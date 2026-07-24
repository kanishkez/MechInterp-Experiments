import torch, gc
import pandas as pd
import numpy as np
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

print('=== EXP 21: ROUTING VECTOR EXTRACTION (QWEN 2.5 7B) ===')

model_name = 'Qwen/Qwen2.5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
num_layers = len(model.model.layers)

categories = {
    "Math": [
        "What is 2+2?", "Solve for x: 2x = 10", "What is the derivative of x^2?",
        "Calculate 15 * 12", "Is 17 a prime number?", "What is the square root of 144?",
        "Integrate 2x dx", "What is 5! (factorial)?", "Convert 100 Celsius to Fahrenheit",
        "Solve 3y - 7 = 11", "Find the area of a circle with radius 5", "What is 2^10?",
        "Is 0 an even number?", "What is the log base 10 of 1000?", "Evaluate sin(pi/2)",
        "What is 1 + 2 + 3 + ... + 100?", "Simplify x^3 * x^4", "What is the Pythagorean theorem?",
        "Solve x^2 - 4 = 0", "What is 100 divided by 0?"
    ],
    "Coding": [
        "def fibonacci(n):", "Write a python script to parse JSON", "class Singleton:",
        "How do I reverse a linked list?", "import pandas as pd", "function bubbleSort(arr) {",
        "Write a SQL query to join two tables", "What is the difference between TCP and UDP?",
        "Explain Big O notation", "def binary_search(arr, target):", "How to fix a NullPointerException?",
        "Write a regex to match an email", "FROM ubuntu:latest", "git commit -m",
        "What is a closure in JavaScript?", "public static void main(String[] args)",
        "How to use Docker compose?", "Write a rust macro", "def quicksort(arr):",
        "What is REST API?"
    ],
    "Factual": [
        "The capital of France is", "Who wrote Romeo and Juliet?", "When did WWII end?",
        "What is the largest planet in our solar system?", "Who painted the Mona Lisa?",
        "What is the chemical symbol for Gold?", "How many continents are there?",
        "Who was the first president of the USA?", "What is the speed of light?",
        "When was the Declaration of Independence signed?", "What is the tallest mountain in the world?",
        "Who invented the telephone?", "What is the main ingredient in guacamole?",
        "Where is the Great Wall of China?", "What currency is used in Japan?",
        "Who discovered penicillin?", "What is the hardest natural substance?",
        "When did the Titanic sink?", "Who is the CEO of Tesla?", "What is the capital of Australia?"
    ],
    "Creative": [
        "Write a poem about the sea.", "Once upon a time in a dark forest",
        "Create a short story about a time traveler.", "Write a haiku about winter.",
        "Draft a speech for a retiring teacher.", "Describe a futuristic city.",
        "Write a song lyric about heartbreak.", "Invent a new color and describe it.",
        "Write a dialogue between a cat and a dog.", "Create a fantasy world map description.",
        "Write a motivational quote.", "Describe the taste of a lemon to an alien.",
        "Write a romantic letter.", "Draft a movie pitch about a rogue AI.",
        "Write a joke about a programmer.", "Describe a haunted house.",
        "Write a recipe for disaster.", "Create a superhero origin story.",
        "Write a bedtime story for a 5 year old.", "Describe a perfect day."
    ],
    "Translation": [
        "Translate to French: Hello", "How do you say 'Thank you' in Spanish?",
        "Translate 'Good morning' to German", "What is the Italian word for 'Apple'?",
        "Translate to Japanese: I love you", "How to say 'Where is the bathroom?' in Chinese",
        "Translate 'Goodbye' to Russian", "What is 'Cat' in Arabic?",
        "Translate to Portuguese: How are you?", "How do you say 'Water' in Hindi?",
        "Translate 'My name is' to Korean", "What is the French word for 'Dog'?",
        "Translate to Spanish: I need help", "How to say 'Yes' in Turkish?",
        "Translate 'No' to Greek", "What is 'Friend' in Swahili?",
        "Translate to German: What time is it?", "How do you say 'Beautiful' in Italian?",
        "Translate 'Food' to Vietnamese", "What is the Dutch word for 'House'?"
    ]
}

# 1. Extract mean vectors for each category at each layer
category_vectors = {}  # cat -> layer -> vector (tensor)

for category, prompts in categories.items():
    print(f"Extracting vectors for {category}...")
    
    # We will accumulate the difference vectors over the 20 prompts
    diff_accum = {l: [] for l in range(num_layers)}
    
    for text in prompts:
        base_prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>"
        prompt_asst = base_prompt + "assistant\n"
        prompt_user = base_prompt + "user\n"
        
        ids_asst = tokenizer(prompt_asst, return_tensors='pt').to('cuda')
        ids_user = tokenizer(prompt_user, return_tensors='pt').to('cuda')
        
        res_asst = {}
        res_user = {}
        
        def cache_res(store_dict, li):
            def fn(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                store_dict[li] = out[:, -1:, :].detach().clone()
            return fn
            
        hks_asst = [model.model.layers[li].register_forward_hook(cache_res(res_asst, li)) for li in range(num_layers)]
        with torch.no_grad(): model(**ids_asst)
        for hk in hks_asst: hk.remove()
        
        hks_user = [model.model.layers[li].register_forward_hook(cache_res(res_user, li)) for li in range(num_layers)]
        with torch.no_grad(): model(**ids_user)
        for hk in hks_user: hk.remove()
        
        for li in range(num_layers):
            diff = res_asst[li] - res_user[li]
            diff_accum[li].append(diff)
            
    # Compute mean over prompts
    category_vectors[category] = {}
    for li in range(num_layers):
        mean_diff = torch.cat(diff_accum[li], dim=0).mean(dim=0, keepdim=True)
        category_vectors[category][li] = mean_diff

# 2. Compute cross-category cosine similarities at their causal commit layers
# From our previous exp15 causal audit, we found these approximate mean causal commit layers:
causal_layers = {
    "Factual": 5,    # early
    "Math": 6,       # early
    "Translation": 9,
    "Coding": 12,    # mid-late
    "Creative": 20   # late
}

# Create a matrix of cosine similarities between categories at their respective causal layers
n_cats = len(causal_layers)
cat_names = list(causal_layers.keys())
cos_matrix = np.zeros((n_cats, n_cats))

for i, cat1 in enumerate(cat_names):
    for j, cat2 in enumerate(cat_names):
        vec1 = category_vectors[cat1][causal_layers[cat1]].squeeze().float()
        vec2 = category_vectors[cat2][causal_layers[cat2]].squeeze().float()
        
        cos_sim = F.cosine_similarity(vec1, vec2, dim=0).item()
        cos_matrix[i, j] = cos_sim

print("\n=== COSINE SIMILARITY BETWEEN CAUSAL VECTORS ===")
df_cos = pd.DataFrame(cos_matrix, index=cat_names, columns=cat_names)
print(df_cos.round(3))

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cos_matrix, annot=True, xticklabels=cat_names, yticklabels=cat_names, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Cosine Similarity of Routing Vectors (at empirical causal layers)")
plt.tight_layout()
plt.savefig("routing_vector_cosine.png", dpi=300)
print("\nSaved routing_vector_cosine.png")

# Save the mean vectors for the intervention experiment
torch.save(category_vectors, "routing_vectors.pt")
print("Saved routing_vectors.pt")
