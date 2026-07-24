import torch, gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import torch.nn.functional as F

print('=== EXP 33: EFFECTIVE DIMENSIONALITY (FIXED) ===')

model_name = 'Qwen/Qwen2.5-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")

LAYER = 20

# 1. Extract a diverse set of hidden states to run PCA on (>40 prompts)
prompts = [
    "The capital of France is", "Write a short story about a time traveler.",
    "If x+5=10, x is", "Translate to Spanish: Hello world",
    "Write a Python function to reverse a list", "Describe the smell of rain.",
    "Who painted the Mona Lisa?", "Explain quantum physics to a child.",
    "The chemical symbol for water is", "The first president of the United States was",
    "The largest planet in the solar system is", "The currency used in Japan is", 
    "The author of Romeo and Juliet is", "The process of plants making food is called", 
    "The boiling point of water in Celsius is", "The tallest mountain on Earth is",
    "The primary language spoken in Brazil is", "If x + 5 = 10, then x is", 
    "The square root of 144 is", "The area of a circle with radius r is",
    "The derivative of x^2 is", "15 percent of 200 is", 
    "The next number in the sequence 2, 4, 6, 8 is", "The value of 5 factorial (5!) is", 
    "The sum of angles in a triangle is", "If a rectangle is 4 by 5, the perimeter is",
    "The absolute value of -10 is", "Write a short, imaginative story about a time-traveling cat:", 
    "Brainstorm a unique name for a futuristic coffee shop:",
    "Describe the feeling of standing on a mountain at sunrise in a poetic way:", 
    "Invent a new holiday and describe its main tradition:",
    "Write a colorful metaphor about the concept of friendship:", 
    "Imagine a world where gravity reverses for one hour every day. Describe it:",
    "Give me three weird but interesting ideas for a sci-fi novel:", 
    "Write a funny haiku about a robot trying to eat a sandwich:",
    "Describe what the color blue sounds like to someone who can't see:", 
    "Come up with a creative excuse for why you didn't do your homework:",
    "What is the airspeed velocity of an unladen swallow?",
    "Explain the theory of relativity in simple terms.",
    "Write a SQL query to join two tables."
]

def get_h(m, p):
    ids = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m(**ids, output_hidden_states=True)
    return out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()

model_base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="auto")

h_inst = np.array([get_h(model, "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n") for p in prompts])
h_base = np.array([get_h(model_base, "<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n") for p in prompts])
h_diff = h_inst - h_base

del model_base
gc.collect()
torch.cuda.empty_cache()

# 2. Run PCA to get orthogonal basis vectors
max_components = min(32, h_diff.shape[0])
pca = PCA(n_components=max_components)
pca.fit(h_diff)
components = torch.tensor(pca.components_).to('cuda').float() # Shape: (k, 3584)

# 3. Iteratively project out top-k components and measure recovery loss
# Test on a held-out prompt (not in the PCA fit set)
test_prompt = "<|im_start|>user\nWrite a short poem about a mechanical bird.<|im_end|>\n<|im_start|>assistant\n"
ids = tokenizer(test_prompt, return_tensors='pt').to('cuda')

# Target (Instruct)
with torch.no_grad():
    out_target = model(**ids)
probs_target = F.softmax(out_target.logits[0, -1, :].float(), dim=-1).cpu().numpy()

# Baseline (Corrupt User)
base_prompt = "<|im_start|>user\nWrite a short poem about a mechanical bird.<|im_end|>\n<|im_start|>user\n"
ids_base = tokenizer(base_prompt, return_tensors='pt').to('cuda')
with torch.no_grad():
    out_base = model(**ids_base)
probs_base = F.softmax(out_base.logits[0, -1, :].float(), dim=-1).cpu().numpy()

def kl(p, q):
    return (p * (np.log(np.clip(p, 1e-10, None)) - np.log(np.clip(q, 1e-10, None)))).sum()

kl_gap = kl(probs_base, probs_target)
print(f"Initial KL Gap: {kl_gap:.4f}")

k_values = list(range(17)) # 0 to 16
for k in k_values:
    if k > 0:
        directions_to_remove = components[:k]
    else:
        directions_to_remove = None
        
    def pre_hook(m, args, current_k=k, current_dirs=directions_to_remove):
        h = args[0]
        h_last = h[:, -1:, :].float()
        
        if current_dirs is not None:
            for i in range(current_k):
                d = current_dirs[i]
                d = d / torch.norm(d)
                proj = (h_last * d).sum(-1, keepdim=True) * d
                h_last = h_last - proj
                
        h[:, -1:, :] = h_last.to(h.dtype)
        return (h,) + args[1:]
        
    # Hook the input to layer 20
    hk = model.model.layers[LAYER].register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        out_ablated = model(**ids)
    hk.remove()
    
    probs_ablated = F.softmax(out_ablated.logits[0, -1, :].float(), dim=-1).cpu().numpy()
    kl_ablated = kl(probs_ablated, probs_target)
    
    recovery = max(0, (kl_gap - kl_ablated) / kl_gap * 100)
    print(f"Dimensions Removed: {k:2d} | KL Recovery: {recovery:6.2f}%")

