import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import gc

models_to_run = [
    "meta-llama/Meta-Llama-3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "allenai/Llama-3.1-Tulu-3-8B"
]

prompts = [
    # Math / Reasoning (GSM8K style)
    "If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?",
    "A store sells apples for $0.50 each and oranges for $0.75 each. If I buy 4 apples and 3 oranges, how much do I spend?",
    "Solve for x: 3x + 5 = 20.",
    "A rectangle has a length of 10 and a width of 5. What is its area?",
    "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
    
    # Coding
    "Write a Python function to compute the Fibonacci sequence up to n.",
    "How do you reverse a string in JavaScript?",
    "Explain what a pointer is in C++.",
    "Write a SQL query to find the second highest salary from an Employee table.",
    "What is the time complexity of binary search?",
    
    # Factual / QA
    "Who was the first president of the United States?",
    "What is the capital of Japan?",
    "What is the chemical symbol for Gold?",
    "In what year did the Apollo 11 moon landing occur?",
    "Who wrote the play 'Romeo and Juliet'?"
]

def compute_js_divergence(logits_p, logits_q):
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd.item()

def compute_cka(X, Y):
    # X, Y shape: (num_prompts, hidden_dim)
    # Centered Kernel Alignment (Linear)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    dot_XX = torch.sum(X * X).item()
    dot_YY = torch.sum(Y * Y).item()
    dot_XY = torch.sum(X * Y).item() ** 2
    
    if dot_XX == 0 or dot_YY == 0:
        return 0.0
    return np.sqrt(dot_XY / (dot_XX * dot_YY))

def main():
    print("=== Tulu 3 Macroscopic Divergence ===")
    
    # We will store the last token hidden states and logits for all models
    # hidden_states[model_name][layer] = tensor(num_prompts, hidden_dim)
    # final_logits[model_name] = tensor(num_prompts, vocab_size)
    
    all_hidden_states = {}
    all_final_logits = {}
    
    tokenizer = None
    
    for model_name in models_to_run:
        print(f"\n--- Loading {model_name} ---")
        try:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            model.eval()
            
            num_layers = len(model.model.layers)
            
            model_hiddens = [[] for _ in range(num_layers)]
            model_logits = []
            
            with torch.no_grad():
                for p in prompts:
                    inputs = tokenizer(p, return_tensors="pt").to(model.device)
                    # output_hidden_states=True returns (embeddings, layer1, layer2, ...)
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    # Logits for the last token
                    last_token_logits = outputs.logits[0, -1, :].cpu()
                    model_logits.append(last_token_logits)
                    
                    # Hidden states for the last token, skipping embeddings (index 0)
                    hiddens = outputs.hidden_states[1:] 
                    for layer_idx, h in enumerate(hiddens):
                        last_token_h = h[0, -1, :].cpu()
                        model_hiddens[layer_idx].append(last_token_h)
            
            all_final_logits[model_name] = torch.stack(model_logits)
            
            all_hidden_states[model_name] = []
            for layer_idx in range(num_layers):
                all_hidden_states[model_name].append(torch.stack(model_hiddens[layer_idx]))
                
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")
            
    print("\n--- Computing Metrics ---")
    comparisons = [
        ("meta-llama/Meta-Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT", "Base \u2192 SFT"),
        ("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "SFT \u2192 DPO"),
        ("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B", "DPO \u2192 RLVR")
    ]
    
    jsd_results = {}
    cka_results = {}
    
    num_layers = 32
    
    for mod_A, mod_B, label in comparisons:
        if mod_A not in all_final_logits or mod_B not in all_final_logits:
            print(f"Skipping {label} due to missing models.")
            continue
            
        # JS Divergence on final output logits
        jsd = compute_js_divergence(all_final_logits[mod_A], all_final_logits[mod_B])
        jsd_results[label] = jsd
        
        # Layer-wise CKA
        cka_layer = []
        for l in range(num_layers):
            h_A = all_hidden_states[mod_A][l].float()
            h_B = all_hidden_states[mod_B][l].float()
            cka = compute_cka(h_A, h_B)
            cka_layer.append(cka)
            
        cka_results[label] = cka_layer
        print(f"Comparison: {label}")
        print(f"Final JS Divergence: {jsd:.4f}")
        print(f"Min CKA: {min(cka_layer):.4f} at layer {np.argmin(cka_layer)}")
        print(f"Max CKA: {max(cka_layer):.4f} at layer {np.argmax(cka_layer)}")
        
    print("\n--- Generating Plots ---")
    
    plt.figure(figsize=(10, 6))
    for label, cka_vals in cka_results.items():
        plt.plot(range(num_layers), cka_vals, marker='o', label=label)
    
    plt.title('Representational Similarity (CKA) Across Tulu-3 Post-Training')
    plt.xlabel('Layer')
    plt.ylabel('CKA Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tulu_01_cka_plot.png', dpi=300)
    print("Saved plot to tulu_01_cka_plot.png")

if __name__ == '__main__':
    main()
