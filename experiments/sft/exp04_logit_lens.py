import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

MODELS = {
    "Base": "NousResearch/Meta-Llama-3.1-8B",
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "DPO": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}

def load_prompts(tokenizer):
    with open('/marimo/tulu_gold_trajectories.json', 'r') as f:
        trajectories = json.load(f)
    
    prompts = [t['prompt'] for t in trajectories]
    domains = [t['domain'] for t in trajectories]
    
    inputs_list = []
    for p in prompts:
        message = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        inputs_list.append(formatted)
    return inputs_list, domains

def compute_logit_lens():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"])
    prompts, domains = load_prompts(tokenizer)
    
    results = {
        name: {dom: {layer: [] for layer in range(33)} for dom in set(domains)} 
        for name in MODELS.keys()
    }
    
    for name, path in MODELS.items():
        print(f"Loading {name} ({path})...")
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16)
        model.eval()
        
        # We need the final layer norm and unembedding matrix
        norm = model.model.norm
        lm_head = model.lm_head
        
        with torch.no_grad():
            for i, p in enumerate(tqdm(prompts, desc=f"Logit Lens {name}")):
                dom = domains[i]
                
                # Output all hidden states
                out = model(**p.to(model.device), output_hidden_states=True)
                
                # The final logits (target)
                final_logits = out.logits[0, -1, :].float()
                p_final = F.softmax(final_logits, dim=-1)
                
                # Loop through all layers
                # out.hidden_states has 33 elements: [embedding, layer1_out, ..., layer32_out]
                for layer_idx, h in enumerate(out.hidden_states):
                    h_t0 = h[0, -1, :] # Last token
                    
                    # Apply final layer norm and unembedding
                    h_norm = norm(h_t0)
                    layer_logits = lm_head(h_norm).float()
                    p_layer = F.softmax(layer_logits, dim=-1)
                    
                    m = 0.5 * (p_final + p_layer)
                    kl_final = F.kl_div(m.log(), p_final, reduction='sum')
                    kl_layer = F.kl_div(m.log(), p_layer, reduction='sum')
                    jsd = 0.5 * (kl_final + kl_layer)
                    
                    results[name][dom][layer_idx].append(jsd.item())
                    
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
    # Aggregate (mean)
    aggregated = {}
    for name in results:
        aggregated[name] = {}
        for dom in results[name]:
            aggregated[name][dom] = [np.mean(results[name][dom][l]) for l in range(33)]
            
    with open('/marimo/evolution_logit_lens.json', 'w') as f:
        json.dump(aggregated, f)
        
    print("\n=== EVOLUTION LOGIT LENS PAYLOAD ===")
    print("Finished writing evolution_logit_lens.json")
    print("=== END JSON PAYLOAD ===")

if __name__ == '__main__':
    compute_logit_lens()
