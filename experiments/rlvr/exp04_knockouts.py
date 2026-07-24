import json
import torch
import gc
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "DPO": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def bootstrap_ci(data, n_boot=1000, ci=95):
    data = np.array(data)
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

def load_dataset():
    with open('/marimo/semantic_agreement_dataset.json', 'r') as f:
        ds = json.load(f)
    dataset_a = [e for e in ds if e.get('dataset') == 'A'][:100]
    dataset_d = [e for e in ds if e.get('dataset') == 'D']
    return dataset_a, dataset_d

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def run_ablation(model, inputs, layers_to_ablate, component='mlp'):
    handles = []
    
    def make_ablation_hook():
        def hook(module, inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            
            # Zero-ablate the sequence last token
            if h.dim() == 3:
                h[0, -1, :] = 0.0
            else:
                h[-1, :] = 0.0
                
            return (h,) + out[1:] if is_tuple else h
        return hook

    for l in layers_to_ablate:
        if component == 'mlp':
            handles.append(model.model.layers[l].mlp.register_forward_hook(make_ablation_hook()))
        else:
            handles.append(model.model.layers[l].self_attn.register_forward_hook(make_ablation_hook()))
            
    with torch.no_grad():
        out = model(**inputs).logits[0, -1, :].detach().cpu()
        
    for h in handles:
        h.remove()
        
    return out

def evaluate_knockouts():
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"])
    dataset_a, dataset_d = load_dataset()
    print(f"Dataset A: {len(dataset_a)}, Dataset D: {len(dataset_d)}")

    all_prompts = []
    for ds_tag, ds in [('A', dataset_a), ('D', dataset_d)]:
        for p in ds:
            all_prompts.append((ds_tag, p['prompt'] if isinstance(p, dict) else p))

    results = {}
    
    # Define the ablation conditions
    ablation_conditions = {
        "L13_only": [13],
        "L11": [11],
        "L11_12": [11, 12],
        "L11_12_13": [11, 12, 13],
        "L11_12_13_14": [11, 12, 13, 14],
        "L11_15_all": [11, 12, 13, 14, 15],
        "Surround_No_L13": [11, 12, 14, 15]
    }

    for model_name, model_id in MODELS.items():
        print(f"\n--- Loading {model_name} ---")
        clear_gpu()
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
        model.eval()
        
        model_results = {cond: {'A': [], 'D': []} for cond in ablation_conditions}
        
        for ds_tag, prompt in tqdm(all_prompts, desc=f"Evaluating {model_name}"):
            formatted = format_prompt(tokenizer, prompt)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            
            # Baseline
            with torch.no_grad():
                baseline_logits = model(**inputs).logits[0, -1, :].detach().cpu()
            baseline_token = baseline_logits.argmax().item()
            
            # Run conditions
            for cond_name, layers in ablation_conditions.items():
                ablated_logits = run_ablation(model, inputs, layers, component='mlp')
                ablated_token = ablated_logits.argmax().item()
                
                # Did the preferred token survive? (1 = yes, 0 = flipped)
                survived = 1 if ablated_token == baseline_token else 0
                model_results[cond_name][ds_tag].append(survived)
                
        results[model_name] = model_results
        del model
        
    print("\n=== RESULTS ===")
    summary = {}
    for model_name, model_results in results.items():
        summary[model_name] = {}
        print(f"\nModel: {model_name}")
        for cond_name, cond_res in model_results.items():
            summary[model_name][cond_name] = {}
            for ds_tag, surv_list in cond_res.items():
                mean, lower, upper = bootstrap_ci(surv_list)
                summary[model_name][cond_name][ds_tag] = {
                    "mean": mean,
                    "lower": lower, 
                    "upper": upper,
                    "n": len(surv_list)
                }
                print(f"  {cond_name} (Dataset {ds_tag}): {mean*100:.1f}% survival [95% CI: {lower*100:.1f}% - {upper*100:.1f}%]")

    with open('/marimo/exp6_knockouts.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
if __name__ == '__main__':
    evaluate_knockouts()
