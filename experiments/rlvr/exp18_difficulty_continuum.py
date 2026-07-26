import json
import torch
import gc
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def get_activations(model, inputs):
    cache = {}
    handles = []
    def hook(module, inp, out, layer_idx):
        h = out[0] if isinstance(out, tuple) else out
        # We only care about the last token of the prefill for trajectory divergence
        cache[layer_idx] = h[:, -1, :].detach().clone().float()
    for l in range(32):
        handles.append(model.model.layers[l].register_forward_hook(
            lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
        ))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles: h.remove()
    return cache

def compute_l2_distance(act_sft, act_rlvr):
    # Shape: (batch, hidden)
    diff = act_sft - act_rlvr
    l2 = torch.norm(diff, dim=-1)
    norm_sft = torch.norm(act_sft, dim=-1)
    # relative L2 distance
    return (l2 / norm_sft).cpu().numpy()

def compute_cosine_distance(act_sft, act_rlvr):
    # Shape: (batch, hidden)
    cos_sim = torch.nn.functional.cosine_similarity(act_sft, act_rlvr, dim=-1)
    return (1.0 - cos_sim).cpu().numpy()

def run_experiment():
    print("Loading large-scale dataset...")
    try:
        with open('/marimo/large_scale_difficulty_dataset.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found. Ensure exp16 completed.")
        return
        
    print("Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    sft_model = AutoModelForCausalLM.from_pretrained(MODELS["SFT"], torch_dtype=torch.bfloat16, device_map='auto').eval()
    rlvr_model = AutoModelForCausalLM.from_pretrained(MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map='auto').eval()

    # Sort dataset by difficulty
    ds = sorted(ds, key=lambda x: x['difficulty'])
    
    # Sample a manageable number of examples uniformly across the spectrum.
    # This keeps the experiment runnable even when exp16 generates a smaller dataset.
    max_examples = min(300, len(ds))
    if len(ds) > max_examples:
        indices = np.linspace(0, len(ds)-1, max_examples).astype(int)
        sampled_ds = [ds[i] for i in indices]
    else:
        sampled_ds = ds

    results = []
    
    BATCH_SIZE = 8
    
    for i in tqdm(range(0, len(sampled_ds), BATCH_SIZE), desc="Computing Trajectory Divergence"):
        batch = sampled_ds[i:i+BATCH_SIZE]
        prompts = [format_prompt(tokenizer, e["prompt"]) for e in batch]
        inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(sft_model.device)
        
        cache_sft = get_activations(sft_model, inp)
        cache_rlvr = get_activations(rlvr_model, inp)
        
        for j, e in enumerate(batch):
            example_data = {
                "difficulty": e["difficulty"],
                "quadrant": e["quadrant"],
                "l2_distance_by_layer": {},
                "cosine_distance_by_layer": {}
            }
            
            for l in range(32):
                sft_act = cache_sft[l][j:j+1]
                rlvr_act = cache_rlvr[l][j:j+1]
                
                example_data["l2_distance_by_layer"][l] = float(compute_l2_distance(sft_act, rlvr_act)[0])
                example_data["cosine_distance_by_layer"][l] = float(compute_cosine_distance(sft_act, rlvr_act)[0])
                
            results.append(example_data)
            
        del cache_sft
        del cache_rlvr
        clear_gpu()
        
    with open('/marimo/exp18_difficulty_divergence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Saved raw divergence results to /marimo/exp18_difficulty_divergence_results.json")
    
if __name__ == "__main__":
    run_experiment()
