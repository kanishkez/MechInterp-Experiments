"""
Phase 14 — Causal Trajectory Transfer

Tests the Capability-Frontier Routing Hypothesis (Prediction 2 & 3).
Goal: Determine if the RLVR trajectory can be transplanted into SFT and pinpoint the causal transition point.

Method:
Patch residual stream activations from RLVR into SFT (and vice-versa).
1. Individual Layers: L0, L2, L5, L8, L10, L12, L14, L16, L20, L25, L31
2. Cumulative Segments: L0-5, L0-10, L0-12, L0-15, L0-20
Compare Dataset A (Core) vs Dataset D (Frontier).
"""
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}
INDIVIDUAL_LAYERS = [0, 2, 5, 8, 10, 12, 14, 16, 20, 25, 31]
CUMULATIVE_SEGMENTS = [(0, 5), (0, 10), (0, 12), (0, 15), (0, 20)]

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def cache_activations(model, inputs, layers_to_cache):
    cache = {}
    handles = []
    def hook(module, inp, out, layer_idx):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:  # Only during prefill
            cache[layer_idx] = h.detach().clone()
    for l in layers_to_cache:
        handles.append(model.model.layers[l].register_forward_hook(
            lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
        ))
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for h in handles: h.remove()
    return cache

def patch_activations(model, inputs, tokenizer, cache, layers_to_patch, max_new_tokens=128):
    handles = []
    def hook(module, inp, out, layer_idx):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1 and layer_idx in cache:
            h[:, :, :] = cache[layer_idx][:, :, :]
        return (h,) + out[1:] if isinstance(out, tuple) else h
    for l in layers_to_patch:
        handles.append(model.model.layers[l].register_forward_hook(
            lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
        ))
    try:
        with torch.no_grad():
            prefill_out = model(**inputs, use_cache=True)
            past_kv = prefill_out.past_key_values
            next_token = prefill_out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        for h in handles: h.remove()
    generated = [next_token]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_token)
            if next_token.item() == tokenizer.eos_token_id:
                break
    return torch.cat([inputs['input_ids'], torch.cat(generated, dim=1)], dim=1)

def run_experiment():
    print("Loading Dataset...")
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    val = [e for e in ds if e['split'] == 'validation']
    
    quadrants = {
        "Dataset_A_Core": [e for e in val if e['predictions']['SFT']['correct'] and e['predictions']['RLVR']['correct']][:25],
        "Dataset_D_Frontier": [e for e in val if not e['predictions']['SFT']['correct'] and e['predictions']['RLVR']['correct']][:25]
    }
    
    print("Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    sft_model = AutoModelForCausalLM.from_pretrained(MODELS["SFT"], torch_dtype=torch.bfloat16, device_map='auto').eval()
    rlvr_model = AutoModelForCausalLM.from_pretrained(MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map='auto').eval()

    results = {"RLVR_to_SFT": {}, "SFT_to_RLVR": {}}

    for q_name, examples in quadrants.items():
        if not examples: continue
        results["RLVR_to_SFT"][q_name] = {"individual": {}, "cumulative": {}}
        results["SFT_to_RLVR"][q_name] = {"individual": {}, "cumulative": {}}
        
        for e in tqdm(examples, desc=f"Evaluating {q_name}"):
            inp = tokenizer(format_prompt(tokenizer, e['prompt']), return_tensors="pt").to(sft_model.device)
            true_ans = str(e['true_answer']).strip().lower()
            is_gsm = (e['source'] == "gsm8k")
            
            try:
                # RLVR -> SFT Transfer
                rlvr_cache = cache_activations(rlvr_model, inp, INDIVIDUAL_LAYERS + [l for start, end in CUMULATIVE_SEGMENTS for l in range(start, end+1)])
                
                for l in INDIVIDUAL_LAYERS:
                    out = patch_activations(sft_model, inp, tokenizer, rlvr_cache, [l])
                    pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                    hit = (true_ans == pred) if is_gsm else (true_ans in pred)
                    results["RLVR_to_SFT"][q_name]["individual"].setdefault(str(l), []).append(hit)
                    
                for start, end in CUMULATIVE_SEGMENTS:
                    layers = list(range(start, end+1))
                    out = patch_activations(sft_model, inp, tokenizer, rlvr_cache, layers)
                    pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                    hit = (true_ans == pred) if is_gsm else (true_ans in pred)
                    key = f"L{start}-{end}"
                    results["RLVR_to_SFT"][q_name]["cumulative"].setdefault(key, []).append(hit)

                # SFT -> RLVR Transfer
                sft_cache = cache_activations(sft_model, inp, INDIVIDUAL_LAYERS + [l for start, end in CUMULATIVE_SEGMENTS for l in range(start, end+1)])
                
                for l in INDIVIDUAL_LAYERS:
                    out = patch_activations(rlvr_model, inp, tokenizer, sft_cache, [l])
                    pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                    hit = (true_ans == pred) if is_gsm else (true_ans in pred)
                    results["SFT_to_RLVR"][q_name]["individual"].setdefault(str(l), []).append(hit)
                    
                for start, end in CUMULATIVE_SEGMENTS:
                    layers = list(range(start, end+1))
                    out = patch_activations(rlvr_model, inp, tokenizer, sft_cache, layers)
                    pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                    hit = (true_ans == pred) if is_gsm else (true_ans in pred)
                    key = f"L{start}-{end}"
                    results["SFT_to_RLVR"][q_name]["cumulative"].setdefault(key, []).append(hit)
                
                del rlvr_cache
                del sft_cache
                clear_gpu()
            except Exception as exc:
                print(f"Error evaluating example: {exc}")
                clear_gpu()
                continue

    # Average results
    for direction in results:
        for q_name in results[direction]:
            for type_name in results[direction][q_name]:
                for k in results[direction][q_name][type_name]:
                    results[direction][q_name][type_name][k] = sum(results[direction][q_name][type_name][k]) / len(results[direction][q_name][type_name][k]) * 100

    with open('/marimo/exp_phase14_trajectory_patching_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to /marimo/exp_phase14_trajectory_patching_results.json")

if __name__ == "__main__":
    run_experiment()
