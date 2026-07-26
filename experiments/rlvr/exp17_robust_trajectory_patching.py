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

def extract_final_answer(text):
    if "#### " in text:
        return text.split("#### ")[-1].strip()
    return text.strip()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def cache_activations(model, inputs, layers_to_cache):
    cache = {}
    handles = []
    def hook(module, inp, out, layer_idx):
        h = out[0] if isinstance(out, tuple) else out
        # Only capture during prefill (seq_len > 1)
        if h.shape[1] > 1:
            cache[layer_idx] = h.detach().clone()
    for l in layers_to_cache:
        handles.append(model.model.layers[l].register_forward_hook(
            lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
        ))
    try:
        with torch.no_grad():
            # Use model.forward() directly to guarantee hooks fire during prefill
            model(**inputs, use_cache=False)
    finally:
        for h in handles: h.remove()
    return cache

def patch_activations(model, inputs, tokenizer, cache, layers_to_patch, max_new_tokens=128):
    handles = []
    def hook(module, inp, out, layer_idx):
        h = out[0] if isinstance(out, tuple) else out
        # Only patch during prefill (seq_len > 1) and if we have the cache
        if h.shape[1] > 1 and layer_idx in cache:
            h[:, :, :] = cache[layer_idx][:, :, :]
        return (h,) + out[1:] if isinstance(out, tuple) else h
    for l in layers_to_patch:
        handles.append(model.model.layers[l].register_forward_hook(
            lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
        ))
    try:
        with torch.no_grad():
            # Step 1: Run patched prefill to get first token and KV cache
            prefill_out = model(**inputs, use_cache=True)
            past_kv = prefill_out.past_key_values
            next_token = prefill_out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        for h in handles: h.remove()

    # Step 2: Custom greedy decode loop (avoids HF generate's internal _prefill re-call)
    generated = [next_token]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_token)
            if (next_token == tokenizer.eos_token_id).all():
                break

    all_generated = torch.cat(generated, dim=1)
    return torch.cat([inputs['input_ids'], all_generated], dim=1)

def run_experiment():
    print("Loading semantic-agreement dataset...")
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found. Ensure semantic_agreement_dataset_full.json is present.")
        return

    val = [e for e in ds if e['split'] == 'validation']
    print(f"Validation Set Size: {len(val)}")

    quadrants = {
        "A_both_pass": [e for e in val if e['predictions']['SFT']['correct'] and e['predictions']['RLVR']['correct']],
        "C_both_fail": [e for e in val if not e['predictions']['SFT']['correct'] and not e['predictions']['RLVR']['correct']],
        "D_sft_fail_rlvr_pass": [e for e in val if not e['predictions']['SFT']['correct'] and e['predictions']['RLVR']['correct']]
    }

    # Keep this runnable in the live notebook by subsampling each quadrant.
    quadrants = {k: v[:8] for k, v in quadrants.items()}
    print("Quadrant sizes:", {k: len(v) for k, v in quadrants.items()})

    print("Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    sft_model = AutoModelForCausalLM.from_pretrained(MODELS["SFT"], torch_dtype=torch.bfloat16, device_map='auto').eval()
    rlvr_model = AutoModelForCausalLM.from_pretrained(MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map='auto').eval()

    results = {
        "Part1_L0-12": {"RLVR_to_SFT": {}, "SFT_to_RLVR": {}},
        "Part2_Cumulative": {"RLVR_to_SFT": {}, "SFT_to_RLVR": {}}
    }
    
    # -------------------------------------------------------------
    # Part 1: Replicate the L0-12 trajectory transfer properly
    # -------------------------------------------------------------
    print("\n--- Running Part 1: L0-12 Transfer on sampled quadrants ---")
    BATCH_SIZE = 50
    for q_name, examples in quadrants.items():
        if not examples: continue
        results["Part1_L0-12"]["RLVR_to_SFT"][q_name] = []
        results["Part1_L0-12"]["SFT_to_RLVR"][q_name] = []
        
        for i in range(0, len(examples), BATCH_SIZE):
            batch = examples[i:i+BATCH_SIZE]
            prompts = [format_prompt(tokenizer, e["prompt"]) for e in batch]
            inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(sft_model.device)
            
            # RLVR -> SFT Transfer (L0-12)
            rlvr_cache = cache_activations(rlvr_model, inp, list(range(13)))
            out_sft = patch_activations(sft_model, inp, tokenizer, rlvr_cache, list(range(13)))
            
            # SFT -> RLVR Transfer (L0-12)
            sft_cache = cache_activations(sft_model, inp, list(range(13)))
            out_rlvr = patch_activations(rlvr_model, inp, tokenizer, sft_cache, list(range(13)))
            
            for j, e in enumerate(batch):
                true_ans = str(e['true_answer']).strip().lower()
                is_gsm = (e['source'] == "gsm8k")
                
                # Check RLVR->SFT
                gen_sft = tokenizer.decode(out_sft[j][inp.input_ids.shape[1]:], skip_special_tokens=True)
                pred_sft = extract_final_answer(gen_sft).strip().lower()
                hit_sft = (true_ans == pred_sft) if is_gsm else (true_ans in pred_sft)
                results["Part1_L0-12"]["RLVR_to_SFT"][q_name].append(hit_sft)
                
                # Check SFT->RLVR
                gen_rlvr = tokenizer.decode(out_rlvr[j][inp.input_ids.shape[1]:], skip_special_tokens=True)
                pred_rlvr = extract_final_answer(gen_rlvr).strip().lower()
                hit_rlvr = (true_ans == pred_rlvr) if is_gsm else (true_ans in pred_rlvr)
                results["Part1_L0-12"]["SFT_to_RLVR"][q_name].append(hit_rlvr)
                
            del rlvr_cache
            del sft_cache
            clear_gpu()

    # Calculate Confidence Intervals for Part 1
    def get_ci(data):
        if not data: return 0, 0, 0
        arr = np.array(data) * 100
        mean = np.mean(arr)
        # 95% bootstrap CI
        boot = np.random.choice(arr, (1000, len(arr)), replace=True)
        means = np.mean(boot, axis=1)
        return mean, np.percentile(means, 2.5), np.percentile(means, 97.5)

    print("\nPart 1 Results (L0-12 Patching):")
    for q_name in quadrants.keys():
        m1, l1, u1 = get_ci(results["Part1_L0-12"]["RLVR_to_SFT"].get(q_name, []))
        m2, l2, u2 = get_ci(results["Part1_L0-12"]["SFT_to_RLVR"].get(q_name, []))
        print(f"[{q_name}] RLVR->SFT: {m1:.1f}% [{l1:.1f}-{u1:.1f}] | SFT->RLVR: {m2:.1f}% [{l2:.1f}-{u2:.1f}]")


    # -------------------------------------------------------------
    # Part 2: Find exact causal boundary on Dataset D
    # -------------------------------------------------------------
    print("\n--- Running Part 2: Exact Causal Boundary on Dataset D ---")
    frontier_examples = quadrants["D_sft_fail_rlvr_pass"]
    
    if frontier_examples:
        for i in range(0, len(frontier_examples), BATCH_SIZE):
            batch = frontier_examples[i:i+BATCH_SIZE]
            prompts = [format_prompt(tokenizer, e["prompt"]) for e in batch]
            inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(sft_model.device)
            
            # Cache all layers for both models
            rlvr_cache = cache_activations(rlvr_model, inp, list(range(32)))
            sft_cache = cache_activations(sft_model, inp, list(range(32)))
            
            for max_layer in tqdm(range(32), desc="Patching layers cumulatively"):
                layers_to_patch = list(range(max_layer + 1))
                key = f"L0-{max_layer}"
                results["Part2_Cumulative"]["RLVR_to_SFT"].setdefault(key, [])
                results["Part2_Cumulative"]["SFT_to_RLVR"].setdefault(key, [])
                
                out_sft = patch_activations(sft_model, inp, tokenizer, rlvr_cache, layers_to_patch)
                out_rlvr = patch_activations(rlvr_model, inp, tokenizer, sft_cache, layers_to_patch)
                
                for j, e in enumerate(batch):
                    true_ans = str(e['true_answer']).strip().lower()
                    is_gsm = (e['source'] == "gsm8k")
                    
                    pred_sft = extract_final_answer(tokenizer.decode(out_sft[j][inp.input_ids.shape[1]:], skip_special_tokens=True)).strip().lower()
                    hit_sft = (true_ans == pred_sft) if is_gsm else (true_ans in pred_sft)
                    results["Part2_Cumulative"]["RLVR_to_SFT"][key].append(hit_sft)
                    
                    pred_rlvr = extract_final_answer(tokenizer.decode(out_rlvr[j][inp.input_ids.shape[1]:], skip_special_tokens=True)).strip().lower()
                    hit_rlvr = (true_ans == pred_rlvr) if is_gsm else (true_ans in pred_rlvr)
                    results["Part2_Cumulative"]["SFT_to_RLVR"][key].append(hit_rlvr)
                    
            del rlvr_cache
            del sft_cache
            clear_gpu()

    # Save all raw boolean results so we can compute CIs later if needed
    with open('/marimo/exp17_robust_patching_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Done! Saved to /marimo/exp17_robust_patching_results.json")

if __name__ == "__main__":
    run_experiment()
