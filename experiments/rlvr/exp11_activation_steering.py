"""
Phase E — Causal State Steering Across Models (Killer Experiment)

Tests the RLVR State-Selection Hypothesis:
1. Extract direction d_L = E[h_L | SFT success] - E[h_L | SFT failure] from Discovery set.
2. Sufficiency: SFT h'_L = h_L + α d_L (evaluate on SFT failure cases).
3. Necessity: RLVR h'_L = h_L - α d_L (evaluate on RLVR success cases).

Controls: Random direction, Orthogonal direction.
Layer Sweep: [5, 10, 14, 20, 25, 31]
Alpha Sweep: [0.0, 0.5, 1.0, 2.0, 4.0]
"""
import json
import torch
import gc
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT_MODEL_ID  = "allenai/Llama-3.1-Tulu-3-8B-SFT"
RLVR_MODEL_ID = "allenai/Llama-3.1-Tulu-3-8B"

LAYERS = [5, 10, 14, 20, 25, 31]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def extract_h_l(model, tokenizer, prompts, layer):
    acts = []
    current = {}
    
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.dim() == 3:
            current[0] = h[0, -1, :].detach().to(torch.float32).cpu()
        else:
            current[0] = h[-1, :].detach().to(torch.float32).cpu()
            
    handle = model.model.layers[layer].register_forward_hook(hook)
    
    for prompt in tqdm(prompts, desc=f"Extract L{layer}", leave=False):
        current = {}
        inp = tokenizer(format_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inp)
        acts.append(current[0])
        
    handle.remove()
    return acts

def steered_forward(model, inputs, layer, direction_gpu, alpha):
    handle_list = []
    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if h.dim() == 3:
            h = h.clone()
            h[0, -1, :] = h[0, -1, :] + alpha * direction_gpu.to(h.device)
        else:
            h = h.clone()
            h[-1, :] = h[-1, :] + alpha * direction_gpu.to(h.device)
        return (h,) + out[1:] if is_tuple else h

    handle_list.append(model.model.layers[layer].register_forward_hook(hook))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=model.config.eos_token_id)
    for h in handle_list:
        h.remove()
    return out

def run_bidirectional_steering():
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    disc = [e for e in ds if e['split'] == 'discovery']
    val  = [e for e in ds if e['split'] == 'validation']
    print(f"Discovery: {len(disc)}, Validation: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    disc_prompts = [e['prompt'] for e in disc]
    disc_sft_labels = np.array([e['predictions']['SFT']['correct'] for e in disc])

    results = {"sft_sufficiency": {}, "rlvr_necessity": {}}

    print("=== Loading SFT Model ===")
    clear_gpu()
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    sft_model.eval()

    val_fail = [e for e in val if not e['predictions']['SFT']['correct']]
    print(f"SFT Fail Validation: {len(val_fail)}")

    directions = {}
    for layer in LAYERS:
        print(f"\n[SFT] Processing Layer {layer}...")
        disc_acts = extract_h_l(sft_model, tokenizer, disc_prompts, layer)
        disc_acts = torch.stack(disc_acts)
        
        success_mask = disc_sft_labels == True
        failure_mask = disc_sft_labels == False
        
        if success_mask.sum() == 0 or failure_mask.sum() == 0:
            print("Not enough class diversity. Skipping.")
            continue
            
        d = disc_acts[success_mask].mean(0) - disc_acts[failure_mask].mean(0)
        d = d / (d.norm() + 1e-8)
        directions[layer] = d
        
        results["sft_sufficiency"][layer] = {}
        for alpha in ALPHAS:
            steered_correct = []
            for e in tqdm(val_fail, desc=f"SFT Sufficiency α={alpha}", leave=False):
                inp = tokenizer(format_prompt(tokenizer, e['prompt']), return_tensors="pt").to(sft_model.device)
                if alpha == 0.0:
                    with torch.no_grad():
                        out = sft_model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                else:
                    out = steered_forward(sft_model, inp, layer, d, alpha)
                
                pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                true = str(e['true_answer']).strip().lower()
                hit = (true in pred) if e['source'] != "gsm8k" else (true == pred)
                steered_correct.append(hit)
                
            acc = np.mean(steered_correct) * 100
            results["sft_sufficiency"][layer][alpha] = float(acc)
            print(f"  α={alpha}: {acc:.1f}%")

    del sft_model; clear_gpu()

    print("\n=== Loading RLVR Model ===")
    rlvr_model = AutoModelForCausalLM.from_pretrained(RLVR_MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    rlvr_model.eval()

    val_success = [e for e in val if e['predictions']['RLVR']['correct']]
    print(f"RLVR Success Validation: {len(val_success)}")

    for layer in LAYERS:
        if layer not in directions: continue
        print(f"\n[RLVR] Processing Layer {layer}...")
        d = directions[layer]
        
        results["rlvr_necessity"][layer] = {}
        for alpha in ALPHAS:
            steered_correct = []
            # We use -alpha for degradation
            for e in tqdm(val_success, desc=f"RLVR Necessity α={alpha}", leave=False):
                inp = tokenizer(format_prompt(tokenizer, e['prompt']), return_tensors="pt").to(rlvr_model.device)
                if alpha == 0.0:
                    with torch.no_grad():
                        out = rlvr_model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                else:
                    out = steered_forward(rlvr_model, inp, layer, d, -alpha)
                
                pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                true = str(e['true_answer']).strip().lower()
                hit = (true in pred) if e['source'] != "gsm8k" else (true == pred)
                steered_correct.append(hit)
                
            acc = np.mean(steered_correct) * 100
            results["rlvr_necessity"][layer][alpha] = float(acc)
            print(f"  α={alpha} (negative): {acc:.1f}%")

    del rlvr_model; clear_gpu()

    with open('/marimo/exp_phaseE_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to /marimo/exp_phaseE_results.json")

if __name__ == "__main__":
    run_bidirectional_steering()
