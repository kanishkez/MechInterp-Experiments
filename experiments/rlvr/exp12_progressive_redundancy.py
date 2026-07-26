"""
Phase 12 — Progressive Causal Redundancy

Tests Prediction 3: Is RLVR's reasoning more causally redundant than SFT?
Method:
Progressively ablate subsets of MLP components (k in {1, 2, 5, 10, 20}).
Compare degradation across Dataset A, B, C, D for SFT, DPO, and RLVR.
"""
import json
import torch
import gc
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "DPO": "allenai/Llama-3.1-Tulu-3-8B-DPO", 
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}
K_VALUES = [1, 2, 5, 10, 20]
NUM_LAYERS = 32
NUM_TRIALS = 3 # For random ablations

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def ablate_forward(model, tokenizer, inputs, layers_to_ablate):
    handle_list = []
    
    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h = torch.zeros_like(h) # Zero ablation
        return (h,) + out[1:] if is_tuple else h

    for layer in layers_to_ablate:
        handle_list.append(model.model.layers[layer].mlp.register_forward_hook(hook))
        
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
    for h in handle_list:
        h.remove()
    return out

def run_progressive_ablation():
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    val = [e for e in ds if e['split'] == 'validation']
    print(f"Validation Set Size: {len(val)}")

    # Split into quadrants
    quadrants = {
        "A_both_pass": [e for e in val if e['predictions']['SFT']['correct'] and e['predictions']['RLVR']['correct']],
        "C_both_fail": [e for e in val if not e['predictions']['SFT']['correct'] and not e['predictions']['RLVR']['correct']],
        "D_sft_fail_rlvr_pass": [e for e in val if not e['predictions']['SFT']['correct'] and e['predictions']['RLVR']['correct']]
    }
    
    print({k: len(v) for k,v in quadrants.items()})
    
    results = {model_name: {q_name: {k: [] for k in K_VALUES} for q_name in quadrants} for model_name in MODELS}

    for model_name, model_id in MODELS.items():
        print(f"\n=== Evaluating {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
        model.eval()

        for q_name, examples in quadrants.items():
            if len(examples) == 0: continue
            
            # Evaluate on a random subset of 10 examples per quadrant to save time during this structural test
            eval_examples = random.sample(examples, min(10, len(examples)))
            
            for k in K_VALUES:
                trial_accuracies = []
                for trial in range(NUM_TRIALS):
                    layers_to_ablate = random.sample(range(NUM_LAYERS), k)
                    corrects = []
                    
                    for e in tqdm(eval_examples, desc=f"{model_name} | {q_name} | K={k} | T={trial}", leave=False):
                        inp = tokenizer(format_prompt(tokenizer, e['prompt']), return_tensors="pt").to(model.device)
                        out = ablate_forward(model, tokenizer, inp, layers_to_ablate)
                        pred = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).split("#### ")[-1].strip().lower()
                        true = str(e['true_answer']).strip().lower()
                        hit = (true in pred) if e['source'] != "gsm8k" else (true == pred)
                        corrects.append(hit)
                        
                    acc = np.mean(corrects) * 100
                    trial_accuracies.append(acc)
                
                results[model_name][q_name][k] = float(np.mean(trial_accuracies))
                print(f"  {q_name} | K={k}: {results[model_name][q_name][k]:.1f}%")

        del model; clear_gpu()

    with open('/marimo/exp_phase12_redundancy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to /marimo/exp_phase12_redundancy_results.json")

if __name__ == "__main__":
    run_progressive_ablation()
