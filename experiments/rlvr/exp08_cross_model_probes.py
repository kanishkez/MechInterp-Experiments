import json
import torch
import gc
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}
PROBE_LAYER = 15

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def extract_h_l(model, tokenizer, prompts):
    acts = []
    handles = []
    current_act = {}
    
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.dim() == 3:
            current_act[0] = h[0, -1, :].detach().to(torch.float32).cpu().clone().numpy()
        else:
            current_act[0] = h[-1, :].detach().to(torch.float32).cpu().clone().numpy()
            
    h = model.model.layers[PROBE_LAYER].register_forward_hook(hook)
        
    for prompt in tqdm(prompts, desc=f"Extracting L{PROBE_LAYER} acts"):
        current_act = {}
        formatted = format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            model(**inputs)
            
        acts.append(current_act[0])
        
    h.remove()
    return np.array(acts)

def run_cross_model_probe():
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found. Aborting.")
        return
        
    # Split into discovery and validation
    discovery = [e for e in ds if e['split'] == 'discovery']
    validation = [e for e in ds if e['split'] == 'validation']
    print(f"Loaded {len(discovery)} Discovery and {len(validation)} Validation examples.")
    
    prompts_train = [p['prompt'] for p in discovery]
    prompts_test = [p['prompt'] for p in validation]
    prompts_all = prompts_train + prompts_test
    
    labels_sft_train = [p['predictions']['SFT']['correct'] for p in discovery]
    labels_sft_test = [p['predictions']['SFT']['correct'] for p in validation]
    
    labels_rlvr_train = [p['predictions']['RLVR']['correct'] for p in discovery]
    labels_rlvr_test = [p['predictions']['RLVR']['correct'] for p in validation]
    
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"])
    
    all_acts = {}
    for model_name, model_id in MODELS.items():
        print(f"\n--- Loading {model_name} for extraction ---")
        clear_gpu()
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
        model.eval()
        acts = extract_h_l(model, tokenizer, prompts_all)
        all_acts[model_name] = {
            "train": acts[:len(prompts_train)],
            "test": acts[len(prompts_train):]
        }
        del model
        clear_gpu()
        
    print("\n--- Training Probes ---")
    
    probes = {}
    scalers = {}
    
    # Train probes
    for train_model in ["SFT", "RLVR"]:
        X_train = all_acts[train_model]["train"]
        y_train = labels_sft_train if train_model == "SFT" else labels_rlvr_train
        
        if len(set(y_train)) < 2:
            print(f"Skipping {train_model} probe: Only 1 class present in training data.")
            continue
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train_scaled, y_train)
        
        probes[train_model] = clf
        scalers[train_model] = scaler
        
    # Evaluate 2x2 Matrix
    matrix = {}
    print("\n--- 2x2 Transfer Matrix (Validation Set) ---")
    
    for train_model in ["SFT", "RLVR"]:
        if train_model not in probes:
            continue
            
        clf = probes[train_model]
        scaler = scalers[train_model]
        
        for test_model in ["SFT", "RLVR"]:
            X_test = all_acts[test_model]["test"]
            y_test = labels_sft_test if test_model == "SFT" else labels_rlvr_test
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"Train {train_model} -> Test {test_model}: {acc*100:.1f}%")
            matrix[f"{train_model}_to_{test_model}"] = float(acc)
            
    # Calculate Cosine Similarity of weights
    if "SFT" in probes and "RLVR" in probes:
        w_sft = probes["SFT"].coef_[0]
        w_rlvr = probes["RLVR"].coef_[0]
        
        cos_sim = np.dot(w_sft, w_rlvr) / (np.linalg.norm(w_sft) * np.linalg.norm(w_rlvr))
        print(f"\nCosine similarity of probe weights: {cos_sim:.4f}")
        matrix["cosine_similarity"] = float(cos_sim)
        
    with open('/marimo/exp8_2x2_transfer_matrix.json', 'w') as f:
        json.dump(matrix, f, indent=2)
        
if __name__ == "__main__":
    run_cross_model_probe()
