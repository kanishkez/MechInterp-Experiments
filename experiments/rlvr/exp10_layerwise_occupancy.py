import json
import torch
import gc
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}
PROBE_LAYERS = [5, 10, 15, 20, 25, 31]

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def extract_at_layers(model, tokenizer, prompts, layers):
    acts = {l: [] for l in layers}
    current = {}

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 3:
                current[l] = h[0, -1, :].detach().to(torch.float32).cpu().numpy()
            else:
                current[l] = h[-1, :].detach().to(torch.float32).cpu().numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]

    for prompt in tqdm(prompts, desc="Extracting"):
        current = {}
        formatted = format_prompt(tokenizer, prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for l in layers:
            acts[l].append(current[l])

    for h in handles:
        h.remove()

    return {l: np.array(acts[l]) for l in layers}

def run_layerwise_occupancy():
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found. Aborting.")
        return

    # Separate discovery (train clusters) and validation (test occupancy)
    disc = [e for e in ds if e['split'] == 'discovery']
    val  = [e for e in ds if e['split'] == 'validation']
    print(f"Discovery: {len(disc)}, Validation: {len(val)}")

    prompts_disc = [e['prompt'] for e in disc]
    prompts_val  = [e['prompt'] for e in val]

    y_disc_sft  = np.array([e['predictions']['SFT']['correct']  for e in disc])
    y_disc_rlvr = np.array([e['predictions']['RLVR']['correct'] for e in disc])
    y_val_sft   = np.array([e['predictions']['SFT']['correct']  for e in val])
    y_val_rlvr  = np.array([e['predictions']['RLVR']['correct'] for e in val])

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"])

    acts_disc = {}
    acts_val  = {}

    for model_name, model_id in MODELS.items():
        print(f"\n--- {model_name} ---")
        clear_gpu()
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
        model.eval()
        acts_disc[model_name] = extract_at_layers(model, tokenizer, prompts_disc, PROBE_LAYERS)
        acts_val[model_name]  = extract_at_layers(model, tokenizer, prompts_val,  PROBE_LAYERS)
        del model
        clear_gpu()

    results = {}

    for l in PROBE_LAYERS:
        print(f"\n=== Layer {l} ===")
        # Cluster on discovery using SFT acts (shared PCA space for both models)
        X_disc_sft  = acts_disc["SFT"][l]
        X_disc_rlvr = acts_disc["RLVR"][l]
        X_all_disc  = np.vstack([X_disc_sft, X_disc_rlvr])

        pca = PCA(n_components=min(20, X_all_disc.shape[1]))
        X_pca_disc = pca.fit_transform(X_all_disc)

        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(X_pca_disc)

        # Now project validation data and predict cluster
        X_val_sft  = pca.transform(acts_val["SFT"][l])
        X_val_rlvr = pca.transform(acts_val["RLVR"][l])
        c_val_sft  = km.predict(X_val_sft)
        c_val_rlvr = km.predict(X_val_rlvr)

        # Discover which cluster corresponds to success:
        # label success-associated cluster as the one where RLVR correct examples dominate
        disc_rlvr_clusters = km.predict(X_pca_disc[len(X_disc_sft):])
        cluster_success = {}
        for k in range(3):
            mask = disc_rlvr_clusters == k
            if mask.sum() > 0:
                cluster_success[k] = y_disc_rlvr[mask].mean()
            else:
                cluster_success[k] = 0.0
        success_cluster = max(cluster_success, key=cluster_success.get)

        p_success_sft  = np.mean(c_val_sft  == success_cluster)
        p_success_rlvr = np.mean(c_val_rlvr == success_cluster)

        print(f"  Success cluster={success_cluster}")
        print(f"  P(success | SFT)  = {p_success_sft*100:.1f}%")
        print(f"  P(success | RLVR) = {p_success_rlvr*100:.1f}%")

        results[l] = {
            "success_cluster": int(success_cluster),
            "P_success_SFT":   float(p_success_sft),
            "P_success_RLVR":  float(p_success_rlvr)
        }

    with open('/marimo/exp_phaseC_layerwise_occupancy.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDone. Results saved to /marimo/exp_phaseC_layerwise_occupancy.json")

if __name__ == "__main__":
    run_layerwise_occupancy()
