"""
Phase E — Activation Steering (Killer Experiment)

1. From Discovery set, compute the success→failure mean-diff direction d_L at layer L (L=14 by default).
2. On Validation set SFT-Fail examples, add α·d_L to h_L during SFT forward pass.
3. Measure ΔP(correct) over a range of α values.

If steering SFT toward the RLVR-associated state rescues hard problems, we have
causal evidence that RLVR changes the state-selection policy over a manipulable
latent variable.
"""
import json
import torch
import gc
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT_MODEL_ID  = "allenai/Llama-3.1-Tulu-3-8B-SFT"
RLVR_MODEL_ID = "allenai/Llama-3.1-Tulu-3-8B"

# The layer where the state becomes causally sufficient (from Exp 9 curve: L14)
STEER_LAYER = 14
ALPHAS      = [0.0, 0.5, 1.0, 2.0, 4.0]

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

    for prompt in tqdm(prompts, desc=f"Extract L{layer}"):
        current = {}
        inp = tokenizer(format_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inp)
        acts.append(current[0])

    handle.remove()
    return acts

def extract_correct(model, tokenizer, prompts, true_answers, sources):
    """Greedy decode and check exact/loose match."""
    correct = []
    for prompt, ans, src in tqdm(zip(prompts, true_answers, sources), total=len(prompts), desc="Evaluating"):
        inp = tokenizer(format_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        pred = gen.split("#### ")[-1].strip().lower()
        true = str(ans).strip().lower()
        hit  = (true in pred) if src != "gsm8k" else (true == pred)
        correct.append(hit)
    return correct

def steered_forward(model, inputs, layer, direction_gpu, alpha):
    """Run forward pass with h_L += alpha * direction injected."""
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
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=model.config.eos_token_id)
    for h in handle_list:
        h.remove()
    return out

def run_activation_steering():
    try:
        with open('/marimo/semantic_agreement_dataset_full.json', 'r') as f:
            ds = json.load(f)
    except FileNotFoundError:
        print("Dataset not found. Aborting.")
        return

    disc = [e for e in ds if e['split'] == 'discovery']
    val  = [e for e in ds if e['split'] == 'validation']
    print(f"Discovery: {len(disc)}, Validation: {len(val)}")

    # ── Step 1: Compute steering direction on Discovery set using SFT acts ──
    # We want: d = mean(h | SFT success) - mean(h | SFT failure)
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_ID, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    disc_prompts = [e['prompt'] for e in disc]
    disc_sft_labels = np.array([e['predictions']['SFT']['correct'] for e in disc])

    print("\n--- Loading SFT for direction extraction ---")
    clear_gpu()
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    sft_model.eval()

    disc_acts = extract_h_l(sft_model, tokenizer, disc_prompts, STEER_LAYER)
    disc_acts = torch.stack(disc_acts)  # (N, D)

    success_mask = disc_sft_labels == True
    failure_mask = disc_sft_labels == False

    if success_mask.sum() == 0 or failure_mask.sum() == 0:
        print("Not enough class diversity in Discovery set for SFT direction. Using RLVR direction instead.")
        del sft_model; clear_gpu()

        rlvr_model = AutoModelForCausalLM.from_pretrained(RLVR_MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
        rlvr_model.eval()
        disc_rlvr_labels = np.array([e['predictions']['RLVR']['correct'] for e in disc])
        disc_rlvr_acts = extract_h_l(rlvr_model, tokenizer, disc_prompts, STEER_LAYER)
        disc_rlvr_acts = torch.stack(disc_rlvr_acts)
        del rlvr_model; clear_gpu()

        # Also extract SFT on discovery again for mean
        sft_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
        sft_model.eval()

        # direction: RLVR mean - SFT mean on same discovery set
        d = disc_rlvr_acts.mean(0) - disc_acts.mean(0)
    else:
        d = disc_acts[success_mask].mean(0) - disc_acts[failure_mask].mean(0)

    # Normalize direction
    d = d / (d.norm() + 1e-8)
    print(f"Steering direction norm: {d.norm().item():.4f}")

    # ── Step 2: Evaluate on Validation SFT-Fail examples with alpha sweep ──
    val_fail = [e for e in val if not e['predictions']['SFT']['correct']]
    print(f"\nValidation SFT-Fail examples: {len(val_fail)}")

    if len(val_fail) == 0:
        print("No SFT failures in validation set. Aborting steering experiment.")
        return

    val_prompts    = [e['prompt']      for e in val_fail]
    val_answers    = [e['true_answer'] for e in val_fail]
    val_sources    = [e['source']      for e in val_fail]

    results = {}

    for alpha in ALPHAS:
        print(f"\n--- Alpha = {alpha} ---")
        steered_correct = []

        for prompt, ans, src in tqdm(zip(val_prompts, val_answers, val_sources),
                                     total=len(val_prompts), desc=f"α={alpha}"):
            inp = tokenizer(format_prompt(tokenizer, prompt), return_tensors="pt").to(sft_model.device)

            if alpha == 0.0:
                with torch.no_grad():
                    out = sft_model.generate(**inp, max_new_tokens=256, do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id)
            else:
                out = steered_forward(sft_model, inp, STEER_LAYER, d, alpha)

            gen  = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
            pred = gen.split("#### ")[-1].strip().lower()
            true = str(ans).strip().lower()
            hit  = (true in pred) if src != "gsm8k" else (true == pred)
            steered_correct.append(hit)

        acc = np.mean(steered_correct) * 100
        print(f"  α={alpha}: {acc:.1f}% correct on SFT-Fail Validation")
        results[str(alpha)] = float(acc)

    del sft_model; clear_gpu()

    print("\n=== Activation Steering Results ===")
    for alpha, acc in results.items():
        print(f"  α={alpha}: {acc:.1f}%")

    with open('/marimo/exp_phaseE_activation_steering.json', 'w') as f:
        json.dump({"steer_layer": STEER_LAYER, "results": results}, f, indent=2)
    print("Done. Saved to /marimo/exp_phaseE_activation_steering.json")

if __name__ == "__main__":
    run_activation_steering()
