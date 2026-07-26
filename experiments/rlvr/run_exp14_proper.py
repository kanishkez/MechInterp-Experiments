"""
exp14 — Causal Trajectory Transfer (production run)

Fixes vs. previous attempt:
 - Builds dataset from scratch using GSM8K (clean, verifiable gold answers)
 - Targets >= 50 examples per quadrant
 - Saves per-example correctness vectors so CIs can be computed post-hoc
 - Writes to /marimo/exp14_production_results.json and tees a .log
 - Prints per-example lines so there's a visible audit trail

Usage (on remote GPU):
    python run_exp14_proper.py 2>&1 | tee /marimo/exp14_production.log
"""

import gc
import json
import os
import re
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B",
}
INDIVIDUAL_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 31]
CUMULATIVE_SEGMENTS = [
    (0, 4),
    (0, 8),
    (0, 10),
    (0, 12),
    (0, 14),
    (0, 16),
    (0, 20),
    (0, 24),
    (0, 28),
    (0, 31),
]
MAX_EXAMPLES_PER_QUADRANT = 60  # aim for ≥50 after any failures
MAX_NEW_TOKENS = 200
BATCH_SIZE = 1  # safe for activation patching (no padding complexity)
OUT_PATH = "/marimo/exp14_production_results.json"
LOG_PATH = "/marimo/exp14_production.log"
DATASET_PATH = "/marimo/exp14_gsm8k_quadrants.json"


# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def extract_answer(text: str) -> str:
    """Extract final numeric answer from model output or gold."""
    if not text:
        return ""
    # GSM8K canonical: "#### <number>"
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Boxed
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Last number in text as fallback
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def is_correct(pred_text: str, gold_answer: str) -> bool:
    pred = extract_answer(pred_text)
    gold = gold_answer.strip().replace(",", "")
    return pred == gold


def format_prompt(tokenizer, prompt: str) -> str:
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


# ── activation patching ───────────────────────────────────────────────────────
def cache_prefill(model, inputs, layers):
    """Run a forward pass and capture residual stream after each requested layer."""
    cache = {}
    handles = []

    def hook(module, inp, out, l_idx):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:  # prefill only (seq_len > 1)
            cache[l_idx] = h.detach().clone()

    for l in layers:
        handles.append(
            model.model.layers[l].register_forward_hook(
                lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
            )
        )
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return cache


def patched_generate(model, inputs, cache, patch_layers, max_new_tokens, eos_id):
    """
    Greedy decode with activation patching on the prefill pass only.
    Avoids HF .generate() so the prefill hook fires exactly once.
    """
    handles = []

    def hook(module, inp, out, l_idx):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1 and l_idx in cache:
            h[:, :, :] = cache[l_idx]
        return (h,) + out[1:] if isinstance(out, tuple) else h

    for l in patch_layers:
        handles.append(
            model.model.layers[l].register_forward_hook(
                lambda m, i, o, l_idx=l: hook(m, i, o, l_idx)
            )
        )
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()

    generated = [next_tok]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_tok)
            if next_tok.item() == eos_id:
                break
    return torch.cat(generated, dim=1)


# ── dataset building ──────────────────────────────────────────────────────────
def build_quadrant_dataset(tokenizer, sft_model, rlvr_model, target_per_quadrant=60):
    """
    Evaluate both models on GSM8K train split, assign quadrant labels,
    return balanced sample. Saves intermediate to DATASET_PATH.
    """
    if os.path.exists(DATASET_PATH):
        log(f"Loading cached quadrant dataset from {DATASET_PATH}")
        with open(DATASET_PATH) as f:
            data = json.load(f)
        for k in data:
            log(f"  {k}: {len(data[k])} examples")
        return data

    log("Building quadrant dataset from GSM8K ...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    log(f"  Loaded {len(gsm)} GSM8K train examples")

    quadrants = {"A_Core": [], "B_SFT_only": [], "C_Both_fail": [], "D_Frontier": []}

    for i, ex in enumerate(tqdm(gsm, desc="Evaluating both models")):
        if all(len(v) >= target_per_quadrant for v in quadrants.values()):
            break

        prompt = ex["question"]
        gold = extract_answer(ex["answer"])
        if not gold:
            continue

        fmt = format_prompt(tokenizer, prompt)
        inp = tokenizer(fmt, return_tensors="pt").to(sft_model.device)

        try:
            with torch.no_grad():
                sft_out = sft_model.generate(
                    **inp,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                rlvr_out = rlvr_model.generate(
                    **inp,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            sft_text = tokenizer.decode(
                sft_out[0][inp.input_ids.shape[1] :], skip_special_tokens=True
            )
            rlvr_text = tokenizer.decode(
                rlvr_out[0][inp.input_ids.shape[1] :], skip_special_tokens=True
            )
            sft_ok = is_correct(sft_text, gold)
            rlvr_ok = is_correct(rlvr_text, gold)

            record = {
                "prompt": prompt,
                "gold": gold,
                "source": "gsm8k",
                "sft_text": sft_text,
                "rlvr_text": rlvr_text,
                "sft_correct": sft_ok,
                "rlvr_correct": rlvr_ok,
            }
            if sft_ok and rlvr_ok:
                key = "A_Core"
            elif sft_ok and not rlvr_ok:
                key = "B_SFT_only"
            elif not sft_ok and not rlvr_ok:
                key = "C_Both_fail"
            else:
                key = "D_Frontier"

            if len(quadrants[key]) < target_per_quadrant:
                quadrants[key].append(record)
                if i % 50 == 0:
                    log(
                        f"  i={i}  A={len(quadrants['A_Core'])}  B={len(quadrants['B_SFT_only'])}  C={len(quadrants['C_Both_fail'])}  D={len(quadrants['D_Frontier'])}"
                    )

            del sft_out, rlvr_out
            clear_gpu()
        except Exception as exc:
            log(f"  Error at i={i}: {exc}")
            clear_gpu()
            continue

    for k, v in quadrants.items():
        log(f"  Final {k}: {len(v)} examples")

    with open(DATASET_PATH, "w") as f:
        json.dump(quadrants, f, indent=2)
    log(f"Saved quadrant dataset to {DATASET_PATH}")
    return quadrants


# ── main experiment ───────────────────────────────────────────────────────────
def run_patching_experiment(quadrants, tokenizer, sft_model, rlvr_model):
    all_layers = sorted(
        set(
            INDIVIDUAL_LAYERS
            + [l for s, e in CUMULATIVE_SEGMENTS for l in range(s, e + 1)]
        )
    )

    results = {
        "metadata": {
            "individual_layers": INDIVIDUAL_LAYERS,
            "cumulative_segments": [f"L{s}-{e}" for s, e in CUMULATIVE_SEGMENTS],
            "quadrant_sizes": {k: len(v) for k, v in quadrants.items()},
        },
        "RLVR_to_SFT": {
            q: {
                "individual": {str(l): [] for l in INDIVIDUAL_LAYERS},
                "cumulative": {f"L{s}-{e}": [] for s, e in CUMULATIVE_SEGMENTS},
            }
            for q in quadrants
        },
        "SFT_to_RLVR": {
            q: {
                "individual": {str(l): [] for l in INDIVIDUAL_LAYERS},
                "cumulative": {f"L{s}-{e}": [] for s, e in CUMULATIVE_SEGMENTS},
            }
            for q in quadrants
        },
    }

    eos_id = tokenizer.eos_token_id

    for q_name, examples in quadrants.items():
        log(f"\n{'=' * 60}")
        log(f"Quadrant: {q_name}  ({len(examples)} examples)")
        log(f"{'=' * 60}")

        for idx, ex in enumerate(tqdm(examples, desc=q_name)):
            prompt = ex["prompt"]
            gold = ex["gold"]
            fmt = format_prompt(tokenizer, prompt)
            inp = tokenizer(fmt, return_tensors="pt").to(sft_model.device)

            try:
                # Cache activations from both models
                rlvr_cache = cache_prefill(rlvr_model, inp, all_layers)
                sft_cache = cache_prefill(sft_model, inp, all_layers)

                # ── Individual layers ───────────────────────────────────────
                for l in INDIVIDUAL_LAYERS:
                    key = str(l)
                    # RLVR → SFT
                    gen = patched_generate(
                        sft_model, inp, rlvr_cache, [l], MAX_NEW_TOKENS, eos_id
                    )
                    text_r2s = tokenizer.decode(gen[0], skip_special_tokens=True)
                    hit_r2s = is_correct(text_r2s, gold)
                    results["RLVR_to_SFT"][q_name]["individual"][key].append(
                        int(hit_r2s)
                    )

                    # SFT → RLVR
                    gen = patched_generate(
                        rlvr_model, inp, sft_cache, [l], MAX_NEW_TOKENS, eos_id
                    )
                    text_s2r = tokenizer.decode(gen[0], skip_special_tokens=True)
                    hit_s2r = is_correct(text_s2r, gold)
                    results["SFT_to_RLVR"][q_name]["individual"][key].append(
                        int(hit_s2r)
                    )

                # ── Cumulative segments ─────────────────────────────────────
                for seg_start, seg_end in CUMULATIVE_SEGMENTS:
                    seg_layers = list(range(seg_start, seg_end + 1))
                    seg_key = f"L{seg_start}-{seg_end}"

                    # RLVR → SFT
                    gen = patched_generate(
                        sft_model, inp, rlvr_cache, seg_layers, MAX_NEW_TOKENS, eos_id
                    )
                    text_r2s = tokenizer.decode(gen[0], skip_special_tokens=True)
                    hit_r2s = is_correct(text_r2s, gold)
                    results["RLVR_to_SFT"][q_name]["cumulative"][seg_key].append(
                        int(hit_r2s)
                    )

                    # SFT → RLVR
                    gen = patched_generate(
                        rlvr_model, inp, sft_cache, seg_layers, MAX_NEW_TOKENS, eos_id
                    )
                    text_s2r = tokenizer.decode(gen[0], skip_special_tokens=True)
                    hit_s2r = is_correct(text_s2r, gold)
                    results["SFT_to_RLVR"][q_name]["cumulative"][seg_key].append(
                        int(hit_s2r)
                    )

                # Per-example log line
                cum_r2s = results["RLVR_to_SFT"][q_name]["cumulative"].get("L0-12", [])
                cum_s2r = results["SFT_to_RLVR"][q_name]["cumulative"].get("L0-12", [])
                log(
                    f"  [{q_name}] ex {idx + 1}/{len(examples)}  gold={gold}  "
                    f"R→S@L0-12={cum_r2s[-1] if cum_r2s else '?'}  "
                    f"S→R@L0-12={cum_s2r[-1] if cum_s2r else '?'}"
                )

                del rlvr_cache, sft_cache
                clear_gpu()

            except Exception as exc:
                log(f"  ERROR [{q_name}] ex {idx}: {exc}")
                clear_gpu()
                continue

        # Checkpoint after each quadrant
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        log(f"Checkpoint saved → {OUT_PATH}")

    return results


# ── summary ───────────────────────────────────────────────────────────────────
def print_summary(results):
    import numpy as np

    def ci(lst):
        if not lst:
            return 0.0, 0.0, 0.0
        arr = np.array(lst, dtype=float) * 100
        mean = np.mean(arr)
        boot = np.random.choice(arr, (2000, len(arr)), replace=True)
        lo, hi = np.percentile(np.mean(boot, axis=1), [2.5, 97.5])
        return mean, lo, hi

    log("\n" + "=" * 70)
    log("CUMULATIVE PATCHING SUMMARY — Dataset D (Frontier)")
    log("=" * 70)
    for seg in [f"L{s}-{e}" for s, e in CUMULATIVE_SEGMENTS]:
        r2s = (
            results["RLVR_to_SFT"]
            .get("D_Frontier", {})
            .get("cumulative", {})
            .get(seg, [])
        )
        s2r = (
            results["SFT_to_RLVR"]
            .get("D_Frontier", {})
            .get("cumulative", {})
            .get(seg, [])
        )
        m1, l1, u1 = ci(r2s)
        m2, l2, u2 = ci(s2r)
        log(
            f"  {seg:8s}  RLVR→SFT: {m1:5.1f}% [{l1:.1f}–{u1:.1f}]  n={len(r2s)}"
            f"    SFT→RLVR: {m2:5.1f}% [{l2:.1f}–{u2:.1f}]  n={len(s2r)}"
        )

    log("\nINDIVIDUAL LAYER SUMMARY — Dataset D")
    for l in INDIVIDUAL_LAYERS:
        key = str(l)
        r2s = (
            results["RLVR_to_SFT"]
            .get("D_Frontier", {})
            .get("individual", {})
            .get(key, [])
        )
        s2r = (
            results["SFT_to_RLVR"]
            .get("D_Frontier", {})
            .get("individual", {})
            .get(key, [])
        )
        m1, l1, u1 = ci(r2s)
        m2, l2, u2 = ci(s2r)
        log(
            f"  L{key:2s}  RLVR→SFT: {m1:5.1f}% [{l1:.1f}–{u1:.1f}]  n={len(r2s)}"
            f"    SFT→RLVR: {m2:5.1f}% [{l2:.1f}–{u2:.1f}]  n={len(s2r)}"
        )

    log("\nCROSS-QUADRANT COMPARISON — L0-12 Cumulative")
    for q in results["RLVR_to_SFT"]:
        r2s = results["RLVR_to_SFT"][q]["cumulative"].get("L0-12", [])
        s2r = results["SFT_to_RLVR"][q]["cumulative"].get("L0-12", [])
        m1, l1, u1 = ci(r2s)
        m2, l2, u2 = ci(s2r)
        log(
            f"  {q:15s}  RLVR→SFT@L0-12: {m1:5.1f}% [{l1:.1f}–{u1:.1f}]  n={len(r2s)}"
            f"    SFT→RLVR@L0-12: {m2:5.1f}% [{l2:.1f}–{u2:.1f}]  n={len(s2r)}"
        )


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("=== exp14 Production Run — Causal Trajectory Transfer ===")
    log(f"Output: {OUT_PATH}")

    log("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading SFT model ...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    log(f"  SFT on {next(sft_model.parameters()).device}")

    log("Loading RLVR model ...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    log(f"  RLVR on {next(rlvr_model.parameters()).device}")

    # GPU status
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {used:.1f}/{total:.0f} GB used after model load")

    quadrants = build_quadrant_dataset(
        tokenizer, sft_model, rlvr_model, target_per_quadrant=MAX_EXAMPLES_PER_QUADRANT
    )
    results = run_patching_experiment(quadrants, tokenizer, sft_model, rlvr_model)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nFinal results saved → {OUT_PATH}")

    print_summary(results)
    log("\nDone.")
