"""
exp14 v3 - Causal Trajectory Transfer (with resume support baked in)
Resumes from /marimo/exp14_production_results.json if it exists.
Skips quadrants already at 60 examples.
"""

import gc
import json
import os
import re
import time

import torch

# ── config ────────────────────────────────────────────────────────────────────
MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B",
}
INDIVIDUAL_LAYERS = [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 16, 20, 24, 28, 31]
CUMULATIVE_SEGMENTS = [
    (0, 4),
    (0, 6),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 14),
    (0, 16),
    (0, 20),
    (0, 24),
    (0, 28),
    (0, 31),
]
TARGET_PATCH_EXAMPLES = 60
DATASET_EVAL_N = 2000
MAX_NEW_TOKENS = 256
OUT_PATH = "/marimo/exp14_production_results.json"
DATASET_PATH = "/marimo/exp14_gsm8k_quadrants.json"
LOG_PATH = "/marimo/exp14_production.log"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def extract_answer(text):
    if not text:
        return ""
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


def is_correct(pred_text, gold_answer):
    pred = extract_answer(pred_text)
    gold = gold_answer.strip().replace(",", "")
    return pred == gold and pred != ""


def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


# ── Phase 1: vLLM dataset build ───────────────────────────────────────────────
def build_dataset_vllm():
    if os.path.exists(DATASET_PATH):
        log(f"Loading cached dataset from {DATASET_PATH}")
        with open(DATASET_PATH) as f:
            data = json.load(f)
        for k, v in data.items():
            log(f"  {k}: {len(v)} examples")
        return data

    log("Phase 1: Building dataset with vLLM ...")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split=f"train[:{DATASET_EVAL_N}]")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"])

    golds = []
    for ex in gsm:
        gold = extract_answer(ex["answer"])
        if not gold:
            continue
        fmt = format_prompt(tokenizer, ex["question"])
        golds.append({"prompt": ex["question"], "gold": gold, "formatted": fmt})

    log(f"  {len(golds)} prompts formatted")
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    for model_key in ("SFT", "RLVR"):
        log(f"  Running {model_key} ...")
        llm = LLM(
            model=MODELS[model_key],
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
        )
        outputs = llm.generate([g["formatted"] for g in golds], sampling)
        for i, g in enumerate(golds):
            g[f"{model_key}_text"] = outputs[i].outputs[0].text
            g[f"{model_key}_correct"] = is_correct(
                outputs[i].outputs[0].text, g["gold"]
            )
        del llm
        clear_gpu()

    results = {"A_Core": [], "B_SFT_only": [], "C_Both_fail": [], "D_Frontier": []}
    for g in golds:
        sft_ok = g.get("SFT_correct", False)
        rlvr_ok = g.get("RLVR_correct", False)
        if sft_ok and rlvr_ok:
            key = "A_Core"
        elif sft_ok and not rlvr_ok:
            key = "B_SFT_only"
        elif not sft_ok and not rlvr_ok:
            key = "C_Both_fail"
        else:
            key = "D_Frontier"
        g["quadrant"] = key
        results[key].append(g)

    for k, v in results.items():
        log(f"  {k}: {len(v)}")

    with open(DATASET_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Saved to {DATASET_PATH}")
    return results


# ── Phase 2: activation patching ─────────────────────────────────────────────
def cache_prefill(model, inputs, layers):
    cache = {}
    handles = []

    def hook(mod, inp, out, l_idx):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:
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
    handles = []

    def hook(mod, inp, out, l_idx):
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


def run_patching(quadrants, existing_results=None):
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("Phase 2: Activation Patching")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    log("Loading SFT model ...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    log("Loading RLVR model ...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {used:.1f}/{total:.0f} GB")

    all_layers = sorted(
        set(INDIVIDUAL_LAYERS)
        | {l for s, e in CUMULATIVE_SEGMENTS for l in range(s, e + 1)}
    )
    eos_id = tokenizer.eos_token_id

    # Build result container, pre-seeded with any existing data
    res = {
        "metadata": {
            "individual_layers": INDIVIDUAL_LAYERS,
            "cumulative_segments": [f"L{s}-{e}" for s, e in CUMULATIVE_SEGMENTS],
            "quadrant_sizes": {k: len(v) for k, v in quadrants.items()},
        },
        "RLVR_to_SFT": {},
        "SFT_to_RLVR": {},
    }
    for direction in ("RLVR_to_SFT", "SFT_to_RLVR"):
        for q in quadrants:
            # Start from existing data if available, else empty
            if existing_results and q in existing_results.get(direction, {}):
                res[direction][q] = existing_results[direction][q]
            else:
                res[direction][q] = {
                    "individual": {str(l): [] for l in INDIVIDUAL_LAYERS},
                    "cumulative": {f"L{s}-{e}": [] for s, e in CUMULATIVE_SEGMENTS},
                }

    for q_name, examples in quadrants.items():
        subset = examples[:TARGET_PATCH_EXAMPLES]
        log(f"\n--- Quadrant: {q_name}  ({len(subset)} examples) ---")

        for idx, ex in enumerate(tqdm(subset, desc=q_name)):
            prompt = ex["prompt"]
            gold = ex["gold"]
            fmt = format_prompt(tokenizer, prompt)
            inp = tokenizer(fmt, return_tensors="pt").to(sft_model.device)

            try:
                rlvr_cache = cache_prefill(rlvr_model, inp, all_layers)
                sft_cache = cache_prefill(sft_model, inp, all_layers)

                for l in INDIVIDUAL_LAYERS:
                    lkey = str(l)
                    gen = patched_generate(
                        sft_model, inp, rlvr_cache, [l], MAX_NEW_TOKENS, eos_id
                    )
                    res["RLVR_to_SFT"][q_name]["individual"][lkey].append(
                        int(
                            is_correct(
                                tokenizer.decode(gen[0], skip_special_tokens=True), gold
                            )
                        )
                    )
                    gen = patched_generate(
                        rlvr_model, inp, sft_cache, [l], MAX_NEW_TOKENS, eos_id
                    )
                    res["SFT_to_RLVR"][q_name]["individual"][lkey].append(
                        int(
                            is_correct(
                                tokenizer.decode(gen[0], skip_special_tokens=True), gold
                            )
                        )
                    )

                for seg_start, seg_end in CUMULATIVE_SEGMENTS:
                    seg_layers = list(range(seg_start, seg_end + 1))
                    skey = f"L{seg_start}-{seg_end}"
                    gen = patched_generate(
                        sft_model, inp, rlvr_cache, seg_layers, MAX_NEW_TOKENS, eos_id
                    )
                    hit_r2s = int(
                        is_correct(
                            tokenizer.decode(gen[0], skip_special_tokens=True), gold
                        )
                    )
                    res["RLVR_to_SFT"][q_name]["cumulative"][skey].append(hit_r2s)
                    gen = patched_generate(
                        rlvr_model, inp, sft_cache, seg_layers, MAX_NEW_TOKENS, eos_id
                    )
                    hit_s2r = int(
                        is_correct(
                            tokenizer.decode(gen[0], skip_special_tokens=True), gold
                        )
                    )
                    res["SFT_to_RLVR"][q_name]["cumulative"][skey].append(hit_s2r)

                r12 = res["RLVR_to_SFT"][q_name]["cumulative"].get("L0-12", [])
                s12 = res["SFT_to_RLVR"][q_name]["cumulative"].get("L0-12", [])
                log(
                    f"  [{q_name}] ex {idx + 1:3d}/{len(subset)}  gold={gold:>8s}  "
                    f"R->S@L0-12={'Y' if r12 and r12[-1] else 'N'}  "
                    f"S->R@L0-12={'Y' if s12 and s12[-1] else 'N'}"
                )

                del rlvr_cache, sft_cache
                clear_gpu()

            except Exception as exc:
                log(f"  ERROR [{q_name}] ex {idx}: {exc}")
                import traceback

                traceback.print_exc()
                clear_gpu()
                continue

        # Save checkpoint after each quadrant
        with open(OUT_PATH, "w") as f:
            json.dump(res, f, indent=2)
        log(f"Checkpoint saved -> {OUT_PATH}")

    return res


def print_summary(res):
    import numpy as np

    def ci(lst):
        if not lst:
            return 0.0, 0.0, 0.0
        arr = np.array(lst, dtype=float) * 100
        mean = np.mean(arr)
        boot = np.random.choice(arr, (2000, len(arr)), replace=True)
        lo, hi = np.percentile(np.mean(boot, axis=1), [2.5, 97.5])
        return mean, lo, hi

    log("\n" + "=" * 72)
    log("RESULTS: RLVR->SFT Cumulative Patching -- Dataset D (Frontier)")
    log("=" * 72)
    for seg in [f"L{s}-{e}" for s, e in CUMULATIVE_SEGMENTS]:
        r2s = (
            res["RLVR_to_SFT"].get("D_Frontier", {}).get("cumulative", {}).get(seg, [])
        )
        s2r = (
            res["SFT_to_RLVR"].get("D_Frontier", {}).get("cumulative", {}).get(seg, [])
        )
        m1, l1, u1 = ci(r2s)
        m2, l2, u2 = ci(s2r)
        log(
            f"  {seg:8s}  RLVR->SFT: {m1:5.1f}% [{l1:.1f}-{u1:.1f}] n={len(r2s)}"
            f"    SFT->RLVR: {m2:5.1f}% [{l2:.1f}-{u2:.1f}] n={len(s2r)}"
        )

    log("\nCROSS-QUADRANT: L0-12 Cumulative")
    for q in res.get("RLVR_to_SFT", {}):
        r2s = res["RLVR_to_SFT"][q]["cumulative"].get("L0-12", [])
        s2r = res["SFT_to_RLVR"][q]["cumulative"].get("L0-12", [])
        m1, l1, u1 = ci(r2s)
        m2, l2, u2 = ci(s2r)
        log(
            f"  {q:15s}  RLVR->SFT: {m1:5.1f}% [{l1:.1f}-{u1:.1f}] n={len(r2s)}"
            f"    SFT->RLVR: {m2:5.1f}% [{l2:.1f}-{u2:.1f}] n={len(s2r)}"
        )


if __name__ == "__main__":
    log("=== exp14 v3 -- Causal Trajectory Transfer ===")
    log(f"Output  : {OUT_PATH}")
    log(f"Dataset : {DATASET_PATH}")

    quadrants = build_dataset_vllm()

    # Load existing checkpoint, skip completed quadrants
    existing = {}
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
        for q, d in existing.get("RLVR_to_SFT", {}).items():
            if len(d.get("cumulative", {}).get("L0-12", [])) >= TARGET_PATCH_EXAMPLES:
                done.add(q)
        if done:
            log(f"[RESUME] Skipping completed quadrants: {done}")
            quadrants = {k: v for k, v in quadrants.items() if k not in done}

    res = run_patching(quadrants, existing_results=existing)

    # Merge any skipped quadrants back into final results
    for direction in ("RLVR_to_SFT", "SFT_to_RLVR"):
        for q, d in existing.get(direction, {}).items():
            if q not in res.get(direction, {}):
                res.setdefault(direction, {})[q] = d

    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    log(f"\nFinal results -> {OUT_PATH}")

    print_summary(res)
    log("Done.")
