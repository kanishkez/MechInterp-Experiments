"""
exp14 v5 - Causal Trajectory Transfer (methodologically corrected)

Fixes applied after auditing v4's results:
 1. Quadrant labels built with the SAME decode pipeline used for patching
    (fixed attention_mask extension, batched HF generate) -- eliminates
    the vLLM-vs-HF inference engine mismatch that made v4's "recovery"
    indistinguishable from baseline noise.
 2. Reports an EXPLICIT unpatched baseline (no hooks at all) alongside
    every patched condition, computed on the same batches, so results
    are baseline-adjusted deltas, not raw accuracy.
 3. Uses the corrected attention_mask-extension decode loop everywhere.

Dataset: /marimo/exp14_gsm8k_quadrants_v2.json (built by exp14_rebuild_v2.py)
"""

import gc
import json
import os
import re
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8
OUT_PATH = "/marimo/exp14_v5_corrected_results.json"
DATASET_PATH = "/marimo/exp14_gsm8k_quadrants_v2.json"
LOG_PATH = "/marimo/exp14_v5.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


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


def is_correct(pred_text, gold):
    pred = extract_answer(pred_text)
    return pred == gold.strip().replace(",", "") and pred != ""


def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def cache_prefill_batch(model, input_ids, attention_mask, layers):
    cache = {}
    handles = []

    def make_hook(l_idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                cache[l_idx] = h.detach().clone()

        return hook

    for l in layers:
        handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return cache


def patched_generate_batch(
    model,
    input_ids,
    attention_mask,
    cache,
    patch_layers,
    max_new_tokens,
    eos_id,
    pad_id,
):
    """CORRECTED: attention_mask is properly extended at every decode step."""
    handles = []

    def make_hook(l_idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1 and l_idx in cache:
                h[:, :, :] = cache[l_idx]
            return (h,) + out[1:] if isinstance(out, tuple) else h

        return hook

    for l in patch_layers:
        handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True
            )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()

    B = input_ids.shape[0]
    generated = [next_tok]
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
    cur_mask = attention_mask

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            feed = next_tok.clone()
            feed[done] = pad_id
            cur_mask = torch.cat(
                [
                    cur_mask,
                    torch.ones((B, 1), dtype=cur_mask.dtype, device=cur_mask.device),
                ],
                dim=1,
            )
            out = model(
                input_ids=feed,
                past_key_values=past_kv,
                attention_mask=cur_mask,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            next_tok[done] = pad_id
            generated.append(next_tok)
            done |= next_tok.squeeze(-1) == eos_id
            if done.all():
                break
    return torch.cat(generated, dim=1)


def run_patching(quadrants, existing_results=None):
    log("Loading tokenizer + models ...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    device = next(sft_model.parameters()).device

    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {used:.1f}/{total:.0f} GB")

    all_layers = sorted(
        set(INDIVIDUAL_LAYERS)
        | {l for s, e in CUMULATIVE_SEGMENTS for l in range(s, e + 1)}
    )
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    res = {
        "metadata": {
            "individual_layers": INDIVIDUAL_LAYERS,
            "cumulative_segments": [f"L{s}-{e}" for s, e in CUMULATIVE_SEGMENTS],
            "quadrant_sizes": {k: len(v) for k, v in quadrants.items()},
            "batch_size": BATCH_SIZE,
            "notes": "v5: self-consistent quadrant labeling + fixed attention_mask decode",
        },
        "RLVR_to_SFT": {},
        "SFT_to_RLVR": {},
        "SFT_unpatched_baseline": {},
        "RLVR_unpatched_baseline": {},
    }
    for direction in ("RLVR_to_SFT", "SFT_to_RLVR"):
        for q in quadrants:
            if existing_results and q in existing_results.get(direction, {}):
                res[direction][q] = existing_results[direction][q]
            else:
                res[direction][q] = {
                    "individual": {str(l): [] for l in INDIVIDUAL_LAYERS},
                    "cumulative": {f"L{s}-{e}": [] for s, e in CUMULATIVE_SEGMENTS},
                }
    for q in quadrants:
        res["SFT_unpatched_baseline"][q] = (
            existing_results.get("SFT_unpatched_baseline", {}).get(q, [])
            if existing_results
            else []
        )
        res["RLVR_unpatched_baseline"][q] = (
            existing_results.get("RLVR_unpatched_baseline", {}).get(q, [])
            if existing_results
            else []
        )

    for q_name, examples in quadrants.items():
        subset = examples[:TARGET_PATCH_EXAMPLES]
        log(f"\n--- Quadrant: {q_name} ({len(subset)} examples) ---")

        for batch_start in range(0, len(subset), BATCH_SIZE):
            batch = subset[batch_start : batch_start + BATCH_SIZE]
            prompts = [format_prompt(tokenizer, ex["prompt"]) for ex in batch]
            golds = [ex["gold"] for ex in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
            B = input_ids.shape[0]

            try:
                rlvr_cache = cache_prefill_batch(
                    rlvr_model, input_ids, attention_mask, all_layers
                )
                sft_cache = cache_prefill_batch(
                    sft_model, input_ids, attention_mask, all_layers
                )

                # ── Unpatched baselines (empty patch list) ──────────────────
                gen_sft_base = patched_generate_batch(
                    sft_model,
                    input_ids,
                    attention_mask,
                    {},
                    [],
                    MAX_NEW_TOKENS,
                    eos_id,
                    pad_id,
                )
                gen_rlvr_base = patched_generate_batch(
                    rlvr_model,
                    input_ids,
                    attention_mask,
                    {},
                    [],
                    MAX_NEW_TOKENS,
                    eos_id,
                    pad_id,
                )
                for j in range(B):
                    t_sft = tokenizer.decode(gen_sft_base[j], skip_special_tokens=True)
                    t_rlvr = tokenizer.decode(
                        gen_rlvr_base[j], skip_special_tokens=True
                    )
                    res["SFT_unpatched_baseline"][q_name].append(
                        int(is_correct(t_sft, golds[j]))
                    )
                    res["RLVR_unpatched_baseline"][q_name].append(
                        int(is_correct(t_rlvr, golds[j]))
                    )

                # ── Individual layers ────────────────────────────────────────
                for l in INDIVIDUAL_LAYERS:
                    lkey = str(l)
                    gen = patched_generate_batch(
                        sft_model,
                        input_ids,
                        attention_mask,
                        rlvr_cache,
                        [l],
                        MAX_NEW_TOKENS,
                        eos_id,
                        pad_id,
                    )
                    for j in range(B):
                        text = tokenizer.decode(gen[j], skip_special_tokens=True)
                        res["RLVR_to_SFT"][q_name]["individual"][lkey].append(
                            int(is_correct(text, golds[j]))
                        )
                    gen = patched_generate_batch(
                        rlvr_model,
                        input_ids,
                        attention_mask,
                        sft_cache,
                        [l],
                        MAX_NEW_TOKENS,
                        eos_id,
                        pad_id,
                    )
                    for j in range(B):
                        text = tokenizer.decode(gen[j], skip_special_tokens=True)
                        res["SFT_to_RLVR"][q_name]["individual"][lkey].append(
                            int(is_correct(text, golds[j]))
                        )

                # ── Cumulative segments ──────────────────────────────────────
                for seg_start, seg_end in CUMULATIVE_SEGMENTS:
                    seg_layers = list(range(seg_start, seg_end + 1))
                    skey = f"L{seg_start}-{seg_end}"
                    gen = patched_generate_batch(
                        sft_model,
                        input_ids,
                        attention_mask,
                        rlvr_cache,
                        seg_layers,
                        MAX_NEW_TOKENS,
                        eos_id,
                        pad_id,
                    )
                    for j in range(B):
                        text = tokenizer.decode(gen[j], skip_special_tokens=True)
                        res["RLVR_to_SFT"][q_name]["cumulative"][skey].append(
                            int(is_correct(text, golds[j]))
                        )
                    gen = patched_generate_batch(
                        rlvr_model,
                        input_ids,
                        attention_mask,
                        sft_cache,
                        seg_layers,
                        MAX_NEW_TOKENS,
                        eos_id,
                        pad_id,
                    )
                    for j in range(B):
                        text = tokenizer.decode(gen[j], skip_special_tokens=True)
                        res["SFT_to_RLVR"][q_name]["cumulative"][skey].append(
                            int(is_correct(text, golds[j]))
                        )

                base_sft = res["SFT_unpatched_baseline"][q_name][-B:]
                base_rlvr = res["RLVR_unpatched_baseline"][q_name][-B:]
                r12 = res["RLVR_to_SFT"][q_name]["cumulative"].get("L0-12", [])[-B:]
                s12 = res["SFT_to_RLVR"][q_name]["cumulative"].get("L0-12", [])[-B:]
                log(
                    f"  [{q_name}] ex {batch_start + 1}-{batch_start + B}/{len(subset)}  "
                    f"golds={golds}  baseline_SFT={base_sft}  baseline_RLVR={base_rlvr}  "
                    f"R->S@L0-12={r12}  S->R@L0-12={s12}"
                )

                del rlvr_cache, sft_cache
                clear_gpu()
            except Exception as exc:
                log(f"  ERROR [{q_name}] batch {batch_start}: {exc}")
                import traceback

                traceback.print_exc()
                clear_gpu()
                continue

            # Save a crash-safe checkpoint after every batch so we never lose
            # more than one batch of work on sandbox resets.
            with open(OUT_PATH, "w") as f:
                json.dump(res, f, indent=2)
            log(f"Checkpoint saved -> {OUT_PATH} (batch {batch_start})")

        with open(OUT_PATH, "w") as f:
            json.dump(res, f, indent=2)
        log(f"Quadrant checkpoint saved -> {OUT_PATH}")

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

    log("\n" + "=" * 78)
    log("BASELINE-ADJUSTED RESULTS -- Dataset D (Frontier)")
    log("=" * 78)
    base_sft = res["SFT_unpatched_baseline"].get("D_Frontier", [])
    base_rlvr = res["RLVR_unpatched_baseline"].get("D_Frontier", [])
    bm, bl, bu = ci(base_sft)
    log(f"  UNPATCHED SFT baseline:  {bm:5.1f}% [{bl:.1f}-{bu:.1f}] n={len(base_sft)}")
    bm2, bl2, bu2 = ci(base_rlvr)
    log(
        f"  UNPATCHED RLVR baseline: {bm2:5.1f}% [{bl2:.1f}-{bu2:.1f}] n={len(base_rlvr)}"
    )
    log("")
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
            f"  {seg:8s}  RLVR->SFT: {m1:5.1f}% [{l1:.1f}-{u1:.1f}] (delta={m1 - bm:+.1f}) n={len(r2s)}"
            f"    SFT->RLVR: {m2:5.1f}% [{l2:.1f}-{u2:.1f}] (delta={m2 - bm2:+.1f}) n={len(s2r)}"
        )


if __name__ == "__main__":
    log("=== exp14 v5 -- Methodologically Corrected ===")
    log(f"Output  : {OUT_PATH}")
    log(f"Dataset : {DATASET_PATH}")

    with open(DATASET_PATH) as f:
        quadrants = json.load(f)
    for k, v in quadrants.items():
        log(f"  {k}: {len(v)} examples")

    existing = {}
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
        for q, d in existing.get("RLVR_to_SFT", {}).items():
            if len(d.get("cumulative", {}).get("L0-12", [])) >= TARGET_PATCH_EXAMPLES:
                done.add(q)
        if done:
            log(f"[RESUME] Skipping completed: {done}")
            quadrants = {k: v for k, v in quadrants.items() if k not in done}

    res = run_patching(quadrants, existing_results=existing)

    for direction in (
        "RLVR_to_SFT",
        "SFT_to_RLVR",
        "SFT_unpatched_baseline",
        "RLVR_unpatched_baseline",
    ):
        for q, d in existing.get(direction, {}).items():
            if q not in res.get(direction, {}):
                res.setdefault(direction, {})[q] = d

    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    log(f"\nFinal results -> {OUT_PATH}")
    print_summary(res)
    log("Done.")
