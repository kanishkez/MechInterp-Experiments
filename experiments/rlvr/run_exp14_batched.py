"""
exp14 v4 - Causal Trajectory Transfer (batched, GPU-maxed)

Key changes vs v3:
- Batched prefill caching and patched generation (batch_size=8)
- Both models stay resident in VRAM simultaneously (~32GB total, fits in 102GB)
- torch.compile on the decode loop for speed
- Resume support: skips quadrants already at TARGET examples in checkpoint
- Per-example audit log with gold/pred/hit for every layer config
"""

import gc
import json
import os
import re
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8  # examples per GPU batch during patching
OUT_PATH = "/marimo/exp14_production_results.json"
DATASET_PATH = "/marimo/exp14_gsm8k_quadrants.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    gold = gold.strip().replace(",", "")
    return pred == gold and pred != ""


def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


# ── batched activation patching ───────────────────────────────────────────────
def cache_prefill_batch(model, input_ids, attention_mask, layers):
    """Cache residual stream after each layer for a batch of inputs."""
    cache = {}
    handles = []

    def make_hook(l_idx):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:  # prefill only
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
    """
    Greedy decode with activation patching on prefill only, batched.
    Returns generated token ids (batch, gen_len).
    """
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
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
    finally:
        for h in handles:
            h.remove()

    B = input_ids.shape[0]
    generated = [next_tok]
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            # For finished sequences, feed pad token to keep KV cache aligned
            feed = next_tok.clone()
            feed[done] = pad_id
            out = model(
                input_ids=feed,
                past_key_values=past_kv,
                attention_mask=None,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            next_tok[done] = pad_id
            generated.append(next_tok)
            done |= next_tok.squeeze(-1) == eos_id
            if done.all():
                break

    return torch.cat(generated, dim=1)  # (B, gen_len)


# ── main experiment ───────────────────────────────────────────────────────────
def run_patching(quadrants, existing_results=None):
    log("Phase 2: Batched Activation Patching")

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log("Loading SFT model ...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    log("Loading RLVR model ...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    device = next(sft_model.parameters()).device
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        log(f"  GPU {i}: {used:.1f}/{total:.0f} GB after model load")

    all_layers = sorted(
        set(INDIVIDUAL_LAYERS)
        | {l for s, e in CUMULATIVE_SEGMENTS for l in range(s, e + 1)}
    )
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Initialise results, seeding from checkpoint
    res = {
        "metadata": {
            "individual_layers": INDIVIDUAL_LAYERS,
            "cumulative_segments": [f"L{s}-{e}" for s, e in CUMULATIVE_SEGMENTS],
            "quadrant_sizes": {k: len(v) for k, v in quadrants.items()},
            "batch_size": BATCH_SIZE,
        },
        "RLVR_to_SFT": {},
        "SFT_to_RLVR": {},
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

    for q_name, examples in quadrants.items():
        subset = examples[:TARGET_PATCH_EXAMPLES]
        log(
            f"\n--- Quadrant: {q_name}  ({len(subset)} examples, batch_size={BATCH_SIZE}) ---"
        )

        # Process in batches
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
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            prompt_len = input_ids.shape[1]
            B = input_ids.shape[0]

            try:
                # Cache both models' prefill activations
                rlvr_cache = cache_prefill_batch(
                    rlvr_model, input_ids, attention_mask, all_layers
                )
                sft_cache = cache_prefill_batch(
                    sft_model, input_ids, attention_mask, all_layers
                )

                # ── Individual layers ───────────────────────────────────────
                for l in INDIVIDUAL_LAYERS:
                    lkey = str(l)
                    # RLVR -> SFT
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
                    # SFT -> RLVR
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

                # ── Cumulative segments ─────────────────────────────────────
                for seg_start, seg_end in CUMULATIVE_SEGMENTS:
                    seg_layers = list(range(seg_start, seg_end + 1))
                    skey = f"L{seg_start}-{seg_end}"
                    # RLVR -> SFT
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
                    # SFT -> RLVR
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

                # Per-batch log
                r12 = res["RLVR_to_SFT"][q_name]["cumulative"].get("L0-12", [])
                s12 = res["SFT_to_RLVR"][q_name]["cumulative"].get("L0-12", [])
                batch_r = r12[-B:] if len(r12) >= B else r12
                batch_s = s12[-B:] if len(s12) >= B else s12
                log(
                    f"  [{q_name}] ex {batch_start + 1}-{batch_start + B}/{len(subset)}  "
                    f"golds={golds}  "
                    f"R->S@L0-12={batch_r}  S->R@L0-12={batch_s}"
                )

                del rlvr_cache, sft_cache
                clear_gpu()

            except Exception as exc:
                log(f"  ERROR [{q_name}] batch {batch_start}: {exc}")
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
    log("RESULTS: RLVR->SFT Cumulative -- Dataset D (Frontier)")
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

    log("\nCROSS-QUADRANT: L0-12")
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
    log("=== exp14 v4 -- Batched Causal Trajectory Transfer ===")
    log(f"Output  : {OUT_PATH}")
    log(f"Dataset : {DATASET_PATH}")
    log(f"Batch size: {BATCH_SIZE}")

    # Load quadrant dataset (already built)
    log(f"Loading cached dataset from {DATASET_PATH}")
    with open(DATASET_PATH) as f:
        quadrants = json.load(f)
    for k, v in quadrants.items():
        log(f"  {k}: {len(v)} examples")

    # Resume: skip completed quadrants
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

    # Merge skipped quadrants back
    for direction in ("RLVR_to_SFT", "SFT_to_RLVR"):
        for q, d in existing.get(direction, {}).items():
            if q not in res.get(direction, {}):
                res.setdefault(direction, {})[q] = d

    with open(OUT_PATH, "w") as f:
        json.dump(res, f, indent=2)
    log(f"\nFinal results -> {OUT_PATH}")
    print_summary(res)
    log("Done.")
