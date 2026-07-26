"""
Rebuild GSM8K quadrants using the SAME single-example, no-padding HF decode
pipeline that the patching experiment uses. This removes the vLLM-vs-HF
inference mismatch that was confounding exp14's causal transfer results.

Scans GSM8K train examples until each quadrant has >= TARGET examples
(or the scan budget is exhausted), saves the labeled dataset.
"""

import gc
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B",
}
MAX_NEW_TOKENS = 256
TARGET_PER_QUADRANT = 60
SCAN_BUDGET = 1500  # max GSM8K examples to scan
OUT_PATH = "/marimo/exp14_gsm8k_quadrants_v2.json"


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
    return pred == gold.strip().replace(",", "") and pred != ""


def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def gen_single(model, tokenizer, prompt, max_new_tokens, eos_id):
    """Single-example, zero-padding greedy decode -- matches the patching pipeline."""
    inp = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(**inp, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    generated = [next_tok]
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_tok)
            if next_tok.item() == eos_id:
                break
    gen = torch.cat(generated, dim=1)
    return tokenizer.decode(gen[0], skip_special_tokens=True)


if __name__ == "__main__":
    log("=== Rebuilding quadrants with self-consistent HF pipeline ===")
    gsm = load_dataset("openai/gsm8k", "main", split=f"train[:{SCAN_BUDGET}]")
    log(f"Loaded {len(gsm)} GSM8K examples")

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"])
    eos_id = tokenizer.eos_token_id

    log("Loading SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    log("Loading RLVR model...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    quadrants = {"A_Core": [], "B_SFT_only": [], "C_Both_fail": [], "D_Frontier": []}

    for i, ex in enumerate(gsm):
        if all(len(v) >= TARGET_PER_QUADRANT for v in quadrants.values()):
            log(f"All quadrants full at i={i}, stopping early")
            break

        gold = extract_answer(ex["answer"])
        if not gold:
            continue
        prompt = ex["question"]
        fmt = format_prompt(tokenizer, prompt)

        try:
            sft_text = gen_single(sft_model, tokenizer, fmt, MAX_NEW_TOKENS, eos_id)
            rlvr_text = gen_single(rlvr_model, tokenizer, fmt, MAX_NEW_TOKENS, eos_id)
            sft_ok = is_correct(sft_text, gold)
            rlvr_ok = is_correct(rlvr_text, gold)

            if sft_ok and rlvr_ok:
                key = "A_Core"
            elif sft_ok and not rlvr_ok:
                key = "B_SFT_only"
            elif not sft_ok and not rlvr_ok:
                key = "C_Both_fail"
            else:
                key = "D_Frontier"

            if len(quadrants[key]) < TARGET_PER_QUADRANT:
                quadrants[key].append(
                    {
                        "prompt": prompt,
                        "gold": gold,
                        "SFT_text": sft_text,
                        "RLVR_text": rlvr_text,
                        "SFT_correct": sft_ok,
                        "RLVR_correct": rlvr_ok,
                    }
                )

            if i % 25 == 0:
                log(
                    f"  i={i}  A={len(quadrants['A_Core'])} B={len(quadrants['B_SFT_only'])} "
                    f"C={len(quadrants['C_Both_fail'])} D={len(quadrants['D_Frontier'])}"
                )

            clear_gpu()
        except Exception as exc:
            log(f"  Error at i={i}: {exc}")
            clear_gpu()
            continue

    for k, v in quadrants.items():
        log(f"Final {k}: {len(v)}")

    with open(OUT_PATH, "w") as f:
        json.dump(quadrants, f, indent=2)
    log(f"Saved to {OUT_PATH}")
