"""
Rebuild GSM8K quadrants using a FIXED batched HF decode pipeline
(attention_mask properly extended each step). This will be the SAME
generation function used later for patching, eliminating the vLLM-vs-HF
inference-engine mismatch that confounded the first exp14 run.
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
SCAN_BUDGET = 2500
BATCH = 16
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


def gen_batch_fixed_mask(
    model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id
):
    """Batched decode with attention_mask PROPERLY extended at every step.
    This is the canonical decode function -- used for labeling AND patching."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

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


if __name__ == "__main__":
    log("=== Rebuilding quadrants: batched HF decode, fixed attention mask ===")
    gsm = load_dataset("openai/gsm8k", "main", split=f"train[:{SCAN_BUDGET}]")
    log(f"Loaded {len(gsm)} GSM8K examples, batch size {BATCH}")

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    log("Loading SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    log("Loading RLVR model...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    device = next(sft_model.parameters()).device

    quadrants = {"A_Core": [], "B_SFT_only": [], "C_Both_fail": [], "D_Frontier": []}

    all_examples = []
    for ex in gsm:
        gold = extract_answer(ex["answer"])
        if gold:
            all_examples.append({"prompt": ex["question"], "gold": gold})

    for batch_start in range(0, len(all_examples), BATCH):
        if all(len(v) >= TARGET_PER_QUADRANT for v in quadrants.values()):
            log(f"All quadrants full at batch_start={batch_start}, stopping")
            break

        batch = all_examples[batch_start : batch_start + BATCH]
        prompts = [format_prompt(tokenizer, ex["prompt"]) for ex in batch]
        golds = [ex["gold"] for ex in batch]

        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        try:
            gen_sft = gen_batch_fixed_mask(
                sft_model,
                enc["input_ids"],
                enc["attention_mask"],
                MAX_NEW_TOKENS,
                eos_id,
                pad_id,
            )
            gen_rlvr = gen_batch_fixed_mask(
                rlvr_model,
                enc["input_ids"],
                enc["attention_mask"],
                MAX_NEW_TOKENS,
                eos_id,
                pad_id,
            )

            for j in range(len(batch)):
                sft_text = tokenizer.decode(gen_sft[j], skip_special_tokens=True)
                rlvr_text = tokenizer.decode(gen_rlvr[j], skip_special_tokens=True)
                sft_ok = is_correct(sft_text, golds[j])
                rlvr_ok = is_correct(rlvr_text, golds[j])

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
                            "prompt": batch[j]["prompt"],
                            "gold": golds[j],
                            "SFT_text": sft_text,
                            "RLVR_text": rlvr_text,
                            "SFT_correct": sft_ok,
                            "RLVR_correct": rlvr_ok,
                        }
                    )

            log(
                f"  batch {batch_start}-{batch_start + len(batch)}: "
                f"A={len(quadrants['A_Core'])} B={len(quadrants['B_SFT_only'])} "
                f"C={len(quadrants['C_Both_fail'])} D={len(quadrants['D_Frontier'])}"
            )

            clear_gpu()
        except Exception as exc:
            log(f"  Error at batch {batch_start}: {exc}")
            import traceback

            traceback.print_exc()
            clear_gpu()
            continue

        # Periodic checkpoint
        if batch_start % (BATCH * 10) == 0:
            with open(OUT_PATH, "w") as f:
                json.dump(quadrants, f, indent=2)

    for k, v in quadrants.items():
        log(f"Final {k}: {len(v)}")

    with open(OUT_PATH, "w") as f:
        json.dump(quadrants, f, indent=2)
    log(f"Saved to {OUT_PATH}")
