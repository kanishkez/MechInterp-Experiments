"""
Diagnose why the custom HF batched-decode pipeline disagrees with vLLM.

Hypothesis: left-padding + KV cache requires the attention_mask to be
EXTENDED at every decode step (concat 1s for new tokens). The batched
patching script passes attention_mask=None after prefill, which can
corrupt position_ids / causal masking for padded sequences.

Test plan:
 1. batch_size=1 (no padding needed) vs vLLM labels -> isolates padding bugs
 2. batch_size=8 with FIXED attention_mask extension vs vLLM labels
 3. batch_size=8 with the ORIGINAL (buggy) attention_mask=None vs vLLM labels
"""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B",
}
MAX_NEW_TOKENS = 256


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


def gen_batch1_no_padding(model, tokenizer, prompt, max_new_tokens, eos_id):
    """Single-example generation, zero padding, matches original exp14 approach."""
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


def gen_batch_fixed_mask(
    model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id
):
    """Batched decode with PROPERLY extended attention_mask at each step."""
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
            # CRITICAL FIX: extend attention mask with 1s for the new token position
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


def gen_batch_broken_mask(
    model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id
):
    """Reproduces the ORIGINAL bug: attention_mask=None after prefill."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

    B = input_ids.shape[0]
    generated = [next_tok]
    done = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
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
    return torch.cat(generated, dim=1)


if __name__ == "__main__":
    with open("/marimo/exp14_gsm8k_quadrants.json") as f:
        quad = json.load(f)
    # Use first 16 examples of D_Frontier (known: SFT=0/60, RLVR=60/60 per vLLM)
    d_examples = quad["D_Frontier"][:16]

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    print("Loading SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    device = next(sft_model.parameters()).device

    prompts = [format_prompt(tokenizer, ex["prompt"]) for ex in d_examples]
    golds = [ex["gold"] for ex in d_examples]

    # ── Test 1: batch_size=1, no padding ──────────────────────────────────
    print("\n=== TEST 1: batch_size=1, no padding (matches original exp14 style) ===")
    hits1 = []
    for i, (p, g) in enumerate(zip(prompts, golds)):
        text = gen_batch1_no_padding(sft_model, tokenizer, p, MAX_NEW_TOKENS, eos_id)
        hit = is_correct(text, g)
        hits1.append(int(hit))
        print(f"  ex {i + 1}: gold={g}  hit={hit}")
    print(
        f"Test 1 SFT accuracy (batch=1, no pad): {sum(hits1)}/{len(hits1)} = {sum(hits1) / len(hits1) * 100:.1f}%"
    )

    # ── Test 2: batch_size=8, FIXED mask extension ────────────────────────
    print("\n=== TEST 2: batch_size=8, attention_mask PROPERLY extended ===")
    hits2 = []
    BATCH = 8
    for i in range(0, len(prompts), BATCH):
        bp = prompts[i : i + BATCH]
        bg = golds[i : i + BATCH]
        enc = tokenizer(
            bp, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        gen = gen_batch_fixed_mask(
            sft_model,
            enc["input_ids"],
            enc["attention_mask"],
            MAX_NEW_TOKENS,
            eos_id,
            pad_id,
        )
        for j in range(len(bp)):
            text = tokenizer.decode(gen[j], skip_special_tokens=True)
            hit = is_correct(text, bg[j])
            hits2.append(int(hit))
            print(f"  ex {i + j + 1}: gold={bg[j]}  hit={hit}")
    print(
        f"Test 2 SFT accuracy (batch=8, fixed mask): {sum(hits2)}/{len(hits2)} = {sum(hits2) / len(hits2) * 100:.1f}%"
    )

    # ── Test 3: batch_size=8, BROKEN mask (attention_mask=None) ───────────
    print("\n=== TEST 3: batch_size=8, attention_mask=None (reproduces the bug) ===")
    hits3 = []
    for i in range(0, len(prompts), BATCH):
        bp = prompts[i : i + BATCH]
        bg = golds[i : i + BATCH]
        enc = tokenizer(
            bp, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        gen = gen_batch_broken_mask(
            sft_model,
            enc["input_ids"],
            enc["attention_mask"],
            MAX_NEW_TOKENS,
            eos_id,
            pad_id,
        )
        for j in range(len(bp)):
            text = tokenizer.decode(gen[j], skip_special_tokens=True)
            hit = is_correct(text, bg[j])
            hits3.append(int(hit))
            print(f"  ex {i + j + 1}: gold={bg[j]}  hit={hit}")
    print(
        f"Test 3 SFT accuracy (batch=8, broken mask): {sum(hits3)}/{len(hits3)} = {sum(hits3) / len(hits3) * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    print("SUMMARY (all should be ~0% per vLLM ground truth):")
    print(f"  Test 1 (batch=1, no pad):      {sum(hits1) / len(hits1) * 100:.1f}%")
    print(f"  Test 2 (batch=8, fixed mask):   {sum(hits2) / len(hits2) * 100:.1f}%")
    print(f"  Test 3 (batch=8, broken mask):  {sum(hits3) / len(hits3) * 100:.1f}%")
    print("=" * 60)

    with open("/marimo/exp14_diagnosis_results.json", "w") as f:
        json.dump(
            {
                "batch1_nopad": hits1,
                "batch8_fixed_mask": hits2,
                "batch8_broken_mask": hits3,
            },
            f,
            indent=2,
        )
