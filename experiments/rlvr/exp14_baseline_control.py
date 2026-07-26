"""
Critical control: what is SFT's accuracy on Dataset D with ZERO patched layers,
using the EXACT same custom HF decode pipeline as the patching experiment?

If this baseline is already ~20% (matching the "recovery" numbers), the entire
exp14 causal-transfer signal is an artifact of pipeline differences between
vLLM (used to build quadrants) and raw HF forward-pass decoding (used to patch),
not a real activation-patching effect.
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
    gold = gold.strip().replace(",", "")
    return pred == gold and pred != ""


def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def unpatched_generate_batch(
    model, input_ids, attention_mask, max_new_tokens, eos_id, pad_id
):
    """Identical decode loop to patched_generate_batch, but with ZERO hooks."""
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
    print("Loading dataset...")
    with open("/marimo/exp14_gsm8k_quadrants.json") as f:
        quad = json.load(f)
    d_examples = quad["D_Frontier"][:60]
    print(
        f"Dataset D: {len(d_examples)} examples (confirmed SFT=0/60, RLVR=60/60 in vLLM build)"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SFT model...")
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODELS["SFT"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    print("Loading RLVR model...")
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        MODELS["RLVR"], torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    device = next(sft_model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    sft_hits = []
    rlvr_hits = []
    sft_texts = []
    rlvr_texts = []

    BATCH = 8
    for i in range(0, len(d_examples), BATCH):
        batch = d_examples[i : i + BATCH]
        prompts = [format_prompt(tokenizer, ex["prompt"]) for ex in batch]
        golds = [ex["gold"] for ex in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        gen_sft = unpatched_generate_batch(
            sft_model,
            enc["input_ids"],
            enc["attention_mask"],
            MAX_NEW_TOKENS,
            eos_id,
            pad_id,
        )
        gen_rlvr = unpatched_generate_batch(
            rlvr_model,
            enc["input_ids"],
            enc["attention_mask"],
            MAX_NEW_TOKENS,
            eos_id,
            pad_id,
        )

        for j in range(len(batch)):
            t_sft = tokenizer.decode(gen_sft[j], skip_special_tokens=True)
            t_rlvr = tokenizer.decode(gen_rlvr[j], skip_special_tokens=True)
            hit_sft = is_correct(t_sft, golds[j])
            hit_rlvr = is_correct(t_rlvr, golds[j])
            sft_hits.append(int(hit_sft))
            rlvr_hits.append(int(hit_rlvr))
            sft_texts.append(t_sft)
            rlvr_texts.append(t_rlvr)
            print(
                f"ex {i + j + 1}/60  gold={golds[j]}  SFT_hit={hit_sft}  RLVR_hit={hit_rlvr}"
            )

    sft_acc = sum(sft_hits) / len(sft_hits) * 100
    rlvr_acc = sum(rlvr_hits) / len(rlvr_hits) * 100
    print()
    print("=" * 60)
    print(f"UNPATCHED BASELINE on Dataset D (this pipeline, HF greedy decode):")
    print(f"  SFT:  {sft_acc:.1f}%  ({sum(sft_hits)}/{len(sft_hits)})")
    print(f"  RLVR: {rlvr_acc:.1f}%  ({sum(rlvr_hits)}/{len(rlvr_hits)})")
    print("=" * 60)

    with open("/marimo/exp14_baseline_control_results.json", "w") as f:
        json.dump(
            {
                "sft_accuracy": sft_acc,
                "rlvr_accuracy": rlvr_acc,
                "sft_hits": sft_hits,
                "rlvr_hits": rlvr_hits,
                "sft_texts": sft_texts,
                "rlvr_texts": rlvr_texts,
            },
            f,
            indent=2,
        )
    print("Saved to /marimo/exp14_baseline_control_results.json")
