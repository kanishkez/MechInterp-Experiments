import gc
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "SFT": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "RLVR": "allenai/Llama-3.1-Tulu-3-8B"
}

DEFAULT_COUNTS = {
    # Keep this large enough to support the downstream uniform sample of 500
    # examples in exp18, but small enough to run without vLLM.
    "gsm8k": 100,
    "math": 50,
    "arc": 25,
}
MAX_NEW_TOKENS = 300
BATCH_SIZE = 4

def extract_final_answer(text):
    if "#### " in text:
        return text.split("#### ")[-1].strip()
    return text.strip()


def format_prompt(tokenizer, prompt):
    chat = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def load_counts():
    return {
        "gsm8k": int(os.getenv("EXP16_GSM8K_N", DEFAULT_COUNTS["gsm8k"])),
        "math": int(os.getenv("EXP16_MATH_N", DEFAULT_COUNTS["math"])),
        "arc": int(os.getenv("EXP16_ARC_N", DEFAULT_COUNTS["arc"])),
    }


def batched_generate(model, tokenizer, prompts):
    outputs = []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def run_generation_sweep():
    print("Loading Mixed Reasoning Datasets for Large Scale Eval...")

    counts = load_counts()
    print(f"Target counts: {counts}")

    # Load a moderately sized mixed reasoning set without requiring vLLM.
    gsm8k = load_dataset("openai/gsm8k", "main", split=f"train[:{counts['gsm8k']}]")
    # Load MATH
    try:
        math_ds = load_dataset(
            "hendrycks/competition_math", split=f"train[:{counts['math']}]"
        )
    except:
        math_ds = []
    # Load ARC
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"train[:{counts['arc']}]")
    except:
        arc = []

    aggregated_prompts = []

    for ex in gsm8k:
        aggregated_prompts.append({
            "prompt": ex["question"],
            "true_answer": extract_final_answer(ex["answer"]),
            "full_truth": ex["answer"],
            "source": "gsm8k",
            "difficulty": 1000 + len(ex["answer"])  # Medium base + length
        })

    for ex in math_ds:
        aggregated_prompts.append({
            "prompt": ex["problem"],
            "true_answer": ex["solution"],
            "full_truth": ex["solution"],
            "source": "math",
            "difficulty": 3000 + len(ex["solution"])  # Hard base + length
        })

    for ex in arc:
        choices_str = "\n".join([f"{l}: {t}" for l, t in zip(ex['choices']['label'], ex['choices']['text'])])
        prompt = f"{ex['question']}\n\nChoices:\n{choices_str}\n\nPlease output your final answer (just the correct label) formatted exactly as: #### <label>"
        aggregated_prompts.append({
            "prompt": prompt,
            "true_answer": str(ex["answerKey"]),
            "full_truth": prompt,  # proxy
            "source": "arc",
            "difficulty": len(prompt) # Easy base + length
        })

    print(f"Loaded {len(aggregated_prompts)} total diverse reasoning prompts.")

    tokenizer = AutoTokenizer.from_pretrained(MODELS["SFT"], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    for ex in aggregated_prompts:
        ex["formatted_prompt"] = format_prompt(tokenizer, ex["prompt"])

    results = {
        i: {
            "prompt": ex["prompt"],
            "formatted_prompt": ex["formatted_prompt"],
            "true_answer": ex["true_answer"],
            "source": ex["source"],
            "difficulty": ex["difficulty"],
            "predictions": {},
        }
        for i, ex in enumerate(aggregated_prompts)
    }

    prompts = [ex["formatted_prompt"] for ex in aggregated_prompts]

    for model_name, model_id in MODELS.items():
        print(f"\nEvaluating {model_name} with HF greedy decode...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

        outputs = batched_generate(model, tokenizer, prompts)

        for i, gen_text in enumerate(outputs):
            pred_answer = extract_final_answer(gen_text)
            true_ans = results[i]["true_answer"].strip().lower()
            pred_ans = pred_answer.strip().lower()

            if results[i]["source"] == "gsm8k":
                is_correct = (true_ans == pred_ans)
            else:
                is_correct = (true_ans in pred_ans)

            results[i]["predictions"][model_name] = {
                "text": gen_text,
                "correct": is_correct,
            }

        del model
        gc.collect()
        torch.cuda.empty_cache()

    final_dataset = []
    for idx, data in results.items():
        sft_c = data["predictions"]["SFT"]["correct"]
        rlvr_c = data["predictions"]["RLVR"]["correct"]
        if sft_c and rlvr_c:
            data["quadrant"] = "A_Core"
        elif not sft_c and rlvr_c:
            data["quadrant"] = "D_Frontier"
        elif sft_c and not rlvr_c:
            data["quadrant"] = "B_SFT_Pass"
        else:
            data["quadrant"] = "C_Fail"
        final_dataset.append(data)
        
    print(f"\nSweep Complete.")
    print("A_Core:", len([x for x in final_dataset if x['quadrant'] == 'A_Core']))
    print("D_Frontier:", len([x for x in final_dataset if x['quadrant'] == 'D_Frontier']))

    with open('/marimo/large_scale_difficulty_dataset.json', 'w') as f:
        json.dump(final_dataset, f, indent=2)
    print("Saved to /marimo/large_scale_difficulty_dataset.json")

if __name__ == "__main__":
    run_generation_sweep()
