import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

SFT_MODEL = "allenai/Llama-3.1-Tulu-3-8B-SFT"
RLVR_MODEL = "allenai/Llama-3.1-Tulu-3-8B" # RLVR is typically the final Tulu 3 release

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(RLVR_MODEL)
    
    print("Loading models...")
    # Load models in bf16
    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL, device_map="auto", torch_dtype=torch.bfloat16
    )
    rlvr_model = AutoModelForCausalLM.from_pretrained(
        RLVR_MODEL, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    print("Loading dataset...")
    # Using a small reasoning dataset or hardcoding a few prompts for speed
    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"
    ]
    
    chat_prompts = []
    for p in prompts:
        chat = [
            {"role": "user", "content": p}
        ]
        chat_prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    
    results = {}
    
    # We will sweep across all 32 layers
    num_layers = rlvr_model.config.num_hidden_layers
    
    for idx, prompt in enumerate(tqdm(chat_prompts)):
        inputs = tokenizer(prompt, return_tensors="pt").to(rlvr_model.device)
        
        # 1. Clean RLVR run (baseline)
        with torch.no_grad():
            rlvr_out = rlvr_model(**inputs, output_hidden_states=True)
            baseline_logits = rlvr_out.logits[0, -1, :]
            
        target_token_id = baseline_logits.argmax().item()
        baseline_prob = torch.nn.functional.softmax(baseline_logits, dim=-1)[target_token_id].item()
        
        # 2. Corrupt SFT run (source)
        with torch.no_grad():
            sft_out = sft_model(**inputs, output_hidden_states=True)
        sft_hiddens = sft_out.hidden_states # list of 33 tensors
        
        layer_results = []
        # 3. Patching
        for layer_idx in range(num_layers):
            def patch_hook(module, input, output):
                # Inject SFT's residual stream at the final token
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                
                # Replace the entire sequence or just the last token?
                # The question asks about patching individual activations (like at the last token). Let's patch the entire sequence for this layer to see if that layer's computation is indispensable.
                # Actually, let's patch just the last token's residual stream as it contains the accumulated context.
                if len(h.shape) == 3:
                    h[:, -1, :] = sft_hiddens[layer_idx][:, -1, :]
                elif len(h.shape) == 2:
                    h[-1, :] = sft_hiddens[layer_idx][0, -1, :]
                    
                return (h,) + output[1:] if is_tuple else h
            
            hook_handle = rlvr_model.model.layers[layer_idx].register_forward_hook(patch_hook)
            
            with torch.no_grad():
                patched_out = rlvr_model(**inputs)
                
            hook_handle.remove()
            
            patched_logits = patched_out.logits[0, -1, :]
            patched_prob = torch.nn.functional.softmax(patched_logits, dim=-1)[target_token_id].item()
            
            prob_drop = baseline_prob - patched_prob
            
            layer_results.append({
                "layer": layer_idx,
                "patched_prob": float(patched_prob),
                "prob_drop": float(prob_drop)
            })
            
        results[idx] = {
            "prompt": prompt,
            "baseline_prob": float(baseline_prob),
            "layer_results": layer_results
        }
        
    with open("exp2_activation_patching_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Finished Experiment 2!")

if __name__ == "__main__":
    main()
