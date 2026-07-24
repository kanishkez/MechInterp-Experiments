import torch
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F
from tqdm import tqdm

def compute_js_divergence(logits_p, logits_q):
    # Logits shape: [batch, pos, d_vocab]
    # Convert to probabilities
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    
    # Midpoint distribution
    m = 0.5 * (p + q)
    
    # KL(P || M) and KL(Q || M)
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean', log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean', log_target=False)
    
    return 0.5 * (kl_pm + kl_qm).item()

def run_exp1(models, prompts, max_tokens=10):
    # Extract model names in lineage order
    lineage = ["Base", "SFT", "DPO", "RLVR"]
    results = {f"{lineage[i]}_vs_{lineage[i+1]}": [] for i in range(len(lineage)-1)}
    
    # We will just take the first few prompts to keep time manageable in the demo
    sample_prompts = prompts[:10]
    
    with torch.no_grad():
        for i in range(len(lineage)-1):
            model_A_name = lineage[i]
            model_B_name = lineage[i+1]
            model_A = models[model_A_name]
            model_B = models[model_B_name]
            
            divergences = []
            
            for prompt in tqdm(sample_prompts, desc=f"{model_A_name} vs {model_B_name}"):
                tokens_A = model_A.to_tokens(prompt)
                tokens_B = model_B.to_tokens(prompt)
                
                # Check tokenization matches
                if not torch.equal(tokens_A, tokens_B):
                    continue
                    
                # Run through models and get accumulated residual streams
                _, cache_A = model_A.run_with_cache(tokens_A, names_filter=lambda x: x.endswith("resid_post"))
                _, cache_B = model_B.run_with_cache(tokens_B, names_filter=lambda x: x.endswith("resid_post"))
                
                num_layers = model_A.cfg.n_layers
                layer_js = []
                
                for layer in range(num_layers):
                    # Get resid_post at this layer: [1, pos, d_model]
                    resid_A = cache_A[f"blocks.{layer}.hook_resid_post"]
                    resid_B = cache_B[f"blocks.{layer}.hook_resid_post"]
                    
                    # Apply final layer norm and unembed
                    # Note: Llama-3 unembedding is tied or untied, transformer_lens handles this via model.unembed
                    # We need to apply final LN first.
                    ln_resid_A = model_A.ln_final(resid_A)
                    logits_A = model_A.unembed(ln_resid_A)
                    
                    ln_resid_B = model_B.ln_final(resid_B)
                    logits_B = model_B.unembed(ln_resid_B)
                    
                    # Compute JS divergence on the final prompt token (pos = -1)
                    js_div = compute_js_divergence(logits_A[:, -1, :], logits_B[:, -1, :])
                    layer_js.append(js_div)
                    
                divergences.append(layer_js)
                
            # Average across prompts
            avg_divergences = np.mean(divergences, axis=0)
            results[f"{model_A_name}_vs_{model_B_name}"] = avg_divergences

    # Plotting
    fig = go.Figure()
    for pair, divs in results.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(divs))),
            y=divs,
            mode='lines+markers',
            name=pair
        ))
    
    fig.update_layout(
        title="Layer-wise JS Divergence Across Post-Training Lineage",
        xaxis_title="Layer",
        yaxis_title="JS Divergence (Final Prompt Token)",
        template="plotly_white"
    )
    return fig

# Note: In marimo, we will call `fig = run_exp1(models, prompts)` and then output `fig`
