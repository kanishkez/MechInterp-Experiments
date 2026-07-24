import torch
import json
import numpy as np

def center(K):
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return H @ K @ H

def cka(X, Y):
    # X, Y: [n_samples, n_features]
    K = X @ X.T
    L = Y @ Y.T
    
    Kc = center(K)
    Lc = center(L)
    
    hsic = torch.sum(Kc * Lc)
    varK = torch.sqrt(torch.sum(Kc * Kc))
    varL = torch.sqrt(torch.sum(Lc * Lc))
    
    return (hsic / (varK * varL)).item()

def compute_evolution_cka():
    print("Loading extracted hiddens...")
    h_base = torch.load('/marimo/evolution_hiddens_Base.pt')
    h_sft = torch.load('/marimo/evolution_hiddens_SFT.pt')
    h_dpo = torch.load('/marimo/evolution_hiddens_DPO.pt')
    h_rlvr = torch.load('/marimo/evolution_hiddens_RLVR.pt')
    
    with open('/marimo/evolution_domains.json', 'r') as f:
        domains = json.load(f)
        
    unique_domains = list(set(domains))
    results = {
        "Base_vs_SFT": {dom: [] for dom in unique_domains},
        "SFT_vs_DPO": {dom: [] for dom in unique_domains},
        "DPO_vs_RLVR": {dom: [] for dom in unique_domains}
    }
    
    for layer in range(33):
        for dom in unique_domains:
            idx = [i for i, d in enumerate(domains) if d == dom]
            
            x_b = h_base[layer][idx]
            x_s = h_sft[layer][idx]
            x_d = h_dpo[layer][idx]
            x_r = h_rlvr[layer][idx]
            
            results["Base_vs_SFT"][dom].append(cka(x_b, x_s))
            results["SFT_vs_DPO"][dom].append(cka(x_s, x_d))
            results["DPO_vs_RLVR"][dom].append(cka(x_d, x_r))
            
    with open('/marimo/evolution_cka_results.json', 'w') as f:
        json.dump(results, f)
        
    print("\n=== EVOLUTION CKA PAYLOAD ===")
    print("Finished computing stratified CKA")
    print("=== END JSON PAYLOAD ===")

if __name__ == '__main__':
    compute_evolution_cka()
