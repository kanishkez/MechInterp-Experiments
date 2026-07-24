import torch, gc, numpy as np
gc.collect(); torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

print("=== EXP 6: ROLE-ROUTING vs SFT-ROUTING ===")
print("Q: Does SFT wire up the same edges that the model uses to distinguish roles?")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
m_inst = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16).cuda()
m_base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B",          dtype=torch.bfloat16).cuda()

# The two diffs we compare:
# Diff A (SFT-routing):  Base(ChatML/Assistant) vs Instruct(ChatML/Assistant)
# Diff B (Role-routing): Instruct(ChatML/Assistant) vs Instruct(ChatML/User)
PROMPT_ASST = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
PROMPT_USER = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>user\n"   # swap final role

MAX_L = 21  # edge patching up to layer 20

def get_mlp_cache(model, prompt):
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    cache = {}
    def hook(name):
        def fn(m, inp, out):
            cache[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return fn
    hooks = [model.model.layers[i].mlp.register_forward_hook(hook(f"mlp_{i}")) for i in range(MAX_L)]
    with torch.no_grad(): model(**ids)
    for h in hooks: h.remove()
    return cache, ids

def get_clean_probs(model, ids):
    with torch.no_grad():
        out = model(**ids)
    logits = out.logits[0,-1,:].float()
    return F.softmax(logits, dim=-1)

def edge_matrix(model, clean_prompt, corrupt_prompt):
    clean_cache, clean_ids = get_mlp_cache(model, clean_prompt)
    corrupt_cache, _       = get_mlp_cache(model, corrupt_prompt)
    clean_probs = get_clean_probs(model, clean_ids)
    mat = np.zeros((MAX_L, MAX_L))
    for src in range(MAX_L):
        for tgt in range(src+1, MAX_L):
            def pre_hook(mod, args, sn=f"mlp_{src}"):
                inp = args[0]
                return (inp - clean_cache[sn] + corrupt_cache[sn],) + args[1:]
            hk = model.model.layers[tgt].post_attention_layernorm.register_forward_pre_hook(pre_hook)
            with torch.no_grad():
                out = model(**clean_ids)
            hk.remove()
            pp = F.softmax(out.logits[0,-1,:].float(), dim=-1)
            kl = (clean_probs*(clean_probs.clamp(1e-10).log() - pp.clamp(1e-10).log())).sum().item()
            mat[src, tgt] = max(kl, 0)
    return mat

print("\nComputing Diff A: Base(Asst) vs Instruct(Asst) = SFT routing delta...")
mat_base_asst = edge_matrix(m_base, PROMPT_ASST, PROMPT_USER)
mat_inst_asst = edge_matrix(m_inst, PROMPT_ASST, PROMPT_USER)
diff_A = mat_inst_asst - mat_base_asst     # positive = edge gained by SFT

print("Computing Diff B: Instruct(Asst) vs Instruct(User) = Role routing delta...")
mat_inst_user = edge_matrix(m_inst, PROMPT_ASST, PROMPT_USER)   # asst→user within Instruct
# Diff B: how much does each edge matter MORE when role=assistant vs role=user?
mat_inst_asst2, _ = (mat_inst_asst, None)   # already have it
diff_B = mat_inst_asst - mat_inst_user       # positive = edge more important in asst mode

def top_k(mat, k=50):
    edges = [(i,j,mat[i,j]) for i in range(MAX_L) for j in range(i+1, MAX_L)]
    edges.sort(key=lambda x: x[2], reverse=True)
    return set((i,j) for i,j,_ in edges[:k])

topA = top_k(diff_A)
topB = top_k(diff_B)
overlap = topA & topB
jaccard = len(overlap) / len(topA | topB)

print(f"\n--- Overlap of SFT-routing and Role-routing (top 50 edges each) ---")
print(f"  Edges GAINED by SFT (Diff A): {len(topA)}")
print(f"  Edges MORE active in Asst role (Diff B): {len(topB)}")
print(f"  Shared: {len(overlap)}")
print(f"  Jaccard: {jaccard:.4f}")

# Flatten matrices and compute Pearson correlation
def flatten_upper(mat):
    vals = []
    for i in range(mat.shape[0]):
        for j in range(i+1, mat.shape[1]):
            vals.append(mat[i,j])
    return np.array(vals)

a = flatten_upper(diff_A)
b = flatten_upper(diff_B)
corr = np.corrcoef(a, b)[0,1]
print(f"\n  Pearson correlation (Diff A vs Diff B): {corr:.4f}")
print(f"  Interpretation: {'HIGH — SFT wires up role-routing policy' if corr > 0.5 else ('MODERATE' if corr > 0.2 else 'LOW — SFT is NOT simply role-routing')}")

print("\nDone!")
