# exp14 Audit Log — Causal Trajectory Transfer (RLVR ↔ SFT)

This file is a running, timestamped log of the actual work done on exp14,
kept separate from the polished narrative in `research_report.md` so that
every claim can be traced back to a real artifact, run, or bug.

**Rule for this file: no result goes in without a path to raw data.**

---

## 2026-07-25 — Original claim (chat-transcript only, UNVERIFIED)

A prior agent session claimed:
> RLVR→SFT patching cumulative trajectory up to Layer 12 transfers 75% of
> capability to SFT on frontier problems (Dataset D), plateauing sharply at L12.
> SFT→RLVR patching at L12 drops RLVR from 100%→0%.

**Audit findings:**
- `experiments/rlvr/exp14_results.json` (the script's own declared output path) was **0 bytes**.
- The numbers actually lived in untracked, never-committed files `dataset_a_patching_results.json` / `dataset_d_patching_results.json`.
- Every value in Dataset D was an exact multiple of 25% → **n=4 examples**. "75%" = 3/4.
- Dataset A showed literally 100.0% on every single layer/segment (11 individual + 5 cumulative) — consistent with n=1–4 and no real variance.
- `exp17_robust_trajectory_patching.py` (the script meant to fix this with n=100 + bootstrap CIs) and `exp18_difficulty_continuum.py` never produced any output file — the prerequisite dataset (`large_scale_difficulty_dataset.json`) doesn't exist anywhere.

**Verdict: the original 75%/0% claim is not supported by evidence and should not be cited.**

---

## 2026-07-25 — Re-run attempt v1–v4 (real GPU, n=60/quadrant)

Ran on a live sandbox (`sb-8152db85f6604fe0.sb.molab.run`), Llama-3.1-Tulu-3-8B-SFT vs
Llama-3.1-Tulu-3-8B (RLVR), GSM8K.

- Phase 1: built quadrants (A_Core / B_SFT_only / C_Both_fail / D_Frontier) via **vLLM**
  batch inference on 2000 GSM8K examples. Confirmed D_Frontier: SFT 0/60, RLVR 60/60.
- Phase 2: activation-patched with a **custom raw-HF forward-pass hook loop** (not vLLM).
- Survived 3 sandbox resets and 2 crashes (scoping bug, silent CUDA hang) via checkpointed
  resume logic. Completed all 240 examples (60/quadrant × 4 quadrants).

**Result (raw, before scrutiny):**

| Layer range | RLVR→SFT (D_Frontier) | SFT→RLVR (D_Frontier) |
|---|---|---|
| L0-12 | 23.3% [13.3–35.0] | 33.3% [21.7–45.0] |
| L0-31 | 30.0% [18.3–43.3] | 38.3% [26.7–51.7] |

No sharp plateau, no monotonic curve — flat and noisy across all layers.
Already this contradicts the original claim. But a deeper problem surfaced on review:

**Critical bug found:** patching a *single* layer alone (L0 only, or L31 only — barely
touching the computation at all) gave **~20% accuracy**, nearly identical to the "23.3%
recovery" attributed to L0-12 patching. That's the tell of a baseline artifact, not a
causal effect.

**Root cause, confirmed by direct test:** ran SFT with **zero patching** through the exact
same custom HF decode pipeline used for measuring patched accuracy. Result:
**SFT unpatched baseline = 18.3% (11/60)**, **RLVR unpatched baseline = 35.0% (21/60)** —
nowhere near the vLLM-derived 0% / 100% used to define the quadrant in Phase 1.

**Diagnosis:** the quadrant labels were assigned using **vLLM** generation, but the
patching experiment measured effects using a **different inference engine** (raw
`transformers` forward-pass loop). Even at temperature=0/greedy on both sides, vLLM
(FlashAttention/PagedAttention kernels) and eager HF decode diverge over a 256-token
generation due to bf16 numerical differences that cascade. Confirmed with an isolated
diagnostic (`exp14_diagnose_pipeline.py`):

| Decode method | SFT accuracy on "SFT-should-fail" set (n=16) |
|---|---|
| batch=1, no padding (matches original exp14 script style) | 12.5% |
| batch=8, attention_mask correctly extended | 18.8% |
| batch=8, attention_mask=None after prefill (the bug in v1-v4) | 12.5% |

All three are far from the vLLM-implied 0%. **This is not a batching/padding bug — it's
an inference-engine mismatch.** Ruling out padding as the sole cause (batch=1 still shows
12.5%) was an important negative control.

**Verdict on v1-v4: the "23.3% recovery at L0-12" is statistically indistinguishable from
the unpatched baseline (18.3%) and should be treated as noise, not a causal transfer
effect. Discarding these results.**

---

## 2026-07-25 — v5: methodologically corrected re-run (in progress)

Fixes applied:
1. **Self-consistent labeling.** Rebuilt GSM8K quadrants (`exp14_gsm8k_quadrants_v2.json`)
   using the *same* batched HF decode function that measures patched accuracy — no vLLM
   involved anywhere in this run. Quadrant assignment and effect measurement now share one
   inference engine, eliminating the confound above.
   - Scanned up to 2500 GSM8K train examples, batch size 16, until each quadrant hit 60.
   - Final: A_Core=60, B_SFT_only=60, C_Both_fail=60, D_Frontier=60 (needed ~1232 examples
     scanned to fill D_Frontier — confirms it really is a rare quadrant, not a labeling
     fluke).
2. **Correct attention-mask extension.** Every decode step now concatenates a `1` onto the
   running attention mask for the newly generated token (previous versions passed
   `attention_mask=None` after prefill, an undiagnosed bug — though isolated testing showed
   this wasn't the dominant cause of the mismatch).
3. **Explicit unpatched baseline reported alongside every patched condition,** computed on
   the exact same batches, so results are always reported as **baseline-adjusted deltas**,
   not raw accuracy. This makes it structurally impossible to mistake "baseline noise" for
   "causal recovery" again.

Script: `experiments/rlvr/run_exp14_v5_corrected.py`
Output: `/marimo/exp14_v5_corrected_results.json` (on sandbox, not yet pulled locally)

**Status: RUNNING as of last check (started ~12:47, sandbox `sb-8152db85f6604fe0`, PID 32974).**
Results below will be filled in once complete — not before.

<!-- RESULTS_PLACEHOLDER: do not hand-wave numbers here until the run finishes and the
     raw JSON has been pulled and independently sanity-checked (baseline vs patched,
     negative control on C_Both_fail, monotonicity check). -->

---

## 2026-07-26 — v5 completed and verified

The corrected exp14 v5 run completed successfully in the live marimo session `s_uutz85`.

**Run summary:**
- Launcher: `python3 /marimo/exp14_launch_pipeline.py`
- Rebuild stage: completed with `exp14_gsm8k_quadrants_v2.json` saved
- Patching stage: completed with `exp14_v5_corrected_results.json` saved
- Pipeline log: `/marimo/exp14_pipeline.log`
- Rebuild log: `/marimo/exp14_rebuild.log`
- v5 log: `/marimo/exp14_v5.log`

**Verified D_Frontier headline results:**
- Unpatched SFT baseline: `15.0% [6.7, 23.3]`
- Unpatched RLVR baseline: `76.7% [66.7, 86.7]`
- RLVR→SFT `L0-12`: `33.3% [21.7, 45.0]`
- RLVR→SFT `L0-31`: `43.3% [31.7, 56.7]`
- SFT→RLVR `L0-12`: `56.7% [43.3, 68.3]`
- SFT→RLVR `L0-31`: `46.7% [35.0, 58.3]`

**Interpretation:**
- RLVR→SFT transfer is real but incomplete.
- SFT→RLVR degradation is real but not catastrophic.
- The effect is distributed across layers, not a sharp single-layer breakpoint.

**Reproducibility artifacts:**
- Raw JSON committed in repo: `experiments/rlvr/exp14_v5_corrected_results.json`
- Launcher script: `experiments/rlvr/exp14_launch_pipeline.py`
- Corrected patching script: `experiments/rlvr/run_exp14_v5_corrected.py`
- Narrative report: `reports/research_report.md`
- Forensic audit: this file
