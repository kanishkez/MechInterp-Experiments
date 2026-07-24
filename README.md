# Mechanistic Interpretability: From Base to SFT to RLVR

## 1. Project Title
Mechanistic Interpretability of Post-Training: Tracing Computational Shifts from Base to SFT to RLVR

## 2. Research Question
What mechanistically changes when a pretrained language model undergoes supervised fine-tuning (SFT), and subsequently preference optimization / reinforcement learning with verifiable rewards (RLVR)?

## 3. Abstract
This repository contains a reproducible, scientifically rigorous set of mechanistic interpretability experiments investigating how language models change through the post-training pipeline. We audit previous claims regarding sparse circuitry and computational bottlenecks, and present a systematic investigation of latent state trajectories across SFT, DPO, and RLVR.

## 4. Key Findings
- SFT induces macroscopic divergence primarily in later layers, though this is sensitive to prompt formatting.
- Causal interventions demonstrate that instruction-following is likely governed by distributed routing policies rather than a single sparse "assistant circuit."
- RLVR and SFT share representational space (cosine sim ~0.625), but RLVR systematically shifts latent state occupancy and activation thresholds.

## 5. Current Hypotheses
The primary hypothesis under investigation is the **State Occupancy Hypothesis**: RLVR improves reasoning not by instantiating novel circuits, but by changing the input-dependent selection of existing latent computational states, increasing the probability that difficult inputs enter a successful state trajectory.

## 6. Experimental Roadmap
- [x] Phase 1: Base vs SFT Macroscopic Divergence
- [x] Phase 2: SFT Routing Falsification
- [x] Phase 3: SFT → DPO → RLVR Component Audits
- [ ] Phase 4: State Occupancy Verification
- [ ] Phase 5: Cross-Model Generalization

## 7. Models
- Qwen 2.5 7B (Base & Instruct)
- Meta Llama 3 8B (Base & Instruct)
- Llama-3.1-Tulu-3-8B (Base, SFT, DPO, RLVR)

## 8. Dataset Construction
See `data/README.md`. Datasets are stratified across reasoning, coding, QA, and creative writing. For RLVR analysis, datasets are split into four quadrants (SFT pass/fail × RLVR pass/fail).

## 9. Reproduction Instructions
```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests/
```
Run individual experiments via `python experiments/<track>/<script>.py`.

## 10. Hardware Requirements
Minimum: 1x 80GB A100 (or equivalent) for 7B/8B model inference with activation caching. 2x A100 recommended for SAE projection and trajectory clustering.

## 11. Experiment Table
| Track | Experiment | Script |
|---|---|---|
| SFT | JS Divergence | `experiments/sft/exp01_js_divergence.py` |
| SFT | CKA | `experiments/sft/exp02_cka.py` |
| SFT Routing | Steering Vectors | `experiments/sft_routing/steering_vectors.py` |
| RLVR | Geometry | `experiments/rlvr/exp01_geometry.py` |
| RLVR | Occupancy | `experiments/rlvr/exp10_layerwise_occupancy.py` |

## 12. Links to Figures
*(Figures will be populated in `figures/` as experiments run)*

## 13. Known Limitations
- CKA and linear probing carry inherent geometric assumptions that do not fully capture non-linear causal dynamics.
- Layerwise clustering is susceptible to collapsing into dataset class priors.

## 14. Falsified Hypotheses
- ❌ **L13 MLP Bottleneck**: Ablation studies show L13 is not a singular bottleneck for RLVR reasoning.
- ❌ **Gain-Only Hypothesis**: RLVR does not solely increase activation magnitude on identical SFT neurons.

## 15. Open Questions
- Is the success state fundamentally different for RLVR, or simply accessed more reliably?
- How much of the divergence is causal vs representational artifact?

## 16. Citation
*(Citation pending publication)*
