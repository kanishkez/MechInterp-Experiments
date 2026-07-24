# Mechanistic Interpretability: From Base to SFT to RLVR

## 1. Abstract
This report documents a large-scale mechanistic interpretability investigation into how large language models change when undergoing Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR).

## 2. Introduction
Understanding the structural and computational changes induced by post-training is crucial for AI safety and alignment. We analyze representations, causal circuitry, and latent states across training stages.

## 3. Research Questions
- What mechanistically changes when a pretrained language model undergoes supervised fine-tuning?
- Does SFT install a sparse "assistant circuit" or change distributed routing?
- How does RLVR fundamentally alter the model's forward pass compared to SFT?

## 4. Models
- Qwen 2.5 7B Base & Instruct
- Meta Llama 3 8B Base & Instruct
- Llama-3.1-Tulu-3-8B (Base, SFT, DPO, RLVR)

## 5. Dataset Construction
Datasets were generated across Code, Creative Writing, QA, and Reasoning domains, carefully split into Discovery and Validation sets. Bias controls were implemented to cover all four SFT/RLVR success/failure quadrants.

## 6. Experimental Methodology
- Representational Similarity (CKA, Probing, JSD)
- Activation Patching (Causal interventions)
- Trajectory Clustering (Latent state occupancy)
- Activation Steering (Dose-response causal sufficiency)

## 7. Base vs Instruct Results
Initial experiments indicated significant late-layer divergence, but this requires robust validation across prompt templates.

## 8. SFT Routing Results
Activation steering and dimensionality experiments demonstrated that instruction-following behavior can be induced by low-rank interventions, suggesting distributed routing over a singular sparse circuit.

## 9. SFT Falsification Results
The hypothesis that a singular sparse assistant circuit governs all SFT behavior has been challenged by robust role-token interventions and cross-domain generalization audits.

## 10. SFT → DPO → RLVR Results
RLVR introduces significant shifts in activation profiles. Cross-model probing achieved ~0.625 cosine similarity between SFT and RLVR success directions, suggesting related but non-identical computational spaces.

## 11. RLVR Falsification Results
- The original hypothesis that L13 MLP acts as an RLVR bottleneck was **falsified**.
- The hypothesis that RLVR simply increases gain on SFT neurons was **falsified**.

## 12. Unified Mechanistic Hypothesis
*(To be updated as validation experiments conclude)*

## 13. Competing Hypotheses
| Prediction | H1 (New Circuit) | H2 (Gain) | H3 (Routing) | H4 (State Occ) | H5 (Redundancy) | Result |
|---|---|---|---|---|---|---|
| L13 bottleneck falsified | No | Yes | Yes | Yes | Yes | Supported |
| Gain only | No | Yes | No | No | No | Falsified |
| New weights dominate | Yes | No | No | No | No | Weak evidence |
| Divergent late states | No | Yes | Yes | Yes | No | Pending |

## 14. Causal Evidence
Activation steering (Phase E) and path patching demonstrate causal links, though necessity/sufficiency bounds are still being formally established.

## 15. Statistical Validation
All core claims will be supported by bootstrap confidence intervals and negative controls.

## 16. Limitations
- Highly dependent on the geometric assumptions of CKA/Linear Probing.
- SAE features are sensitive to dictionary size and training hyperparameters.

## 17. Open Questions
- To what extent are the identified latent states merely representations of the input class priors?

## 18. Final Conclusions
*(To be written)*
