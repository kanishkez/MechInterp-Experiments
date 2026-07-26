# Experiment Log

This file is the canonical chronological log for the project. It records what each experiment tried to test, how it was run, what kind of graph or visualization it used, and the practical conclusion we drew from it.

The numbering below follows the project narrative in `reports/research_report.md`.

## Shared Setup

The early experiments used Qwen 2.5 Base vs Qwen 2.5 Instruct to understand how supervised fine-tuning changes a model.

The later RLVR experiments used the Llama-3.1-Tulu-3-8B family to study how reinforcement learning with verifiable rewards changes frontier reasoning.

Across the project, the recurring technical methods were:

| Method | What it measured |
|---|---|
| JS divergence | Layerwise distributional drift in output logits |
| CKA | Geometric similarity between residual-stream representations |
| Linear probes | Whether latent concepts were linearly decodable |
| Logit lens | What the model would predict if computation stopped at an intermediate layer |
| DLA | Which components directly wrote to the final logits |
| Delta DLA | Which components changed most between two models |
| Activation patching | Whether a component or trajectory was causally sufficient or necessary |
| SAE analysis | Feature-level structure inside the residual stream |
| Feature steering | Whether a learned feature could causally change behavior |
| Path patching | Whether a specific causal route through the network mattered |
| Bootstrap validation | Whether the result survived resampling and negative controls |

## Experiment Index

| Exp | Question | How it was done | Graphs / visuals used | Main result |
|---|---|---|---|---|
| 1 | Does SFT change output behavior everywhere or mainly near the end of the network? | Ran base and instruct models on 400 matched prompts across code, creative, reasoning, and QA, then measured layerwise JS divergence on the residual-to-logit projection. | Line plot of JS divergence over layers. | Most of the pass stays similar; the big split happens late, near the output layers. |
| 2 | Are the internal geometries still the same after SFT? | Extracted residual streams across all layers and computed CKA on matched prompts. | CKA heatmap and layerwise similarity curves. | Representations stay highly similar for most of the network, with a late divergence and a reasoning-domain mid-layer dip. |
| 3 | Are semantic concepts already present in the base model? | Trained cross-validated linear probes at every layer on balanced datasets of 500 instances per task. | Accuracy-by-layer probe curves and cross-model transfer matrix. | The same concepts are already linearly decodable in both models; SFT mainly changes usage, not basic presence. |
| 4 | When does the instruct model start looking different in token space? | Applied the logit lens to intermediate residual streams and tracked the correct-token probability. | Logit-lens line plots by layer. | The instruct model begins accumulating confidence earlier; the base model remains confused longer. |
| 5 | Which components directly write to the final token choice? | Computed DLA across all attention heads and MLPs, layer by layer. | Per-component bar charts and layerwise attribution plots. | Late MLPs dominate the direct logit-writing signal; attention contributes far less directly. |
| 6 | Which changes are truly caused by SFT rather than present in both models? | Subtracted base DLA from instruct DLA and compared the result across tasks. | Delta-DLA heatmap / task-by-layer comparison. | Creative prompts show little change; code prompts spike strongly in the late MLPs. |
| 7 | Can a specific head be causally responsible for the change? | Patched selected instruct attention heads with base activations and measured performance drop. | Activation-patching matrix and causal drop plots. | Late-layer heads, especially Layer 27 heads 9 and 13, caused a substantial performance shift when removed. |
| 8 | Do the base and instruct models share an SAE coordinate system? | Ran an instruction SAE on base-model activations and checked reconstruction quality on 200,000 tokens. | Reconstruction-quality plots and cosine-similarity summaries. | The same SAE decoded the base model well; the latent coordinate system is strongly shared. |
| 9 | What features does the routing head manipulate? | Compared SAE features before and after the routing head on matched prompts. | Feature delta bars and before/after feature maps. | The head suppresses one feature and amplifies another, consistent with a toggle-like role. |
| 10 | Can those features be used to steer behavior? | Clamped high-activation SAE features in the base model and inspected the generated text. | Steering dose-response curves and max-activation snippet grids. | Steering the target feature induced structured code and instruction-following behavior. |
| 11 | What triggers the template-specific feature change? | Fed the same content once as raw text and once with the chat template, then measured feature activation. | Paired feature-activation comparison plots. | The chat template is a strong trigger that flips the internal feature mix. |
| 12 | Is the circuit robust if upstream features are ablated? | Zeroed upstream features and measured downstream survival across 400 prompts. | Ablation bars and survival curves. | Early context features matter, and a Layer 23 hub-like feature was especially critical. |
| 13 | Did SFT actually change the routing weights? | Compared the Layer 27 head OV projection between base and instruct. | OV-weight comparison chart. | The OV weights were effectively unchanged, which suggests SFT mostly reuses pre-existing circuitry. |
| 14 | What is the minimal causal graph for instruction-following? | Ran exhaustive pairwise path patching across 42 component nodes and thresholded the median recovery. | Causal graph visualization of strong edges. | The instruct graph is much sparser than the base graph, and the topological pathway is more efficient. |
| 15 | Are the selected nodes stable under resampling? | Bootstrapped 50 random 80/20 splits and compared top nodes against a random null. | Bootstrap boxplots and null-comparison bands. | The candidate nodes were consistently better than the null baseline. |
| 16 | Can the feature analysis scale to a large atlas? | Automated the representational mapping pipeline across a much larger feature corpus. | Atlas-style lifecycle maps and large-scale feature summaries. | The automated pipeline catalogued a very large feature set and reproduced the bottleneck pattern at scale. |
| 17 | Does the pattern replicate on another model family? | Re-ran the JS-divergence and patching logic on Llama-3-8B. | Cross-model layer curves and patching summaries. | The late-layer divergence pattern generalized beyond Qwen. |
| 18 | Did RLVR rewrite the representation geometry? | Compared weight changes and CKA across SFT, DPO, and RLVR on identical token sequences. | Weight-delta plots and CKA heatmaps. | Weight changes were tiny and CKA stayed near-identical, so RLVR did not radically rewrite the geometry. |

## Base-to-SFT Interpretation

The base-to-instruct experiments showed a consistent pattern.

The model already contains most of the semantic content before SFT.

SFT changes how those latent variables are routed and when they become visible in the output distribution.

The strongest evidence for that claim came from the late-layer divergence, the late-layer DLA concentration, the feature-level routing head analysis, and the path-patching graph.

## Graph Map

| Graph or visualization | Used in | Why it mattered |
|---|---|---|
| Layerwise line plots | Exp 1, 4, 6, 17, 18 | Showed where in the forward pass the models separated |
| Heatmaps | Exp 2, 6, 18 | Made geometry and divergence structure visible across layers and tasks |
| Accuracy-by-layer curves | Exp 3, 15 | Showed when latent concepts were decodable and whether nodes were stable |
| Bar charts | Exp 5, 7, 9, 10, 12, 13 | Localized which components or features mattered most |
| Causal graphs | Exp 7, 14 | Turned component effects into a network-level mechanism |
| Dose-response plots | Exp 10 | Tested whether steering a feature could actually move behavior |
| Bootstrap intervals | Exp 15, 18 | Kept the claims honest under resampling and noise |
| Atlas maps | Exp 16 | Scaled the feature analysis beyond one-off examples |

## RLVR Interpretation

The RLVR experiments changed the question from "what is present in the model" to "what computation is selected when the model solves frontier problems."

The corrected exp14 result showed partial causal transfer between SFT and RLVR on frontier GSM8K cases, but not a full copy of the RLVR capability.

The later RLVR analyses supported a distributed, partially transferable trajectory change rather than a single bottleneck or a single reasoning vector.

## Live RLVR Notebook Appendix

The following runtime experiments were run in the live marimo session during this work. These are operational scripts and result files, separate from the report numbering above.

| Runtime script | What it did | What happened |
|---|---|---|
| `experiments/rlvr/exp14_baseline_control.py` | Checked the unpatched HF greedy baseline on `D_Frontier` with the same pipeline used for patching. | Confirmed the negative control and established the baseline needed for the exp14 story. |
| `experiments/rlvr/exp16_large_scale_eval.py` | Replaced the original `vllm` path with batched `transformers` greedy decoding and generated a large-scale difficulty dataset. | Completed successfully and wrote `large_scale_difficulty_dataset.json`. |
| `experiments/rlvr/exp17_robust_trajectory_patching.py` | Tested robust trajectory transfer across quadrants and cumulative patching. | Completed successfully and produced a partial, distributed transfer pattern. |
| `experiments/rlvr/exp18_difficulty_continuum.py` | Measured SFT/RLVR trajectory divergence across a sampled difficulty continuum. | Completed successfully and wrote the divergence results file. |

## Live RLVR Results

These are the most important live results from the notebook runs.

| Measurement | Value |
|---|---:|
| `exp14` unpatched SFT baseline on `D_Frontier` | `15.0%` |
| `exp14` unpatched RLVR baseline on `D_Frontier` | `76.7%` |
| `exp14` RLVR→SFT `L0-12` | `33.3%` |
| `exp14` RLVR→SFT `L0-31` | `43.3%` |
| `exp14` SFT→RLVR `L0-12` | `56.7%` |
| `exp14` SFT→RLVR `L0-31` | `46.7%` |
| `exp17` RLVR→SFT on `D_sft_fail_rlvr_pass` | `87.5%` |
| `exp17` SFT→RLVR on `D_sft_fail_rlvr_pass` | `62.5%` |
| `exp18` examples analyzed | `125` |
| `exp18` strongest divergence layer | `Layer 31` |

## Takeaway

The project moved from representational similarity to causal circuit localization and then to RLVR trajectory analysis.

The final story is not that one single layer or one single head contains "reasoning."

The stronger story is that SFT and RLVR both reconfigure how the model moves through internal states, with the RLVR change being causally relevant, partially transferable, and distributed across multiple layers.
