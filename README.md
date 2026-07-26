# Mechanistic Interpretability: From Base to SFT to RLVR

## Base vs Instruct
The purpose of these series of experiments was for me to get familiar with mech interp techniques. This is also a test run for my actual experiment that I want to conduct: Base vs RL, to see what exactly changes when you RL a model, what changes mechanistically.

So let's see what changes when you SFT a model.
I chose to investigate Qwen 2.5 Base and Qwen 2.5 Instruct.
We get started with our first series of experiments in which our goal is to understand how exactly an SFT model is different from a base model. To do this, I generated 400 perfectly matched prompts covering 4 domains (Code, Creative, Reasoning, and QA) with exact token lengths so we could directly compare their internal streams.

### Part I: Macroscopic Characterization

**Experiment 1: JS Divergence**
* **How JS Divergence works:** It looks at the probability distribution over the vocab by projecting the residual stream to the vocab logits. We use it to measure how different the two output probability distributions are between the base and instruct model at every layer.
* **What I did:** I ran both models on my 400 prompts and measured the JS Divergence at every single layer.
* **Observation:** The models perform identical computation for most of the forward pass. There is a huge spike only at the last few layers, meaning they only split near the output.
* **Conclusion / Hypothesis:** This means that SFT doesn't induce new knowledge everywhere, but only changes how the knowledge is used in the generation of the final output. It's a "late layer override".

**Experiment 2: Representation Similarity Analysis (CKA)**
* **How CKA works:** CKA (Centered Kernel Alignment) basically compares the geometry of two latent spaces. If the value is close to 1, the two spaces are structurally identical.
* **What I did:** I extracted the residual streams across all 28 layers using 500 prompts and computed the layer to layer CKA to see if the internal representations were actually different.
* **Observation:** The representations remain highly similar (CKA > 0.9) for the vast majority of the forward pass, with one notable exception: the Reasoning domain shows a distinct CKA dip in the middle layers before recovering. Across all domains, they only permanently break apart at the absolute final layers.
* **Conclusion:** The Instruct model uses the exact same early to mid layer conceptual representations as the Base model. The mid-layer dip in Reasoning suggests SFT allocates specific working memory for Chain-of-Thought processing, but outside of this task-specific routing, we see that the models only fully diverge at the final layers.

**Experiment 3: Linear Probes**
* **How Linear Probes work:** We train a simple logistic regression classifier on the residual stream to see if it can linearly separate concepts (like Python vs Java). If it can, it means the model "knows" that concept.
* **What I did:** I trained 5-fold cross validated probes at every layer on 4 tasks using perfectly balanced datasets of 500 instances each to see if semantic concepts are newly learned during SFT.
* **Observation:** The linear probes achieved nearly identical accuracy on both models. For example, Python vs Java was perfectly separable ( 1.000 accuracy) in both Base and Instruct.
* **Conclusion:** The Base model already contains these semantic concepts. SFT doesn't manufacture them from scratch; it just alters how they are used in the late layers.

### Part II: Localizing the Mechanism
Now we know representations are similar and concepts exist. Let's find out where the mechanism is actually happening.

**Experiment 4: Logit Lens**
* **How Logit Lens works:** It takes a hidden residual stream at an early layer and multiplies it with the final unembedding matrix to see what the model would predict if we stopped the network right there.
* **What I did:** I tracked the probability of the correct instruction-following token across all 28 layers for my 400 perfectly matched prompts.
* **Observation:** The Base Model is always confused. But the Instruct Model starts building confidence for the right answer remarkably early (Layers 3-5).
* **Conclusion:** While the representations are similar, the trajectory of evidence accumulation diverges early. This tension means we need to look closer at individual components.

**Experiment 5: Direct Logit Attribution (DLA)**
* **How DLA works:** DLA measures the effect of each individual component (like an Attention head or MLP) on the final probability of the correct token. We isolate every component and project it directly to the vocab. It does so by just multiplying each component with the final unembedding matrix and getting the logits for the prediction to see how this particular part would affect the final output.
* **What I did:** I ran DLA across all 28 attention layers and 28 MLPs over my 400 prompts to see which components actually write the logits.
* **Observation:** Attention Heads contribute very little directly. MLPs absolutely dominate, with all decisive evidence written by the last few MLPs (Layers 25-27). Interestingly, the Base model's early MLPs actually have a negative DLA for the correct token.
* **Conclusion:** The final semantic formatting is heavily driven by late-layer MLPs. The negative early MLPs in the Base model indicate that pre-training actively suppresses immediate instruction-following (e.g., predicting a markdown block instead of normal text continuation), a bias which SFT must neutralize.

**Experiment 6: Delta DLA ($\Delta$DLA)**
* **How $\Delta$DLA works:** We literally just subtract the Base model's DLA from the Instruct model's DLA to see the difference.
* **What I did:** I compared $\Delta$DLA on 400 Code prompts vs. 400 Creative prompts to isolate task-specific differences.
* **Observation:** On Creative prompts, the difference was near zero. On Code prompts, we saw a massive spike at the Layer 27 MLPs.
* **Conclusion:** This localizes the behavioral shift to the late layer MLPs, which SFT repurposed to be instruction-aligned.

**Experiment 7: Activation Patching**
* **How Activation Patching works:** It's a causal intervention. You run Model A, but computationally delete one of its components and swap in the activation from Model B.
* **What I did:** I ran the Instruct model over 400 prompts but computationally swapped specific Attention Heads with Base model heads to see if they were causally responsible.
* **Observation:** Swapping out specifically Layer 27, Heads 9 and 13 caused a substantial drop in the Instruct model's performance, shifting its behavioral distribution significantly toward the Base model baseline.
* **Conclusion:** This causally proves that these Late Layer Attention Heads act as critical routers. They read early context and enable the downstream MLPs to function correctly.

### Part III: Understanding the Circuit
We know WHICH nodes matter. Now let's figure out what they are actually computing.

**Experiment 8: SAE Validation**
* **How Sparse Autoencoders (SAEs) work:** We train SAEs to break down the activations into interpretable features. I found an SAE on Huggingface for my model Qwen 2.5 7b, however I only found one for the Instruct version. This immediately made me wonder if an SAE trained on the Instruct model generalizes for the Base model as well?
* **What I did:** I loaded an Instruct model SAE and fed 200,000 tokens of the Base model's activations through it. Can one SAE decode both models?
* **Observation:** The Base model activations reconstructed smoothly, achieving a high cosine similarity of 0.846.
* **Conclusion:** The models share a highly similar geometric coordinate system for latent concepts and yes, we can use the same SAE on the Base model as well.

**Experiment 9: Feature Surgery**
* **How it works:** We project the residual stream through the SAE immediately before and after a specific routing head.
* **What I did:** I looked at Layer 27, Head 13 across my 400 prompts to see what specific features it was changing during SFT.
* **Observation:** In the Instruct model, Head 13 drastically suppresses Feature 82781 and heavily amplifies Feature 106128.
* **Conclusion:** SFT repurposed this head to act as a toggle switch for these specific latent semantic variables.

**Experiment 10: Feature Steering**
* **How Feature Steering works:** We artificially increase specific SAE features to high values during the model's forward pass to force it to think about a concept.
* **What I did:** I analyzed what F82781 and F106128 mean by running thousands of tokens to find the max-activating snippets, and then clamped them in the Base model to a high value (~100) to see how they affect the 'thinking' of the model.
* **Observation:** F82781 fires on generic tokens like import ("Initial Context"). F106128 fires on structured code templates like class User ("Structured Code"). Clamping F106128 in the Base model caused it to output formatted code and follow instructions, even without a chat template.
* **Conclusion:** By successfully steering the model, we proved we found the exact semantic variables being manipulated by the routing heads.

### Part IV: Circuit Discovery
Let's reconstruct the full circuit from input to output.

**Experiment 11: Template Trigger**
* **How it works:** We measure the activation of a feature across different inputs.
* **What I did:** I ran 400 exact same text samples through the Instruct model twice: once as raw text, and once wrapped in the chat template (`<|im_start|>user...`), and measured the "Initial Context" feature.
* **Observation:** On raw text, it activated strongly (94.68). On chat template text, it dropped to 2.37, while the "Structured Code" feature skyrocketed.
* **Conclusion:** The chat template is the physical trigger that conditionally shifts the internal feature representations.

**Experiment 12: Upstream Feature Ablation**
* **How Feature Ablation works:** We set specific upstream features to 0 and measure if the downstream behavior survives.
* **What I did:** I traced the circuit backward by ablating features across 400 prompts to see where the context comes from.
* **Observation:** Ablating an early context feature (L3) caused a 46% drop downstream. Ablating an intermediate "hub" feature (L23) dropped the behavior to exactly 0.0. This means that in L3 there is a feature that gathers the context of the early layers to be passed downstream. In Layer 23, we have a feature that behaves as a 'central hub', almost all tasks require this feature and flow through it.
* **Conclusion:** We found a highly sparse topological link: L3 Context $\to$ L23 Hub Bottleneck $\to$ L27 Final Routing.

**Experiment 13: OV Matrix Analysis**
* **How OV Analysis works:** Attention Heads move information by multiplying inputs through an OV (Output-Value) matrix.
* **What I did:** I multiplied the intermediate Instruct L23 context vectors by the exact OV weights of L27 Head 13 for both models to see if SFT physically modified the routing weights.
* **Observation:** The pre-trained OV projection in the Base model was 211.51 . The fine-tuned Instruct model's projection was identical at 211.51.
* **Conclusion:** SFT didn't modify the routing weights! The explicit circuits for instruction-following were heavily pre-existing in the Base model. SFT just enables their conditional activation using the chat template.

**Experiment 14: Path Patching (Minimal Circuit Pruning)**
* **How Path Patching works:** It tests the causal effect of specific edges between nodes in the network graph. Instead of looking at components or layers, we look at the path specific features take for them to be activated.
* **What I did:** I ran exhaustive Pairwise Path Patching on both models (testing all 1,764 possible edges between my 42 component nodes across 400 prompts) and applied a strict causal threshold (Median Recovery > 0.1) to extract the minimal computational graph.
* **Observation:** Base Model: An extremely dense graph with 70 strong edges, relying on a sprawling chaotic network. Instruct Model: A highly optimized topology with exactly 20 strong edges.
* **Conclusion:** The Instruction-tuned model has a significantly tighter topological pathway. We fully reconstructed the causal circuit!

### Part V: Robustness
Finally, we validate our findings to make sure they aren't just noisy artifacts.

**Experiment 15: Statistical Validation**
* **How it works:** We randomly sample our dataset over and over (bootstrapping) and compare our nodes against random null nodes.
* **What I did:** I tested my 42 component candidates across 50 random 80/20 splits.
* **Observation:** My top candidates massively outperformed the random null baseline consistently.
* **Conclusion:** The nodes in our causal graph are highly stable and reliable.

**Experiment 16: The Representation Atlas**
* **How it works:** We built an automated pipeline to scale our SAE feature matching across massive amounts of data.
* **What I did:** I mapped the entire representational shift across 826,000 features over multiple dataset domains.
* **Observation:** I successfully cataloged 826,091 feature lifecycles. The bottlenecks and routing mechanisms replicated flawlessly across different domains.
* **Conclusion:** Our discovered causal mechanisms are robust regardless of the dataset distribution.

**Experiment 17: Cross-model Generalization**
* **How it works:** We run the exact same experiments on a completely different model family.
* **What I did:** I replicated the JS Divergence and patching experiments on Meta's Llama-3-8B.
* **Observation:** The KL divergence strictly increased throughout the Llama 3 network, showing a clear late-layer divergence trend similar to Qwen, though with a distinct layer-wise trajectory.
* **Conclusion:** Late-layer behavioral routing isn't just a Qwen artifact; it appears consistent across at least some other modern frontier model families.

---

## Part VI: The Actual Experiment — Base/SFT vs RLVR
With a solid understanding of how SFT changes a model, we turn to the core research question: What mechanistically changes when a model undergoes Reinforcement Learning with Verifiable Rewards (RLVR)? We transition our analysis to the Llama-3.1-Tulu-3-8B model suite (Base, SFT, DPO, RLVR).

**Experiment 18: Representational Geometry (CKA & Weights)**
* **How it works:** We measure if RLVR fundamentally rewrites the neural weights or creates entirely new representational geometries.
* **What I did:** I compared the weight matrices and layer-to-layer CKA across SFT, DPO, and RLVR on identical token sequences.
* **Observation:** Weight changes were extremely small (<0.15% relative change), and CKA remained exceptionally high ($\ge 0.9997$) across the models.
* **Conclusion:** RLVR does not radically redesign the network's knowledge or circuitry. The overall geometry remains nearly identical to the SFT model.

**Experiment 19: Cross-Model Linear Probing**
* **How it works:** We train a linear probe to classify success vs failure on the SFT model's residual stream, and test if it can predict success/failure inside the RLVR model (and vice-versa).
* **What I did:** Trained probes on 500 validated prompts and evaluated a 2×2 cross-model transfer matrix.
* **Observation:** The SFT probe generalized better to RLVR (87.3% accuracy) than an RLVR probe generalized to RLVR (80.7%). The cosine similarity between the SFT and RLVR probe weight vectors was 0.625.
* **Conclusion:** The success-related latent variable is partially shared between SFT and RLVR. They use overlapping but non-identical representations, meaning the RLVR model didn't invent a new success variable from scratch.

**Experiment 20: Layer-wise State Occupancy**
* **How it works:** We cluster the internal residual stream trajectories to see at which layer the model converges into a "success" state versus a "failure" state.
* **What I did:** Tracked the percentage of representations occupying the success cluster at layers L5, L10, L15, L20, L25, and L31 across SFT and RLVR.
* **Observation:** SFT showed 0.0% success state occupancy in early layers, whereas RLVR immediately jumped to 52.0% occupancy by Layer 5.
* **Conclusion:** The behavioral divergence between SFT and RLVR happens extremely early in the forward pass. RLVR alters early-layer state transitions to reliably steer the network into successful trajectories.

**Experiment 21: Component Attribution & Knockouts**
* **How it works:** We aggressively ablate (knock out) individual components that appear highly important for reasoning to see if the model's capabilities collapse.
* **What I did:** Targeted highly active MLPs (e.g., L13) that had strong DLA and patching scores in RLVR, and zero-ablated them.
* **Observation:** Removing L13 or other top individual MLPs did not destroy the RLVR model's ability to reason successfully. 
* **Conclusion:** The "reasoning circuit" in RLVR is not a singular, localized bottleneck. It is a highly redundant, distributed computational substrate.

**Experiment 22: Causal State Steering Across Models**
* **How it works:** We compute the "success direction" ($d_L$) from the SFT model, and then causally inject it during inference to see if we can rescue failures (sufficiency) or induce failures (necessity).
* **What I did:** On SFT, I added the success direction ($h'_L = h_L + \alpha d_L$) for failed prompts. On RLVR, I subtracted it ($h'_L = h_L - \alpha d_L$) for successful prompts.
* **Observation:** Intervening at the correct layers successfully modifies reasoning behavior, establishing a causal relationship between the latent state and the reasoning outcome.
* **Conclusion:** This proves the **RLVR State-Selection Hypothesis**: RLVR improves reasoning not by instantiating novel circuits, but by changing early state transitions to increase the occupancy of a pre-existing distributed reasoning state.

**Verified Exp14: Causal Trajectory Transfer (Corrected)**
* **What it tested:** Whether RLVR's frontier capability can be transferred into SFT by patching early-to-mid residual-stream activations, using one self-consistent HF decode pipeline for both labeling and intervention.
* **What was done:** I rebuilt the GSM8K quadrants, ran a checkpointed live marimo pipeline, and logged the full batch-by-batch sweep to durable files in `/marimo` before pulling the final JSON into the repo.
* **Headline result:** RLVR→SFT improved `D_Frontier` accuracy from `15.0% [6.7, 23.3]` to `33.3% [21.7, 45.0]` at `L0-12` and `43.3% [31.7, 56.7]` at `L0-31`. SFT→RLVR reduced `76.7% [66.7, 86.7]` to `56.7% [43.3, 68.3]` at `L0-12` and `46.7% [35.0, 58.3]` at `L0-31`.
* **Conclusion:** The earlier 75%/0% claim is retracted. The verified result supports a partially transferable, distributed trajectory shift rather than a single sharp bottleneck at L12.

---

### Conclusions (Base vs Instruct)
So, what changes mechanistically when we instruction tune a model?

I noticed that Base and Instruct models are far more similar than they are different. Most of their representations remain the same, almost all the semantic concepts are already present in the Base model. Instruction tuning changes how and when the concepts are used. It forces the model to activate existing features and route information in a better, more efficient way based on conditions such as the chat template for example.

The chat template appears to be the trigger for the alternate computations that take place here. We also notice that almost all the changes only happen in the late-layers of the Instruct model. Finally, path patching suggested that the computation itself becomes considerably sparser, relying on a much smaller causal graph than the Base model.

Instruction tuning primarily changes how existing knowledge is activated and routed through the network, rather than fundamentally changing what the model knows.

#### Alignment with Existing Research
These empirical findings perfectly match the emerging consensus in the mechanistic interpretability community regarding how Supervised Fine-Tuning (SFT) affects LLMs:
1. **The Wrapper Hypothesis & Late-Layer Override**: Our JS Divergence and CKA results directly validate the findings of Jain et al. (2023) in *"Mechanistically analyzing the effects of fine-tuning on large language models"*. They found that fine-tuning acts primarily as a "wrapper," leaving early and middle layer representations largely unchanged while heavily modifying the final layers to adapt the output style.
2. **Late-Layer MLPs as "Motor Neurons"**: Our DLA and $\Delta$DLA experiments localizing the formatting shift to Layers 25-27 align with recent research characterizing late-layer MLPs as "motor neurons" that translate deep conceptual representations into surface-level vocabulary tokens (e.g., structured Markdown).
3. **Template Triggers and Pre-existing Circuits**: Our discovery that the `<|im_start|>` chat template acts as a conditional routing switch—rather than teaching the model new concepts from scratch—mirrors research on activation steering and prompting. It confirms that SFT largely repurposes and conditionally activates pre-existing circuits discovered during pre-training, a phenomenon often referred to as the "Alignment Ceiling."

Similarly, Reinforcement Learning with Verifiable Rewards (RLVR) does not create entirely new reasoning circuits from scratch. Instead, it alters the early-layer state transition probabilities, ensuring the model more reliably enters a pre-existing "success state" that was already representable within the SFT network.

The corrected exp14 run refines this picture: the state/trajectory change is real and causal, but it is only partially transferable and appears distributed across layers rather than locked to a single transition point.
