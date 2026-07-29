# Exhaustive Literature Review: Alignment-Aware SFT and Related Works

**Sources searched**: arXiv (10+ independent queries across technique families), covering 2019–2026. Papers examined: ~45+ unique works.

---

## Summary Verdict

> [!IMPORTANT]
> The **specific combination** — using intra-model, cross-prompt hidden-state alignment as a *training-time auxiliary loss* to address the knowing-using gap — has **not been found in any existing work**. However, every individual component exists in the prior literature, and several works are close enough to require honest differentiation. Our novelty claim should be modest: a **novel application and combination** of existing techniques to a newly-identified SFT failure mode.

The honest novelty statement:
> *"To our knowledge, we are the first to propose using intra-model, cross-prompt representation alignment as a training-time auxiliary loss during SFT, motivated by the mechanistically-identified knowledge-circuit misalignment (the Knowing-Using Gap). Prior work either uses representation distillation for model compression (requiring a separate teacher model), or uses activation steering for inference-time behavior control (not updating weights)."*

---

## Part 1: The Layer-Wise Knowledge Distillation Family

This is the most technically similar family of work. All methods here match hidden states across layers during training, but exclusively for model compression between two separate models.

### TinyBERT (Jiao et al., 2019 — arXiv:1909.10351)
**What they do**: Distills a large BERT into a tiny one by matching: (1) embedding layer outputs, (2) attention weight matrices, and (3) hidden state vectors layer-by-layer, using MSE loss.

**Formula**: $\mathcal{L}_\text{hidn} = \text{MSE}(H^S W_h, H^T)$ where $W_h$ is a learnable projection matrix from student-dim to teacher-dim.

**Key differences from ours**:
- Two-model (Teacher→Student), not intra-model self-distillation.
- Matches same-position tokens in the *same* prompt, not entity-tokens across different prompts.
- Uses MSE, not Cosine Distance.
- Goal: compression, not routing alignment within one model.

**Should we borrow?** Yes — one specific idea: TinyBERT adds a **learnable projection matrix** $W_h$ because their hidden dimensions differ. We should consider this if the representation geometry at $l_s^\text{early}$ differs substantially from $l_t$. This is already partially addressed by our Variant 2 (Probing Loss), which is structurally equivalent.

---

### Patient Knowledge Distillation / PKD (Sun et al., 2019)
**What they do**: Forces the student to be "patient" — instead of only matching the teacher's final layer output, it matches the teacher's last $k$ intermediate layers iteratively, waiting for each layer to stabilize before moving on.

**Key differences from ours**:
- Still two-model, same prompt.
- The "patience" concept is philosophically analogous to our warmup period $K$ before activating the alignment loss.

**Should we borrow?** The **warmup-before-alignment** design our plan already uses is exactly the "patience" insight from PKD. We're already doing the right thing intuitively.

---

### MiniLLM (Gu et al., 2023 — arXiv:2306.08543)
**What they do**: Distills large autoregressive LLMs into smaller ones. Key insight: switches from forward KL divergence (which causes the student to spread probability mass too broadly) to **reverse KL divergence** for on-policy output distillation.

**Key differences from ours**:
- Two-model, output-space only (logit distributions), not hidden states.
- Does not address internal routing or multi-hop reasoning.

**Should we borrow?** No — reverse KLD is for output probability distribution mismatch. Our problem is in the representation space, not the output space. OPRD already empirically debunked output-space losses as insufficient.

---

### IBKD — Text Representation Distillation via Information Bottleneck (Zhang et al., 2023 — arXiv:2311.05472)
**What they do**: Distills knowledge between models by maximizing mutual information between teacher and student final representations, while minimizing mutual information between the student representation and the input data. Uses the Information Bottleneck principle rather than direct MSE or KL matching.

**Key differences from ours**:
- Two-model, same prompt, compression goal.
- Focuses on a single representation layer (the final output representation), not cross-layer routing.
- Uses mutual information objectives (not cosine distance).

**Should we borrow?** The mutual information framing is intellectually interesting but computationally expensive. Our Cosine Distance is a simpler, cheaper proxy that achieves the same directional alignment goal. No direct adoption recommended.

---

### OPRD — On-Policy Representation Distillation (Yang et al., 2026 — arXiv:2606.06021)

**What they do**: Lifts KD from the output space into the hidden-state space, aligning intermediate representations of a student model to a teacher model during on-policy rollouts. Bypasses the LM head entirely, providing a deterministic per-sample gradient. Extends to cross-architecture via a "frozen projector pair."

**How our method differs (detailed comparison)**:

| Feature | OPRD (Yang et al.) | Our Proposed Method (Faster-SFT) |
| :--- | :--- | :--- |
| **Goal** | Model compression / capability transfer | Fixing the internal Knowing-Using Gap |
| **Models Involved** | Two models (Teacher → Student) | One model (Self-Distillation) |
| **Prompt Used** | Same prompt for both models | **Different prompts** (Memorization Teacher → Reasoning Student) |
| **Layer Alignment** | Matches Layer $X$ of Teacher to Layer $Y$ of Student | Matches **Storage Layers** ($l_s$) to **Reasoning Layer** ($l_t$) within the same model |
| **Loss Metric** | Cosine similarity on hidden states | Cosine similarity on hidden states |
| **Loss Target** | Compresses knowledge into a smaller architecture | Forces knowledge to route through specific pre-trained reasoning circuits |
| **Teacher in memory?** | Yes — full large model must stay loaded | No — teacher is just a cached hidden state from a previous forward pass |

**Should we borrow?**

> [!TIP]
> **Validation of Representation over Output Loss (Why no KL-Divergence):**
> OPRD's primary thesis is that matching output-space probability distributions (like KL Divergence) creates a "high-variance gradient estimator" and an "information bottleneck." By aligning hidden states directly (like we do with Cosine Distance), OPRD proved that you get a deterministic, dense per-sample gradient. **This strongly validates our decision to use Cosine Distance on the latent vectors instead of KL divergence on the logits.**

> [!NOTE]
> **The "Frozen Projector" Concept:**
> For cross-architecture distillation, OPRD uses a "Frozen Projector Pair" (linear transformations) to align representations that have different dimensions or structures.
> *Takeaway:* If we ever find that the entity representation at the early storage layer ($l_s$) has fundamentally shifted its geometrical structure by the time it reaches the target layer ($l_t$), we could insert a lightweight, frozen linear projector to translate the vector space before applying our Cosine Distance loss. This is conceptually equivalent to our Variant 2 "Probing Loss."

---

## Part 2: The Knowledge Editing Family

This family tries to insert facts into specific layers by surgically modifying weights. They converge on the same routing problem we have identified — but none proposed a training-time loss as the solution.

### ROME (Meng et al., 2022) / MEMIT (Meng et al., 2023)
**What they do**: Identify the exact MLP layers where a fact is stored (via causal tracing), then surgically modify those weights using a rank-one update to insert a new fact.

**Key differences from ours**:
- ROME/MEMIT modify weights at a *specific layer* (typically layer 13–17 in GPT-2-XL).
- They insert facts into late-storage MLP layers, not middle-layer reasoning circuits.
- They are post-hoc weight edits, not training-time auxiliary losses.

**Critically relevant extensions**: Two 2024–2026 papers extending ROME directly discover the **same routing problem** we are solving:

#### "On the Limitations of Rank-One Model Editing in Answering Multi-hop Questions" (arXiv:2601.04600)
Identifies the **"hopping-too-late" problem**: when ROME edits deeper layers, those layers lack access to necessary intermediate representations for multi-hop reasoning. This is mechanistically **the same failure mode** we identified — facts written to late layers arrive too late in the forward pass for middle-layer reasoning circuits to use them.

#### "ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall" (arXiv:2510.07896)
Discovers that "implicit subjects function as query neurons, which sequentially activate corresponding value neurons across transformer layers" — multi-hop reasoning is a layer-sequential process. Proposes editing at **multiple layers** along this chain rather than a single late layer.

#### "Enhancing Multi-hop Reasoning through Knowledge Erasure in LLM Editing" (arXiv:2408.12456)
Hypothesizes that residual single-hop knowledge after editing causes edited models to revert to their original answers on multi-hop queries. Validates this experimentally.

> [!IMPORTANT]
> These papers provide strong, independent confirmation of our core hypothesis from the mechanistic interpretability direction. They did not propose a training-time loss, but they provide converging evidence that the layer-routing problem is real and structurally important. They should be cited as supporting evidence.

---

## Part 3: The Activation Steering / Representation Engineering Family

These methods work directly on representations — but at inference time only, never updating weights.

### Representation Engineering / Activation Steering (various, 2023–2026)
**What they do**: Identify specific directions in the residual stream corresponding to behaviors (e.g., "hallucination," "tool use"), and either steer or suppress activations at inference time.

Representative papers found:
- **ASA (arXiv:2602.02935)**: Identifies that tool-use necessity is linearly decodable from mid-layer activations, yet the model doesn't act on it. Proposes inference-time mid-layer steering.
- **AAC (arXiv:2603.10195)**: Uses layer-wise linear probing to identify "Hallucination Nodes" and suppresses them via a forward hook at inference time.
- **FairSteer (arXiv:2504.14492)**: Computes "steering vectors" from contrastive activations, injects them at inference time.
- **MSRS (arXiv:2508.10599)**: Multi-attribute steering via orthogonal subspace allocation at inference time.

**Key differences from ours**:
- All activation steering methods are **inference-time**, not training-time. They patch the model during generation but do not update weights.
- Our method updates weights permanently via backpropagation. The model routes correctly on its own after training, without any inference-time interventions.

**Critical insight to borrow**: The **linear probe approach** these papers use is exactly what we use in our Metric 1 (Layer Profiling) and Variant 2 (Probing Loss). These papers validate that linear probing is a reliable and computationally cheap way to identify "knowledge-accessible" layers. We are on solid methodological ground here.

---

## Part 4: The Mechanistic Interpretability / Knowledge Elicitation Family

### MechELK (arXiv:2605.28825)
**What they do**: Three-stage framework to elicit *latent knowledge* — knowledge encoded in the model's representations but not reflected in outputs. Specifically: (1) Locate latent knowledge via Sparse Autoencoder (SAE) feature analysis, (2) Amplify it via targeted fine-tuning, (3) Evaluate faithfulness.

**Key differences from ours**:
- MechELK elicits knowledge that already exists in a pre-trained model (it's already "in there" but suppressed).
- We are trying to inject *new* facts via SFT and then route them into the right layers.

**What overlaps**: MechELK's Stage 2 ("Amplify via targeted fine-tuning") is the closest prior work philosophically — they specifically fine-tune targeting specific layers to make latent knowledge more accessible. This is the same philosophy as our alignment loss.

> [!NOTE]
> MechELK targets *pre-existing* latent knowledge; we target *newly injected SFT facts*. That is the key distinction. If MechELK appears in a reviewer's head, our paper should explicitly draw this line.

---

## Part 5: The Self-Improvement / Cross-Prompt SFT Family

### "LLMs Can Self-Improve" (Huang et al., 2022 — arXiv:2210.11610)
**What they do**: Use the model's own CoT reasoning outputs as training targets. The model generates "high-confidence" rationale-augmented answers and fine-tunes on those self-generated solutions.

**Key differences from ours**:
- Operates in output space (generated text), not representation space (hidden states).
- Does not target specific layers.
- Does not use a dual-prompt (memorization vs. reasoning) structure.

**Philosophical overlap**: The idea of a model supervising itself (self-distillation) is present. We extend this to the *representation level*: the model uses its own hidden states from one prompt type to supervise another.

---

## Part 6: Targeted Lexical Injection (arXiv:2506.15415) — Most Surprising Find

**What they do**: Shows that Swahili-English word alignment reaches near-perfect cosine similarity at **Layer 2** of Llama (early layers). Fine-tunes specifically on early-layer LoRA to inject cross-lingual lexical alignment.

**Key overlap**:
- Demonstrates that targeting **early layers specifically** (via targeted LoRA rank allocation) is a viable and effective SFT strategy.
- Validates that early-layer representations are the most "pure" and lexically grounded.

**This directly supports** our plan's $l_s^\text{early}$ selection strategy and the idea that early layers are the reliable "knowledge crystallization" point.

---

## Overall Comparison Table

| Paper | Goal | Teacher-Student? | Hidden State Loss? | Cross-Prompt? | Layer-Targeted? | Training-Time? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| TinyBERT | Compression | ✅ Two-model | ✅ MSE | ❌ | ❌ All layers | ✅ |
| PKD | Compression | ✅ Two-model | ✅ MSE | ❌ | ❌ Last-k layers | ✅ |
| MiniLLM | Compression | ✅ Two-model | ❌ KL-div logits | ❌ | ❌ | ✅ |
| IBKD | Compression | ✅ Two-model | ✅ Mutual info | ❌ | ❌ Final layer | ✅ |
| OPRD | Compression | ✅ Two-model | ✅ Cosine | ❌ | ✅ Selected | ✅ |
| ROME/MEMIT | Knowledge editing | ❌ One-model | ❌ Weight edit | ❌ | ✅ Specific | ❌ Post-hoc |
| ACE | KE (multi-hop) | ❌ One-model | ❌ Weight edit | ❌ | ✅ Multi-layer | ❌ Post-hoc |
| Activation Steering | Behavior control | ❌ One-model | ✅ Direction | ❌ | ✅ Specific | ❌ Inference |
| MechELK | Latent knowledge | ❌ One-model | ✅ SAE features | ❌ | ✅ Specific | ✅ (partial) |
| LLM Self-Improve | Reasoning ability | ❌ One-model | ❌ Output text | ✅ CoT prompts | ❌ | ✅ |
| Targeted Lex. Injection | Cross-lingual | ❌ One-model | ❌ LoRA weights | ❌ | ✅ Early layers | ✅ |
| **Ours (RepDist)** | **SFT routing** | **❌ One-model** | **✅ Cosine** | **✅ Mem→Reason** | **✅ Profiled** | **✅** |

The combination of (**One-model** + **Hidden State Loss** + **Cross-Prompt** + **Layer-Targeted** + **Training-Time**) is unique to our method across all surveyed work.

---

## Actionable Recommendations

1. **Adopt from TinyBERT**: Consider adding a small learnable projector matrix between $h_E^{l_s}$ and $h_E^{l_t}$ if their dimensions or geometric structure differ post-warmup. Already partially addressed by Variant 2 (Probing Loss), but worth testing in ablations as a Variant 5.

2. **Adopt from OPRD**: Explicitly cite OPRD and state that their empirical findings validate our choice of Cosine Distance over KL-divergence — their gradient variance analysis is our strongest theoretical backing for loss design.

3. **Adopt from ACE + ROME multi-hop failures**: Cite these as independent mechanistic confirmation of the "hopping-too-late" routing problem. Crucially, **none of them proposed a training-time auxiliary loss as the solution** — that is our contribution.

4. **Add MechELK to Related Works with clear distinction**: MechELK is the closest training-time method. The explicit distinction is: *pre-existing latent knowledge* vs. *newly injected SFT facts*. Reviewers will raise this.

5. **The novelty claim to make (humbly)**:
   > "To our knowledge, we are the first to propose using intra-model, cross-prompt representation alignment as a training-time auxiliary loss during SFT, motivated by the mechanistically-identified knowledge-circuit misalignment (the Knowing-Using Gap). Prior work either uses representation distillation for model compression (requiring a separate teacher model), or uses activation steering for inference-time behavior control (not updating weights). Our work is the first to turn self-patching into a differentiable training objective."

---

*Literature search conducted via arXiv API across 10+ independent queries covering the following technique families: layer-wise knowledge distillation, on-policy representation distillation, knowledge editing, mechanistic interpretability, activation steering, contrastive SFT, and self-improvement. Papers reviewed: arXiv:1909.10351 (TinyBERT), arXiv:2306.08543 (MiniLLM), arXiv:2311.05472 (IBKD), arXiv:2606.06021 (OPRD), arXiv:2601.04600 (ROME multi-hop limits), arXiv:2510.07896 (ACE), arXiv:2408.12456 (Knowledge Erasure), arXiv:2605.28825 (MechELK), arXiv:2602.02935 (ASA), arXiv:2603.10195 (AAC), arXiv:2504.14492 (FairSteer), arXiv:2508.10599 (MSRS), arXiv:2506.15415 (Targeted Lexical Injection), arXiv:2210.11610 (LLM Self-Improve), arXiv:2607.08393 (Knowing-Using Gap).*
