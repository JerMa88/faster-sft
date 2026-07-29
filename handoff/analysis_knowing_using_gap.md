# Analysis: Why Middle Layers Are Slow & Self-Patching as a Remedy

## Paper Under Analysis
**"Why Memorized Knowledge Fails to Generalize in LLM Finetuning"** (arXiv:2607.08393)
— identifies the **Knowing-Using Gap**: fine-tuned LLMs memorize new facts in early/late layers but fail to use them for multi-hop reasoning because the middle layers (where reasoning circuits live) never receive the representation.

## Grounding Literature
- [Physics of LM Part 3.1](file:///Users/zma/Documents/programs/faster-sft/related_works/Physics_of_LM_Part3.1_Knowledge_Storage_Extraction_Allen-Zhu_Li.pdf) — Knowledge Storage & Extraction (arXiv:2309.14316)
- [Physics of LM Part 3.2](file:///Users/zma/Documents/programs/faster-sft/related_works/Physics_of_LM_Part3.2_Knowledge_Manipulation_Allen-Zhu_Li.pdf) — Knowledge Manipulation (arXiv:2309.14402)
- [Physics of LM Part 3.3](file:///Users/zma/Documents/programs/faster-sft/related_works/Physics_of_LM_Part3.3_Knowledge_Capacity_Scaling_Allen-Zhu_Li.pdf) — Knowledge Capacity Scaling Laws (arXiv:2404.05405)

---

## Question 1: Why Are Middle Layers Slow to Adapt During SFT?

### Short Answer
**It's neither that middle layers are "less expressive" nor purely a gradient-flow problem. It's a *routing* problem: SFT's gradient signal first writes facts into layers that are easy to fit (early/late), and the middle-layer reasoning circuits—whose pre-trained weights are already specialized for compositional processing—receive almost no learning signal about the new fact.**

### Detailed Reasoning

#### 1A. The optimization landscape favors early/late layers

During SFT with a factual-recall loss (e.g., "Which protein is expressed in embryo? → IGFBP3"), the gradient directly rewards layers that produce the correct next-token. In a transformer:

- **Late layers** (final ~20% of depth) sit closest to the output logits. Their MLPs act as "memory heads" that resolve entity→attribute lookups via key-value associations (Meng et al., 2022 — ROME; Geva et al., 2023). The gradient signal is strong and direct here.
- **Early layers** (first ~10-20%) encode entity identity via attention to name tokens. The embedding + early-MLP pathway can quickly learn to associate a new entity name with a new fact vector. Allen-Zhu & Li (Part 3.1) showed this directly: with knowledge augmentation, facts are stored **linearly on entity name embeddings** in early layers. Without augmentation, knowledge disperses across all token positions and becomes inaccessible.
- **Middle layers** (~30-70% depth) run the pre-trained **reasoning circuits**: bridge-entity resolution, multi-hop composition, intersection filtering. These circuits were learned during pre-training on billions of tokens and are *already functionally specialized*. Fine-tuning on a few thousand new facts does not generate a loss signal that pushes these circuits to incorporate the new knowledge, because the memorization objective is already satisfied by the early/late pathway.

> [!IMPORTANT]
> The middle layers aren't inherently harder to backpropagate through. They're hard to **adapt** because the SFT loss is satisfied *before* the gradient signal ever needs to reach them. Memorization saturates in ~2-3 epochs; generalization (which requires middle-layer integration) lags by 4-5 epochs and often never fully converges.

#### 1B. Allen-Zhu & Li's "Physics of Language Models" explains *why* this is structural

Allen-Zhu & Li's key findings that ground this understanding:

| PoLM Finding | Implication for the Knowing-Using Gap |
|---|---|
| **Part 3.1**: Knowledge is only extractable if it's stored *linearly on entity-name tokens*. Without augmentation, the model memorizes the fact but disperses it across all tokens, making QA extraction ~0% accurate. | SFT writes knowledge onto the entity position in early layers (where linear storage is easiest), but this encoding is *not in the format* that middle-layer reasoning circuits expect as input. The middle layers need the fact to arrive as a contextualized representation through the residual stream, not as a static early-layer embedding. |
| **Part 3.2**: Models excel at retrieval but fail at classification, comparison, and inverse search — unless **Chain-of-Thought** explicitly decomposes the task. CoT is needed for both training *and* inference. | The middle layers can manipulate knowledge *only if it's presented to them in the right format at the right position*. SFT doesn't naturally produce this routing. CoT partially works by forcing the model to first *recall* (using early/late) and then *reason* (using middle) in separate forward passes / generation steps. But as the paper shows, even CoT remains far below self-patching performance. |
| **Part 3.2**: Inverse knowledge search is ~0% accurate — the autoregressive architecture cannot "look back." | Multi-hop reasoning requires the model to resolve a bridge entity (Fact A → bridge → Fact B). If Fact A is stored in late layers, the resolution happens *after* the middle-layer composition window. The fact literally arrives too late in the forward pass. This is why *late→mid* patching is so effective. |
| **Part 3.3**: Models can store ~2 bits/parameter; capacity isn't the bottleneck. | The gap isn't about running out of room. There's plenty of capacity. It's about where knowledge is *routed* during the forward pass. |

#### 1C. It's not a vanishing gradient problem per se

A common intuition is "middle layers are hard because of vanishing gradients." This is **not quite right** for modern transformers with:
- Residual connections (additive skip connections at every layer)
- LayerNorm / RMSNorm
- Pre-norm architectures (used in Qwen, LLaMA)

These ensure gradients flow healthily to all layers. The paper's own evidence refutes a pure gradient-flow explanation:

1. **Early→mid patching works just as well as late→mid.** If the problem were that early layers couldn't get gradient signal, early layers wouldn't have the knowledge to begin with. But they do — early layers encode the fact perfectly.
2. **The gap persists across model scales** (1B to 8B), which wouldn't happen if it were a simple gradient pathology that deeper models would overcome.
3. **Full fine-tuning shows the *same* pattern as LoRA** (Table 3 in the paper). FFT memorizes faster but doesn't close the generalization gap — ruling out "LoRA's rank is too low for middle layers."

> [!TIP]
> **The real bottleneck**: After memorization loss hits zero, the gradient for the memorization objective vanishes naturally. There is no remaining loss signal to push the representation into the middle-layer reasoning path. The fact is "learned" from the loss's perspective, but it's learned in the wrong place. Continued training *sometimes* leads to grokking-like delayed generalization as weight diffusion slowly moves the representation, but this is unreliable and slow.

---

## Question 2: Self-Patching — Implementation, SOTA Status, and Improvement Opportunities

### 2A. What Self-Patching Does (Implementation)

Self-patching is a **post-hoc causal intervention**, not a training method. The algorithm:

```
For each layer pair (l_src, l_tgt):
  1. Run the memorization prompt P_mem through the model
  2. Cache the hidden state at entity position from layer l_src:
       z = h[l_src][entity_tokens](P_mem)
  3. Run the generalization prompt P_gen through the model
  4. At layer l_tgt, replace the entity-position hidden state:
       h̃[l_tgt][entity_tokens] ← z
  5. Continue the forward pass from l_tgt onward
  6. Measure: ΔAcc = Acc(patched) - Acc(unpatched)
```

This produces an L×L **permeation heatmap** where each cell shows the accuracy gain from patching source→target.

### 2B. Key Findings from the Heatmap

The paper discovers **two concentrated clusters** of effective patches (Figure 5):

```
Cluster 1: Late→Mid  (source ~0.8L → target ~0.5L)
Cluster 2: Early→Mid (source ~0.1L → target ~0.5L)
```

**Fixed heuristic**: Using just these two predetermined layer pairs (no per-instance search), they recover **58–75% of oracle headroom**.

### 2C. Is It SOTA?

> [!WARNING]
> **Self-patching is a *diagnostic tool*, not a training or inference method.** The authors are very explicit about this (Section 5.5, Appendix H). It requires:
> - Knowing the correct answer (to evaluate ΔAcc)
> - Access to the memorization prompt
> - Running modified forward passes at inference time
>
> So it's not directly comparable to training methods. There is no existing "SOTA" in this exact problem framing, because the paper *defines* the problem (Knowing-Using Gap) and the diagnostic (self-patching) for the first time.

**What it is SOTA at**: Demonstrating that the gap is a routing problem and quantifying the recoverable headroom. It substantially outperforms:
- **CoT prompting**: Only partially helps chaining, sometimes hurts intersection
- **Irrelevant patching** (control): Shows the gain is fact-specific, not just a perturbation artifact
- **Standard fine-tuning convergence**: Even at convergence, unpatched models plateau far below patched performance

### 2D. What Can Be Improved — Research Directions

Here are concrete directions to move from diagnostic to practical method, ordered roughly by feasibility:

#### Direction 1: Alignment-Aware Training Loss (Most Promising)

Instead of just training on the memorization loss, add an **auxiliary loss that rewards middle-layer activation of the new fact**:

```
L_total = L_memorization + λ · L_alignment
```

Where `L_alignment` could be:
- **Representation matching**: Force `h[mid_layer][entity]` during generalization prompts to be close to `h[late_layer][entity]` during memorization prompts. Essentially, **distill the patching operation into the training objective**.
- **Contrastive routing loss**: Ensure the entity representation at the middle layer is more similar to the *correct* fact than to random facts.
- **Probing loss** (inspired by PoLM 3.1): Train a linear probe on middle-layer representations to predict the fact's attributes, and backpropagate through the probe.

> [!TIP]
> The paper's fixed heuristic (source at ~0.8L/~0.1L, target at ~0.5L) gives you the exact layer indices to use for this auxiliary loss. No per-instance search needed.

#### Direction 2: Adaptive Patching at Inference Time

The fixed heuristic recovers 58–75%. The remaining 25–42% gap is from **instance-specific variation** in where knowledge is stored. Ideas:

- **Learned patch selector**: Train a lightweight classifier that, given a query, predicts the optimal (l_src, l_tgt) pair. The paper's heatmaps provide training data for this.
- **Soft patching**: Instead of hard-copying representations, learn a **gating/interpolation** between the original and patched representation: `h̃ = α·h_original + (1-α)·z_patched`, where α is learned.
- **Multi-pass inference**: Run the model once normally, identify which entity-layer combinations have high-confidence representations (via a fast probe), then re-run with targeted patches.

#### Direction 3: Architecture Modifications

- **Cross-layer attention / representation sharing**: Add skip connections specifically from early→mid and late→mid for entity-position tokens. This is analogous to what DenseNet does for CNNs, but targeted.
- **Mixture-of-Depths routing**: Use adaptive computation to *extend* processing at middle layers for queries that require multi-hop reasoning.
- **LoRA with layer-asymmetric rank allocation**: Assign higher LoRA rank to middle layers during SFT, since those are the layers that need the most adaptation but receive the least gradient signal under standard training.

#### Direction 4: Training Data / Curriculum Interventions (Cheapest)

Inspired directly by Allen-Zhu & Li's Part 3.1:

- **Knowledge augmentation during SFT**: Present facts in multiple phrasings, orderings, and contexts so the gradient forces the model to build a *generalizable* representation rather than a position-specific one.
- **Interleave memorization and reasoning**: Instead of training all facts first, interleave fact-memorization batches with reasoning batches that *use* the same facts. This forces the gradient to care about middle-layer routing *while* the memorization gradient is still active.
- **Curriculum**: Teach simple recall first, then same-hop reasoning, then multi-hop — so the loss signal at each stage pushes representations deeper into the reasoning layers.

#### Direction 5: Extend Beyond KG Facts

The current paper only validates on knowledge-graph facts (biomedical + academic). Key extensions:
- **Procedural knowledge** (how-to instructions)
- **Mathematical facts** (formulas, theorems)
- **Coding patterns** (API signatures → usage)
- **Safety-critical knowledge** (alignment facts the model should *use* not just *recall*)

---

## Summary Table

| Aspect | Current Paper | Allen-Zhu & Li (PoLM) | Gap / Opportunity |
|---|---|---|---|
| **What's diagnosed** | Knowledge stored in wrong layers after SFT | Knowledge stored in inaccessible format without augmentation | Both identify a *storage-access mismatch*, but from different angles |
| **Root cause** | Routing: fact never reaches middle-layer reasoning circuits | Encoding: fact not linearly stored on entity tokens without augmentation | The two are complementary — PoLM explains *how* knowledge should be encoded; this paper explains *where* it needs to go |
| **Fix (diagnostic)** | Self-patching: copy representation to middle layer | Knowledge augmentation in pretraining | Neither is a training-time fix for SFT specifically |
| **Fix (practical)** | Fixed heuristic (58-75% recovery) | More diverse training data | **An alignment-aware training loss that combines both insights is the clear next step** |
| **Scope** | KG facts, biomedical + academic | Synthetic biographies | Both need extension to real-world diverse knowledge |

---

## Saved Literature

All relevant PDFs are saved in [related_works/](file:///Users/zma/Documents/programs/faster-sft/related_works/):

| File | Paper |
|---|---|
| [2607.08393v1.pdf](file:///Users/zma/Documents/programs/faster-sft/related_works/2607.08393v1.pdf) | "Why Memorized Knowledge Fails to Generalize in LLM Finetuning" |
| [Physics_of_LM_Part3.1_Knowledge_Storage_Extraction_Allen-Zhu_Li.pdf](file:///Users/zma/Documents/programs/faster-sft/related_works/Physics_of_LM_Part3.1_Knowledge_Storage_Extraction_Allen-Zhu_Li.pdf) | Physics of LM: Part 3.1, Knowledge Storage and Extraction (arXiv:2309.14316) |
| [Physics_of_LM_Part3.2_Knowledge_Manipulation_Allen-Zhu_Li.pdf](file:///Users/zma/Documents/programs/faster-sft/related_works/Physics_of_LM_Part3.2_Knowledge_Manipulation_Allen-Zhu_Li.pdf) | Physics of LM: Part 3.2, Knowledge Manipulation (arXiv:2309.14402) |
| [Physics_of_LM_Part3.3_Knowledge_Capacity_Scaling_Allen-Zhu_Li.pdf](file:///Users/zma/Documents/programs/faster-sft/related_works/Physics_of_LM_Part3.3_Knowledge_Capacity_Scaling_Allen-Zhu_Li.pdf) | Physics of LM: Part 3.3, Knowledge Capacity Scaling Laws (arXiv:2404.05405) |
