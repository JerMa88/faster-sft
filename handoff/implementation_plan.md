# Alignment-Aware SFT: Closing the Knowing-Using Gap During Training

## Goal

Prove that adding an **alignment-aware auxiliary loss** during SFT produces a **Pareto improvement** over standard SFT: both **faster convergence** (smaller temporal lag ΔT) **and** a **higher final ceiling** (smaller accuracy gap ΔA) on downstream multi-hop reasoning tasks — without increasing model size or training data.

## Background

The Knowing-Using Gap paper (arXiv:2607.08393) shows that SFT writes facts into early/late layers while middle-layer reasoning circuits go unmodified. Self-patching (a post-hoc diagnostic) recovers 58–75% of the reasoning headroom by manually copying representations into mid-layers. We turn this diagnostic into a differentiable training objective.

Allen-Zhu & Li's PoLM (Parts 3.1–3.3) shows that knowledge must be linearly encoded on entity tokens to be extractable, and that knowledge manipulation requires explicit routing (CoT) — further supporting the need for training-time alignment.

---

## Part 1: Empirical Layer Profiling

> [!IMPORTANT]
> We do **not** use arbitrary depth fractions (e.g., 0.1L, 0.5L, 0.8L) to pick layer indices. Instead, we run a dedicated profiling pass on the pre-trained (un-finetuned) model to identify the three critical layer indices via three independent metrics. These indices become the **data-driven hyperparameters** for the alignment loss.

### The Three Critical Layer Types

| Layer Role | Symbol | What Happens There |
|---|---|---|
| **Early storage** | $l_s^\text{early}$ | Knowledge is first encoded linearly on entity tokens |
| **Reasoning bottleneck** | $l_t$ | Where reasoning circuits operate; where the fact is needed but missing |
| **Late storage** | $l_s^\text{late}$ | Knowledge is re-encoded near the output in final MLP passes |

### Metric 1: Per-Layer Linear Probe Accuracy → Finds $l_s^\text{early}$ and $l_s^\text{late}$

**Rationale**: PoLM Part 3.1 shows that extractable knowledge is stored *linearly* on entity name tokens. We can detect storage layers by measuring how accurately a linear classifier can decode the target attribute from the entity-position hidden state at each layer.

**Procedure**:

1. Run the pre-trained model on a held-out set of 200 memorization prompts $P_\text{mem}^{(i)}$ with known answer $y^{*(i)}$
2. Extract the entity-position hidden state $h_E^l$ at every layer $l = 1, \ldots, L$ using forward hooks
3. Train a separate logistic regression probe $\phi_l$ for each layer on 160 examples, evaluate on 40:

$$\text{ProbeAcc}(l) = \frac{1}{40} \sum_{i} \mathbb{1}\!\left[\arg\max \phi_l\!\left(h_E^l(P_\text{mem}^{(i)})\right) = y^{*(i)}\right]$$

4. Plot $\text{ProbeAcc}(l)$ vs. $l$. The profile will show:
   - A **first local peak** at low layer depth → $l_s^\text{early}$ (the earliest layer with high probe accuracy, i.e., earliest storage crystallization)
   - A **second high-accuracy plateau** at high layer depth → $l_s^\text{late}$ (the last layer before probe accuracy stops rising, i.e., final storage)

**Selection rule**:

$$l_s^\text{early} = \arg\min_{l : \text{ProbeAcc}(l) > \theta_\text{early}} l \quad \text{(first layer exceeding threshold } \theta_\text{early} = 0.6 \text{)}$$

$$l_s^\text{late} = \arg\max_{l : \text{ProbeAcc}(l) > \theta_\text{late}} l \quad \text{(last layer with consistently high probe accuracy, } \theta_\text{late} = 0.85 \text{)}$$

### Metric 2: Self-Patching Gain Map → Finds $l_t$

**Rationale**: The paper's own self-patching scan produces a layer-pair heatmap $A[l_s, l_t] = \Delta\text{Acc}$ that directly tells us which target layer $l_t$ produces the largest reasoning gain when the correct representation is injected. This is the most direct measure of which layer is the reasoning bottleneck.

**Procedure**:

1. Run a **baseline SFT** for $K$ warmup epochs (just enough for memorization to saturate, $A_\text{mem} > 0.95$)
2. For every layer pair $(l_s, l_t)$, execute the self-patching scan on 100 facts from the STaRK-Prime validation set:

$$A[l_s, l_t] = \frac{1}{100} \sum_i \left(\text{Acc}(\tilde{M}(P_\text{gen}^{(i)}), y^{*(i)}) - \text{Acc}(M(P_\text{gen}^{(i)}), y^{*(i)})\right)$$

where $\tilde{M}$ denotes the model with $h_E^{l_t}(P_\text{gen}) \leftarrow h_E^{l_s}(P_\text{mem})$

3. Identify the target layer:

$$l_t = \arg\max_{l_t} \max_{l_s} A[l_s, l_t]$$

i.e., the single target layer that, when patched from *any* source layer, gives the maximum reasoning gain.

> [!TIP]
> This directly operationalizes Figure 5 from the paper. The heatmap concentration cluster(s) reveal $l_t$ empirically for Qwen2.5-1.5B, not by assumption.

### Metric 3: Logit-Lens Curvature → Confirms $l_s^\text{early}$ Transition

**Rationale**: The "logit lens" (nostalgebraist 2020) projects each layer's residual stream through the unembedding matrix to get an intermediate "prediction" distribution. The layer where this prediction **first stabilizes** on the correct entity-attribute answer marks where early knowledge crystallizes into an accessible form.

**Procedure**:

1. For each layer $l$, compute the unembedded logit distribution: $p^l = \text{softmax}(W_U h_E^l(P_\text{mem}))$
2. Compute the **layer-wise KL divergence** between consecutive layers: $d(l) = D_\text{KL}(p^l \| p^{l-1})$
3. The **early storage transition** is at the layer where $d(l)$ drops sharply and $p^l$ assigns high probability to $y^*$:

$$l_s^\text{early,confirm} = \arg\min_{l : p^l(y^*) > 0.5} l$$

**Usage**: Use this as a *cross-check* for $l_s^\text{early}$ from Metric 1. If both metrics agree within ±2 layers, proceed. If they disagree, use the self-patching map (Metric 2) as the tie-breaker.

### Layer Profiling Output

The profiling phase produces a `layer_profile.json`:

```json
{
  "model": "Qwen2.5-1.5B",
  "L": 28,
  "l_s_early": <int>,
  "l_s_late": <int>,
  "l_t": <int>,
  "probe_accuracy_per_layer": [...],
  "logit_lens_kl_per_layer": [...],
  "self_patching_heatmap": [[...]]
}
```

This file is loaded by all downstream training runs, making $l_s^\text{early}$, $l_s^\text{late}$, $l_t$ empirically grounded and reproducible.

---

## Part 2: Mathematical Formulation of Alignment Losses

### Notation

| Symbol | Meaning |
|---|---|
| $M$ | Transformer with $L$ layers |
| $h_t^l(P)$ | Hidden state at layer $l$, token position $t$, for prompt $P$ |
| $E$ | Entity anchor token span (entity name tokens in the prompt) |
| $h_E^l(P)$ | Mean-pooled hidden state over $E$ at layer $l$: $\frac{1}{\|E\|}\sum_{t \in E} h_t^l(P)$ |
| $P_\text{mem}$ | Memorization prompt (direct recall QA) |
| $P_\text{gen}$ | Generalization prompt (multi-hop reasoning QA) |
| $l_s^\text{early}, l_s^\text{late}$ | Source layers (empirically determined by profiling) |
| $l_t$ | Target layer (empirically determined by profiling) |
| $\mathcal{L}_\text{SFT}$ | Standard next-token prediction cross-entropy loss |
| $\lambda$ | Alignment loss weight (hyperparameter) |
| $K$ | Warmup epochs before alignment loss activates |

### Overall Training Objective

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{SFT}(P_\text{mem}) + \mathbb{1}[\text{epoch} \geq K] \cdot \lambda \cdot \mathcal{L}_\text{align}$$

The SFT loss is applied to $P_\text{mem}$ (memorization prompts). The alignment loss bridges the gap between memorization ($P_\text{mem}$) and reasoning ($P_\text{gen}$).

**Dual forward pass**: Within each batch, each fact contributes two forward passes:
1. $P_\text{mem}$ under `torch.no_grad()` → caches source representations (teacher)
2. $P_\text{gen}$ with gradient → receives both SFT loss and alignment loss

---

### Loss Variant 1: Representation Distillation (RepDist)

**Intuition**: Directly operationalize self-patching as a differentiable loss. Force the mid-layer entity representation during $P_\text{gen}$ to match the source-layer entity representation from $P_\text{mem}$.

$$\mathcal{L}_\text{RepDist} = \frac{1}{2} \sum_{l_s \in \{l_s^\text{early}, l_s^\text{late}\}} \left(1 - \cos\!\left(h_E^{l_t}(P_\text{gen}),\; \text{sg}\!\left[h_E^{l_s}(P_\text{mem})\right]\right)\right)$$

where $\text{sg}[\cdot]$ is the stop-gradient operator (the teacher is detached), and $\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$.

> [!NOTE]
> We use **cosine distance** rather than MSE because residual stream hidden states at different layers have different norms (due to layer-norm scaling). Cosine similarity focuses on directional alignment only, which is the geometrically meaningful quantity — PoLM Part 3.1 shows knowledge is stored as a *direction* in the entity embedding space.

---

### Loss Variant 2: Probing Loss (ProbeLoss)

**Intuition**: PoLM Part 3.1 shows knowledge is only usable if it is *linearly decodable* from entity-position embeddings. We train a frozen linear probe $\phi$ to decode the fact from the late-layer entity representation, then use it as a fixed "knowledge detector" at the mid-layer, backpropagating the decoding error through the model.

**Step 1 — Pre-train the probe** (done once before main training, on the warmup checkpoint):

$$\phi^* = \arg\min_\phi \mathbb{E}_{P_\text{mem}}\!\left[\text{CE}\!\left(\phi\!\left(h_E^{l_s^\text{late}}(P_\text{mem})\right),\; y^*\right)\right]$$

$\phi$ is a linear layer $\mathbb{R}^d \to \mathbb{R}^{|V|}$ mapping the entity embedding to a distribution over answer tokens. Trained for 10 epochs on the memorization set, then frozen.

**Step 2 — Apply at mid-layer during main training**:

$$\mathcal{L}_\text{Probe} = \text{CE}\!\left(\phi^*\!\left(h_E^{l_t}(P_\text{gen})\right),\; y^*\right)$$

Gradient flows through $h_E^{l_t}(P_\text{gen})$ but **not** through the frozen $\phi^*$. This pushes $h_E^{l_t}$ into the subspace where the fact is linearly decodable — the precise condition PoLM shows is necessary for extractability.

---

### Loss Variant 3: Contrastive Routing (ContraRoute)

**Intuition**: InfoNCE-style loss that pulls the mid-layer generalization representation toward the source-layer representation of the *same* fact, while pushing it away from other facts in the batch. Operates on the geometry of the representation space rather than requiring a decoder.

$$\mathcal{L}_\text{Contra} = -\frac{1}{2}\sum_{l_s \in \{l_s^\text{early}, l_s^\text{late}\}} \log \frac{\exp\!\left(\text{sim}(q, k^+_{l_s}) / \tau\right)}{\exp\!\left(\text{sim}(q, k^+_{l_s}) / \tau\right) + \sum_{j \neq i} \exp\!\left(\text{sim}(q, k^-_{j,l_s}) / \tau\right)}$$

where for fact $i$:
- $q = h_E^{l_t}(P_\text{gen}^{(i)})$ — the mid-layer reasoning-prompt representation (query)
- $k^+_{l_s} = \text{sg}\!\left[h_E^{l_s}(P_\text{mem}^{(i)})\right]$ — source-layer representation of the **same** fact (positive key, detached)
- $k^-_{j,l_s} = \text{sg}\!\left[h_E^{l_s}(P_\text{mem}^{(j)})\right]$ for $j \neq i$ — source-layer representations of **other** facts in the batch (negative keys, detached)
- $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$ — cosine similarity
- $\tau = 0.07$ — temperature

> [!NOTE]
> Gradients only flow through $q$ (the $P_\text{gen}$ mid-layer representation). All keys are stop-gradient. In-batch negatives are free to compute, requiring no extra forward passes. Batch size should be ≥8 to provide sufficient negatives.

---

### Loss Variant 4: Hybrid (RepDist + ProbeLoss)

$$\mathcal{L}_\text{Hybrid} = \alpha \cdot \mathcal{L}_\text{RepDist} + (1 - \alpha) \cdot \mathcal{L}_\text{Probe}$$

where $\alpha = 0.5$ initially (ablated in Phase 7). The RepDist term enforces directional alignment of representations; the ProbeLoss term enforces linear decodability. These are complementary: a representation can be directionally aligned but not linearly decodable if the probe's weight matrix doesn't match the aligned direction.

---

## Part 3: Experiment Matrix

### Conditions (Single Seed Exploration Phase)

| Condition | Training Method | Loss | Datasets |
|---|---|---|---|
| **Baseline-LoRA** | LoRA (r=16, q/v/o) | $\mathcal{L}_\text{SFT}$ only | STaRK-Prime, STaRK-MAG |
| **Baseline-FFT** | Full fine-tuning | $\mathcal{L}_\text{SFT}$ only | STaRK-Prime, STaRK-MAG |
| **RepDist-LoRA** | LoRA | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{RepDist}$ | STaRK-Prime, STaRK-MAG |
| **RepDist-FFT** | FFT | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{RepDist}$ | STaRK-Prime, STaRK-MAG |
| **ProbeLoss-LoRA** | LoRA | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Probe}$ | STaRK-Prime, STaRK-MAG |
| **ProbeLoss-FFT** | FFT | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Probe}$ | STaRK-Prime, STaRK-MAG |
| **ContraRoute-LoRA** | LoRA | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Contra}$ | STaRK-Prime, STaRK-MAG |
| **ContraRoute-FFT** | FFT | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Contra}$ | STaRK-Prime, STaRK-MAG |
| **Hybrid-LoRA** | LoRA | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Hybrid}$ | STaRK-Prime, STaRK-MAG |
| **Hybrid-FFT** | FFT | $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Hybrid}$ | STaRK-Prime, STaRK-MAG |

**Total exploration runs**: 10 conditions × 2 datasets = **20 runs** (1 seed each)

### Validation Phase

Pick the top-performing loss variant across both datasets. Re-run that variant + baselines with **3 seeds**. Report mean ± std with 95% Wilson confidence intervals (matching the original paper's methodology).

---

## Part 4: Evaluation Metrics

| Metric | Formula | Proves |
|---|---|---|
| **$A_\text{mem}$** | Exact-match on memorization prompts | Sanity check — must be $\geq 0.99$ for all methods |
| **$A_\text{gen}$(chaining)** | Exact-match on 2-hop chaining QA | Higher ceiling claim |
| **$A_\text{gen}$(intersection)** | MRR on intersection QA | Generality across task types |
| **$\Delta T$** | $T_\text{gen} - T_\text{mem}$ where $T = $ epoch of saturation | Faster convergence claim |
| **$\Delta A$** | $A_\text{mem}(T_\text{max}) - A_\text{gen}(T_\text{max})$ | Ceiling gap reduction |
| **Residual headroom** | Oracle self-patching gain on aligned checkpoint | Should be $\ll$ baseline headroom |
| **Training curves** | Per-epoch $A_\text{mem}$, $A_\text{gen}$ | Visual Pareto improvement |
| **Cost overhead** | Wall-clock seconds/epoch, peak GPU memory | Practical feasibility |

### Saturation Time Definition (matching the paper, eq. 2)

$$T_\text{gen} = \min\{t : A_\text{gen}(t') \geq \theta_\text{sat},\ \forall t' \in [t, t+w]\}$$

where $\theta_\text{sat}$ is method-specific (90% of the method's own final accuracy) and $w = 3$ epochs stability window.

### Success Criteria

> [!IMPORTANT]
> The experiment succeeds if, for at least one alignment loss variant:
> 1. $\Delta T_\text{ours} < \Delta T_\text{baseline}$ — faster convergence, **AND**
> 2. $A_\text{gen,ours}(T_\text{max}) > A_\text{gen,baseline}(T_\text{max})$ — higher ceiling, **AND**
> 3. $A_\text{mem,ours} \geq 0.98$ — memorization not degraded
>
> with statistical significance: 95% Wilson CIs non-overlapping between our method and baseline.

---

## Part 5: Datasets

### STaRK-Prime (Biomedical KG)
- **Download**: https://stark.stanford.edu/dataset_prime.html
- Biomedical knowledge graph with protein–drug–disease relations
- We use the memorization-to-generalization QA pairs built from it (Appendix A of arXiv:2607.08393): 1000 facts per run, with chaining and intersection generalization tasks
- Entity anchors are the head entity names (known from the KG)

### STaRK-MAG (Academic KG)
- **Download**: https://stark.stanford.edu/dataset_mag.html
- Academic Microsoft Academic Graph with author–paper–venue relations
- Same QA construction pipeline as STaRK-Prime
- Cross-domain validation that results generalize beyond biomedical

---

## Part 6: Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | Qwen2.5-1.5B | Paper's primary model; 28 layers |
| **$l_s^\text{early}, l_s^\text{late}, l_t$** | **Empirically determined by Phase 1.5 profiling** | Data-driven; not arbitrary |
| LoRA rank | 16 | Moderate; matches common practice |
| LoRA target modules | q_proj, v_proj, o_proj | Extended to output projection |
| Learning rate (LoRA) | 2e-4 | Standard for LoRA SFT |
| Learning rate (FFT) | 2e-5 | Standard for full fine-tuning |
| Batch size | 8 | Fits in single GPU with dual forward pass |
| Max epochs | 50 | Paper shows convergence by ~15-20 epochs; 50 gives margin |
| $\lambda$ | 0.1 (ablation: {0.01, 0.1, 0.5, 1.0}) | Start moderate; ablate |
| $K$ (warmup epochs) | 3 (ablation: {0, 1, 3, 5}) | Memorization saturates at epoch ~2-3 |
| $\tau$ (ContraRoute temperature) | 0.07 | Standard InfoNCE default |
| $\alpha$ (Hybrid balance) | 0.5 (ablation: {0.3, 0.5, 0.7}) | Equal weighting initially |
| Optimizer | AdamW | Standard |
| Weight decay | 0.01 | Standard |
| LR schedule | Cosine with linear warmup (100 steps) | Standard |
| Eval frequency | Every epoch | Match paper's per-epoch tracking |
| Saturation window $w$ | 3 | Match paper's definition (eq. 2) |

---

## Part 7: Code Structure

```
faster-sft/
├── related_works/                       # Literature (committed)
├── src/
│   ├── config.py                        # Dataclass configs for all conditions
│   ├── data/
│   │   ├── stark_dataset.py             # STaRK-Prime and STaRK-MAG loading/processing
│   │   └── paired_dataloader.py         # Yields (P_mem, P_gen, entity_span, y*) tuples
│   ├── profiling/
│   │   ├── linear_probe.py              # Per-layer probe accuracy: identifies l_s_early, l_s_late
│   │   ├── logit_lens.py                # Logit-lens KL curvature: confirms l_s_early
│   │   └── self_patch_scan.py           # Self-patching gain map: identifies l_t
│   ├── losses/
│   │   ├── base.py                      # Abstract AlignmentLoss interface
│   │   ├── rep_distill.py               # Variant 1: RepDist (cosine distillation)
│   │   ├── probe_loss.py                # Variant 2: ProbeLoss (frozen linear probe)
│   │   ├── contrastive.py               # Variant 3: ContraRoute (InfoNCE)
│   │   └── hybrid.py                    # Variant 4: Hybrid = α·RepDist + (1-α)·Probe
│   ├── models/
│   │   ├── hooks.py                     # Forward hooks for intermediate hidden states
│   │   └── model_utils.py               # Model loading, LoRA application
│   ├── training/
│   │   ├── trainer.py                   # Dual forward pass, warmup schedule, combined loss
│   │   └── callbacks.py                 # Per-epoch eval, checkpointing, wandb logging
│   └── evaluation/
│       ├── metrics.py                   # A_mem, A_gen, ΔT, ΔA, Wilson CIs
│       ├── self_patching.py             # Oracle self-patching for residual headroom
│       └── evaluator.py                 # Full eval pipeline
├── scripts/
│   ├── prepare_data.py                  # Download STaRK, build QA pairs
│   ├── run_profiling.py                 # Phase 1.5: produce layer_profile.json
│   ├── pretrain_probe.py                # Pre-train linear probe for ProbeLoss
│   ├── run_experiment.py                # Main experiment launcher
│   ├── run_self_patching_eval.py        # Post-training residual headroom measurement
│   └── plot_results.py                  # Training curves, comparison tables
├── configs/
│   ├── baseline_{lora,fft}.yaml
│   ├── repdist_{lora,fft}.yaml
│   ├── probe_{lora,fft}.yaml
│   ├── contra_{lora,fft}.yaml
│   └── hybrid_{lora,fft}.yaml
├── requirements.txt
└── README.md
```

---

## Part 8: Implementation Phases

### Phase 0: Environment Setup
- [ ] Set up Python environment: PyTorch ≥2.1, Transformers, PEFT, datasets, wandb
- [ ] Verify Qwen2.5-1.5B loads and runs inference (forward pass, greedy decode)
- [ ] Confirm forward hook extraction works for any layer index

### Phase 1: Data Preparation
- [ ] Download STaRK-Prime from https://stark.stanford.edu/dataset_prime.html
- [ ] Download STaRK-MAG from https://stark.stanford.edu/dataset_mag.html
- [ ] Build memorization-to-generalization QA pairs following arXiv:2607.08393 Appendix A pipeline (chaining + intersection tasks, 1000 facts each)
- [ ] Implement entity span finder: tokenize prompts, locate entity name tokens via exact-match with the KG entity string; handle multi-token BPE spans
- [ ] Implement `paired_dataloader.py`: each batch item is a (P_mem, P_gen, entity_span_mem, entity_span_gen, y*) tuple
- [ ] Validate: spot-check 50 pairs, confirm entity spans are correctly identified

### Phase 1.5: Layer Profiling
- [ ] Implement `linear_probe.py`: train per-layer logistic regression on 160 facts, evaluate on 40
- [ ] Implement `logit_lens.py`: project each layer's residual stream through unembedding matrix, compute KL divergence between consecutive layers
- [ ] Implement `self_patch_scan.py`: full L×L scan on 100 validation facts (early SFT checkpoint after K=3 epochs)
- [ ] Run all three metrics on Qwen2.5-1.5B
- [ ] Plot `ProbeAcc(l)`, `KL(l)`, and the self-patching heatmap $A[l_s, l_t]$
- [ ] Apply selection rules to determine $l_s^\text{early}$, $l_s^\text{late}$, $l_t$; cross-validate with logit-lens confirmation
- [ ] Save `layer_profile.json`; all downstream runs load this file

### Phase 2: Core Training Infrastructure
- [ ] Implement `hooks.py`: register `nn.Module.register_forward_hook` at layers $l_s^\text{early}$, $l_s^\text{late}$, $l_t$; accumulate into a `RepresentationCache`
- [ ] Implement `trainer.py` dual forward pass:
  1. Forward $P_\text{mem}$ with `torch.no_grad()` → populate cache for source layers
  2. Forward $P_\text{gen}$ with gradients → populate cache for target layer; compute $\mathcal{L}_\text{SFT}$
  3. Compute $\mathcal{L}_\text{align}$ from cached representations
  4. Combine: `loss = L_sft + (λ if epoch >= K else 0) * L_align`
  5. Backward, optimizer step
- [ ] Implement `callbacks.py`: per-epoch eval, checkpoint at epochs {1, 3, 5, 10, 15, 20, 30, 50}

### Phase 3: Loss Implementations
- [ ] `rep_distill.py`: cosine loss from $h_E^{l_t}(P_\text{gen})$ toward detached $h_E^{l_s}(P_\text{mem})$; average over both source layers
- [ ] `pretrain_probe.py` + `linear_probe.py`: train $\phi^*$ on late-layer entity representations for 10 epochs; freeze
- [ ] `probe_loss.py`: CE loss of $\phi^*(h_E^{l_t}(P_\text{gen}))$ vs. $y^*$; confirm gradient flows through $h_E^{l_t}$ only
- [ ] `contrastive.py`: InfoNCE with in-batch negatives; confirm $\tau=0.07$ default, log effective batch-negative count
- [ ] `hybrid.py`: weighted sum with $\alpha$ parameter; confirm both constituent gradients are present
- [ ] Unit tests: (a) gradient only flows through $P_\text{gen}$ branch, (b) loss is 0 when representations already match (RepDist), (c) loss equals CE of perfect probe (ProbeLoss)

### Phase 4: Evaluation Infrastructure
- [ ] `metrics.py`: exact-match $A_\text{mem}$, $A_\text{gen}$; saturation time $T_\text{gen}$ per eq. 2 with $w=3$; $\Delta T$, $\Delta A$; 95% Wilson CIs
- [ ] `self_patching.py`: post-training oracle scan over all $L \times L$ layer pairs on 1000 facts; report max gain (oracle headroom)
- [ ] `evaluator.py`: run full eval from a checkpoint path; output metrics dict + per-epoch CSV

### Phase 5: Baseline Runs
- [ ] Run Baseline-LoRA on STaRK-Prime and STaRK-MAG (2 runs)
- [ ] Run Baseline-FFT on STaRK-Prime and STaRK-MAG (2 runs)
- [ ] Verify against paper's Table 3: expect $A_\text{mem} \approx 0.998$, $A_\text{gen}^\text{chain} \approx 0.078$, $A_\text{gen}^\text{int} \approx 0.793$ for Qwen2.5-1.5B LoRA on STaRK-Prime (within ±0.02 tolerance)
- [ ] Run oracle self-patching on baseline checkpoints to establish upper-bound headroom

### Phase 6: Alignment Loss Exploration (1 seed)
- [ ] Run all 4 alignment losses × 2 training methods × 2 datasets = **16 runs**
- [ ] Log per-epoch: $A_\text{mem}$, $A_\text{gen}^\text{chain}$, $A_\text{gen}^\text{int}$, $\mathcal{L}_\text{SFT}$, $\mathcal{L}_\text{align}$, GPU memory
- [ ] Rank conditions by: primary = $A_\text{gen}^\text{chain}$ at epoch 50; secondary = $\Delta T$ reduction

### Phase 7: Validation and Ablations
- [ ] Re-run winning loss variant + baselines with **3 seeds** on both datasets
- [ ] Compute 95% Wilson CIs; confirm statistical significance
- [ ] Run oracle self-patching on all alignment-trained checkpoints; confirm reduced residual headroom
- [ ] $\lambda$ ablation: {0.01, 0.1, 0.5, 1.0} for winning loss (4 additional runs)
- [ ] $K$ ablation: {0, 1, 3, 5} warmup epochs for winning loss (4 additional runs)
- [ ] Layer index sensitivity: re-run winning loss with $l_t \pm 2$ layers to confirm profile-selected $l_t$ is indeed optimal

### Phase 8: Analysis and Reporting
- [ ] Training curve plots: $A_\text{mem}$, $A_\text{gen}^\text{chain}$ vs. epoch for all 10 conditions side-by-side
- [ ] Tables: match paper's format — Table 3 (ΔT comparison), Table 4 (oracle vs. no-patch vs. our method), Table 7 (heuristic vs. oracle)
- [ ] Layer profiling figures: ProbeAcc(l), KL(l) curves, and self-patching heatmap at warmup checkpoint
- [ ] Cost table: wall-clock time/epoch, peak GPU memory, additional compute overhead vs. baseline
- [ ] Write summary of which loss variant won and why

---

## Verification Plan

### Automated Tests
```bash
# Unit tests: gradient flow, output shapes, loss correctness
python -m pytest tests/test_losses.py -v

# Integration: single training step with dual forward pass
python -m pytest tests/test_trainer.py -v

# Data: verify paired prompts share facts, entity spans correct
python -m pytest tests/test_data.py -v

# Profiling: verify probe, logit-lens, self-patch scan produce valid outputs
python -m pytest tests/test_profiling.py -v
```

### Baseline Reproduction Gate
Before proceeding to Phase 6, confirm: `Baseline-LoRA on STaRK-Prime → A_mem ≥ 0.978, A_gen_chain ≤ 0.098` (within ±0.02 of paper's 0.998 and 0.078). If outside this range, debug before running expensive conditions.

### Open Questions (Resolved)

| Question | Resolution |
|---|---|
| Layer index selection | Empirical profiling (Phase 1.5) via probe accuracy, logit-lens, and self-patching scan |
| STaRK data access | Direct download from stark.stanford.edu/dataset_{prime,mag}.html |
| Synthetic biography dataset | **Dropped** — out of scope for this experiment |
| Entity span identification | Exact-match tokenizer lookup; known entity names from STaRK KG |
| GPU memory with dual forward pass | `no_grad` on P_mem pass eliminates activation storage; only P_gen activations held in memory |
