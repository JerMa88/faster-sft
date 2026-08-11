# Complete Overhaul: KUG Replication and KUG Resolution Experiments

This plan outlines a complete overhaul of our experimental methodology. Previous experiments are being discarded because they computed Cross-Entropy loss exclusively on $P_{gen}$ and ignored $P_{mem}$, making them fundamentally incomparable to the original Knowing-Using Gap (KUG) paper (which exclusively supervised $P_{mem}$). 

We will constrain these new experiments solely to `Qwen/Qwen2.5-1.5B` to ensure rapid iteration while validating the exact KUG hypotheses and our proposed novel training regimes.

## User Review Required

> [!WARNING]
> **Methodology Overhaul:** We are completely abandoning the previous `faster-sft` loss implementations. We will rebuild the training loop from scratch to ensure strict dataset/split parity with `Mem2Gen-71FF` and Appendix B of the Dai et al. paper.

> [!IMPORTANT]
> **New Diagnostic Requirements:** All three experiments below must run epoch-by-epoch diagnostics using the original `layer_patching.py` code. We must output **Permeation Dynamics Heatmaps** (resembling Figure 9) and **Patchscope views** of memorization-related layers at the head-entity position to prove exactly where knowledge successfully routes (success cases) or fails to route (failure cases).

## Experimental Setup

### Data Pipeline & Task Separation
- Strictly adhere to the data splits described in Appendix B of the KUG paper.
- Extract and rigorously separate `chaining`, `intersection`, and `fact_checking` tasks.
- For all training runs, $A_{mem}$ and $A_{gen}$ will be tracked independently for each task type.

---

### Experiment 1: Strict KUG Baseline Replication
**Loss Function:** $\mathcal{L} = \mathcal{L}_{CE}(P_{mem})$
- **Method:** Fine-tune standard LoRA exclusively on the memorization prompts.
- **Goal:** Reproduce the exact KUG curves seen in Figure 7. We expect to see $A_{mem}$ hit ~1.0 quickly. Generalization should remain near zero for chaining and fact-checking (the KUG), but should naturally rise alongside memorization for the intersection task.

### Experiment 2: Supervised Dual-Loss + Alignment (Novel)
**Loss Function:** $\mathcal{L} = \mathcal{L}_{CE}(P_{mem}) + \lambda_{gen} \mathcal{L}_{CE}(P_{gen}) + \lambda_{align} \mathcal{L}_{align} + \lambda_{grok} \mathcal{L}_{grok}$
- **Method:** Supervise the network on both the memorization storage and the generalization task simultaneously, while guiding the internal representation routing using our Alignment and Grokking loss variants.
- **Goal:** Determine if supervising both endpoints—while applying structural alignment—solves the KUG without degrading the near-perfect $A_{mem}$ baseline. We will verify if this improves both $A_{mem}$ and $A_{gen}$ structurally via the permeation heatmaps.

### Experiment 3: Two-Stage Sequential Training (Novel)
**Loss Function:**
- Epoch 1 to 15: $\mathcal{L} = \mathcal{L}_{CE}(P_{mem})$ 
- Epoch 16 to 50: $\mathcal{L} = \mathcal{L}_{CE}(P_{gen})$
- **Method:** Freeze the task objective to $P_{mem}$ until Epoch 15 (which Experiment 1 should prove is when true memorization saturation occurs). Then, abruptly switch the training objective exclusively to $P_{gen}$.
- **Goal:** Observe if the model can successfully map the previously saturated facts to the reasoning task, or if it immediately suffers catastrophic forgetting of $A_{mem}$ the moment $P_{mem}$ is removed from the loss function. This directly tests the durability of the internal knowledge representations.

## Verification Plan

### Automated Tests
- Verify the exact data subsets extracted match the entity/relation counts from `Mem2Gen-71FF/data_generation`.
- Run a 2-epoch dry run of Experiment 1 with `layer_patching.py` active to ensure permeation heatmaps render correctly to disk without OOM errors.

### Manual Verification
- The user will visually inspect the Figure 7 replication curves from Experiment 1 before Experiment 2 and 3 commence, to ensure baseline comparability is rock-solid.
