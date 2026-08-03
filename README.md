# Alignment-Aware Supervised Fine-Tuning (faster-sft)

This repository contains the codebase for investigating and mitigating the **Knowing-Using Gap (KUG)** in Large Language Models during Supervised Fine-Tuning (SFT). Our goal is to prove that adding an **alignment-aware auxiliary loss** during SFT produces a Pareto improvement over standard SFT: both faster convergence (smaller temporal lag $\Delta T$) and a higher final ceiling (smaller accuracy gap $\Delta A$) on downstream multi-hop reasoning tasks.

## Methodology Overview

Our experimental approach is grounded in resolving the routing failures observed when models are updated with new facts. The KUG hypothesis predicts that standard SFT injects facts into early storage layers (high $A_{mem}$) but these facts fail to properly route through middle-layer reasoning circuits (low $A_{gen}$). As training progresses, $A_{mem}$ often peaks early and declines due to catastrophic forgetting, while $A_{gen}$ remains near zero.

To address this, we apply an **alignment-aware auxiliary loss** to actively bring storage representations and reasoning representations closer together during training. 

### Methods and Alignment Variants Experimented

We are experimenting with the following Loss functions (where $\lambda=0.1$):

1. **Standard SFT (Baseline-LoRA)**: $\mathcal{L}_\text{SFT}$
2. **RepDist-LoRA**: $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{RepDist}$
   - Minimizes the representational distance between early storage layers and later reasoning layers.
3. **ContraRoute-LoRA**: $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Contra}$
   - Applies contrastive learning to push correct reasoning pathways closer while pushing away incorrect or non-routed pathways.
4. **ProbeLoss-LoRA**: $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Probe}$
   - Uses linear probes on entity embeddings to enforce decodability at specific layer checkpoints.
5. **Hybrid-LoRA**: $\mathcal{L}_\text{SFT} + \lambda \cdot \mathcal{L}_\text{Hybrid}$
   - A combination of representation distance and contrastive routing methods.

### Key References and Theoretical Grounding

Our methodologies draw directly from recent literature on model editing and activation pathways:
- **Evaluation Methodology (arXiv:2607.08393)**: We rely on multi-token greedy generation and string-level exact match for evaluating $A_{mem}$ and $A_{gen}$.
- **Targeted Lexical Injection (arXiv:2506.15415)**: Validates our use of Centered Kernel Alignment (CKA) between layers to identify representation transition points.
- **ROME/MEMIT "Hopping-Too-Late" Problem (arXiv:2601.04600)**: Informs our Oracle Self-Patching Headroom Analysis to measure routing failures in middle layers.
- **ACE Sequential Activation Chain (arXiv:2510.07896)**: Serves as the theoretical basis for ensuring facts route correctly through early storage to late reasoning layers.

## Current Status

**Phase 1-5 (v1 Baseline and Alignment Sweep): COMPLETED**
- **Evaluation Complete**: We successfully ran the full 62-run matrix evaluation (eval_v3) across all 6 models and 2 datasets.
- **KUG Confirmed**: The Knowing-Using Gap was independently confirmed across 12/12 baseline runs.
- **Alignment Validated**: The v1 Hybrid alignment loss effectively doubled generalization and boosted AUC by over 75% on some models.

**Phase 6 (v2 Alignment Sweep): IN PROGRESS**
- Despite the successes of v1, absolute generalization still falls short of the Oracle ceiling (which is theoretically bounded at ~60%).
- Based on rigorous mathematical proofs of routing failures during our alignment process (detailed in `related_works_research.md`), we are currently running the **v2 Alignment Sweep** (`run_alignment_sweep.sh` saving to `outputs/runs_v2`) incorporating four architectural fixes:
  1. **BridgeAlign**: Dual-span chaining alignment to prevent hop-1 failures.
  2. **DynLayerAlign**: Dynamic layer targeting matching representation mobility.
  3. **KG-HardInfoNCE**: Relational hard-negative mining to force representation isotropy.
  4. **TopoPrefixAlign**: Graph schema attention prefixing to reduce entropy.
  
Jobs are currently executing on the SLURM cluster.