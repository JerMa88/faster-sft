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

## Repository Structure

The repository is organized to separate model execution logic, scripts, and SLURM jobs cleanly:

- `src/`: Core Python modules for data loading, loss functions (Baseline, RepDist, Probe, Contrastive, Hybrid), and evaluation metrics.
- `scripts/`: Python utility scripts broken down by domain.
  - `data_prep/`: Scripts to prepare and format the STaRK datasets.
  - `training/`: Core SFT training loop, probing, and profiling scripts.
  - `evaluation/`: Scripts to run multi-token generative evaluation.
  - `analysis/`: Scripts for parsing logs, computing statistics, and aggregating results.
- `slurm_jobs/`: SLURM bash scripts used to launch multi-GPU training and evaluation on the cluster.
  - `archive/`: Deprecated V1 scripts.
- `outputs/`: Model checkpoints, logs, and evaluation results.
- `hf_cache/`: Local cache directory for HuggingFace models.

## How to Run the Experiments

This guide explains how to run the end-to-end pipeline, from data preparation to evaluation.

### 1. Data Preparation
To prepare the dataset and synthetic knowledge bases:
```bash
python scripts/data_prep/prepare_data.py
python scripts/data_prep/prepare_v2_data.py
```

### 2. Run Baseline SFT
Run the baseline supervised fine-tuning using the `run_sft.sh` SLURM script. This script handles data prep (if missing), profiling, probe pre-training, and baseline SFT training automatically.
```bash
sbatch slurm_jobs/run_sft.sh <model_key>
# Example: sbatch slurm_jobs/run_sft.sh gemma4-e4b
```
*(Valid models: `llama3.2-3b`, `qwen3.5-2b`, `gemma4-e4b`, `antares-1b`, `nanbeige4.2-3b`, `lfm2.5-1.2b`)*

### 3. Run Alignment Sweep (V2)
Once the baseline SFT is complete, run the alignment sweep (which trains the model using the Hybrid, Contrastive, Probe, and RepDist auxiliary losses):
```bash
sbatch slurm_jobs/run_alignment_sweep.sh <model_key>
```
*Note: This sweep skips baseline logic and strictly trains the alignment variants using $\lambda=0.1$. Results are saved in `outputs/runs_v2`.*

### 4. Evaluation
Evaluate the V2 alignment sweep runs using the parallel evaluator. This job will spin up parallel workers to speed up the multi-token generation evaluation.
```bash
sbatch slurm_jobs/run_eval_v2_parallel.sh <model_key>
```
The results will be written to `eval_results.json` inside each model's checkpoint directory.

### 5. Consolidate Results
Once evaluations are completed, aggregate the results into a markdown table:
```bash
python scripts/analysis/consolidate_v2_results.py
```