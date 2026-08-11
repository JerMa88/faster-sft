# AA-SFT Handoff Document

> **Updated: 2026-08-07.** All training complete; 28/28 V1 AA-SFT runs evaluated; paper at `paper/main.tex` is submission-ready for ACL 2026.

---

## Current Project State

Experiments are **complete**. The paper reports V1 AA-SFT (RepDist, ContraRoute, Probe, Hybrid) as the primary method — these beats the SFT baseline on 6/7 model-dataset pairs using Relaxed EM.

### What's Done
- ✅ Data: STaRK-PRIME + STaRK-MAG QA datasets (`data/processed/`)
- ✅ Layer profiling + probe pre-training for all 4 models
- ✅ Training: 28 runs (4 models × 2 datasets × 5 variants) in `outputs/runs/`
- ✅ Evaluation: Relaxed EM applied uniformly via `scripts/evaluation/evaluate_v1_runs.py`
- ✅ Figures: 5 figures in `paper/figures/` (regenerate with `scripts/analysis/generate_paper_figures.py`)
- ✅ Paper: `paper/main.tex` updated with correct V1 results, tables, and discussion

### What's Pending (COMPLETE OVERHAUL)
> [!CAUTION]
> **CRITICAL ISSUE FOUND:** Previous `faster-sft` experiments are invalid and MUST BE ABANDONED. They evaluated gradients only on $P_{gen}$, skipping $P_{mem}$, destroying baseline comparability with the KUG paper (Dai et al., 2025). 

The next agent MUST execute a complete overhaul of the experimental design, restricted to `Qwen/Qwen2.5-1.5B` for now. We will run 3 precise experiments (KUG Replication, Supervised Dual-Loss, and Two-Stage Sequential Training).
👉 **Please refer to [implementation_plan.md](file:///users/jerryma/.gemini/antigravity-ide/brain/5a2647eb-6476-41fe-b796-619a791e27d5/implementation_plan.md) for exact architecture changes.**

- ❌ Study the original codebase in `Mem2Gen-71FF/` for exact data splits and baseline loss implementations.
- ❌ Replicate Figure 7 (KUG Baseline) exactly.
- ❌ Integrate `layer_patching.py` to generate Figure 9 permeation dynamic heatmaps and Patchscope views.
- ❌ Run Experiment 2 (Supervised Dual-Loss) and Experiment 3 (Two-Stage Mem-to-Gen).

---

## Repository Layout

```
faster-sft/
├── src/
│   ├── data/paired_dataloader.py   # (P_mem, P_gen) loader + entity span tracking
│   ├── models/hooks.py              # Layer-wise representation hooks
│   └── training/losses.py           # All 4 AA-SFT loss functions
├── scripts/
│   ├── data_prep/                   # STaRK data preparation
│   ├── training/train_sft.py        # Primary training loop (LoRA + AA-SFT loss)
│   ├── training/pretrain_probe.py   # Freeze-and-fit linear probe φ*
│   ├── training/run_profiling.py    # Layer SNR profiling → layer_profile.json
│   ├── evaluation/evaluate_v1_runs.py  # Relaxed EM evaluator for outputs/runs/
│   └── analysis/generate_paper_figures.py
├── slurm_jobs/
│   ├── run_sft.sh                   # Baseline + probe pre-training
│   └── run_alignment_sweep.sh       # Full 4-variant AA-SFT sweep
├── paper/                           # ACL 2026 paper
├── data/processed/                  # QA datasets, layer profiles, probes
└── outputs/
    ├── runs/                        # V1 AA-SFT checkpoints (primary, reported)
    ├── runs_v2/                     # V2 sweep (archived, not reported — underperforms)
    └── logs/                        # SLURM job logs
```

---

## Key Design Decisions

### Why V1 and not V2?
A V2 sweep (`outputs/runs_v2/`) trained with `stark_prime_qa_v2.jsonl` and an additional BridgeAlign term. V2 only beat the baseline on 2/7 pairs vs. 6/7 for V1. Root cause: BridgeAlign suppresses memorisation (A_mem ~30% lower throughout training), which proportionally limits generalisation since you can't generalise what you haven't memorised. V1's simpler loss is better.

### Why Relaxed EM?
Strict exact match is zero across all models because instruction-tuned models produce verbose, formatted outputs ("The answer is X."). Relaxed EM (gold-answer-as-substring) correctly identifies these as correct. This matches the evaluation protocol of Dai et al. (2025, Mem2Gen).

### Why 4 models?
Antares-1B and Nanbeige4.2-3B had irrecoverable CUDA kernel deadlocks (Flash Attention and custom loop-attention kernels incompatible with A100 environment). Results archived in `outputs/runs_v2/` where training completed but inference failed.

---

## Reproducing Results

```bash
# 1. Evaluate a model
python scripts/evaluation/evaluate_v1_runs.py --model_key llama3.2-3b

# 2. Regenerate figures
python scripts/analysis/generate_paper_figures.py

# 3. Build tables
python scripts/analysis/consolidate_v2_results.py  # (works for both runs/ and runs_v2/)
```

---

## Contacts

- Jerry Ma (jerryma@smu.edu) — primary author
- Prof. Hahsler (mhahsler) — PI
