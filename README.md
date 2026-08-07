# Alignment-Aware Supervised Fine-Tuning (AA-SFT)

This repository contains the codebase for investigating and mitigating the **Knowing-Using Gap (KUG)** in Large Language Models during Supervised Fine-Tuning (SFT). We show that adding an **alignment-aware auxiliary loss** during SFT produces a consistent Pareto improvement over standard SFT: higher generalisation accuracy ($A_\text{gen}$) with no loss in memorisation accuracy ($A_\text{mem}$).

**Results:** AA-SFT beats the SFT baseline on **6/7 model-dataset pairs** (relaxed exact match). Best configurations achieve $A_\text{gen} = 0.271$ (Qwen3.5-2B, Probe/Hybrid, STaRK-PRIME) and $A_\text{gen} = 0.261$ (Llama3.2-3B, Probe, STaRK-PRIME), closing ~32–35% of the oracle headroom at **zero inference cost**.

---

## Methodology

Standard SFT injects facts into early-layer MLP neurons (high $A_\text{mem}$) but these facts fail to route through mid-layer reasoning circuits (near-zero $A_\text{gen}$) — the **Knowing-Using Gap**. We fix this with training-time auxiliary losses that align the hidden-state representation of the same fact across two differently-framed prompts: a memorisation prompt $P_\text{mem}$ and a generalisation prompt $P_\text{gen}$.

### AA-SFT Loss Variants ($\lambda=0.1$, warmup $K=3$ epochs)

| Variant | Loss | Description |
|---|---|---|
| **Baseline** | $\mathcal{L}_\text{SFT}$ | Standard cross-entropy SFT, no alignment |
| **RepDist** | $\mathcal{L}_\text{SFT} + \lambda \mathcal{L}_\text{RepDist}$ | Cosine distance between storage and reasoning representations |
| **ContraRoute** | $\mathcal{L}_\text{SFT} + \lambda \mathcal{L}_\text{Contra}$ | InfoNCE contrastive loss with in-batch negatives |
| **Probe** | $\mathcal{L}_\text{SFT} + \lambda \mathcal{L}_\text{Probe}$ | Frozen linear probe enforces decodability at reasoning layer |
| **Hybrid** | $\mathcal{L}_\text{SFT} + \lambda \mathcal{L}_\text{Hybrid}$ | 0.5×Probe + 0.5×ContraRoute |

All runs: LoRA ($r=16$, $\alpha=32$), AdamW ($\eta=2\times10^{-4}$), 50 epochs, seed 42, NVIDIA A100 (80 GB) via SLURM.

### Evaluation

**Primary metric: Relaxed Exact Match (Relaxed EM)** — gold answer is a case-insensitive substring of the model output. Strict EM is effectively zero across all models due to instruction-following verbosity.

---

## Repository Structure

```
faster-sft/
├── src/                         # Core Python modules
│   ├── data/paired_dataloader.py  # (P_mem, P_gen) pair loader + entity span tracking
│   ├── models/hooks.py            # Representation extraction hooks
│   └── training/losses.py         # RepDist, ContraRoute, Probe, Hybrid loss implementations
├── scripts/
│   ├── data_prep/                 # STaRK data preparation scripts
│   ├── training/                  # train_sft.py, pretrain_probe.py, run_profiling.py
│   ├── evaluation/                # evaluate_all_v2.py, evaluate_v1_runs.py
│   └── analysis/                  # generate_paper_figures.py, consolidate results
├── slurm_jobs/                    # SLURM launch scripts for training and evaluation
├── paper/                         # ACL paper (main.tex, refs.bib, figures/)
├── data/processed/                # STaRK-PRIME and STaRK-MAG QA datasets, layer profiles
└── outputs/
    ├── runs/                      # AA-SFT training checkpoints + eval_results.json
    └── logs/                      # SLURM job logs
```

---

## How to Run the Experiments

### 1. Data Preparation

```bash
python scripts/data_prep/prepare_data.py
```
Produces `data/processed/stark_prime_qa.jsonl` and `data/processed/stark_mag_qa.jsonl` (1,000 pairs each).

### 2. Layer Profiling + Probe Pre-training

```bash
# Per-model (run once per model_key before training)
python scripts/training/run_profiling.py --model_key qwen3.5-2b
python scripts/training/pretrain_probe.py --model_key qwen3.5-2b
```
Outputs: `data/processed/layer_profile_<model>.json` and `data/processed/probe_phi_<model>.pt`

### 3. Run Baseline SFT

```bash
sbatch slurm_jobs/run_sft.sh <model_key>
# e.g.: sbatch slurm_jobs/run_sft.sh llama3.2-3b
```
Valid model keys: `llama3.2-3b`, `qwen3.5-2b`, `gemma4-e4b`, `lfm2.5-1.2b`

### 4. Run AA-SFT Alignment Sweep

All 4 alignment variants (RepDist, ContraRoute, Probe, Hybrid) for both STaRK-PRIME and STaRK-MAG:

```bash
sbatch slurm_jobs/run_alignment_sweep.sh <model_key>
```
Results saved to `outputs/runs/<model_key>/stark_{prime,mag}/`.

> **Note:** Antares-1B and Nanbeige4.2-3B are excluded — CUDA kernel deadlocks during inference (Flash Attention / loop-attention incompatibility with A100).

### 5. Evaluate

```bash
python scripts/evaluation/evaluate_v1_runs.py --model_key llama3.2-3b
```
Results written to `eval_results.json` inside each checkpoint directory. Uses **Relaxed EM** as primary metric.

### 6. Generate Figures

```bash
python scripts/analysis/generate_paper_figures.py
```
Outputs all 5 paper figures to `paper/figures/`.

---

## Key Results (Relaxed EM, peak $A_\text{gen}$)

| Model | Dataset | SFT Baseline | Best AA-SFT | Best Loss | Δ |
|---|---|---|---|---|---|
| Llama3.2-3B | PRIME | 0.254 | **0.261** | Probe | +2.8% |
| Llama3.2-3B | MAG   | 0.086 | **0.104** | Hybrid | +21% |
| Qwen3.5-2B  | PRIME | 0.244 | **0.271** | Probe/Hybrid | +11% |
| Qwen3.5-2B  | MAG   | 0.019 | 0.019 | — | 0% (tie) |
| LFM2.5-1.2B | PRIME | 0.068 | **0.069** | Hybrid | +1.5% |
| LFM2.5-1.2B | MAG   | 0.013 | **0.014** | Probe/Hybrid | +8% |
| Gemma4-E4B  | MAG   | 0.009 | **0.011** | ContraRoute | +22% |

**6/7 pairs improved.** Loss function choice is a second-order factor; dataset structure is the primary determinant of KUG magnitude.

---

## Paper

See [`paper/`](paper/) for the full ACL 2026 draft: *"Routing Facts to Reasoning Circuits: Alignment-Aware SFT Closes the Knowing-Using Gap"*.
