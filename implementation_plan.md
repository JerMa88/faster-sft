# Overhaul of KUG Experiments: Replication, 2-Stage SFT, and Joint Supervised SFT

This implementation plan outlines the complete experimental overhaul for `Qwen/Qwen2.5-1.5B` on the Knowing-Using Gap (KUG). Previous experiments were invalidated because they skipped $P_{mem}$ gradient updates; this overhaul ensures strict baseline parity with Dai et al. (2025, NeurIPS submission / `Mem2Gen-71FF`) while introducing two novel training regimes to address the KUG.

---

## Direct Answers to User Inquiries

### 1. Are the KUG Original Authors guaranteed NOT to have done these experiments?
**YES, 100% CERTAIN.**
- In the KUG paper (Dai et al., 2025), the authors **exclusively trained using standard SFT on $P_{mem}$** (memorization loss only). 
- They analyzed the resulting Knowing-Using Gap purely via **inference-time activation patching** (self-patching hidden states at inference without modifying training).
- The authors **did NOT** test multi-stage fine-tuning ($P_{mem} \to P_{gen}$ sequential training) nor **joint supervised loss** ($P_{mem} + P_{gen}$ simultaneous training).
- Testing whether fine-tuning dynamics themselves can bridge the KUG (without needing inference-time patching) is an unstudied, novel next step.

### 2. Data Preprocessing & Task Disaggregation Strategy
**YES, WE WILL RE-PREPROCESS AND RIGOROUSLY DISAGGREGATE THE DATA.**
- We will construct a dedicated data preprocessor (`prepare_kug_datasets.py`) that explicitly separates and balances:
  1. `chaining` tasks ($E_1 \to E_2 \to E_3$ reasoning requiring intermediate entity bridge recovery)
  2. `intersection` tasks (identifying a target entity satisfying multiple joint constraints)
  3. `fact_checking` tasks (statement verification as true/false/unknown)
- **Why this is critical:** `intersection` is an easier task that naturally generalizes under standard SFT. Testing `intersection` separately allows us to verify whether **Method 2 (2-Stage SFT)** and **Method 3 (Joint Supervised SFT)** preserve performance on easy tasks **without catastrophic forgetting**, while evaluating whether they successfully unlock generalization on hard tasks (`chaining` and `fact_checking`).
- **Baseline Filtering:** Pre-existing zero-shot correct facts on the base model will be filtered out so accuracy reflects true knowledge injection.

### 3. Decoupled Training & Evaluation Architecture with W&B Resume
**SEPARATED TRAINING AND EVALUATION FOR MAXIMUM EFFICIENCY:**
1. **Training Phase (Pure Optimization):**
   - Training runs without inline generation/eval for maximum GPU throughput.
   - Saves adapter weights at EVERY epoch (`checkpoint-epoch-1` through `checkpoint-epoch-50`).
   - Logs training loss ($L_{mem}$, $L_{gen}$, total loss, grokking loss) and weight/gradient norms ($\|W\|_2, \|\Delta W\|_2, \|\nabla W\|_2$) to a unique W&B run ID.
   - Writes `wandb_run_id` to `run_metadata.json` in the checkpoint directory.
2. **Evaluation Phase (Post-Training SLURM Job):**
   - Submitted as a separate GPU SLURM job after training completes.
   - Reads `wandb_run_id` from `run_metadata.json` and resumes the **SAME W&B RUN** using `wandb.init(id=wandb_run_id, resume="allow")`.
   - Iterates through `checkpoint-epoch-1` to `checkpoint-epoch-50` on GPU:
     - Evaluates $A_{mem}$ and $A_{gen}$ disaggregated by task type (`chaining`, `intersection`, `fact_checking`).
     - Computes Permeation Dynamics Heatmaps ($l_{src} \to l_{tgt}$ self-patching at $E_{head}$, matching Figure 9).
     - Computes Patchscope hidden state decodings at $E_{head}$ (matching Figure 8).
     - Logs epoch evaluation metrics, heatmaps, and Patchscope tables to the SAME W&B run.

---

## User Review Required & Safety Rules

> [!CAUTION]
> **STRICT CPU SAFETY DIRECTIVE:** DO NOT run PyTorch or safetensors model loading scripts on CPU or in local bash sessions! Doing so will exhaust CPU RAM and brick the host VM. ALL PyTorch model loading, training, dry-runs, and evaluations MUST be launched on GPU via SLURM (`sbatch`).

> [!IMPORTANT]
> **Task-by-Task Git Workflow:** For each coding task, we will write code, test non-model logic locally (or run model tests via GPU SLURM), debug completely, and make a Git commit with a detailed technical message before moving to the next task.

> [!WARNING]
> **SLURM Job Submission & VRAM Optimization:**
> - Submit ONE job at a time via SLURM `sbatch`.
> - Maximize GPU VRAM utilization ($\ge 70$ GB target on 80 GB A100/A800 GPUs).
> - After submitting a 2-epoch dry run, inspect logs and GPU VRAM usage. If VRAM is too low (< 70 GB), cancel the job (`scancel`), increase per-device batch size or sequence padding, and resubmit.

---

## Experimental Setup & 3 Training Regimes

### Hyperparameters (Exact KUG Paper Parity)
- **Base Model:** `Qwen/Qwen2.5-1.5B`
- **LoRA Config:** $r=16, \alpha=32$, dropout $0.05$, target modules: `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}`
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$, lr $2 \times 10^{-4}$)
- **Target VRAM Utilization:** $\ge 70$ GB (tuned via batch size and sequence padding)
- **Epochs:** 50 epochs per experiment
- **Datasets:** STaRK-PRIME & STaRK-MAG datasets (1,000 multi-fact split) with task breakdown: `chaining`, `intersection`, `fact_checking`.

---

### Training Methods

#### 1. Method 1: KUG Baseline Replication (Replicating Figure 7)
- **Loss:** $\mathcal{L} = \mathcal{L}_{CE}(P_{mem})$
- **Protocol:** Standard SFT on memorization prompts $P_{mem}$ only for 50 epochs.
- **Goal:** Reproduce the exact KUG curves from Figure 7 in Dai et al.: $A_{mem}$ saturates to ~1.0 around epoch 15, while $A_{gen}$ remains near zero for `chaining` and `fact_checking` while `intersection` $A_{gen}$ rises alongside $A_{mem}$.

#### 2. Method 2: 2-Stage Mem-then-Gen SFT (Sequential Training)
- **Loss:** 
  - **Epochs 1 to 15:** $\mathcal{L} = \mathcal{L}_{CE}(P_{mem})$ (Memorization Phase)
  - **Epochs 16 to 50:** $\mathcal{L} = \mathcal{L}_{CE}(P_{gen})$ (Generalization Phase)
- **Protocol:** Train on $P_{mem}$ until epoch 15 (where Method 1 proves memorization reaches saturation), then hard-switch the objective exclusively to $P_{gen}$.
- **Goal:** Test if $A_{mem}$ suffers **catastrophic collapse / forgetting** when $P_{mem}$ gradients are removed, or if $A_{gen}$ can build upon pre-memorized representations without destroying knowledge storage across easy (`intersection`) and hard (`chaining`, `fact_checking`) tasks.

#### 3. Method 3: Joint Supervised SFT (Simultaneous Supervision)
- **Loss:** $\mathcal{L} = \mathcal{L}_{CE}(P_{mem}) + \mathcal{L}_{CE}(P_{gen})$
- **Protocol:** Supervise both memorization and generalization tasks concurrently across all 50 epochs.
- **Goal:** Determine if simultaneous supervision on both endpoints resolves the KUG, outperforming both Method 1 (Baseline) and Method 2 (2-Stage) on $A_{mem}$ and $A_{gen}$ across all task categories.

---

## Detailed Task Breakdown & Execution Plan

### Task 1: Dataset Preprocessing & Disaggregation Script
- **File:** `scripts/data_prep/prepare_kug_datasets.py`
- **Actions:** Build dataset preprocessing script loading STaRK-PRIME and STaRK-MAG, formatting $P_{mem}, P_{gen}$, target entities $y^*$, bridge entities, and tagging tasks as `chaining`, `intersection`, `fact_checking`.
- **Testing:** Run data formatting tests locally (JSON schema validation & count checks, no PyTorch).
- **Git Commit:** `"feat(data_prep): add disaggregated dataset preprocessing script for STaRK-PRIME and STaRK-MAG"`

### Task 2: Paired Dataloader with Entity Span Indexing
- **File:** `src/data/paired_dataloader.py`
- **Actions:** Update dataloader to output tokenized $P_{mem}, P_{gen}$ inputs, attention masks, task category tags, and exact token position spans for head-entity $E_{head}$.
- **Testing:** Run tokenization & span index tests.
- **Git Commit:** `"feat(data): refactor PairedSTaRKDataset to support task-type tags and head-entity token span indexing"`

### Task 3: Decoupled Fast Training Loop with W&B & Weight Norm Logging
- **File:** `src/training/train_kug_overhaul.py`
- **Actions:** Construct unified trainer supporting `--method baseline`, `--method two_stage`, and `--method joint`. Log $L_{mem}, L_{gen}$, total loss, grokking loss, $\|W\|_2, \|\Delta W\|_2, \|\nabla W\|_2$ to W&B. Save adapter weights for EVERY epoch (`checkpoint-epoch-X`) and store `wandb_run_id` in `run_metadata.json`.
- **Git Commit:** `"feat(training): implement fast decoupled trainer for KUG baseline, 2-stage, and joint loss saving per-epoch adapter weights"`

### Task 4: Standalone Evaluation & Diagnostic Suite with W&B Resume
- **File:** `src/analysis/kug_eval_diagnostics.py`
- **Actions:** Standalone script that loads `run_metadata.json`, resumes the **SAME W&B run ID**, iterates through per-epoch adapter checkpoints, evaluates $A_{mem}$ and $A_{gen}$ disaggregated by task type (`chaining`, `intersection`, `fact_checking`), computes Permeation Dynamics Heatmaps ($l_{src} \to l_{tgt}$ self-patching at $E_{head}$), runs Patchscope hidden state decodings, and logs metrics/heatmaps to the W&B run.
- **Git Commit:** `"feat(eval): add standalone evaluation and diagnostic script with W&B run ID resumption"`

### Task 5: SLURM Job Scripts for Training and Evaluation
- **Files:** `slurm_jobs/run_kug_train.sh`, `slurm_jobs/run_kug_eval.sh`
- **Actions:** Create SLURM batch scripts for single GPU submission with 80GB memory, proper environment exports (`HF_HOME`, `PYTHONPATH`), W&B API key setup, and VRAM logging via `nvidia-smi`.
- **Git Commit:** `"feat(slurm): add SLURM batch submission scripts for training and evaluation jobs"`

### Task 6: Method 1 (Baseline Replication) Execution & Verification
- **Actions:**
  1. Submit `run_kug_train.sh` for Method 1 via `sbatch`.
  2. Perform 2-epoch dry run check, monitor VRAM ($\ge 70$ GB target). Tune batch size if VRAM utilization is low.
  3. Allow training to complete (saving 50 per-epoch adapter checkpoints), then submit `run_kug_eval.sh`.
  4. Verify $A_{mem} \to 1.0$ at epoch 15 and confirm baseline parity with Figure 7 on W&B.
- **Git Commit:** `"eval(baseline): complete Method 1 baseline replication and W&B diagnostic logging"`

### Task 7: Method 2 (2-Stage Mem-then-Gen SFT) Execution & Verification
- **Actions:**
  1. Submit Method 2 training job via `sbatch`.
  2. Monitor VRAM and 2-epoch dry run on W&B.
  3. Submit evaluation job after training completes.
  4. Analyze if $A_{mem}$ suffers catastrophic collapse after Epoch 15 switch on `chaining` vs `intersection`.
- **Git Commit:** `"eval(two_stage): complete Method 2 sequential 2-stage training and catastrophic forgetting evaluation"`

### Task 8: Method 3 (Joint Supervised SFT) Execution & Verification
- **Actions:**
  1. Submit Method 3 training job via `sbatch`.
  2. Monitor VRAM and dry run on W&B.
  3. Submit evaluation job after training completes.
  4. Compare $A_{mem}$ and $A_{gen}$ against Baseline and 2-Stage SFT across all task types.
- **Git Commit:** `"eval(joint): complete Method 3 joint supervised SFT and final multi-method comparative analysis"`

---

## Verification Plan

### Automated Tests
1. **Data Schema Validation:** Test `prepare_kug_datasets.py` outputs for JSON structural validity, non-empty $P_{mem}, P_{gen}$, target entity strings, and correct task labels.
2. **SLURM GPU Dry Run (2 Epochs):** Submit 2-epoch dry-run jobs for each training method via `sbatch` to verify CUDA memory, VRAM utilization ($\ge 70$ GB target), per-epoch adapter checkpoint creation, and W&B logging without OOM errors.

### Empirical & Visual Verification
1. **Baseline Parity:** Verify Method 1 output curves match Figure 7 from Dai et al. ($A_{mem} \to 1.0$ at epoch ~15; $A_{gen} \approx 0$ for chaining; $A_{gen}$ rises for intersection).
2. **Comparative Analysis:** Compare $A_{mem}$ and $A_{gen}$ trajectories across Methods 1, 2, and 3:
   - Check if Method 2 causes catastrophic collapse of $A_{mem}$ after Epoch 15 on `chaining` vs `intersection`.
   - Check if Method 3 achieves high $A_{mem}$ and $A_{gen}$ simultaneously across all task types.
3. **Mechanistic Verification:** Inspect W&B-logged Permeation Heatmaps (resembling Figure 9) and Patchscope hidden state decodings (resembling Figure 8) for success vs failure cases.
