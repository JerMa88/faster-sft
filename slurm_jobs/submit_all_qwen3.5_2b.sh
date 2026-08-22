#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Master Submission Script: Full Suite of 8 Methods for Qwen/Qwen3.5-2B
#  SLURM Allocation Account: eclarson_trm_0001
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

mkdir -p outputs/logs outputs/kug_overhaul_v2

echo "================================================================"
echo "  Submitting Full Suite of 8 Methods for Qwen/Qwen3.5-2B"
echo "  SLURM Account: eclarson_trm_0001"
echo "================================================================"

# ── 1. Submit SFT Methods ───────────────────────────────────────────────────
echo "[1/8] Submitting Method 1: Baseline SFT (Memory-Only)..."
JOB_BASELINE=$(sbatch --parsable slurm_jobs/run_qwen3.5_train_sft.sh baseline 16 2)
echo "      -> Submitted Job ID: ${JOB_BASELINE}"

echo "[2/8] Submitting Method 2: 2-Stage SFT (Sequential)..."
JOB_TWO_STAGE=$(sbatch --parsable slurm_jobs/run_qwen3.5_train_sft.sh two_stage 16 2)
echo "      -> Submitted Job ID: ${JOB_TWO_STAGE}"

echo "[3/8] Submitting Method 3: Joint Multitask SFT..."
JOB_JOINT=$(sbatch --parsable slurm_jobs/run_qwen3.5_train_sft.sh joint 16 2)
echo "      -> Submitted Job ID: ${JOB_JOINT}"

# ── 2. Submit RLVR Methods (Chained with dependency on Baseline Stage 1) ────
echo "[4/8] Submitting Method 4: Basic 2-Stage RLVR (after baseline)..."
JOB_RLVR=$(sbatch --parsable --dependency="afterok:${JOB_BASELINE}" slurm_jobs/run_qwen3.5_train_rlvr.sh two_stage_rlvr 8 8)
echo "      -> Submitted Job ID: ${JOB_RLVR} (Dependency: ${JOB_BASELINE})"

echo "[5/8] Submitting Method 5: Breadcrumb Permeation RLVR (after baseline)..."
JOB_BREADCRUMB=$(sbatch --parsable --dependency="afterok:${JOB_BASELINE}" slurm_jobs/run_qwen3.5_train_rlvr.sh two_stage_breadcrumb_rlvr 8 8)
echo "      -> Submitted Job ID: ${JOB_BREADCRUMB} (Dependency: ${JOB_BASELINE})"

echo "[6/8] Submitting Method 6: 2-Step CoT RLVR (after baseline)..."
JOB_COT=$(sbatch --parsable --dependency="afterok:${JOB_BASELINE}" slurm_jobs/run_qwen3.5_train_rlvr.sh two_stage_cot_rlvr 8 8)
echo "      -> Submitted Job ID: ${JOB_COT} (Dependency: ${JOB_BASELINE})"

echo "[7/8] Submitting Method 7: OPRD + Curriculum RLVR (after baseline)..."
JOB_OPRD=$(sbatch --parsable --dependency="afterok:${JOB_BASELINE}" slurm_jobs/run_qwen3.5_train_rlvr.sh two_stage_oprd_curriculum_rlvr 8 8 0.0 0.15)
echo "      -> Submitted Job ID: ${JOB_OPRD} (Dependency: ${JOB_BASELINE})"

echo "[8/8] Submitting Method 8: Knowledge-Circuit Routing (KCR) RLVR (after baseline)..."
JOB_KCR=$(sbatch --parsable --dependency="afterok:${JOB_BASELINE}" slurm_jobs/run_qwen3.5_train_rlvr.sh two_stage_kcr_curriculum_rlvr 8 8 0.15 0.0)
echo "      -> Submitted Job ID: ${JOB_KCR} (Dependency: ${JOB_BASELINE})"

echo "================================================================"
echo "  All 8 jobs successfully queued under eclarson_trm_0001!"
echo "  Baseline SFT:        ${JOB_BASELINE}"
echo "  2-Stage SFT:         ${JOB_TWO_STAGE}"
echo "  Joint Multitask SFT: ${JOB_JOINT}"
echo "  2-Stage RLVR:        ${JOB_RLVR}"
echo "  Breadcrumb RLVR:     ${JOB_BREADCRUMB}"
echo "  2-Step CoT RLVR:     ${JOB_COT}"
echo "  OPRD Curriculum:     ${JOB_OPRD}"
echo "  KCR RLVR:            ${JOB_KCR}"
echo "================================================================"
