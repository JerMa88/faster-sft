#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG Overhaul v2 Training SLURM Job (Completion-Only Loss)
#
#  Usage: sbatch run_kug_train_v2.sh [method] [batch_size] [grad_accum]
#    method:     baseline | two_stage | joint  (default: baseline)
#    batch_size: integer                        (default: 64)
#    grad_accum: integer                        (default: 1)
#
#  Examples:
#    sbatch run_kug_train_v2.sh baseline 64 1
#    sbatch run_kug_train_v2.sh two_stage 64 1
#    sbatch run_kug_train_v2.sh joint 64 1
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_train_v2
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/kug_train_v2_%j.out
#SBATCH --error=outputs/logs/kug_train_v2_%j.err

set -euo pipefail

# ── Argument parsing with validation ────────────────────────────────────────
METHOD="${1:-baseline}"
BATCH_SIZE="${2:-64}"
GRAD_ACCUM="${3:-1}"

# Validate method
if [[ "${METHOD}" != "baseline" && "${METHOD}" != "two_stage" && "${METHOD}" != "joint" ]]; then
    echo "ERROR: Invalid method '${METHOD}'. Must be: baseline | two_stage | joint"
    echo "Usage: sbatch run_kug_train_v2.sh [method] [batch_size] [grad_accum]"
    exit 1
fi

# Validate batch_size is a positive integer
if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -lt 1 ]]; then
    echo "ERROR: Invalid batch_size '${BATCH_SIZE}'. Must be a positive integer."
    exit 1
fi

echo "================================================================"
echo "  KUG Overhaul v2 Training (Completion-Only Loss)"
echo "  Method:     ${METHOD}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Grad Accum: ${GRAD_ACCUM}"
echo "  Host:       $(hostname)"
echo "  Date:       $(date)"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "================================================================"

nvidia-smi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

mkdir -p outputs/logs outputs/kug_overhaul_v2

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"
# Prevent CUDA OOM fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

# Verify Python and script exist
if [[ ! -f "${PYTHON}" ]]; then
    echo "ERROR: Python not found at ${PYTHON}"
    exit 1
fi
if [[ ! -f "src/training/train_kug_overhaul.py" ]]; then
    echo "ERROR: Training script not found"
    exit 1
fi
if [[ ! -f "data/processed/kug_dataset_all.jsonl" ]]; then
    echo "ERROR: Dataset not found at data/processed/kug_dataset_all.jsonl"
    exit 1
fi

# ── Background VRAM monitor ──────────────────────────────────────────────────
VRAM_LOG="outputs/logs/kug_vram_monitor_${SLURM_JOB_ID}.log"
(
    while true; do
        echo "$(date '+%H:%M:%S') | $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" >> "${VRAM_LOG}"
        sleep 30
    done
) &
MONITOR_PID=$!
trap "kill ${MONITOR_PID} 2>/dev/null || true; echo '=== Peak VRAM ==='; sort -t'|' -k2 -rn ${VRAM_LOG} | head -3" EXIT

echo ""
echo "--- Launching training: method=${METHOD}, batch_size=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM} ---"

${PYTHON} src/training/train_kug_overhaul.py \
    --method "${METHOD}" \
    --model_name_or_path "Qwen/Qwen2.5-1.5B" \
    --dataset_path "data/processed/kug_dataset_all.jsonl" \
    --output_dir "outputs/kug_overhaul_v2" \
    --wandb_project "kug_overhaul_qwen1.5b" \
    --num_epochs 50 \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 2e-4

echo "Training job finished successfully."
nvidia-smi
