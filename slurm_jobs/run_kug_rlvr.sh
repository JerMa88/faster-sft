#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG 2-Stage RLVR Training SLURM Job (GRPO with Verifiable Rewards)
#
#  Usage: sbatch run_kug_rlvr.sh [batch_size] [grad_accum] [num_rollouts] [kl_beta]
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_rlvr
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/kug_rlvr_%j.out
#SBATCH --error=outputs/logs/kug_rlvr_%j.err

set -euo pipefail

BATCH_SIZE="${1:-16}"
GRAD_ACCUM="${2:-4}"
NUM_ROLLOUTS="${3:-4}"
KL_BETA="${4:-0.04}"
LR="${5:-5e-5}"

echo "================================================================"
echo "  KUG 2-Stage RLVR Training (GRPO with Verifiable Rewards)"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Grad Accum:   ${GRAD_ACCUM}"
echo "  Rollouts (K): ${NUM_ROLLOUTS}"
echo "  KL Beta:      ${KL_BETA}"
echo "  Learning Rate:${LR}"
echo "  Host:         $(hostname)"
echo "  Date:         $(date)"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "================================================================"

nvidia-smi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

mkdir -p outputs/logs outputs/kug_overhaul_v2/two_stage_rlvr_qwen2.5-1.5b

# Copy Stage 1 (Epochs 1-15) checkpoints so full 50-epoch eval is seamless
STAGE1_DIR="outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b"
RLVR_DIR="outputs/kug_overhaul_v2/two_stage_rlvr_qwen2.5-1.5b"

echo "Copying Stage 1 (Epochs 1-15) checkpoints into ${RLVR_DIR}..."
for ep in $(seq 1 15); do
    if [[ -d "${STAGE1_DIR}/checkpoint-epoch-${ep}" && ! -d "${RLVR_DIR}/checkpoint-epoch-${ep}" ]]; then
        cp -r "${STAGE1_DIR}/checkpoint-epoch-${ep}" "${RLVR_DIR}/"
    fi
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

# Background VRAM monitor
VRAM_LOG="outputs/logs/kug_rlvr_vram_${SLURM_JOB_ID}.log"
(
    while true; do
        echo "$(date '+%H:%M:%S') | $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" >> "${VRAM_LOG}"
        sleep 30
    done
) &
MONITOR_PID=$!
trap "kill ${MONITOR_PID} 2>/dev/null || true; echo '=== Peak VRAM ==='; sort -t'|' -k2 -rn ${VRAM_LOG} | head -3" EXIT

echo ""
echo "--- Launching RLVR Training (Epochs 16 to 50) ---"

${PYTHON} src/training/train_kug_rlvr.py \
    --method "two_stage_rlvr" \
    --model_name_or_path "Qwen/Qwen2.5-1.5B" \
    --init_checkpoint "${STAGE1_DIR}/checkpoint-epoch-15" \
    --dataset_path "data/processed/kug_dataset_all.jsonl" \
    --output_dir "outputs/kug_overhaul_v2" \
    --wandb_project "kug_overhaul_qwen1.5b" \
    --start_epoch 16 \
    --end_epoch 50 \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --temperature 0.7 \
    --top_p 0.9 \
    --kl_beta "${KL_BETA}" \
    --learning_rate "${LR}"

echo "RLVR training completed successfully."
nvidia-smi
