#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG 2-Stage RLVR + OPRD + Curriculum Annealing Training SLURM Job
#  Combines Online Policy Representation Distillation (OPRD) with
#  Bridge-Penalized Curriculum Annealing (Phase 1 -> Phase 2 -> Phase 3 Penalty)
#
#  Usage: sbatch run_kug_oprd_curriculum_rlvr.sh
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_oprd_rlvr
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/kug_oprd_rlvr_%j.out
#SBATCH --error=outputs/logs/kug_oprd_rlvr_%j.err

set -euo pipefail

BATCH_SIZE="${1:-8}"
GRAD_ACCUM="${2:-8}"
NUM_ROLLOUTS="${3:-8}"
KL_BETA="${4:-0.04}"
LR="${5:-5e-5}"
OPRD_WEIGHT="${6:-0.10}"
METHOD="two_stage_oprd_curriculum_rlvr"

echo "================================================================"
echo "  KUG 2-Stage RLVR + OPRD + Curriculum Annealing"
echo "  Method:        ${METHOD}"
echo "  Batch Size:    ${BATCH_SIZE}"
echo "  Grad Accum:    ${GRAD_ACCUM}"
echo "  Rollouts (K):  ${NUM_ROLLOUTS}"
echo "  KL Beta:       ${KL_BETA}"
echo "  OPRD Weight:   ${OPRD_WEIGHT}"
echo "  Learning Rate: ${LR}"
echo "  Host:          $(hostname)"
echo "  Date:          $(date)"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "================================================================"

nvidia-smi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

mkdir -p outputs/logs "outputs/kug_overhaul_v2/${METHOD}_qwen2.5-1.5b"

# Copy Stage 1 (Epochs 1-15) checkpoints so full 50-epoch eval is seamless
STAGE1_DIR="outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b"
RLVR_DIR="outputs/kug_overhaul_v2/${METHOD}_qwen2.5-1.5b"

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

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

# Background VRAM monitor
(
    while true; do
        DATE=$(date +%H:%M:%S)
        VRAM=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
        echo "${DATE} | ${VRAM}" >> "outputs/logs/vram_${METHOD}_${SLURM_JOB_ID}.log"
        sleep 30
    done
) &
VRAM_PID=$!
trap "kill ${VRAM_PID} 2>/dev/null || true" EXIT

echo "Starting RLVR + OPRD + Curriculum Annealing training on GPU..."
${PYTHON} src/training/train_kug_rlvr.py \
    --method "${METHOD}" \
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
    --temperature 0.85 \
    --top_p 0.95 \
    --max_prompt_length 256 \
    --max_new_tokens 96 \
    --kl_beta "${KL_BETA}" \
    --oprd_weight "${OPRD_WEIGHT}" \
    --curriculum_anneal \
    --use_cot \
    --clip_eps 0.2 \
    --learning_rate "${LR}" \
    --weight_decay 0.01

echo "Training completed."
echo "=== Peak VRAM ==="
tail -n 10 "outputs/logs/vram_${METHOD}_${SLURM_JOB_ID}.log" || true
