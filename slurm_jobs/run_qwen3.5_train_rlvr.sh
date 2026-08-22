#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG Qwen3.5-2B 2-Stage RLVR Training SLURM Job (GRPO + Thinking Routing)
#
#  Usage: sbatch run_qwen3.5_train_rlvr.sh [method] [batch_size] [grad_accum] [kcr_weight] [oprd_weight]
#    method: two_stage_rlvr | two_stage_breadcrumb_rlvr | two_stage_cot_rlvr |
#            two_stage_oprd_curriculum_rlvr | two_stage_kcr_curriculum_rlvr
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=qwen3.5_rlvr
#SBATCH --account=eclarson_trm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/qwen3.5_rlvr_%j.out
#SBATCH --error=outputs/logs/qwen3.5_rlvr_%j.err

set -euo pipefail

METHOD="${1:-two_stage_kcr_curriculum_rlvr}"
BATCH_SIZE="${2:-4}"
GRAD_ACCUM="${3:-16}"
NUM_ROLLOUTS="${4:-8}"
KL_BETA="${5:-0.04}"
LR="${6:-5e-5}"
MODEL_NAME="Qwen/Qwen3.5-2B"

if [[ "${METHOD}" == "two_stage_kcr_curriculum_rlvr" ]]; then
    KCR_WEIGHT="0.15"
    OPRD_WEIGHT="0.0"
elif [[ "${METHOD}" == "two_stage_oprd_curriculum_rlvr" ]]; then
    KCR_WEIGHT="0.0"
    OPRD_WEIGHT="0.15"
else
    KCR_WEIGHT="0.0"
    OPRD_WEIGHT="0.0"
fi

echo "================================================================"
echo "  KUG Qwen3.5-2B 2-Stage RLVR (Thinking-Preserving GRPO)"
echo "  Method:        ${METHOD}"
echo "  Model:         ${MODEL_NAME}"
echo "  Batch Size:    ${BATCH_SIZE}"
echo "  Grad Accum:    ${GRAD_ACCUM}"
echo "  Rollouts (K):  ${NUM_ROLLOUTS}"
echo "  KL Beta:       ${KL_BETA}"
echo "  KCR Weight:    ${KCR_WEIGHT}"
echo "  OPRD Weight:   ${OPRD_WEIGHT}"
echo "  Learning Rate: ${LR}"
echo "  Account:       eclarson_trm_0001"
echo "  Host:          $(hostname)"
echo "  Date:          $(date)"
echo "  Job ID:        ${SLURM_JOB_ID}"
echo "================================================================"

nvidia-smi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

mkdir -p outputs/logs "outputs/kug_overhaul_v2/${METHOD}_qwen3.5-2b"

# Copy Stage 1 (Epochs 1-15) checkpoints so full 50-epoch eval is seamless
STAGE1_DIR="outputs/kug_overhaul_v2/baseline_qwen3.5-2b"
RLVR_DIR="outputs/kug_overhaul_v2/${METHOD}_qwen3.5-2b"

if [[ -d "${STAGE1_DIR}" ]]; then
    echo "Copying Stage 1 (Epochs 1-15) checkpoints into ${RLVR_DIR}..."
    for ep in $(seq 1 15); do
        if [[ -d "${STAGE1_DIR}/checkpoint-epoch-${ep}" && ! -d "${RLVR_DIR}/checkpoint-epoch-${ep}" ]]; then
            cp -r "${STAGE1_DIR}/checkpoint-epoch-${ep}" "${RLVR_DIR}/"
        fi
    done
fi

INIT_CKPT="${STAGE1_DIR}/checkpoint-epoch-15"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="/users/jerryma/.conda/envs/torch2.8/bin/python"

EXTRA_FLAGS=""
if [[ "${METHOD}" != "two_stage_rlvr" ]]; then
    EXTRA_FLAGS="--use_cot"
fi

${PYTHON_BIN} src/training/train_kug_rlvr.py \
    --method "${METHOD}" \
    --model_name_or_path "${MODEL_NAME}" \
    --init_checkpoint "${INIT_CKPT}" \
    --dataset_path "data/processed/kug_dataset_all.jsonl" \
    --output_dir "outputs/kug_overhaul_v2" \
    --wandb_project "kug_overhaul_qwen3.5_2b" \
    --start_epoch 16 \
    --end_epoch 50 \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_rollouts "${NUM_ROLLOUTS}" \
    --temperature 0.85 \
    --top_p 0.95 \
    --max_prompt_length 256 \
    --max_new_tokens 128 \
    --kl_beta "${KL_BETA}" \
    --clip_eps 0.2 \
    --learning_rate "${LR}" \
    --weight_decay 0.01 \
    --kcr_weight "${KCR_WEIGHT}" \
    --oprd_weight "${OPRD_WEIGHT}" \
    --curriculum_anneal \
    --use_thinking \
    ${EXTRA_FLAGS}

echo "=== RLVR Training Completed Successfully ==="
