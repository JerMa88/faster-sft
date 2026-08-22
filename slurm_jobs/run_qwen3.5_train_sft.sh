#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG Qwen3.5-2B SFT Training SLURM Job (Thinking-Preserving Completion Loss)
#
#  Usage: sbatch run_qwen3.5_train_sft.sh [method] [batch_size] [grad_accum]
#    method: baseline | two_stage | joint  (default: baseline)
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=qwen3.5_sft
#SBATCH --account=eclarson_trm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/qwen3.5_sft_%j.out
#SBATCH --error=outputs/logs/qwen3.5_sft_%j.err

set -euo pipefail

METHOD="${1:-baseline}"
BATCH_SIZE="${2:-16}"
GRAD_ACCUM="${3:-2}"
MODEL_NAME="Qwen/Qwen3.5-2B"

if [[ "${METHOD}" != "baseline" && "${METHOD}" != "two_stage" && "${METHOD}" != "joint" ]]; then
    echo "ERROR: Invalid method '${METHOD}'. Must be: baseline | two_stage | joint"
    exit 1
fi

echo "================================================================"
echo "  KUG Qwen3.5-2B SFT (Thinking-Preserving Completion Loss)"
echo "  Method:     ${METHOD}"
echo "  Model:      ${MODEL_NAME}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Grad Accum: ${GRAD_ACCUM}"
echo "  Account:    eclarson_trm_0001"
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="/users/jerryma/.conda/envs/torch2.8/bin/python"

${PYTHON_BIN} src/training/train_kug_overhaul.py \
    --method "${METHOD}" \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_path "data/processed/kug_dataset_all.jsonl" \
    --output_dir "outputs/kug_overhaul_v2" \
    --wandb_project "kug_overhaul_qwen3.5_2b" \
    --num_epochs 50 \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_length 512 \
    --use_thinking

echo "=== SFT Training Completed Successfully ==="
