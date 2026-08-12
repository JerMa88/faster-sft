#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG v2 Dry Run — 2 Epochs, VRAM Profiling
#  Verifies completion-only loss, VRAM utilization (>70GB target), W&B logging
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_dryrun_v2
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/kug_dryrun_v2_%j.out
#SBATCH --error=outputs/logs/kug_dryrun_v2_%j.err

set -euo pipefail

BATCH_SIZE="${1:-16}"
GRAD_ACCUM="${2:-1}"

echo "================================================================"
echo "  KUG v2 Dry Run (Completion-Only Loss)"
echo "  Batch Size: ${BATCH_SIZE}, Grad Accum: ${GRAD_ACCUM}"
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "================================================================"

nvidia-smi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}"

mkdir -p outputs/logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

echo ""
echo "--- Running 2-epoch dry run to check VRAM and W&B logging ---"
${PYTHON} src/training/train_kug_overhaul.py \
    --method baseline \
    --model_name_or_path "Qwen/Qwen2.5-1.5B" \
    --dataset_path "data/processed/kug_dataset_all.jsonl" \
    --output_dir "outputs/kug_dryrun_v2" \
    --wandb_project "kug_overhaul_qwen1.5b" \
    --num_epochs 2 \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 2e-4

echo ""
echo "--- Post-run VRAM status ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

echo "Dry run finished."
