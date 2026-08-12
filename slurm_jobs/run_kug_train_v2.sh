#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG Overhaul v2 Training SLURM Job (Completion-Only Loss)
#  Usage: sbatch run_kug_train_v2.sh <method> [batch_size] [grad_accum]
#    method: baseline | two_stage | joint
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_train_v2
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/kug_train_v2_%x_%j.out
#SBATCH --error=outputs/logs/kug_train_v2_%x_%j.err

set -euo pipefail

METHOD="${1:-baseline}"
BATCH_SIZE="${2:-16}"
GRAD_ACCUM="${3:-1}"

echo "================================================================"
echo "  Starting KUG Training v2 Job (Completion-Only Loss)"
echo "  Method: ${METHOD}"
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

echo "Executing v2 training script..."
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

echo "Training job finished."
nvidia-smi
