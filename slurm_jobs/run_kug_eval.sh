#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG Overhaul Evaluation SLURM Job
#  Usage: sbatch run_kug_eval.sh <checkpoint_dir>
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_eval
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/kug_eval_%x_%j.out
#SBATCH --error=outputs/logs/kug_eval_%x_%j.err

set -euo pipefail

CKPT_DIR="${1}"

if [[ -z "${CKPT_DIR}" ]]; then
    echo "Usage: sbatch run_kug_eval.sh <checkpoint_dir>"
    exit 1
fi

echo "================================================================"
echo "  Starting KUG Standalone Evaluation Job"
echo "  Checkpoint Dir: ${CKPT_DIR}"
echo "  Host: $(hostname)"
echo "  Date: $(date)"
echo "================================================================"

nvidia-smi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

echo "Executing standalone evaluation script..."
${PYTHON} scripts/analysis/fast_kug_eval.py \
    --ckpt_dir "${CKPT_DIR}" \
    --eval_dataset_path "data/processed/kug_dataset_all.jsonl" \
    --sample_size 200

echo "Evaluation job finished."
nvidia-smi
