#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Fast KUG Accuracy Evaluation SLURM Job
#  Evaluates all 50 epoch checkpoints in ~3 minutes (accuracy metrics only)
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=fast_eval_v2
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/fast_eval_%j.out
#SBATCH --error=outputs/logs/fast_eval_%j.err

set -euo pipefail

CKPT_DIR="${1:-outputs/kug_overhaul_v2/baseline_qwen2.5-1.5b}"
SAMPLE_SIZE="${2:-200}"

echo "================================================================"
echo "  Fast KUG Accuracy Evaluation"
echo "  Checkpoint Dir: ${CKPT_DIR}"
echo "  Sample Size: ${SAMPLE_SIZE}"
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

echo "Running fast accuracy evaluation across all 50 epoch checkpoints..."
${PYTHON} scripts/analysis/fast_kug_eval.py \
    --ckpt_dir "${CKPT_DIR}" \
    --eval_dataset_path "data/processed/kug_dataset_all.jsonl" \
    --sample_size "${SAMPLE_SIZE}"

echo "Fast evaluation completed."
