#!/bin/bash
#SBATCH --job-name=inspect_fc
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/logs/inspect_fc_%j.out
#SBATCH --error=outputs/logs/inspect_fc_%j.err

set -euo pipefail

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}"

export PYTHONUNBUFFERED=1
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

echo "Running Full Fact Checking Test..."
${PYTHON} scripts/analysis/test_fc_eval.py
