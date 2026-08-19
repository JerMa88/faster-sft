#!/bin/bash
#SBATCH --job-name=cot_eval_comp
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/cot_comp_%j.out
#SBATCH --error=outputs/logs/cot_comp_%j.err

set -euo pipefail

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

${PYTHON} scripts/analysis/compare_cot_eval_prompts.py
