#!/bin/bash
#SBATCH --job-name=eval_v2
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/eval_v2_%x_%j.out
#SBATCH --error=outputs/logs/eval_v2_%x_%j.err
set -euo pipefail

MODEL_KEY="${1:-}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python3"
HF_CACHE="./hf_cache"

if [[ -f ~/.hf_token ]]; then
    export HF_TOKEN=$(cat ~/.hf_token)
fi

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
export PYTHONUNBUFFERED=1

echo "══════════════════════════════════════════════════"
echo "  Parallel Evaluation V2 (Fixes 1-4) — Model: ${MODEL_KEY}"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════"

if [[ -n "${MODEL_KEY}" ]]; then
    ${PYTHON} scripts/evaluate_all_v2.py --model_key "${MODEL_KEY}" 2>&1
else
    ${PYTHON} scripts/evaluate_all_v2.py 2>&1
fi

echo "══════════════════════════════════════════════════"
echo "  Finished at $(date)"
echo "══════════════════════════════════════════════════"
