#!/bin/bash
#SBATCH --job-name=eval_v3_align
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/eval_v3_align_%j.out
#SBATCH --error=outputs/logs/eval_v3_align_%j.err
set -euo pipefail

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python3"
HF_CACHE="./hf_cache"

# Read HF token
if [[ -f ~/.hf_token ]]; then
    export HF_TOKEN=$(cat ~/.hf_token)
fi

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"

echo "══════════════════════════════════════════════════"
echo "  Evaluation V3 — All Runs (skips already done)"
echo "  (Multi-token generation + string match)"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════"

# Run the batch evaluator — it scans all outputs/runs/*/*/
# and evaluates any run with checkpoints
${PYTHON} scripts/evaluate_all.py 2>&1

echo ""
echo "Running alignment analysis..."
${PYTHON} scripts/analyze_alignment.py 2>&1 || true

echo ""
echo "══════════════════════════════════════════════════"
echo "  All done at $(date)"
echo "══════════════════════════════════════════════════"
