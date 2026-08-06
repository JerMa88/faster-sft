#!/bin/bash
#SBATCH --job-name=eval_sft
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err

set -euo pipefail

if [[ -f "${HOME}/.hf_token" ]]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi

export PYTHONUNBUFFERED=1
WORK_DIR="/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

cd "${WORK_DIR}"
${PYTHON} scripts/evaluate_all.py
