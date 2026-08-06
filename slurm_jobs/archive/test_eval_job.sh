#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/test_eval_%j.out
#SBATCH --error=outputs/logs/test_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
if [[ -f "${HOME}/.hf_token" ]]; then export HF_TOKEN="$(cat "${HOME}/.hf_token")"; fi

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}"
/users/jerryma/.conda/envs/torch2.8/bin/python scripts/test_evaluator.py
