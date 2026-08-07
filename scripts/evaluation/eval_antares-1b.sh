#!/bin/bash
#SBATCH --job-name=eval_antares-1b
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/outputs/logs/eval_antares-1b_%j.out
#SBATCH --error=/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/outputs/logs/eval_antares-1b_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
if [[ -f "${HOME}/.hf_token" ]]; then export HF_TOKEN="$(cat "${HOME}/.hf_token")"; fi

cd "/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
mkdir -p outputs/logs
/users/jerryma/.conda/envs/torch2.8/bin/python scripts/evaluation/evaluate_all_v2.py --model_key antares-1b --batch_size 8
