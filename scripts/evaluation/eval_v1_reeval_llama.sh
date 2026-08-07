#!/bin/bash
#SBATCH --job-name=eval_v1_llama
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/outputs/logs/eval_v1_llama_%j.out
#SBATCH --error=/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft/outputs/logs/eval_v1_llama_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
if [[ -f "${HOME}/.hf_token" ]]; then export HF_TOKEN="$(cat "${HOME}/.hf_token")"; fi

cd "/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
mkdir -p outputs/logs
/users/jerryma/.conda/envs/torch2.8/bin/python scripts/evaluation/evaluate_v1_runs.py --model_key llama3.2-3b --batch_size 16
