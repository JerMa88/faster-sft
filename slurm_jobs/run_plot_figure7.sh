#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Plot Figure 7 — SFT Training Dynamics
#  Parses fast_eval log and generates 3-panel A_mem vs A_gen plot
#
#  Usage: sbatch slurm_jobs/run_plot_figure7.sh <eval_log> [out_png]
# ═══════════════════════════════════════════════════════════════

#SBATCH --job-name=plot_fig7
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/logs/plot_figure7_%j.out
#SBATCH --error=outputs/logs/plot_figure7_%j.err

set -euo pipefail

EVAL_LOG="${1:-outputs/logs/fast_eval_474756.out}"
OUT_PNG="${2:-outputs/figures/figure7_sft_baseline.png}"

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"
mkdir -p outputs/figures

echo "Plotting Figure 7 from: ${EVAL_LOG}"
${PYTHON} scripts/plotting/plot_figure7_replication.py \
    --log "${EVAL_LOG}" \
    --out "${OUT_PNG}" \
    --theme all

echo "Done. Outputs saved to ./figures/ and ${OUT_PNG}"
