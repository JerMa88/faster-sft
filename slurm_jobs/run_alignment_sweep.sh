#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Alignment Loss Sweep — runs only the 4 alignment variants (no baseline)
#  Usage:  sbatch run_alignment_sweep.sh <model_key>
#
#  Requires: baseline already completed (data prep, profiling, baseline run)
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=align
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/logs/align_%x_%j.out
#SBATCH --error=outputs/logs/align_%x_%j.err

set -euo pipefail

MODEL_KEY="${1:-qwen3.5-2b}"

if [[ -f "${HOME}/.hf_token" ]]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi

export PYTHONUNBUFFERED=1

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"
HF_CACHE="${WORK_DIR}/hf_cache"

# Map model key → HuggingFace model ID
case "${MODEL_KEY}" in
  llama3.2-3b)      MODEL_ID="meta-llama/Llama-3.2-3B-Instruct" ;;
  qwen3.5-2b)       MODEL_ID="Qwen/Qwen3.5-2B" ;;
  gemma4-e4b)       MODEL_ID="google/gemma-4-E4B-it" ;;
  antares-1b)       MODEL_ID="fdtn-ai/antares-1b" ;;
  nanbeige4.2-3b)   MODEL_ID="Nanbeige/Nanbeige4.2-3B" ;;
  lfm2.5-1.2b)      MODEL_ID="LiquidAI/LFM2.5-1.2B-Base" ;;
  *)
    echo "[ERROR] Unknown model key: '${MODEL_KEY}'"
    exit 1
    ;;
esac

scontrol update JobId="${SLURM_JOB_ID}" JobName="align_${MODEL_KEY}" 2>/dev/null || true

echo "════════════════════════════════════════════════════════════════"
echo "  Alignment Loss Sweep"
echo "  Model key  : ${MODEL_KEY}"
echo "  Model ID   : ${MODEL_ID}"
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Node       : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started    : $(date)"
echo "════════════════════════════════════════════════════════════════"

cd "${WORK_DIR}"

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# GPU diagnostics
nvidia-smi 2>/dev/null | head -12 || echo "nvidia-smi unavailable"
echo ""

PRIME_DATA="${WORK_DIR}/data/processed/stark_prime_qa_v2.jsonl"
MAG_DATA="${WORK_DIR}/data/processed/stark_mag_qa_v2.jsonl"
PROFILE_PATH="${WORK_DIR}/data/processed/layer_profile_${MODEL_KEY}.json"
PROBE_PATH="${WORK_DIR}/data/processed/probe_phi_${MODEL_KEY}.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Pretrain φ* probe if not yet done (needed for probe/hybrid variants)
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "${PROBE_PATH}" ]]; then
    echo ">>> Pretraining φ* probe for ${MODEL_KEY} …"
    ${PYTHON} scripts/training/pretrain_probe.py \
        --model_key   "${MODEL_KEY}" \
        --data_path   "${PRIME_DATA}" \
        --layer_profile "${PROFILE_PATH}" \
        --probe_epochs 10 \
        --hf_cache    "${HF_CACHE}" \
        2>&1
    echo ">>> φ* probe done at $(date)"
else
    echo ">>> φ* probe for ${MODEL_KEY} already exists — skipping."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Alignment-Aware LoRA — all 4 variants, both datasets
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> Alignment-aware LoRA sweep …"

LAMBDA=0.1
WARMUP=3

for LOSS_VARIANT in rep_distill contrastive probe hybrid; do
    for DATASET in "prime" "mag"; do
        if [[ "${DATASET}" == "prime" ]]; then
            DATA_FILE="${PRIME_DATA}"
        else
            DATA_FILE="${MAG_DATA}"
        fi

        # Skip if already done
        RUN_DIR="outputs/runs_v2/${MODEL_KEY}/stark_${DATASET}"
        EXPECTED_SLUG="${MODEL_ID//\//-}_${LOSS_VARIANT}_lam${LAMBDA}_seed42"
        # Use the -- replacement like train_sft.py does
        EXPECTED_SLUG2="${MODEL_ID//\//-}"
        EXPECTED_SLUG2="${EXPECTED_SLUG2//-/--}"
        
        EXISTING=$(find "${RUN_DIR}" -maxdepth 2 -name "checkpoint_epoch50" -type d 2>/dev/null | \
                   grep -i "${LOSS_VARIANT}" | head -1 || true)
        if [[ -n "${EXISTING}" ]]; then
            echo "    >> ${LOSS_VARIANT} ${DATASET^^} already complete — skipping."
            continue
        fi

        echo "    >> ${LOSS_VARIANT} — STaRK-${DATASET^^} …"
        ${PYTHON} scripts/training/train_sft.py \
            --model_id      "${MODEL_ID}" \
            --model_key     "${MODEL_KEY}" \
            --data_path     "${DATA_FILE}" \
            --loss_variant  "${LOSS_VARIANT}" \
            --probe_path    "${PROBE_PATH}" \
            --epochs        50 \
            --lambda_align  "${LAMBDA}" \
            --warmup_epochs "${WARMUP}" \
            --batch_size    0 \
            --lr            2e-4 \
            --lora_rank     16 \
            --layer_profile "${PROFILE_PATH}" \
            --out_dir       "outputs/runs_v2/${MODEL_KEY}/stark_${DATASET}" \
            --hf_cache      "${HF_CACHE}" \
            --seed          42 \
            2>&1
        echo "    >> ${LOSS_VARIANT} ${DATASET^^} done at $(date)"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ALIGNMENT SWEEP COMPLETE for ${MODEL_KEY}"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════════"

find "outputs/runs/${MODEL_KEY}" -name "metrics.json" -printf "  %p\n" 2>/dev/null || true
