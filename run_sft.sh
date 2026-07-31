#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  Alignment-Aware SFT — Single-Model SLURM Job
#  Usage:  sbatch run_sft.sh <model_key>
#
#  Model keys and their HuggingFace IDs:
#    llama3.2-3b      → meta-llama/Llama-3.2-3B-Instruct
#    qwen3.5-2b       → Qwen/Qwen3.5-2B
#    gemma2-2b        → google/gemma-2-2b-it
#    antares-1b       → fdtn-ai/antares-1b
#    nanbeige4.2-3b   → Nanbeige/Nanbeige4.2-3B
#    lfm2.5-1.2b      → LiquidAI/LFM2.5-1.2B-Base
#
#  Examples:
#    sbatch run_sft.sh llama3.2-3b
#    sbatch run_sft.sh qwen3.5-1.5b
#    for m in llama3.2-3b qwen3.5-2b gemma2-2b antares-1b nanbeige4.2-3b lfm2.5-1.2b; do
#        sbatch run_sft.sh "$m"
#    done
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=sft_%a
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/logs/sft_%x_%j.out
#SBATCH --error=outputs/logs/sft_%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0. Argument & environment setup
# ─────────────────────────────────────────────────────────────────────────────
MODEL_KEY="${1:-qwen3.5-1.5b}"   # first positional arg becomes the model key

# Load HuggingFace token for gated models (llama3.2-3b, antares-1b).
# Save your token once with: echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token
if [[ -f "${HOME}/.hf_token" ]]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
    echo ">>> HF_TOKEN loaded from ~/.hf_token"
elif [[ -z "${HF_TOKEN:-}" ]]; then
    echo ">>> [WARN] HF_TOKEN not set — gated models (llama3.2-3b, antares-1b) will fail to download."
fi

export PYTHONUNBUFFERED=1

WORK_DIR="/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"
HF_CACHE="${WORK_DIR}/hf_cache"

# Map model key → HuggingFace model ID
case "${MODEL_KEY}" in
  llama3.2-3b)      MODEL_ID="meta-llama/Llama-3.2-3B-Instruct" ;;
  qwen3.5-2b)       MODEL_ID="Qwen/Qwen3.5-2B" ;;
  gemma2-2b)        MODEL_ID="google/gemma-2-2b-it" ;;
  antares-1b)       MODEL_ID="fdtn-ai/antares-1b" ;;
  nanbeige4.2-3b)   MODEL_ID="Nanbeige/Nanbeige4.2-3B" ;;
  lfm2.5-1.2b)      MODEL_ID="LiquidAI/LFM2.5-1.2B-Base" ;;
  *)
    echo "[ERROR] Unknown model key: '${MODEL_KEY}'"
    echo "        Valid keys: llama3.2-3b qwen3.5-2b gemma2-2b antares-1b nanbeige4.2-3b lfm2.5-1.2b"
    exit 1
    ;;
esac

# Re-name the SLURM job to include the model key (visible in squeue)
scontrol update JobId="${SLURM_JOB_ID}" JobName="sft_${MODEL_KEY}" 2>/dev/null || true

echo "════════════════════════════════════════════════════════════════"
echo "  Alignment-Aware SFT Experiment"
echo "  Model key  : ${MODEL_KEY}"
echo "  Model ID   : ${MODEL_ID}"
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Node       : $(hostname)"
echo "  GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started    : $(date)"
echo "════════════════════════════════════════════════════════════════"
echo ""

cd "${WORK_DIR}"
mkdir -p outputs/logs outputs/runs "${HF_CACHE}"

# GPU diagnostics
nvidia-smi 2>/dev/null | head -12 || echo "nvidia-smi unavailable"
echo ""

# Export HF cache so all Python sub-processes see it
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
# Maximise VRAM usage: allow full memory fraction, enable TF32 on A100
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Preparation (skip if both processed files already exist)
# ─────────────────────────────────────────────────────────────────────────────
PRIME_DATA="${WORK_DIR}/data/processed/stark_prime_qa.jsonl"
MAG_DATA="${WORK_DIR}/data/processed/stark_mag_qa.jsonl"

if [[ ! -f "${PRIME_DATA}" || ! -f "${MAG_DATA}" ]]; then
    echo ">>> [1/4] Building STaRK QA pairs …"
    ${PYTHON} scripts/prepare_data.py \
        --dataset both \
        --num_facts 1000 \
        2>&1
    echo ">>> Data preparation done at $(date)"
else
    echo ">>> [1/4] STaRK data already exists — skipping."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Layer Profiling — Phase 1.5
#    Run once per model (profile saved with model key in filename).
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_PATH="${WORK_DIR}/data/processed/layer_profile_${MODEL_KEY}.json"

if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo ">>> [2/5] Running layer profiling for ${MODEL_KEY} …"
    ${PYTHON} scripts/run_profiling.py \
        --model_id    "${MODEL_ID}" \
        --data_path   "${PRIME_DATA}" \
        --n_probe_samples 200 \
        --out_path    "data/processed/layer_profile_${MODEL_KEY}.json" \
        --hf_cache    "${HF_CACHE}" \
        2>&1
    echo ">>> Layer profiling done at $(date)"
else
    echo ">>> [2/5] Layer profile for ${MODEL_KEY} already exists — skipping."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2b. Pretrain φ* probe (needed for ProbeLoss and Hybrid variants)
#     Run once per model on STaRK-Prime memorization set.
# ─────────────────────────────────────────────────────────────────────────────
PROBE_PATH="${WORK_DIR}/data/processed/probe_phi_${MODEL_KEY}.pt"

if [[ ! -f "${PROBE_PATH}" ]]; then
    echo ">>> [2b/5] Pretraining φ* probe for ${MODEL_KEY} …"
    ${PYTHON} scripts/pretrain_probe.py \
        --model_key   "${MODEL_KEY}" \
        --data_path   "${PRIME_DATA}" \
        --layer_profile "${PROFILE_PATH}" \
        --probe_epochs 10 \
        --hf_cache    "${HF_CACHE}" \
        2>&1
    echo ">>> φ* probe done at $(date)"
else
    echo ">>> [2b/5] φ* probe for ${MODEL_KEY} already exists — skipping."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. Baseline LoRA (no alignment loss) — STaRK-Prime and STaRK-MAG
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [3/5] Baseline LoRA runs …"

for DATASET in "prime" "mag"; do
    if [[ "${DATASET}" == "prime" ]]; then
        DATA_FILE="${PRIME_DATA}"
    else
        DATA_FILE="${MAG_DATA}"
    fi

    echo "    >> Baseline LoRA — STaRK-${DATASET^^} …"
    ${PYTHON} scripts/train_sft.py \
        --model_id      "${MODEL_ID}" \
        --model_key     "${MODEL_KEY}" \
        --data_path     "${DATA_FILE}" \
        --loss_variant  baseline \
        --epochs        50 \
        --lambda_align  0.0 \
        --warmup_epochs 3 \
        --batch_size    0 \
        --lr            2e-4 \
        --lora_rank     16 \
        --layer_profile "${PROFILE_PATH}" \
        --out_dir       "outputs/runs/${MODEL_KEY}/stark_${DATASET}" \
        --hf_cache      "${HF_CACHE}" \
        --seed          42 \
        2>&1
    echo "    >> Baseline LoRA ${DATASET^^} done at $(date)"
done
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 3b. Self-Patching Scan — update l_t in layer_profile after epoch-3 checkpoint
#     Uses the STaRK-Prime baseline epoch-3 checkpoint (the warmup checkpoint).
#     This refines l_t from the heuristic ~0.5L to the empirically optimal value.
# ─────────────────────────────────────────────────────────────────────────────
WARMUP_CKPT="${WORK_DIR}/outputs/runs/${MODEL_KEY}/stark_prime/baseline--${MODEL_ID//\//-}-baseline_lam0.0_seed42/checkpoint_epoch3"

# Find the actual checkpoint directory (run slug may vary slightly)
WARMUP_CKPT_FOUND=$(find "${WORK_DIR}/outputs/runs/${MODEL_KEY}/stark_prime" \
    -maxdepth 2 -name "checkpoint_epoch3" -type d 2>/dev/null | head -1)

if [[ -n "${WARMUP_CKPT_FOUND}" ]]; then
    echo ">>> [3b/6] Running self-patch scan on warmup checkpoint …"
    echo "    Checkpoint: ${WARMUP_CKPT_FOUND}"
    ${PYTHON} -m src.profiling.self_patch_scan \
        --checkpoint    "${WARMUP_CKPT_FOUND}" \
        --model_id      "${MODEL_ID}" \
        --data_path     "${PRIME_DATA}" \
        --profile_path  "${PROFILE_PATH}" \
        --n_samples     100 \
        --plots_dir     "outputs/plots/${MODEL_KEY}" \
        --hf_cache      "${HF_CACHE}" \
        2>&1
    echo ">>> Self-patch scan done at $(date). l_t updated in ${PROFILE_PATH}"
else
    echo ">>> [3b/6] Warmup checkpoint (epoch3) not found — skipping self-patch scan."
    echo "    Alignment variants will use heuristic l_t from layer profile."
fi
echo ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Alignment-Aware LoRA — all 4 variants, both datasets
# ─────────────────────────────────────────────────────────────────────────────
echo ">>> [4/6] Alignment-aware LoRA sweep …"

LAMBDA=0.1
WARMUP=3

for LOSS_VARIANT in rep_distill contrastive probe hybrid; do
    for DATASET in "prime" "mag"; do
        if [[ "${DATASET}" == "prime" ]]; then
            DATA_FILE="${PRIME_DATA}"
        else
            DATA_FILE="${MAG_DATA}"
        fi

        echo "    >> ${LOSS_VARIANT} — STaRK-${DATASET^^} …"
        ${PYTHON} scripts/train_sft.py \
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
            --out_dir       "outputs/runs/${MODEL_KEY}/stark_${DATASET}" \
            --hf_cache      "${HF_CACHE}" \
            --seed          42 \
            2>&1
        echo "    >> ${LOSS_VARIANT} ${DATASET^^} done at $(date)"
    done
done

echo ""

echo "════════════════════════════════════════════════════════════════"
echo "  ALL RUNS COMPLETE for ${MODEL_KEY}"
echo "  Finished: $(date)"
echo "  Outputs:  ${WORK_DIR}/outputs/runs/${MODEL_KEY}/"
echo "════════════════════════════════════════════════════════════════"

# List generated metrics files
find "outputs/runs/${MODEL_KEY}" -name "metrics.json" -printf "  %p\n" 2>/dev/null || true
