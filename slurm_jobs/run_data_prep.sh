#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
#  KUG Dataset Preparation SLURM Job — v4 (Paper-Faithful KG Triplets)
#
#  Builds chaining/intersection/fact-checking from RAW KG triplets exactly
#  as the Mem2Gen paper does. Runs exhaustive validation before completing.
#  If ANY check fails, exits non-zero — DO NOT submit training after failure.
#
#  Usage: sbatch slurm_jobs/run_data_prep.sh [num_facts] [seed]
#    num_facts: target samples per dataset (default: 1000)
#    seed:      random seed                (default: 42)
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=kug_data_prep
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/kug_data_prep_%j.out
#SBATCH --error=outputs/logs/kug_data_prep_%j.err

set -euo pipefail

NUM_FACTS="${1:-1000}"
SEED="${2:-42}"

echo "================================================================"
echo "  KUG Dataset Preparation (v4 — Paper-Faithful KG Triplets)"
echo "  num_facts per dataset: ${NUM_FACTS}"
echo "  seed:                  ${SEED}"
echo "  Host:                  $(hostname)"
echo "  Date:                  $(date)"
echo "  Job ID:                ${SLURM_JOB_ID}"
echo "================================================================"

WORK_DIR="/work/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/faster-sft"
cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

mkdir -p outputs/logs data/processed

export PYTHONUNBUFFERED=1
export PYTHONPATH="${WORK_DIR}"
export HF_HOME="${WORK_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

PYTHON="/users/jerryma/.conda/envs/torch2.8/bin/python"

[[ -f "${PYTHON}" ]] || { echo "ERROR: Python not found at ${PYTHON}"; exit 1; }
[[ -f "scripts/data_prep/prepare_kug_datasets.py" ]] || { echo "ERROR: prep script missing"; exit 1; }

# ── Verify raw KG files exist ─────────────────────────────────────────────
for DS in prime mag; do
    for FILE in edge_index.pt node_info.pkl edge_types.pt edge_type_dict.pkl node_types.pt node_type_dict.pkl; do
        F="data/raw/stark_${DS}_skb/extracted/processed/${FILE}"
        [[ -f "${F}" ]] || { echo "ERROR: Missing required KG file: ${F}"; exit 1; }
    done
    echo "  [${DS}] All 6 raw KG files verified."
done

# ── Run v4 data preparation (includes built-in validation) ────────────────
echo ""
echo "--- Running v4 dataset preparation ---"
${PYTHON} scripts/data_prep/prepare_kug_datasets.py \
    --num_facts "${NUM_FACTS}" \
    --seed "${SEED}"

RC=$?
if [[ $RC -ne 0 ]]; then
    echo ""
    echo "!!! DATASET PREPARATION FAILED (exit=$RC) — DO NOT PROCEED WITH TRAINING !!!"
    exit $RC
fi

echo ""
echo "--- Output files ---"
ls -lh data/processed/kug_dataset_*.jsonl

# ── Standalone second-pass validation ────────────────────────────────────
echo ""
echo "--- Second-pass standalone validation ---"
${PYTHON} - <<'PYEOF'
import json, sys
from pathlib import Path
from collections import Counter

path = Path("data/processed/kug_dataset_all.jsonl")
records = [json.loads(l) for l in path.open() if l.strip()]
print(f"Total records loaded: {len(records)}")

fails = []

# 1. Task distribution
tasks = Counter(r["task_type"] for r in records)
print(f"Task distribution: {dict(tasks)}")
for t in ["chaining", "intersection", "fact_checking"]:
    if tasks[t] == 0:
        fails.append(f"FAIL: No samples for '{t}'")

# 2. \nAnswer: separator
bad_mem = sum(1 for r in records if "\nAnswer:" not in r["p_mem"])
bad_gen = sum(1 for r in records if "\nAnswer:" not in r["p_gen"])
print(f"Missing \\nAnswer: in p_mem={bad_mem}, p_gen={bad_gen}")
if bad_mem: fails.append(f"FAIL: {bad_mem} records missing \\nAnswer: in p_mem")
if bad_gen: fails.append(f"FAIL: {bad_gen} records missing \\nAnswer: in p_gen")

# 3. FC balance
fc = [r for r in records if r["task_type"] == "fact_checking"]
fc_true  = sum(1 for r in fc if r["p_gen"].split("\nAnswer:")[-1].strip().lower() == "true")
fc_false = sum(1 for r in fc if r["p_gen"].split("\nAnswer:")[-1].strip().lower() == "false")
balance  = fc_true / max(1, fc_true + fc_false)
print(f"Fact-checking: TRUE={fc_true}, FALSE={fc_false}, balance={balance:.1%}")
if not (0.45 <= balance <= 0.55):
    fails.append(f"FAIL: FC balance {balance:.1%} not in 45-55%")
else:
    print("  PASS: FC balance OK")

# 4. CRITICAL: Intersection P_mem answer == P_gen answer
inter = [r for r in records if r["task_type"] == "intersection"]
bad = [r["id"] for r in inter
       if r["p_mem"].split("\nAnswer:")[-1].strip().lower()
       != r["p_gen"].split("\nAnswer:")[-1].strip().lower()]
print(f"Intersection answer match: {len(inter)-len(bad)}/{len(inter)} match")
if bad:
    fails.append(f"FAIL: {len(bad)}/{len(inter)} intersection P_mem!=P_gen: {bad[:3]}")
else:
    print("  PASS: All intersection P_mem==P_gen answers")

# 5. FC FALSE: distractor must not equal the true tail (exact match check)
import re as _re
fc_false_rec = [r for r in fc if r["p_gen"].split("\nAnswer:")[-1].strip().lower() == "false"]
leaks = []
for r in fc_false_rec[:100]:
    true_ans = r["target_entity"].lower()
    if len(true_ans) <= 3:
        continue
    gen_prompt = r["p_gen"].split("\nAnswer:")[0].lower()
    m = _re.search(r"relation with (.+?)\.", gen_prompt)
    distractor_used = m.group(1).strip() if m else ""
    if distractor_used == true_ans:
        leaks.append(r["id"])
print(f"FC FALSE distractor check: {len(leaks)}/100 have distractor==true tail (exact)")
if leaks:
    fails.append(f"FAIL: {len(leaks)}/100 FALSE FC records have distractor==true tail: {leaks[:3]}")
else:
    print("  PASS: FALSE FC distractors correct")

# 6. Print one sample per task type
print("\n--- Sample Records ---")
for task in ["chaining", "intersection", "fact_checking"]:
    r = next((x for x in records if x["task_type"] == task), None)
    if r:
        print(f"\n  [{task.upper()}]  id={r['id']}")
        print(f"  P_mem: {r['p_mem'][:200]!r}")
        print(f"  P_gen: {r['p_gen'][:200]!r}")

# Verdict
if fails:
    print("\n!!! VALIDATION FAILED !!!")
    for f in fails: print(f"  {f}")
    print("\n>>> DO NOT SUBMIT TRAINING <<<")
    sys.exit(1)
else:
    print("\n=== ALL VALIDATION CHECKS PASSED ===")
    print("Data is paper-faithful. Safe to proceed with training.")
PYEOF

RC=$?
if [[ $RC -ne 0 ]]; then
    echo ""
    echo "!!! SECOND-PASS VALIDATION FAILED — DO NOT SUBMIT TRAINING JOBS !!!"
    exit $RC
fi

echo ""
echo "================================================================"
echo "  ALL CHECKS PASSED — data is ready for training."
echo "  Next step: sbatch slurm_jobs/run_kug_train_v2.sh baseline 64 1"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "================================================================"
