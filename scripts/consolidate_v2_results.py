import os
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def consolidate():
    eval_files = glob.glob(str(ROOT / "outputs" / "runs_v2" / "*" / "*" / "*" / "eval_results.json"))
    print(f"Found {len(eval_files)} evaluated run results in runs_v2.")

    summary = {}
    for ef in eval_files:
        path = Path(ef)
        run_slug = path.parent.name
        dataset_name = path.parents[0].name
        model_key = path.parents[1].name
        
        with open(ef, "r") as f:
            data = json.load(f)
            
        key = f"{model_key}_{dataset_name}_{run_slug}"
        summary[key] = data

    out_file = ROOT / "outputs" / "final_evaluation_summary_v2.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Consolidated summary saved to {out_file}")

if __name__ == "__main__":
    consolidate()
