import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def analyze():
    v1_path = ROOT / "outputs" / "final_evaluation_summary.json"
    v2_path = ROOT / "outputs" / "final_evaluation_summary_v2.json"

    if not v1_path.exists() or not v2_path.exists():
        print("Missing summary files.")
        return

    with open(v1_path) as f:
        v1_data = json.load(f)
    with open(v2_path) as f:
        v2_data = json.load(f)

    print(f"--- V1 vs V2 Comparison Summary ---")
    print(f"{'Model/Dataset':<30} | {'V1 Final A_gen':<15} | {'V2 Final A_gen':<15} | {'Rel Gain':<10}")
    print("-" * 75)

    for v2_key, v2_val in v2_data.items():
        v2_agen = v2_val.get("A_gen_final", 0.0)
        v2_amem = v2_val.get("A_mem_final", 0.0)
        run_dir = v2_val.get("run_dir", "")
        
        # Determine model key and dataset from run_dir
        parts = Path(run_dir).parts
        model_key = parts[-3]
        dataset = parts[-2]
        variant = parts[-1]

        v1_key = f"{model_key}_{dataset}"
        v1_val = v1_data.get(v1_key, {})
        v1_agen = v1_val.get("A_gen_final", 0.001)

        rel_gain = ((v2_agen - v1_agen) / max(v1_agen, 1e-4)) * 100.0
        print(f"{model_key} ({dataset}) [{variant.split('_')[1]}]: {v1_agen:.4f} -> {v2_agen:.4f} ({rel_gain:+.1f}%)")

if __name__ == "__main__":
    analyze()
