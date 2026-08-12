import os
import json
import wandb
import matplotlib.pyplot as plt

def fetch_run_data(run_id, entity="jerma88-smu", project="kug_overhaul_qwen1.5b"):
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        return history
    except Exception as e:
        print(f"Error fetching run {run_id}: {e}")
        return None

def main():
    runs = {
        "baseline": "gn15thp3",
        "two_stage": "k9ojf5au",
        "joint": "kq47ojz9"
    }

    results = {}
    for method, run_id in runs.items():
        print(f"Fetching W&B data for {method} (Run ID: {run_id})...")
        df = fetch_run_data(run_id)
        if df is not None:
            # Filter rows where epoch and eval metrics exist
            eval_cols = [c for c in df.columns if c.startswith("eval/")]
            print(f"  Columns found: {eval_cols}")
            print(f"  Total rows: {len(df)}")
            results[method] = df

if __name__ == "__main__":
    main()
