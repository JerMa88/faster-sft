"""
src/evaluation/evaluator.py
============================
Checkpoint evaluator — runs a trained model on P_mem and P_gen sets,
computes A_mem and A_gen, builds the accuracy-vs-epoch curve from a
run directory, and saves eval_results.json.

Usage (standalone):
    python -m src.evaluation.evaluator \
        --run_dir  outputs/runs/qwen3.5-1.5b/stark_prime/baseline-... \
        --model_id Qwen/Qwen3.5-1.5B \
        --data_path data/processed/stark_prime_qa.jsonl

Design:
  - Each training run saves checkpoints at epochs {1, 3, 5, 10, 15, 20, 30, 50}
  - Evaluator loads the checkpoint, uses model.generate() to produce
    multi-token answers, then compares via string_exact_match.
  - Works with both base models and PEFT/LoRA adapters (auto-detected).
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HF_CACHE = str(ROOT / "hf_cache")
os.environ.setdefault("HF_HOME",            HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE",  HF_CACHE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import (
    string_accuracy, strict_accuracy, strict_accuracy_with_indicators,
    relaxed_accuracy_with_indicators, relaxed_exact_match,
    accuracy_with_wilson_ci, convergence_epoch, auc_curve,
)



# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(
    checkpoint_path: str,
    base_model_id:   str,
    device:          torch.device,
    dtype:           torch.dtype,
    hf_cache:        str,
):
    """
    Load model from checkpoint_path.
    Auto-detects LoRA adapters (adapter_config.json present).
    """
    ckpt = Path(checkpoint_path)
    is_peft = (ckpt / "adapter_config.json").exists()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, cache_dir=hf_cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- MONKEY PATCH FOR NANBEIGE ---
    import transformers
    if hasattr(transformers, "DynamicCache"):
        if not hasattr(transformers.DynamicCache, "from_legacy_cache"):
            transformers.DynamicCache.from_legacy_cache = lambda past_key_values: transformers.DynamicCache()
        if not hasattr(transformers.DynamicCache, "to_legacy_cache"):
            transformers.DynamicCache.to_legacy_cache = lambda self: ()
    # ---------------------------------

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model_id if is_peft else str(ckpt), cache_dir=hf_cache, trust_remote_code=True)
    if "nanbeige" in base_model_id.lower() or "nanbeige" in str(ckpt).lower():
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            if "type" not in config.rope_scaling:
                config.rope_scaling["type"] = "linear"
            if "factor" not in config.rope_scaling:
                config.rope_scaling["factor"] = 1.0

    if is_peft:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft not installed — cannot load LoRA checkpoint.")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, config=config, cache_dir=hf_cache,
            torch_dtype=dtype,
            device_map="cuda" if device.type == "cuda" else "cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), config=config, cache_dir=hf_cache,
            torch_dtype=dtype,
            device_map="cuda" if device.type == "cuda" else "cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt), cache_dir=hf_cache, trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Generation-based evaluation (fixes the BPE token mismatch bug)
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_MARKER = "Answer:"

@torch.no_grad()
def _run_generative_eval(
    model,
    tokenizer,
    data_path:    str,
    device:       torch.device,
    dtype:        torch.dtype,
    kind:         str = "gen",   # "mem" or "gen"
    max_new_tokens: int = 32,
    batch_size:   int = 16,
) -> tuple[list[str], list[str]]:
    """
    Multi-token generative evaluation.

    For each example:
    1. Construct the prompt (P_mem or P_gen) up to "Answer: " (exclusive of entity)
    2. Use model.generate() to produce the answer
    3. Decode generated tokens → string
    4. Compare against target_entity string

    Returns (predictions, targets) both as lists of strings.
    """
    import json as _json

    # Read raw data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(_json.loads(line))

    all_preds   = []
    all_targets = []

    # Process in batches
    for batch_start in range(0, len(data), batch_size):
        batch_items = data[batch_start : batch_start + batch_size]

        prompts  = []
        targets  = []
        for item in batch_items:
            target_entity = item["target_entity"]
            targets.append(target_entity)

            if kind == "mem":
                # P_mem: "Context: {doc}\nQuery: What entity is this about?\nAnswer: "
                full_text = f"Context: {item['document']}\nQuery: What entity is this about?\nAnswer: "
            else:
                # P_gen: "Query: {query}\nAnswer: "
                full_text = f"Query: {item['query']}\nAnswer: "

            prompts.append(full_text)

def _manual_generate_item(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    """
    Per-example manual autoregressive generation with KV-caching.
    Avoids left-padding positional corruption for custom architectures (e.g. Nanbeige).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    generated = input_ids.clone()
    past_key_values = None
    curr_input_ids = input_ids

    for step in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
            try:
                outputs = model(input_ids=curr_input_ids, past_key_values=past_key_values, use_cache=True)
                past = getattr(outputs, "past_key_values", None)
                if past is not None:
                    past_key_values = past
                else:
                    past_key_values = None
            except Exception:
                outputs = model(input_ids=generated)
                past_key_values = None

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        curr_input_ids = next_token if past_key_values is not None else generated

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    gen_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


@torch.no_grad()
def _run_generative_eval(
    model,
    tokenizer,
    data_path:    str,
    device:       torch.device,
    dtype:        torch.dtype,
    kind:         str = "gen",   # "mem" or "gen"
    max_new_tokens: int = 32,
    batch_size:   int = 16,
) -> tuple[list[str], list[str]]:
    """
    Multi-token generative evaluation.
    Auto-detects custom models like Nanbeige and uses per-item zero-padding KV-cache generation.
    """
    import json as _json

    # Read raw data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(_json.loads(line))

    all_preds   = []
    all_targets = []

    # Check if model is Nanbeige or custom architecture requiring per-item generation
    is_custom_model = False
    model_name = str(type(model)).lower() + str(getattr(model, "name_or_path", "")).lower()
    if hasattr(model, "config"):
        model_name += str(getattr(model.config, "_name_or_path", "")).lower() + str(getattr(model.config, "model_type", "")).lower()
    if "nanbeige" in model_name or "antares" in model_name:
        # antares-1b: model.generate() silently hangs on some checkpoints;
        # use per-item manual KV-cache generation (same path as Nanbeige)
        is_custom_model = True

    if is_custom_model:
        # Route custom models through single-example zero-padding KV-cache generation
        for item in data:
            target_entity = item["target_entity"]
            all_targets.append(target_entity)
            if kind == "mem":
                full_text = f"Context: {item['document']}\nQuery: What entity is this about?\nAnswer: "
            else:
                full_text = f"Query: {item['query']}\nAnswer: "
            
            pred_text = _manual_generate_item(model, tokenizer, full_text, max_new_tokens, device, dtype)
            all_preds.append(pred_text)
        return all_preds, all_targets

    # Process standard models in batches
    for batch_start in range(0, len(data), batch_size):
        batch_items = data[batch_start : batch_start + batch_size]

        prompts  = []
        targets  = []
        for item in batch_items:
            target_entity = item["target_entity"]
            targets.append(target_entity)

            if kind == "mem":
                full_text = f"Context: {item['document']}\nQuery: What entity is this about?\nAnswer: "
            else:
                full_text = f"Query: {item['query']}\nAnswer: "

            prompts.append(full_text)

        # Tokenize prompts (left-pad for generation)
        tokenizer.padding_side = "left"
        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480,  # leave room for generation
        ).to(device)

        try:
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(device.type == "cuda")):
                gen_ids = model.generate(
                    **encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = encodings["input_ids"].shape[1]
            for b_idx in range(len(batch_items)):
                generated_tokens = gen_ids[b_idx, prompt_len:]
                pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                all_preds.append(pred_text)
                all_targets.append(targets[b_idx])

        except Exception as gen_err:
            # Fallback for individual items in batch if model.generate fails
            for b_idx, prompt in enumerate(prompts):
                pred_text = _manual_generate_item(model, tokenizer, prompt, max_new_tokens, device, dtype)
                all_preds.append(pred_text)
                all_targets.append(targets[b_idx])

    return all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# Single-checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    base_model_id:   str,
    data_path:       str,
    device:          torch.device,
    dtype:           torch.dtype,
    batch_size:      int = 16,
    hf_cache:        str = "./hf_cache",
    verbose:         bool = True,
) -> dict:
    """
    Evaluate a single checkpoint using multi-token generation + string match.
    Returns metrics dict with A_mem and A_gen.
    """
    if verbose:
        print(f"  Evaluating: {Path(checkpoint_path).name} …")

    model, tokenizer = _load_model(
        checkpoint_path, base_model_id, device, dtype, hf_cache
    )

    mem_preds: list = []
    mem_targets: list = []
    gen_preds: list = []
    gen_targets: list = []
    try:
        # Generative evaluation
        mem_preds, mem_targets = _run_generative_eval(
            model, tokenizer, data_path, device, dtype,
            kind="mem", batch_size=batch_size,
        )
        gen_preds, gen_targets = _run_generative_eval(
            model, tokenizer, data_path, device, dtype,
            kind="gen", batch_size=batch_size,
        )
    finally:
        # Always free GPU memory — even if a CUDA error is thrown mid-eval.
        # Without this, a CUDA crash leaves the device in a broken state and
        # every subsequent run in the same process will fail immediately.
        del model
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass  # GPU already in bad state; best-effort cleanup

    if not mem_preds:
        raise RuntimeError("No predictions generated — eval aborted (CUDA crash or empty dataset).")

    # ── Primary metric: relaxed EM (gold ⊆ pred, unidirectional substring) ──
    # Follows Mem2Gen / KUG paper convention; appropriate for multi-word STARK
    # paper titles where the model often generates the correct title followed by
    # trailing tokens that strict EM would unfairly penalize.
    a_mem_relaxed, mem_indicators = relaxed_accuracy_with_indicators(mem_preds, mem_targets)
    a_gen_relaxed, gen_indicators = relaxed_accuracy_with_indicators(gen_preds, gen_targets)

    # ── Secondary metric: strict EM (no substring, ACL conservative bound) ──
    a_mem_strict, mem_strict_ind = strict_accuracy_with_indicators(mem_preds, mem_targets)
    a_gen_strict, gen_strict_ind = strict_accuracy_with_indicators(gen_preds, gen_targets)

    if verbose:
        print(f"    Relaxed: A_mem={a_mem_relaxed:.3f}  A_gen={a_gen_relaxed:.3f}  [PRIMARY — gold-in-pred]")
        print(f"    Strict : A_mem={a_mem_strict:.3f}  A_gen={a_gen_strict:.3f}  [secondary — exact]")
        # Spot-check using relaxed indicators
        for i in range(min(3, len(mem_preds))):
            match_sym = "✓" if mem_indicators[i] else "✗"
            print(f"      [{match_sym}] pred='{mem_preds[i][:50]}' "
                  f"target='{mem_targets[i][:50]}'")

    return {
        "checkpoint":     str(checkpoint_path),
        # Relaxed EM (PRIMARY — gold-in-pred substring, follows KUG/Mem2Gen)
        "A_mem":          round(a_mem_relaxed, 4),
        "A_gen":          round(a_gen_relaxed, 4),
        # Strict EM (SECONDARY — conservative ACL bound)
        "A_mem_strict":   round(a_mem_strict, 4),
        "A_gen_strict":   round(a_gen_strict, 4),
        # Per-instance correctness vectors for McNemar test (relaxed, primary)
        "mem_correct":    mem_indicators,
        "gen_correct":    gen_indicators,
        # Per-instance correctness vectors (strict, for sensitivity analysis)
        "mem_correct_strict": mem_strict_ind,
        "gen_correct_strict": gen_strict_ind,
        "n_examples":     len(mem_preds),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Full run evaluation (all checkpoints → accuracy curve)
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_EPOCHS = [1, 3, 5, 10, 15, 20, 30, 50]


def evaluate_run(
    run_dir:       str,
    base_model_id: str,
    data_path:     str,
    device:        torch.device,
    dtype:         torch.dtype,
    batch_size:    int   = 16,
    hf_cache:      str   = "./hf_cache",
    threshold:     float = 0.95,
    verbose:       bool  = True,
) -> dict:
    """
    Evaluate all saved checkpoints in a run directory.
    Returns a summary dict with the full accuracy curve and convergence metrics.
    """
    run_dir  = Path(run_dir)
    checkpts = sorted(
        [d for d in run_dir.glob("checkpoint_epoch*") if d.is_dir()],
        key=lambda d: int(d.name.replace("checkpoint_epoch", ""))
    )

    if not checkpts:
        raise FileNotFoundError(f"No checkpoint_epoch* directories in {run_dir}")

    if verbose:
        print(f"\n  Run: {run_dir.name}")
        print(f"  Found {len(checkpts)} checkpoints: "
              f"{[c.name for c in checkpts]}")

    epoch_results = []
    cuda_dead = False  # track if GPU is in a bad state
    for ckpt in checkpts:
        epoch_num = int(ckpt.name.replace("checkpoint_epoch", ""))
        if cuda_dead:
            print(f"    [SKIP] checkpoint_epoch{epoch_num} — GPU in bad state, skipping.")
            continue
        try:
            result = evaluate_checkpoint(
                str(ckpt), base_model_id, data_path,
                device, dtype, batch_size, hf_cache, verbose
            )
            result["epoch"] = epoch_num
            epoch_results.append(result)
        except RuntimeError as e:
            err_str = str(e)
            print(f"    [ERROR] checkpoint_epoch{epoch_num}: {err_str[:200]}")
            if "CUDA" in err_str or "cuda" in err_str:
                # Best-effort GPU state reset so the next run can start fresh
                print("    [INFO] CUDA error detected — attempting device reset…")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
                # If the GPU is truly dead, mark it so we skip remaining checkpoints
                # but let the outer loop continue to the next *run* (different model load)
                cuda_dead = True
            # Continue to next checkpoint (or break this run); do not propagate


    if not epoch_results:
        raise RuntimeError(
            f"All checkpoints failed for run {run_dir.name}. "
            "Check CUDA errors above."
        )

    # Build accuracy curves
    epochs = [r["epoch"] for r in epoch_results]

    # ── Relaxed EM curves (PRIMARY — follows KUG/Mem2Gen) ──
    a_mem_curve         = [r["A_mem"]        for r in epoch_results]
    a_gen_curve         = [r["A_gen"]        for r in epoch_results]

    # ── Strict EM curves (SECONDARY — conservative ACL bound) ──
    a_mem_curve_strict  = [r["A_mem_strict"] for r in epoch_results]
    a_gen_curve_strict  = [r["A_gen_strict"] for r in epoch_results]

    # Final epoch correctness indicators
    final_result         = epoch_results[-1]
    mem_correct_relaxed  = final_result["mem_correct"]         # relaxed
    gen_correct_relaxed  = final_result["gen_correct"]         # relaxed
    mem_correct_strict   = final_result.get("mem_correct_strict", mem_correct_relaxed)
    gen_correct_strict   = final_result.get("gen_correct_strict", gen_correct_relaxed)

    # Wilson CIs for final epoch (both relaxed and strict)
    mem_ci_relaxed = accuracy_with_wilson_ci(mem_correct_relaxed)
    gen_ci_relaxed = accuracy_with_wilson_ci(gen_correct_relaxed)
    mem_ci_strict  = accuracy_with_wilson_ci(mem_correct_strict)
    gen_ci_strict  = accuracy_with_wilson_ci(gen_correct_strict)

    # Convergence on relaxed A_gen (primary)
    t_conv         = convergence_epoch(a_gen_curve,        threshold, start_epoch=epochs[0])
    auc            = auc_curve(a_gen_curve)

    # Convergence on strict A_gen (secondary)
    t_conv_strict  = convergence_epoch(a_gen_curve_strict, threshold, start_epoch=epochs[0])
    auc_strict     = auc_curve(a_gen_curve_strict)

    # Baseline gate: A_mem (relaxed, primary) >= 0.978 at epoch 3
    # Using relaxed EM consistent with primary metric convention
    gate_result = None
    for r in epoch_results:
        if r["epoch"] == 3:
            gate_result = r["A_mem"] >= 0.978
            if verbose:
                status = "✅ PASS" if gate_result else "❌ FAIL"
                print(f"\n  Baseline gate (A_mem_relaxed @ epoch3 >= 0.978): "
                      f"{r['A_mem']:.3f} → {status}")

    # Check for per-task-type breakdown (chaining vs intersection) in dataset
    task_type_breakdown = {}
    try:
        data_items = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data_items.append(json.loads(line))

        task_types = [item.get("task_type") for item in data_items]
        if any(tt in ("chaining", "intersection") for tt in task_types):
            for tt in ("chaining", "intersection"):
                indices = [i for i, t in enumerate(task_types) if t == tt]
                if indices:
                    sub_mem = [mem_correct_relaxed[i] for i in indices]
                    sub_gen = [gen_correct_relaxed[i] for i in indices]
                    mem_stats = accuracy_with_wilson_ci(sub_mem)
                    gen_stats = accuracy_with_wilson_ci(sub_gen)
                    task_type_breakdown[tt] = {
                        "n_examples": len(indices),
                        "A_mem_strict": mem_stats["acc"],
                        "A_mem_strict_ci": [mem_stats["ci_lo"], mem_stats["ci_hi"]],
                        "A_gen_strict": gen_stats["acc"],
                        "A_gen_strict_ci": [gen_stats["ci_lo"], gen_stats["ci_hi"]],
                        "mem_correct": sub_mem,
                        "gen_correct": sub_gen,
                    }
    except Exception as tt_err:
        if verbose:
            print(f"  [NOTE] Task-type breakdown skipped: {tt_err}")

    summary = {
        "run_dir":            str(run_dir),
        "base_model_id":      base_model_id,
        "data_path":          data_path,
        "epochs":             epochs,
        # ── Relaxed EM (PRIMARY — gold-in-pred, follows KUG/Mem2Gen) ──
        "A_mem_curve":        [round(a, 4) for a in a_mem_curve],
        "A_gen_curve":        [round(a, 4) for a in a_gen_curve],
        "A_mem_final":        round(a_mem_curve[-1], 4),
        "A_gen_final":        round(a_gen_curve[-1], 4),
        "A_mem_ci":           [round(mem_ci_relaxed["ci_lo"], 4), round(mem_ci_relaxed["ci_hi"], 4)],
        "A_gen_ci":           [round(gen_ci_relaxed["ci_lo"], 4), round(gen_ci_relaxed["ci_hi"], 4)],
        "T_conv":             t_conv,
        "AUC":                round(auc, 4),
        # ── Strict EM (SECONDARY — exact match, conservative ACL bound) ──
        "A_mem_curve_strict": [round(a, 4) for a in a_mem_curve_strict],
        "A_gen_curve_strict": [round(a, 4) for a in a_gen_curve_strict],
        "A_mem_strict_final": round(a_mem_curve_strict[-1], 4),
        "A_gen_strict_final": round(a_gen_curve_strict[-1], 4),
        "A_mem_strict_ci":    [round(mem_ci_strict["ci_lo"], 4), round(mem_ci_strict["ci_hi"], 4)],
        "A_gen_strict_ci":    [round(gen_ci_strict["ci_lo"], 4), round(gen_ci_strict["ci_hi"], 4)],
        "T_conv_strict":      t_conv_strict,
        "AUC_strict":         round(auc_strict, 4),
        "threshold":          threshold,
        "baseline_gate":      gate_result,
        # Final epoch correctness vectors (for McNemar significance testing)
        "mem_correct_final":  mem_correct_relaxed,
        "gen_correct_final":  gen_correct_relaxed,
        # Per-task-type breakdown (if task_type present in data)
        "task_types":         task_type_breakdown,
        "per_epoch":          epoch_results,
    }

    # Save
    out_path = run_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"\n  Saved → {out_path}")
        print(f"  Strict:  A_gen_final={summary['A_gen_strict_final']:.3f}  "
              f"T_conv={t_conv_strict}  AUC={auc_strict:.3f}")
        print(f"  Relaxed: A_gen_final={summary['A_gen_final']:.3f}  "
              f"T_conv={t_conv}  AUC={auc:.3f}")


    return summary



# ─────────────────────────────────────────────────────────────────────────────
# Cross-run comparison (for the paper table)
# ─────────────────────────────────────────────────────────────────────────────

def compare_runs(
    run_dirs:      list[str],
    base_model_id: str,
    data_path:     str,
    device:        torch.device,
    dtype:         torch.dtype,
    out_path:      str = "outputs/comparison.json",
    **eval_kwargs,
) -> dict:
    """
    Evaluate multiple run directories and produce a comparison table.
    Saves to out_path. Returns a dict keyed by run_dir name.
    """
    comparison = {}
    for run_dir in run_dirs:
        name = Path(run_dir).name
        try:
            summary = evaluate_run(
                run_dir, base_model_id, data_path, device, dtype, **eval_kwargs
            )
            comparison[name] = {
                "A_gen_final": summary["A_gen_final"],
                "T_conv":      summary["T_conv"],
                "AUC":         summary["AUC"],
                "A_mem_final": summary["A_mem_final"],
            }
        except Exception as e:
            comparison[name] = {"error": str(e)}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved → {out_path}")
    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT run checkpoints")
    parser.add_argument("--run_dir",    required=True,
                        help="Path to a run directory (outputs/runs/.../)")
    parser.add_argument("--model_id",   required=True,
                        help="Base HF model ID (for tokenizer + PEFT base)")
    parser.add_argument("--data_path",  default="data/processed/stark_prime_qa.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--threshold",  type=float, default=0.95)
    parser.add_argument("--hf_cache",   default="./hf_cache")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    evaluate_run(
        args.run_dir, args.model_id, args.data_path,
        device, dtype, args.batch_size, args.hf_cache, args.threshold,
        verbose=True,
    )


if __name__ == "__main__":
    main()
