# Alignment-Aware SFT Handoff Document

This document summarizes the current state of the Alignment-Aware SFT experiment and provides instructions for resuming the work on a more capable machine with full network access and GPUs.

## Current Project State

We have completed the implementation of the core training pipeline and validated it end-to-end on CPU using a mocked configuration (synthetic data + local cache model). 

### Accomplished Implementation Details:
1. **Dynamic Hooks (`src/models/hooks.py`)**: 
   - A robust mechanism to extract hidden states from arbitrary layers during the forward pass.
   - Designed to handle both standard architectures (Llama/Qwen `model.layers`) and dual-stack architectures (like the `sapientinc/HRM-Text-1B` we used as a fallback).
   - Accurately handles `PeftModel` un-wrapping for LoRA adapters.
   - Extracts mean-pooled states dynamically from `entity_span` slices.

2. **Dataloader (`src/data/paired_dataloader.py`)**:
   - Tokenizes the `(P_mem, P_gen)` mapping format.
   - Includes logic to precisely trace back from padding tokens to compute exact `entity_span` indices across disparate tokenizations.

3. **Loss Mechanics (`src/training/losses.py`)**:
   - InfoNCE Contrastive alignment penalty implemented, pulling representations of extracted entities in early layers toward their representations in late memorization layers.
   - Includes a fallback cosine similarity distance metric (`rep_distill_loss`).

4. **Training Loop (`scripts/train_sft.py`)**:
   - The dual forward pass is fully implemented. It orchestrates the `RepresentationCache`, computes Causal LM loss on `P_gen`, and injects the alignment penalty through backpropagation.

## Setup on the New Machine

When you transition to a machine with an active GPU and internet connection:

### 1. Re-target the Model
In `tests/test_hooks.py` and `scripts/train_sft.py`, we temporarily set the `model_id` to `sapientinc/HRM-Text-1B` to bypass the `[Errno 49] Can't assign requested address` huggingface.co block.
- Change `model_id` back to `"Qwen/Qwen2.5-1.5B"`.
- Remove `local_files_only=True` if you are downloading the weights for the first time.
- Switch `device_map="cpu"` to `device_map="cuda"` (or `"auto"`).

### 2. Fetch the True STaRK Datasets
Due to the network block on S3/HF Hub, the data processing step was halted, and we generated 2,000 synthetic pairs (`data/processed/synthetic_qa.jsonl`).
- Run `scripts/prepare_data.py` (you may need to revert the forced `curl -4` hacks depending on your new machine's DNS configuration).
- The script should now naturally fetch the STaRK-Prime and STaRK-MAG datasets via standard Hugging Face/S3 avenues.
- Point `data_path` in `scripts/train_sft.py` to the actual processed STaRK `.jsonl` files instead of the synthetic ones.

### 3. Run Layer Profiling
We hardcoded $l_s = 24$ and $l_t = 10$ for the 32-layer HRM-Text model.
- You should execute a full sweep (Phase 1.5 in `task.md`) to plot the Logit-Lens KL divergence and Linear Probe accuracy over the 28 layers of Qwen2.5-1.5B to rigorously select $l_s$ and $l_t$.

### 4. Hardware Optimization
- With GPUs available, ensure `bfloat16` or `float16` precision is added to the `AutoModelForCausalLM.from_pretrained` call in `train_sft.py`.
- Adjust `batch_size` in the dataloader from the CPU-friendly `2` to whatever saturates your GPU VRAM.

## Repository Layout
- `scripts/`: Execution scripts for data prep, training (`train_sft.py`), and synthetic data generation.
- `src/models/`: Contains the critical `hooks.py` that power the extraction logic.
- `src/data/`: `paired_dataloader.py`.
- `src/training/`: `losses.py`.
- `tests/`: Basic validation scripts for the hooks.
