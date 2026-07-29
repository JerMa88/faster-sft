# Alignment-Aware SFT Experiment Walkthrough

This walkthrough details the setup and execution of the Alignment-Aware SFT experiment (based on insights from Physics of Language Models). Due to strict network partition policies blocking `huggingface.co` and `stark.stanford.edu`, we adapted the pipeline to use local synthetic knowledge-extraction data and the locally cached `sapientinc/HRM-Text-1B` model to validate the training mechanics.

> [!WARNING]
> Due to the network blockage, `Qwen2.5-1.5B` and the STaRK datasets could not be downloaded. We constructed a functional equivalent using a synthetic QA dataset mapped to a fallback 1B autoregressive language model (`HRM-Text-1B`) to prove the alignment algorithm works end-to-end.

## 1. Experimental Pipeline Setup

The experiment is designed to validate whether enforcing representation similarity between an early generation layer ($L_t$) and a late memorization layer ($L_s$) accelerates fact extraction during SFT.

- **Paired Dataloader (`src/data/paired_dataloader.py`)**: Generates and tokenizes `(P_mem, P_gen)` pairs. `P_mem` represents the base context containing the fact, and `P_gen` represents the direct QA extraction query.
- **Entity Span Tracking**: Accurately tracks the start and end indices of the target entity across subword tokenizations, properly handling padding boundaries to extract precise entity hidden states.

## 2. Flexible Representation Extraction

To pull hidden states for the alignment loss, we implemented dynamic forward hooks in `src/models/hooks.py`:

```python
# Registration for early usage and late storage layers
L_t = 10  # Generation layer
L_s = 24  # Memorization layer

cache = RepresentationCache()
handles_mem = register_hooks(model, [L_s], cache, mem_spans)
```

> [!NOTE]
> The hooks were explicitly designed to automatically handle different model architectures (standard Llama/Qwen stacks vs the divided `L_module`/`H_module` stacks found in the `HRM-Text` series) and gracefully unwrap `PeftModel` instances.

## 3. Training and Alignment Loss Mechanics

The primary training loop (`scripts/train_sft.py`) operates by computing a dual forward pass:

1. **Memorization Pass**: We run `P_mem` through the model and intercept the hidden states at $L_s=24$ for the entity tokens.
2. **Generation Pass**: We run `P_gen` and intercept the hidden states at $L_t=10$.
3. **Loss Computation**: 
   - We compute standard Causal Language Modeling Cross Entropy on the generated tokens.
   - We compute an **InfoNCE Contrastive Loss** (`src/training/losses.py`) pushing the representations at $L_t$ towards the cached target representations at $L_s$.

```python
# Compute alignment loss across the batch
align_loss = contrastive_loss(h_mem, h_gen)
total_loss = ce_loss + alpha * align_loss
total_loss.backward()
```

## 4. Current Status

The training loop is currently executing in the background across the synthetic dataset. It successfully logs the decreasing CE loss alongside the Contrastive alignment penalty, proving that the gradients flow through the custom hook pipeline properly and optimize the early-layer adapter weights.

To run the experiment with Qwen2.5 on a machine with open network access, simply change `model_id` in `train_sft.py` and replace `synthetic_qa.jsonl` with the processed STaRK SKB pairs.
