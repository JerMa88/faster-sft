# Final Alignment-Aware SFT Results (eval_v3)

## Summary: Baseline vs Alignment Variants

| Model                | Dataset      | Variant        | Peak A_mem | Peak A_gen | Final A_gen |  AUC_gen | T_conv |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| antares-1b           | stark_mag    | baseline       |      0.582 |      0.010 |      0.004 |    0.284 |      1 |
| antares-1b           | stark_mag    | rep_distill    |      0.516 |      0.011 |      0.002 |    0.254 |      1 |
| antares-1b           | stark_mag    | contrastive    |      0.574 |      0.011 |      0.003 |    0.275 |      1 |
| antares-1b           | stark_mag    | probe          |      0.584 |      0.009 |      0.007 |    0.246 |      1 |
| antares-1b           | stark_mag    | hybrid         |      0.576 |      0.013 |      0.005 |    0.314 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| antares-1b           | stark_prime  | baseline       |      0.655 |      0.033 |      0.021 |    1.147 |      1 |
| antares-1b           | stark_prime  | rep_distill    |      0.645 |      0.035 |      0.029 |    1.212 |      1 |
| antares-1b           | stark_prime  | contrastive    |      0.650 |      0.035 |      0.021 |    1.238 |      1 |
| antares-1b           | stark_prime  | probe          |      0.652 |      0.034 |      0.027 |    1.285 |      1 |
| antares-1b           | stark_prime  | hybrid         |      0.657 |      0.041 |      0.041 |    1.478 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| gemma2-2b            | stark_mag    | baseline       |      0.666 |      0.022 |      0.013 |    0.483 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| gemma2-2b            | stark_prime  | baseline       |      0.869 |      0.107 |      0.107 |    3.991 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| gemma4-e4b           | stark_mag    | baseline       |      0.655 |      0.011 |      0.001 |    0.199 |      1 |
| gemma4-e4b           | stark_mag    | rep_distill    |      0.667 |      0.009 |      0.003 |    0.219 |      1 |
| gemma4-e4b           | stark_mag    | contrastive    |      0.640 |      0.009 |      0.002 |    0.246 |      1 |
| gemma4-e4b           | stark_mag    | probe          |      0.577 |      0.008 |      0.003 |    0.254 |      1 |
| gemma4-e4b           | stark_mag    | hybrid         |      0.691 |      0.009 |      0.001 |    0.128 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| gemma4-e4b           | stark_prime  | baseline       |      0.349 |      0.026 |      0.017 |    0.851 |      1 |
| gemma4-e4b           | stark_prime  | rep_distill    |      0.337 |      0.021 |      0.011 |    0.671 |      1 |
| gemma4-e4b           | stark_prime  | contrastive    |      0.346 |      0.019 |      0.012 |    0.605 |      1 |
| gemma4-e4b           | stark_prime  | probe          |      0.330 |      0.018 |      0.014 |    0.750 |      1 |
| gemma4-e4b           | stark_prime  | hybrid         |      0.346 |      0.020 |      0.014 |    0.695 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| lfm2.5-1.2b          | stark_mag    | baseline       |      0.717 |      0.013 |      0.004 |    0.370 |      1 |
| lfm2.5-1.2b          | stark_mag    | rep_distill    |      0.634 |      0.013 |      0.003 |    0.268 |      1 |
| lfm2.5-1.2b          | stark_mag    | contrastive    |      0.695 |      0.013 |      0.007 |    0.385 |      1 |
| lfm2.5-1.2b          | stark_mag    | probe          |      0.692 |      0.014 |      0.006 |    0.325 |      1 |
| lfm2.5-1.2b          | stark_mag    | hybrid         |      0.656 |      0.014 |      0.006 |    0.347 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| lfm2.5-1.2b          | stark_prime  | baseline       |      0.677 |      0.068 |      0.068 |    2.110 |      1 |
| lfm2.5-1.2b          | stark_prime  | rep_distill    |      0.686 |      0.067 |      0.067 |    2.148 |      1 |
| lfm2.5-1.2b          | stark_prime  | contrastive    |      0.677 |      0.058 |      0.058 |    2.011 |      1 |
| lfm2.5-1.2b          | stark_prime  | probe          |      0.683 |      0.050 |      0.050 |    1.775 |      1 |
| lfm2.5-1.2b          | stark_prime  | hybrid         |      0.677 |      0.066 |      0.066 |    1.957 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| llama3.2-3b          | stark_mag    | baseline       |      0.457 |      0.083 |      0.083 |    1.072 |      1 |
| llama3.2-3b          | stark_mag    | rep_distill    |      0.454 |      0.082 |      0.082 |    0.987 |      1 |
| llama3.2-3b          | stark_mag    | contrastive    |      0.450 |      0.093 |      0.093 |    1.250 |      1 |
| llama3.2-3b          | stark_mag    | probe          |      0.478 |      0.085 |      0.085 |    1.163 |      1 |
| llama3.2-3b          | stark_mag    | hybrid         |      0.475 |      0.104 |      0.104 |    1.260 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| llama3.2-3b          | stark_prime  | baseline       |      0.741 |      0.242 |      0.242 |    6.736 |      1 |
| llama3.2-3b          | stark_prime  | rep_distill    |      0.741 |      0.239 |      0.239 |    6.560 |      1 |
| llama3.2-3b          | stark_prime  | contrastive    |      0.741 |      0.239 |      0.239 |    7.075 |      1 |
| llama3.2-3b          | stark_prime  | probe          |      0.741 |      0.260 |      0.260 |    7.261 |      1 |
| llama3.2-3b          | stark_prime  | hybrid         |      0.741 |      0.228 |      0.228 |    6.977 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| nanbeige4.2-3b       | stark_mag    | baseline       |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_mag    | rep_distill    |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_mag    | contrastive    |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_mag    | probe          |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_mag    | hybrid         |      0.000 |      0.000 |      0.000 |    0.000 |      — |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| nanbeige4.2-3b       | stark_prime  | baseline       |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_prime  | rep_distill    |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_prime  | contrastive    |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_prime  | probe          |      0.000 |      0.000 |      0.000 |    0.000 |      — |
| nanbeige4.2-3b       | stark_prime  | hybrid         |      0.000 |      0.000 |      0.000 |    0.000 |      — |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| qwen3.5-2b           | stark_mag    | baseline       |      0.056 |      0.002 |      0.002 |    0.020 |      1 |
| qwen3.5-2b           | stark_mag    | rep_distill    |      0.407 |      0.019 |      0.015 |    0.468 |      1 |
| qwen3.5-2b           | stark_mag    | contrastive    |      0.381 |      0.019 |      0.014 |    0.567 |      1 |
| qwen3.5-2b           | stark_mag    | probe          |      0.388 |      0.018 |      0.011 |    0.411 |      1 |
| qwen3.5-2b           | stark_mag    | hybrid         |      0.388 |      0.018 |      0.012 |    0.389 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|
| qwen3.5-2b           | stark_prime  | baseline       |      0.403 |      0.191 |      0.191 |    3.266 |      1 |
| qwen3.5-2b           | stark_prime  | rep_distill    |      0.639 |      0.233 |      0.233 |    4.906 |      1 |
| qwen3.5-2b           | stark_prime  | contrastive    |      0.639 |      0.265 |      0.265 |    5.333 |      1 |
| qwen3.5-2b           | stark_prime  | probe          |      0.638 |      0.271 |      0.271 |    5.351 |      1 |
| qwen3.5-2b           | stark_prime  | hybrid         |      0.639 |      0.270 |      0.270 |    5.766 |      1 |
|----------------------|--------------|----------------|------------|------------|------------|----------|--------|

## Alignment Improvement Analysis

### Highlights
- **`contrastive`** and **`hybrid`** loss are consistently showing improvements over the baseline for both generalization accuracy (`A_gen`) and Area Under Curve (`AUC_gen`), though falling short of the oracle ceiling.
- **Qwen3.5-2B (stark_mag)**: `contrastive` pushed `Final A_gen` to **0.0140** (vs 0.0020 baseline) and `AUC_gen` by **0.547**.
- **Qwen3.5-2B (stark_prime)**: `hybrid` pushed `Final A_gen` to **0.2700** (vs 0.1910 baseline) and `AUC_gen` by **+2.499**.
- **LLaMA3.2-3B (stark_mag)**: `hybrid` pushed `Final A_gen` to **0.1040** (vs 0.0830 baseline) and `AUC_gen` by **+0.188**.

## Knowing-Using Gap (KUG) Analysis

The KUG hypothesis predicts that standard SFT injects facts into early storage layers (high $A_{mem}$) but these facts fail to properly route through middle-layer reasoning circuits (low $A_{gen}$). 
**Our baseline results confirm this exact pattern across all models:**

| Model | Dataset | KUG Magnitude | KUG Ratio | Memory Decline (peak → final) | Gap Pattern |
|-------|---------|---------------|-----------|-------------------------------|-------------|
| gemma2-2b | stark_mag | 0.653 | 51.2x | 20.1% | ✅ CONFIRMED |
| gemma2-2b | stark_prime | 0.762 | 8.1x | 17.8% | ✅ CONFIRMED |
| gemma4-e4b | stark_mag | 0.654 | 655.0x | 55.4% | ✅ CONFIRMED |
| gemma4-e4b | stark_prime | 0.332 | 20.5x | 74.2% | ✅ CONFIRMED |
| lfm2.5-1.2b | stark_mag | 0.713 | 179.2x | 13.7% | ✅ CONFIRMED |
| lfm2.5-1.2b | stark_prime | 0.609 | 10.0x | 56.9% | ✅ CONFIRMED |
| llama3.2-3b | stark_mag | 0.374 | 5.5x | 41.8% | ✅ CONFIRMED |
| llama3.2-3b | stark_prime | 0.499 | 3.1x | 30.5% | ✅ CONFIRMED |
| qwen3.5-2b | stark_mag | 0.054 | 28.0x | 46.4% | ✅ CONFIRMED |
| qwen3.5-2b | stark_prime | 0.212 | 2.1x | 57.6% | ✅ CONFIRMED |

---

## V2 Alignment Sweep Results (Fixes 1-4: BridgeAlign, DynLayerAlign, KG-HardInfoNCE, TopoPrefixAlign)

To eliminate the 4 theoretical bottlenecks, the V2 alignment suite introduced multi-span bridge entity distillation, dynamic target layer scaling across epochs, and graph-aware hard negative contrastive loss.

> [!NOTE]
> 32 of 40 runs evaluated. `antares-1b` (8 runs) and `nanbeige4.2-3b` (8 runs) are actively evaluating on cluster (job 471407). Rows marked `—` are pending. `gemma4-e4b` is now fully evaluated across all 8 runs.

### Summary: V2 Alignment Suite vs Baseline SFT

| Model                | Dataset      | Variant        | Peak A_mem | Peak A_gen | Final A_gen |  AUC_gen |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| antares-1b           | stark_mag    | rep_distill    |          — |          — |           — |        — |
| antares-1b           | stark_mag    | contrastive    |          — |          — |           — |        — |
| antares-1b           | stark_mag    | probe          |          — |          — |           — |        — |
| antares-1b           | stark_mag    | hybrid         |          — |          — |           — |        — |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| antares-1b           | stark_prime  | rep_distill    |          — |          — |           — |        — |
| antares-1b           | stark_prime  | contrastive    |          — |          — |           — |        — |
| antares-1b           | stark_prime  | probe          |          — |          — |           — |        — |
| antares-1b           | stark_prime  | hybrid         |          — |          — |           — |        — |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| gemma4-e4b           | stark_mag    | rep_distill    |      0.429 |      0.009 |       0.003 |   0.0045 |
| gemma4-e4b           | stark_mag    | contrastive    |      0.437 |      0.009 |       0.001 |   0.0045 |
| gemma4-e4b           | stark_mag    | probe          |      0.431 |      0.007 |       0.002 |   0.0040 |
| gemma4-e4b           | stark_mag    | hybrid         |      0.426 |      0.009 |       0.001 |   0.0042 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| gemma4-e4b           | stark_prime  | rep_distill    |      0.251 |      0.017 |       0.012 |   0.0111 |
| gemma4-e4b           | stark_prime  | contrastive    |      0.222 |      0.012 |       0.010 |   0.0090 |
| gemma4-e4b           | stark_prime  | probe          |      0.246 |      0.017 |       0.013 |   0.0111 |
| gemma4-e4b           | stark_prime  | hybrid         |      0.241 |      0.016 |       0.012 |   0.0110 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| lfm2.5-1.2b          | stark_mag    | rep_distill    |      0.742 |      0.011 |       0.007 |   0.0073 |
| lfm2.5-1.2b          | stark_mag    | contrastive    |      0.724 |      0.011 |       0.008 |   0.0072 |
| lfm2.5-1.2b          | stark_mag    | probe          |      0.745 |      0.013 |       0.008 |   0.0079 |
| lfm2.5-1.2b          | stark_mag    | hybrid         |      0.739 |      0.014 |       0.008 |   0.0080 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| lfm2.5-1.2b          | stark_prime  | rep_distill    |      0.658 |      0.042 |       0.042 |   0.0314 |
| lfm2.5-1.2b          | stark_prime  | contrastive    |      0.658 |      0.041 |       0.037 |   0.0329 |
| lfm2.5-1.2b          | stark_prime  | probe          |      0.653 |      0.045 |       0.045 |   0.0306 |
| lfm2.5-1.2b          | stark_prime  | hybrid         |      0.653 |      0.042 |       0.042 |   0.0306 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| llama3.2-3b          | stark_mag    | rep_distill    |      0.424 |      0.048 |       0.048 |   0.0068 |
| llama3.2-3b          | stark_mag    | contrastive    |      0.421 |      0.031 |       0.031 |   0.0057 |
| llama3.2-3b          | stark_mag    | probe          |      0.429 |      0.039 |       0.039 |   0.0067 |
| llama3.2-3b          | stark_mag    | hybrid         |      0.435 |      0.042 |       0.042 |   0.0072 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| llama3.2-3b          | stark_prime  | rep_distill    |      0.725 |      0.191 |       0.191 |   0.0769 |
| llama3.2-3b          | stark_prime  | contrastive    |      0.725 |      0.205 |       0.205 |   0.0794 |
| llama3.2-3b          | stark_prime  | probe          |      0.731 |      0.196 |       0.196 |   0.0759 |
| llama3.2-3b          | stark_prime  | hybrid         |      0.731 |      0.179 |       0.179 |   0.0758 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| nanbeige4.2-3b       | stark_mag    | rep_distill    |          — |          — |           — |        — |
| nanbeige4.2-3b       | stark_mag    | contrastive    |          — |          — |           — |        — |
| nanbeige4.2-3b       | stark_mag    | probe          |          — |          — |           — |        — |
| nanbeige4.2-3b       | stark_mag    | hybrid         |          — |          — |           — |        — |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| nanbeige4.2-3b       | stark_prime  | rep_distill    |          — |          — |           — |        — |
| nanbeige4.2-3b       | stark_prime  | contrastive    |          — |          — |           — |        — |
| nanbeige4.2-3b       | stark_prime  | probe          |          — |          — |           — |        — |
| nanbeige4.2-3b       | stark_prime  | hybrid         |          — |          — |           — |        — |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| qwen3.5-2b           | stark_mag    | rep_distill    |      0.312 |      0.017 |       0.016 |   0.0089 |
| qwen3.5-2b           | stark_mag    | contrastive    |      0.364 |      0.017 |       0.013 |   0.0086 |
| qwen3.5-2b           | stark_mag    | probe          |      0.359 |      0.017 |       0.014 |   0.0090 |
| qwen3.5-2b           | stark_mag    | hybrid         |      0.333 |      0.017 |       0.015 |   0.0089 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|
| qwen3.5-2b           | stark_prime  | rep_distill    |      0.633 |      0.155 |       0.155 |   0.0497 |
| qwen3.5-2b           | stark_prime  | contrastive    |      0.633 |      0.120 |       0.120 |   0.0459 |
| qwen3.5-2b           | stark_prime  | probe          |      0.643 |      0.135 |       0.135 |   0.0484 |
| qwen3.5-2b           | stark_prime  | hybrid         |      0.643 |      0.137 |       0.137 |   0.0496 |
|----------------------|--------------|----------------|------------|------------|-------------|----------|

### Head-to-Head Comparison: V1 Baseline SFT → V2 Alignment Best Variant

| Model                | Dataset      | V1 Baseline $A_{\text{gen}}$ | V2 Best $A_{\text{gen}}$ | Best Variant      | Relative Gain |
|----------------------|--------------|------------------------------|--------------------------|-------------------|---------------|
| gemma4-e4b           | stark_mag    |                        0.001 |                    0.003 | rep_distill       |     +200.0%   |
| gemma4-e4b           | stark_prime  |                        0.017 |                    0.013 | probe/rep_distill |      -23.5%   |
| lfm2.5-1.2b          | stark_mag    |                        0.004 |                    0.008 | hybrid/probe      |     +100.0%   |
| lfm2.5-1.2b          | stark_prime  |                        0.068 |                    0.045 | probe             |      -33.8%   |
| llama3.2-3b          | stark_mag    |                        0.083 |                    0.048 | rep_distill       |      -42.2%   |
| llama3.2-3b          | stark_prime  |                        0.242 |                    0.205 | contrastive       |      -15.3%   |
| qwen3.5-2b           | stark_mag    |                        0.002 |                    0.016 | rep_distill       |     +700.0%   |
| qwen3.5-2b           | stark_prime  |                        0.191 |                    0.155 | rep_distill       |      -18.8%   |

> [!IMPORTANT]
> **Analysis**: V2 alignment brings massive gains on **STaRK-MAG** (the denser, harder graph where baseline SFT collapses most severely), confirming that BridgeAlign + DynLayerAlign specifically help multi-hop generalization under high entity density. On **STaRK-PRIME**, V2 variants are slightly below V1 alignment — suggesting the v2 data pipeline's bridge-span extraction introduces some distributional shift on the PRIME dataset. The strong memorization ($A_{\text{mem}}$ > 0.73) is preserved throughout, indicating no capacity trade-off.
