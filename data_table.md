| Model | Dataset | Variant | Peak A_mem | Peak A_gen | Final A_gen | AUC_gen |
|---|---|---|---|---|---|---|
| gemma4-e4b      | stark_mag   | rep_distill |      0.429 |      0.009 |       0.003 |   0.004 |
| gemma4-e4b      | stark_mag   | contrastive |      0.437 |      0.009 |       0.001 |   0.004 |
| gemma4-e4b      | stark_mag   | probe       |      0.431 |      0.007 |       0.002 |   0.004 |
| gemma4-e4b      | stark_mag   | hybrid      |      0.426 |      0.009 |       0.001 |   0.004 |
|---|---|---|---|---|---|---|
| gemma4-e4b      | stark_prime | contrastive |      0.222 |      0.012 |       0.010 |   0.009 |
| gemma4-e4b      | stark_prime | probe       |      0.246 |      0.017 |       0.013 |   0.011 |
| gemma4-e4b      | stark_prime | hybrid      |      0.241 |      0.016 |       0.012 |   0.011 |
|---|---|---|---|---|---|---|
| lfm2.5-1.2b     | stark_mag   | rep_distill |      0.742 |      0.011 |       0.007 |   0.007 |
| lfm2.5-1.2b     | stark_mag   | contrastive |      0.724 |      0.011 |       0.008 |   0.007 |
| lfm2.5-1.2b     | stark_mag   | probe       |      0.745 |      0.013 |       0.008 |   0.008 |
| lfm2.5-1.2b     | stark_mag   | hybrid      |      0.739 |      0.014 |       0.008 |   0.008 |
|---|---|---|---|---|---|---|
| lfm2.5-1.2b     | stark_prime | rep_distill |      0.658 |      0.042 |       0.042 |   0.031 |
| lfm2.5-1.2b     | stark_prime | contrastive |      0.658 |      0.041 |       0.037 |   0.033 |
| lfm2.5-1.2b     | stark_prime | probe       |      0.653 |      0.045 |       0.045 |   0.031 |
| lfm2.5-1.2b     | stark_prime | hybrid      |      0.653 |      0.042 |       0.042 |   0.031 |
|---|---|---|---|---|---|---|
| llama3.2-3b     | stark_mag   | rep_distill |      0.424 |      0.048 |       0.048 |   0.007 |
| llama3.2-3b     | stark_mag   | contrastive |      0.421 |      0.031 |       0.031 |   0.006 |
| llama3.2-3b     | stark_mag   | probe       |      0.429 |      0.039 |       0.039 |   0.007 |
| llama3.2-3b     | stark_mag   | hybrid      |      0.435 |      0.042 |       0.042 |   0.007 |
|---|---|---|---|---|---|---|
| llama3.2-3b     | stark_prime | rep_distill |      0.725 |      0.191 |       0.191 |   0.077 |
| llama3.2-3b     | stark_prime | contrastive |      0.725 |      0.205 |       0.205 |   0.079 |
| llama3.2-3b     | stark_prime | probe       |      0.731 |      0.196 |       0.196 |   0.076 |
| llama3.2-3b     | stark_prime | hybrid      |      0.731 |      0.179 |       0.179 |   0.076 |
|---|---|---|---|---|---|---|
| nanbeige4.2-3b  | stark_mag   | baseline    |      0.000 |      0.000 |       0.000 |   0.000 |
|---|---|---|---|---|---|---|
| nanbeige4.2-3b  | stark_prime | baseline    |      0.000 |      0.000 |       0.000 |   0.000 |
|---|---|---|---|---|---|---|
| qwen3.5-2b      | stark_mag   | baseline    |      0.407 |      0.019 |       0.015 |   0.010 |
| qwen3.5-2b      | stark_mag   | rep_distill |      0.312 |      0.017 |       0.016 |   0.009 |
| qwen3.5-2b      | stark_mag   | contrastive |      0.364 |      0.017 |       0.013 |   0.009 |
| qwen3.5-2b      | stark_mag   | probe       |      0.359 |      0.017 |       0.014 |   0.009 |
| qwen3.5-2b      | stark_mag   | hybrid      |      0.333 |      0.017 |       0.015 |   0.009 |
|---|---|---|---|---|---|---|
| qwen3.5-2b      | stark_prime | baseline    |      0.639 |      0.233 |       0.233 |   0.057 |
| qwen3.5-2b      | stark_prime | rep_distill |      0.633 |      0.155 |       0.155 |   0.050 |
| qwen3.5-2b      | stark_prime | contrastive |      0.633 |      0.120 |       0.120 |   0.046 |
| qwen3.5-2b      | stark_prime | probe       |      0.643 |      0.135 |       0.135 |   0.048 |
| qwen3.5-2b      | stark_prime | hybrid      |      0.643 |      0.137 |       0.137 |   0.050 |
|---|---|---|---|---|---|---|