# ACL 2026 Paper — Routing Facts to Reasoning Circuits

## Files

| File | Description |
|---|---|
| `main.tex` | Full paper LaTeX source (~646 lines, 9 sections) |
| `refs.bib` | BibTeX references |
| `figures/` | All 5 paper figures (PDF, vector, 300 DPI) |
| `README.md` | This file |

## To Compile

```bash
# Option 1: Overleaf — upload main.tex, refs.bib, and figures/
# Option 2: Local LaTeX
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## To Regenerate Figures

From repo root:
```bash
python3 scripts/analysis/generate_paper_figures.py
```
Figures are saved to `paper/figures/` as PDFs.

## Figure Map

| File | Description |
|---|---|
| `fig1_kug_scatter.pdf` | KUG scatter: A_mem vs A_gen, 7 baseline runs (relaxed EM) |
| `fig2_learning_curves.pdf` | Learning curves: A_mem and A_gen vs epoch (Llama/PRIME, Qwen/PRIME) |
| `fig3_v1_gains.pdf` | AA-SFT variant grouped bar chart vs SFT baseline (dashed) |
| `fig4_baseline_vs_aasft.pdf` | Side-by-side SFT baseline vs AA-SFT best, % gain annotated, 6/7 pairs |
| `fig5_oracle_headroom.pdf` | Oracle headroom stacked bar (AA-SFT closes ~32–35% gap) |

## Paper Structure

| Section | Title |
|---|---|
| §1 | Introduction |
| §2 | Background |
| §3 | Problem Formalization |
| §4 | Method: AA-SFT (RepDist, ContraRoute, Probe, Hybrid) |
| §5 | Experimental Setup |
| §6 | Results (2 tables: KUG baseline + AA-SFT alignment) |
| §7 | Analysis |
| §8 | Related Work |
| §9 | Conclusion |

**Total:** ~8 pages content + references (ACL Long Paper limit)

## Key Numbers (for reviewer reference)

- Models: Llama3.2-3B, Qwen3.5-2B, LFM2.5-1.2B, Gemma4-E4B
- Datasets: STaRK-PRIME, STaRK-MAG
- Metric: Relaxed EM (gold answer substring of output, case-insensitive)
- Best result: A_gen = 0.271 (Qwen3.5-2B Probe/Hybrid, STaRK-PRIME)
- Beats SFT baseline: 6/7 model-dataset pairs
- Oracle headroom closure: ~32–35% at zero inference cost
