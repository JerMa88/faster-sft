# ACL 2026 Paper — Routing Facts to Reasoning Circuits

## Files

| File | Description |
|---|---|
| `main.tex` | Full paper LaTeX source (~760 lines, 9 sections) |
| `refs.bib` | BibTeX references (21 entries) |
| `figures/` | All 5 paper figures (PDF, vector, 300 DPI) |
| `README.md` | This file |

## To Compile

```bash
# Option 1: Overleaf — just upload main.tex, refs.bib, and figures/
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
| `fig1_kug_scatter.pdf` | KUG scatter: A_mem vs A_gen, all 12 baseline runs |
| `fig2_learning_curves.pdf` | Learning curves: A_mem and A_gen vs epoch |
| `fig3_v1_gains.pdf` | V1 variant grouped bar chart |
| `fig4_v2_vs_v1.pdf` | V2 vs V1 relative % gain |
| `fig5_oracle_headroom.pdf` | Oracle headroom stacked bar |

## Paper Structure

| Section | Title | ~Pages |
|---|---|---|
| §1 | Introduction | 1.0 |
| §2 | Background | 0.7 |
| §3 | Problem Formalization | 0.7 |
| §4 | Method: AA-SFT (V1 + V2) | 1.5 |
| §5 | Experimental Setup | 0.5 |
| §6 | Results (3 tables) | 1.5 |
| §7 | Analysis | 0.8 |
| §8 | Related Work (comparison table) | 0.5 |
| §9 | Conclusion | 0.3 |
| | References | ~1.0 |

Total: ~8 pages content + references (ACL Long Paper limit)
