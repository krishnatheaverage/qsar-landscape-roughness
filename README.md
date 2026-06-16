# Structure–Activity Landscape Roughness Predicts Per-Compound QSAR Error Independently of the Applicability Domain

This repository contains the full data-analysis pipeline, results, figures, and the LaTeX manuscript for a study showing that a **per-compound, structure-only measure of local structure-activity landscape roughness predicts where a QSAR model will make large errors**, and that this signal is statistically distinct from (and for activity cliffs, opposite to) the classical applicability domain.


## Overview

Machine-learning models for molecular potency rely on the similarity principle: similar structures tend to have similar activity. Activity cliffs (pairs of very similar molecules with very different potency) violate that principle and are where models fail. Existing roughness measures are either global (one number per dataset, e.g. ROGI, MODI, SARI) or pairwise (e.g. SALI), and applicability-domain or uncertainty methods target sparse-region extrapolation rather than the dense-region discontinuities that cliffs represent.

This work defines a small family of **local roughness descriptors** computed from a compound's nearest training neighbors, including activity-free variants that need no measured activity, and evaluates them on the 30-target MoleculeACE benchmark plus two physicochemical datasets.

## Key findings

1. **The per-compound reliability tools in use miss cliffs.** RF tree variance ranks general error well (flagging AUC 0.71) but is at chance on activity cliffs (0.52); the applicability domain is oriented backwards, since cliffs are dense, not sparse.
2. **Structure-only local roughness is the missing signal, and the right per-compound scale for calibration.** A conformal predictor conditioned on roughness keeps 90% interval coverage flat across the roughness range and lifts cliff coverage toward nominal (87% → 90%), where conditioning on tree variance or the applicability domain leaves coverage falling with roughness and broken on cliffs.
3. **Complementary, not competitive.** Variance ranks ordinary error, roughness ranks cliffs (AUC 0.69 vs variance's 0.52); a combined variance + roughness score inherits both and beats variance on cliffs on 29/30 targets.
4. **Distinct from the applicability domain** (survives partial-correlation and mixed-effects control), **activity-free** and available before assay, holds across three model classes (RF, gradient boosting, SVR) and distance metrics, and replicates on aqueous solubility and lipophilicity.
5. A landscape measure that uses the query's **own activity** correlates more strongly (Dirichlet energy median Spearman rho = 0.65) but is a mechanistic **upper bound, not deployable**: computing it needs the potency being predicted.

Positioning: Sheridan (JCIM 2012) used per-compound RF tree variance and the prediction value as reliability signals, and the reliability-density-neighbourhood method (Aniceto et al., J. Cheminform. 2016) did per-instance local-neighbourhood reliability; both, like the applicability domain, are organized around proximity to the training data. The distinct claim here is the **activity-free local-roughness axis** and its **opposition to the applicability domain for cliffs**, the failure those proximity-based signals miss.

## Repository structure

```
qsar-landscape-roughness/
  README.md
  requirements.txt
  LICENSE
  CITATION.cff
  .gitignore
  src/          analysis pipeline and figure/table scripts
  src/config.py central path configuration (repo-relative, env-overridable)
  results/      aggregated result tables (CSV)
  paper/        LaTeX manuscript (main.tex) + supporting information, and reviewer memo
  paper/figures/ all figures and the table-of-contents graphic
```

## Data

- **MoleculeACE benchmark** (van Tilborg, Alenicheva, Grisoni, *J. Chem. Inf. Model.* 2022): 30 ChEMBL targets with activity-cliff labels and fixed train/test splits. Install via `pip install MoleculeACE`; the data ship inside the package under `MoleculeACE/Data/benchmark_data`.
- **MoleculeNet physicochemical sets** (Wu et al., *Chem. Sci.* 2018): ESOL aqueous solubility (Delaney 2004) and Lipophilicity (logD). `src/cross_domain.py` downloads these CSVs automatically.

These datasets are redistributed under their own licenses by their original authors; this repository does not include the raw benchmark files.

## Installation

```bash
pip install -r requirements.txt
pip install MoleculeACE          # provides the 30-target benchmark data
# to build the paper you need a LaTeX toolchain (TeX Live / MiKTeX / Overleaf),
# or the self-contained `tectonic` engine.
```

A CPU-only PyTorch is sufficient for the graph-network experiments.

## Reproducing the results

**One command** runs the whole pipeline (descriptors → statistics → every figure and table → both PDFs):

```bash
./reproduce.sh
```

Heavy steps cache per-target intermediates to disk and skip work already done, so the script is safe to re-run. To run stages individually, in logical order:

```bash
python src/build_features.py all        # roughness descriptors + RF error per target -> cache/
python src/analyze.py                   # -> results/all_per_compound.csv, tableA, tableB
python src/gnn.py all                   # fixed-schedule GIN errors -> cache_gnn/
python src/gnn_tuned.py all             # validated/early-stopped GIN errors -> cache_gnn2/
python src/analyze_gnn.py               # -> gnn_fixed_vs_tuned.csv, Figure 5
python src/robustness.py ecfp4          # k-sweep under ECFP4/Tanimoto  -> cache
python src/robustness.py ecfp6          # k-sweep under ECFP6/Tanimoto  -> cache
python src/robustness.py desc           # k-sweep under descriptor/Euclidean -> cache
python src/robustness_analyze.py        # -> robustness_summary.csv, Figure 3
python src/model2.py all                # gradient boosting + SVR errors -> cache_models/
python src/model_agnostic_analyze.py    # -> model_agnostic_results.csv (Table 1)
python src/cross_domain.py              # ESOL + Lipophilicity -> cross_domain_results.csv
python src/make_figure1.py              # main three-panel figure (Figure 1)
python src/enrichment.py                # triage/enrichment (Figure 2)
python src/make_figure4.py              # cross-domain validation figure (Figure 4)
python src/toc_graphic.py               # ACS table-of-contents graphic
python src/make_si_tables.py            # -> results/per_target_correlations.csv + paper/si_tables.tex
python src/conformal.py                 # risk-score complementarity + conditional conformal intervals -> conformal_results.csv, conformal_curve.csv, complementarity_results.csv, Figure 6, Table 2
```

**Paths and configuration.** All input and output locations are defined in `src/config.py` relative to the repository root, so the pipeline runs from a fresh checkout with no path editing. The MoleculeACE benchmark directory is located automatically from the installed package, and `cross_domain.py` downloads the two MoleculeNet CSVs on first run. You can override any location with environment variables (`QSAR_CACHE`, `QSAR_RESULTS`, `QSAR_FIGURES`, `QSAR_DATA`, `MOLECULEACE_DATA`). Cached per-target intermediates are written to `cache/` (gitignored) and regenerated on demand.

## Building the paper (LaTeX)

The manuscript and supporting information are written in LaTeX under `paper/`:

| File | Contents |
|---|---|
| `paper/main.tex` | the manuscript (article class, ACS-style references, self-contained bibliography — no `.bib`/biber needed) |
| `paper/supporting_information.tex` | the SI (per-target tables, robustness sweep, GNN details) |
| `paper/si_tables.tex` | per-target table bodies; regenerate with `python src/make_si_tables.py` |
| `paper/figures/` | all figures + the TOC graphic |
| `paper/Makefile` | build targets |

```bash
cd paper
make            # builds main.pdf and supporting_information.pdf via latexmk
make tectonic   # alternative: build both with the self-contained tectonic engine
```

Both documents also compile out of the box on Overleaf.

## Figures

Figures 1-4 and 6 are rendered in **color-independent ggplot2** (grayscale-distinguishable shades, line types, and point shapes, so they read in black-and-white) by the R scripts in `src/figures_R/`. The Python analysis scripts emit tidy plot-data CSVs (`results/fig1_panelA.csv`, `fig2_curves.csv`, `figure4_plot_values.csv`, `conformal_curve.csv`, `robustness_summary.csv`, ...); each R script reads the matching CSV, so the figures never re-derive statistics. Rendering needs R with `ggplot2`, `patchwork`, `ggrepel`, `readr`, `dplyr`, `tidyr`; `reproduce.sh` runs them after the Python steps (and falls back to the matplotlib figures if R is absent). Figure 5 (the graph-network supporting figure) is matplotlib, since its per-compound regeneration needs the GNN caches.

Filenames are descriptive; the manuscript numbers figures in order of appearance:

| Manuscript | File | Rendered by |
|---|---|---|
| Figure 1 | `figure1_main.png` | `src/figures_R/fig1.R` (vertical 3-panel) |
| Figure 2 | `figure5_enrichment.png` | `src/figures_R/fig2.R` |
| Figure 3 | `robustness_figure.png` | `src/figures_R/fig3.R` |
| Figure 4 | `figure4_crossdomain.png` | `src/figures_R/fig4.R` |
| Figure 5 | `gnn_tuned_figure.png` | `src/analyze_gnn.py` (matplotlib) |
| Figure 6 | `figure6_conformal.png` | `src/figures_R/fig6.R` |
| TOC graphic | `TOC_graphic.png` | `src/toc_graphic.py` |

## Result tables

| File | Contents |
|---|---|
| `all_per_compound.csv` | Per-test-compound roughness descriptors, applicability-domain and uncertainty controls, random-forest error, activity, and cliff label, for all 30 targets |
| `tableA_spearman_vs_error.csv` | Per-target Spearman of each descriptor vs error, aggregated (median, sign consistency, FDR-adjusted p) |
| `tableB_partial_AD.csv` | Partial Spearman after controlling for the applicability domain |
| `tableE_global_indices.csv` | Global ROGI index vs mean error and cliff fraction |
| `robustness_summary.csv` | Correlations across neighborhood size k and three distance metrics |
| `gnn_fixed_vs_tuned.csv` | Graph-network error and roughness-quartile gap, both training regimens |
| `model_agnostic_results.csv` | Roughness-error correlation for RF, gradient boosting, and SVR |
| `cross_domain_results.csv` | ESOL and Lipophilicity validation |
| `per_target_correlations.csv` | Per-target Spearman of each descriptor vs error, partial (AD-controlled), and cliff fraction (SI Table S1) |
| `conformal_results.csv` | 90% conformal interval coverage (marginal / cliff / non-cliff) and width, by conditioning variable (variance, roughness, AD, combined) and construction (Figure 6, SI Table S4) |
| `conformal_curve.csv` | conformal coverage vs local-roughness quantile, per conditioning variable (Figure 6A) |
| `complementarity_results.csv` | flagging ROC-AUC of variance / roughness / combined on top-quartile error vs activity cliffs (Table 2) |

## Status and limitations

This is a complete, self-contained, fully in-silico study. Honest caveats: the activity-free effect sizes, while consistent across all targets, are modest; the graph-convolutional baseline was not architecture-optimized and is reported as a supporting observation only; one descriptor (a local Holder exponent) did not predict error and is reported as a negative result. The manuscript has not yet had independent expert review; `paper/review_memo.md` contains a structured self-critique anticipating likely reviewer concerns.

## Citation

If you use this code or build on these results, please cite the manuscript (in preparation) and the underlying benchmark:

> Harish, K. Structure–Activity Landscape Roughness Predicts Per-Compound QSAR Error Independently of the Applicability Domain. Manuscript, 2026.
>
> van Tilborg, D.; Alenicheva, A.; Grisoni, F. Exposing the Limitations of Molecular Machine Learning with Activity Cliffs. *J. Chem. Inf. Model.* 2022, 62 (23), 5938-5951.

## License

Code is released under the MIT License (see `LICENSE`). Benchmark datasets remain under the licenses of their original publishers.
