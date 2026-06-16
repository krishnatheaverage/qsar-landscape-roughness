#!/usr/bin/env bash
# Reproduce every result table, figure, and PDF from a fresh checkout.
#
# Requires the pinned environment (requirements.txt or environment.yml) plus the
# MoleculeACE benchmark (installed by those files) and a LaTeX toolchain or the
# `tectonic` engine for the final PDF build. Heavy steps cache per-target
# intermediates under cache*/ and skip work that is already done, so the script
# is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

echo "[ 1/13] per-compound roughness descriptors + random-forest error"
$PY src/build_features.py all
echo "[ 2/13] core statistics -> all_per_compound.csv, tableA, tableB"
$PY src/analyze.py
echo "[ 3/13] graph network, fixed 60-epoch schedule"
$PY src/gnn.py all
echo "[ 4/13] graph network, validated early stopping"
$PY src/gnn_tuned.py all
echo "[ 5/13] graph-network analysis -> Figure 5"
$PY src/analyze_gnn.py
echo "[ 6/13] robustness sweep (three distance metrics)"
$PY src/robustness.py ecfp4
$PY src/robustness.py ecfp6
$PY src/robustness.py desc
echo "[ 7/13] robustness analysis -> Figure 3"
$PY src/robustness_analyze.py
echo "[ 8/13] gradient boosting + SVR error"
$PY src/model2.py all
echo "[ 9/13] model-agnostic table -> Table 1"
$PY src/model_agnostic_analyze.py
echo "[10/13] cross-domain validation (ESOL + lipophilicity)"
$PY src/cross_domain.py
echo "[11/13] main-text figures + SI tables"
$PY src/make_figure1.py     # Figure 1
$PY src/enrichment.py       # Figure 2
$PY src/make_figure4.py     # Figure 4
$PY src/toc_graphic.py      # TOC graphic
$PY src/make_si_tables.py   # SI Tables S1, S3
echo "[12/13] roughness-conditioned conformal intervals -> Figure 6"
$PY src/conformal.py
echo "[12b/13] render publication figures in color-independent ggplot2 (R)"
# The Python steps above also emit tidy plot-data CSVs (results/fig*.csv etc.);
# these R scripts read them and write the final grayscale-safe figures.
if command -v Rscript >/dev/null 2>&1; then
  for r in fig1 fig2 fig3 fig4 fig6; do QSAR_ROOT="$PWD" Rscript "src/figures_R/$r.R"; done
else
  echo "  (Rscript not found; keeping the matplotlib figures. Install R with"
  echo "   ggplot2/patchwork/ggrepel/readr/dplyr/tidyr for the publication figures.)"
fi
echo "[13/13] build manuscript + supporting information PDFs"
cd paper
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf main.tex && latexmk -pdf supporting_information.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex && tectonic supporting_information.tex
else
  echo "  (no latexmk or tectonic found; skipping PDF build)"
fi
echo "Done. Figures in paper/figures/, tables in results/, PDFs in paper/."
