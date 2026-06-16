"""make_gnn_panelB.py -- tidy Panel-B data for the GNN figure (fig_gnn.R).

Emits results/gnn_quartile_gap.csv with columns:
    regime (fixed-60 | tuned), rough_q (1..4), mean_gap, sem, source

Panel B = mean per-compound (GNN error - RF error) by within-target local-roughness
quartile (Q1 smooth .. Q4 rough), for the fixed-60 and tuned GIN regimes.

- FIXED-60 is computed exactly from results/all_per_compound.csv (gnn_err is the
  fixed-60 prediction; rough_q is the within-dataset dirichlet quartile, recomputed
  here to match analyze_gnn.py).
- TUNED needs per-compound tuned predictions, which live in the gitignored cache
  (cache/gnn2). If that cache is present this script computes the tuned bars exactly;
  if it is absent it falls back to values DIGITIZED from the committed figure so the
  plot still renders, marking source='figure_digitized'. Re-run where cache/gnn2
  exists (or after `python src/gnn_tuned.py`) to replace them with exact numbers.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, CACHE_DIR, CACHE_GNN2
import numpy as np, pandas as pd

df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))
# within-target roughness quartile from the Dirichlet energy (matches analyze_gnn.py)
df["rough_q"] = df.groupby("dataset")["dirichlet"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4]))
df["gap_fixed"] = df["gnn_err"] - df["rf_err"]

rows = []
def add(regime, gapcol, frame, source):
    g = frame.groupby("rough_q")[gapcol]
    for q in [1, 2, 3, 4]:
        rows.append(dict(regime=regime, rough_q=int(q),
                         mean_gap=float(g.mean().loc[q]),
                         sem=float(g.sem().loc[q]), source=source))

add("fixed-60", "gap_fixed", df, "data")

# --- tuned: exact if the cache is here, else digitized fallback ---
have_cache = os.path.isdir(CACHE_GNN2) and len(os.listdir(CACHE_GNN2)) > 0
if have_cache:
    tuned = pd.concat([pd.read_csv(os.path.join(CACHE_GNN2, f))
                       for f in sorted(os.listdir(CACHE_DIR))], ignore_index=True)
    assert len(tuned) == len(df) and (tuned.smiles.values == df.smiles.values).mean() == 1.0
    df["gap_tuned"] = tuned.gnn_err.values - df["rf_err"].values
    add("tuned", "gap_tuned", df, "data")
    print("tuned bars: computed exactly from cache/gnn2")
else:
    # values read from the committed gnn_tuned_figure.png (Q1..Q4); approximate.
    DIG = {1: (0.273, 0.007), 2: (0.218, 0.008), 3: (0.191, 0.009), 4: (0.165, 0.012)}
    for q, (m, s) in DIG.items():
        rows.append(dict(regime="tuned", rough_q=q, mean_gap=m, sem=s,
                         source="figure_digitized"))
    print("WARNING: cache/gnn2 not found -> tuned bars are DIGITIZED from the figure; "
          "re-run with the cache (or `python src/gnn_tuned.py`) for exact values.")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS_DIR, "gnn_quartile_gap.csv"), index=False)
print("\nsaved results/gnn_quartile_gap.csv")
print(out.to_string(index=False))
