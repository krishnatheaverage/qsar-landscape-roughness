"""robustness_analyze.py -- k- and metric-stability of the central correlations."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

METRICS = ["ecfp4", "ecfp6", "desc"]
NAME = {"ecfp4": "ECFP4 / Tanimoto", "ecfp6": "ECFP6 / Tanimoto", "desc": "RDKit desc / Euclidean"}
COL = {"ecfp4": "#2166ac", "ecfp6": "#67a9cf", "desc": "#ef8a62"}
K_LIST = [5, 10, 15, 20, 30]

def partial_spearman(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[m], y[m], Z[m]
    if len(x) < 15: return np.nan
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(xr))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    rx = xr - Zr @ np.linalg.lstsq(Zr, xr, rcond=None)[0]
    ry = yr - Zr @ np.linalg.lstsq(Zr, yr, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 1e-9 and ry.std() > 1e-9 else np.nan

data = {m: pd.read_csv(os.path.join(CACHE_DIR, f"robustness_{m}.csv")) for m in METRICS}

def agg(df, k, col, partial=False):
    sub = df[df.k == k]; out = []
    for _, g in sub.groupby("dataset"):
        if partial:
            out.append(partial_spearman(g[col].values, g.rf_err.values, g[["nn_dist", "local_dens"]].values))
        else:
            s = g[[col, "rf_err"]].dropna()
            if len(s) > 10: out.append(spearmanr(s[col], s.rf_err).statistic)
    return np.nanmedian(out)

print("=== k- and metric-stability of median Spearman vs RF error (30 targets) ===")
print(f"{'metric':24s} {'k':>3s} | {'local_var':>9s} {'nbr_disp':>9s} {'nbr_disp|AD':>11s}")
res = []
for m in METRICS:
    for k in K_LIST:
        lv = agg(data[m], k, "local_var")
        nd = agg(data[m], k, "nbr_disp")
        ndp = agg(data[m], k, "nbr_disp", partial=True)
        res.append(dict(metric=m, k=k, local_var=lv, nbr_disp=nd, nbr_disp_partial=ndp))
        print(f"{NAME[m]:24s} {k:3d} | {lv:9.3f} {nd:9.3f} {ndp:11.3f}")
R = pd.DataFrame(res); R.to_csv(os.path.join(RESULTS_DIR, "robustness_summary.csv"), index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for m in METRICS:
    sub = R[R.metric == m]
    ax[0].plot(sub.k, sub.local_var, "-o", color=COL[m], label=NAME[m])
    ax[1].plot(sub.k, sub.nbr_disp, "-o", color=COL[m], label=NAME[m])
    ax[2].plot(sub.k, sub.nbr_disp_partial, "-o", color=COL[m], label=NAME[m])
titles = ["local_var  (landscape roughness, uses y)",
          "nbr_disp  (y-free)",
          "nbr_disp | AD  (partial, controls NN-dist + density)"]
for a, t in zip(ax, titles):
    a.set_xlabel("neighbourhood size k"); a.set_title(t, fontsize=10)
    a.axhline(0, color="k", lw=.5); a.set_xticks(K_LIST); a.legend(fontsize=8, frameon=False)
ax[0].set_ylabel("median Spearman ρ vs RF error")
fig.suptitle("Robustness: the roughness–error relationship is stable across k and across distance metric",
             fontsize=12, weight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "robustness_figure.png"), dpi=150, bbox_inches="tight")
print("\nfigure saved -> robustness_figure.png")
