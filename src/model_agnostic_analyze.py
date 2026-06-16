# Compare landscape-roughness vs model-error rank correlations across RF/GBT/SVR over 30 targets.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

def partial(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), 1)
    x, y, Z = x[m], y[m], Z[m]
    if len(x) < 15: return np.nan
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(xr))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    rx = xr - Zr @ np.linalg.lstsq(Zr, xr, rcond=None)[0]
    ry = yr - Zr @ np.linalg.lstsq(Zr, yr, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]

rows = []
for f in sorted(x for x in os.listdir("cache") if x.endswith(".csv") and not x.startswith("robustness")):
    c = pd.read_csv(os.path.join("cache", f)); m = pd.read_csv(os.path.join("cache_models", f))
    c["gbt_err"] = m.gbt_err.values; c["svr_err"] = m.svr_err.values; rows.append(c)
df = pd.concat(rows, ignore_index=True)

print("=== Model-agnosticism: median Spearman across 30 targets ===")
print(f"{'model':28s} {'rho_dirichlet':>13s} {'rho_nbr_disp':>13s} {'partial_nbr|AD':>15s}")
out = []
for lab, ec in [("Random forest (primary)", "rf_err"), ("Gradient boosting", "gbt_err"), ("SVR (RBF kernel)", "svr_err")]:
    rd, rn, pn = [], [], []
    for _, g in df.groupby("dataset"):
        rd.append(spearmanr(g.dirichlet, g[ec]).statistic)
        s = g[["nbr_disp", ec]].dropna(); rn.append(spearmanr(s.nbr_disp, s[ec]).statistic)
        pn.append(partial(g.nbr_disp.values, g[ec].values, g[["nn_sim", "local_dens"]].values))
    print(f"{lab:28s} {np.median(rd):13.3f} {np.median(rn):13.3f} {np.median(pn):15.3f}")
    out.append(dict(model=lab, rho_dirichlet=round(float(np.median(rd)), 3),
                    rho_nbr_disp=round(float(np.median(rn)), 3),
                    partial_nbr_disp_AD=round(float(np.median(pn)), 3)))
pd.DataFrame(out).to_csv(os.path.join(RESULTS_DIR, "model_agnostic_results.csv"), index=False)
print("\nsaved model_agnostic_results.csv")
