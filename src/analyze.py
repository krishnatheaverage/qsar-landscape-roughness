"""
analyze.py -- turn the cached per-compound data into the paper's stats.

(A) per-target Spearman of each predictor vs RF error, aggregated over the 30 targets
    (median, sign-consistency, Wilcoxon vs 0, BH-FDR).
(B) AD confound check: partial Spearman of each roughness measure vs error, controlling
    for NN similarity + local density. does roughness survive once you remove the
    "i'm far from training data" effect?
(C) cliff-classification AUC from the y-free measures (back to the earlier null).
(D) deep-learning gap: per-compound (gnn_err - rf_err) vs roughness, + quartile means.
(E) mixed-effects sanity check on within-target z-scored data.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, warnings, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

CACHE, CACHE_GNN = CACHE_DIR, CACHE_GNN
OUT = RESULTS_DIR

# load + merge (rows are already in the same order)
frames = []
for f in sorted(os.listdir(CACHE)):
    d = pd.read_csv(os.path.join(CACHE, f))
    g = pd.read_csv(os.path.join(CACHE_GNN, f))
    assert len(d) == len(g)
    d["gnn_err"] = g["gnn_err"].values
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["gap"] = df["gnn_err"] - df["rf_err"]                 # >0 = GNN worse than RF
print(f"Loaded {len(df)} test compounds across {df.dataset.nunique()} targets "
      f"({df.cliff_mol.mean()*100:.1f}% cliffs)\n")

LANDSCAPE = ["dirichlet", "lipschitz"]                    # use the query's y
APRIORI   = ["nbr_disp", "sali_max", "sali_mean", "holder"]  # y-free
BASELINES = ["nn_sim", "local_dens", "rf_var", "mol_size"]   # AD + uncertainty + size
SIGN = {"holder": -1, "nn_sim": -1, "local_dens": -1}        # ones we expect to go negative
def oriented(col):  # flip so higher = more expected error
    return SIGN.get(col, 1) * df[col]

def agg_spearman(target="rf_err"):
    rows = []
    for col in LANDSCAPE + APRIORI + BASELINES:
        rhos = []
        for ds, sub in df.groupby("dataset"):
            s = sub[[col, target]].dropna()
            if len(s) > 10:
                rhos.append(spearmanr(s[col], s[target]).statistic)
        rhos = np.array(rhos)
        rhos_oriented = rhos * SIGN.get(col, 1)
        p = wilcoxon(rhos_oriented).pvalue
        rows.append(dict(predictor=col, median_rho=np.median(rhos),
                         oriented_median=np.median(rhos_oriented),
                         frac_consistent=np.mean(rhos_oriented > 0), wilcoxon_p=p, n=len(rhos)))
    r = pd.DataFrame(rows)
    r["fdr_p"] = multipletests(r.wilcoxon_p, method="fdr_bh")[1]
    return r.sort_values("oriented_median", ascending=False)

print("=== (A) Per-target Spearman vs RF error (aggregated over 30 targets) ===")
A = agg_spearman("rf_err")
print(A.to_string(index=False,
      formatters={c: "{:.3f}".format for c in ["median_rho","oriented_median","frac_consistent","wilcoxon_p","fdr_p"]}))
A.to_csv(os.path.join(OUT, "tableA_spearman_vs_error.csv"), index=False)

# (B) partial Spearman, controlling for AD (nn_sim + local_dens)
def partial_spearman(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[m], y[m], Z[m]
    if len(x) < 15: return np.nan
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(xr))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    rx = xr - Zr @ np.linalg.lstsq(Zr, xr, rcond=None)[0]
    ry = yr - Zr @ np.linalg.lstsq(Zr, yr, rcond=None)[0]
    if rx.std() < 1e-9 or ry.std() < 1e-9: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

print("\n=== (B) AD-confound test: zero-order vs partial Spearman (control = nn_sim + local_dens) ===")
print(f"{'predictor':12s} {'zero-order':>11s} {'partial|AD':>11s} {'retained%':>10s}")
Brows = []
for col in LANDSCAPE + APRIORI:
    z0, zp = [], []
    for ds, sub in df.groupby("dataset"):
        x = (sub[col] * SIGN.get(col, 1)).values; y = sub["rf_err"].values
        Z = sub[["nn_sim", "local_dens"]].values
        s = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(s) > 15:
            z0.append(spearmanr(s.x, s.y).statistic)
            zp.append(partial_spearman(x, y, Z))
    z0, zp = np.array(z0), np.array([v for v in zp if v == v])
    ret = np.median(zp) / np.median(z0) * 100 if np.median(z0) != 0 else np.nan
    print(f"{col:12s} {np.median(z0):11.3f} {np.median(zp):11.3f} {ret:9.0f}%")
    Brows.append(dict(predictor=col, zero_order=np.median(z0), partial_AD=np.median(zp), retained_pct=ret))
pd.DataFrame(Brows).to_csv(os.path.join(OUT, "tableB_partial_AD.csv"), index=False)

# (C) cliff AUC from the y-free measures
print("\n=== (C) Flagging labelled cliffs from y-free constructs (per-target AUC, median) ===")
for col in APRIORI + ["rf_var", "nn_sim"]:
    aucs = []
    for ds, sub in df.groupby("dataset"):
        s = sub[[col, "cliff_mol"]].dropna()
        if s.cliff_mol.nunique() == 2 and len(s) > 10:
            aucs.append(roc_auc_score(s.cliff_mol, s[col] * SIGN.get(col, 1)))
    print(f"  {col:12s}: AUC {np.median(aucs):.3f}  [{np.min(aucs):.3f}, {np.max(aucs):.3f}]")

# (D) deep-learning gap
print("\n=== (D) Deep-learning gap: (GNN err - RF err) vs roughness ===")
print(f"  Overall mean gap (GNN-RF): {df.gap.mean():+.3f}  | GNN worse on {(df.groupby('dataset').gap.mean()>0).mean()*100:.0f}% of targets")
print(f"  Mean gap  cliffs={df[df.cliff_mol==1].gap.mean():+.3f}  non-cliffs={df[df.cliff_mol==0].gap.mean():+.3f}")
for col in ["dirichlet", "sali_mean", "nbr_disp"]:
    rhos = [spearmanr(s[col], s["gap"]).statistic
            for ds, sub in df.groupby("dataset") for s in [sub[[col, "gap"]].dropna()] if len(s) > 10]
    rhos = np.array(rhos)
    print(f"  rho(gap, {col:9s}) median={np.median(rhos):+.3f}  consistent={np.mean(rhos>0)*100:.0f}%  wilcoxon_p={wilcoxon(rhos).pvalue:.1e}")
# gap by roughness quartile (within target)
df["rough_q"] = df.groupby("dataset")["dirichlet"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1,2,3,4]))
qg = df.groupby("rough_q").gap.mean()
print("  mean gap by roughness quartile (Q1 smooth -> Q4 rough):",
      "  ".join(f"Q{q}={qg.loc[q]:+.3f}" for q in [1,2,3,4]))

# (E) mixed-effects check (within-target z-scored)
print("\n=== (E) Mixed-effects: per-target z-scored error ~ roughness + AD controls + (1|target) ===")
z = df.copy()
for c in ["rf_err", "sali_mean", "nbr_disp", "dirichlet", "nn_sim", "local_dens", "mol_size"]:
    z[c+"_z"] = z.groupby("dataset")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
z = z.replace([np.inf, -np.inf], np.nan)
for rough in ["sali_mean_z", "nbr_disp_z", "dirichlet_z"]:
    d2 = z.dropna(subset=[rough, "rf_err_z", "nn_sim_z", "local_dens_z", "mol_size_z"])
    md = smf.mixedlm(f"rf_err_z ~ {rough} + nn_sim_z + local_dens_z + mol_size_z",
                     d2, groups=d2["dataset"]).fit(method="lbfgs", disp=False)
    b, p = md.params[rough], md.pvalues[rough]
    ci = md.conf_int().loc[rough]
    print(f"  {rough:14s}: beta={b:+.3f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]  p={p:.1e}")

df.to_csv(os.path.join(OUT, "all_per_compound.csv"), index=False)
print("\nSaved: tableA_spearman_vs_error.csv, tableB_partial_AD.csv, all_per_compound.csv")
