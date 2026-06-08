"""make_figure1.py -- main-text Figure 1 (panels A,B,D; panel C demoted) + Results numbers."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, wilcoxon, mannwhitneyu
from sklearn.metrics import roc_auc_score

df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))
SIGN = {"holder": -1, "nn_sim": -1, "local_dens": -1}
FAMILY = {"dirichlet": "landscape", "lipschitz": "landscape",
          "nbr_disp": "a-priori", "sali_mean": "a-priori", "sali_max": "a-priori", "holder": "a-priori",
          "nn_sim": "applicability domain", "local_dens": "applicability domain", "rf_var": "uncertainty"}
COL = {"landscape": "#b2182b", "a-priori": "#2166ac", "applicability domain": "#878787", "uncertainty": "#1b7837"}
LABEL = {"dirichlet": "Dirichlet*", "lipschitz": "Lipschitz/SALI*", "nbr_disp": "nbr dispersion",
         "sali_mean": "SALI-density (mean)", "sali_max": "SALI-density (max)", "holder": "Hölder exp.",
         "nn_sim": "NN similarity", "local_dens": "local density", "rf_var": "RF variance"}

def per_target_rho(col, target="rf_err"):
    out = []
    for _, sub in df.groupby("dataset"):
        s = sub[[col, target]].dropna()
        if len(s) > 10: out.append(spearmanr(s[col], s[target]).statistic * SIGN.get(col, 1))
    return np.array(out)

def partial_spearman(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[m], y[m], Z[m]
    if len(x) < 15: return np.nan
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(xr))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    rx = xr - Zr @ np.linalg.lstsq(Zr, xr, rcond=None)[0]
    ry = yr - Zr @ np.linalg.lstsq(Zr, yr, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 1e-9 and ry.std() > 1e-9 else np.nan

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

# Panel A
order = ["dirichlet", "lipschitz", "rf_var", "nbr_disp", "sali_mean", "sali_max", "nn_sim", "local_dens", "holder"]
meds = [np.median(per_target_rho(c)) for c in order]
q1 = [np.percentile(per_target_rho(c), 25) for c in order]; q3 = [np.percentile(per_target_rho(c), 75) for c in order]
ypos = np.arange(len(order))[::-1]
ax[0].barh(ypos, meds, color=[COL[FAMILY[c]] for c in order],
           xerr=[np.array(meds)-np.array(q1), np.array(q3)-np.array(meds)], error_kw=dict(ecolor="#444", lw=1, capsize=3))
ax[0].set_yticks(ypos); ax[0].set_yticklabels([LABEL[c] for c in order], fontsize=9)
ax[0].axvline(0, color="k", lw=.6); ax[0].set_xlabel("Spearman ρ vs per-compound error\n(median ± IQR, 30 targets)", fontsize=9)
ax[0].set_title("A  What predicts where QSAR errs", fontsize=11, loc="left", weight="bold")
ax[0].legend([plt.Rectangle((0,0),1,1,color=COL[k]) for k in COL], COL.keys(), fontsize=7.2, loc="lower right", frameon=False)
ax[0].text(0.02, 0.02, "* uses the query's own activity", transform=ax[0].transAxes, fontsize=7, style="italic")

# Panel B
constructs = ["dirichlet", "lipschitz", "nbr_disp", "sali_mean"]
z0, zp = [], []
for c in constructs:
    a, b = [], []
    for _, sub in df.groupby("dataset"):
        x = (sub[c] * SIGN.get(c, 1)).values; y = sub["rf_err"].values; Z = sub[["nn_sim", "local_dens"]].values
        ss = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(ss) > 15: a.append(spearmanr(ss.x, ss.y).statistic); b.append(partial_spearman(x, y, Z))
    z0.append(np.nanmedian(a)); zp.append(np.nanmedian(b))
for i in range(len(constructs)):
    ax[1].plot([0, 1], [z0[i], zp[i]], "-o", color=COL[FAMILY[constructs[i]]], lw=2, ms=7)
    ax[1].text(1.04, zp[i], LABEL[constructs[i]], va="center", fontsize=8.5, color=COL[FAMILY[constructs[i]]])
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["zero-order ρ", "partial ρ\n(control: NN-sim + density)"], fontsize=9)
ax[1].set_xlim(-0.15, 1.8); ax[1].set_ylabel("Spearman ρ vs error", fontsize=9)
ax[1].set_title("B  Roughness survives the applicability-domain control", fontsize=10.5, loc="left", weight="bold")

# Panel D
flaggers = ["sali_mean", "sali_max", "nbr_disp", "holder", "rf_var", "nn_sim"]
aucs = {}
for c in flaggers:
    a = []
    for _, sub in df.groupby("dataset"):
        s = sub[[c, "cliff_mol"]].dropna()
        if s.cliff_mol.nunique() == 2 and len(s) > 10: a.append(roc_auc_score(s.cliff_mol, s[c] * SIGN.get(c, 1)))
    aucs[c] = np.array(a)
ypos = np.arange(len(flaggers))[::-1]
ax[2].barh(ypos, [np.median(aucs[c]) for c in flaggers], color=[COL[FAMILY[c]] for c in flaggers],
           xerr=[[np.median(aucs[c])-np.percentile(aucs[c],25) for c in flaggers],
                 [np.percentile(aucs[c],75)-np.median(aucs[c]) for c in flaggers]], error_kw=dict(ecolor="#444", lw=1, capsize=3))
ax[2].axvline(0.5, color="k", lw=.8, ls="--"); ax[2].set_xlim(0.3, 0.85)
ax[2].set_yticks(ypos); ax[2].set_yticklabels([LABEL[c] for c in flaggers], fontsize=9)
ax[2].set_xlabel("AUC for flagging labelled cliffs\n(median ± IQR)", fontsize=9)
ax[2].set_title("C  Flagging cliffs a-priori (no activity used)", fontsize=10.5, loc="left", weight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "figure1_main.png"), dpi=160, bbox_inches="tight")

# ---- exact numbers for the Results prose ----
cliff = df.cliff_mol.astype(bool)
ratios = df.groupby("dataset").apply(lambda g: g[g.cliff_mol==1].rf_err.mean()/g[g.cliff_mol==0].rf_err.mean())
per_t_higher = df.groupby("dataset").apply(lambda g: g[g.cliff_mol==1].rf_err.mean() > g[g.cliff_mol==0].rf_err.mean())
print("CLIFF vs NON-CLIFF RF error:")
print(f"  overall mean: cliff={df[cliff].rf_err.mean():.3f}  non={df[~cliff].rf_err.mean():.3f}")
print(f"  median per-target ratio={ratios.median():.2f}x ; cliff>non on {int(per_t_higher.sum())}/30 targets")
print(f"  Wilcoxon (per-target cliff vs non mean): p={wilcoxon(df.groupby('dataset').apply(lambda g: g[g.cliff_mol==1].rf_err.mean()), df.groupby('dataset').apply(lambda g: g[g.cliff_mol==0].rf_err.mean())).pvalue:.1e}")
A = pd.read_csv(os.path.join(RESULTS_DIR, "tableA_spearman_vs_error.csv"))
print("\nTABLE A fdr_p:", {r.predictor: f"{r.fdr_p:.1e}" for r in A.itertuples()})
print("Panel A medians:", {LABEL[c]: round(m,3) for c, m in zip(order, meds)})
print("Panel B zero->partial:", {constructs[i]: (round(z0[i],3), round(zp[i],3)) for i in range(len(constructs))})
print("Panel D AUC:", {LABEL[c]: round(float(np.median(aucs[c])),3) for c in flaggers})
print("NN-sim RAW cliff AUC (not oriented):", round(float(np.median([roc_auc_score(s.cliff_mol, s.nn_sim) for _,sub in df.groupby('dataset') for s in [sub[['nn_sim','cliff_mol']].dropna()] if s.cliff_mol.nunique()==2])),3))
print("figure1_main.png saved")
