"""
conformal.py -- which per-compound variable should a conformal predictor be conditioned on?

Random-forest tree variance is the strongest general-purpose reliability signal (it ranks
top-quartile errors well) but is at chance on activity cliffs; the applicability domain is
oriented backward for cliffs. We test whether the activity-free local roughness (the pairwise
SALI density) is the per-compound scale that captures the cliff failure these miss, and whether
it complements rather than replaces variance.

Two analyses on results/all_per_compound.csv (rf_err = |pred - y| plus roughness / variance /
AD columns):

(1) Complementarity of risk scores. Per-target ROC-AUC for flagging top-quartile-error compounds
    and labeled activity cliffs, for tree variance, roughness, and a combined score
    (within-target standardized variance + roughness). -> complementarity_results.csv

(2) Conditional conformal coverage. Split-conformal 90% intervals from the held-out residuals
    (test compounds split 50/50 into calibration/evaluation). An unconditional predictor is
    compared with Mondrian predictors that stratify calibration into quintiles of a conditioning
    variable -- tree variance, roughness, the applicability domain, or the combined score -- and
    with locally-weighted (normalized) scaling by roughness or AD. We report marginal, cliff, and
    non-cliff coverage, interval width, and coverage as a function of roughness quantile.
    -> conformal_results.csv, conformal_curve.csv, paper/figures/figure6_conformal.png

Averaged over 50 random splits, aggregated as the median across the 30 targets.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR
import numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ALPHA = 0.10      # 90% intervals
REPS  = 50
NB    = 5         # conditioning strata and roughness-curve quantiles
df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))

def z(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)

def clean(v):
    v = np.asarray(v, float)
    return np.where(np.isfinite(v), v, np.nanmedian(v))

def auc(score, y):
    score = np.asarray(score, float); y = np.asarray(y).astype(bool)
    ok = np.isfinite(score); score, y = score[ok], y[ok]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos < 5 or nneg < 5: return np.nan
    r = rankdata(score)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def cq(scores, alpha):
    s = np.sort(scores); n = len(s)
    if n == 0: return np.inf
    return s[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1]

def qbins(v_cal, v_pts, nb):
    thr = np.quantile(v_cal, np.linspace(0, 1, nb + 1)[1:-1])
    return np.minimum(np.digitize(v_pts, thr), nb - 1)

# ---------- (1) complementarity of flagging scores ----------
crows = []
for ds, g in df.groupby("dataset"):
    g = g.reset_index(drop=True)
    var = g["rf_var"].values; rough = g["sali_mean"].values
    comb = z(clean(var)) + z(clean(rough))
    hi = (g["rf_err"] >= g["rf_err"].quantile(0.75)).values
    cl = (g["cliff_mol"] == 1).values
    crows.append(dict(dataset=ds,
        err_variance=auc(var, hi),  err_roughness=auc(rough, hi),  err_combined=auc(comb, hi),
        cliff_variance=auc(var, cl), cliff_roughness=auc(rough, cl), cliff_combined=auc(comb, cl)))
C = pd.DataFrame(crows)
comp = pd.DataFrame([
    dict(target="top-quartile error", variance=C.err_variance.median(),
         roughness=C.err_roughness.median(), combined=C.err_combined.median()),
    dict(target="activity cliffs", variance=C.cliff_variance.median(),
         roughness=C.cliff_roughness.median(), combined=C.cliff_combined.median()),
]).round(3)
comp.to_csv(os.path.join(RESULTS_DIR, "complementarity_results.csv"), index=False)
print("== flagging ROC-AUC (median over 30 targets) ==")
print(comp.to_string(index=False))
print("combined > variance on cliffs:", int((C.cliff_combined > C.cliff_variance).sum()), "/ 30 targets\n")

# ---------- (2) conditional conformal coverage ----------
# Mondrian conditioning variables (all activity-free / available before assay)
MOND = ["variance", "roughness", "AD", "combined"]
NORM = ["roughness", "AD"]            # locally-weighted (normalized) scales
crv_methods = ["standard", "variance", "AD", "roughness"]   # lines drawn in the figure
rows, curve_rows = [], []
for rep in range(REPS):
    rng = np.random.RandomState(rep)
    for ds, g in df.groupby("dataset"):
        g = g.reset_index(drop=True)
        resid = g["rf_err"].values; cliff = g["cliff_mol"].values.astype(bool)
        var = clean(g["rf_var"].values); rough = clean(g["sali_mean"].values)
        ad = -clean(g["nn_sim"].values)                  # higher = more AD risk
        comb = z(var) + z(rough)
        VAR = {"variance": var, "roughness": rough, "AD": ad, "combined": comb}
        idx = rng.permutation(len(g)); h = len(g) // 2; cal, ev = idx[:h], idx[h:]
        if len(cal) < NB * 8 or len(ev) < NB * 8: continue
        ev_rq = qbins(rough[ev], rough[ev], NB)          # roughness quintile of each eval point

        methods = {}
        qg = cq(resid[cal], ALPHA)
        methods["standard"] = (resid[ev] <= qg, np.full(len(ev), 2 * qg))
        for nm in MOND:
            v = VAR[nm]; cb = qbins(v[cal], v[cal], NB); eb = qbins(v[cal], v[ev], NB)
            qb = {b: cq(resid[cal][cb == b], ALPHA) for b in range(NB)}
            hw = np.array([qb[b] for b in eb])
            methods[nm] = (resid[ev] <= hw, 2 * hw)
        for nm in NORM:
            s = VAR[nm]; s = s - np.nanmin(s); sig = s + 0.1 * (np.nanmedian(s) + 1e-9)
            qn = cq(resid[cal] / sig[cal], ALPHA)
            methods[nm + "-norm"] = (resid[ev] <= qn * sig[ev], 2 * qn * sig[ev])

        wstd = methods["standard"][1]
        for nm, (cov, w) in methods.items():
            rows.append(dict(method=nm, dataset=ds, marginal=cov.mean(),
                cliff=cov[cliff[ev]].mean() if cliff[ev].sum() >= 5 else np.nan,
                noncliff=cov[~cliff[ev]].mean() if (~cliff[ev]).sum() >= 5 else np.nan,
                relwidth=np.median(w) / np.median(wstd)))
            if nm in crv_methods:
                for b in range(NB):
                    m = ev_rq == b
                    if m.sum() > 0:
                        curve_rows.append(dict(method=nm, dataset=ds, qbin=b, cov=cov[m].mean()))

per = pd.DataFrame(rows).groupby(["method", "dataset"]).mean(numeric_only=True).reset_index()
std_cliff = per[per.method == "standard"].set_index("dataset")["cliff"]
ORDER = ["standard", "variance", "AD", "roughness", "combined", "roughness-norm", "AD-norm"]
LABELS = {"standard": "standard (unconditional)", "variance": "Mondrian: tree variance",
          "AD": "Mondrian: applicability domain", "roughness": "Mondrian: roughness",
          "combined": "Mondrian: variance + roughness",
          "roughness-norm": "locally-weighted: roughness", "AD-norm": "locally-weighted: AD"}
summ = []
for m in ORDER:
    t = per[per.method == m].set_index("dataset")
    summ.append(dict(method=LABELS[m],
        marginal_cov=round(float(t["marginal"].median()), 3),
        cliff_cov=round(float(t["cliff"].median()), 3),
        noncliff_cov=round(float(t["noncliff"].median()), 3),
        median_rel_width=round(float(t["relwidth"].median()), 2),
        frac_targets_cliff_improved=round(float((t["cliff"] > std_cliff).mean()), 2)))
summary = pd.DataFrame(summ)
summary.to_csv(os.path.join(RESULTS_DIR, "conformal_results.csv"), index=False)
print("== conformal coverage (median over 30 targets) ==")
print(summary.to_string(index=False))

Cu = pd.DataFrame(curve_rows).groupby(["method", "qbin"])["cov"].mean().reset_index()
Cu.to_csv(os.path.join(RESULTS_DIR, "conformal_curve.csv"), index=False)

# ---------- figure ----------
COL = {"standard": "#555555", "variance": "#1b7837", "AD": "#878787", "roughness": "#2166ac"}
LAB = {"standard": "unconditional", "variance": "tree variance", "AD": "applicability domain",
       "roughness": "roughness"}
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7), gridspec_kw={"width_ratios": [1.55, 1]})

xq = np.arange(NB)
for m in crv_methods:
    y = [Cu[(Cu.method == m) & (Cu.qbin == b)]["cov"].iloc[0] for b in range(NB)]
    ax[0].plot(xq, y, "-o", color=COL[m], lw=2, ms=6, label=LAB[m])
ax[0].axhline(0.90, ls="--", color="k", lw=1); ax[0].text(NB - 1, 0.905, "nominal 90%", fontsize=8, ha="right")
ax[0].set_xticks(xq); ax[0].set_xticklabels(["Q1\n(smooth)", "Q2", "Q3", "Q4", "Q5\n(rough)"])
ax[0].set_xlabel("local roughness quantile"); ax[0].set_ylabel("90% interval coverage")
ax[0].set_title("A  Coverage across the roughness range", loc="left", weight="bold", fontsize=11)
ax[0].legend(frameon=False, fontsize=8.3, loc="lower left", title="conditioning variable", title_fontsize=8.3)

barm = ["standard", "variance", "AD", "roughness", "combined"]
bcol = {"standard": "#555555", "variance": "#1b7837", "AD": "#878787",
        "roughness": "#2166ac", "combined": "#762a83"}
cliffv = [per[per.method == m].set_index("dataset")["cliff"].median() for m in barm]
ax[1].bar(range(len(barm)), cliffv, color=[bcol[m] for m in barm])
ax[1].axhline(0.90, ls="--", color="k", lw=1)
ax[1].set_ylim(0.80, 0.93)
ax[1].set_xticks(range(len(barm)))
ax[1].set_xticklabels(["none", "var.", "AD", "rough.", "var.+\nrough."], fontsize=8.5)
ax[1].set_ylabel("coverage on activity cliffs")
ax[1].set_title("B  Cliff coverage by conditioning", loc="left", weight="bold", fontsize=11)

fig.suptitle("Roughness, not tree variance or the applicability domain, is the per-compound scale that keeps coverage valid on cliffs",
             weight="bold", fontsize=10.8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "figure6_conformal.png"), dpi=160, bbox_inches="tight")
print("\nfigure6_conformal.png + conformal_curve.csv + complementarity_results.csv saved")
