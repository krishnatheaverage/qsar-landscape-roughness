"""
conformal.py -- roughness-conditioned conformal prediction intervals.

The question is not how to normalize a conformal predictor but which per-compound
variable to condition it on. Split-conformal regression gives a marginal coverage
guarantee; locally-weighted (normalized) and Mondrian variants make coverage
conditional on a per-compound difficulty estimate but leave the choice of estimate
open. We show the activity-free roughness (the pairwise-SALI density) is that estimate.

For each target the test compounds are split 50/50 into calibration and evaluation,
and 90% intervals are built five ways from the held-out residuals:
  standard        constant width (global conformal quantile)
  rough-norm      locally-weighted: width scaled by sigma = SALI density
  AD-norm         locally-weighted: width scaled by sigma = applicability-domain risk
  rough-Mondrian  stratify calibration into SALI-density quintiles, calibrate within
  AD-Mondrian     stratify calibration into applicability-domain quintiles, calibrate within
Both conditioning variables are activity-free (available before assay). We report
marginal, activity-cliff, and non-cliff coverage, interval width, and coverage as a
function of roughness quantile, averaged over 50 splits and aggregated as the median
across the 30 targets. Reads results/all_per_compound.csv; writes
results/conformal_results.csv, results/conformal_curve.csv, paper/figures/figure6_conformal.png.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ALPHA = 0.10      # 90% intervals
REPS  = 50
NB    = 5         # roughness/AD strata and curve quantiles
df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))
CURVE_METHODS = ["standard", "rough-Mondrian", "AD-Mondrian"]

def cq(scores, alpha):
    s = np.sort(scores); n = len(s)
    if n == 0: return np.inf
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return s[k - 1]

def cleanv(g, col, sign):
    v = sign * g[col].values.astype(float)
    return np.where(np.isfinite(v), v, np.nanmedian(v))

def inner_quantile_bins(v_cal, v_pts, nb):
    thr = np.quantile(v_cal, np.linspace(0, 1, nb + 1)[1:-1])
    return np.minimum(np.digitize(v_pts, thr), nb - 1)

rows, curve_rows = [], []
for rep in range(REPS):
    rng = np.random.RandomState(rep)
    for ds, g in df.groupby("dataset"):
        g = g.reset_index(drop=True)
        resid = g["rf_err"].values
        cliff = g["cliff_mol"].values.astype(bool)
        idx = rng.permutation(len(g)); h = len(g) // 2
        cal, ev = idx[:h], idx[h:]
        if len(cal) < NB * 8 or len(ev) < NB * 8:
            continue
        rough = cleanv(g, "sali_mean", 1)
        ad = cleanv(g, "nn_sim", -1)
        ev_rbin = inner_quantile_bins(rough[ev], rough[ev], NB)   # roughness quintile of each eval point

        methods = {}
        qg = cq(resid[cal], ALPHA)
        methods["standard"] = (resid[ev] <= qg, np.full(len(ev), 2 * qg))
        for nm, col, sign in [("rough-norm", "sali_mean", 1), ("AD-norm", "nn_sim", -1)]:
            s = cleanv(g, col, sign); s = s - np.nanmin(s)
            sig = s + 0.1 * (np.nanmedian(s) + 1e-9)
            qn = cq(resid[cal] / sig[cal], ALPHA)
            methods[nm] = (resid[ev] <= qn * sig[ev], 2 * qn * sig[ev])
        for nm, var in [("rough-Mondrian", rough), ("AD-Mondrian", ad)]:
            cb = inner_quantile_bins(var[cal], var[cal], NB)
            eb = inner_quantile_bins(var[cal], var[ev], NB)
            qb = {b: cq(resid[cal][cb == b], ALPHA) for b in range(NB)}
            hw = np.array([qb[b] for b in eb])
            methods[nm] = (resid[ev] <= hw, 2 * hw)

        wstd = methods["standard"][1]
        for nm, (cov, w) in methods.items():
            rows.append(dict(method=nm, dataset=ds, marginal=cov.mean(),
                cliff=cov[cliff[ev]].mean() if cliff[ev].sum() >= 5 else np.nan,
                noncliff=cov[~cliff[ev]].mean() if (~cliff[ev]).sum() >= 5 else np.nan,
                relwidth=np.median(w) / np.median(wstd)))
            if nm in CURVE_METHODS:
                for b in range(NB):
                    mm = ev_rbin == b
                    if mm.sum() > 0:
                        curve_rows.append(dict(method=nm, dataset=ds, qbin=b, cov=cov[mm].mean()))

R = pd.DataFrame(rows)
per = R.groupby(["method", "dataset"]).mean(numeric_only=True).reset_index()
std_cliff = per[per.method == "standard"].set_index("dataset")["cliff"]
ORDER = ["standard", "rough-norm", "AD-norm", "rough-Mondrian", "AD-Mondrian"]
summary = []
for m in ORDER:
    t = per[per.method == m].set_index("dataset")
    summary.append(dict(method=m,
        marginal_cov=round(float(t["marginal"].median()), 3),
        cliff_cov=round(float(t["cliff"].median()), 3),
        noncliff_cov=round(float(t["noncliff"].median()), 3),
        median_rel_width=round(float(t["relwidth"].median()), 2),
        frac_targets_cliff_improved=round(float((t["cliff"] > std_cliff).mean()), 2)))
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join(RESULTS_DIR, "conformal_results.csv"), index=False)
print(summary.to_string(index=False))

C = pd.DataFrame(curve_rows).groupby(["method", "qbin"])["cov"].mean().reset_index()
C.to_csv(os.path.join(RESULTS_DIR, "conformal_curve.csv"), index=False)

# ---- figure ----
COL = {"standard": "#555555", "rough-Mondrian": "#2166ac", "AD-Mondrian": "#878787"}
LAB = {"standard": "standard (unconditional)", "rough-Mondrian": "roughness-conditioned",
       "AD-Mondrian": "applicability-domain-conditioned"}
fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.5, 1]})

xq = np.arange(NB)
for m in CURVE_METHODS:
    y = [C[(C.method == m) & (C.qbin == b)]["cov"].iloc[0] for b in range(NB)]
    ax[0].plot(xq, y, "-o", color=COL[m], label=LAB[m], lw=2, ms=6)
ax[0].axhline(0.90, ls="--", color="k", lw=1); ax[0].text(NB - 1, 0.905, "nominal 90%", fontsize=8, ha="right")
ax[0].set_xticks(xq); ax[0].set_xticklabels(["Q1\n(smooth)", "Q2", "Q3", "Q4", "Q5\n(rough)"])
ax[0].set_xlabel("local roughness quantile"); ax[0].set_ylabel("90% interval coverage")
ax[0].set_title("A  Coverage across the roughness range", loc="left", weight="bold", fontsize=11)
ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")

wm = ["standard", "rough-Mondrian", "AD-Mondrian"]
ax[1].bar(range(len(wm)), [summary[summary.method == m]["median_rel_width"].iloc[0] for m in wm],
          color=[COL[m] for m in wm])
ax[1].axhline(1.0, ls="--", color="k", lw=1)
ax[1].set_xticks(range(len(wm))); ax[1].set_xticklabels(["standard", "roughness", "AD"], fontsize=9)
ax[1].set_ylabel("median interval width (relative to standard)")
ax[1].set_title("B  Interval width", loc="left", weight="bold", fontsize=11)

fig.suptitle("Conditioning conformal intervals on roughness keeps coverage valid where the landscape is rough",
             weight="bold", fontsize=11.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "figure6_conformal.png"), dpi=160, bbox_inches="tight")
print("figure6_conformal.png + conformal_curve.csv saved")
