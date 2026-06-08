"""
conformal.py -- roughness-aware (cliff-aware) conformal prediction intervals.

Split-conformal regression on the held-out per-compound residuals. For each target
the test compounds are split 50/50 into a calibration and an evaluation set; 90%
prediction intervals are built three ways and scored for marginal, activity-cliff,
and non-cliff coverage, plus interval width:

  standard          nonconformity = |residual|              -> constant-width intervals
  roughness (SALI)  nonconformity = |residual| / sigma(x)   -> sigma = activity-free SALI density
  AD (1 - NN sim)   nonconformity = |residual| / sigma(x)   -> sigma = applicability-domain risk

Both sigma choices are activity-free, so the predictor is deployable before assay.
Averaged over 50 random calibration/evaluation splits and reported as the median
across the 30 targets. Reads results/all_per_compound.csv (rf_err = |pred - y| plus
the roughness / AD columns); writes results/conformal_results.csv and
paper/figures/figure6_conformal.png.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ALPHA = 0.10   # 90% prediction intervals
REPS  = 50
df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))

METHODS = [("standard", None, 1),
           ("roughness (SALI)", "sali_mean", 1),
           ("AD (1-NN sim)", "nn_sim", -1)]

def conformal_q(scores, alpha):
    n = len(scores); k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(scores)[k - 1]

def sigma_for(g, col, sign):
    """Per-compound nonconformity scale (>0). sign flips AD so higher = riskier."""
    if col is None:
        return np.ones(len(g))
    s = sign * g[col].values.astype(float)
    s = np.where(np.isfinite(s), s, np.nanmedian(s))
    s = s - np.nanmin(s)
    return s + 0.1 * (np.nanmedian(s) + 1e-9)   # floor keeps widths well-defined

def evaluate():
    rec = {name: [] for name, _, _ in METHODS}
    for rep in range(REPS):
        rng = np.random.RandomState(rep)
        for ds, g in df.groupby("dataset"):
            g = g.reset_index(drop=True)
            resid = g["rf_err"].values
            cliff = g["cliff_mol"].values.astype(bool)
            idx = rng.permutation(len(g)); h = len(g) // 2
            cal, ev = idx[:h], idx[h:]
            if len(cal) < 10 or len(ev) < 10:
                continue
            wstd = None
            for name, col, sign in METHODS:                 # standard is first -> wstd set
                sig = sigma_for(g, col, sign)
                q = conformal_q(resid[cal] / sig[cal], ALPHA)
                hw = q * sig[ev]
                cov = resid[ev] <= hw
                w = 2 * hw
                ce = cliff[ev]
                if name == "standard":
                    wstd = w
                rec[name].append(dict(
                    dataset=ds,
                    cov=cov.mean(),
                    cov_cliff=cov[ce].mean() if ce.sum() >= 5 else np.nan,
                    cov_noncliff=cov[~ce].mean() if (~ce).sum() >= 5 else np.nan,
                    relwidth=np.median(w) / np.median(wstd)))
    std_cliff = pd.DataFrame(rec["standard"]).groupby("dataset")["cov_cliff"].mean()
    out = []
    for name, _, _ in METHODS:
        t = pd.DataFrame(rec[name]).groupby("dataset").mean(numeric_only=True)
        out.append(dict(
            method=name,
            marginal_cov=round(float(t["cov"].median()), 3),
            cliff_cov=round(float(t["cov_cliff"].median()), 3),
            noncliff_cov=round(float(t["cov_noncliff"].median()), 3),
            median_rel_width=round(float(t["relwidth"].median()), 2),
            frac_targets_cliff_improved=round(float((t["cov_cliff"] > std_cliff).mean()), 2)))
    return pd.DataFrame(out)

res = evaluate()
res.to_csv(os.path.join(RESULTS_DIR, "conformal_results.csv"), index=False)
print(res.to_string(index=False))

# ---- figure ----
COL = {"standard": "#555555", "roughness (SALI)": "#2166ac", "AD (1-NN sim)": "#878787"}
methods = res.method.tolist()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})

xg = np.arange(2); w = 0.26
for i, m in enumerate(methods):
    r = res[res.method == m].iloc[0]
    ax[0].bar(xg + (i - 1) * w, [r["cliff_cov"], r["noncliff_cov"]], w, color=COL[m], label=m)
ax[0].axhline(0.90, ls="--", color="k", lw=1)
ax[0].text(1.38, 0.903, "nominal 90%", fontsize=8, ha="right")
ax[0].set_xticks(xg); ax[0].set_xticklabels(["Activity cliffs", "Non-cliff"])
ax[0].set_ylim(0.75, 1.0); ax[0].set_ylabel("90% interval coverage")
ax[0].set_title("A  Coverage by compound type", loc="left", weight="bold", fontsize=11)
ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")

ax[1].bar(range(len(methods)), [res[res.method == m].iloc[0]["median_rel_width"] for m in methods],
          color=[COL[m] for m in methods])
ax[1].axhline(1.0, ls="--", color="k", lw=1)
ax[1].set_xticks(range(len(methods)))
ax[1].set_xticklabels(["standard", "roughness\n(SALI)", "AD\n(1-NN sim)"], fontsize=8.5)
ax[1].set_ylabel("median interval width (relative to standard)")
ax[1].set_title("B  Interval width", loc="left", weight="bold", fontsize=11)

fig.suptitle("Roughness-aware conformal intervals restore coverage on activity cliffs",
             weight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "figure6_conformal.png"), dpi=160, bbox_inches="tight")
print("figure6_conformal.png saved")
