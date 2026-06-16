"""analyze_gnn.py -- fixed-60 vs tuned GIN, and re-check the panel-C reversal."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))  # has fixed-60 gnn_err
# pull in the tuned gnn_err (cache_gnn2, same sorted order as cache)
tuned = []
for f in sorted(x for x in os.listdir(CACHE_DIR) if x.endswith(".csv") and not x.startswith("robustness")):
    tuned.append(pd.read_csv(os.path.join(CACHE_GNN2, f)))
tuned = pd.concat(tuned, ignore_index=True)
assert len(tuned) == len(df) and (tuned.smiles.values == df.smiles.values).mean() == 1.0, "alignment!"
df["gnn_err_tuned"] = tuned.gnn_err.values
df["gap_fixed"] = df.gnn_err - df.rf_err
df["gap_tuned"] = df.gnn_err_tuned - df.rf_err

# did tuning actually help the fit?
per_t = df.groupby("dataset").agg(mae_fixed=("gnn_err","mean"), mae_tuned=("gnn_err_tuned","mean"),
                                  mae_rf=("rf_err","mean")).reset_index()
print("=== GNN fit: tuned vs fixed-60 (mean MAE) ===")
print(f"  overall mean MAE  fixed-60={df.gnn_err.mean():.3f}  tuned={df.gnn_err_tuned.mean():.3f}  RF={df.rf_err.mean():.3f}")
print(f"  tuned better on {int((per_t.mae_tuned < per_t.mae_fixed).sum())}/30 targets; "
      f"GNN still > RF on {int((per_t.mae_tuned > per_t.mae_rf).sum())}/30")

# re-check the reversal with the tuned model
print("\n=== Deep-learning gap (GNN - RF) by model ===")
for tag, gcol in [("fixed-60", "gap_fixed"), ("tuned", "gap_tuned")]:
    rhos = np.array([spearmanr(s["dirichlet"], s[gcol]).statistic
                     for _, sub in df.groupby("dataset") for s in [sub[["dirichlet", gcol]].dropna()] if len(s) > 10])
    qcol = df.groupby("dataset")["dirichlet"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1,2,3,4]))
    qg = df.groupby(qcol)[gcol].mean()
    print(f"  [{tag:8s}] mean gap cliffs={df[df.cliff_mol==1][gcol].mean():+.3f} "
          f"non={df[df.cliff_mol==0][gcol].mean():+.3f} | rho(gap,rough) med={np.median(rhos):+.3f} "
          f"wilcoxon_p={wilcoxon(rhos).pvalue:.1e} | quartiles "
          + " ".join(f"Q{q}={qg.loc[q]:+.3f}" for q in [1,2,3,4]))

# figure: MAE scatter + quartile bars, fixed vs tuned
df["rough_q"] = df.groupby("dataset")["dirichlet"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1,2,3,4]))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
lim = [0.2, 1.25]
ax[0].plot(lim, lim, "k--", lw=.8, alpha=.6)
ax[0].scatter(per_t.mae_fixed, per_t.mae_tuned, s=42, c="#762a83", edgecolors="white")
ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel("GNN test MAE — fixed 60 epochs")
ax[0].set_ylabel("GNN test MAE — tuned (early stopping)")
ax[0].set_title("A  Tuned vs fixed-60 GNN MAE\n(on/above diagonal: tuning gave no systematic gain)", fontsize=10.5, loc="left", weight="bold")
ax[0].text(0.05,0.92,"both regimes: GNN MAE > RF on 30/30",transform=ax[0].transAxes,fontsize=8,color="#555")

qf = df.groupby("rough_q").gap_fixed.mean(); qfse = df.groupby("rough_q").gap_fixed.sem()
qt = df.groupby("rough_q").gap_tuned.mean(); qtse = df.groupby("rough_q").gap_tuned.sem()
x = np.arange(4); w = 0.38
ax[1].bar(x-w/2, [qf.loc[q] for q in [1,2,3,4]], w, yerr=[qfse.loc[q] for q in [1,2,3,4]],
          label="fixed-60", color="#c2a5cf", capsize=3)
ax[1].bar(x+w/2, [qt.loc[q] for q in [1,2,3,4]], w, yerr=[qtse.loc[q] for q in [1,2,3,4]],
          label="tuned", color="#762a83", capsize=3)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Q1\nsmooth","Q2","Q3","Q4\nrough"])
ax[1].axhline(0, color="k", lw=.6); ax[1].set_ylabel("mean (GNN error − RF error)")
ax[1].set_title("B  Gap still shrinks from smooth→rough after tuning", fontsize=10.5, loc="left", weight="bold")
ax[1].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "gnn_tuned_figure.png"), dpi=150, bbox_inches="tight")
per_t.to_csv(os.path.join(RESULTS_DIR, "gnn_fixed_vs_tuned.csv"), index=False)
print("\nsaved gnn_tuned_figure.png, gnn_fixed_vs_tuned.csv")
