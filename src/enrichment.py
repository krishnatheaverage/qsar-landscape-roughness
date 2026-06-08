"""
enrichment.py -- operational triage value of a structure-only roughness flag.

Question: if a modeler flags the top f% of predictions as low-confidence, what fraction
of the genuinely problematic compounds is captured, and how does roughness compare with
the standard reliability signals used in their intended direction?

Two targets to enrich (within each of the 30 tasks, then averaged across tasks):
  - high-error compounds  : top-quartile per-compound RF error
  - labelled activity cliffs

Risk scores (higher = flagged first), each in its intended orientation:
  - roughness     : nbr_disp (high-error task) / sali_mean (cliff task)   [y-free]
  - applic. domain: -nn_sim  (low similarity = risky; standard AD orientation)
  - uncertainty   : rf_var   (high tree variance = risky)
  - random        : shuffled
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)

df = pd.read_csv(os.path.join(RESULTS_DIR, "all_per_compound.csv"))
FRACS = np.arange(0.05, 0.51, 0.05)

def recall_curve(score_col, sign, pos_mask_fn):
    """Avg recall vs fraction flagged, across targets; score risk = sign*column."""
    curves = []
    for _, g in df.groupby("dataset"):
        pos = pos_mask_fn(g)
        if pos.sum() < 3 or (~pos).sum() < 3: continue
        if score_col == "random":
            risk = np.random.rand(len(g))
        else:
            risk = sign * g[score_col].values
        order = np.argsort(-risk)  # most-risky first
        pos_ord = pos.values[order]
        rec = [pos_ord[:max(1, int(round(f*len(g))))].sum() / pos.sum() for f in FRACS]
        curves.append(rec)
    return np.array(curves).mean(0)

def highedge(g): return g.rf_err >= g.rf_err.quantile(0.75)
def cliffs(g):   return g.cliff_mol == 1

scorers_err = [("roughness (nbr_disp)", "nbr_disp", 1, "#2166ac"),
               ("applicability domain", "nn_sim", -1, "#878787"),
               ("uncertainty (RF var)", "rf_var", 1, "#1b7837"),
               ("random", "random", 1, "#bbbbbb")]
scorers_cliff = [("roughness (SALI-density)", "sali_mean", 1, "#2166ac"),
                 ("applicability domain", "nn_sim", -1, "#878787"),
                 ("uncertainty (RF var)", "rf_var", 1, "#1b7837"),
                 ("random", "random", 1, "#bbbbbb")]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
print("=== Recall of positives captured by flagging the top f% (avg over 30 targets) ===")
for axi, (title, scorers, posfn) in enumerate([
        ("A  Catching high-error predictions", scorers_err, highedge),
        ("B  Catching activity cliffs", scorers_cliff, cliffs)]):
    print(f"\n[{title}]")
    for name, col, sign, color in scorers:
        rc = recall_curve(col, sign, posfn)
        ax[axi].plot(FRACS*100, rc*100, "-o", ms=4, color=color, label=name)
        i20 = np.argmin(np.abs(FRACS - 0.20))
        print(f"  {name:26s} recall@20%flag = {rc[i20]*100:4.1f}%  (EF20 = {rc[i20]/0.20:.2f})")
    ax[axi].plot([0,50],[0,50],"k:",lw=.7)
    ax[axi].set_xlabel("% of compounds flagged as low-confidence")
    ax[axi].set_ylabel("% of target compounds captured (recall)")
    ax[axi].set_title(title, fontsize=10.5, loc="left", weight="bold")
    ax[axi].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("Operational triage: a structure-only roughness flag captures more problem compounds than applicability domain or uncertainty",
             fontsize=11, weight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR, "figure5_enrichment.png"), dpi=160, bbox_inches="tight")
print("\nfigure5_enrichment.png saved")
