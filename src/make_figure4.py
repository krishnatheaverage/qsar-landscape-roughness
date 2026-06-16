"""make_figure4.py -- cross-domain validation figure (Figure 4).
grouped bars of zero-order vs AD-controlled partial Spearman, for the landscape
Dirichlet energy (panel A) and the activity-free neighbour dispersion (panel B),
across the 30-target bioactivity median and the two MoleculeNet physchem sets.
reads cross_domain_results.csv (ESOL + Lipophilicity) and tableA/tableB for the
bioactivity medians, so run after cross_domain.py and analyze.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, FIGURES_DIR
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cd = pd.read_csv(os.path.join(RESULTS_DIR, "cross_domain_results.csv")).set_index("dataset")
A  = pd.read_csv(os.path.join(RESULTS_DIR, "tableA_spearman_vs_error.csv")).set_index("predictor")
B  = pd.read_csv(os.path.join(RESULTS_DIR, "tableB_partial_AD.csv")).set_index("predictor")

esol, lipo = "ESOL (logS solubility)", "Lipophilicity (logD)"
groups = ["Bioactivity\n(30-target median)", "ESOL\n(solubility)", "Lipophilicity\n(logD)"]

# panel A: landscape Dirichlet energy; panel B: activity-free neighbour dispersion
dir_zero  = [A.loc["dirichlet", "median_rho"], cd.loc[esol, "rho_dirichlet"], cd.loc[lipo, "rho_dirichlet"]]
dir_part  = [B.loc["dirichlet", "partial_AD"], cd.loc[esol, "partial_dir_AD"], cd.loc[lipo, "partial_dir_AD"]]
disp_zero = [A.loc["nbr_disp", "median_rho"],  cd.loc[esol, "rho_nbr_disp"],  cd.loc[lipo, "rho_nbr_disp"]]
disp_part = [B.loc["nbr_disp", "partial_AD"],  cd.loc[esol, "partial_disp_AD"], cd.loc[lipo, "partial_disp_AD"]]

def lighten(hexc, f=0.5):
    h = hexc.lstrip("#"); rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in rgb)

panels = [("A  Landscape roughness (Dirichlet)",          "#b2182b", dir_zero,  dir_part),
          ("B  Activity-free roughness (nbr dispersion)", "#2166ac", disp_zero, disp_part)]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
x, w = np.arange(len(groups)), 0.38
for a, (title, color, zero, part) in zip(ax, panels):
    a.bar(x - w/2, zero, w, color=color, label="zero-order ρ")
    a.bar(x + w/2, part, w, facecolor=lighten(color), hatch="///",
          edgecolor="black", linewidth=0.7, label="partial ρ | AD")
    a.set_xticks(x); a.set_xticklabels(groups, fontsize=9)
    a.axhline(0, color="k", lw=0.6)
    a.set_ylim(0, 0.75)
    a.set_title(title, fontsize=11, loc="left", weight="bold")
    a.legend(frameon=False, fontsize=9.5,
             loc="upper center" if title.startswith("A") else "upper right")
ax[0].set_ylabel("Spearman ρ vs per-compound error", fontsize=10)
fig.suptitle("External validation: the roughness–error relationship replicates outside bioactivity",
             fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "figure4_crossdomain.png"), dpi=160, bbox_inches="tight")
print("figure4_crossdomain.png saved")

# tidy CSV of the exact plotted bar heights, so the R/ggplot2 port reads one clean
# source that mirrors panel A/B bar-for-bar (no values re-derived downstream).
group_labels = ["Bioactivity (30-target median)", "ESOL (solubility)", "Lipophilicity (logD)"]
_rows = []
for gi, glab in enumerate(group_labels):
    _rows.append(dict(panel="A", panel_label="Landscape roughness (Dirichlet)",
                      group=glab, group_order=gi, series="zero-order", value=dir_zero[gi]))
    _rows.append(dict(panel="A", panel_label="Landscape roughness (Dirichlet)",
                      group=glab, group_order=gi, series="partial|AD", value=dir_part[gi]))
    _rows.append(dict(panel="B", panel_label="Activity-free roughness (nbr dispersion)",
                      group=glab, group_order=gi, series="zero-order", value=disp_zero[gi]))
    _rows.append(dict(panel="B", panel_label="Activity-free roughness (nbr dispersion)",
                      group=glab, group_order=gi, series="partial|AD", value=disp_part[gi]))
pd.DataFrame(_rows).to_csv(os.path.join(RESULTS_DIR, "figure4_plot_values.csv"), index=False)
print("figure4_plot_values.csv saved")
