"""toc_graphic.py -- ACS TOC/abstract graphic (<= 3.25 x 1.75 in)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(3.25, 1.75), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 13.2); ax.axis("off")

x1 = np.linspace(0.6, 5.0, 100)
y1 = 3.2 + 0.55*np.sin(x1*0.7) + 0.18*x1
ax.plot(x1, y1, color="#1b7837", lw=3, solid_capstyle="round", zorder=3)
xc = 5.35
ax.plot([xc, xc], [y1[-1], 8.3], color="#b2182b", lw=3, solid_capstyle="round", zorder=3)
x2 = np.linspace(5.7, 9.2, 80)
y2 = 8.3 + 0.5*np.sin((x2-5.7)*0.9) - 0.05*(x2-5.7)
ax.plot(x2, y2, color="#1b7837", lw=3, solid_capstyle="round", zorder=3)
ax.axvspan(4.9, 5.8, color="#b2182b", alpha=0.10, zorder=0)

def mol(xx, yy, c):
    ax.add_patch(RegularPolygon((xx, yy), numVertices=6, radius=0.42,
                                orientation=np.pi/6, facecolor="white",
                                edgecolor=c, lw=1.6, zorder=5))
mol(4.7, y1[-1], "#333333"); mol(6.0, 8.3, "#333333")
ax.add_patch(FancyArrowPatch((4.95, y1[-1]+0.2), (5.78, 8.05),
             connectionstyle="arc3,rad=0.25", arrowstyle="-", lw=1.2,
             ls=(0,(3,2)), color="#b2182b", zorder=4))
ax.text(4.55, 6.5, "large\nΔpotency", color="#b2182b", fontsize=5.2, ha="center", style="italic")

ax.text(2.6, 1.9, "smooth -> reliable", color="#1b7837", fontsize=6.4, ha="center", weight="bold")
ax.text(5.55, 9.4, "activity cliff", color="#b2182b", fontsize=6.4, ha="center", weight="bold")
ax.text(5.55, 8.9, "high local roughness", color="#b2182b", fontsize=5.4, ha="center")

ax.annotate("", xy=(9.7, 0.7), xytext=(0.5, 0.7), arrowprops=dict(arrowstyle="->", lw=1, color="#666"))
ax.annotate("", xy=(0.5, 9.6), xytext=(0.5, 0.7), arrowprops=dict(arrowstyle="->", lw=1, color="#666"))
ax.text(5.0, 0.05, "structural similarity", fontsize=5.8, ha="center", color="#444")
ax.text(0.05, 5.0, "activity", fontsize=5.8, va="center", rotation=90, color="#444")

ax.text(5.0, 12.6, "Predicting where a QSAR model fails", fontsize=7.2, ha="center", weight="bold", color="#111")
ax.text(5.0, 11.2, "structure-only roughness flags per-compound error,", fontsize=5.5, ha="center", color="#333")
ax.text(5.0, 10.6, "independent of the applicability domain", fontsize=5.5, ha="center", color="#333")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plt.savefig(os.path.join(FIGURES_DIR, "TOC_graphic.png"), dpi=300, facecolor="white")
print("TOC_graphic.png saved")
