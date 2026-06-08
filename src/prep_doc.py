# -*- coding: utf-8 -*-
"""prep_doc.py -- turn manuscript.md into manuscript.json for the docx builder."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import re, json, os

SUP = lambda n: f"\u27e6{n}\u27e7"  # superscript marker ⟦n⟧
src = open(os.path.join(PAPER_DIR, "manuscript.md"), encoding="utf-8").read()

# citations -> superscripts (numbered in order of appearance)
CITES = [
    # grouped first
    (" (Vamathevan et al., 2019; Walters and Barzilay, 2021)", SUP("1,2")),
    (" (Maggiora, 2006; Stumpfe and Bajorath, 2012)", SUP("4,5")),
    (" (van Tilborg et al., 2022; Dablander et al., 2023)", SUP("7,8")),
    (" (Netzeva et al., 2005; Sahigara et al., 2012)", SUP("14,15")),
    (" (Norinder et al., 2014; Hirschfeld et al., 2020)", SUP("16,17")),
    ("van Tilborg et al. (2022)", "van Tilborg et al." + SUP("7")),
    # label-form (tail "; cite)")
    ("; Delaney, 2004)", SUP("19") + ")"),
    ("; Wu et al., 2018)", SUP("20") + ")"),
    ("; Rogers and Hahn, 2010)", SUP("22") + ")"),
    ("; Guha and Van Drie, 2008)", SUP("13") + ")"),
    ("; Xu et al., 2019)", SUP("24") + ")"),
    # singletons
    (" (Johnson and Maggiora, 1990)", SUP("3")),
    (" (Cruz-Monteagudo et al., 2014)", SUP("6")),
    (" (Aldeghi et al., 2022)", SUP("9")),
    (" (Graff et al., 2023)", SUP("10")),
    (" (Golbraikh et al., 2014)", SUP("11")),
    (" (Peltason and Bajorath, 2007)", SUP("12")),
    (" (Guha and Van Drie, 2008)", SUP("13")),
    (" (van Tilborg et al., 2022)", SUP("7")),
    (" (Gaulton et al., 2012)", SUP("18")),
    (" (Bemis and Murcko, 1996)", SUP("21")),
    (" (Breiman, 2001)", SUP("23")),
    (" (Benjamini and Hochberg, 1995)", SUP("25")),
]

# grab the body (Introduction up to References)
body = src[src.index("## Introduction"):src.index("## References")]
for old, new in CITES:
    body = body.replace(old, new)

# make sure no author-year cites slipped through (the "(2026.03)" version string is fine)
leftover = [m for m in re.findall(r"\([^()]*(?:19|20)\d\d[^()]*\)", body) if m != "(2026.03)"]
assert not leftover, f"unconverted citations: {leftover}"

# renumber figures to order of appearance (swap via temp placeholders)
body = body.replace("Figure 5", "\x01").replace("Figure 2", "\x02").replace("Figure 3", "\x03")
body = body.replace("\x01", "Figure 2").replace("\x02", "Figure 3").replace("\x03", "Figure 5")

FIGS = {
    "Activity-free flagging of cliff-prone compounds": dict(num=1, path=os.path.join(FIGURES_DIR, "figure1_main.png"),
        caption="Figure 1. Local structure\u2013activity landscape roughness predicts per-compound QSAR error across 30 bioactivity targets. (A) Median Spearman correlation between each descriptor and per-compound random-forest error (median \u00b1 interquartile range over 30 targets); landscape descriptors (red) use the query compound\u2019s own activity, the activity-free descriptors (blue) do not, and applicability-domain (gray) and uncertainty (green) baselines are shown for reference. (B) Zero-order versus applicability-domain-controlled partial correlations; the relationship is retained or strengthened under control. (C) Area under the ROC curve for flagging benchmark-labeled activity cliffs from activity-free descriptors; nearest-neighbor similarity (gray) falls below 0.5 in the extrapolation orientation because cliffs cluster among close analogs."),
    "Operational value: triaging unreliable predictions": dict(num=2, path=os.path.join(FIGURES_DIR, "figure5_enrichment.png"),
        caption="Figure 2. Operational triage value of a structure-only roughness flag. Fraction of (A) high-error predictions (top-quartile per-compound error) and (B) labeled activity cliffs recovered as a function of the percentage of compounds flagged as low-confidence, averaged over 30 targets, ranking by roughness (blue), the applicability domain in its standard orientation (gray), random-forest tree variance (green), or at random (dotted)."),
    "Robustness to neighborhood size and distance metric": dict(num=3, path=os.path.join(FIGURES_DIR, "robustness_figure.png"),
        caption="Figure 3. Stability of the roughness\u2013error relationship across neighborhood size k and molecular distance metric. Median Spearman correlation over 30 targets between per-compound random-forest error and (left) a landscape roughness measure, (center) the activity-free neighborhood dispersion, and (right) the neighborhood dispersion after controlling for the applicability domain, for ECFP4/Tanimoto, ECFP6/Tanimoto, and a 12-descriptor Euclidean representation."),
    "External validation beyond bioactivity": dict(num=4, path=os.path.join(FIGURES_DIR, "figure4_crossdomain.png"),
        caption="Figure 4. External validation on physicochemical regression benchmarks. Zero-order and applicability-domain-controlled (partial) Spearman correlations between roughness and per-compound error for (A) the landscape Dirichlet energy and (B) the activity-free neighborhood dispersion, for the 30-target bioactivity median and for aqueous solubility (ESOL) and lipophilicity (logD)."),
    "Where a graph-convolutional model underperforms": dict(num=5, path=os.path.join(FIGURES_DIR, "gnn_tuned_figure.png"),
        caption="Figure 5. Comparison with a graph-convolutional baseline. (A) Per-target test mean absolute error of the graph isomorphism network under a fixed 60-epoch schedule versus validated early stopping; both regimens exceed the random-forest error on all 30 targets. (B) Mean (network \u2212 random forest) per-compound error by within-target roughness quartile (Q1 smoothest to Q4 roughest) for both regimens; the deficit is largest where the landscape is smooth."),
}
for f in FIGS.values():
    assert os.path.exists(f["path"]), f["path"]

# inline tokenizer (bold/italic/sub/superscript)
def tok(s):
    """** bold, * italic, _x/_{..} subscript, ^x/^{..} superscript, U+27E6..U+27E7 citation."""
    runs = []; i = 0; n = len(s); bold = False; ital = False
    while i < n:
        c = s[i]
        if c == "\u27e6":
            j = s.index("\u27e7", i); runs.append({"t": s[i+1:j], "sup": True}); i = j + 1; continue
        if s[i:i+2] == "**":
            bold = not bold; i += 2; continue
        if c == "*":
            ital = not ital; i += 1; continue
        if c == "_" or c == "^":
            is_sup = (c == "^"); i += 1
            if i < n and s[i] == "{":
                j = s.index("}", i); txt = s[i+1:j]; i = j + 1
            else:
                txt = s[i] if i < n else ""; i += 1
            runs.append({"t": txt, ("sup" if is_sup else "sub"): True}); continue
        j = i
        while j < n and s[j] not in "*_^\u27e6":
            j += 1
        r = {"t": s[i:j]}
        if bold: r["b"] = True
        if ital: r["i"] = True
        runs.append(r); i = j
    return runs

# parse the body into blocks
blocks = []
lines = body.split("\n")
cur_sub = None
def flush_fig():
    if cur_sub in FIGS:
        from PIL import Image
        f = FIGS[cur_sub]
        W, H = Image.open(f["path"]).size
        dw = 600; dh = round(dw * H / W)
        blocks.append({"type": "figure", "path": f["path"], "w": dw, "h": dh})
        blocks.append({"type": "caption", "runs": tok(f["caption"])})

i = 0
while i < len(lines):
    ln = lines[i].rstrip()
    if ln.startswith("## "):
        flush_fig(); cur_sub = None
        blocks.append({"type": "h1", "runs": tok(ln[3:].strip())})
    elif ln.startswith("### "):
        flush_fig()
        cur_sub = ln[4:].strip()
        blocks.append({"type": "h2", "runs": tok(cur_sub)})
    elif ln.startswith("|"):
        tbl = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            tbl.append(lines[i]); i += 1
        rows = []
        for r in tbl:
            if re.match(r"^\s*\|[\s:|-]+\|\s*$", r):  # separator
                continue
            cells = [c.strip().replace("\x00", "|") for c in r.replace("\\|", "\x00").strip().strip("|").split("|")]
            rows.append(cells)
        blocks.append({"type": "table", "header": [tok(c) for c in rows[0]],
                       "rows": [[tok(c) for c in row] for row in rows[1:]]})
        continue
    elif ln.startswith("  ") and ln.strip():
        blocks.append({"type": "equation", "runs": tok(ln.strip())})
    elif ln.strip() and not re.match(r"^-{3,}$", ln.strip()):
        blocks.append({"type": "p", "runs": tok(ln.strip())})
    i += 1
flush_fig()

# references, in citation order
REFS = [
 "Vamathevan, J.; Clark, D.; Czodrowski, P.; Dunham, I.; Ferran, E.; Lee, G.; et al. Applications of Machine Learning in Drug Discovery and Development. *Nat. Rev. Drug Discov.* 2019, 18 (6), 463\u2013477.",
 "Walters, W. P.; Barzilay, R. Critical Assessment of AI in Drug Discovery. *Expert Opin. Drug Discov.* 2021, 16 (9), 937\u2013947.",
 "Johnson, M. A.; Maggiora, G. M. *Concepts and Applications of Molecular Similarity*; Wiley: New York, 1990.",
 "Maggiora, G. M. On Outliers and Activity Cliffs\u2014Why QSAR Often Disappoints. *J. Chem. Inf. Model.* 2006, 46 (4), 1535.",
 "Stumpfe, D.; Bajorath, J. Exploring Activity Cliffs in Medicinal Chemistry. *J. Med. Chem.* 2012, 55 (7), 2932\u20132942.",
 "Cruz-Monteagudo, M.; Medina-Franco, J. L.; P\u00e9rez-Castillo, Y.; Nicolotti, O.; Cordeiro, M. N. D. S.; Borges, F. Activity Cliffs in Drug Discovery: Dr Jekyll or Mr Hyde? *Drug Discov. Today* 2014, 19 (8), 1069\u20131080.",
 "van Tilborg, D.; Alenicheva, A.; Grisoni, F. Exposing the Limitations of Molecular Machine Learning with Activity Cliffs. *J. Chem. Inf. Model.* 2022, 62 (23), 5938\u20135951.",
 "Dablander, M.; Hanser, T.; Lambiotte, R.; Morris, G. M. Exploring QSAR Models for Activity-Cliff Prediction. *J. Cheminform.* 2023, 15, 47.",
 "Aldeghi, M.; Graff, D. E.; Frey, N.; Morrone, J. A.; Pyzer-Knapp, E. O.; Jordan, K. E.; Coley, C. W. Roughness of Molecular Property Landscapes and Its Impact on Modellability. *J. Chem. Inf. Model.* 2022, 62 (19), 4660\u20134671.",
 "Graff, D. E.; Pyzer-Knapp, E. O.; Jordan, K. E.; Shakhnovich, E. I.; Coley, C. W. Evaluating the Roughness of Structure\u2013Property Relationships Using Pretrained Molecular Representations. *Digital Discovery* 2023, 2 (5), 1452\u20131460.",
 "Golbraikh, A.; Muratov, E.; Fourches, D.; Tropsha, A. Data Set Modelability by QSAR. *J. Chem. Inf. Model.* 2014, 54 (1), 1\u20134.",
 "Peltason, L.; Bajorath, J. SAR Index: Quantifying the Nature of Structure\u2013Activity Relationships. *J. Med. Chem.* 2007, 50 (23), 5571\u20135578.",
 "Guha, R.; Van Drie, J. H. Structure\u2013Activity Landscape Index: Identifying and Quantifying Activity Cliffs. *J. Chem. Inf. Model.* 2008, 48 (3), 646\u2013658.",
 "Netzeva, T. I.; Worth, A. P.; Aldenberg, T.; Benigni, R.; Cronin, M. T. D.; Gramatica, P.; et al. Current Status of Methods for Defining the Applicability Domain of (Quantitative) Structure\u2013Activity Relationships. *ATLA, Altern. Lab. Anim.* 2005, 33 (2), 155\u2013173.",
 "Sahigara, F.; Mansouri, K.; Ballabio, D.; Mauri, A.; Consonni, V.; Todeschini, R. Comparison of Different Approaches to Define the Applicability Domain of QSAR Models. *Molecules* 2012, 17 (5), 4791\u20134810.",
 "Norinder, U.; Carlsson, L.; Boyer, S.; Eklund, M. Introducing Conformal Prediction in Predictive Modeling. A Transparent and Flexible Alternative to Applicability Domain Determination. *J. Chem. Inf. Model.* 2014, 54 (6), 1596\u20131603.",
 "Hirschfeld, L.; Swanson, K.; Yang, K.; Barzilay, R.; Coley, C. W. Uncertainty Quantification Using Neural Networks for Molecular Property Prediction. *J. Chem. Inf. Model.* 2020, 60 (8), 3770\u20133780.",
 "Gaulton, A.; Bellis, L. J.; Bento, A. P.; Chambers, J.; Davies, M.; Hersey, A.; et al. ChEMBL: A Large-Scale Bioactivity Database for Drug Discovery. *Nucleic Acids Res.* 2012, 40 (D1), D1100\u2013D1107.",
 "Delaney, J. S. ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. *J. Chem. Inf. Comput. Sci.* 2004, 44 (3), 1000\u20131005.",
 "Wu, Z.; Ramsundar, B.; Feinberg, E. N.; Gomes, J.; Geniesse, C.; Pappu, A. S.; Leswing, K.; Pande, V. MoleculeNet: A Benchmark for Molecular Machine Learning. *Chem. Sci.* 2018, 9 (2), 513\u2013530.",
 "Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular Frameworks. *J. Med. Chem.* 1996, 39 (15), 2887\u20132893.",
 "Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* 2010, 50 (5), 742\u2013754.",
 "Breiman, L. Random Forests. *Mach. Learn.* 2001, 45 (1), 5\u201332.",
 "Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. How Powerful Are Graph Neural Networks? In *International Conference on Learning Representations (ICLR)*, 2019.",
 "Benjamini, Y.; Hochberg, Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *J. R. Stat. Soc. B* 1995, 57 (1), 289\u2013300.",
]

doc = {
 "title": "Local Structure\u2013Activity Landscape Roughness Predicts Per-Compound QSAR Error Independently of the Applicability Domain",
 "authors": "Krishna Harish",
 "affiliation": "Elkins High School, Missouri City, Texas, United States",
 "corresponding": "*Corresponding author. E-mail: krishnaharish2009@gmail.com",
 "abstract": "Machine-learning models for molecular potency are most useful when their predictions can be trusted, yet they fail systematically on activity cliffs (pairs of structurally similar molecules with large potency differences) and give no warning when a prediction lands on one. Existing measures of structure\u2013activity landscape roughness are either global (one scalar per data set) or pairwise (single molecule pairs), while applicability-domain and uncertainty methods flag extrapolation into sparse chemical space rather than the dense-region discontinuities cliffs represent. We introduce a family of local roughness functionals computed from a compound\u2019s nearest training neighbors and evaluate them on thirty bioactivity benchmarks. Local roughness predicts the per-compound error of a random-forest model on every target and retains its predictive value after the applicability domain is controlled in partial-correlation and mixed-effects analyses, establishing it as a distinct axis of reliability. An activity-free descriptor, the pairwise-SALI density of a compound\u2019s neighbors, flags labeled activity cliffs with a median ROC-AUC of 0.69, whereas nearest-neighbor similarity is oriented oppositely because cliffs cluster among close analogs. These relationships hold across three model classes, are robust to neighborhood size and distance metric, and replicate on physicochemical regression benchmarks outside bioactivity; in a triage setting, ranking predictions by roughness preferentially recovers the activity cliffs that uncertainty- and applicability-domain flags miss. The result is a model-agnostic, structure-only estimate of where a QSAR prediction should and should not be trusted.",
 "keywords": "activity cliffs; QSAR; structure\u2013activity landscape; applicability domain; uncertainty quantification; cheminformatics; machine-learning reliability",
 "blocks": blocks,
 "references": [tok(r) for r in REFS],
 "toc_graphic": os.path.join(FIGURES_DIR, "TOC_graphic.png"),
}
json.dump(doc, open(os.path.join(ROOT, "manuscript.json"), "w"), ensure_ascii=False)
print("manuscript.json written:", len(blocks), "blocks,", len(REFS), "refs")
print("block types:", {b["type"]: sum(1 for x in blocks if x["type"] == b["type"]) for b in blocks})
