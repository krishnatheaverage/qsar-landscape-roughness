# Head-to-head: local roughness vs prior per-compound reliability methods (Sheridan, RDN-adapt, variance, AD).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, benchmark_dir
import numpy as np, pandas as pd, warnings
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore"); np.random.seed(0)

DATADIR = benchmark_dir()
K = 10
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def feats(smiles):
    fps, arrs, ok = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: ok.append(False); continue
        fp = _gen.GetFingerprint(m); a = np.zeros(2048, np.int8); DataStructs.ConvertToNumpyArray(fp, a)
        fps.append(fp); arrs.append(a); ok.append(True)
    return fps, np.array(arrs), np.array(ok)

def tertile_bins(v):
    q = np.quantile(v, [1/3, 2/3])
    return np.digitize(v, q)

def per_target(name):
    df = pd.read_csv(os.path.join(DATADIR, name + ".csv"))
    tr = df[df.split == "train"].reset_index(drop=True); te = df[df.split == "test"].reset_index(drop=True)
    tr_fps, tr_X, tr_ok = feats(tr.smiles.tolist()); te_fps, te_X, te_ok = feats(te.smiles.tolist())
    tr = tr[tr_ok].reset_index(drop=True); te = te[te_ok].reset_index(drop=True)
    tr_y, te_y = tr.y.values, te.y.values
    cliff = te.cliff_mol.values.astype(int)
    rng = max(tr_y.max() - tr_y.min(), 1e-6)

    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt", n_jobs=-1,
                               random_state=0, oob_score=True, bootstrap=True).fit(tr_X, tr_y)
    te_tree = np.stack([t.predict(te_X) for t in rf.estimators_])
    te_pred = te_tree.mean(0); te_sd = te_tree.std(0); rf_err = np.abs(te_pred - te_y)
    tr_tree = np.stack([t.predict(tr_X) for t in rf.estimators_]); tr_sd = tr_tree.std(0)
    oob_pred = rf.oob_prediction_; oob_err = np.abs(oob_pred - tr_y)

    # neighbours of each test compound among training
    te_nn_sim = np.zeros(len(te)); roughness = np.zeros(len(te)); rdn_risk = np.zeros(len(te))
    # --- RDN (regression adaptation, transparently labelled) ---
    W = np.clip(1 - tr_sd/rng, 0, 1) * np.clip(1 - oob_err/rng, 0, 1)         # Wi = (1-normSTD)*agreement
    # per-train avg distance to its K nearest *other* training neighbours
    tr_avgd = np.zeros(len(tr))
    for i in range(len(tr)):
        ds = np.sort(1 - np.array(DataStructs.BulkTanimotoSimilarity(tr_fps[i], tr_fps)))[1:]  # drop self
        tr_avgd[i] = np.mean(ds[:K])
    q1, q3 = np.quantile(tr_avgd, [.25, .75]); refval = q3 + 1.5*(q3 - q1)
    Di = np.zeros(len(tr))
    for i in range(len(tr)):
        ds = np.sort(1 - np.array(DataStructs.BulkTanimotoSimilarity(tr_fps[i], tr_fps)))[1:]
        w = ds[ds <= refval][:K]
        Di[i] = np.mean(w) if len(w) else refval
    radius = np.clip(W * np.maximum(Di, 1e-3), 1e-3, None)

    for j, fp in enumerate(te_fps):
        s = np.array(DataStructs.BulkTanimotoSimilarity(fp, tr_fps)); order = np.argsort(-s); nn = order[:K]
        te_nn_sim[j] = s[nn[0]]
        sv = []
        for a in range(len(nn)):
            sa = np.array(DataStructs.BulkTanimotoSimilarity(tr_fps[nn[a]], [tr_fps[t] for t in nn]))
            for b in range(a+1, len(nn)):
                sv.append(abs(tr_y[nn[a]] - tr_y[nn[b]]) / max(1 - sa[b], 1e-3))
        roughness[j] = np.mean(sv) if sv else np.nan
        d_te = 1 - s
        rdn_risk[j] = np.min(d_te / radius)        # <=1 inside a reliable sphere; higher = less reliable

    # --- Sheridan 3-measure binned error model (calibrated on OOB training) ---
    tr_nn1 = np.array([np.sort(np.array(DataStructs.BulkTanimotoSimilarity(tr_fps[i], tr_fps)))[-2]
                       for i in range(len(tr))])     # similarity to nearest *other* training cpd
    b_sim, b_sd, b_pr = tertile_bins(tr_nn1), tertile_bins(tr_sd), tertile_bins(oob_pred)
    bin_rmse = {}
    for key in set(zip(b_sim, b_sd, b_pr)):
        m = (b_sim == key[0]) & (b_sd == key[1]) & (b_pr == key[2])
        bin_rmse[key] = np.sqrt(np.mean(oob_err[m]**2))
    glob = np.sqrt(np.mean(oob_err**2))
    qs_sim = np.quantile(tr_nn1, [1/3, 2/3]); qs_sd = np.quantile(tr_sd, [1/3, 2/3]); qs_pr = np.quantile(oob_pred, [1/3, 2/3])
    sher = np.array([bin_rmse.get((np.digitize(te_nn_sim[j], qs_sim), np.digitize(te_sd[j], qs_sd),
                                   np.digitize(te_pred[j], qs_pr)), glob) for j in range(len(te))])

    scores = {"roughness": roughness, "ensemble variance": te_sd, "distance-AD": 1 - te_nn_sim,
              "Sheridan (3-measure)": sher, "RDN (regr. adapt.)": rdn_risk}
    hi = (rf_err >= np.quantile(rf_err, 0.75)).astype(int)
    row = {}
    for nm, sc in scores.items():
        ok = np.isfinite(sc)
        row[nm + "|err"] = roc_auc_score(hi[ok], sc[ok]) if hi[ok].min() != hi[ok].max() else np.nan
        row[nm + "|cliff"] = roc_auc_score(cliff[ok], sc[ok]) if (cliff[ok].min() != cliff[ok].max()) else np.nan
    return row

names = sorted(f[:-4] for f in os.listdir(DATADIR) if f.endswith(".csv"))
rows = []
for n in names:
    try: rows.append(per_target(n)); print(f"  {n} done")
    except Exception as e: print(f"  {n} SKIP {e}")
R = pd.DataFrame(rows)
methods = ["roughness", "ensemble variance", "distance-AD", "Sheridan (3-measure)", "RDN (regr. adapt.)"]
out = pd.DataFrame({"method": methods,
                    "top_quartile_error_AUC": [R[m+"|err"].median() for m in methods],
                    "activity_cliff_AUC":     [R[m+"|cliff"].median() for m in methods]})
out.to_csv(os.path.join(RESULTS_DIR, "headtohead_results.csv"), index=False)
print("\n=== Head-to-head (median per-target ROC-AUC across 30 targets) ===")
print(out.to_string(index=False))
