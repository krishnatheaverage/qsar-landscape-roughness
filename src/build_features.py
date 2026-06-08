"""
build_features.py  --  main feature pipeline (everything except the GNN).

for each MoleculeACE target: train an RF on the train split, then for every test
molecule compute a bunch of per-compound numbers from its nearest training neighbours.

landscape roughness (uses the query's own activity y_i):
    dirichlet   : sim-weighted mean squared activity gap
    lipschitz   : max |dy| / (1 - tanimoto)   (aggregated SALI)

a-priori roughness (no query y needed, so usable at predict time):
    nbr_disp    : std of the k nearest neighbours' activities
    sali_max    : max pairwise SALI among the query's k neighbours
    sali_mean   : mean of those
    highsim_disp: std of activities for neighbours with tanimoto >= 0.9 (MMP-ish)
    holder      : local Holder exponent of the training landscape
                  (slope of log|dy| vs log d over neighbour pairs; low = rough)

AD controls (the stuff we need to beat):
    nn_sim      : max tanimoto to any training mol (1 = sitting on the training data)
    local_dens  : mean of the k nearest sims
    mol_size    : heavy-atom count

targets:
    rf_err      : |pred - y|
    rf_var      : variance across the trees (uncertainty baseline)

cached to cache/<dataset>.csv (test rows only).
usage:  python3 build_features.py DATASET1 DATASET2 ...   |   python3 build_features.py all
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, sys, warnings, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
np.random.seed(0)

DATADIR = benchmark_dir()
CACHE = CACHE_DIR; os.makedirs(CACHE, exist_ok=True)
K = 10            # neighbours for dispersion / SALI / dirichlet / lipschitz
K_HOLDER = 25     # bigger neighbourhood for the holder slope fit
HIGHSIM = 0.9     # MMP-ish similarity cutoff
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurize(smiles):
    fps, arrs, sizes, ok = [], [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False); continue
        fp = _gen.GetFingerprint(m); a = np.zeros(2048, np.int8); DataStructs.ConvertToNumpyArray(fp, a)
        fps.append(fp); arrs.append(a); sizes.append(m.GetNumHeavyAtoms()); ok.append(True)
    return fps, np.array(arrs), np.array(sizes), np.array(ok)

def holder_exponent(neigh_fps, neigh_y):
    """slope of log|dy| vs log(dist) over all pairs in the neighbour set."""
    n = len(neigh_fps)
    if n < 4: return np.nan
    logd, logdy = [], []
    for a in range(n):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(neigh_fps[a], neigh_fps))
        for b in range(a + 1, n):
            d = 1.0 - sims[b]; dy = abs(neigh_y[a] - neigh_y[b])
            if d > 1e-3 and dy > 1e-6:
                logd.append(np.log(d)); logdy.append(np.log(dy))
    if len(logd) < 4: return np.nan
    return float(np.polyfit(logd, logdy, 1)[0])   # slope = the exponent

def process(name):
    out = os.path.join(CACHE, name + ".csv")
    if os.path.exists(out):
        print(f"  [skip] {name}"); return
    df = pd.read_csv(os.path.join(DATADIR, name + ".csv"))
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    tr_fps, tr_X, _, tr_ok = featurize(tr.smiles.tolist())
    te_fps, te_X, te_sz, te_ok = featurize(te.smiles.tolist())
    tr = tr[tr_ok].reset_index(drop=True); te = te[te_ok].reset_index(drop=True)
    tr_y = tr.y.values; te_y = te.y.values

    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=0).fit(tr_X, tr_y)
    tree_preds = np.stack([t.predict(te_X) for t in rf.estimators_])   # [n_trees, n_te]
    pred = tree_preds.mean(0); rf_var = tree_preds.var(0)
    rf_err = np.abs(pred - te_y)

    rec = []
    for i, (fp_i, y_i, sz_i) in enumerate(zip(te_fps, te_y, te_sz)):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_i, tr_fps))
        order = np.argsort(-sims)
        nn = order[:K]; s_nn, y_nn = sims[nn], tr_y[nn]
        nh = order[:K_HOLDER]
        dy = np.abs(y_i - y_nn)
        w = s_nn + 1e-9
        dirichlet = float(np.sum(w * dy**2) / np.sum(w))
        lipschitz = float(np.max(dy / np.clip(1 - s_nn, 1e-3, None)))
        nbr_disp = float(np.std(y_nn))
        # pairwise SALI among the k neighbours (doesn't touch the query's y)
        sali_vals = []
        for a in range(len(nn)):
            sims_a = np.array(DataStructs.BulkTanimotoSimilarity(tr_fps[nn[a]], [tr_fps[j] for j in nn]))
            for b in range(a + 1, len(nn)):
                d = max(1 - sims_a[b], 1e-3); sali_vals.append(abs(tr_y[nn[a]] - tr_y[nn[b]]) / d)
        sali_max = float(np.max(sali_vals)) if sali_vals else np.nan
        sali_mean = float(np.mean(sali_vals)) if sali_vals else np.nan
        hs = sims >= HIGHSIM
        highsim_disp = float(np.std(tr_y[hs])) if hs.sum() >= 2 else np.nan
        holder = holder_exponent([tr_fps[j] for j in nh], tr_y[nh])
        rec.append(dict(
            dataset=name, smiles=te.smiles.iloc[i], y=y_i, cliff_mol=int(te.cliff_mol.iloc[i]),
            rf_err=rf_err[i], rf_var=float(rf_var[i]),
            dirichlet=dirichlet, lipschitz=lipschitz,
            nbr_disp=nbr_disp, sali_max=sali_max, sali_mean=sali_mean,
            highsim_disp=highsim_disp, holder=holder,
            nn_sim=float(sims[nn[0]]), local_dens=float(np.mean(s_nn)), mol_size=int(sz_i)))
    pd.DataFrame(rec).to_csv(out, index=False)
    print(f"  [done] {name}: n_test={len(rec)}  (RF cliff/non err: "
          f"{np.mean([r['rf_err'] for r in rec if r['cliff_mol']]):.3f}/"
          f"{np.mean([r['rf_err'] for r in rec if not r['cliff_mol']]):.3f})")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["all"]:
        args = sorted(f[:-4] for f in os.listdir(DATADIR) if f.endswith(".csv"))
    for nm in args:
        process(nm)
    print("BATCH COMPLETE")
