"""
model2.py [all|DATASET...] -- second/third model classes for the model-agnosticism test.
Trains HistGradientBoosting and SVR(RBF) on ECFP4 per target; records per-compound test
error aligned to cache/<target>.csv. Output: cache_models/<target>.csv [smiles, gbt_err, svr_err].
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, sys, warnings, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
DATADIR = benchmark_dir()
CACHE = CACHE_DIR; OUT = CACHE_MODELS; os.makedirs(OUT, exist_ok=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def feat(smiles):
    X, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: ok.append(False); continue
        fp = gen.GetFingerprint(m); a = np.zeros(2048, np.int8); DataStructs.ConvertToNumpyArray(fp, a)
        X.append(a); ok.append(True)
    return np.array(X), np.array(ok)

def run(name):
    fout = os.path.join(OUT, name + ".csv")
    if os.path.exists(fout): print("  [skip]", name); return
    cache = pd.read_csv(os.path.join(CACHE, name + ".csv"))
    df = pd.read_csv(os.path.join(DATADIR, name + ".csv")); tr = df[df.split == "train"]
    trX, trok = feat(tr.smiles.tolist()); teX, teok = feat(cache.smiles.tolist())
    tr_y = tr.y.values[trok]; te_y = cache.y.values
    gbt = HistGradientBoostingRegressor(random_state=0).fit(trX, tr_y)
    svr = SVR(C=10.0, gamma="scale").fit(trX, tr_y)
    pd.DataFrame({"smiles": cache.smiles.values,
                  "gbt_err": np.abs(gbt.predict(teX) - te_y),
                  "svr_err": np.abs(svr.predict(teX) - te_y)}).to_csv(fout, index=False)
    print(f"  [done] {name}: GBT MAE={np.abs(gbt.predict(teX)-te_y).mean():.3f} "
          f"SVR MAE={np.abs(svr.predict(teX)-te_y).mean():.3f} RF MAE={cache.rf_err.mean():.3f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["all"]: args = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith(".csv"))
    for nm in args: run(nm)
    print("BATCH COMPLETE")
