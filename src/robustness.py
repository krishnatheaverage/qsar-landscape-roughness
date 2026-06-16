# Recompute roughness/AD metrics per test compound across k under a chosen distance metric for all targets.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, sys, warnings, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors, Crippen
from scipy.spatial.distance import cdist
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

DATADIR = benchmark_dir()
CACHE = CACHE_DIR
K_LIST = [5, 10, 15, 20, 30]
metric = sys.argv[1]
OUT = os.path.join(CACHE_DIR, f"robustness_{metric}.csv")

gen4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
gen6 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
DESC = [lambda m: Descriptors.MolWt(m), lambda m: Crippen.MolLogP(m),
        lambda m: rdMolDescriptors.CalcTPSA(m), lambda m: rdMolDescriptors.CalcNumHBD(m),
        lambda m: rdMolDescriptors.CalcNumHBA(m), lambda m: rdMolDescriptors.CalcNumRotatableBonds(m),
        lambda m: rdMolDescriptors.CalcNumAromaticRings(m), lambda m: rdMolDescriptors.CalcFractionCSP3(m),
        lambda m: m.GetNumHeavyAtoms(), lambda m: rdMolDescriptors.CalcNumRings(m),
        lambda m: rdMolDescriptors.CalcNumHeteroatoms(m), lambda m: Descriptors.MolMR(m)]

def fps(smiles, g):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(g.GetFingerprint(m) if m else None)
    return out

def descmat(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for f in DESC] if m else [np.nan]*len(DESC))
    return np.array(rows, float)

def dist_test_train(te_smiles, tr_smiles):
    if metric in ("ecfp4", "ecfp6"):
        g = gen4 if metric == "ecfp4" else gen6
        trf = fps(tr_smiles, g); tef = fps(te_smiles, g)
        D = np.ones((len(tef), len(trf)))
        for i, fp in enumerate(tef):
            if fp is None: continue
            D[i] = 1.0 - np.array(DataStructs.BulkTanimotoSimilarity(fp, trf))
        return D
    else:
        Xtr = descmat(tr_smiles); Xte = descmat(te_smiles)
        mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0) + 1e-9
        Xtr = np.where(np.isfinite(Xtr), Xtr, mu); Xte = np.where(np.isfinite(Xte), Xte, mu)
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        return cdist(Xte, Xtr, "euclidean")

rows = []
for f in sorted(x for x in os.listdir(CACHE) if x.endswith(".csv") and not x.startswith("robustness")):
    name = f[:-4]
    cache = pd.read_csv(os.path.join(CACHE, name + ".csv"))
    df = pd.read_csv(os.path.join(DATADIR, name + ".csv"))
    tr = df[df.split == "train"].reset_index(drop=True)
    tr_y = tr.y.values
    D = dist_test_train(cache.smiles.tolist(), tr.smiles.tolist())
    order = np.argsort(D, axis=1)
    for i in range(len(cache)):
        oi = order[i]; di = D[i, oi]; yi = cache.y.iloc[i]
        nn_dist = float(di[0])
        for k in K_LIST:
            nn = oi[:k]; dk = di[:k]; ynn = tr_y[nn]
            rows.append(dict(dataset=name, metric=metric, k=k,
                             local_var=float(np.mean((yi - ynn)**2)),
                             nbr_disp=float(np.std(ynn)),
                             nn_dist=nn_dist, local_dens=float(np.mean(dk)),
                             rf_err=float(cache.rf_err.iloc[i]), cliff_mol=int(cache.cliff_mol.iloc[i])))
    print(f"  {name} done", flush=True)

pd.DataFrame(rows).to_csv(OUT, index=False)
print("SAVED", OUT, "rows", len(rows))
