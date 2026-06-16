# External check on two MoleculeNet physchem benchmarks (ESOL, Lipophilicity) with scaffold splits.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import urllib.request
def _ensure(_fn, _url):
    _dest = os.path.join(DATA_DIR, _fn)
    if not os.path.exists(_dest):
        print('downloading', _fn, '...'); urllib.request.urlretrieve(_url, _dest)
    return _dest
_BASE = 'https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest'
_ensure('ESOL_delaney-processed.csv', _BASE + '/ESOL_delaney-processed.csv')
_ensure('Lipophilicity.csv', _BASE + '/Lipophilicity.csv')
import warnings, numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, rankdata
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
np.random.seed(0)

DATASETS = {
    "ESOL (logS solubility)":      (os.path.join(DATA_DIR, "ESOL_delaney-processed.csv"), "smiles", "measured log solubility in mols per litre"),
    "Lipophilicity (logD)":        (os.path.join(DATA_DIR, "Lipophilicity.csv"), "smiles", "exp"),
}
K = 10
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurize(smiles):
    fps, arrs, ok = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: ok.append(False); continue
        fp = gen.GetFingerprint(m); a = np.zeros(2048, np.int8); DataStructs.ConvertToNumpyArray(fp, a)
        fps.append(fp); arrs.append(a); ok.append(True)
    return fps, np.array(arrs), np.array(ok)

def scaffold_split(smiles, frac_train=0.8):
    groups = {}
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else f"_bad{i}"
        groups.setdefault(scaf, []).append(i)
    n_train = int(frac_train * len(smiles)); tr, te = [], []
    for g in sorted(groups.values(), key=len, reverse=True):
        (tr if len(tr) + len(g) <= n_train else te).extend(g)
    return np.array(tr), np.array(te)

def partial_spearman(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    x, y, Z = x[m], y[m], Z[m]
    xr, yr = rankdata(x), rankdata(y)
    Zr = np.column_stack([np.ones(len(xr))] + [rankdata(Z[:, j]) for j in range(Z.shape[1])])
    rx = xr - Zr @ np.linalg.lstsq(Zr, xr, rcond=None)[0]
    ry = yr - Zr @ np.linalg.lstsq(Zr, yr, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])

print(f"{'dataset':26s} {'n_tr':>5s} {'n_te':>5s} {'RF_MAE':>7s} | "
      f"{'rho_dir':>8s} {'rho_disp':>9s} {'rho_sali':>9s} | {'disp|AD':>8s} {'dir|AD':>8s}")
print("-" * 110)
rows = []
for name, (path, scol, ycol) in DATASETS.items():
    df = pd.read_csv(path)[[scol, ycol]].dropna().reset_index(drop=True)
    df.columns = ["smiles", "y"]
    tr_idx, te_idx = scaffold_split(df.smiles.tolist())
    tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    tr_fps, tr_X, tr_ok = featurize(tr.smiles.tolist()); te_fps, te_X, te_ok = featurize(te.smiles.tolist())
    tr, te = tr[tr_ok].reset_index(drop=True), te[te_ok].reset_index(drop=True)
    tr_y, te_y = tr.y.values, te.y.values
    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=0).fit(tr_X, tr_y)
    err = np.abs(rf.predict(te_X) - te_y)
    dirich, disp, sali, nnsim, dens = [], [], [], [], []
    for fp_i, y_i in zip(te_fps, te_y):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_i, tr_fps)); nn = np.argsort(-sims)[:K]
        s_nn, y_nn = sims[nn], tr_y[nn]; w = s_nn + 1e-9
        dirich.append(np.sum(w * (y_i - y_nn)**2) / np.sum(w)); disp.append(np.std(y_nn))
        sv = []
        for a in range(len(nn)):
            sa = np.array(DataStructs.BulkTanimotoSimilarity(tr_fps[nn[a]], [tr_fps[j] for j in nn]))
            for b in range(a + 1, len(nn)):
                sv.append(abs(tr_y[nn[a]] - tr_y[nn[b]]) / max(1 - sa[b], 1e-3))
        sali.append(np.mean(sv) if sv else np.nan)
        nnsim.append(sims[nn[0]]); dens.append(np.mean(s_nn))
    dirich, disp, sali = map(np.array, (dirich, disp, sali)); nnsim, dens = np.array(nnsim), np.array(dens)
    Z = np.column_stack([nnsim, dens])
    r_dir = spearmanr(dirich, err); r_disp = spearmanr(disp, err); r_sali = spearmanr(sali, err)
    p_disp = partial_spearman(disp, err, Z); p_dir = partial_spearman(dirich, err, Z)
    print(f"{name:26s} {len(tr):5d} {len(te):5d} {err.mean():7.3f} | "
          f"{r_dir.statistic:8.3f} {r_disp.statistic:9.3f} {r_sali.statistic:9.3f} | {p_disp:8.3f} {p_dir:8.3f}")
    rows.append(dict(dataset=name, n_train=len(tr), n_test=len(te), rf_mae=err.mean(),
                     rho_dirichlet=r_dir.statistic, p_dirichlet=r_dir.pvalue,
                     rho_nbr_disp=r_disp.statistic, p_nbr_disp=r_disp.pvalue,
                     rho_sali_mean=r_sali.statistic, partial_disp_AD=p_disp, partial_dir_AD=p_dir))
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "cross_domain_results.csv"), index=False)
print("\nsaved cross_domain_results.csv")
