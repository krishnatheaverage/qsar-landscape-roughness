"""
gnn_tuned.py [all|DATASET...] -- retrain the GIN *properly* to remove the undertraining
confound behind panel C: 10% validation split, up to 250 epochs, best-checkpoint
selection on val MSE with early stopping (patience 25), wider hidden (128) + dropout.
Saves per-compound test error to cache_gnn2/<dataset>.csv aligned to cache/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, sys, copy, warnings, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
import torch, torch.nn as nn
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)

DATADIR = benchmark_dir()
CACHE = CACHE_DIR; OUT = CACHE_GNN2; os.makedirs(OUT, exist_ok=True)
ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]; HIDDEN = 64; LAYERS = 3
EPOCHS = 150; PATIENCE = 20; BS = 128; LR = 1e-3

def atom_feats(m):
    F = []
    for a in m.GetAtoms():
        z = a.GetAtomicNum()
        oh = [1.0 if z == t else 0.0 for t in ATOMS] + [1.0 if z not in ATOMS else 0.0]
        F.append(oh + [a.GetDegree()/4, a.GetFormalCharge(), float(a.GetIsAromatic()),
                       a.GetTotalNumHs()/4, float(a.IsInRing())])
    return np.array(F, np.float32)

def mol_graph(s):
    m = Chem.MolFromSmiles(s)
    if m is None or m.GetNumAtoms() == 0: return None
    n = m.GetNumAtoms(); A = np.zeros((n, n), np.float32)
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); A[i, j] = A[j, i] = 1.0
    return atom_feats(m), A

def collate(graphs):
    n = max(g[0].shape[0] for g in graphs); F = graphs[0][0].shape[1]; B = len(graphs)
    X = np.zeros((B, n, F), np.float32); A = np.zeros((B, n, n), np.float32); M = np.zeros((B, n), np.float32)
    for k, (xf, a) in enumerate(graphs):
        nk = xf.shape[0]; X[k, :nk] = xf; A[k, :nk, :nk] = a; M[k, :nk] = 1.0
    return torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(M)

class GIN(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        dims = [fdim] + [HIDDEN]*LAYERS
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], HIDDEN), nn.ReLU(),
                                                 nn.Linear(HIDDEN, HIDDEN)) for i in range(LAYERS)])
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(0.1), nn.Linear(HIDDEN, 1))
    def forward(self, X, A, M):
        H = X; readout = 0
        for mlp in self.mlps:
            H = mlp(H + torch.bmm(A, H)) * M.unsqueeze(-1)
            readout = readout + H.sum(1) / M.sum(1, keepdim=True).clamp(min=1)
        return self.head(readout).squeeze(-1)

def predict(model, graphs):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(graphs), 128):
            X, A, M = collate(graphs[s:s+128]); out.append(model(X, A, M).numpy())
    return np.concatenate(out)

def run(name):
    fout = os.path.join(OUT, name + ".csv")
    if os.path.exists(fout): print(f"  [skip] {name}"); return
    cache = pd.read_csv(os.path.join(CACHE, name + ".csv"))
    df = pd.read_csv(os.path.join(DATADIR, name + ".csv")); tr = df[df.split == "train"]
    g_all = [mol_graph(s) for s in tr.smiles]; y_all = tr.y.values
    keep = [i for i, g in enumerate(g_all) if g is not None]
    g_all = [g_all[i] for i in keep]; y_all = y_all[keep]
    te_g = [mol_graph(s) for s in cache.smiles]; te_y = cache.y.values
    # 10% validation split
    idx = np.random.permutation(len(g_all)); nval = max(8, len(g_all)//10)
    vi, ti = idx[:nval], idx[nval:]
    mu, sd = y_all[ti].mean(), y_all[ti].std() + 1e-9
    model = GIN(g_all[0][0].shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    lossf = nn.MSELoss()
    # size-bucketed batches (minimise padding); shuffle batch ORDER each epoch
    ti_sorted = ti[np.argsort([g_all[i][0].shape[0] for i in ti])]
    batches = [ti_sorted[s:s+BS] for s in range(0, len(ti_sorted), BS)]
    best_val, best_state, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(batches)
        for bi in batches:
            X, A, M = collate([g_all[i] for i in bi])
            yb = torch.tensor((y_all[bi]-mu)/sd, dtype=torch.float32)
            opt.zero_grad(); lossf(model(X, A, M), yb).backward(); opt.step()
        vp = predict(model, [g_all[i] for i in vi]) * sd + mu
        vmse = float(np.mean((vp - y_all[vi])**2)); sched.step(vmse)
        if vmse < best_val - 1e-4:
            best_val, best_state, wait = vmse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(best_state)
    pred = predict(model, te_g) * sd + mu
    gnn_err = np.abs(pred - te_y)
    pd.DataFrame({"smiles": cache.smiles.values, "gnn_err": gnn_err}).to_csv(fout, index=False)
    cliff = cache.cliff_mol.values.astype(bool)
    print(f"  [done] {name}: tuned MAE={gnn_err.mean():.3f} (was via fixed-60) "
          f"cliff/non {gnn_err[cliff].mean():.3f}/{gnn_err[~cliff].mean():.3f} | RF {cache.rf_err.mean():.3f} | ep~{ep}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["all"]:
        args = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith(".csv"))
    for nm in args:
        run(nm)
    print("BATCH COMPLETE")
