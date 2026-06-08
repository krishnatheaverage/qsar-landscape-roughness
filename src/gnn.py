"""
gnn.py -- small GIN baseline, plain PyTorch on CPU.
trains on each target's train split and saves the per-compound test error, in the
same row order as cache/<dataset>.csv.

not trying to be a SOTA GNN -- just a message-passing model with a smoothness prior
(similar graphs -> similar preds), so we can check whether its cliff penalty piles
up in the rough regions.

out: cache_gnn/<dataset>.csv  [smiles, gnn_err]
usage: python3 gnn.py DATASET ...  |  python3 gnn.py all
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ROOT, PAPER_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS, RESULTS_DIR, FIGURES_DIR, benchmark_dir
import os, sys, warnings, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
import torch, torch.nn as nn
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
torch.set_num_threads(4)
SEED = 0; torch.manual_seed(SEED); np.random.seed(SEED)

DATADIR = benchmark_dir()
CACHE = CACHE_DIR; CACHE_GNN = CACHE_GNN; os.makedirs(CACHE_GNN, exist_ok=True)
ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]; HIDDEN = 64; LAYERS = 3; EPOCHS = 60; BS = 64; LR = 1e-3

def atom_feats(m):
    F = []
    for a in m.GetAtoms():
        z = a.GetAtomicNum()
        oh = [1.0 if z == t else 0.0 for t in ATOMS] + [1.0 if z not in ATOMS else 0.0]
        F.append(oh + [a.GetDegree() / 4, a.GetFormalCharge(), float(a.GetIsAromatic()),
                       a.GetTotalNumHs() / 4, float(a.IsInRing())])
    return np.array(F, dtype=np.float32)

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
        dims = [fdim] + [HIDDEN] * LAYERS
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], HIDDEN), nn.ReLU(),
                                                 nn.Linear(HIDDEN, HIDDEN)) for i in range(LAYERS)])
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
    def forward(self, X, A, M):
        H = X; readout = 0
        for mlp in self.mlps:
            neigh = torch.bmm(A, H)             # sum over neighbours
            H = mlp(H + neigh) * M.unsqueeze(-1)  # GIN-0, mask the padding
            cnt = M.sum(1, keepdim=True).clamp(min=1)
            readout = readout + H.sum(1) / cnt    # masked mean-pool, summed over layers
        return self.head(readout).squeeze(-1)

def run(name):
    out = os.path.join(CACHE_GNN, name + ".csv")
    if os.path.exists(out): print(f"  [skip] {name}"); return
    cache = pd.read_csv(os.path.join(CACHE, name + ".csv"))          # test rows + y, in order
    df = pd.read_csv(os.path.join(DATADIR, name + ".csv"))
    tr = df[df.split == "train"]
    tr_g = [mol_graph(s) for s in tr.smiles]; tr_y = tr.y.values
    keep = [i for i, g in enumerate(tr_g) if g is not None]
    tr_g = [tr_g[i] for i in keep]; tr_y = tr_y[keep]
    te_g = [mol_graph(s) for s in cache.smiles]; te_y = cache.y.values
    mu, sd = tr_y.mean(), tr_y.std() + 1e-9
    fdim = tr_g[0][0].shape[1]
    model = GIN(fdim); opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.MSELoss()
    idx = np.arange(len(tr_g))
    model.train()
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for s in range(0, len(idx), BS):
            bi = idx[s:s + BS]
            X, A, M = collate([tr_g[i] for i in bi])
            yb = torch.tensor((tr_y[bi] - mu) / sd, dtype=torch.float32)
            opt.zero_grad(); loss = lossf(model(X, A, M), yb); loss.backward(); opt.step()
    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(te_g), 128):
            X, A, M = collate(te_g[s:s + 128]); preds.append(model(X, A, M).numpy())
    pred = np.concatenate(preds) * sd + mu
    gnn_err = np.abs(pred - te_y)
    pd.DataFrame({"smiles": cache.smiles.values, "gnn_err": gnn_err}).to_csv(out, index=False)
    cliff = cache.cliff_mol.values.astype(bool)
    print(f"  [done] {name}: GNN test MAE={gnn_err.mean():.3f} (cliff/non "
          f"{gnn_err[cliff].mean():.3f}/{gnn_err[~cliff].mean():.3f}) | RF MAE={cache.rf_err.mean():.3f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["all"]:
        args = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith(".csv"))
    for nm in args:
        run(nm)
    print("BATCH COMPLETE")
