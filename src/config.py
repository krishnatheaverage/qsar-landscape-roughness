# -*- coding: utf-8 -*-
"""Central path configuration. All locations are relative to the repository root
so the pipeline runs from a fresh checkout. Override any of them with environment
variables (QSAR_CACHE, QSAR_RESULTS, QSAR_FIGURES, QSAR_DATA, MOLECULEACE_DATA)."""
import os

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR   = os.path.join(ROOT, "paper")
FIGURES_DIR = os.environ.get("QSAR_FIGURES", os.path.join(PAPER_DIR, "figures"))
RESULTS_DIR = os.environ.get("QSAR_RESULTS", os.path.join(ROOT, "results"))
DATA_DIR    = os.environ.get("QSAR_DATA",    os.path.join(ROOT, "data"))
CACHE_DIR   = os.environ.get("QSAR_CACHE",   os.path.join(ROOT, "cache"))
CACHE_GNN    = os.path.join(CACHE_DIR, "gnn")
CACHE_GNN2   = os.path.join(CACHE_DIR, "gnn2")
CACHE_MODELS = os.path.join(CACHE_DIR, "models")

for _d in (FIGURES_DIR, RESULTS_DIR, DATA_DIR, CACHE_DIR, CACHE_GNN, CACHE_GNN2, CACHE_MODELS):
    os.makedirs(_d, exist_ok=True)

def benchmark_dir():
    """Locate the MoleculeACE benchmark_data directory (30 ChEMBL targets)."""
    p = os.environ.get("MOLECULEACE_DATA")
    if p:
        return p
    try:
        import MoleculeACE
        return os.path.join(os.path.dirname(MoleculeACE.__file__), "Data", "benchmark_data")
    except Exception:
        raise SystemExit("MoleculeACE benchmark data not found. Run `pip install MoleculeACE` "
                         "or set the MOLECULEACE_DATA environment variable to the benchmark_data path.")
