"""Regenerate the AYC CNC partition using the K selected by the experiment.

The stability scan is performed by ``experiments.run_hsi``.  This helper only
rebuilds the full graph and final labels for the K recorded in
``results/ayc_paper_metrics.json``; it never silently substitutes a hard-coded
cluster count.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)   # comparison/ parent = project root
# --- CNC package path bootstrap: make the three category folders importable ---
for _c in ("core", "comparison", "visualization"):
    _p = os.path.join(PROJ, _c)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# --- end bootstrap ---
import cnc
from datasets import load_ayc

OUT = os.path.join(PROJ, "figures", "paper_figures")
os.makedirs(OUT, exist_ok=True)

print("[regen] loading AYC (PCA->6) ...")
X, _ = load_ayc()
print("[regen] X shape", X.shape)

print("[regen] building resonance affinity ...")
R, k, n_edges = cnc.resonance_affinity(X, k=20, rank_exponent=2.0)
print("[regen] n_edges", n_edges)

metrics_path = os.path.join(PROJ, "results", "ayc_paper_metrics.json")
if not os.path.exists(metrics_path):
    raise FileNotFoundError("run the AYC experiment first: " + metrics_path)
with open(metrics_path, encoding="utf-8") as f:
    K = int(json.load(f)["K"])
print(f"[regen] spectral partition K={K} ...")
cnc_labels = cnc.spectral_partition(R, K, seed=0)

from sklearn.cluster import KMeans
print(f"[regen] KMeans reference K={K} ...")
km = KMeans(K, n_init=10, random_state=0).fit_predict(X)

cnc_path = os.path.join(OUT, f"ayc_cnc_labels_K{K}.npy")
km_path = os.path.join(OUT, f"ayc_kmeans_labels_K{K}.npy")
np.save(cnc_path, cnc_labels.astype(np.int16))
np.save(km_path, km.astype(np.int16))
# Preserve historical filenames when the selected value is K=20.
if K == 20:
    np.save(os.path.join(OUT, "ayc_cnc_labels_K20.npy"), cnc_labels.astype(np.int16))
    np.save(os.path.join(OUT, "ayc_kmeans_labels_K20.npy"), km.astype(np.int16))
print("[regen] saved labels. unique cnc:", np.unique(cnc_labels).size,
      "unique km:", np.unique(km).size)

# spatial coherence using original 200x200 layout
side = int(round(X.shape[0] ** 0.5))
print("[regen] side", side)
