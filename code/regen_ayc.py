"""Regenerate the Asian Games Village (AYC) CNC partition at K=20 and a K-means
reference, using the exact CNC pipeline from the manuscript. We use the
JSON-reported selected K=20 (avoiding the expensive stability/continuity loop)
and only run the final spectral partition on the full graph for the map.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "code" else HERE
sys.path.insert(0, os.path.join(PROJ, "code"))
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

K = 20
print(f"[regen] spectral partition K={K} ...")
cnc_labels = cnc.spectral_partition(R, K, seed=0)

from sklearn.cluster import KMeans
print("[regen] KMeans reference K=20 ...")
km = KMeans(K, n_init=10, random_state=0).fit_predict(X)

np.save(os.path.join(OUT, "ayc_cnc_labels_K20.npy"), cnc_labels.astype(np.int16))
np.save(os.path.join(OUT, "ayc_kmeans_labels_K20.npy"), km.astype(np.int16))
print("[regen] saved labels. unique cnc:", np.unique(cnc_labels).size,
      "unique km:", np.unique(km).size)

# spatial coherence using original 200x200 layout
side = int(round(X.shape[0] ** 0.5))
print("[regen] side", side)
