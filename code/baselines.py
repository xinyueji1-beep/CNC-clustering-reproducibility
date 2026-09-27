# -*- coding: utf-8 -*-
"""
Baseline clustering algorithms for comparison with CNC.

Each function returns cluster labels (and K where relevant).  All functions
take numpy arrays and a fixed seed for reproducibility.

  * kmeans_labels       -- KMeans (oracle K)
  * dbscan_auto         -- DBSCAN with per-dataset auto eps (median k-distance)
  * dpc                 -- Density-Peaks Clustering (oracle K, in-house)
  * affinity_propagation_labels -- AP (auto K)
  * hdbscan_labels      -- HDBSCAN (auto K)
  * snn_partition       -- binary SNN affinity + CNC spectral partition (oracle K)
  * rbf_partition       -- Gaussian/RBF affinity + spectral partition (oracle K)
  * rbf_partition_knn   -- kNN-truncated Gaussian affinity (for large N, where
                           the dense O(N^2) kernel does not fit in memory)

Metrics: ari, ami.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as ami
from sklearn.neighbors import NearestNeighbors

from cnc import spectral_partition, resonance_affinity  # reuse CNC's partition


# ---------------------------------------------------------------------------
# KMeans (oracle K)
# ---------------------------------------------------------------------------
def kmeans_labels(X: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    return KMeans(K, n_init=10, random_state=seed).fit_predict(X)


# ---------------------------------------------------------------------------
# DBSCAN with per-dataset auto eps
# ---------------------------------------------------------------------------
def auto_eps_dbscan(X: np.ndarray, min_samples: int = 5) -> float:
    """Choose eps as the median MinPts-th-nearest-neighbor distance.

    ``NearestNeighbors`` includes the point itself in its first column, so we
    request ``min_samples + 1`` neighbors and read the last column, i.e. the
    genuine MinPts-th neighbor distance.  This adapts to each dataset's density
    scale rather than using a single global eps that is unfair across 2D, 20-D,
    and 64-D inputs.
    """
    kd = NearestNeighbors(n_neighbors=min_samples + 1).fit(X).kneighbors(X)[0][:, -1]
    return float(np.median(kd))


def dbscan_auto(X: np.ndarray, min_samples: int = 5):
    eps = auto_eps_dbscan(X, min_samples=min_samples)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels.tolist()) - {-1})
    return labels, n_clusters, eps


# ---------------------------------------------------------------------------
# Density-Peaks Clustering (Rodriguez & Laio 2014), in-house
# ---------------------------------------------------------------------------
def dpc(X: np.ndarray, K: int, dc_frac: float = 0.02) -> np.ndarray:
    """DPC: dc at the dc_frac-quantile of pairwise distances; Gaussian-kernel
    local density rho; delta = min distance to a higher-density point; top-K
    centers by gamma = rho*delta; assign each point to its nearest
    higher-density neighbor's cluster.  Oracle K supplied by the caller."""
    n = X.shape[0]
    D = cdist(X, X)
    dc = np.quantile(D[np.triu_indices(n, 1)], dc_frac)
    if dc <= 0:
        dc = np.quantile(D[np.triu_indices(n, 1)], 0.5)
    rho = np.sum(np.exp(-(D / dc) ** 2), axis=1) - 1.0  # exclude self

    delta = np.zeros(n)
    nearest_hd = -np.ones(n, dtype=int)
    # Deterministic tie-handling: order by decreasing rho, ties broken by
    # point index (lexsort: primary key -rho, secondary key index).
    order = np.lexsort((np.arange(n), -rho))
    for rank, i in enumerate(order):
        if rank == 0:
            delta[i] = D[i].max()
            continue
        higher = order[:rank]
        j = higher[np.argmin(D[i, higher])]
        delta[i] = D[i, j]
        nearest_hd[i] = j

    gamma = rho * delta
    centers = np.argsort(-gamma)[:K]
    labels = -np.ones(n, dtype=int)
    for c_idx, c in enumerate(centers):
        labels[c] = c_idx

    def _resolve(i: int) -> None:
        """Follow the nearest-higher-density chain to the first labeled point.

        Every chain step moves strictly earlier in ``order`` (higher local
        density), and the rank-0 point always has the largest gamma and is
        therefore labeled, so the chain terminates.
        """
        chain = []
        j = i
        while labels[j] == -1:
            chain.append(j)
            j = int(nearest_hd[j])
        for c in chain:
            labels[c] = labels[j]

    # assign in decreasing rho order
    for i in order:
        if labels[i] == -1:
            _resolve(i)
    return labels


# ---------------------------------------------------------------------------
# Affinity Propagation (auto K)
# ---------------------------------------------------------------------------
def affinity_propagation_labels(X: np.ndarray):
    ap = AffinityPropagation(random_state=0, convergence_iter=25, max_iter=300)
    labels = ap.fit_predict(X)
    return labels, int(len(np.unique(labels)))


# ---------------------------------------------------------------------------
# HDBSCAN (auto K)
# ---------------------------------------------------------------------------
def hdbscan_labels(X: np.ndarray, min_cluster_size: int = 15):
    import hdbscan
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    n_clusters = len(set(labels.tolist()) - {-1})
    return labels, n_clusters


# ---------------------------------------------------------------------------
# Binary SNN affinity + CNC spectral partition (oracle K)
# ---------------------------------------------------------------------------
def snn_partition(X: np.ndarray, K: int, k: int = 20, seed: int = 0) -> np.ndarray:
    """Shared-nearest-neighbor affinity with rank_exponent=0 (binary count),
    partitioned with CNC's spectral machinery at the oracle K."""
    R, _, _ = resonance_affinity(X, k=k, rank_exponent=0.0)
    return spectral_partition(R, K, seed=seed)


# ---------------------------------------------------------------------------
# RBF / Gaussian affinity + spectral partition (oracle K)
# ---------------------------------------------------------------------------
def rbf_partition(X: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Gaussian RBF affinity with sigma = median pairwise distance, partitioned
    with CNC's spectral machinery at the oracle K."""
    n = X.shape[0]
    D = cdist(X, X)
    sigma = float(np.median(D))
    if sigma <= 0:
        sigma = 1.0
    A = csr_matrix(np.exp(-(D ** 2) / (2.0 * sigma ** 2)))
    return spectral_partition(A, K, seed=seed)


def rbf_partition_knn(X: np.ndarray, K: int, k: int = 20, seed: int = 0,
                      sigma_samples: int = 2000) -> np.ndarray:
    """Gaussian RBF affinity truncated to the k nearest neighbours.

    Used where the dense kernel of :func:`rbf_partition` is memory-prohibitive:
    for N = 10,249 the dense form needs about 3\,GB and its shift-invert
    factorization fails.  The scale sigma is the median pairwise distance of at
    most ``sigma_samples`` points; every other entry of the full kernel is set
    to zero, keeping O(N*k) nonzeros.  Returned labels are produced by the same
    spectral partition as :func:`rbf_partition`.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n > sigma_samples:
        sub = X[rng.choice(n, sigma_samples, replace=False)]
    else:
        sub = X
    sigma = float(np.median(cdist(sub, sub)))
    if sigma <= 0:
        sigma = 1.0
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = np.exp(-(dist.ravel() ** 2) / (2.0 * sigma ** 2))
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)          # symmetrize the truncated kernel
    return spectral_partition(A, K, seed=seed)
