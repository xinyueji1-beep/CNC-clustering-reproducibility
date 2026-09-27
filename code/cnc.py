# -*- coding: utf-8 -*-
"""
CNC (Consensus-Neighborhood Clustering) core algorithm.

This module contains only the new method:
  * resonance_affinity   -- rank-weighted shared-neighbor cosine graph
  * spectral_partition   -- normalized-Laplacian spectral partition
  * stability_select_K   -- subsampling-NMI consensus for automatic K (argmax)
  * hierarchical_stability_select -- adjacent-partition-agreement refinement (default)
  * cnc_cluster          -- end-to-end: affinity -> K -> labels
  * block_purity / block_recovery -- local-support diagnostics

CPU/GPU
-------
The module detects CuPy for sparse GPU eigensolves and uses cuML for GPU
neighbor search and KMeans when available. CPU implementations remain available.
Force a backend with::

    import cnc
    cnc.set_device("gpu")   # or "cpu"

The package runner defaults to the CPU reference path. Its optional GPU mode
uses CuPy for sparse eigensolves; cuML additionally accelerates neighbor search
and KMeans where that library is supported and installed.
"""
from __future__ import annotations

import numpy as np
import os
import sys
from pathlib import Path
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# GPU backend detection (optional)
# ---------------------------------------------------------------------------
_GPU = False
_CP = None        # cupy module, or None
_CUML_NEIGHBORS = None
_CUML_KMEANS = None
_CUPY_EIGSH = None
_DLL_DIR_HANDLES = []


def _prepare_cuda_dll_path() -> None:
    """Expose CUDA DLLs bundled in the active conda environment on Windows."""
    if os.name != "nt":
        return
    prefix = Path(sys.prefix)
    candidates = [prefix / "bin", prefix / "Library" / "bin"]
    if not any((prefix / "bin").glob("nvrtc-builtins64_*.dll")):
        return
    os.environ.setdefault("CUDA_PATH", str(prefix))
    path_entries = {entry.lower() for entry in os.environ.get("PATH", "").split(os.pathsep)}
    for directory in candidates:
        if directory.is_dir():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            if str(directory).lower() not in path_entries:
                os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
                path_entries.add(str(directory).lower())


def _detect_gpu() -> bool:
    """Enable CuPy's sparse eigensolver and optional cuML helpers."""
    global _GPU, _CP, _CUML_NEIGHBORS, _CUML_KMEANS, _CUPY_EIGSH
    try:
        _prepare_cuda_dll_path()
        import cupy as cp
        import cupyx.scipy.sparse  # noqa: F401
        import cupyx.scipy.sparse.linalg as cupy_linalg
        cp.cuda.get_device_id()  # raises if no GPU
        cp.arange(1, dtype=cp.float32).sum().get()
        _CP = cp
        _CUPY_EIGSH = cupy_linalg.eigsh
        try:
            from cuml.neighbors import NearestNeighbors as cuNN
            from cuml.cluster import KMeans as cuKMeans
        except Exception:
            cuNN = cuKMeans = None
        _CUML_NEIGHBORS = cuNN
        _CUML_KMEANS = cuKMeans
        _GPU = True
        return True
    except Exception:
        _GPU = False
        return False


def set_device(name: str) -> None:
    """Force ``"cpu"`` or ``"gpu"``; GPU mode requires a working CuPy install."""
    global _GPU
    if name == "gpu":
        if not _detect_gpu():
            print("[cnc] CuPy CUDA backend unavailable, falling back to CPU.")
            _GPU = False
        else:
            _GPU = True
    elif name == "cpu":
        _GPU = False
    else:
        raise ValueError(f"device must be 'cpu' or 'gpu'; got {name!r}")


# auto-detect on import
_detect_gpu()


def using_gpu() -> bool:
    return _GPU


def backend_summary() -> str:
    if not _GPU:
        return "CPU (SciPy eigensolver, scikit-learn neighbors and KMeans)"
    helpers = "cuML neighbors/KMeans" if _CUML_NEIGHBORS is not None else "CPU neighbors/KMeans"
    return f"GPU CuPy eigensolver; {helpers}"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def deterministic_v0(n: int, seed: int = 0):
    """A fixed nonzero starting vector for eigsh, for reproducibility.

    The ``1009*n`` offset makes the vector depend on the graph size as well
    as the seed, matching the original implementation.  Returns a CuPy
    array on GPU, NumPy array on CPU.
    """
    rng = np.random.default_rng(seed + 1009 * n)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    if _GPU:
        v = _CP.asarray(v)
    return v


# ---------------------------------------------------------------------------
# 1. Rank-weighted shared-neighbor cosine resonance affinity
# ---------------------------------------------------------------------------
def resonance_affinity(X: np.ndarray, k: int = 20, rank_exponent: float = 2.0):
    """Build the CNC resonance-affinity graph R.

    Steps
    -----
    1. Directed kNN adjacency N with rank weights w=(k-r)^p.
    2. Symmetric shared-neighbor adjacency A = N N^T.
    3. Row-normalize to cosine: R[i,j] = A[i,j] / (||a_i|| ||a_j||).

    Returns
    -------
    R : csr_matrix  (n, n) float64, symmetric, diagonal set to 0
    k : int
    n_edges : int
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if k < 1 or k >= n:
        raise ValueError(f"k must satisfy 1 <= k < n; got k={k}, n={n}")

    # --- directed kNN with rank weights ---
    # NearestNeighbors(k) includes the point itself at distance 0, so we ask
    # for k+1 and drop the self-column to get exactly k genuine neighbors.
    if _GPU and _CUML_NEIGHBORS is not None:
        nn = _CUML_NEIGHBORS(n_neighbors=k + 1).fit(X)
        _, idx_gpu = nn.kneighbors(X)
        idx = _CP.asnumpy(idx_gpu)[:, 1:]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    if rank_exponent == 0:
        vals = np.ones(n * k, dtype=np.float64)
    else:
        ranks = np.arange(k, dtype=np.float64)
        vals = np.tile((k - ranks) ** rank_exponent, n)
    N = csr_matrix((vals, (rows, cols)), shape=(n, n))

    M = (N @ N.T).tocsr()
    norms = np.sqrt(M.diagonal())
    norms[norms <= 0] = 1.0
    inv_norm = diags(1.0 / norms)
    R = (inv_norm @ M @ inv_norm).tocsr()
    R.setdiag(0.0)
    R.eliminate_zeros()
    R.data = np.clip(R.data, 0.0, 1.0)
    return R, k, int(M.nnz)


# ---------------------------------------------------------------------------
# 2. Spectral partition on a sparse affinity
# ---------------------------------------------------------------------------
def spectral_partition(A, K: int, seed: int = 0) -> np.ndarray:
    """Normalized-Laplacian spectral partition of a (sparse) graph.

    Uses the sparse eigensolver (shift-invert for the smallest eigenvalues)
    followed by row-normalized embedding and KMeans. CuPy accelerates the GPU
    eigensolve; cuML optionally accelerates neighbor search and KMeans.
    """
    emb = _spectral_embedding(A, K, seed=seed)
    if _GPU and _CUML_KMEANS is not None:
        km = _CUML_KMEANS(n_clusters=K, n_init=10, random_state=seed)
        return _CP.asnumpy(km.fit_predict(emb))
    return KMeans(K, n_init=10, random_state=seed).fit_predict(emb)


def _spectral_embedding(A, K: int, seed: int = 0) -> np.ndarray:
    """Return the row-normalized K-dimensional Laplacian embedding."""
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    n = A.shape[0]
    if K < 2 or K >= n:
        raise ValueError(f"K must satisfy 2 <= K < n; got K={K}, n={n}")

    d = np.asarray(A.sum(axis=1)).ravel()
    d[d <= 0] = 1e-12
    dinv = diags(1.0 / np.sqrt(d))
    L = (diags(np.ones(n)) - dinv @ A @ dinv).tocsr()
    v0 = deterministic_v0(n, seed=seed)

    if _GPU:
        L_gpu = _CP.sparse.csr_matrix(L)
        ev, evec = _CUPY_EIGSH(L_gpu, k=K, which="SA",
                               tol=1e-4, maxiter=4000, v0=v0)
        ev = _CP.asnumpy(ev)
        evec = _CP.asnumpy(evec)
    else:
        try:
            ev, evec = eigsh(L, k=K, sigma=-1e-6, which="LM", tol=1e-4,
                             maxiter=2000, v0=v0)
        except Exception:
            ev, evec = eigsh(L, k=K, which="SM", tol=1e-3, maxiter=4000, v0=v0)

    order = np.argsort(ev)
    emb = evec[:, order[:K]]
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
# ---------------------------------------------------------------------------
# 3. Stability selection for K (unsupervised)
# ---------------------------------------------------------------------------
def stability_curve_stats(A, Kmax: int = 10, B: int = 40, frac: float = 0.8,
                          sample_cap: int | None = None, seed: int = 0):
    """Full stability curve with per-candidate uncertainty.

    Same resampling protocol as :func:`stability_select_K`, but instead of only
    the mean it returns, for every candidate K, the mean subsample-partition NMI,
    the standard error of that mean over the valid subsample pairs, and the
    number of contributing pairs.  The standard error makes a flat plateau
    distinguishable from a sharp maximum.
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    n = A.shape[0]
    if B < 2:
        raise ValueError(f"B must be at least 2; got B={B}")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1]; got frac={frac}")
    sample_size = int(frac * n)
    if sample_cap is not None:
        if sample_cap < 1:
            raise ValueError(f"sample_cap must be positive; got {sample_cap}")
        sample_size = min(sample_size, int(sample_cap))
    if sample_size <= Kmax:
        raise ValueError(
            f"subsample size must exceed Kmax; got size={sample_size}, Kmax={Kmax}"
        )

    rng = np.random.default_rng(seed)
    idxs = [rng.choice(n, size=sample_size, replace=False) for _ in range(B)]
    pi_list = [{p: q for q, p in enumerate(perm)} for perm in idxs]

    embeddings = []
    for perm in idxs:
        Ab = A[perm][:, perm]
        embeddings.append(_spectral_embedding(Ab, Kmax, seed=seed))

    stats = {}
    for K in range(2, Kmax + 1):
        labs = []
        for emb in embeddings:
            if _GPU and _CUML_KMEANS is not None:
                km = _CUML_KMEANS(n_clusters=K, n_init=10, random_state=seed)
                labs.append(_CP.asnumpy(km.fit_predict(emb[:, :K])))
            else:
                labs.append(KMeans(K, n_init=10, random_state=seed).fit_predict(emb[:, :K]))
        nmis = []
        for i in range(B):
            for j in range(i + 1, B):
                inter = np.intersect1d(idxs[i], idxs[j], assume_unique=True)
                if len(inter) < 4:
                    continue
                li = labs[i][[pi_list[i][p] for p in inter]]
                lj = labs[j][[pi_list[j][p] for p in inter]]
                if len(np.unique(li)) < 2 or len(np.unique(lj)) < 2:
                    continue
                nmis.append(normalized_mutual_info_score(li, lj))
        if nmis:
            arr = np.asarray(nmis, dtype=float)
            stats[K] = {
                "mean": float(arr.mean()),
                "se": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
                "n_pairs": int(arr.size),
            }
        else:
            stats[K] = {"mean": -1.0, "se": 0.0, "n_pairs": 0}
    return stats


def one_se_select_K(stats: dict) -> int:
    """Parsimony rule for flat stability curves.

    Returns the smallest candidate K whose mean stability is within one standard
    error of the largest mean (Breiman's one-standard-error rule applied to the
    stability curve).  When the curve has a sharp maximum the rule returns that
    maximizer; when the curve is flat it returns the most parsimonious scale
    that is statistically indistinguishable from the maximum.
    """
    means = {int(k): float(v["mean"]) for k, v in stats.items()}
    best = max(means, key=means.get)
    threshold = means[best] - float(stats[best]["se"])
    return min(k for k, m in means.items() if m >= threshold)


def hierarchical_stability_select(A, Kmax: int = 10, B: int = 40,
                                   frac: float = 0.8, sample_cap: int | None = None,
                                   seed: int = 0, cont_margin: float = 0.10,
                                   cont_floor: float = 0.05):
    """Stability selection refined by adjacent-partition agreement.

    Plain stability selection (``stability_select_K``) reports the cluster count
    that is most self-consistent under resampling. That count is not necessarily
    the semantically meaningful one: a coarse stability maximum such as ``K=2`` on
    ``sklearn.datasets.load_digits`` attains the highest subsample NMI because
    it is the coarsest split, yet increasing the count by one changes much of
    the partition (the adjacent-partition NMI is low). A retained count should
    therefore be both self-stable and reasonably similar to its neighboring
    candidate partition.

    This selector therefore:
      1. evaluates the full stability curve (mean subsample NMI per ``K``);
      2. computes adjacent-partition agreement for every ``K``, defined as the
         NMI between the ``K``-cluster and ``(K+1)``-cluster partitions;
      3. if the plain argmax ``K*`` is a coarse stability maximum --- its
         agreement is well below the median agreement --- the argmax is
         rejected and the most self-stable alternative whose continuity is
         near the median is returned instead; otherwise ``K*`` is kept.

    A coarse stability maximum is a cluster count that is highly self-consistent
    yet differs substantially from the next candidate partition. This is the
    failure observed for the plain argmax on
    ``load_digits`` (``K=2``) and on the four-Gaussian ``Blobs`` (``K=2``),
    where the coarsest split wins the self-stability race but is not the
    semantically meaningful scale.

    The two free constants ``cont_margin`` and ``cont_floor`` control how far
    below the median agreement a count must fall to trigger replacement, and how
    close to the median a replacement must be. Their defaults are held fixed
    across every experiment in the manuscript.

    Returns ``(best_K, stability, continuity)``.
    """
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    stats = stability_curve_stats(A, Kmax=Kmax, B=B, frac=frac,
                                  sample_cap=sample_cap, seed=seed)
    stab = {int(K): float(v["mean"]) for K, v in stats.items()}

    # Continuity c(K) = NMI between the K- and (K+1)-cluster partitions.
    continuity = {}
    for K in range(2, Kmax):
        lK = spectral_partition(A, K, seed=seed)
        lK1 = spectral_partition(A, K + 1, seed=seed)
        continuity[K] = normalized_mutual_info_score(lK, lK1)

    best_K = max(stab, key=stab.get)
    if continuity:
        median_cont = float(np.median(list(continuity.values())))
        # The continuity C(K) = NMI(P_K, P_{K+1}) is defined only for K in
        # [2, Kmax-1]; the boundary candidate K=Kmax has no finer neighbour
        # to refine into, so it is not assigned a continuity value.  This
        # matters for two reasons.  First, an argmax that lands exactly on the
        # The boundary cannot be tested for the coarse-stability-maximum signature,
        # so it is kept as-is (the "not definable" case is distinct from
        # "continuous").  Second, the boundary candidate is excluded from the
        # alternative set below, so it can never be selected as a replacement
        # on the strength of a phantom continuity value.
        best_cont = continuity.get(best_K)
        if best_cont is not None and best_cont < median_cont - cont_margin:
            alternatives = [
                K for K in range(2, Kmax)  # exclude boundary K=Kmax
                if continuity[K] >= median_cont - cont_floor
            ]
            if alternatives:
                best_K = max(alternatives, key=lambda K: stab[K])

    return best_K, stab, continuity


def stability_select_K(A, Kmax: int = 10, B: int = 40, frac: float = 0.8,
                       sample_cap: int | None = None, seed: int = 0):
    """Subsampling-without-replacement NMI consensus to pick K.

    For each candidate K in {2,...,Kmax}, draw B subsamples without replacement,
    compute one Kmax-dimensional spectral embedding per subsample, and cluster
    its leading K coordinates.  The optional sample_cap bounds the subsample
    size for large graphs. Returns (best_K, stability_dict).
    """
    stats = stability_curve_stats(A, Kmax=Kmax, B=B, frac=frac,
                                  sample_cap=sample_cap, seed=seed)
    stab = {K: v["mean"] for K, v in stats.items()}
    best_K = max(stab, key=stab.get)
    return best_K, stab


# ---------------------------------------------------------------------------
# 4. End-to-end CNC
# ---------------------------------------------------------------------------
def cnc_cluster(X: np.ndarray, k: int = 20, rank_exponent: float = 2.0,
               Kmax: int = 10, B: int = 40, frac: float = 0.8,
               sample_cap: int | None = None, seed: int = 0,
               selector: str = "hss"):
    """Run CNC end-to-end and return labels, chosen K, and diagnostics.

    The ``selector`` argument chooses the cluster-count rule:

      * ``"hss"`` (default) -- hierarchical stability selection: the
        adjacent-partition-agreement refinement implemented by
        :func:`hierarchical_stability_select`, which rejects a coarse stability
        maximum (e.g. ``K=2`` on Digits / Blobs) when its next partition differs
        substantially;
      * ``"argmax"`` -- the plain subsampling-NMI argmax of
        :func:`stability_select_K`.

    Returns dict with keys: labels, K, stab, R, n_edges.
    """
    R, _, n_edges = resonance_affinity(X, k=k, rank_exponent=rank_exponent)
    if selector == "hss":
        K, stab, _ = hierarchical_stability_select(R, Kmax=Kmax, B=B, frac=frac,
                                                   sample_cap=sample_cap, seed=seed)
    elif selector == "argmax":
        K, stab = stability_select_K(R, Kmax=Kmax, B=B, frac=frac,
                                     sample_cap=sample_cap, seed=seed)
    else:
        raise ValueError(f"unknown selector {selector!r}; expected 'hss' or 'argmax'")
    labels = spectral_partition(R, K, seed=seed)
    return {
        "labels": labels,
        "K": K,
        "stab": stab,
        "R": R,
        "n_edges": n_edges,
    }


# ---------------------------------------------------------------------------
# 5. Block-recovery diagnostics (local support / local separation)
# ---------------------------------------------------------------------------
def block_purity(labels: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of points whose cluster majority label matches their true label."""
    n = len(labels)
    correct = 0
    for k in np.unique(labels):
        mask = labels == k
        vals, counts = np.unique(y_true[mask], return_counts=True)
        correct += counts[np.argmax(counts)]
    return float(correct / n)


def block_recovery(R, labels: np.ndarray, y_true: np.ndarray) -> dict:
    """Measure local support and local separation on the affinity graph."""
    R = R.tocoo()
    r, c, v = R.row, R.col, R.data
    same = y_true[r] == y_true[c]
    within = v[same]
    cross = v[~same]
    return {
        "local_support": float(within.sum() / (v.sum() + 1e-12)),
        "mean_within": float(within.mean() if len(within) else 0.0),
        "mean_cross": float(cross.mean() if len(cross) else 0.0),
        "n_within": int(len(within)),
        "n_cross": int(len(cross)),
    }
