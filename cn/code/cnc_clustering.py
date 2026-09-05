"""
CNC -- Consensus-Neighborhood Clustering (邻域共识聚类)  v4 (unified, single-path)

============================================================================
Design rationale (novelty -- linearly independent of existing methods)
============================================================================
The clustering criterion is whether two points "see the same neighborhood",
rather than Euclidean distance, density peaks, or message passing. Points on
the same local manifold should have highly similar nearest-neighbor rankings;
points across clusters undergo a sudden change of neighborhood structure.

Core primitive -- resonance score R(i,j):
    take the k nearest neighbors of each point, assign decreasing weights by
    rank w(rank) = (k - rank)^2.  Treat as sparse vectors; the resonance of two
    points is the cosine similarity of these vectors (contributing only on
    shared neighbors).  Finer than Shared Nearest Neighbor (SNN): it looks at
    both WHICH neighbors are shared AND whether their RANKINGS agree.
    This is the ONLY similarity source -- no Euclidean term is mixed in.

============================================================================
Unified partitioning pipeline (v4: single path, unsupervised, principled)
============================================================================
    (1) Resonance affinity graph: build affinity matrix A (pure neighborhood
        consensus, no distance term) using the primitive above.
    (2) Stability selection for K (unsupervised): bootstrap subsampling
        (80% of points per round) -> spectral clustering on the subsampled
        affinity graph -> NMI consensus across subsamples -> pick K with the
        highest consensus.  This is a recognized robust K-estimator for
        spectral clustering; it avoids the silhouette trap and the eigengap
        overestimation on dense spectra.
    (3) Spectral partition: spectral clustering on the full resonance affinity
        graph (normalized Laplacian + embedding + KMeans) -> final K clusters.

Dependencies: numpy, scipy, scikit-learn, matplotlib.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

# ---------------- reproducible, location-independent output paths ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_FIGS = os.path.join(ROOT, "paper", "figs")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(PAPER_FIGS, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Resonance graph construction (novelty primitive)
# ----------------------------------------------------------------------------
def resonance_graph(X, k=10):
    """Return (edges, n), edges = [(i, j, R), ...].
    R(i,j) = cosine similarity of neighborhood-weight vectors (contributing
    only on shared neighbors), lying in [0, 1]."""
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]
    ranks = np.arange(k)
    weights = (k - ranks).astype(float) ** 2   # sharper: nearest neighbors dominate

    wdict = [dict(zip(idx[i], weights)) for i in range(n)]
    norm = np.array([np.sqrt(sum(v * v for v in wdict[i].values())) for i in range(n)])

    # strict inner product: dot(i,j) = sum_{c in N(i)∩N(j)} w_i(c) * w_j(c)
    dot = {}
    for i in range(n):
        nb_i = wdict[i]
        for c, wi_c in nb_i.items():
            for j, wi_j in nb_i.items():        # j is also a neighbor of i
                if j <= i:
                    continue
                wj_c = wdict[j].get(c)          # is c also a neighbor of j
                if wj_c is None:
                    continue
                dot[(i, j)] = dot.get((i, j), 0.0) + wi_c * wj_c

    edges = []  # (i, j, R)
    for (i, j), d in dot.items():
        denom = norm[i] * norm[j]
        if denom <= 0:
            continue
        r = min(d / denom, 1.0)
        edges.append((i, j, r))
    return edges, n


def resonance_affinity(X, k=20, power=1.0):
    """Build the resonance affinity matrix A (pure neighborhood consensus,
    optional power sharpening)."""
    edges, n = resonance_graph(X, k)
    A = np.zeros((n, n))
    for i, j, r in edges:
        A[i, j] = r ** power
        A[j, i] = r ** power
    return A, n, edges


# ----------------------------------------------------------------------------
# 2. Spectral partition
# ----------------------------------------------------------------------------
def _spectral_labels(A, K, seed=0):
    """Spectral clustering on affinity graph A -> K cluster labels."""
    n = A.shape[0]
    d = A.sum(axis=1)
    d[d <= 0] = 1e-12
    Dinv = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - Dinv @ A @ Dinv          # normalized Laplacian
    w, V = np.linalg.eigh(L)
    w = np.clip(w, 0, None)
    emb = V[:, :K]                            # first K eigenvectors
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(emb)


def spectral_partition(A, K, seed=0):
    """Sparse, scalable spectral partition on affinity graph A (sparse or
    dense) -> K cluster labels, using the normalized Laplacian and a sparse
    eigensolver.  Identical machinery to the full CNC pipeline, exposed so that
    alternative affinities (e.g., the binary SNN variant, standard RBF) can be
    partitioned under the same conditions."""
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    n = A.shape[0]
    d = np.asarray(A.sum(1)).ravel()
    d[d <= 0] = 1e-12
    dinv = diags(1.0 / np.sqrt(d))
    L = (diags(np.ones(n)) - dinv @ A @ dinv).tocsr()
    try:
        ev, evec = eigsh(L, k=K + 1, sigma=0.0, which="LM", tol=1e-4, maxiter=2000)
    except Exception:
        ev, evec = eigsh(L, k=K + 1, which="SM", tol=1e-3, maxiter=4000)
    order = np.argsort(ev)
    emb = evec[:, order[1:K + 1]]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return KMeans(K, n_init=10, random_state=seed).fit_predict(emb)


# ----------------------------------------------------------------------------
# 3. Stability selection for K (unsupervised, principled)
# ----------------------------------------------------------------------------
def stability_select_K(A, Kmax=10, B=40, frac=0.8, seed=0, all_pairs=True):
    """Bootstrap subsampling + NMI consensus across subsamples; pick K with the
    highest consensus.  Fully unsupervised.  Returns (best_K, stab_dict)."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    stab = {}
    for K in range(2, Kmax + 1):
        labs, idxs = [], []
        for b in range(B):
            perm = rng.choice(n, size=int(frac * n), replace=False)
            Ab = A[np.ix_(perm, perm)]
            labs.append(_spectral_labels(Ab, K))
            idxs.append(perm)
        pi_list = [{p: q for q, p in enumerate(idxs[i])} for i in range(B)]
        nmis = []
        if all_pairs:
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
        else:
            for i in range(B - 1):
                inter = np.intersect1d(idxs[i], idxs[i + 1], assume_unique=True)
                if len(inter) < 4:
                    continue
                li = labs[i][[pi_list[i][p] for p in inter]]
                lj = labs[i + 1][[pi_list[i + 1][p] for p in inter]]
                if len(np.unique(li)) < 2 or len(np.unique(lj)) < 2:
                    continue
                nmis.append(normalized_mutual_info_score(li, lj))
        stab[K] = float(np.mean(nmis)) if nmis else -1.0
    best_K = max(stab, key=stab.get)
    return best_K, stab


# ----------------------------------------------------------------------------
# 4. CNC entry point (unified single path)
# ----------------------------------------------------------------------------
def cnc_cluster(X, k=20, Kmax=10, B=40, frac=0.8, seed=0):
    """CNC clustering (unified single path).

    Pipeline: resonance affinity graph -> stability selection for K ->
    spectral partition.  Returns (labels, K, diag) where
    diag = (K, n_edges, R_min, R_max, R_med, R_p90, max_stability)."""
    A, n, edges = resonance_affinity(X, k)
    if not edges:
        return np.zeros(n, dtype=int), 1, (1, 0, 0, 1, 0.5, 0.5, 1.0)
    R = np.array([e[2] for e in edges])
    Rs = np.sort(R)[::-1]
    K, stab = stability_select_K(A, Kmax, B, frac, seed)
    labels = _spectral_labels(A, K)
    diag = (K, len(edges), float(Rs.min()), float(Rs.max()),
            float(np.percentile(Rs, 50)), float(np.percentile(Rs, 90)),
            max(stab.values()))
    return labels, K, diag


# ----------------------------------------------------------------------------
# 5. Benchmark + visualization (synthetic data)
# ----------------------------------------------------------------------------
def score(true, pred):
    mask = pred != -1
    if mask.sum() == 0:
        return float("nan"), float("nan")
    return (adjusted_rand_score(true[mask], pred[mask]),
            adjusted_mutual_info_score(true[mask], pred[mask]))


def main():
    k, Kmax, B = 20, 10, 40
    Xa, ya = make_blobs(n_samples=600, centers=4, random_state=0)
    Xa = Xa @ np.array([[0.6, -0.6], [-0.4, 0.8]]) * 0.5

    datasets = {
        "Moons (non-convex)": make_moons(n_samples=600, noise=0.05, random_state=0),
        "Circles (concentric)": make_circles(n_samples=600, factor=0.5, noise=0.05, random_state=0),
        "Blobs (spherical)": make_blobs(n_samples=600, centers=4, random_state=0),
        "Aniso (anisotropic)": (Xa, ya),
    }

    methods, results, n_cnc = {}, {}, {}
    for name, data in datasets.items():
        X, y = data
        n_true = len(np.unique(y))
        lab, K, diag = cnc_cluster(X, k=k, Kmax=Kmax, B=B)
        km = KMeans(n_clusters=n_true, n_init=10, random_state=0).fit_predict(X)
        db = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)
        methods[name] = {"X": X, "y": y, "cnc": lab, "km": km, "db": db}
        results[name] = {
            "CNC": score(y, lab),
            "KMeans": score(y, km),
            "DBSCAN": score(y, db),
        }
        n_cnc[name] = K
        print(f"[debug {name}] CNC K={K} (trueK={n_true}) | "
              f"R[min={diag[2]:.3f}, med={diag[4]:.3f}, p90={diag[5]:.3f}, max={diag[3]:.3f}]")

    # ---- Fig 1: CNC clustering results ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, m) in zip(axes.ravel(), methods.items()):
        lab = m["cnc"]
        uniq = np.unique(lab)
        cmap = plt.get_cmap("tab10")
        for ci, c in enumerate(uniq):
            sel = lab == c
            ax.scatter(m["X"][sel, 0], m["X"][sel, 1], s=11, c=[cmap(ci % 10)])
        ax.set_title(f"{name}  (CNC clusters={len(uniq)})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CNC v4 (unified) -- resonance affinity + stability-selected spectral "
                 f"clustering  (k={k}, B={B}, auto K)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PAPER_FIGS, "cnc_result.png"), dpi=130)
    fig.savefig(os.path.join(PAPER_FIGS, "cnc_synthetic.png"), dpi=130)
    plt.close(fig)

    # ---- Fig 2: ARI of 3 methods ----
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    x = np.arange(len(names)); w = 0.26
    ari_c = [results[n]["CNC"][0] for n in names]
    ari_km = [results[n]["KMeans"][0] for n in names]
    ari_db = [results[n]["DBSCAN"][0] for n in names]
    ax.bar(x - w, ari_c, w, label="CNC (unified)", color="#1f78b4")
    ax.bar(x, ari_km, w, label="KMeans", color="#d95f0e")
    ax.bar(x + w, ari_db, w, label="DBSCAN", color="#31a354")
    for xi, (kc, kk, kd) in enumerate(zip(ari_c, ari_km, ari_db)):
        ax.text(xi - w, kc + 0.02, f"K={n_cnc[names[xi]]}", ha="center", fontsize=8, color="#1f78b4")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Adjusted Rand Index (higher is better)")
    ax.set_ylim(0, 1.12); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Clustering quality vs ground truth (ARI)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cnc_benchmark.png"), dpi=130)
    fig.savefig(os.path.join(PAPER_FIGS, "cnc_benchmark.pdf"), dpi=130)
    plt.close(fig)

    # ---- numbers ----
    print("=" * 80)
    print(f"{'dataset':<22}{'CNC(ARI/AMI)':<18}{'K':<5}{'KMeans':<14}{'DBSCAN':<8}")
    print("-" * 80)
    for name in names:
        r = results[name]
        def fmt(t): return f"{t[0]:.2f}/{t[1]:.2f}"
        print(f"{name:<22}{fmt(r['CNC']):<18}{n_cnc[name]:<5}{fmt(r['KMeans']):<14}{fmt(r['DBSCAN']):<8}")
    print("=" * 80)
    print(f"saved: {os.path.join(PAPER_FIGS, 'cnc_result.png')}, "
          f"{os.path.join(RESULTS_DIR, 'cnc_benchmark.png')}")


if __name__ == "__main__":
    main()
