# -*- coding: utf-8 -*-
"""Expanded benchmark for the CNC paper (CMC submission version).

Adds three baselines that the original submission lacked:
  * SNN  -- Shared-Nearest-Neighbor affinity (the binary-weight special case
            of CNC) followed by the *same* spectral partition, so CNC and SNN
            are compared at identical K and only the affinity differs;
  * RBF  -- standard (Gaussian / RBF) spectral clustering, the baseline CNC
            is claimed to beat, run at the oracle true K;
  * HDBSCAN -- variable-density density clustering (the method that fixes
            DBSCAN's global-epsilon weakness), fully automatic K.

It also adds two larger / higher-dimensional / real-labeled scenarios to
address the "toy dataset" criticism:
  * Gauss20D -- 3000 points in 20 dimensions, 5 Gaussian components;
  * Digits   -- sklearn ``load_digits`` (1797 x 64, 10 real classes).

Outputs:
  * results/benchmark_results.json  (all numbers, paper source of truth)
  * figures/paper_figures/fig_benchmark_expanded.pdf  (5 established methods)
  * figures/paper_figures/fig_baseline_compare.pdf     (CNC vs SNN/RBF/HDBSCAN)

Run:  python code/benchmark_expanded.py
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs, load_digits
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score)
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

try:
    import hdbscan as HDB
    HAVE_HDB = True
except Exception:
    HAVE_HDB = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnc_clustering as C

ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")
PAPER_FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PAPER_FIGS, exist_ok=True)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# ---------------- DPC (Rodriguez & Laio 2014, Science) ----------------
def dpc(X, K, dc_frac=0.02):
    """Density-Peaks Clustering, auto-select top-K centers by gamma=rho*delta."""
    n = X.shape[0]
    D = cdist(X, X)
    triu = D[np.triu_indices(n, 1)]
    dc = float(np.quantile(triu, dc_frac))
    rho = np.exp(-(D / dc) ** 2).sum(axis=1) - 1.0
    order = np.argsort(rho)[::-1]
    delta = np.zeros(n)
    for i in range(n):
        higher = np.where(rho > rho[i])[0]
        if higher.size == 0:
            delta[i] = D[i].max()
        else:
            delta[i] = D[i, higher].min()
    gamma = rho * delta
    centers = np.argsort(gamma)[::-1][:K]
    center_of = -np.ones(n, dtype=int)
    for ci, c in enumerate(centers):
        center_of[c] = ci
    for i in order:
        if center_of[i] != -1:
            continue
        higher = np.where(rho > rho[i])[0]
        j = higher[np.argmin(D[i, higher])]
        center_of[i] = center_of[j]
    return center_of.astype(int)


# ---------------- AP (Frey & Dueck 2007, Science) ----------------
def ap_cluster(X):
    """Affinity Propagation; returns (labels, n_clusters)."""
    ap = AffinityPropagation(random_state=0, damping=0.5).fit(X)
    lab = ap.labels_
    ncl = len(set(lab.tolist()) - {-1})
    return lab, ncl


# ---------------- SNN affinity (binary-weight special case of CNC) ----
def snn_affinity(X, k=20):
    """Shared-Nearest-Neighbor count matrix as a sparse affinity.

    For every point we record its k-NN; the affinity between i and j is the
    number of neighbors they share. This is exactly CNC's resonance score
    with binary (instead of rank-weighted, cosine-normalized) weights.
    """
    n = X.shape[0]
    _, idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X).kneighbors(X)
    idx = idx[:, 1:]
    rows, cols = [], []
    for nb in idx:
        r = np.repeat(nb, k)
        c = np.tile(nb, k)
        rows.append(r)
        cols.append(c)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    A = np.zeros((n, n))
    np.add.at(A, (rows, cols), 1.0)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    from scipy.sparse import csr_matrix
    return csr_matrix(A)


# ---------------- RBF (Gaussian) affinity: standard spectral baseline ----
def rbf_affinity(X, sigma=None):
    """Standard Gaussian (RBF) affinity with sigma = median pairwise distance."""
    D = cdist(X, X)
    if sigma is None:
        d = D[D > 0]
        sigma = float(np.median(d)) if d.size else 1.0
    A = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(A, 0.0)
    from scipy.sparse import csr_matrix
    return csr_matrix(A)


def spectral_at_K(A, K, seed=0):
    """Reuse CNC's normalized-Laplacian spectral partition at a fixed K."""
    return C.spectral_partition(A, K, seed=seed)


def metric(y, pred):
    pred = np.asarray(pred)
    return (float(adjusted_rand_score(y, pred)),
            float(adjusted_mutual_info_score(y, pred)))


def hdbscan_cluster(X, min_cluster_size=15):
    """HDBSCAN (variable-density). Returns (labels, K) with automatic K."""
    if not HAVE_HDB:
        return np.full(X.shape[0], -1, dtype=int), 0
    c = HDB.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None)
    lab = c.fit_predict(X)
    K = len(set(lab.tolist()) - {-1})
    return lab, K


def run_one(X, y, k=20, Kmax=10, B=40, oracle_K=None):
    """Run all methods on one dataset; return a dict of per-method results."""
    n_true = int(len(np.unique(y)))
    Ktrue = n_true if oracle_K is None else oracle_K

    # CNC (automatic K via stability selection)
    lab, K_cnc, _ = C.cnc_cluster(X, k=k, Kmax=Kmax, B=B)
    # KMeans (oracle K)
    km = KMeans(n_clusters=Ktrue, n_init=10, random_state=0).fit_predict(X)
    # DBSCAN (fixed global epsilon -- the known weak-strawman)
    db = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)
    nd = len(set(db.tolist()) - {-1})
    # DPC (oracle K)
    dp = dpc(X, Ktrue)
    # AP (automatic)
    ap_lab, n_ap = ap_cluster(X)
    # SNN affinity + spectral at oracle K (CNC's binary special case)
    Asnn = snn_affinity(X, k=k)
    snn_lab = spectral_at_K(Asnn, Ktrue)
    # RBF spectral at oracle K (standard spectral clustering baseline)
    Arbf = rbf_affinity(X)
    rbf_lab = spectral_at_K(Arbf, Ktrue)
    # HDBSCAN (automatic K, variable density)
    hdb_lab, n_hdb = hdbscan_cluster(X)

    res = {
        "n_true": n_true,
        "CNC":        {"K": int(K_cnc),  "ARI": metric(y, lab)[0],    "AMI": metric(y, lab)[1]},
        "KMeans":     {"K": Ktrue,       "ARI": metric(y, km)[0],     "AMI": metric(y, km)[1]},
        "DBSCAN":     {"K": int(nd),     "ARI": metric(y, db)[0],     "AMI": metric(y, db)[1]},
        "DPC":        {"K": Ktrue,       "ARI": metric(y, dp)[0],     "AMI": metric(y, dp)[1]},
        "AP":         {"K": int(n_ap),   "ARI": metric(y, ap_lab)[0], "AMI": metric(y, ap_lab)[1]},
        "SNN":        {"K": Ktrue,       "ARI": metric(y, snn_lab)[0],"AMI": metric(y, snn_lab)[1]},
        "RBFSpectral":{"K": Ktrue,       "ARI": metric(y, rbf_lab)[0],"AMI": metric(y, rbf_lab)[1]},
        "HDBSCAN":    {"K": int(n_hdb),  "ARI": metric(y, hdb_lab)[0],"AMI": metric(y, hdb_lab)[1]},
    }
    return res


def build_datasets(N=1000):
    Xa, ya = make_blobs(n_samples=N, centers=4, random_state=0)
    Xa = Xa @ np.array([[0.6, -0.6], [-0.4, 0.8]]) * 0.5
    Xv, yv = make_blobs(n_samples=N, centers=4, cluster_std=[0.4, 0.8, 0.3, 1.2],
                        random_state=2)
    datasets = {
        "Moons":       make_moons(n_samples=N, noise=0.05, random_state=0),
        "Circles":     make_circles(n_samples=N, factor=0.5, noise=0.05, random_state=0),
        "Blobs":       make_blobs(n_samples=N, centers=4, random_state=0),
        "Aniso":       (Xa, ya),
        "VariedBlobs": (Xv, yv),
    }
    return datasets


def main():
    k, Kmax, B = 20, 10, 40

    # --- five classical synthetic-structure datasets (enlarged to N=1000) ---
    datasets = build_datasets(N=1000)
    out = {"synthetic": {}}
    for name, data in datasets.items():
        X, y = data
        r = run_one(X, y, k=k, Kmax=Kmax, B=B)
        out["synthetic"][name] = r
        log(f"{name:12s} CNC(K={r['CNC']['K']}) ARI={r['CNC']['ARI']:.2f} | "
            f"KM={r['KMeans']['ARI']:.2f} DB={r['DBSCAN']['ARI']:.2f}(K={r['DBSCAN']['K']}) "
            f"DPC={r['DPC']['ARI']:.2f} AP={r['AP']['ARI']:.2f}(K={r['AP']['K']}) | "
            f"SNN={r['SNN']['ARI']:.2f} RBF={r['RBFSpectral']['ARI']:.2f} "
            f"HDB={r['HDBSCAN']['ARI']:.2f}(K={r['HDBSCAN']['K']})")

    # --- high-dimensional Gaussian mixture (scale / dimensionality) ---
    rng = np.random.RandomState(3)
    G = 5
    cen = rng.randn(G, 20) * 6.0
    Xg, yg = [], []
    for c in range(G):
        Xg.append(rng.randn(600, 20) + cen[c])
        yg += [c] * 600
    Xg, yg = np.vstack(Xg), np.array(yg)
    rg = run_one(Xg, yg, k=20, Kmax=10, B=20)
    out["Gauss20D"] = rg
    log(f"{'Gauss20D':12s} CNC(K={rg['CNC']['K']}) ARI={rg['CNC']['ARI']:.2f} | "
        f"SNN={rg['SNN']['ARI']:.2f} RBF={rg['RBFSpectral']['ARI']:.2f} "
        f"HDB={rg['HDBSCAN']['ARI']:.2f}(K={rg['HDBSCAN']['K']})")

    # --- real, labeled, high-dimensional scenario: sklearn digits ---
    dig = load_digits()
    Xd, yd = dig.data.astype(np.float64), dig.target
    # z-score per feature for fair affinity
    mu, sd = Xd.mean(0), Xd.std(0); sd[sd < 1e-9] = 1.0
    Xd = (Xd - mu) / sd
    rd = run_one(Xd, yd, k=20, Kmax=15, B=20, oracle_K=10)
    out["Digits"] = rd
    log(f"{'Digits':12s} CNC(K={rd['CNC']['K']}) ARI={rd['CNC']['ARI']:.2f} | "
        f"KM={rd['KMeans']['ARI']:.2f} DB={rd['DBSCAN']['ARI']:.2f}(K={rd['DBSCAN']['K']}) "
        f"DPC={rd['DPC']['ARI']:.2f} AP={rd['AP']['ARI']:.2f}(K={rd['AP']['K']}) | "
        f"SNN={rd['SNN']['ARI']:.2f} RBF={rd['RBFSpectral']['ARI']:.2f} "
        f"HDB={rd['HDBSCAN']['ARI']:.2f}(K={rd['HDBSCAN']['K']})")

    out_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"saved: {out_path}")

    # ---- Figure 1: ARI of the five established methods on the 5 synthetic ----
    R = out["synthetic"]
    order = ["Moons", "Circles", "Blobs", "Aniso", "VariedBlobs"]
    methods = ["CNC", "KMeans", "DBSCAN", "DPC", "AP"]
    colors = ["#1f78b4", "#d95f0e", "#31a354", "#984ea3", "#e41a1c"]
    styles = {"CNC": "auto K", "KMeans": "oracle K", "DBSCAN": "fixed eps",
              "DPC": "oracle K", "AP": "auto K"}
    x = np.arange(len(order)); w = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for m, ci in zip(methods, range(len(methods))):
        vals = [R[d][m]["ARI"] for d in order]
        ax.bar(x + (ci - 2) * w, vals, w, label=f"{m} ({styles[m]})", color=colors[ci])
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("Adjusted Rand Index (ARI)")
    ax.set_ylim(0, 1.12); ax.axhline(0, color="k", lw=0.6); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Clustering quality vs ground truth (ARI) -- 5 established methods, 5 synthetic datasets",
                 fontsize=12)
    ax.legend(fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig_path = os.path.join(PAPER_FIGS, "fig_benchmark_expanded.pdf")
    fig.savefig(fig_path, dpi=150)
    log(f"saved: {fig_path}")

    # ---- Figure 2: CNC vs its three closest baselines (SNN/RBF/HDBSCAN) ----
    bc_order = ["Moons", "Circles", "Blobs", "Aniso", "VariedBlobs", "Gauss20D", "Digits"]
    bc_methods = ["CNC", "SNN", "RBFSpectral", "HDBSCAN"]
    bc_colors = ["#1f78b4", "#999999", "#e41a1c", "#31a354"]
    bc_styles = {"CNC": "auto K", "SNN": "binary (oracle K)",
                 "RBFSpectral": "RBF (oracle K)", "HDBSCAN": "auto K"}
    xb = np.arange(len(bc_order)); wb = 0.2
    fig2, ax2 = plt.subplots(figsize=(12, 5.2))
    for m, ci in zip(bc_methods, range(len(bc_methods))):
        src = out["synthetic"] if m == "CNC" else None
        vals = []
        for d in bc_order:
            if d in out["synthetic"]:
                vals.append(out["synthetic"][d][m]["ARI"])
            elif d == "Gauss20D":
                vals.append(out["Gauss20D"][m]["ARI"])
            else:
                vals.append(out["Digits"][m]["ARI"])
        ax2.bar(xb + (ci - 1.5) * wb, vals, wb, label=f"{m} ({bc_styles[m]})", color=bc_colors[ci])
    ax2.set_xticks(xb); ax2.set_xticklabels(bc_order, fontsize=9, rotation=15)
    ax2.set_ylabel("Adjusted Rand Index (ARI)")
    ax2.set_ylim(0, 1.12); ax2.axhline(0, color="k", lw=0.6); ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("CNC vs its closest baselines: SNN (binary special case), RBF-spectral, HDBSCAN",
                  fontsize=12)
    ax2.legend(fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig2.tight_layout(rect=[0, 0.1, 1, 1])
    fig2_path = os.path.join(PAPER_FIGS, "fig_baseline_compare.pdf")
    fig2.savefig(fig2_path, dpi=150)
    log(f"saved: {fig2_path}")


if __name__ == "__main__":
    main()
