# -*- coding: utf-8 -*-
"""
Cross-affinity validation of the adjacent-partition-agreement refinement.

This script compares the same five cluster-number selectors -- plain argmax,
one-standard-error (1-SE), eigengap, silhouette, and the proposed
adjacent-partition-agreement refinement (HSS) -- on TWO affinities:

  * CNC  : the rank-weighted shared-neighbour resonance graph (resonance_affinity)
  * RBF  : a kNN-truncated Gaussian (RBF) spectral affinity

The manuscript reports the seven labeled benchmarks by default. The two
hyperspectral datasets can be added explicitly with ``--include-large``.

Outputs:
  results/cross_affinity_validation.json   -- per-dataset, per-affinity, per-selector K & ARI
  results/margin_sensitivity_expanded.json -- margin x floor grid on ALL labeled benchmarks

Run from the code/ directory (so cnc / baselines / datasets import cleanly).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

import cnc
import experiments as E
import datasets as ds

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 0


# ---------------------------------------------------------------------------
# RBF (Gaussian) spectral affinity, kNN-truncated -- mirrors baselines.rbf_partition_knn
# but returns the sparse affinity matrix so it can drive the same selectors.
# ---------------------------------------------------------------------------
def rbf_affinity(X: np.ndarray, k: int = 20, sigma_samples: int = 2000,
                 sigma: float | None = None) -> csr_matrix:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if sigma is None:
        rng = np.random.default_rng(SEED)
        sub = X[rng.choice(n, min(sigma_samples, n), replace=False)] if n > sigma_samples else X
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
    A = A.maximum(A.T)          # symmetrize
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


# ---------------------------------------------------------------------------
# Per-affinity five-selector comparison (identical protocol to run_kselection)
# ---------------------------------------------------------------------------
def ari_of(A, y, K, seed=SEED):
    if y is None or len(np.unique(y)) < 2:
        return None
    lab = cnc.spectral_partition(A, K, seed=seed)
    return round(float(adjusted_rand_score(y, lab)), 4)


def compare_affinity(A, y, Kmax, B, sample_cap=None):
    n = A.shape[0]
    skip_sil = n > 3000            # silhouette_score is O(n^2); skip on large scenes
    stats = cnc.stability_curve_stats(A, Kmax=Kmax, B=B, frac=0.8,
                                     sample_cap=sample_cap, seed=SEED)
    stab = {int(k): float(v["mean"]) for k, v in stats.items()}
    continuity = E.continuity_independent(A, Kmax, seed=SEED)
    K_argmax = max(stab, key=stab.get)
    K_hss = E.hss_apply(stab, continuity)
    K_1se = cnc.one_se_select_K(stats)
    K_eig = E.eigengap_select(A, Kmax)
    rec = {
        "argmax":     {"K": int(K_argmax), "ARI": ari_of(A, y, K_argmax)},
        "1-SE":       {"K": int(K_1se),    "ARI": ari_of(A, y, K_1se)},
        "eigengap":   {"K": int(K_eig),    "ARI": ari_of(A, y, K_eig)},
        "HSS":        {"K": int(K_hss),    "ARI": ari_of(A, y, K_hss)},
    }
    if not skip_sil:
        K_sil = E.silhouette_select(A, Kmax)
        rec["silhouette"] = {"K": int(K_sil), "ARI": ari_of(A, y, K_sil)}
    return rec, stab, continuity


# ---------------------------------------------------------------------------
# Dataset list: (name, X, y, k_for_affinity, Kmax, B, sample_cap)
# matches run_kselection / run_indian_pines exactly (seed=0 data).
# ---------------------------------------------------------------------------
def labeled_benchmarks():
    syn = ds.synthetic_benchmarks()          # Moons, Circles, Blobs, Aniso, VariedBlobs (seed=0)
    Xg, yg = ds.make_gauss20d()
    Xd, yd = ds.make_digits()
    return [
        ("Moons",        *syn["Moons"],        20, 10, 40, None),
        ("Circles",      *syn["Circles"],      20, 10, 40, None),
        ("Blobs",        *syn["Blobs"],        20, 10, 40, None),
        ("Aniso",        *syn["Aniso"],        20, 10, 40, None),
        ("VariedBlobs",  *syn["VariedBlobs"],  20, 10, 40, None),
        ("Gauss20D",     Xg, yg,               20, 10, 40, None),
        ("Digits",       Xd, yd,               20, 15, 20, None),
    ]


def main(include_large: bool = False):
    t0 = time.time()
    out = {"seed": SEED, "datasets": {}}

    for name, X, y, k, Kmax, B, cap in labeled_benchmarks():
        true_K = int(len(np.unique(y)))
        # --- CNC resonance affinity ---
        R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
        cnc_rec, _, _ = compare_affinity(R, y, Kmax, B, cap)
        # --- RBF spectral affinity ---
        A = rbf_affinity(X, k=k)
        rbf_rec, _, _ = compare_affinity(A, y, Kmax, B, cap)
        out["datasets"][name] = {
            "true_K": true_K,
            "CNC": cnc_rec,
            "RBF": rbf_rec,
        }
        print(f"{name:12s} true={true_K} | CNC HSS K={cnc_rec['HSS']['K']} "
              f"argmax K={cnc_rec['argmax']['K']} | RBF HSS K={rbf_rec['HSS']['K']} "
              f"argmax K={rbf_rec['argmax']['K']}")

    # The manuscript's cross-affinity table covers the seven labeled benchmarks.
    if include_large:
        Xh, yh = ds.load_indian_pines()
        true_K = int(len(np.unique(yh)))
        Rh, _, _ = cnc.resonance_affinity(Xh, k=20)
        cnc_ip, _, _ = compare_affinity(Rh, yh, 20, 20, 4000)
        Ah = rbf_affinity(Xh, k=20)
        rbf_ip, _, _ = compare_affinity(Ah, yh, 20, 20, 4000)
        out["datasets"]["IndianPines"] = {
            "true_K": true_K,
            "CNC": cnc_ip,
            "RBF": rbf_ip,
        }
        print(f"{'IndianPines':12s} true={true_K} | CNC HSS K={cnc_ip['HSS']['K']} "
              f"| RBF HSS K={rbf_ip['HSS']['K']}")

    if include_large:
        Xa, _ = ds.load_ayc()
        Ra, _, _ = cnc.resonance_affinity(Xa, k=20)
        ayc_cnc, _, _ = compare_affinity(Ra, None, 24, 20, 4000)
        Aa = rbf_affinity(Xa, k=20)
        ayc_rbf, _, _ = compare_affinity(Aa, None, 24, 20, 4000)
        out["datasets"]["AsianGamesVillage"] = {
            "true_K": None,
            "CNC": ayc_cnc,
            "RBF": ayc_rbf,
        }
        print(f"{'AsianGamesVillage':12s} | CNC HSS K={ayc_cnc['HSS']['K']} "
              f"| RBF HSS K={ayc_rbf['HSS']['K']}")

    with open(os.path.join(RESULTS, "cross_affinity_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] cross_affinity_validation.json  ({time.time()-t0:.0f}s)")


# ---------------------------------------------------------------------------
# Expanded margin x floor sensitivity grid on both tested carriers.
# ---------------------------------------------------------------------------
def margin_sensitivity():
    t0 = time.time()
    margin_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    floor_grid = [0.02, 0.05, 0.08, 0.10]
    out = {"seed": SEED, "margin_grid": margin_grid, "floor_grid": floor_grid,
           "datasets": {}}
    for name, X, y, k, Kmax, B, cap in labeled_benchmarks():
        R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
        stats = cnc.stability_curve_stats(R, Kmax=Kmax, B=B, frac=0.8, seed=SEED)
        stab = {int(K): float(v["mean"]) for K, v in stats.items()}
        cont = E.continuity_independent(R, Kmax, seed=SEED)
        ari_per_K = {K: ari_of(R, y, K) for K in range(2, Kmax + 1)}
        grid_cnc = {}
        for m in margin_grid:
            for fl in floor_grid:
                Kg = E.hss_apply(stab, cont, m, fl)
                grid_cnc[f"m{m}_f{fl}"] = {"K": int(Kg), "ARI": ari_per_K.get(Kg)}
        # RBF carrier
        A = rbf_affinity(X, k=k)
        stats_r = cnc.stability_curve_stats(A, Kmax=Kmax, B=B, frac=0.8, seed=SEED)
        stab_r = {int(K): float(v["mean"]) for K, v in stats_r.items()}
        cont_r = E.continuity_independent(A, Kmax, seed=SEED)
        ari_r = {K: ari_of(A, y, K) for K in range(2, Kmax + 1)}
        grid_rbf = {}
        for m in margin_grid:
            for fl in floor_grid:
                Kg = E.hss_apply(stab_r, cont_r, m, fl)
                grid_rbf[f"m{m}_f{fl}"] = {"K": int(Kg), "ARI": ari_r.get(Kg)}
        out["datasets"][name] = {
            "true_K": int(len(np.unique(y))),
            "CNC": {"stab": {int(k): round(v, 4) for k, v in stab.items()},
                    "continuity": {int(k): round(v, 4) for k, v in cont.items()},
                    "grid": grid_cnc},
            "RBF": {"stab": {int(k): round(v, 4) for k, v in stab_r.items()},
                    "continuity": {int(k): round(v, 4) for k, v in cont_r.items()},
                    "grid": grid_rbf},
            "ari_per_K": ari_per_K,
        }
        print(f"{name:12s} margin-grid done")
    with open(os.path.join(RESULTS, "margin_sensitivity_expanded.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] margin_sensitivity_expanded.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-large", action="store_true",
        help="also run cross-affinity comparisons on Indian Pines and Asian Games Village",
    )
    args = parser.parse_args()
    main(include_large=args.include_large)
    margin_sensitivity()
