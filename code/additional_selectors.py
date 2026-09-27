# -*- coding: utf-8 -*-
"""
Additional established automatic K methods for the CNC selector comparison.

Reviewer point #1 of the CNC submission asked to contrast the proposed
adjacent-partition-agreement refinement against further established automatic
cluster-number methods beyond argmax / 1-SE / eigengap / silhouette.  This
script adds two reference methods that both operate on the *raw features* X
through the k-means distortion W_K (within-cluster sum of squares):

  * Gap statistic (Tibshirani, Walther & Hastie, 2001):
        Gap(K) = mean_b log W_K^(b) - log W_K
    over B uniform reference datasets drawn in the bounding box of X; choose the
    smallest K with Gap(K) >= Gap(K+1) - s_{K+1}, s_K = sd_K * sqrt(1 + 1/B).

  * Jump statistic (Sugar & James, 2003):
        JS(K) = W_K^{-d/2} - W_{K-1}^{-d/2}   (W_0^{-d/2} := 0)
    with d = number of features; choose K maximizing JS(K).

Because both selectors read only X (not the affinity A), they are, by
construction, independent of the affinity carrier; this is reported rather than
hidden.  For each labeled benchmark we report the K they select and the ARI of
the paper's CNC spectral partition evaluated at that K, so the ARI column is
directly comparable to Table~\\ref{tab:ksel}.

Outputs:
  results/additional_selectors.json   -- per-dataset gap/jump K, ARI, and curves
  (also prints a ready LaTeX tabular to stdout)

Run from the code/ directory.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import cnc
import datasets as ds

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

SEED = 0


# ---------------------------------------------------------------------------
# k-means distortion (within-cluster sum of squares, i.e. KMeans inertia)
# ---------------------------------------------------------------------------
def kmeans_W(X: np.ndarray, K: int, seed: int = SEED) -> float:
    return float(KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X).inertia_)


# ---------------------------------------------------------------------------
# Gap statistic (Tibshirani, Walther & Hastie, 2001)
# ---------------------------------------------------------------------------
def gap_statistic_select(X: np.ndarray, Kmax: int, B: int = 10, seed: int = SEED):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W = {K: kmeans_W(X, K, seed) for K in range(1, Kmax + 1)}
    lo, hi = X.min(axis=0), X.max(axis=0)
    logW_ref = {K: [] for K in range(1, Kmax + 1)}
    for _ in range(B):
        Xb = rng.uniform(lo, hi, size=(n, d))
        for K in range(1, Kmax + 1):
            logW_ref[K].append(np.log(kmeans_W(Xb, K, seed=seed)))
    gap, sk = {}, {}
    for K in range(1, Kmax + 1):
        arr = np.array(logW_ref[K])
        gap[K] = float(arr.mean() - np.log(W[K]))
        sk[K] = float(arr.std(ddof=1) * np.sqrt(1.0 + 1.0 / B))
    chosen = None
    for K in range(1, Kmax):
        if gap[K] >= gap[K + 1] - sk[K + 1]:
            chosen = K
            break
    if chosen is None:
        chosen = max(gap, key=gap.get)
    return chosen, {K: round(gap[K], 4) for K in gap}


# ---------------------------------------------------------------------------
# Jump statistic (Sugar & James, 2003)
# ---------------------------------------------------------------------------
def jump_statistic_select(X: np.ndarray, Kmax: int, seed: int = SEED):
    d = X.shape[1]
    y = d / 2.0
    W = {K: kmeans_W(X, K, seed) for K in range(1, Kmax + 1)}
    term = lambda K: 0.0 if K == 0 else W[K] ** (-y)
    js = {K: float(term(K) - term(K - 1)) for K in range(1, Kmax + 1)}
    chosen = max(js, key=js.get)
    return chosen, {K: round(js[K], 6) for K in js}


def ari_of(A, y, K, seed=SEED):
    if K < 2:
        return None  # degenerate: a single cluster has no meaningful ARI
    lab = cnc.spectral_partition(A, K, seed=seed)
    return round(float(adjusted_rand_score(y, lab)), 4)


def labeled_benchmarks():
    syn = ds.synthetic_benchmarks()
    Xg, yg = ds.make_gauss20d()
    Xd, yd = ds.make_digits()
    return [
        ("Moons",        *syn["Moons"],        20, 10, 40),
        ("Circles",      *syn["Circles"],      20, 10, 40),
        ("Blobs",        *syn["Blobs"],        20, 10, 40),
        ("Aniso",        *syn["Aniso"],        20, 10, 40),
        ("VariedBlobs",  *syn["VariedBlobs"],  20, 10, 40),
        ("Gauss20D",     Xg, yg,               20, 10, 40),
        ("Digits",       Xd, yd,               20, 15, 20),
    ]


def main():
    t0 = time.time()
    out = {"seed": SEED, "datasets": {}}
    print(f"{'Dataset':12s} {'trueK':5s} | {'gap K':6s} {'gap ARI':7s} | {'jump K':6s} {'jump ARI':7s}")
    print("-" * 60)
    for name, X, y, k, Kmax, B in labeled_benchmarks():
        true_K = int(len(np.unique(y)))
        # gap / jump select from raw features only
        Kg, gap_curve = gap_statistic_select(X, Kmax, B=10)
        Kj, jump_curve = jump_statistic_select(X, Kmax)
        # ARI of the paper's CNC spectral partition at the selected K
        R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
        ari_gap = ari_of(R, y, Kg)
        ari_jump = ari_of(R, y, Kj)
        out["datasets"][name] = {
            "true_K": true_K,
            "gap": {"K": int(Kg), "ARI": ari_gap, "curve": gap_curve},
            "jump": {"K": int(Kj), "ARI": ari_jump, "curve": jump_curve},
        }
        def fmt(a):
            return f"{a:.4f}" if a is not None else "n/a"
        print(f"{name:12s} {true_K:<5d} | {Kg:<6d} {fmt(ari_gap):<7} | {Kj:<6d} {fmt(ari_jump):<7}")

    # LaTeX tabular (compact, comparable to tab:ksel)
    rows = []
    for name in out["datasets"]:
        info = out["datasets"][name]
        tk = info["true_K"]
        ga = info['gap']['ARI']; ja = info['jump']['ARI']
        ga_s = f"{ga:.2f}" if ga is not None else "\\textendash"
        ja_s = f"{ja:.2f}" if ja is not None else "\\textendash"
        rows.append(
            f"{name} ({tk}) & {info['gap']['K']}\\,({ga_s}) "
            f"& {info['jump']['K']}\\,({ja_s})"
        )
    print("\n=== LaTeX (tab:ksel-additional body) ===")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Dataset (true $K$) & gap statistic & jump statistic \\\\")
    print("\\midrule")
    for r in rows:
        print(r + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    with open(os.path.join(RESULTS, "additional_selectors.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] additional_selectors.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
