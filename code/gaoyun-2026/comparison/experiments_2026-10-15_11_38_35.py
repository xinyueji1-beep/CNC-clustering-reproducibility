# -*- coding: utf-8 -*-
"""Single entry-point for every experiment in the CNC manuscript.

Each function below reproduces one manuscript result and writes it to
``results/`` (or ``figures/paper_figures/`` for figures). The package runner
also invokes the independent selector and cross-affinity experiments.

The heavy lifting lives in ``cnc.py`` (the method) and ``baselines.py`` (the
comparison methods); ``datasets.py`` supplies every dataset.  This module only
*orchestrates* the experiments.

Run from the ``code`` directory with the hologpu interpreter.
"""
from __future__ import annotations

# --- CNC package path bootstrap: make the three category folders importable ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _c in ("core", "comparison", "visualization"):
    _p = _os.path.join(_ROOT, _c)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import json
import os
import pickle
import time
import warnings

import numpy as np
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

import cnc
import baselines as bl
import datasets as ds

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures", "paper_figures")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
_STATE = {}


# ---------------------------------------------------------------------------
# _STATE cache (checkpoint support)
# ---------------------------------------------------------------------------
# Several steps populate the module-global `_STATE` (e.g. the resonance affinity
# matrix R and stability curves) which later steps consume.  When a step is
# skipped on resume, its `_STATE` contribution would otherwise be missing and a
# later step would silently recompute an expensive precursor.  We therefore
# mirror `_STATE` to small pickles in `results/` so a skipped step can be
# restored without recomputation.  R is a scipy csr_matrix, so the pickles are
# tiny.  Everything is best-effort: a cache miss simply falls back to recompute.
def _state_cache_path(key: str) -> str:
    return os.path.join(RESULTS, f"_state_{key}.pkl")


def _save_state(key: str) -> None:
    if key not in _STATE:
        return
    try:
        with open(_state_cache_path(key), "wb") as _f:
            pickle.dump(_STATE[key], _f)
    except Exception as _exc:  # pragma: no cover - best effort cache
        _note(f"[warn] could not cache '{key}' state: {_exc}")


def _load_state(key: str) -> bool:
    _p = _state_cache_path(key)
    if os.path.exists(_p):
        try:
            with open(_p, "rb") as _f:
                _STATE[key] = pickle.load(_f)
            return True
        except Exception:
            return False
    return False


_warned = False


def _note(msg):
    print(f"[experiments] {msg}", flush=True)


# ===========================================================================
# Shared low-level selectors (eigengap / silhouette + protocol primitives)
# ===========================================================================
def laplacian_eigenvalues(A, k):
    A = csr_matrix(A)
    d = np.asarray(A.sum(axis=1)).ravel()
    d[d <= 0] = 1e-12
    dinv = diags(1.0 / np.sqrt(d))
    L = (diags(np.ones(A.shape[0])) - dinv @ A @ dinv).tocsr()
    v0 = cnc.deterministic_v0(A.shape[0], seed=0)
    try:
        ev, _ = eigsh(L, k=k, sigma=-1e-6, which="LM", tol=1e-4, maxiter=2000, v0=v0)
    except Exception:
        ev, _ = eigsh(L, k=k, which="SM", tol=1e-3, maxiter=4000, v0=v0)
    return np.sort(ev)


def eigengap_select(A, Kmax):
    ev = laplacian_eigenvalues(A, Kmax + 1)
    gaps = {K: float(ev[K] - ev[K - 1]) for K in range(2, Kmax)}
    return max(gaps, key=gaps.get)


def silhouette_select(A, Kmax, seed=0):
    emb = cnc._spectral_embedding(A, Kmax, seed=seed)
    scores = {}
    for K in range(2, Kmax):
        labs = KMeans(K, n_init=10, random_state=seed).fit_predict(emb[:, :K])
        scores[K] = float(silhouette_score(emb[:, :K], labs))
    return max(scores, key=scores.get)


def stability_curve_reuse(A, Kmax, B, frac, seed, sample_cap=None):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    sample_size = int(frac * n)
    if sample_cap is not None:
        sample_size = min(sample_size, int(sample_cap))
    idxs = [rng.choice(n, size=sample_size, replace=False) for _ in range(B)]
    pi = [{p: q for q, p in enumerate(perm)} for perm in idxs]
    embs = [cnc._spectral_embedding(A[perm][:, perm], Kmax, seed=seed) for perm in idxs]
    stats = {}
    for K in range(2, Kmax + 1):
        labs = [KMeans(K, n_init=10, random_state=seed).fit_predict(e[:, :K]) for e in embs]
        nmis = []
        for i in range(B):
            for j in range(i + 1, B):
                inter = np.intersect1d(idxs[i], idxs[j], assume_unique=True)
                if len(inter) < 4:
                    continue
                li = labs[i][[pi[i][p] for p in inter]]
                lj = labs[j][[pi[j][p] for p in inter]]
                if len(np.unique(li)) < 2 or len(np.unique(lj)) < 2:
                    continue
                nmis.append(normalized_mutual_info_score(li, lj))
        stats[K] = float(np.mean(nmis)) if nmis else -1.0
    return stats


def stability_curve_independent(A, Kmax, B, frac, seed, sample_cap=None):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    sample_size = int(frac * n)
    if sample_cap is not None:
        sample_size = min(sample_size, int(sample_cap))
    idxs = [rng.choice(n, size=sample_size, replace=False) for _ in range(B)]
    pi = [{p: q for q, p in enumerate(perm)} for perm in idxs]
    stats = {}
    for K in range(2, Kmax + 1):
        labs = []
        for perm in idxs:
            e = cnc._spectral_embedding(A[perm][:, perm], K, seed=seed)
            labs.append(KMeans(K, n_init=10, random_state=seed).fit_predict(e))
        nmis = []
        for i in range(B):
            for j in range(i + 1, B):
                inter = np.intersect1d(idxs[i], idxs[j], assume_unique=True)
                if len(inter) < 4:
                    continue
                li = labs[i][[pi[i][p] for p in inter]]
                lj = labs[j][[pi[j][p] for p in inter]]
                if len(np.unique(li)) < 2 or len(np.unique(lj)) < 2:
                    continue
                nmis.append(normalized_mutual_info_score(li, lj))
        stats[K] = float(np.mean(nmis)) if nmis else -1.0
    return stats


def continuity_independent(A, Kmax, seed):
    cont = {}
    for K in range(2, Kmax):
        lK = cnc.spectral_partition(A, K, seed=seed)
        lK1 = cnc.spectral_partition(A, K + 1, seed=seed)
        cont[K] = normalized_mutual_info_score(lK, lK1)
    return cont


def continuity_reuse(A, Kmax, seed):
    emb = cnc._spectral_embedding(A, Kmax, seed=seed)
    parts = {K: KMeans(K, n_init=10, random_state=seed).fit_predict(emb[:, :K])
             for K in range(2, Kmax + 1)}
    return {K: normalized_mutual_info_score(parts[K], parts[K + 1])
            for K in range(2, Kmax)}


def hss_apply(stab, continuity, cont_margin=0.10, cont_floor=0.05):
    Kmax = max(stab)
    best = max(stab, key=stab.get)
    if continuity:
        med = float(np.median(list(continuity.values())))
        bc = continuity.get(best)
        if bc is not None and bc < med - cont_margin:
            alts = [K for K in range(2, Kmax) if continuity[K] >= med - cont_floor]
            if alts:
                best = max(alts, key=lambda K: stab[K])
    return best


def run_protocol(A, y, Kmax, B, frac, seed, stab_mode, cont_mode,
                 sample_cap=None, stab=None, cont=None):
    t0 = time.time()
    if stab is None:
        if stab_mode == "reuse":
            stab = stability_curve_reuse(A, Kmax, B, frac, seed, sample_cap)
        else:
            stab = stability_curve_independent(A, Kmax, B, frac, seed, sample_cap)
    if cont is None:
        cont = (continuity_reuse(A, Kmax, seed) if cont_mode == "reuse"
                else continuity_independent(A, Kmax, seed))
    argmax = max(stab, key=stab.get)
    K_final = hss_apply(stab, cont)
    rec = {"argmax": int(argmax), "K_final": int(K_final),
           "seconds": round(time.time() - t0, 2)}
    if y is not None:
        labels = cnc.spectral_partition(A, K_final, seed=seed)
        rec["ARI"] = round(float(adjusted_rand_score(y, labels)), 4)
        rec["AMI"] = round(float(adjusted_mutual_info_score(y, labels)), 4)
    else:
        rec["ARI"] = None
        rec["AMI"] = None
    return rec


# ===========================================================================
# Experiment 1: main benchmark (Tables 8-11)
# ===========================================================================
def run_benchmark():
    _note("benchmark (synthetic + Gauss20D + Digits)")
    from sklearn.datasets import make_moons, make_circles

    def run_one(X, y, true_K, Kmax=10, B=40):
        rec = {"n_true": int(true_K)}
        out = cnc.cnc_cluster(X, k=20, rank_exponent=2.0, Kmax=Kmax, B=B, seed=0)
        rec["CNC"] = {"K": int(out["K"]), "ARI": float(bl.ari(y, out["labels"])),
                      "AMI": float(bl.ami(y, out["labels"]))}
        km = bl.kmeans_labels(X, true_K)
        rec["KMeans"] = {"K": int(true_K), "ARI": float(bl.ari(y, km)), "AMI": float(bl.ami(y, km))}
        db, dbK, eps = bl.dbscan_auto(X)
        rec["DBSCAN"] = {"K": int(dbK), "ARI": float(bl.ari(y, db)), "AMI": float(bl.ami(y, db))}
        dpc_lab = bl.dpc(X, true_K)
        rec["DPC"] = {"K": int(true_K), "ARI": float(bl.ari(y, dpc_lab)), "AMI": float(bl.ami(y, dpc_lab))}
        ap_lab, apK = bl.affinity_propagation_labels(X)
        rec["AP"] = {"K": int(apK), "ARI": float(bl.ari(y, ap_lab)), "AMI": float(bl.ami(y, ap_lab))}
        snn_lab = bl.snn_partition(X, true_K)
        rec["SNN"] = {"K": int(true_K), "ARI": float(bl.ari(y, snn_lab)), "AMI": float(bl.ami(y, snn_lab))}
        rbf_lab = bl.rbf_partition(X, true_K)
        rec["RBFSpectral"] = {"K": int(true_K), "ARI": float(bl.ari(y, rbf_lab)),
                              "AMI": float(bl.ami(y, rbf_lab))}
        try:
            hdb_lab, hdbK = bl.hdbscan_labels(X)
            rec["HDBSCAN"] = {"K": int(hdbK), "ARI": float(bl.ari(y, hdb_lab)),
                              "AMI": float(bl.ami(y, hdb_lab))}
        except Exception:
            pass
        return rec

    results = {"synthetic": {}}
    for name, (X, y) in ds.synthetic_benchmarks().items():
        results["synthetic"][name] = run_one(X, y, len(np.unique(y)))
        r = results["synthetic"][name]["CNC"]
        _note(f"{name:12s} CNC K={r['K']} ARI={r['ARI']:.2f}")
    Xg, yg = ds.make_gauss20d()
    results["Gauss20D"] = run_one(Xg, yg, 5)
    Xd, yd = ds.make_digits()
    results["Digits"] = run_one(Xd, yd, 10, Kmax=15, B=20)
    _dump("benchmark_results.json", results)
    return results


# ===========================================================================
# Experiment 2: k-sensitivity (Table 5)
# ===========================================================================
def run_ksensitivity():
    _note("k-sensitivity")
    ksens = {}
    for name, (X, y) in ds.synthetic_benchmarks().items():
        trueK = len(np.unique(y))
        row = {}
        for k in [10, 15, 20, 25, 30, 40]:
            R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
            lab = cnc.spectral_partition(R, trueK, seed=0)
            row[str(k)] = float(bl.ari(y, lab))
        ksens[name] = row
    _dump("k_sensitivity.json", ksens)
    return ksens


# ===========================================================================
# Experiment 3: rank-exponent ablation (Table 6)
# ===========================================================================
def run_rank_ablation():
    _note("rank-exponent ablation")
    abl = {}
    for name, (X, y) in ds.synthetic_benchmarks().items():
        trueK = len(np.unique(y))
        row = {}
        for p in [0, 1, 2, 3, 4]:
            R, _, _ = cnc.resonance_affinity(X, k=20, rank_exponent=float(p))
            lab = cnc.spectral_partition(R, trueK, seed=0)
            row[str(p)] = float(bl.ari(y, lab))
        abl[name] = row
    _dump("rank_ablation.json", abl)
    return abl


# ===========================================================================
# Experiment 4: multi-seed robustness (Table 7)
# ===========================================================================
def run_multiseed():
    _note("multi-seed robustness")
    seeds = [0, 1, 2, 3, 4]
    records = []
    for s in seeds:
        rec = {}
        for name in ds.synthetic_benchmarks():
            X, y = ds.build_family(name, s)
            trueK = len(np.unique(y))
            out = cnc.cnc_cluster(X, k=20, Kmax=10, B=10, seed=s)
            rec[name] = {
                "CNC_auto": {"K": int(out["K"]), "ARI": float(bl.ari(y, out["labels"]))},
                "CNC_oracleK": {"K": int(trueK),
                                "ARI": float(bl.ari(y, cnc.spectral_partition(
                                    cnc.resonance_affinity(X, k=20)[0], trueK, seed=s)))},
                "SNN_oracleK": {"K": int(trueK),
                                "ARI": float(bl.ari(y, bl.snn_partition(X, trueK, seed=s)))},
                "KMeans_oracleK": {"K": int(trueK),
                                   "ARI": float(bl.ari(y, bl.kmeans_labels(X, trueK, seed=s)))},
            }
        records.append(rec)
    summary = {}
    for name in ds.synthetic_benchmarks():
        summary[name] = {}
        for key in ["CNC_auto", "CNC_oracleK", "SNN_oracleK", "KMeans_oracleK"]:
            vals = [records[s][name][key]["ARI"] for s in range(len(seeds))]
            Ks = [records[s][name][key]["K"] for s in range(len(seeds))]
            summary[name][key] = {"ARI_mean": float(np.mean(vals)), "ARI_std": float(np.std(vals)),
                                  "K_values": [int(k) for k in Ks]}
    _dump("multiseed_benchmark.json", {"seeds": seeds, "summary": summary})
    return summary


# ===========================================================================
# Experiment 5: runtime scaling (Table 12)
# ===========================================================================
def run_runtime():
    _note("runtime scaling")
    records_rt = []
    for n in [500, 1000, 2000, 4000]:
        X, _ = make_blobs(n_samples=n, centers=4, random_state=0)
        t0 = time.time(); R, _, edges = cnc.resonance_affinity(X, k=20)
        tg = time.time() - t0
        t0 = time.time(); cnc.spectral_partition(R, 4, seed=0)
        ts = time.time() - t0
        t0 = time.time(); cnc.stability_select_K(R, Kmax=6, B=5, seed=0)
        tstab = time.time() - t0
        records_rt.append({"N": n, "graph_seconds": round(tg, 3), "spectral_seconds": round(ts, 3),
                           "stability_seconds": round(tstab, 3), "nnz": int(R.nnz)})
    _dump("runtime_scaling.json", {"records": records_rt})
    return records_rt


# ===========================================================================
# Experiment 6: hyperspectral case (Table 13)
# ===========================================================================
def _spatial_continuity(labels, H=200, W=200):
    img = labels.reshape(H, W)
    interior = np.zeros_like(img, dtype=bool)
    interior[1:-1, 1:-1] = True
    same = total = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(img, di, 0), dj, 1)
            same += int(((img == shifted) & interior).sum())
            total += int(interior.sum())
    return float(same / total)


def _spectral_separation(X, labels):
    classes = sorted(np.unique(labels).tolist())
    means = np.vstack([X[labels == c].mean(axis=0) for c in classes])
    if len(classes) < 2:
        return 0.0, 0.0
    distances = np.sqrt(((means[:, None, :] - means[None, :, :]) ** 2).sum(axis=2))
    values = distances[np.triu_indices(len(classes), 1)]
    return float(values.mean()), float(values.std())


def _metric_record(X, labels):
    sep_mean, sep_std = _spectral_separation(X, labels)
    return {
        "spatial_coherence": round(_spatial_continuity(labels), 4),
        "cluster_sizes": {str(int(c)): int(np.sum(labels == c)) for c in sorted(np.unique(labels))},
        "n_clusters": int(len(np.unique(labels))),
        "inter_cluster_sep_mean": round(sep_mean, 2),
        "inter_cluster_sep_std": round(sep_std, 2),
    }


def run_hsi():
    _note("hyperspectral (Asian Games Village)")
    Xh, _ = ds.load_ayc()
    Xh_bands = ds.load_ayc_bands()
    Rh, _, _ = cnc.resonance_affinity(Xh, k=20)
    stats = cnc.stability_curve_stats(Rh, Kmax=24, B=20, frac=0.8,
                                      sample_cap=4000, seed=0)
    stability = {int(k): float(v["mean"]) for k, v in stats.items()}
    K_h = max(stability, key=stability.get)

    cnc_labels = {}
    kmeans_labels = {}
    metrics = {}
    for Kt in sorted(set([4, 6, int(K_h)])):
        cnc_labels[Kt] = cnc.spectral_partition(Rh, Kt, seed=0)
        kmeans_labels[Kt] = bl.kmeans_labels(Xh, Kt, seed=0)
        metrics[f"CNC K={Kt}"] = _metric_record(Xh_bands, cnc_labels[Kt])
        metrics[f"KMeans K={Kt}"] = _metric_record(Xh_bands, kmeans_labels[Kt])

    ayc_metrics = {
        "K": int(K_h),
        "stability": {str(k): float(v) for k, v in stability.items()},
        "metrics": metrics,
        "protocol": {"n_pixels": int(Xh.shape[0]), "n_bands": int(Xh_bands.shape[1]),
                     "pca_components": int(Xh.shape[1]), "k_neighbors": 20,
                     "Kmax": 24, "B": 20, "frac": 0.8, "sample_cap": 4000,
                     "sample_size": 4000, "seed": 0},
    }
    _dump("ayc_paper_metrics.json", ayc_metrics)
    _STATE["ayc"] = {
        "X": Xh,
        "X_bands": Xh_bands,
        "R": Rh,
        "stats": stats,
        "stability": stability,
        "labels": cnc_labels,
        "kmeans_labels": kmeans_labels,
    }
    _save_state("ayc")
    for diagnostic_Kmax in (12, 16):
        diag = cnc.stability_curve_stats(
            Rh, Kmax=diagnostic_Kmax, B=20, frac=0.8,
            sample_cap=4000, seed=0
        )
        _dump(
            f"ayc_kstab_ext{diagnostic_Kmax if diagnostic_Kmax != 16 else ''}.json",
            {
                "Kmax": diagnostic_Kmax,
                "B": 20,
                "frac": 0.8,
                "sample_cap": 4000,
                "seed": 0,
                "stability": {str(k): float(v["mean"]) for k, v in diag.items()},
            },
        )
    _dump(
        "ayc_kstab_ext24.json",
        {
            "Kmax": 24,
            "B": 20,
            "frac": 0.8,
            "sample_cap": 4000,
            "seed": 0,
            "selected_K": int(K_h),
            "stability": {str(k): round(float(v), 4) for k, v in stability.items()},
        },
    )
    np.savez_compressed(os.path.join(RESULTS, "ayc_labels.npz"),
                        **{f"cnc_{k}": v for k, v in cnc_labels.items()},
                        **{f"kmeans_{k}": v for k, v in kmeans_labels.items()})
    _note(f"K={K_h}")
    return ayc_metrics, cnc_labels, kmeans_labels, Xh_bands


# ===========================================================================
# Experiment 7: 1-SE stability stats for AYC
# ===========================================================================
def run_ayc_1se():
    _note("AYC 1-SE stability stats")
    if "ayc" not in _STATE and not _load_state("ayc"):
        run_hsi()
    stats = _STATE["ayc"]["stats"]
    out = {"Kmax": 24, "B": 20, "frac": 0.8, "sample_cap": 4000, "seed": 0,
           "stats": {str(k): {"mean": round(v["mean"], 4), "se": round(v["se"], 4),
                              "n_pairs": v["n_pairs"]} for k, v in stats.items()},
           "selected_1se": int(cnc.one_se_select_K(stats))}
    _dump("ayc_stability_stats.json", out)
    return out


# ===========================================================================
# Experiment 8: Indian Pines labeled benchmark
# ===========================================================================
def run_indian_pines():
    _note("Indian Pines labeled benchmark")
    Xh, y = ds.load_indian_pines()
    K_true = int(len(np.unique(y)))
    res = {"dataset": "Indian Pines", "n_pixels": int(Xh.shape[0]),
           "n_bands": None, "pca_components": int(Xh.shape[1]),
           "n_classes": K_true, "k_neighbors": 20,
           "cnc": {"Kmax": 20, "B": 20, "frac": 0.8, "sample_cap": 4000, "seed": 0},
           "methods": {}, "notes": "DPC and AP omitted (O(N^2) cost at N=10,249)."}

    out = cnc.cnc_cluster(Xh, k=20, rank_exponent=2.0, Kmax=20, B=20,
                          sample_cap=4000, frac=0.8, seed=0)
    res["methods"]["CNC"] = {"K": int(out["K"]), "ARI": float(bl.ari(y, out["labels"])),
                             "AMI": float(bl.ami(y, out["labels"]))}
    km = bl.kmeans_labels(Xh, K_true)
    res["methods"]["KMeans"] = {"K": K_true, "ARI": float(bl.ari(y, km)), "AMI": float(bl.ami(y, km))}
    snn = bl.snn_partition(Xh, K_true, k=20)
    res["methods"]["SNN"] = {"K": K_true, "ARI": float(bl.ari(y, snn)), "AMI": float(bl.ami(y, snn))}
    rbf = bl.rbf_partition_knn(Xh, K_true, k=20, seed=0)
    res["methods"]["RBFSpectral"] = {"K": K_true, "ARI": float(bl.ari(y, rbf)), "AMI": float(bl.ami(y, rbf)),
                                     "variant": "20-NN truncated Gaussian (dense kernel memory-prohibitive)"}
    db, dbK, eps = bl.dbscan_auto(Xh)
    res["methods"]["DBSCAN"] = {"K": int(dbK), "ARI": float(bl.ari(y, db)), "AMI": float(bl.ami(y, db))}
    try:
        hl, hK = bl.hdbscan_labels(Xh)
        res["methods"]["HDBSCAN"] = {"K": int(hK), "ARI": float(bl.ari(y, hl)),
                                     "AMI": float(bl.ami(y, hl))}
    except ModuleNotFoundError as _exc:
        _note(f"[skip] HDBSCAN baseline unavailable ({_exc}); install with "
              f"`pip install hdbscan==0.8.44` and re-run this step to enable it. "
              f"Other baselines are retained.")
        res["methods"]["HDBSCAN"] = {"skipped": True,
                                     "reason": "hdbscan not installed",
                                     "install": "pip install hdbscan==0.8.44"}

    Rh, _, _ = cnc.resonance_affinity(Xh, k=20)
    stats = cnc.stability_curve_stats(Rh, Kmax=20, B=20, frac=0.8, sample_cap=4000, seed=0)
    res["stability_curve"] = {str(k): {"mean": round(v["mean"], 4), "se": round(v["se"], 4),
                                       "n_pairs": v["n_pairs"]} for k, v in stats.items()}
    res["selected_K_1se"] = int(cnc.one_se_select_K(stats))
    _STATE["indian_pines"] = {"X": Xh, "y": y, "R": Rh}
    _STATE["indian_pines"]["stats"] = stats
    _save_state("indian_pines")
    _dump("indian_pines_benchmark.json", res)
    return res


# ===========================================================================
# Experiment 9: k-selection baselines (argmax / 1-SE / eigengap / silhouette / HSS)
# ===========================================================================
def run_kselection():
    _note("k-selection baselines (Table: selector comparison)")
    out = {"seed": 0, "datasets": {}}
    datasets = [
        ("Moons", *ds.synthetic_benchmarks()["Moons"], 20, 10, 40),
        ("Circles", *ds.synthetic_benchmarks()["Circles"], 20, 10, 40),
        ("Blobs", *ds.synthetic_benchmarks()["Blobs"], 20, 10, 40),
        ("Aniso", *ds.synthetic_benchmarks()["Aniso"], 20, 10, 40),
        ("VariedBlobs", *ds.synthetic_benchmarks()["VariedBlobs"], 20, 10, 40),
        ("Gauss20D", *ds.make_gauss20d(), 20, 10, 40),
        ("Digits", *ds.make_digits(), 20, 15, 20),
    ]

    def ari_of(A, y, K, seed=0):
        lab = cnc.spectral_partition(A, K, seed=seed)
        return round(float(adjusted_rand_score(y, lab)), 4)

    for name, X, y, k, Kmax, B in datasets:
        R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
        stats = cnc.stability_curve_stats(R, Kmax=Kmax, B=B, frac=0.8, seed=0)
        stab = {int(kk): float(v["mean"]) for kk, v in stats.items()}
        continuity = continuity_independent(R, Kmax, seed=0)
        K_hss = hss_apply(stab, continuity)
        K_argmax = max(stab, key=stab.get)
        K_1se = cnc.one_se_select_K(stats)
        K_eig = eigengap_select(R, Kmax)
        K_sil = silhouette_select(R, Kmax)
        rec = {
            "true_K": int(len(np.unique(y))),
            "argmax": {"K": K_argmax, "ARI": ari_of(R, y, K_argmax)},
            "1-SE": {"K": K_1se, "ARI": ari_of(R, y, K_1se)},
            "eigengap": {"K": K_eig, "ARI": ari_of(R, y, K_eig)},
            "silhouette": {"K": K_sil, "ARI": ari_of(R, y, K_sil)},
            "HSS": {"K": K_hss, "ARI": ari_of(R, y, K_hss)},
        }
        out["datasets"][name] = rec
        _note(f"{name:12s} HSS K={K_hss}/{rec['HSS']['ARI']} | eigengap K={K_eig}/{rec['eigengap']['ARI']}")

    if "indian_pines" not in _STATE and not _load_state("indian_pines"):
        run_indian_pines()
    Xh = _STATE["indian_pines"]["X"]
    yh = _STATE["indian_pines"]["y"]
    Rh = _STATE["indian_pines"]["R"]
    stats_ip = _STATE["indian_pines"]["stats"]
    stab_ip = {int(kk): float(v["mean"]) for kk, v in stats_ip.items()}
    K_ip_argmax = max(stab_ip, key=stab_ip.get)
    K_ip_1se = cnc.one_se_select_K(stats_ip)
    continuity_ip = continuity_independent(Rh, 20, seed=0)
    K_ip_hss = hss_apply(stab_ip, continuity_ip)
    K_ip_eig = eigengap_select(Rh, 20)
    K_ip_sil = silhouette_select(Rh, 20)
    out["datasets"]["IndianPines"] = {
        "true_K": 16,
        "argmax": {"K": K_ip_argmax, "ARI": ari_of(Rh, yh, K_ip_argmax)},
        "1-SE": {"K": K_ip_1se, "ARI": ari_of(Rh, yh, K_ip_1se)},
        "eigengap": {"K": K_ip_eig, "ARI": ari_of(Rh, yh, K_ip_eig)},
        "silhouette": {"K": K_ip_sil, "ARI": ari_of(Rh, yh, K_ip_sil)},
        "HSS": {"K": K_ip_hss, "ARI": ari_of(Rh, yh, K_ip_hss)},
    }
    _dump("kselection_baselines.json", out)
    return out


# ===========================================================================
# Experiment 10: embedding-protocol ablation (A / B / C)
# ===========================================================================
def run_protocol_ablation():
    _note("embedding-protocol ablation (A/B/C)")
    out = {"seed": 0, "synthetic": {}, "hyperspectral": {}}
    synthetic = [
        ("Moons", *ds.synthetic_benchmarks()["Moons"], 20, 10, 40),
        ("Circles", *ds.synthetic_benchmarks()["Circles"], 20, 10, 40),
        ("Blobs", *ds.synthetic_benchmarks()["Blobs"], 20, 10, 40),
        ("Aniso", *ds.synthetic_benchmarks()["Aniso"], 20, 10, 40),
        ("VariedBlobs", *ds.synthetic_benchmarks()["VariedBlobs"], 20, 10, 40),
        ("Gauss20D", *ds.make_gauss20d(), 20, 10, 40),
        ("Digits", *ds.make_digits(), 20, 15, 20),
    ]
    for name, X, y, k, Kmax, B in synthetic:
        R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
        stab_reuse = stability_curve_reuse(R, Kmax, B, 0.8, 0)
        stab_indep = stability_curve_independent(R, Kmax, B, 0.8, 0)
        cont_indep = continuity_independent(R, Kmax, seed=0)
        cont_reuse = continuity_reuse(R, Kmax, seed=0)
        out["synthetic"][name] = {
            "A_reuse_indep": run_protocol(
                R, y, Kmax, B, 0.8, 0, "reuse", "indep",
                stab=stab_reuse, cont=cont_indep
            ),
            "B_indep_indep": run_protocol(
                R, y, Kmax, B, 0.8, 0, "indep", "indep",
                stab=stab_indep, cont=cont_indep
            ),
            "C_reuse_reuse": run_protocol(
                R, y, Kmax, B, 0.8, 0, "reuse", "reuse",
                stab=stab_reuse, cont=cont_reuse
            ),
        }
        row = out["synthetic"][name]
        _note(f"{name:12s} A K={row['A_reuse_indep']['K_final']} B K={row['B_indep_indep']['K_final']} "
              f"C K={row['C_reuse_reuse']['K_final']}")

    for name, X, y, k, Kmax, B, cap in [
        ("IndianPines", *ds.load_indian_pines(), 20, 20, 20, 4000),
        ("AsianGamesVillage", *ds.load_ayc(), 20, 24, 20, 4000),
    ]:
        if name == "AsianGamesVillage":
            if "ayc" not in _STATE and not _load_state("ayc"):
                run_hsi()
            R = _STATE["ayc"]["R"]
            stab_prod = _STATE["ayc"]["stability"]
            argmax_ayc = max(stab_prod, key=stab_prod.get)
            cont_reuse = continuity_reuse(R, Kmax, seed=0)
            recA = {"argmax": int(argmax_ayc), "K_final": int(argmax_ayc),
                    "ARI": None, "AMI": None, "seconds": None,
                    "note": "A inherited from production run (interior maximum, HSS unchanged)"}
            recC = run_protocol(
                R, y, Kmax, B, 0.8, 0, "reuse", "reuse",
                sample_cap=cap, stab=stab_prod, cont=cont_reuse
            )
        else:
            if "indian_pines" not in _STATE and not _load_state("indian_pines"):
                run_indian_pines()
            ip = _STATE["indian_pines"]
            R = ip["R"]
            stab_ip = {
                int(k): float(v["mean"]) for k, v in ip["stats"].items()
            }
            cont_indep = continuity_independent(R, Kmax, seed=0)
            cont_reuse = continuity_reuse(R, Kmax, seed=0)
            recA = run_protocol(
                R, y, Kmax, B, 0.8, 0, "reuse", "indep",
                sample_cap=cap, stab=stab_ip, cont=cont_indep
            )
            recC = run_protocol(
                R, y, Kmax, B, 0.8, 0, "reuse", "reuse",
                sample_cap=cap, stab=stab_ip, cont=cont_reuse
            )
        out["hyperspectral"][name] = {"A_reuse_indep": recA, "C_reuse_reuse": recC}
        out["hyperspectral"][name]["note"] = (
            "reduced two-way ablation (A vs C, the continuity-protocol contrast); "
            "protocol B (fully independent stability) omitted as prohibitive at this scale"
        )
        _note(f"{name:18s} A K={recA['K_final']} C K={recC['K_final']}")

    _dump("embedding_protocol_ablation.json", out)
    return out


# ===========================================================================
# Experiment 11: multi-seed protocol frequency (A vs C)
# ===========================================================================
def run_protocol_multiseed():
    _note("multi-seed protocol frequency (A vs C)")
    out = {"note": ("multi-seed selection frequency of protocols A vs C on the "
                    "two discriminating datasets; protocol B omitted (cost)"),
           "seeds": [0, 1, 2, 3, 4]}

    def run(A, y, Kmax, B, seed, cont_mode):
        stab = stability_curve_reuse(A, Kmax, B, 0.8, seed)
        cont = (continuity_reuse(A, Kmax, seed) if cont_mode == "reuse"
                else continuity_independent(A, Kmax, seed))
        K_final = hss_apply(stab, cont)
        lab = cnc.spectral_partition(A, K_final, seed=seed)
        return int(K_final), round(float(adjusted_rand_score(y, lab)), 4)

    for name, loader, k, Kmax, B in [
        ("Blobs", lambda s: ds.build_family("Blobs", s), 20, 10, 40),
        ("Digits", lambda s: ds.make_digits(), 20, 15, 20),
    ]:
        rec = {"A": {"K": [], "ARI": []}, "C": {"K": [], "ARI": []}}
        for s in out["seeds"]:
            X, y = loader(s)
            R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
            Ka, aa = run(R, y, Kmax, B, s, "indep")
            Kc, ac = run(R, y, Kmax, B, s, "reuse")
            rec["A"]["K"].append(Ka); rec["A"]["ARI"].append(aa)
            rec["C"]["K"].append(Kc); rec["C"]["ARI"].append(ac)
        for proto in ("A", "C"):
            rec[proto]["K_freq"] = {str(k): rec[proto]["K"].count(k)
                                    for k in sorted(set(rec[proto]["K"]))}
            rec[proto]["ARI_mean"] = round(float(np.mean(rec[proto]["ARI"])), 4)
            rec[proto]["ARI_std"] = round(float(np.std(rec[proto]["ARI"])), 4)
        out[name] = rec
        _note(f"{name}: A freq={rec['A']['K_freq']} C freq={rec['C']['K_freq']}")

    _dump("embedding_protocol_multiseed.json", out)
    return out


# ===========================================================================
# Experiment 12: reviewer-response checks (boundary fix + sensitivity + 1-SE)
# ===========================================================================
def run_reviewer_checks():
    _note("reviewer-response checks (boundary fix + sensitivity grid + 1-SE)")
    out = {"boundary_fix": {}, "sensitivity": {}, "one_se": {}}
    margin_grid = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    floor_grid = [0.02, 0.05, 0.08, 0.10]
    SEED = 0

    benchmarks = [
        ("Moons", *ds.synthetic_benchmarks()["Moons"], 20, 10, 40),
        ("Circles", *ds.synthetic_benchmarks()["Circles"], 20, 10, 40),
        ("Blobs", *ds.synthetic_benchmarks()["Blobs"], 20, 10, 40),
        ("Digits", *ds.make_digits(), 20, 15, 20),
    ]

    for name, X, y, k, Kmax, B in benchmarks:
        R, _, _ = cnc.resonance_affinity(X, k=k, rank_exponent=2.0)
        stats = cnc.stability_curve_stats(R, Kmax=Kmax, B=B, frac=0.8, seed=SEED)
        stab = {int(K): float(v["mean"]) for K, v in stats.items()}
        continuity = continuity_independent(R, Kmax, seed=SEED)
        ari_per_K = {}
        for K in range(2, Kmax + 1):
            ari_per_K[K] = round(float(adjusted_rand_score(
                y, cnc.spectral_partition(R, K, seed=SEED))), 4)

        K_fix = hss_apply(stab, continuity, 0.10, 0.05)
        out["boundary_fix"][name] = {
            "K": K_fix, "ARI": ari_per_K[K_fix],
            "stab": round(stab[K_fix], 4),
            "cont": round(continuity[K_fix], 4) if K_fix in continuity else None,
            "seed": SEED,
        }
        out["sensitivity"][name] = {
            "ari_per_K": ari_per_K,
            "stab": {int(k): round(v, 4) for k, v in stab.items()},
            "continuity": {int(k): round(v, 4) for k, v in continuity.items()},
            "protocol": {"seed": SEED, "Kmax": Kmax, "B": B, "frac": 0.8},
            "grid": {},
        }
        for m in margin_grid:
            for f in floor_grid:
                Kg = hss_apply(stab, continuity, m, f)
                out["sensitivity"][name]["grid"][f"m{m}_f{f}"] = {
                    "tau_margin": m, "tau_floor": f, "K": Kg, "ARI": ari_per_K[Kg],
                    "stab_of_K": round(stab[Kg], 4),
                    "continuity_of_K": round(continuity[Kg], 4) if Kg in continuity else None,
                    "seed": SEED, "Kmax": Kmax, "B": B, "frac": 0.8,
                }
        K_1se = cnc.one_se_select_K(stats)
        out["one_se"][name] = {"K": K_1se, "ARI": ari_per_K[K_1se], "seed": SEED}

    _dump("reviewer_response_checks.json", out)
    return out


# ===========================================================================
# Figure rendering (delegates to plotting.py)
# ===========================================================================
def render_all_figures(benchmark_results=None):
    _note("rendering figures")
    import plotting as P
    if benchmark_results is None:
        benchmark_results = json.load(open(os.path.join(RESULTS, "benchmark_results.json")))
    abl = json.load(open(os.path.join(RESULTS, "rank_ablation.json")))
    summary = json.load(open(os.path.join(RESULTS, "multiseed_benchmark.json")))["summary"]
    records_rt = json.load(open(os.path.join(RESULTS, "runtime_scaling.json")))["records"]
    ayc_metrics = json.load(open(os.path.join(RESULTS, "ayc_paper_metrics.json")))
    labels_npz = np.load(os.path.join(RESULTS, "ayc_labels.npz"))
    Xh_bands = ds.load_ayc_bands()
    cnc_labels = {int(k.replace("cnc_", "")): labels_npz[k] for k in labels_npz.files if k.startswith("cnc_")}
    kmeans_labels = {int(k.replace("kmeans_", "")): labels_npz[k] for k in labels_npz.files if k.startswith("kmeans_")}

    P.render_benchmark_figures(benchmark_results, ROOT)
    P.render_rank_ablation([0, 1, 2, 3, 4], list(ds.synthetic_benchmarks().keys()), abl, ROOT)
    P.render_affinity(ds.synthetic_benchmarks(), cnc, ROOT)
    P.render_multiseed(summary, list(ds.synthetic_benchmarks().keys()), ROOT)
    P.render_runtime(records_rt, ROOT)
    P.render_idea(ROOT)
    _panels = []
    for nm in ["Moons", "Circles", "Blobs", "Aniso"]:
        Xp, yp = ds.synthetic_benchmarks()[nm]
        out = cnc.cnc_cluster(Xp, k=20, Kmax=10, B=40, seed=0)
        _panels.append({"name": nm, "X": Xp, "labels": out["labels"]})
    P.render_synthetic(_panels, ROOT)
    P.render_hsi_figures(Xh_bands, cnc_labels, kmeans_labels,
                         ayc_metrics["stability"], ROOT)
    _note("all figures rendered")


# ===========================================================================
# Top-level driver: run everything in the manuscript order
# ===========================================================================
def clean_outputs():
    """Remove generated results and figures while preserving the package layout."""
    for directory in (RESULTS, FIGS):
        os.makedirs(directory, exist_ok=True)
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                os.remove(path)
    _STATE.clear()


def verify_outputs():
    """Assert the manuscript's load-bearing results after a full reproduction."""
    benchmark = json.load(open(os.path.join(RESULTS, "benchmark_results.json")))
    ksel = json.load(open(os.path.join(RESULTS, "kselection_baselines.json")))
    indian = json.load(open(os.path.join(RESULTS, "indian_pines_benchmark.json")))
    ayc = json.load(open(os.path.join(RESULTS, "ayc_paper_metrics.json")))
    checks = {
        "Blobs_K": benchmark["synthetic"]["Blobs"]["CNC"]["K"] == 4,
        "Blobs_ARI": abs(benchmark["synthetic"]["Blobs"]["CNC"]["ARI"] - 0.827) < 0.01,
        "Digits_K": benchmark["Digits"]["CNC"]["K"] == 9,
        "Digits_ARI": abs(benchmark["Digits"]["CNC"]["ARI"] - 0.742) < 0.01,
        "IndianPines_K": indian["methods"]["CNC"]["K"] == 14,
        "AYC_K": ayc["K"] == 20,
        "Selector_Blobs": ksel["datasets"]["Blobs"]["HSS"]["K"] == 4,
        "Selector_Digits": ksel["datasets"]["Digits"]["HSS"]["K"] == 9,
    }
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        raise AssertionError(f"reproduction checks failed: {missing}")
    return checks


def run_all(include_extended: bool = True, resume: bool = False):
    """Run every manuscript experiment (data-generation layer).

    The extended protocol studies are part of the manuscript package, so they
    are included by default.  ``include_extended=False`` is retained for the
    short core-only path used by the package runner's optional flag.

    ``resume=True`` skips any step whose output JSON already exists, so an
    interrupted run can continue without repeating finished experiments.
    Shared precursor state (e.g. the resonance affinity matrix R) is restored
    from ``results/_state_*.pkl`` caches so later steps do not recompute it.
    """
    t_all = time.time()
    steps = [
        ("benchmark",     run_benchmark,     ["benchmark_results.json"]),
        ("k-sensitivity", run_ksensitivity,  ["k_sensitivity.json"]),
        ("rank-ablation", run_rank_ablation, ["rank_ablation.json"]),
        ("multiseed",     run_multiseed,     ["multiseed_benchmark.json"]),
        ("runtime",       run_runtime,       ["runtime_scaling.json"]),
        ("hsi",           run_hsi,           ["ayc_paper_metrics.json", "ayc_kstab_ext12.json",
                                              "ayc_kstab_ext.json", "ayc_kstab_ext24.json"]),
        ("ayc-1se",       run_ayc_1se,       ["ayc_stability_stats.json"]),
        ("indian-pines",  run_indian_pines,  ["indian_pines_benchmark.json"]),
        ("kselection",    run_kselection,    ["kselection_baselines.json"]),
    ]
    if include_extended:
        steps += [
            ("protocol-ablation",  run_protocol_ablation,  ["embedding_protocol_ablation.json"]),
            ("protocol-multiseed", run_protocol_multiseed, ["embedding_protocol_multiseed.json"]),
        ]
    steps.append(("reviewer-checks", run_reviewer_checks, ["reviewer_response_checks.json"]))

    for name, fn, outs in steps:
        if resume and outs and all(os.path.exists(os.path.join(RESULTS, o)) for o in outs):
            _note(f"[skip] {name} (outputs already exist)")
            continue
        _note(f"[run] {name}")
        fn()
    # NOTE: figure rendering is intentionally NOT called here. It is a separate
    # layer (visualization/) driven by run_all.py / the Jupyter notebook, so that
    # data generation and figure generation stay decoupled.
    report = verify_outputs()
    _note(f"TOTAL {time.time() - t_all:.0f}s")
    return report


def _dump(rel_path, obj):
    path = os.path.join(RESULTS, rel_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    _note(f"saved -> {os.path.relpath(path, ROOT)}")
