# -*- coding: utf-8 -*-
"""
CNC spectral (+ spectral-spatial) segmentation on the Asian Games Village
hyperspectral cube (200x200 px, 64 bands).

This is a self-contained, reproducible driver for the real-data experiment
reported in the paper. It:
  * loads the preprocessed cube (data/ayc_X.npy),
  * builds the resonance affinity graph and performs stability-selected
    spectral clustering (CNC),
  * compares against K-means at the same K,
  * regenerates every real-data figure referenced by paper/main.tex
    (figs/ayc_kstab.png, figs/ayc_compare_clean.png, figs/ayc_spectra_paper.pdf),
  * writes results/ayc_labels.npz and results/ayc_paper_metrics.json, and
  * prints the internal-evaluation table that backs Table 4 of the paper.

All output paths are resolved relative to this file, so the script can be
run from anywhere.  Output figures are written both into paper/figs (for the
LaTeX build) and a supplementary copy into results/.
"""
import os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

# ---------------- reproducible, location-independent paths ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
PAPER_FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PAPER_FIGS, exist_ok=True)

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- load
X = np.load(os.path.join(DATA_DIR, "ayc_X.npy"))          # (40000, 64) raw radiance
H = W = 200
keep = np.arange(54)                                       # drop noisy bands 55-64
Xk = X[:, keep].astype(np.float64)
X_clean = Xk.copy()                                        # raw radiance, 54 bands (for spectra/sep)
mu, sd = Xk.mean(0), Xk.std(0); sd[sd < 1e-6] = 1.0
Z = (Xk - mu) / sd
pca = PCA(n_components=0.99, random_state=0)
P = pca.fit_transform(Z)
n_comp = P.shape[1]
log("preproc: kept bands=54, PCA comps=", n_comp,
    "var=", f"{pca.explained_variance_ratio_.sum():.4f}")


# ---------------------------------------------------------------- CNC core
def resonance_affinity(F, k=20):
    n = F.shape[0]
    _, idx = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(F).kneighbors(F)
    idx = idx[:, 1:]
    ranks = np.arange(k)
    w = (k - ranks).astype(np.float64)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    data = np.tile(w, n)
    N = csr_matrix((data, (rows, cols)), shape=(n, n))
    M = N @ N.T
    nrm = np.sqrt(np.asarray(M.diagonal()).ravel()) + 1e-12
    di = diags(1.0 / nrm)
    A = di @ M @ di
    A.setdiag(0.0); A.eliminate_zeros(); A = A.tocsr()
    return A


def spectral_partition(A, K, seed=0):
    n = A.shape[0]
    d = np.asarray(A.sum(1)).ravel(); d[d <= 0] = 1e-12
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


def stability_select_K(F, Kmax=12, B=20, n_sub=4000, k=15, seed=1):
    n = F.shape[0]
    rng = np.random.default_rng(seed)
    embs, perms = [], []
    for _ in range(B):
        perm = rng.choice(n, size=min(n_sub, n), replace=False)
        A = resonance_affinity(F[perm], k=k)
        d = np.asarray(A.sum(1)).ravel(); d[d <= 0] = 1e-12
        dinv = diags(1.0 / np.sqrt(d))
        L = (diags(np.ones(A.shape[0])) - dinv @ A @ dinv).tocsr()
        try:
            _, V = eigsh(L, k=Kmax + 1, sigma=0.0, which="LM", tol=1e-4, maxiter=2000)
        except Exception:
            _, V = eigsh(L, k=Kmax + 1, which="SM", tol=1e-3, maxiter=4000)
        embs.append(V); perms.append(perm)
    stab = {}
    for K in range(2, Kmax + 1):
        labs = []
        for b in range(B):
            V = embs[b][:, 1:K + 1]
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            labs.append(KMeans(K, n_init=10, random_state=0).fit_predict(V))
        nmis = []
        for i in range(B - 1):
            inter = np.intersect1d(perms[i], perms[i + 1], assume_unique=True)
            if len(inter) < 4: continue
            pi = {p: q for q, p in enumerate(perms[i])}
            pj = {p: q for q, p in enumerate(perms[i + 1])}
            li, lj = labs[i][[pi[p] for p in inter]], labs[i + 1][[pj[p] for p in inter]]
            if len(np.unique(li)) < 2 or len(np.unique(lj)) < 2: continue
            nmis.append(normalized_mutual_info_score(li, lj))
        stab[K] = float(np.mean(nmis)) if nmis else -1.0
    return max(stab, key=stab.get), stab


def spatial_gate(A, sigma=12.0):
    """multiply each affinity edge by a Gaussian spatial gate."""
    n = A.shape[0]
    rows, cols = A.nonzero()
    r1, c1 = rows // W, rows % W
    r2, c2 = cols // W, cols % W
    d2 = (r1 - r2) ** 2 + (c1 - c2) ** 2
    gate = np.exp(-d2 / (2.0 * sigma * sigma))
    Asp = csr_matrix((A.data * gate, (rows, cols)), shape=(n, n))
    Asp.setdiag(0.0); Asp.eliminate_zeros(); Asp = Asp.tocsr()
    return Asp


def spatial_homogeneity(lab):
    img = lab.reshape(H, W)
    same = 0; total = 0
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            nb = img[i-1:i+2, j-1:j+2]
            same += (nb == img[i, j]).sum() - 1
            total += 8
    return same / total


# ---------------------------------------------------------------- run CNC
log("building pure spectral resonance affinity...")
A = resonance_affinity(P, k=20)
log("affinity nnz=", A.nnz)

log("stability selection of K...")
K10, stab = stability_select_K(P, Kmax=12, B=20, n_sub=4000, k=15)
log("stability chose K=", K10, "curve", {k: round(v, 3) for k, v in stab.items()})

# partitions
log("partitioning pure spectral K=4/6/%d..." % K10)
lab_cnc4 = spectral_partition(A, 4)
lab_cnc6 = spectral_partition(A, 6)
lab_cnc10 = spectral_partition(A, K10)

log("building spectral-spatial affinity (sigma=12px)...")
Asp = spatial_gate(A, sigma=12.0)
lab_cnc10s = spectral_partition(Asp, K10)

log("KMeans baseline K=%d..." % K10)
lab_km = KMeans(K10, n_init=10, random_state=0).fit_predict(P)
# Fair, same-K K-means controls (apple-to-apple with CNC).
lab_km4 = KMeans(4, n_init=10, random_state=0).fit_predict(P)
lab_km6 = KMeans(6, n_init=10, random_state=0).fit_predict(P)

hom_cnc4 = spatial_homogeneity(lab_cnc4)
hom_cnc6 = spatial_homogeneity(lab_cnc6)
hom_cnc10 = spatial_homogeneity(lab_cnc10)
hom_cnc10s = spatial_homogeneity(lab_cnc10s)
hom_km = spatial_homogeneity(lab_km)
hom_km4 = spatial_homogeneity(lab_km4)
hom_km6 = spatial_homogeneity(lab_km6)
log("spatial homogeneity (8-neighbor agreement): "
    f"CNC-K4={hom_cnc4:.3f} CNC-K6={hom_cnc6:.3f} CNC-K10={hom_cnc10:.3f} "
    f"CNC-spatial-K10={hom_cnc10s:.3f} | "
    f"KMeans-K4={hom_km4:.3f} KMeans-K6={hom_km6:.3f} KMeans-K{K10}={hom_km:.3f}")


# ---------------------------------------------------------------- figures
cmap = "tab20"

def save(fig, name):
    """Save a figure into paper/figs (for the LaTeX build) and results/ (copy)."""
    p1 = os.path.join(PAPER_FIGS, name)
    p2 = os.path.join(RESULTS_DIR, name)
    fig.savefig(p1, dpi=140)
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    log(f"saved {p1}")

# (1) stability curve -> paper/figs/ayc_kstab.png
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ks = sorted(stab)
ax.plot(ks, [stab[k] for k in ks], "o-", color="#2c7fb8")
ax.axvline(K10, color="#d95f0e", ls="--", label=f"chosen K={K10}")
ax.set_xlabel("K"); ax.set_ylabel("mean bootstrap NMI")
ax.set_title("Stability selection of K", fontsize=11)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "ayc_kstab.png")

# (2) 2x2 comparison clean -> paper/figs/ayc_compare_clean.png
#     Direct same-K (apple-to-apple) comparison: CNC vs K-means at K=4 and K=6.
def plot_map(ax, lab, title):
    ax.imshow(lab.reshape(H, W), cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, lab, title in [(axes[0, 0], lab_cnc4,  "CNC (spectral) K=4"),
                       (axes[0, 1], lab_km4,   "K-means K=4"),
                       (axes[1, 0], lab_cnc6,  "CNC (spectral) K=6"),
                       (axes[1, 1], lab_km6,   "K-means K=6")]:
    plot_map(ax, lab, title)
fig.suptitle("Asian Games Village hyperspectral segmentation: CNC vs K-means at the same K "
             "(54 bands, 462-1020 nm)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(PAPER_FIGS, "ayc_compare_clean.png"), dpi=140)
fig.savefig(os.path.join(PAPER_FIGS, "ayc_compare.png"), dpi=140)
fig.savefig(os.path.join(RESULTS_DIR, "ayc_compare_clean.png"), dpi=140)
fig.savefig(os.path.join(RESULTS_DIR, "ayc_compare.png"), dpi=140)
plt.close(fig)
log(f"saved {os.path.join(PAPER_FIGS, 'ayc_compare.png')}")

# (3) mean spectra (raw radiance, English in-figure) -> paper/figs/ayc_spectra_paper.pdf
band_axis = keep + 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, lab, ttl in [(axes[0], lab_cnc4, "CNC spectral K=4"),
                     (axes[1], lab_cnc10, f"CNC spectral K={K10}")]:
    for c in np.unique(lab):
        m = X_clean[lab == c]
        mn = m.mean(0); st = m.std(0)
        ax.plot(band_axis, mn, label=f"C{c} (n={len(m)})", lw=1.6)
        ax.fill_between(band_axis, mn - st, mn + st, alpha=0.12)
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("band index (1-54, 462-1020 nm)")
    ax.set_ylabel("mean radiance")
    ax.grid(alpha=0.3)
    if ttl.startswith("CNC spectral K=4"):
        ax.legend(fontsize=8, ncol=2)
fig.suptitle("Cluster-mean spectra (Ayuncun hyperspectral, 54 bands)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
p_pdf = os.path.join(PAPER_FIGS, "ayc_spectra_paper.pdf")
p_png = os.path.join(PAPER_FIGS, "ayc_spectra_paper.png")
fig.savefig(p_pdf, dpi=150); fig.savefig(p_png, dpi=150); plt.close(fig)
log(f"saved {p_pdf}")

# (4) supplementary 2x3 full comparison -> results/ayc_compare_full.png
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
plot_map(axes[0, 0], lab_cnc4,  f"CNC (spectral) K=4")
plot_map(axes[0, 1], lab_cnc6,  f"CNC (spectral) K=6")
plot_map(axes[0, 2], lab_cnc10, f"CNC (spectral) K={K10}")
plot_map(axes[1, 0], lab_cnc10s, f"CNC (spectral+spatial) K={K10}")
plot_map(axes[1, 1], lab_km,     f"KMeans K={K10}")
ax = axes[1, 2]
ax.plot(ks, [stab[k] for k in ks], "o-", color="#2c7fb8")
ax.axvline(K10, color="#d95f0e", ls="--", label=f"chosen K={K10}")
ax.set_xlabel("K"); ax.set_ylabel("mean bootstrap NMI")
ax.set_title("Stability selection of K", fontsize=10)
ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Asian Games Village hyperspectral segmentation (supplementary)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(RESULTS_DIR, "ayc_compare_full.png"), dpi=140); plt.close(fig)
log("saved results/ayc_compare_full.png")


# ---------------------------------------------------------------- metrics
def cluster_stats(labels):
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}

def mean_spectra(labels):
    uniq = np.unique(labels)
    return np.array([X_clean[labels == u].mean(axis=0) for u in uniq])   # (K, 54)

def inter_cluster_sep(means):
    K = means.shape[0]
    d = [np.linalg.norm(means[i] - means[j])
         for i in range(K) for j in range(i + 1, K)]
    return float(np.mean(d)), float(np.std(d))

methods = {
    "CNC K=4":        lab_cnc4,
    "CNC K=6":        lab_cnc6,
    "CNC K=10":       lab_cnc10,
    "CNC K=10 spatial": lab_cnc10s,
    "KMeans K=4":     lab_km4,
    "KMeans K=6":     lab_km6,
    "KMeans K=10":    lab_km,
}
out = {}
for name, l in methods.items():
    coh = spatial_homogeneity(l)
    stats = cluster_stats(l)
    ms = mean_spectra(l)
    sep_mean, sep_std = inter_cluster_sep(ms)
    out[name] = {
        "spatial_coherence": round(coh, 4),
        "cluster_sizes": stats,
        "n_clusters": int(len(stats)),
        "inter_cluster_sep_mean": round(sep_mean, 2),
        "inter_cluster_sep_std": round(sep_std, 2),
    }
    print(f"{name:18s} spatial_coh={coh:.4f}  K={len(stats)}  "
          f"inter-sep={sep_mean:.1f}+-{sep_std:.1f}  sizes={list(stats.values())}")

with open(os.path.join(RESULTS_DIR, "ayc_paper_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
log("saved results/ayc_paper_metrics.json")

# save labels for reuse
np.savez(os.path.join(RESULTS_DIR, "ayc_labels.npz"),
         cnc4=lab_cnc4, cnc6=lab_cnc6, cnc10=lab_cnc10,
         cnc10s=lab_cnc10s, km=lab_km, km4=lab_km4, km6=lab_km6,
         bands=keep, components=n_comp, K10=K10)
log("saved results/ayc_labels.npz")


# ---------------------------------------------------------------- summary
print("\n" + "=" * 72)
print("Asian Games Village (亚运村) hyperspectral clustering summary")
print("=" * 72)
print(f"pixels={H*W}  usable bands=54  PCA comps={n_comp}  K(stability)={K10}")
print("Spatial homogeneity (8-neighbor label agreement) - same K comparison:")
print(f"  CNC K=4          : {hom_cnc4:.3f}   KMeans K=4     : {hom_km4:.3f}")
print(f"  CNC K=6          : {hom_cnc6:.3f}   KMeans K=6     : {hom_km6:.3f}")
print(f"  CNC K=10         : {hom_cnc10:.3f}  KMeans K={K10}  : {hom_km:.3f}")
print(f"  CNC K=10 spatial : {hom_cnc10s:.3f}")
print(f"total time {time.time()-t0:.1f}s")
