"""All paper figure rendering.

Every function takes the already-computed data (loaded from JSON) and writes
a PDF/PNG into ``figures/paper_figures/``.  This keeps the experiment notebook
thin and makes figures cheap to re-render after a data change.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

METHOD_COLORS = {
    "CNC": "#2369a8",
    "KMeans": "#d9772b",
    "DBSCAN": "#3a9368",
    "DPC": "#8a62a8",
    "AP": "#c94d4d",
    "SNN": "#6c7f93",
    "RBFSpectral": "#c95a3f",
    "HDBSCAN": "#4e9a72",
}
METHOD_MARKERS = {
    "CNC": "o", "KMeans": "s", "DBSCAN": "^", "DPC": "D", "AP": "P",
}


def _out_dir(ROOT: str) -> str:
    d = os.path.join(ROOT, "figures", "paper_figures")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Benchmark dot plot + CNC-centered gap heatmap
# ---------------------------------------------------------------------------
def _lookup_ari(out: dict, dataset: str, method: str) -> float:
    if dataset in out.get("synthetic", {}):
        return float(out["synthetic"][dataset][method]["ARI"])
    return float(out[dataset][method]["ARI"])


def render_benchmark_figures(out: dict, ROOT: str) -> None:
    d = _out_dir(ROOT)
    order = ["Moons", "Circles", "Blobs", "Aniso", "VariedBlobs"]
    methods = ["CNC", "KMeans", "DBSCAN", "DPC", "AP"]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.35), sharex=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, order):
        y = np.arange(len(methods))
        for yi, method in zip(y, methods):
            value = _lookup_ari(out, dataset, method)
            ax.hlines(yi, 0.0, value, color="#d5dde6", linewidth=3.0, zorder=1)
            ax.scatter(value, yi, s=62, marker=METHOD_MARKERS[method],
                       color=METHOD_COLORS[method], edgecolor="white",
                       linewidth=0.9, zorder=3)
            ax.text(min(value + 0.025, 1.035), yi, f"{value:.2f}",
                    va="center", ha="left", fontsize=8.2,
                    fontweight="bold" if method == "CNC" else "normal")
        ax.set_title(dataset, pad=8, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.14)
        ax.set_ylim(len(methods) - 0.35, -0.65)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(axis="x", color="#dfe5eb", linewidth=0.75)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].axis("off")
    fig.suptitle("Synthetic benchmark: method scores by dataset",
                 fontsize=12, fontweight="bold", y=0.985)
    fig.supxlabel("Adjusted Rand Index (ARI; higher is better)", y=0.035)
    fig.text(0.5, 0.075,
             "CNC, DBSCAN and AP use automatic settings; KMeans and DPC use oracle K.",
             ha="center", fontsize=8.3, color="#5b6573")
    fig.tight_layout(rect=[0.02, 0.105, 0.98, 0.94])
    fig.savefig(os.path.join(d, "fig_benchmark_expanded.pdf"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    order7 = ["Moons", "Circles", "Blobs", "Aniso", "VariedBlobs",
              "Gauss20D", "Digits"]
    baseline_methods = ["SNN", "RBFSpectral", "HDBSCAN"]
    cnc_vals = np.array([_lookup_ari(out, x, "CNC") for x in order7])
    gaps = np.array([
        [_lookup_ari(out, x, m) - c for m in baseline_methods]
        for x, c in zip(order7, cnc_vals)
    ])
    gap_lim = max(0.25, float(np.ceil(np.max(np.abs(gaps)) * 10.0) / 10.0))
    cmap = LinearSegmentedColormap.from_list("g", ["#2166ac", "#f7f7f7", "#b2182b"])
    norm = TwoSlopeNorm(vmin=-gap_lim, vcenter=0.0, vmax=gap_lim)
    fig = plt.figure(figsize=(11.3, 5.85), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 3.5], wspace=0.36)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1], sharey=ax_c)
    y = np.arange(len(order7))
    ax_c.hlines(y, 0.0, cnc_vals, color="#cbd6e1", linewidth=3.2, zorder=1)
    ax_c.scatter(cnc_vals, y, s=76, color=METHOD_COLORS["CNC"],
                 edgecolor="white", linewidth=0.9, zorder=3)
    for yi, v in zip(y, cnc_vals):
        ax_c.text(min(v + 0.035, 1.04), yi, f"{v:.2f}",
                  va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax_c.set_title("CNC ARI", pad=6, fontweight="bold")
    ax_c.set_xlabel("ARI", labelpad=8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(order7)
    ax_c.invert_yaxis()
    ax_c.set_xlim(-0.02, 1.14)
    ax_c.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_c.grid(axis="x", color="#dfe5eb", linewidth=0.75)
    ax_c.set_axisbelow(True)
    im = ax_g.imshow(gaps, cmap=cmap, norm=norm, aspect="auto")
    ax_g.set_title("Baseline ARI - CNC ARI", pad=4, fontweight="bold")
    ax_g.set_xticks(np.arange(len(baseline_methods)))
    ax_g.set_xticklabels(["SNN\noracle K", "RBF-spectral\noracle K",
                         "HDBSCAN\nauto K"], fontsize=8.4)
    ax_g.tick_params(axis="y", left=False, labelleft=False)
    ax_g.set_xlabel("Negative values favor CNC; positive values favor the baseline",
                   labelpad=11, fontsize=8.4)
    for r in range(gaps.shape[0]):
        for c in range(gaps.shape[1]):
            v = gaps[r, c]
            ax_g.text(c, r, f"{v:+.2f}", ha="center", va="center",
                      fontsize=9, color="#17202a",
                      fontweight="bold" if abs(v) >= 0.20 else "normal")
    cb = fig.colorbar(im, ax=ax_g, fraction=0.046, pad=0.035)
    cb.set_label("ARI gap", rotation=270, labelpad=12)
    for ax in (ax_c, ax_g):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("CNC-centered comparison across seven scenarios",
                 fontsize=12, fontweight="bold", y=1.015)
    fig.savefig(os.path.join(d, "fig_baseline_compare.pdf"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rank-exponent ablation
# ---------------------------------------------------------------------------
def render_rank_ablation(exponents: List[int], order: List[str],
                         results: Dict[str, Dict[str, float]], ROOT: str) -> None:
    d = _out_dir(ROOT)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = ["#2369a8", "#d9772b", "#3a9368", "#8a62a8", "#c94d4d"]
    for name, color in zip(order, colors):
        row = results[name]
        vals = [row[str(p)] if str(p) in row else row[p] for p in exponents]
        ax.plot(exponents, vals, marker="o", linewidth=1.8, markersize=5.5,
                label=name, color=color)
    ax.set_xlabel("Rank exponent $p$ in $w=(k-r)^p$")
    ax.set_ylabel("ARI at matched true $K$")
    ax.set_xticks(exponents)
    ax.set_ylim(-0.02, 1.04)
    ax.grid(axis="y", color="#dfe5eb", linewidth=0.75)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Rank-weighting ablation at $N=1000$, $k=20$",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "cnc_rank_ablation.pdf"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-seed robustness
# ---------------------------------------------------------------------------
def render_multiseed(summary: dict, order: List[str], ROOT: str) -> None:
    d = _out_dir(ROOT)
    methods = ["CNC_auto", "CNC_oracleK", "SNN_oracleK", "KMeans_oracleK"]
    labels = ["CNC auto", "CNC matched K", "SNN matched K", "K-means"]
    colors = ["#2369a8", "#4d8bc2", "#6c7f93", "#d9772b"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    width = 0.18
    for i, (m, lab, col) in enumerate(zip(methods, labels, colors)):
        means = [summary[k][m]["ARI_mean"] for k in order]
        stds = [summary[k][m]["ARI_std"] for k in order]
        ax.errorbar(x + (i - 1.5) * width, means, yerr=stds,
                    fmt="o", markersize=5.5, capsize=2.5, linewidth=1.1,
                    color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("ARI (mean +/- one standard deviation)")
    ax.set_ylim(-0.02, 1.06)
    ax.grid(axis="y", color="#dfe5eb", linewidth=0.75)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Five-seed robustness study ($N=1000$, $k=20$)",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_multiseed_robustness.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runtime scaling
# ---------------------------------------------------------------------------
def render_runtime(records: List[dict], ROOT: str) -> None:
    d = _out_dir(ROOT)
    n = np.array([r["N"] for r in records])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, label, color in [
        ("graph_seconds", "Affinity construction", "#2369a8"),
        ("spectral_seconds", "Fixed-K spectral partition", "#d9772b"),
        ("stability_seconds", "Stability selection", "#8a62a8"),
    ]:
        ax.plot(n, [r[key] for r in records], marker="o", linewidth=1.8,
                markersize=5.5, label=label, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(n)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of samples $N$")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.grid(which="both", color="#dfe5eb", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Measured CNC runtime scaling on four Gaussian blobs",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.3)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_runtime_scaling.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Resonance affinity heatmap (needs datasets + cnc module)
# ---------------------------------------------------------------------------
def render_affinity(datasets: Dict[str, tuple], cnc_mod, ROOT: str) -> None:
    d = _out_dir(ROOT)

    def _order(X, y):
        idx = []
        for lab in sorted(np.unique(y)):
            li = np.flatnonzero(y == lab)
            sub = X[li]
            axis = np.argmax(np.ptp(sub, axis=0))
            o = np.argsort(sub[:, axis])
            idx.extend(li[o].tolist())
        return np.array(idx)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    panels = [
        ("Four Gaussian blobs ($K{=}4$)", datasets["Blobs"]),
        ("Two moons ($K{=}2$)", datasets["Moons"]),
    ]
    vmax = 0.7
    im = None
    for ax, (title, (X, y)) in zip(axes, panels):
        A, _, _ = cnc_mod.resonance_affinity(X, k=20)
        Ad = A.toarray() if hasattr(A, "toarray") else np.asarray(A, dtype=float)
        o = _order(X, y)
        Ad = Ad[np.ix_(o, o)]
        im = ax.imshow(Ad, cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
        bounds = [0]
        for lab in sorted(np.unique(y)):
            bounds.append(bounds[-1] + int(np.sum(y == lab)))
        for b in bounds[1:-1]:
            ax.axvline(b - 0.5, color="cyan", lw=0.9, alpha=0.85)
            ax.axhline(b - 0.5, color="cyan", lw=0.9, alpha=0.85)
        ax.set_title(f"Resonance affinity -- {title}", fontsize=10)
        ax.set_xlabel("samples (sorted by label, then by coordinate)")
        ax.set_ylabel("samples (sorted by label, then by coordinate)")
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="$R(i,j)$")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "cnc_affinity.pdf"), dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic clustering scatter (needs per-dataset labels)
# ---------------------------------------------------------------------------
def render_synthetic(panels: List[dict], ROOT: str) -> None:
    """panels: list of dicts with keys 'name', 'X', 'labels'."""
    d = _out_dir(ROOT)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    cmap = plt.get_cmap("tab10")
    for ax, p in zip(axes.ravel(), panels):
        lab = p["labels"]
        uniq = np.unique(lab)
        for ci, c in enumerate(uniq):
            sel = lab == c
            ax.scatter(p["X"][sel, 0], p["X"][sel, 1], s=11,
                       c=[cmap(ci % 10)])
        ax.set_title(f"{p['name']}  (CNC clusters={len(uniq)})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("CNC -- resonance affinity + stability-selected spectral clustering",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(d, "cnc_synthetic.png"), dpi=300)
    fig.savefig(os.path.join(d, "cnc_synthetic.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hyperspectral diagnostics
# ---------------------------------------------------------------------------
def render_hsi_figures(X: np.ndarray, cnc_labels: Dict[int, np.ndarray],
                       kmeans_labels: Dict[int, np.ndarray],
                       stability: Dict[int, float], ROOT: str,
                       height: int = 200, width: int = 200) -> None:
    """Render all HSI figures from the current computed labels and scores."""
    d = _out_dir(ROOT)
    k_values = sorted(int(k) for k in stability)
    stab_values = [float(stability[str(k)] if str(k) in stability else stability[k])
                   for k in k_values]

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(k_values, stab_values, marker="o", color=METHOD_COLORS["CNC"],
            linewidth=1.8, markersize=5)
    best_k = k_values[int(np.argmax(stab_values))]
    ax.axvline(best_k, color="#8a62a8", linestyle="--", linewidth=1.1,
               label=f"selected K={best_k}")
    ax.set_xlabel("Candidate cluster count K")
    ax.set_ylabel("Subsample-partition NMI")
    ax.set_xticks(k_values)
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#dfe5eb", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("HSI stability selection", fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "ayc_kstab.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(d, "ayc_kstab.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    map_keys = [k for k in (4, 6, best_k) if k in cnc_labels]
    if map_keys:
        fig, axes = plt.subplots(2, len(map_keys), figsize=(3.3 * len(map_keys), 5.5),
                                 squeeze=False)
        for col, k in enumerate(map_keys):
            for row, (name, labels) in enumerate((("CNC", cnc_labels[k]),
                                                   ("K-means", kmeans_labels[k]))):
                ax = axes[row, col]
                ax.imshow(labels.reshape(height, width), cmap="tab20", interpolation="nearest")
                ax.set_title(f"{name}, K={k}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("Asian Games Village clustering maps", fontsize=11,
                     fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(d, "ayc_compare.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(d, "ayc_compare_clean.png"), dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    plot_keys = [k for k in (4, best_k) if k in cnc_labels]
    if plot_keys:
        fig, axes = plt.subplots(1, len(plot_keys), figsize=(7.8, 3.1),
                                 squeeze=False)
        band_axis = np.arange(1, X.shape[1] + 1)
        for col, k in enumerate(plot_keys):
            ax = axes[0, col]
            labels = cnc_labels[k]
            for c in sorted(np.unique(labels)):
                mask = labels == c
                ax.plot(band_axis, X[mask].mean(axis=0), linewidth=1.0, alpha=0.78)
            ax.set_title(f"CNC cluster spectra, K={k}", fontsize=10)
            ax.set_xlabel("Retained visible-NIR band index (1-54)")
            ax.set_ylabel("Mean standardized band value")
            ax.grid(axis="y", color="#dfe5eb", linewidth=0.7)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "ayc_spectra.pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(d, "ayc_spectra_paper.pdf"), dpi=300,
                    bbox_inches="tight")
        fig.savefig(os.path.join(d, "ayc_spectra_paper.png"), dpi=220,
                    bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Concept figure (cnc_idea.pdf) -- pure schematic
# ---------------------------------------------------------------------------
def render_idea(ROOT: str) -> None:
    d = _out_dir(ROOT)
    r = np.random.default_rng(0)
    a = r.normal([0, 1.4], 0.42, (90, 2))
    b = r.normal([2.4, -1.4], 0.42, (90, 2))
    pts = np.vstack([a, b])
    lab = np.array([0] * 90 + [1] * 90)
    cols = {0: "#d7301f", 1: "#1f78b4"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (qi, qj, ttl) in zip(axes, [
        (0, 90, "First-order distance: p_i and p_j are close in space"),
        (0, 90, "Second-order consensus: shared neighbourhoods reveal structure"),
    ]):
        for c in (0, 1):
            m = lab == c
            ax.scatter(pts[m, 0], pts[m, 1], s=12, c=cols[c], alpha=0.45, zorder=1)
        for q, mk, lbl in [(qi, "o", "p_i"), (qj, "^", "p_j")]:
            qx, qy = pts[q]
            dists = np.linalg.norm(pts - [qx, qy], axis=1)
            order = np.argsort(dists)
            order = [o for o in order if o not in (qi, qj)][:7]
            qcol = cols[lab[q]]
            ax.scatter(qx, qy, s=120, c=qcol, marker=mk, edgecolor="k",
                       linewidths=1.3, zorder=6)
            ax.annotate(lbl, (qx, qy), fontsize=10, fontweight="bold",
                        textcoords="offset points", xytext=(6, 6), color="k")
            for o in order:
                ax.plot([qx, pts[o, 0]], [qy, pts[o, 1]], c=qcol, lw=1.0,
                        alpha=0.55, zorder=2)
                ax.scatter(pts[o, 0], pts[o, 1], s=46, facecolors="none",
                           edgecolors=qcol, linewidths=1.5, zorder=4)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(d, "cnc_idea.pdf"), dpi=300)
    plt.close(fig)
