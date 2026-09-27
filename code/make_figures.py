# -*- coding: utf-8 -*-
"""Render journal-quality figures for the CNC / adjacent-partition-agreement
manuscript, replacing the weaker auto-generated plots.

Outputs (PDF vector + 300-dpi PNG) into figures/paper_figures/ as Fig1..Fig5,
in manuscript order of appearance:
  Fig1.pdf/png  - framework / architectural diagram of CNC (Method)
  Fig2.pdf/png  - resonance-affinity vs Euclidean k-NN on a non-convex toy (Method)
  Fig3.pdf/png  - core innovation: coarse stability maximum correction (Results)
  Fig4.pdf/png  - CNC vs 7 baselines across 7 scenarios (Results, bar charts)
  Fig5.pdf/png  - real-world hyperspectral application (Asian Games Village, Results)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
# Resolve the project root: if this script lives inside a "code" folder, the
# root is its parent; otherwise the script itself sits at the root.
PROJ = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "code" else HERE
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures", "paper_figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(PROJ, "code"))

# ----------------------------------------------------------------------------
# Shared journal style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 140,
})

# Professional, partly colour-blind-safe palette
PALETTE = {
    "CNC":        "#1f6fb2",
    "KMeans":     "#e6812e",
    "DBSCAN":     "#2ca25f",
    "DPC":        "#9b6ab0",
    "AP":         "#d6453f",
    "SNN":        "#7f8c8d",
    "RBFSpectral":"#13a0a0",
    "HDBSCAN":    "#6a4c93",
}
CNC_BLUE   = "#1f6fb2"
ACCENT     = "#d16002"   # contribution / refinement highlight
GREY       = "#5b6573"
SOFTGRID   = "#dfe5eb"
INK        = "#1b2733"

def panel_letter(ax, letter, xy=(-0.16, 1.02), fs=11):
    ax.text(xy[0], xy[1], letter, transform=ax.transAxes, fontsize=fs,
            fontweight="bold", color=INK, va="bottom", ha="left")

def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=SOFTGRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------
# FIG 1 - framework / architectural diagram (clean, Method)
# ----------------------------------------------------------------------------
def make_fig1():
    fig = plt.figure(figsize=(7.8, 3.4))
    canvas = fig.add_axes([0, 0, 1, 1]); canvas.axis("off")
    canvas.set_xlim(0, 1); canvas.set_ylim(0, 1)

    stages = [
        ("Input data", "Unlabeled points,\nunknown K, often\nnon-convex structure"),
        ("Resonance affinity", "Neighbourhood consensus\nbuilds a sparse graph\n$R(i,j)\\in[0,1]$"),
        ("Spectral partitioning", "Normalised-Laplacian\neigenmap into a\nlow-dim embedding"),
        ("K-selection + refinement", "Stability scan corrected\nby adjacent-partition\nagreement"),
    ]
    n = len(stages)
    w, x0s, gap = 0.212, 0.030, 0.0
    gap = (1 - 2 * x0s - n * w) / (n - 1)
    y0, h = 0.26, 0.46

    for i, (title, desc) in enumerate(stages):
        x0 = x0s + i * (w + gap)
        is_contrib = (i == n - 1)
        box = FancyBboxPatch((x0, y0), w, h,
                             boxstyle="round,pad=0.006,rounding_size=0.02",
                             linewidth=1.8 if is_contrib else 0.9,
                             edgecolor=ACCENT if is_contrib else "#9fb0c0",
                             facecolor="#fff5ec" if is_contrib else "#f6f8fa")
        canvas.add_patch(box)
        canvas.text(x0 + w / 2, y0 + h - 0.06, title, ha="center", va="top",
                    fontsize=8.6, fontweight="bold",
                    color=ACCENT if is_contrib else INK)
        canvas.text(x0 + w / 2, y0 + 0.075, desc, ha="center", va="bottom",
                    fontsize=7.4, color=GREY, linespacing=1.3)
        if i < n - 1:
            ax2 = x0 + w
            canvas.add_patch(FancyArrowPatch((ax2 + 0.006, y0 + h / 2),
                              (ax2 + gap - 0.006, y0 + h / 2),
                              arrowstyle="-|>", mutation_scale=10, lw=1.2,
                              color="#7f8c8d"))

    # contribution bracket / label under the last (highlighted) stage
    lx = x0s + (n - 1) * (w + gap)
    canvas.add_patch(FancyBboxPatch((lx - 0.01, y0 - 0.105), w + 0.02, 0.075,
                         boxstyle="round,pad=0.004,rounding_size=0.012",
                         linewidth=0.0, facecolor="#fbe6d2"))
    canvas.text(lx + w / 2, y0 - 0.067, "Contribution of this work",
                ha="center", va="center", fontsize=7.8, fontweight="bold",
                color=ACCENT)

    # output label
    canvas.text(0.5, 0.095,
                "CNC returns the final partition together with an automatically selected cluster count $K^{*}$",
                ha="center", va="bottom", fontsize=8.0, color=INK)

    canvas.text(0.5, 0.95,
                "Framework of Consensus-Neighbourhood Clustering (CNC)",
                ha="center", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(os.path.join(FIG, "Fig1.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "Fig1.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig1] framework saved")


# ----------------------------------------------------------------------------
# FIG 2 - resonance affinity vs Euclidean k-NN (replaces heatmap, Method)
# ----------------------------------------------------------------------------
def make_fig2():
    import cnc
    from datasets import build_family
    from sklearn.neighbors import NearestNeighbors

    X, y = build_family("Moons", 0)
    y = np.asarray(y)
    # subsample for clean edge drawing
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=220, replace=False)
    Xs, ys = X[idx], y[idx]

    # Euclidean k-NN edges
    knn = NearestNeighbors(n_neighbors=6).fit(Xs)
    _, nn = knn.kneighbors(Xs)
    eucl_edges = [(i, j) for i in range(len(Xs)) for j in nn[i] if j > i]

    # Resonance-affinity edges (top neighbours per node)
    R, _, _ = cnc.resonance_affinity(Xs, k=20, rank_exponent=2.0)
    R = R.toarray()
    res_edges = []
    for i in range(len(Xs)):
        js = np.argsort(R[i])[::-1][:4]
        for j in js:
            if j > i and R[i, j] > 0.02:
                res_edges.append((i, j, R[i, j]))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    col_true = np.where(ys == 0, "#1f6fb2", "#d16002")

    # left: Euclidean k-NN
    ax = axes[0]
    for i, j in eucl_edges:
        ax.plot([Xs[i, 0], Xs[j, 0]], [Xs[i, 1], Xs[j, 1]],
                color="#9aa7b4", lw=0.5, alpha=0.55, zorder=1)
    ax.scatter(Xs[:, 0], Xs[:, 1], s=14, c=col_true, linewidths=0, zorder=2)
    ax.set_title("Euclidean k-NN graph", loc="left", fontsize=10)
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.tick_params(labelsize=7.5)
    panel_letter(ax, "a", xy=(-0.12, 1.04))

    # right: resonance affinity
    ax = axes[1]
    wmax = max([e[2] for e in res_edges]) if res_edges else 1.0
    for i, j, w in res_edges:
        ax.plot([Xs[i, 0], Xs[j, 0]], [Xs[i, 1], Xs[j, 1]],
                color="#1f6fb2", lw=0.5 + 1.6 * (w / wmax), alpha=0.45, zorder=1)
    ax.scatter(Xs[:, 0], Xs[:, 1], s=14, c=col_true, linewidths=0, zorder=2)
    ax.set_title("Resonance-affinity graph (proposed)", loc="left", fontsize=10)
    ax.set_xlabel("feature 1")
    ax.tick_params(labelleft=False, labelsize=7.5)
    panel_letter(ax, "b", xy=(-0.12, 1.04))

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.savefig(os.path.join(FIG, "Fig2.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "Fig2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig2] affinity-vs-knn saved")


# ----------------------------------------------------------------------------
# FIG 3 - core innovation: coarse stability maximum correction (Results)
# ----------------------------------------------------------------------------
def make_fig3():
    ms = json.load(open(os.path.join(RES, "margin_sensitivity_expanded.json")))
    datasets = [("Blobs", 4), ("Digits", 10)]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3))
    payoff = {"Blobs": (0.3186, 0.8273, 2, 4),
              "Digits": (0.0469, 0.7416, 2, 9)}

    for col, (ds, trueK) in enumerate(datasets):
        c = ms["datasets"][ds]["CNC"]
        stab = {int(k): v for k, v in c["stab"].items()}
        cont = {int(k): v for k, v in c["continuity"].items()}
        ks = sorted(stab)
        # ---- curves panel ----
        ax = axes[0, col]
        ax.plot(ks, [stab[k] for k in ks], color=CNC_BLUE, lw=2.0, marker="o",
                ms=4.5, label="Stability (subsample NMI)")
        ax.plot(ks, [cont.get(k, np.nan) for k in ks], color=ACCENT, lw=1.8,
                marker="s", ms=3.8, ls="--", label="Adjacent-partition agreement")
        argmaxK = max(stab, key=stab.get)
        refinedK = payoff[ds][3]
        ax.axvline(argmaxK, color="#c0392b", lw=1.0, ls=":", alpha=0.85)
        ax.axvline(refinedK, color=CNC_BLUE, lw=1.3, ls="--", alpha=0.6)
        ax.scatter([argmaxK], [stab[argmaxK]], color="#c0392b", zorder=6, s=34,
                   marker="x", lw=1.6)
        ax.annotate(f"coarse stability max\\nK={argmaxK}",
                    xy=(argmaxK, stab[argmaxK]),
                    xytext=(argmaxK + 1.2, 0.67 if ds == "Blobs" else 0.70),
                    fontsize=7.0, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9,
                                    connectionstyle="arc3,rad=-0.15"))
        ax.annotate(f"refined K={refinedK}",
                    xy=(refinedK, cont.get(refinedK, stab[refinedK])),
                    xytext=(refinedK + 0.2, 0.61), fontsize=7.2, color=CNC_BLUE,
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=CNC_BLUE, lw=0.9,
                                    connectionstyle="arc3,rad=0.1"))
        ax.set_ylim(0.55, 1.02)
        ax.set_xlabel("Candidate cluster count $K$")
        ax.set_ylabel("Agreement (NMI)")
        ax.set_title(f"{ds}  (true $K$={trueK})", loc="left", fontsize=9.5)
        ax.set_xticks(ks)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=SOFTGRID, lw=0.7); ax.set_axisbelow(True)
        if col == 0:
            ax.legend(frameon=False, fontsize=7.2, loc="upper right")
        panel_letter(ax, chr(97 + col))

        # ---- payoff panel ----
        ax = axes[1, col]
        a_low, a_high, k_low, k_high = payoff[ds]
        bars = [("Plain stability\nargmax", a_low, "#c0392b"),
                ("Refined\n(proposed)", a_high, CNC_BLUE)]
        xx = np.arange(2)
        for j, (lab, val, col_) in enumerate(bars):
            ax.bar(xx[j], val, width=0.56, color=col_, edgecolor="white",
                   linewidth=0.6)
            ax.text(xx[j], val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=col_)
        ax.set_xticks(xx); ax.set_xticklabels([f"{lab}\nK={k}" for lab, _, k in
                   zip([b[0] for b in bars], bars, [k_low, k_high])], fontsize=7.6)
        ax.set_xlim(-0.2, 1.70)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("ARI (higher is better)")
        ax.set_title("Partition accuracy after selection", loc="left", fontsize=9.5)
        ax.text(1.55, (a_low + a_high) / 2,
                f"$\\Delta$ARI\n=+{a_high - a_low:.2f}", fontsize=7.6,
                color=GREY, va="center", ha="center")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=SOFTGRID, lw=0.7); ax.set_axisbelow(True)
        panel_letter(ax, chr(99 + col))

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    fig.savefig(os.path.join(FIG, "Fig3.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "Fig3.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig3] refinement saved")


# ----------------------------------------------------------------------------
# FIG 4 - benchmark advantage (bar charts replace heatmap, Results)
# ----------------------------------------------------------------------------
def make_fig4():
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    bm = json.load(open(os.path.join(RES, "benchmark_results.json")))
    methods = ["CNC", "KMeans", "DBSCAN", "DPC", "AP", "SNN", "RBFSpectral", "HDBSCAN"]
    datasets = ["Moons", "Circles", "Blobs", "Aniso", "VariedBlobs", "Gauss20D", "Digits"]
    def ari(ds, m):
        d = bm["synthetic"][ds] if ds in bm["synthetic"] else bm[ds]
        return float(d[m]["ARI"])
    M = np.array([[ari(ds, m) for ds in datasets] for m in methods])
    mean_ari = M.mean(axis=1)

    fig = plt.figure(figsize=(7.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.24)

    # (a) mean ARI across 7 scenarios, sorted
    axa = fig.add_subplot(gs[0])
    order = np.argsort(mean_ari)[::-1]
    ypos = np.arange(len(methods))
    for yy, idx in zip(ypos, order):
        val = mean_ari[idx]
        col = ACCENT if methods[idx] == "CNC" else "#bcd2e6"
        axa.barh(yy, val, color=col, edgecolor="white", linewidth=0.5)
        axa.text(val + 0.012, yy, f"{val:.2f}", va="center", fontsize=7.4,
                 color=INK, fontweight="bold" if methods[idx] == "CNC" else "normal")
    axa.set_yticks(ypos)
    axa.set_yticklabels([methods[idx] for idx in order], fontsize=8)
    axa.invert_yaxis()
    axa.set_xlim(0, 1.05)
    axa.set_xlabel("Mean ARI across 7 scenarios")
    axa.set_title("Average performance", loc="left", fontsize=10)
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.grid(axis="x", color=SOFTGRID, lw=0.7); axa.set_axisbelow(True)
    panel_letter(axa, "a")

    # (b) small multiples: ARI per scenario
    pos = gs[1].get_position(fig)
    inner = GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[1],
                                    wspace=0.22, hspace=0.62)
    for di, ds in enumerate(datasets):
        r, c = divmod(di, 2)
        ax = fig.add_subplot(inner[r, c])
        yp = np.arange(len(methods))
        for m in range(len(methods)):
            col = ACCENT if methods[m] == "CNC" else "#bcd2e6"
            ax.barh(yp[m], M[m, di], color=col, edgecolor="white", linewidth=0.25)
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "", "1"], fontsize=6.0)
        ax.set_xlim(0, 1.02)
        ax.set_title(ds, fontsize=7.6, loc="left", pad=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(pos.x0, pos.y1 + 0.030, "ARI by scenario", fontsize=10,
             fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(pos.x0 - 0.038, pos.y1 + 0.030, "b", fontsize=11,
             fontweight="bold", color=INK, ha="left", va="bottom")

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])
    fig.savefig(os.path.join(FIG, "Fig4.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "Fig4.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig4] benchmark saved")


# ----------------------------------------------------------------------------
# FIG 5 - hyperspectral real-world application (Results)
# ----------------------------------------------------------------------------
def make_fig5():
    ayv = json.load(open(os.path.join(RES, "ayc_kstab_ext24.json")))
    stab = {int(k): v for k, v in ayv["stability"].items()}
    selK = ayv.get("selected_K", 20)
    ks = sorted(stab)

    fig = plt.figure(figsize=(7.6, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.12)

    # (a) stability selection curve
    ax = fig.add_subplot(gs[0])
    ax.plot(ks, [stab[k] for k in ks], color=CNC_BLUE, lw=1.9, marker="o", ms=3.6)
    ax.axvline(selK, color=ACCENT, lw=1.3, ls="--", alpha=0.8)
    ax.scatter([selK], [stab[selK]], color=ACCENT, s=40, zorder=6)
    ax.annotate(f"selected\nK={selK}", xy=(selK, stab[selK]),
                xytext=(selK - 6.5, stab[selK] - 0.13), fontsize=7.4,
                color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9))
    ax.set_xlabel("Candidate cluster count $K$", labelpad=10)
    ax.set_ylabel("Subsample-partition NMI")
    ax.set_title("Stability selection (real HSI)", loc="left", fontsize=9.0)
    ax.set_xticks([2, 6, 10, 14, 18, 22])
    ax.set_ylim(0.14, 0.82)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=SOFTGRID, lw=0.7); ax.set_axisbelow(True)
    panel_letter(ax, "a")

    # (b,c) cluster maps
    cnc_lab = np.load(os.path.join(FIG, "ayc_cnc_labels_K20.npy"))
    km_lab = np.load(os.path.join(FIG, "ayc_kmeans_labels_K20.npy"))
    side = int(round(cnc_lab.shape[0] ** 0.5))
    for ax_i, (lab, title) in zip([fig.add_subplot(gs[1]), fig.add_subplot(gs[2])],
                                   [(cnc_lab, "CNC  K=20"), (km_lab, "K-means  K=20")]):
        m = lab.reshape(side, side)
        ax_i.imshow(m, cmap="tab20", interpolation="nearest")
        ax_i.set_title(title, fontsize=9.0)
        ax_i.set_xticks([]); ax_i.set_yticks([])
        panel_letter(ax_i, "bc"[0] if title.startswith("CNC") else "bc"[1], xy=(-0.02, 1.04))

    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.savefig(os.path.join(FIG, "Fig5.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "Fig5.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Fig5] hyperspectral saved")

if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    # Fig 5 depends on regenerated AYC labels (background task)
    if os.path.exists(os.path.join(FIG, "ayc_cnc_labels_K20.npy")):
        make_fig5()
        print("[main] all figures done")
    else:
        print("[main] Fig5 skipped (AYC labels not ready yet)")
