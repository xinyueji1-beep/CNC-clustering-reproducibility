# -*- coding: utf-8 -*-
"""Concept figure for the paper: the idea origin of CNC -- first-order distance
versus second-order (neighborhood) consensus.  All in-figure text is English;
the caption lives in paper/main.tex.  Saves paper/figs/fig_idea.pdf (and .png).

Run:  python code/fig_idea.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_FIGS = os.path.join(ROOT, "figures", "paper_figures")
os.makedirs(PAPER_FIGS, exist_ok=True)


def make_two_blobs(sep=2.4, n=90, seed=0):
    r = np.random.default_rng(seed)
    a = r.normal([0, 1.4], 0.42, (n, 2))
    b = r.normal([sep, -1.4], 0.42, (n, 2))
    pts = np.vstack([a, b])
    lab = np.array([0] * n + [1] * n)
    return pts, lab


def draw_neighborhoods(ax, pts, lab, qi, qj, k=7, qlabels=("p_i", "p_j")):
    cols = {0: "#d7301f", 1: "#1f78b4"}
    for c in (0, 1):
        m = lab == c
        ax.scatter(pts[m, 0], pts[m, 1], s=12, c=cols[c], alpha=0.45, zorder=1)
    for q, idx, mk, lbl in [(qi, 0, "o", qlabels[0]),
                             (qj, 1, "^", qlabels[1])]:
        qx, qy = pts[q]
        d = np.linalg.norm(pts - [qx, qy], axis=1)
        order = np.argsort(d)
        order = [o for o in order if o not in (qi, qj)][:k]
        qcol = cols[lab[q]]
        ax.scatter(qx, qy, s=120, c=qcol, marker=mk, edgecolor="k",
                   linewidths=1.3, zorder=6)
        ax.annotate(lbl, (qx, qy), fontsize=10, fontweight="bold",
                     textcoords="offset points", xytext=(6, 6), color="k")
        for rank, o in enumerate(order, 1):
            ax.plot([qx, pts[o, 0]], [qy, pts[o, 1]], c=qcol, lw=1.0,
                    alpha=0.55, zorder=2)
            ax.scatter(pts[o, 0], pts[o, 1], s=46, facecolors="none",
                        edgecolors=qcol, linewidths=1.5, zorder=4)
            ax.annotate(str(rank), (pts[o, 0], pts[o, 1]), fontsize=7,
                         color=qcol, textcoords="offset points", xytext=(2, 2))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

# Panel A: cross-cluster, spatially close -> distance lies
ptsA, labA = make_two_blobs(sep=2.4, seed=11)
red_idx = np.where(labA == 0)[0]
blue_idx = np.where(labA == 1)[0]
center_blue = ptsA[blue_idx].mean(0)
qiA = red_idx[np.argmin(np.linalg.norm(ptsA[red_idx] - center_blue, axis=1))]
center_red = ptsA[red_idx].mean(0)
qjA = blue_idx[np.argmin(np.linalg.norm(ptsA[blue_idx] - center_red, axis=1))]
draw_neighborhoods(axA, ptsA, labA, qiA, qjA, k=7,
                   qlabels=("p_i (red)", "p_j (blue)"))
pi = ptsA[qiA]; pj = ptsA[qjA]
axA.plot([pi[0], pj[0]], [pi[1], pj[1]], c="black", ls="--", lw=1.2, zorder=3)
axA.annotate(r"$d_{ij}$ small", ((pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2),
              fontsize=10, color="black", textcoords="offset points",
              xytext=(4, -14))
axA.set_title("A. cross-cluster & spatially adjacent", fontsize=11)
axA.set_xticks([]); axA.set_yticks([])
axA.text(0.02, 0.97, "distance view: similar  (wrong)\nneighborhood view: diverge  -> R low",
         transform=axA.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800", alpha=0.9))

# Panel B: same cluster, far apart -> distance lies other way
ptsB, labB = make_two_blobs(sep=7.5, seed=22)
red_idx = np.where(labB == 0)[0]
qiB = red_idx[np.argmin(ptsB[red_idx, 0])]
qjB = red_idx[np.argmax(ptsB[red_idx, 0])]
ptsB[:, 1] -= 1.0
draw_neighborhoods(axB, ptsB, labB, qiB, qjB, k=7,
                   qlabels=("p_i (red)", "p_j (red)"))
pi = ptsB[qiB]; pj = ptsB[qjB]
axB.plot([pi[0], pj[0]], [pi[1], pj[1]], c="black", ls="--", lw=1.2, zorder=3)
axB.annotate(r"$d_{ij}$ large", ((pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2),
              fontsize=10, color="black", textcoords="offset points",
              xytext=(0, 8))
axB.set_title("B. same cluster, spatially distant", fontsize=11)
axB.set_xticks([]); axB.set_yticks([])
axB.text(0.02, 0.97, "distance view: dissimilar  (wrong)\nneighborhood view: agree  -> R high",
         transform=axB.transAxes, fontsize=9, va="top",
         bbox=dict(boxstyle="round", fc="#d4edda", ec="#28a745", alpha=0.9))

fig.suptitle("CNC idea origin: why a second-order (neighborhood) affinity beats a first-order distance",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(PAPER_FIGS, "fig_idea.pdf"), dpi=150)
fig.savefig(os.path.join(PAPER_FIGS, "fig_idea.png"), dpi=150)
fig.savefig(os.path.join(PAPER_FIGS, "cnc_idea.pdf"), dpi=150)
print(f"saved: {os.path.join(PAPER_FIGS, 'fig_idea.pdf')} / .png")
