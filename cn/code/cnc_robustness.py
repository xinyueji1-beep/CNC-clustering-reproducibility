"""
Mechanism and robustness study for Consensus-Neighborhood Clustering (CNC).

Produces two artifacts used in the CMC manuscript:

1. cnc_affinity.pdf  -- resonance affinity matrices (sorted by ground-truth
                        label) for the Circles and Blobs datasets, visually
                        confirming Theorem 1 (approximate block-diagonal
                        structure under well-separated manifolds).
2. cnc_ksens.json    -- CNC adjusted Rand index (ARI) as a function of the
                        neighborhood size k, with the final cluster count K
                        fixed to the ground-truth count so that the effect of
                        k alone is isolated.

The script imports the core CNC implementation from cnc_clustering.py.
"""

import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnc_clustering as C  # noqa: E402


def build_datasets():
    """Reproduce the five synthetic benchmarks used in the paper."""
    Xa, ya = make_blobs(n_samples=600, centers=4, random_state=0)
    Xa = Xa @ np.array([[0.6, -0.6], [-0.4, 0.8]]) * 0.5
    Xv, yv = make_blobs(
        n_samples=600, centers=4, cluster_std=[0.4, 0.8, 0.3, 1.2],
        random_state=2,
    )
    return {
        "Moons":       make_moons(n_samples=600, noise=0.05, random_state=0),
        "Circles":     make_circles(n_samples=600, factor=0.5, noise=0.05, random_state=0),
        "Blobs":       make_blobs(n_samples=600, centers=4, random_state=0),
        "Aniso":       (Xa, ya),
        "VariedBlobs": (Xv, yv),
    }


def k_sensitivity():
    """ARI of CNC as a function of neighborhood size k (K fixed to truth)."""
    datasets = build_datasets()
    ks = [10, 15, 20, 25, 30, 40]
    order = ["Moons", "Circles", "Blobs", "Aniso", "VariedBlobs"]
    results = {name: {} for name in order}
    for name in order:
        X, y = datasets[name]
        n_true = int(len(np.unique(y)))
        for k in ks:
            A, _, _ = C.resonance_affinity(X, k=k)
            lab = C._spectral_labels(A, n_true)
            results[name][k] = float(adjusted_rand_score(y, lab))
    return ks, order, results


def _ordered_by_label_then_coordinate(X, y):
    """Sort samples by ground-truth label, then within each label by the
    coordinate along the cluster's largest variance direction. This ordering
    exposes the near block-diagonal structure of the resonance affinity.
    """
    idx = []
    for lab in sorted(np.unique(y)):
        lab_idx = np.flatnonzero(y == lab)
        sub = X[lab_idx]
        axis = np.argmax(np.ptp(sub, axis=0))
        order = np.argsort(sub[:, axis])
        idx.extend(lab_idx[order].tolist())
    return np.array(idx)


def affinity_figure():
    """Sorted resonance affinity heatmaps for Blobs and Moons.

    The matrices are re-ordered by ground-truth label and, within each label,
    by the dominant spatial coordinate. Cyan ticks mark cluster boundaries so
    that the (approximate) block-diagonal structure of Theorem~1 is visible.
    """
    datasets = build_datasets()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    panels = [
        ("Four Gaussian blobs ($K{=}4$)", datasets["Blobs"]),
        ("Two moons ($K{=}2$)", datasets["Moons"]),
    ]
    vmax = 0.7  # clips the brightest 1--2% outliers; exposes intra-block bands
    for ax, (title, (X, y)) in zip(axes, panels):
        A, _, _ = C.resonance_affinity(X, k=20)
        Ad = np.asarray(A, dtype=float)
        order = _ordered_by_label_then_coordinate(X, y)
        Ad = Ad[np.ix_(order, order)]
        im = ax.imshow(Ad, cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
        # cluster-boundary ticks
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
    return fig


def main():
    out_dir = os.path.join(HERE, "..", "paper", "figs")
    os.makedirs(out_dir, exist_ok=True)

    ks, order, results = k_sensitivity()
    print("=== CNC ARI vs neighborhood size k (K fixed to ground truth) ===")
    header = "dataset".ljust(12) + "".join(f"k={k:>4}" for k in ks)
    print(header)
    for name in order:
        row = name.ljust(12) + "".join(f"{results[name][k]:5.2f}" for k in ks)
        print(row)

    with open(os.path.join(out_dir, "cnc_ksens.json"), "w") as f:
        json.dump({"ks": ks, "order": order, "results": results}, f, indent=2)

    fig = affinity_figure()
    fig.savefig(os.path.join(out_dir, "cnc_affinity.pdf"), dpi=300)
    print("saved cnc_affinity.pdf and cnc_ksens.json")


if __name__ == "__main__":
    main()
