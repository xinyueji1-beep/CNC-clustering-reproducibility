"""Targeted K_max sensitivity checks for the revision response.

The study keeps all CNC and selector settings fixed and changes only the
candidate upper bound.  It is intentionally small enough to run as a review
diagnostic rather than replacing the main benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parent / "visualization"))

import cnc
import datasets as ds


def run_one(X, y, bounds, rounds):
    R, _, _ = cnc.resonance_affinity(X, k=min(20, len(X) - 1))
    result = {}
    for Kmax in bounds:
        K, stability, _ = cnc.hierarchical_stability_select(
            R, Kmax=Kmax, B=rounds, frac=0.8, seed=0
        )
        labels = cnc.spectral_partition(R, K, seed=0)
        argmax = max(stability, key=stability.get)
        result[str(Kmax)] = {
            "selected_K": int(K),
            "argmax_K": int(argmax),
            "argmax_at_boundary": bool(argmax == Kmax),
            "ARI": round(float(adjusted_rand_score(y, labels)), 4),
        }
    return result


def main() -> None:
    synthetic = ds.synthetic_benchmarks()
    out = {
        "protocol": {
            "seed": 0,
            "frac": 0.8,
            "Blobs_rounds": 10,
            "Digits_rounds": 5,
            "note": "Compact diagnostic: only Kmax changes; all other settings are fixed.",
        },
        "Blobs": run_one(synthetic["Blobs"][0], synthetic["Blobs"][1],
                          [6, 8, 10, 12, 15], 10),
        "Digits": run_one(*ds.make_digits(), [8, 10, 12, 15, 18, 20], 5),
    }
    path = ROOT / "results" / "reviewer_kmax_sensitivity.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
