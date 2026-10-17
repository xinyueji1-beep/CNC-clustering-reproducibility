"""Conditional sample-size viability check requested by Reviewer 2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parent / "visualization"))

import cnc


def main() -> None:
    centers = np.array([[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]])
    per_cluster_values = [10, 20, 40, 80]
    noise_values = [0.8, 1.5, 2.5]
    # This is a compact diagnostic, not a power study: five independent data
    # seeds and the same fixed algorithm budget are enough to expose gross
    # sample-size/separation dependence without implying inferential precision.
    seeds = list(range(5))
    rows = {}
    for noise in noise_values:
        rows[str(noise)] = {}
        for n_per in per_cluster_values:
            records = []
            for seed in seeds:
                X, y = make_blobs(
                    n_samples=4 * n_per,
                    centers=centers,
                    cluster_std=noise,
                    random_state=seed,
                )
                algorithm_seed = 1000 + seed
                # Keep the neighborhood local as N changes; k=20 would be
                # half of the smallest sample and would confound the study.
                neighborhood_k = min(max(5, int(np.sqrt(len(X)))), len(X) - 1)
                result = cnc.cnc_cluster(
                    X,
                    k=neighborhood_k,
                    Kmax=6,
                    B=10,
                    frac=0.8,
                    seed=algorithm_seed,
                )
                ari = float(adjusted_rand_score(y, result["labels"]))
                records.append({
                    "data_seed": seed,
                    "algorithm_seed": algorithm_seed,
                    "k": neighborhood_k,
                    "selected_K": int(result["K"]),
                    "ARI": round(ari, 4),
                    "correct_K": bool(result["K"] == 4),
                    "ARI_ge_0.8": bool(ari >= 0.8),
                })
            rows[str(noise)][str(n_per)] = {
                "n_total": 4 * n_per,
                "n_repetitions": len(records),
                "correct_K_rate": round(float(np.mean([r["correct_K"] for r in records])), 3),
                "ARI_ge_0.8_rate": round(float(np.mean([r["ARI_ge_0.8"] for r in records])), 3),
                "ARI_mean": round(float(np.mean([r["ARI"] for r in records])), 4),
                "ARI_sd": round(float(np.std([r["ARI"] for r in records], ddof=1)), 4),
                "records": records,
            }
    out = {
        "protocol": {
            "centers": centers.tolist(),
            "cluster_std_values": noise_values,
            "Kmax": 6,
            "B": 10,
            "frac": 0.8,
            "neighborhood_rule": "min(max(5, floor(sqrt(N))), N-1)",
            "seeds": seeds,
            "precision_target": "descriptive target only: ARI >= 0.8 and correct K in at least 80% of repetitions",
            "interpretation": "Conditional four-component, two-dimensional benchmark; not a universal threshold. Each data seed is paired with a distinct prespecified algorithm seed.",
        },
        "per_cluster": rows,
    }
    path = ROOT / "results" / "reviewer_sample_size.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
