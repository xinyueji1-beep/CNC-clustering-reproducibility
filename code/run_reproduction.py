"""Recreate the manuscript's result tables, figures, and validation outputs."""
from __future__ import annotations

import json
import argparse
import os
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CODE_DIR.parent
RESULTS = PACKAGE_ROOT / "results"
sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)

import additional_selectors
import cross_affinity_validation
import experiments


def _read_json(name: str):
    with (RESULTS / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_extended_outputs(include_protocol: bool = True) -> dict[str, bool]:
    expected = {
        "Moons", "Circles", "Blobs", "Aniso", "VariedBlobs", "Gauss20D", "Digits"
    }
    additional = _read_json("additional_selectors.json")["datasets"]
    cross = _read_json("cross_affinity_validation.json")["datasets"]
    margins = _read_json("margin_sensitivity_expanded.json")["datasets"]
    reviewer = _read_json("reviewer_response_checks.json")
    default_cell = "m0.1_f0.05"

    checks = {
        "gap_jump_all_seven": set(additional) == expected and all(
            {"gap", "jump"} <= set(additional[name]) for name in expected
        ),
        "cross_affinity_all_seven": set(cross) == expected and all(
            {"CNC", "RBF"} <= set(cross[name]) for name in expected
        ),
        "expanded_margin_grid_all_seven": set(margins) == expected and all(
            default_cell in margins[name][carrier]["grid"]
            for name in expected for carrier in ("CNC", "RBF")
        ),
        "blobs_transfer": (
            cross["Blobs"]["CNC"]["HSS"]["K"] == 4
            and cross["Blobs"]["RBF"]["HSS"]["K"] == 4
        ),
        "digits_transfer": (
            cross["Digits"]["CNC"]["HSS"]["K"] == 9
            and cross["Digits"]["RBF"]["HSS"]["K"] == 8
        ),
        "blob_default_margin": (
            margins["Blobs"]["CNC"]["grid"][default_cell]["K"] == 4
            and margins["Blobs"]["RBF"]["grid"][default_cell]["K"] == 4
        ),
        "digits_default_margin": (
            margins["Digits"]["CNC"]["grid"][default_cell]["K"] == 9
            and margins["Digits"]["RBF"]["grid"][default_cell]["K"] == 8
        ),
        "reviewer_response_checks_complete": all(
            set(reviewer[name]) == {"Moons", "Circles", "Blobs", "Digits"}
            for name in ("boundary_fix", "sensitivity", "one_se")
        ),
    }
    if include_protocol:
        protocol = _read_json("embedding_protocol_ablation.json")
        multiseed = _read_json("embedding_protocol_multiseed.json")
        checks.update({
            "protocol_ablation_all_synthetic": set(protocol["synthetic"]) == expected,
            "protocol_ablation_both_hyperspectral": set(protocol["hyperspectral"])
            == {"IndianPines", "AsianGamesVillage"},
            "protocol_multiseed_complete": (
                multiseed["seeds"] == [0, 1, 2, 3, 4]
                and {"Blobs", "Digits"} <= set(multiseed)
            ),
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"extended reproduction checks failed: {failed}")
    return checks


def smoke_test() -> None:
    import cnc
    import datasets

    X, _ = datasets.synthetic_benchmarks()["Blobs"]
    result = cnc.cnc_cluster(X[:300], k=10, Kmax=4, B=4, seed=0)
    if len(result["labels"]) != 300 or not 2 <= result["K"] <= 4:
        raise AssertionError("small synthetic smoke test returned an invalid result")
    print(
        f"Smoke test passed: K={result['K']}, backend={cnc.backend_summary()}. "
        "No result files were removed or written.",
        flush=True,
    )


def run_reproduction() -> dict[str, bool]:
    started = time.time()
    print(f"Package root: {PACKAGE_ROOT}", flush=True)
    print(f"Compute backend: {experiments.cnc.backend_summary()}", flush=True)
    print("Removing generated results and figures before the full run.", flush=True)
    experiments.clean_outputs()

    checks = experiments.run_all(include_extended=True)
    additional_selectors.main()
    cross_affinity_validation.main()
    cross_affinity_validation.margin_sensitivity()
    checks.update(verify_extended_outputs())

    # This array is only used as an intermediate when rendering the HSI figures.
    (RESULTS / "ayc_labels.npz").unlink(missing_ok=True)
    if not all(checks.values()):
        raise AssertionError(f"reproduction checks failed: {checks}")

    print(f"All {len(checks)} reproduction checks passed.", flush=True)
    print(f"Total runtime: {time.time() - started:.0f} seconds.", flush=True)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="run one small synthetic check without clearing or writing result files",
    )
    parser.add_argument(
        "--core-only", action="store_true",
        help="run the shorter core manuscript set without the protocol ablation",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="use CuPy for sparse eigensolves when a working CUDA runtime is available",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="force the CPU reference backend (also the default)",
    )
    args = parser.parse_args()
    if args.gpu and args.cpu:
        parser.error("--gpu and --cpu cannot be used together")
    if args.gpu:
        experiments.cnc.set_device("gpu")
    else:
        experiments.cnc.set_device("cpu")
    if args.smoke_test:
        smoke_test()
    else:
        if args.core_only:
            started = time.time()
            print(f"Package root: {PACKAGE_ROOT}", flush=True)
            print(f"Compute backend: {experiments.cnc.backend_summary()}", flush=True)
            print("Removing generated results and figures before the core-only run.", flush=True)
            experiments.clean_outputs()
            checks = experiments.run_all(include_extended=False)
            additional_selectors.main()
            cross_affinity_validation.main()
            cross_affinity_validation.margin_sensitivity()
            checks.update(verify_extended_outputs(include_protocol=False))
            (RESULTS / "ayc_labels.npz").unlink(missing_ok=True)
            if not all(checks.values()):
                raise AssertionError(f"reproduction checks failed: {checks}")
            print(f"All {len(checks)} reproduction checks passed.", flush=True)
            print(f"Total runtime: {time.time() - started:.0f} seconds.", flush=True)
        else:
            run_reproduction()
