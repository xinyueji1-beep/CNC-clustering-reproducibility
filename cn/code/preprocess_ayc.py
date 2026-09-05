# -*- coding: utf-8 -*-
"""
Preprocess the raw Asian Games Village (亚运村) hyperspectral ENVI cube into the
numpy array consumed by code/ayc_cluster_final.py.

Input  : data/ayc_data/亚运村数据/yyc200.hdr  (ENVI header, BSQ, int16)
         data/ayc_data/亚运村数据/yyc200       (binary pixel data, 200x200x64)
Output : data/ayc_X.npy                       (40000, 64) float64 raw radiance

This step is deterministic -- running it reproduces data/ayc_X.npy exactly.
The downstream experiment keeps the first 54 bands (462-1020 nm) and drops the
noisy 55-64 bands; that selection happens inside ayc_cluster_final.py.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "ayc_data", "亚运村数据")
HDR = os.path.join(RAW_DIR, "yyc200.hdr")
BIN = os.path.join(RAW_DIR, "yyc200")
OUT = os.path.join(ROOT, "data", "ayc_X.npy")


def read_envi_bsq(hdr_path, bin_path):
    """Minimal ENVI BSQ int16 reader driven by the header."""
    with open(hdr_path, encoding="utf-8", errors="ignore") as f:
        hdr = f.read()
    meta = {}
    for ln in hdr.split("\n"):
        ln = ln.strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            meta[k.strip().lower()] = v.strip()
    samples = int(meta["samples"])
    lines_n = int(meta["lines"])
    bands = int(meta["bands"])
    dtype_code = int(meta["data type"])     # 2 => int16
    assert dtype_code == 2, f"expected int16 (code 2), got {dtype_code}"
    raw = np.fromfile(bin_path, dtype="<i2")
    cube = raw.reshape((bands, lines_n, samples))          # BSQ
    X_all = cube.reshape((bands, -1)).T.astype(np.float64)  # (samples*lines, bands)
    return X_all


if __name__ == "__main__":
    X = read_envi_bsq(HDR, BIN)
    np.save(OUT, X)
    print(f"saved {OUT}  shape={X.shape} dtype={X.dtype}")
