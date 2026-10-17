# -*- coding: utf-8 -*-
"""Unified dataset construction and loading for every CNC experiment.

Every experiment script previously duplicated the synthetic-dataset builders
(``make_aniso``, ``make_varied``) and the hyperspectral loaders
(``load_indian_pines``, ``load_ayc``).  This module centralises them so that
the whole manuscript is reproduced from one source of truth.

Exposed helpers
---------------
  make_aniso(n, seed)          -- anisotropic Gaussian blobs (2D)
  make_varied(n, seed)         -- varied-variance Gaussian blobs (2D)
  make_gauss20d(n, seed)       -- 20-D Gaussian blobs (5 centres)
  make_digits()                -- sklearn load_digits (X, y)
  build_family(name, seed)     -- regenerate a synthetic family at a data seed
  load_indian_pines()          -- labeled hyperspectral cube -> (X_pca, y)
  load_ayc()                   -- unlabeled 40k-pixel scene -> (X_pca, None)
  load_ayc_bands()             -- raw 54 standardized bands (for spectral metrics)
  synthetic_benchmarks()       -- the five fixed-seed synthetic datasets
"""
from __future__ import annotations

import os

import numpy as np
import scipy.io as sio
from sklearn.datasets import make_moons, make_circles, make_blobs, load_digits

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.dirname(HERE)
DATA_IP = os.path.join(ROOT, "data", "indian_pines")
DATA_AYC = os.path.join(ROOT, "data", "asian_games_village", "ayc_X.npy")


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------
def make_aniso(n=1000, seed=0):
    X, y = make_blobs(n_samples=n, centers=4, random_state=seed)
    X = X.dot([[0.6, -0.6], [-0.4, 0.9]])
    return X, y


def make_varied(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [3, 0], [0, 3], [5, 5]], float)
    stds = [0.3, 0.6, 0.9, 1.2]
    Xs, ys = [], []
    for ci, (c, s) in enumerate(zip(centers, stds)):
        Xs.append(rng.normal(c, s, (n // 4, 2)))
        ys.append(np.full(n // 4, ci))
    return np.vstack(Xs), np.concatenate(ys)


def make_gauss20d(n=3000, seed=0):
    return make_blobs(n_samples=n, centers=5, n_features=20, random_state=seed)


def make_digits():
    return load_digits(return_X_y=True)


def build_family(name, seed):
    """Regenerate a named synthetic family at a given data-generation seed."""
    if name == "Moons":
        return make_moons(n_samples=1000, noise=0.05, random_state=seed)
    if name == "Circles":
        return make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=seed)
    if name == "Blobs":
        return make_blobs(n_samples=1000, centers=4, random_state=seed)
    if name == "Aniso":
        return make_aniso(seed=seed)
    if name == "VariedBlobs":
        return make_varied(seed=seed)
    raise KeyError(name)


def synthetic_benchmarks():
    """The five fixed-seed synthetic datasets used in the main benchmark."""
    return {
        "Moons":       make_moons(n_samples=1000, noise=0.05, random_state=0),
        "Circles":     make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=0),
        "Blobs":       make_blobs(n_samples=1000, centers=4, random_state=0),
        "Aniso":       make_aniso(),
        "VariedBlobs": make_varied(),
    }


# ---------------------------------------------------------------------------
# Hyperspectral loaders
# ---------------------------------------------------------------------------
def _pca_to(Xb, variance=0.99):
    """z-score, then PCA to ``variance`` retained variance; returns components."""
    Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-8)
    U, S, _ = np.linalg.svd(Xb, full_matrices=False)
    var = S ** 2
    n_comp = int(np.searchsorted(np.cumsum(var) / var.sum(), variance) + 1)
    return U[:, :n_comp] * S[:n_comp]


def load_indian_pines():
    """Labeled Indian Pines hyperspectral benchmark -> (X_pca, y)."""
    cube = sio.loadmat(os.path.join(DATA_IP, "Indian_pines_corrected.mat"))["indian_pines_corrected"]
    gt = sio.loadmat(os.path.join(DATA_IP, "Indian_pines_gt.mat"))["indian_pines_gt"]
    mask = gt.ravel() > 0
    X_raw = cube.reshape(-1, cube.shape[2])[mask].astype(np.float64)
    y = gt.ravel()[mask].astype(int)
    return _pca_to(X_raw, 0.99), y


def load_ayc_bands():
    """Asian Games Village: raw 54 retained standardized bands (for metrics)."""
    Xh_bands = np.load(DATA_AYC)[:, :54]
    return (Xh_bands - Xh_bands.mean(0)) / (Xh_bands.std(0) + 1e-8)


def load_ayc():
    """Asian Games Village: PCA to 6 components (unlabeled) -> (X_pca, None)."""
    U, S, _ = np.linalg.svd(load_ayc_bands(), full_matrices=False)
    Xh = U[:, :6] * S[:6]
    return Xh, None
