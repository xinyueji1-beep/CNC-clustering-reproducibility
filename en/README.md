# CNC — CMC/TSP journal template submission (English, reproducible)

This folder is the **English submission** of the **Consensus-Neighborhood
Clustering (CNC)** paper, formatted for the **Computers, Materials & Continua
(CMC) / TSP** journal class. The Chinese version lives in `../cn/`.

Template source: `MBC_CMC_submission_source.zip`. We reuse its document class
(`Definitions/tsp.cls`), section structure, algorithm environment, and the
**Vancouver numeric** (`thebibliography` + `\doilink` DOI) reference style.

---

## 1. Directory layout

```
en/
├── CNC_CMC_submission.tex     # paper source (English, pdfLaTeX)
├── CNC_CMC_submission.pdf     # compiled paper
├── Definitions/               # journal class & packages (tsp.cls etc.)
├── figures/paper_figures/     # 8 figures referenced by the paper
│   ├── cnc_idea.pdf               # idea origin
│   ├── cnc_affinity.pdf           # resonance affinity block structure (Thm 1)
│   ├── cnc_benchmark.pdf          # 3-method ARI bars
│   ├── fig_baseline_compare.pdf   # CNC vs SNN/RBF-spectral/HDBSCAN
│   ├── cnc_synthetic.png          # CNC clustering on synthetic data
│   ├── ayc_kstab.png              # stability-selection curve
│   ├── ayc_compare.png            # Ayuncun spatial segmentation
│   └── ayc_spectra.pdf            # cluster-mean spectra
├── code/
│   ├── cnc_clustering.py          # CNC core (resonance_affinity / spectral_partition) + demo figs
│   ├── benchmark_expanded.py      # synthetic + Digits benchmark (CNC vs 8 methods)
│   ├── cnc_robustness.py          # mechanism/robustness (affinity block + k-sensitivity)
│   ├── ayc_cluster_final.py       # Ayuncun hyperspectral real experiment
│   ├── preprocess_ayc.py          # raw ENVI -> ayc_X.npy
│   ├── fig_idea.py                # idea-origin concept figure
│   └── requirements.txt
├── data/
│   ├── ayc_X.npy                  # preprocessed hyperspectral input (40000×64)
│   ├── benchmark_results.json     # synthetic benchmark numbers (source of truth for tables)
│   └── ayc_data/亚运村数据/         # raw ENVI (yyc200.hdr + yyc200)
└── results/
    ├── ayc_labels.npz             # real-experiment labels (K=4/6/10 + KMeans)
    └── ayc_paper_metrics.json     # real-data internal metrics (Table real)
```

---

## 2. Reproduce

Use the `hologpu` env (or any Python with numpy/scipy/scikit-learn/hdbscan/matplotlib):

```bash
python code/benchmark_expanded.py   # synthetic+Digits benchmark (Tables 2/3, baseline, digits, 2 comparison figs)
python code/cnc_robustness.py      # mechanism/robustness figs (cnc_affinity.pdf + cnc_ksens.json)
python code/cnc_clustering.py      # demo figs referenced by the paper: cnc_synthetic.png, cnc_benchmark.pdf
python code/fig_idea.py            # idea figure (also emits cnc_idea.pdf used by the tex)
python code/ayc_cluster_final.py   # Ayuncun real experiment (Table real, ayc_kstab/compare/spectra figs)
python code/preprocess_ayc.py      # optional: regenerate ayc_X.npy from raw ENVI
```

> **hdbscan** (optional, `pip install hdbscan`) is required only for the HDBSCAN
> baseline row; without it that row degrades to a placeholder (K=0), everything else
> reproduces exactly. **fig_idea.py also writes `cnc_idea.pdf`** (the name the tex
> includes), and **ayc_cluster_final.py also writes `ayc_compare.png` / `ayc_spectra.pdf`**,
> so re-running the scripts regenerates every figure the paper includes.

Compile the PDF with **pdfLaTeX** (the class needs `[T1]{fontenc}` + `libertine`, not XeLaTeX):

```bash
pdflatex CNC_CMC_submission.tex
pdflatex CNC_CMC_submission.tex
```

All scripts resolve paths relative to themselves and run from any directory.

---

## 3. Verification

Re-running `benchmark_expanded.py` and `ayc_cluster_final.py` reproduces the
shipped `results/*.json` entry-by-entry (fixed random seeds). This package was
verified to do so.

## 4. Placeholders to fill before submission

- Authors/affiliations: `\Author`, `\address`, `\corres` (`[First Author]` etc.).
- Dates: `\datereceived`, `\dateaccepted`.
- Repo link in Acknowledgments: `https://github.com/[your-account]/cnc-clustering`.
- A few classic references (e.g. `macqueen1967`) omit `\doilink` (no DOI).
