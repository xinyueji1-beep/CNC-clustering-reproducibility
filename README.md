# CNC reproducibility package

This repository contains the code, data, experimental results, and generated figures for reproducing the CNC clustering experiments.

- `code/`: CNC implementation, baselines, and experiment scripts
- `data/`: Asian Games Village and Indian Pines inputs
- `results/`: verified experiment outputs
- `figures/`: generated experimental figures

Manuscript source files, manuscript PDFs, LaTeX templates, cover letters, and build files are intentionally excluded.

## Environment

Python 3.10 is recommended. Install dependencies from `code/requirements.txt`.

To verify the installation, run:

`python code/run_reproduction.py --smoke-test`