# CNC 论文 — 中英文双版本可独立复现包

本文件夹包含 **Consensus-Neighborhood Clustering (CNC, 邻域共识聚类)** 论文的
**中文版 (`cn/`)** 与 **英文版 (`en/`)** 两份投稿，二者内容对齐、各自**可独立复现**。

```
CNC_paper/
├── README.md          # 本说明（总览）
├── cn/                # 中文版（XeLaTeX，ctex 文档类）
│   ├── paper/main.tex + main.pdf + figs/
│   ├── code/          # 可复现代码
│   ├── data/          # 原始 ENVI + benchmark_results.json + ayc_X.npy
│   └── results/       # 实验产出（JSON / 标签 / 补充图）
└── en/                # 英文版（pdfLaTeX，CMC/TSP 模板）
    ├── CNC_CMC_submission.tex + .pdf + Definitions/
    ├── figures/paper_figures/
    ├── code/
    ├── data/          # 原始 ENVI + benchmark_results.json + ayc_X.npy
    └── results/
```

## 两个版本的差异

| 项 | 中文 `cn/` | 英文 `en/` |
|---|---|---|
| 编译器 | XeLaTeX (`ctex`) | pdfLaTeX (`tsp.cls`) |
| 参考文献 | GB/T 7714 顺序编码制 | Vancouver 顺序编码制 |
| 图目录 | `paper/figs/` | `figures/paper_figures/` |
| 真实数据图名 | `ayc_compare.png` / `ayc_spectra.pdf` | 同名 |

两版**算法、基准数值、真实数据实验、文献池（≥30 篇、近五年>50%、近三年>33%）完全一致**。

## 复现（各自进入子目录执行）

**中文 `cn/`**：见 `cn/README.md` —— 运行 `benchmark_expanded.py`、`cnc_robustness.py`、
`cnc_clustering.py`、`fig_idea.py`、`ayc_cluster_final.py`（可选 `preprocess_ayc.py`），
再 `cd paper && xelatex main.tex && xelatex main.tex`。

**英文 `en/`**：见 `en/README.md` —— 运行同上 6 个脚本，再
`pdflatex CNC_CMC_submission.tex && pdflatex CNC_CMC_submission.tex`。

## 依赖

```
pip install numpy scipy scikit-learn matplotlib
pip install hdbscan        # 可选：仅影响 HDBSCAN 基线行；缺省时该行退化为占位
```
建议在 conda 环境 `hologpu` 中运行（已验证 numpy/scipy/scikit-learn/matplotlib/hdbscan 齐全）。

## 已验证的可复现性

- 重跑 `benchmark_expanded.py` 与 `ayc_cluster_final.py` 得到的
  `results/*.json` 与随包 JSON **逐项一致**（固定随机种子，确定性）。
- 两版论文引用的 **全部 8 张图** 均可由 `code/` 脚本重新生成（脚本已在文件名上与
  LaTeX 引用一一对应，解决了原包"图存在但脚本不再生对应文件名"的问题）。
- 原始 ENVI 数据已同时包含在 `cn/data/ayc_data/` 与 `en/data/ayc_data/`，二者均可从
  原始数据重新生成 `ayc_X.npy`。

## 投稿前待补（占位符，非复现 blocker）

- 作者/单位/邮箱/GitHub 链接（两版均有 `[...]` 占位）。
- 参考文献可按期刊要求由 `thebibliography` 改用 BibTeX。
