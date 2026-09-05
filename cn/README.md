# CNC 邻域共识聚类（中文版）— 可独立复现论文包

本文件夹是 **邻域共识聚类（Consensus-Neighborhood Clustering, CNC）** 论文的
**中文投稿版本**，包含**完整、可独立复现**的数据与代码。论文正文为
`paper/main.tex`（已编译为 `paper/main.pdf`）。英文版本位于同级 `../en/`。

---

## 1. 目录结构

```
cn/
├── README.md                  # 本说明
├── paper/
│   ├── main.tex               # 论文 LaTeX 源（中文，XeLaTeX 编译）
│   ├── main.pdf               # 已编译论文
│   └── figs/                  # 论文引用的 8 张图（均可由 code/ 重新生成）
│       ├── fig_idea.pdf           # 思想来源：一阶距离 vs 二阶邻域共识
│       ├── cnc_affinity.pdf       # 共振亲和矩阵块对角结构（定理 1 佐证）
│       ├── cnc_benchmark.pdf      # 3 方法 ARI 对比
│       ├── fig_baseline_compare.pdf  # CNC vs SNN/RBF-spectral/HDBSCAN
│       ├── cnc_synthetic.png      # CNC 在合成数据上的聚类效果
│       ├── ayc_kstab.png          # 亚运村高光谱稳定性选 K 曲线
│       ├── ayc_compare.png        # 亚运村空间分割对比
│       └── ayc_spectra.pdf        # 各簇平均光谱
├── code/
│   ├── cnc_clustering.py      # CNC 核心算法（含 resonance_affinity / spectral_partition）+ 演示图
│   ├── benchmark_expanded.py  # 合成+真实带标基准（CNC vs 8 方法，含 SNN/RBF/HDBSCAN/Digits/Gauss20D）
│   ├── cnc_robustness.py      # 机制/鲁棒性（亲和块结构图 + k 敏感性）
│   ├── ayc_cluster_final.py   # 亚运村高光谱真实实验（生成全部真实数据图与指标）
│   ├── preprocess_ayc.py      # 原始 ENVI → ayc_X.npy（从原始数据复现输入）
│   ├── fig_idea.py            # 生成思想来源概念图
│   └── requirements.txt       # Python 依赖
├── data/
│   ├── ayc_X.npy              # 预处理后的高光谱输入 (40000×64, float64)
│   ├── benchmark_results.json # 合成基准数值（论文表的数据来源，与 code 产物一致）
│   └── ayc_data/亚运村数据/    # 原始 ENVI 数据（yyc200.hdr + yyc200）
└── results/
    ├── ayc_labels.npz         # 真实实验聚类标签（含 K=4/6/10 与 KMeans）
    ├── ayc_paper_metrics.json # 真实数据内部评价指标（论文表 4 的数据来源）
    └── *.png / *.json         # 复现脚本额外生成的补充图与中间产物
```

---

## 2. 环境依赖

**Python（推荐 3.10+，conda 环境 `hologpu` 亦可）：**
```bash
pip install numpy scipy scikit-learn matplotlib
# 可选（仅影响 HDBSCAN 基线行）：
pip install hdbscan
```
> 未安装 `hdbscan` 时，`benchmark_results.json` 中的 **HDBSCAN** 一行会退化为
> 占位（K=0, ARI/AMI=NaN）；其余 7 个方法与全部真实数据结果不受影响、完全可复现。
> 安装后重跑即可补全 HDBSCAN 真实数值。

**LaTeX（用于编译论文，非 pip 包）：** 带有 XeLaTeX 的发行版（MiKTeX / TeX Live），
含 `ctex` 宏集及 `amsmath, amssymb, amsthm, booktabs, array, multirow, tabularx,
graphicx, subcaption, xcolor, algorithm, algpseudocode, hyperref, geometry`。

---

## 3. 独立复现步骤

所有脚本的输出路径相对本文件夹自动解析，**可从任意目录运行**。

### (a) 合成 + 真实带标基准（生成论文表数值，约数分钟）
```bash
python code/benchmark_expanded.py
```
生成 `results/benchmark_results.json`（8 方法 × {5 合成 + Gauss20D + Digits}）与
`paper/figs/fig_baseline_compare.pdf`。合成数据由固定随机种子生成，确定可复现。

### (b) 机制 / 鲁棒性图
```bash
python code/cnc_robustness.py
```
生成 `paper/figs/cnc_affinity.pdf` 与 `paper/figs/cnc_ksens.json`。

### (c) 合成数据聚类演示图
```bash
python code/cnc_clustering.py
```
生成 `paper/figs/cnc_synthetic.png` 与 `paper/figs/cnc_benchmark.pdf`。

### (d) 思想来源概念图
```bash
python code/fig_idea.py
```
生成 `paper/figs/fig_idea.pdf`。

### (e) 亚运村高光谱真实实验（数分钟）
```bash
python code/ayc_cluster_final.py
```
读取 `data/ayc_X.npy`，完成 CNC 谱聚类与稳定性选 K，生成
`paper/figs/{ayc_kstab.png, ayc_compare.png, ayc_spectra.pdf}`、
`results/{ayc_labels.npz, ayc_paper_metrics.json}`，并打印与论文表 4 对应的指标。

### (f) （可选）从原始 ENVI 复现高光谱输入（确定性）
```bash
python code/preprocess_ayc.py
```
读取 `data/ayc_data/亚运村数据/`，写出 `data/ayc_X.npy`。

### (g) 编译论文 PDF
```bash
cd paper && xelatex main.tex && xelatex main.tex && cd ..
```

---

## 4. 关于可复现性

- **数值（表）**：合成基准与真实数据指标分别由 (a)、(e) 确定性复现，输出 JSON 即论
  文表的数据来源；`data/benchmark_results.json` 与 `results/` 下副本一致。
- **图像**：上述 (a)–(e) 覆盖论文引用的全部 8 张图，重跑即再生，无需手工文件。
- **真实数据**：原始 ENVI 随包提供，可经 (f) 重新生成输入；CNC/KMeans 标签与指标由
  (e) 确定性复现。
- **核验**：本包已实测——重跑 (a)(e) 得到的 JSON 与随包 `results/*.json` 逐项一致。

## 5. 投稿前需补充

`paper/main.tex` 作者/单位字段为占位符 `[作者姓名]` 等，请按目标期刊替换；
参考文献为手写 `thebibliography` 条目，可改 BibTeX。
