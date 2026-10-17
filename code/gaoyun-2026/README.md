# CNC（Consensus-Neighborhood Clustering）论文代码整理

将原始投稿包中的代码按三类重新组织，并提供一个总入口 `run_all.py` 及其 Jupyter 版本 `run_all.ipynb`。

## 目录结构（三类 Python 代码）

```
CNC_code_organized/
├── core/                      # ★ 第 1 层 · 主代码（CNC 核心算法）
│   └── cnc.py                 #   resonance_affinity / spectral_partition /
│                              #   stability_select_K / hierarchical_stability_select /
│                              #   cnc_cluster / block_purity / block_recovery
│
├── comparison/                # ★ 第 2 层 · 算法比较 + 数据生成（调用主代码）
│   ├── datasets.py            #   数据集构造 / 加载（合成 + 高光谱）
│   ├── baselines.py           #   7 个基线聚类：KMeans/DBSCAN/DPC/AP/HDBSCAN/SNN/RBF
│   ├── experiments.py         #   主实验驱动：benchmark / k 敏感 / rank 消融 /
│   │                         #     多种子 / 运行时间 / 高光谱 / Indian Pines /
│   │                         #     K 选择 / 协议消融 / 审稿答复（输出 results/*.json）
│   ├── additional_selectors.py#   额外 K 选择基线（Gap / Jump 统计量）
│   ├── cross_affinity_validation.py  # 跨亲和力验证 + margin×floor 网格
│   ├── regen_ayc.py           #   重新生成 AYC 在 K=20 的标签（供 Fig5 使用）
│   ├── reviewer_kmax_sensitivity.py  # 审稿人补充诊断（可选）
│   └── reviewer_sample_size.py       # 审稿人补充诊断（可选）
│
├── visualization/             # ★ 第 3 层 · 根据数据生成图形
│   ├── plotting.py            #   render_* 系列（由 experiments.render_all_figures 调用）
│   └── make_figures.py        #   make_fig1..make_fig5（Fig1–Fig5 期刊级配图）
│
├── run_all.py                 # 主代码 / 总入口：依次调用第 2 层 + 第 3 层
├── run_all.ipynb              # 同一总入口的 Jupyter 版本（单元逐步运行 / 出图）
├── requirements.txt
├── data/                      # 数据集（已随包复制）
├── results/                   # 第 2 层输出（运行后生成 *.json）
└── figures/
    └── paper_figures/         # 第 3 层输出（运行后生成 *.pdf / *.png）
```

## 三层关系

| 层 | 目录 | 作用 | 输入 → 输出 |
|----|------|------|-------------|
| 1 主代码 | `core/` | CNC 算法本身 | `X` → 亲和图 / 标签 / 选择的 K |
| 2 算法比较+数据生成 | `comparison/` | 调用主代码与基线对比 | `data/` → `results/*.json` |
| 3 出图 | `visualization/` | 读取数据画图 | `results/*.json` → `figures/paper_figures/*` |

第 1 层是“被调用”的库；第 2 层调用它；第 3 层消费第 2 层的产物。

## 运行方式

### A. 命令行（一次性跑完）

```bash
cd CNC_code_organized
pip install -r requirements.txt        # 见下：numpy/scipy/scikit-learn/matplotlib/hdbscan
python run_all.py                       # 全流程：第 2 层（含扩展）+ 第 3 层出图（默认 CPU）
python run_all.py --core-only           # 仅核心实验（不含协议消融）
python run_all.py --figures-only        # 仅重新出图（results/ 已存在时）
python run_all.py --resume              # 断点续传：跳过已存在输出的步骤，从断点继续
python run_all.py --gpu                 # 显式启用 GPU（需要可用 CuPy/CUDA）
```

默认后端固定为 CPU（SciPy + scikit-learn），因此导入 `core/cnc.py` 不会
因为机器上恰好安装了 CuPy 而改变结果。论文复现入口使用固定随机种子；
只有明确传入 `--gpu` 或在 Python 中调用 `cnc.set_device("gpu")` 才会启用 GPU。

### 断点续传（`--resume`）

每个实验步骤都会把结果写成 `results/*.json`（出图步骤写成
`figures/paper_figures/*.pdf`）。加了 `--resume` 后，已存在输出的步骤会
**自动跳过**，从第一个未完成的步骤继续，无需重跑已完成的实验。

- 步骤间共享的前置状态（如共振亲和矩阵 `R`）会缓存在
  `results/_state_*.pkl`，因此续跑时不会为了恢复状态而重复计算昂贵的前置实验。
- 适用场景：运行中断（崩溃 / 关机）、某个依赖缺失被跳过、只想补跑最后几步等。
- 例：某步因缺 `hdbscan` 崩溃，先 `pip install hdbscan==0.8.44`，再
  `python run_all.py --resume` 即可从断点继续，前 7 步不会被重跑。

> 说明：若中断发生在旧版（无缓存）代码下、且后续扩展实验依赖该步骤的状态，
> 续跑时可能补算一次该前置实验（一次性，之后缓存生效）。属正常行为。

### B. Jupyter（你用来逐步调用绘制）

在 `CNC_code_organized/` 目录下用 Jupyter 打开 `run_all.ipynb`，
逐单元运行即可：第 1 层演示 → 第 2 层生成数据 → 第 3 层出图。
每个单元独立，可单独重跑某一张图或某一个实验。

- 首次运行把单元里的 `RESUME = False`；中断后续跑改为 `RESUME = True` 即可，
  已完成的步骤自动跳过。
- 也可以只补画某张图，见 Notebook 末尾的示例。

## 依赖

```
numpy, scipy, scikit-learn, matplotlib, hdbscan
```
（`hdbscan` 仅用于 HDBSCAN 基线，失败时会自动跳过，不影响其余结果。）

## 说明

- 原始算法的数学与实现（`core/cnc.py`）未做任何改动，仅新增了跨目录 import 的路径引导。
- 相较原始投稿包，把 `experiments.run_all()` 内部的出图解耦出来，改由总入口显式调用，
  从而让“数据生成（第 2 层）”与“图形生成（第 3 层）”清晰分离。
- 审稿人补充诊断脚本（`reviewer_*`）默认**不**在 `run_all.py` 中运行，需要时手动调用。
