# -*- coding: utf-8 -*-
"""
CNC 论文代码总入口（orchestrator / 主代码）。

本文件把论文代码分成三个清晰的层次，并依次调用：

  第 1 层  主代码  core/cnc.py
            —— CNC 核心算法（resonance_affinity / spectral_partition /
               stability_select_K / hierarchical_stability_select / cnc_cluster
               / block_purity / block_recovery）。本层是“被调用”的算法库，
               不直接在这里运行。

  第 2 层  算法比较 + 数据生成  comparison/*
            —— 调用第 1 层的主代码，与 7 个基线算法做对比，并把结果写成
               results/*.json。 包含：
                 comparison/datasets.py           数据集加载 / 构造
                 comparison/baselines.py          7 个基线聚类算法
                 comparison/experiments.py        主实验驱动（benchmark / k 敏感性 /
                                                 rank 消融 / 多种子 / 运行时间 /
                                                 高光谱 / Indian Pines / K 选择 / 协议消融 / 审稿答复）
                 comparison/additional_selectors.py   额外 K 选择基线（Gap / Jump）
                 comparison/cross_affinity_validation.py  跨亲和力验证 + margin 网格
                 comparison/regen_ayc.py               AYC 标签再生成（供 Fig5 使用）
                 comparison/reviewer_kmax_sensitivity.py / reviewer_sample_size.py
                                                       审稿人补充诊断（可选，不默认运行）

  第 3 层  根据数据生成图形  visualization/*
            —— 读取 results/*.json，绘制并导出 figures/paper_figures/*.{pdf,png}。
                 visualization/plotting.py          render_* 系列（由 experiments.render_all_figures 调用）
                 visualization/make_figures.py      make_fig1..make_fig5（Fig1–Fig5 期刊级配图）

用法
----
    python run_all.py                 # 全流程：第 2 层（含扩展实验）+ 第 3 层出图（默认 CPU）
    python run_all.py --core-only     # 仅核心实验（不含协议消融），仍会出图
    python run_all.py --gpu           # 显式启用 GPU（需要可用 CuPy/CUDA）
    python run_all.py --figures-only  # 仅重新出图（前提是 results/ 已存在数据）

也可以在 Jupyter 中按单元逐步调用（见同目录 run_all.ipynb）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# 路径引导：把三个分类目录加入 sys.path，使彼此可以 import
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
for _cat in ("core", "comparison", "visualization"):
    _p = os.path.join(HERE, _cat)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

ROOT = HERE
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures", "paper_figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)


# ===========================================================================
# 第 2 层：算法比较 + 数据生成
# ===========================================================================
def run_comparison(include_extended: bool = True, resume: bool = False) -> dict:
    """运行全部对比实验，把结果写入 results/*.json。

    ``resume=True`` 时跳过输出已存在的步骤（含下面的补充实验），从断点续跑。
    """
    import experiments

    checks = experiments.run_all(include_extended=include_extended, resume=resume)

    # 额外的审稿人补充实验（生成 cross_affinity_validation.json /
    # margin_sensitivity_expanded.json / additional_selectors.json）。
    import additional_selectors
    import cross_affinity_validation

    if resume and os.path.exists(os.path.join(RESULTS, "additional_selectors.json")):
        print("[run_all] 跳过 additional_selectors（输出已存在）")
    else:
        additional_selectors.main()
    if resume and os.path.exists(os.path.join(RESULTS, "cross_affinity_validation.json")):
        print("[run_all] 跳过 cross_affinity_validation（输出已存在）")
    else:
        cross_affinity_validation.main()
    if resume and os.path.exists(os.path.join(RESULTS, "margin_sensitivity_expanded.json")):
        print("[run_all] 跳过 margin_sensitivity（输出已存在）")
    else:
        cross_affinity_validation.margin_sensitivity()
    return checks


# ===========================================================================
# 第 3 层：根据数据生成图形
# ===========================================================================
def run_figures(regenerate_ayc_labels: bool = True, resume: bool = False) -> None:
    """读取 results/*.json，绘制并导出 figures/paper_figures/*。

    ``resume=True`` 时跳过输出已存在的图（以 *.pdf 为判据），从断点续跑。
    """
    import experiments

    # 3a. plotting.py 系列（benchmark / rank 消融 / affinity / multiseed /
    #     runtime / idea / synthetic / HSI 图等）
    if resume and os.path.exists(os.path.join(FIGS, "cnc_synthetic.pdf")):
        print("[run_all] 跳过 plotting.py 系列图（输出已存在）")
    else:
        experiments.render_all_figures()

    # 3b. make_figures.py 的 Fig1–Fig5 期刊级配图
    import make_figures

    for fname, fn in (("Fig1", make_figures.make_fig1),
                      ("Fig2", make_figures.make_fig2),
                      ("Fig3", make_figures.make_fig3),
                      ("Fig4", make_figures.make_fig4)):
        if resume and os.path.exists(os.path.join(FIGS, f"{fname}.pdf")):
            print(f"[run_all] 跳过 {fname}（输出已存在）")
        else:
            fn()

    # Fig5 依赖 AYC 标签。其文件名使用 results 中记录的 selected K；
    # 若缺失，则先重新生成（耗时，约几分钟）。
    ayc_k = None
    metrics_path = os.path.join(RESULTS, "ayc_paper_metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, encoding="utf-8") as _f:
            ayc_k = int(json.load(_f)["K"])
    ayc_cnc = os.path.join(FIGS, f"ayc_cnc_labels_K{ayc_k}.npy") if ayc_k else ""
    if regenerate_ayc_labels and not os.path.exists(ayc_cnc):
        print("[run_all] Fig5 标签缺失，正在用 regen_ayc 重新生成 AYC 标签 ...")
        import regen_ayc  # noqa: F401  (模块在导入时即完成生成)
    if os.path.exists(ayc_cnc):
        if resume and os.path.exists(os.path.join(FIGS, "Fig5.pdf")):
            print("[run_all] 跳过 Fig5（输出已存在）")
        else:
            make_figures.make_fig5()
    else:
        print("[run_all] 跳过 Fig5（未找到 AYC 标签；可运行 comparison/regen_ayc.py 后再出图）")


# ===========================================================================
# 主入口
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="CNC 论文代码总入口（三层：主代码 / 算法比较 / 出图）")
    ap.add_argument("--core-only", action="store_true",
                    help="仅运行核心实验（不含协议消融），仍会出图")
    ap.add_argument("--figures-only", action="store_true",
                    help="仅重新出图（前提是 results/ 已存在数据）")
    ap.add_argument("--resume", action="store_true",
                    help="断点续传：跳过输出已存在的步骤（数据生成与出图均适用），从断点继续")
    ap.add_argument("--gpu", action="store_true",
                    help="显式启用 GPU；默认始终使用 CPU 参考实现")
    args = ap.parse_args()

    # Import only after parsing so the backend choice is explicit and visible.
    import cnc
    cnc.set_device("gpu" if args.gpu else "cpu")
    print(f"[run_all] Compute backend: {cnc.backend_summary()}")

    t0 = time.time()
    if args.figures_only:
        run_figures(resume=args.resume)
    else:
        run_comparison(include_extended=not args.core_only, resume=args.resume)
        run_figures(resume=args.resume)

    print(f"\n[run_all] 完成，总耗时 {time.time() - t0:.0f}s")
    print(f"[run_all] 数据: {os.path.relpath(RESULTS, ROOT)}/")
    print(f"[run_all] 图形: {os.path.relpath(FIGS, ROOT)}/")


if __name__ == "__main__":
    main()
