# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo

# scripts/compare_period_stats.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare period length and statistical distribution
for base vs bit-rotation enhanced high-dimensional chaos map.

输出风格：
- 单图：EPS（每维一张），无标题，可带 LaTeX 坐标标签；
- 总图：PDF，包含所有维度子图，并加上标题。
"""

import os
from pathlib import Path
from typing import Tuple
from typing import Any, Dict



import numpy as np
import matplotlib.pyplot as plt

from chaos.R1_LY_Complementary import (
    ComplementaryChaosParams,
    HighDimChaosMap,
    HighDimChaosMapEnhanced,
)
from chaos.analysis.orbit import (
    OrbitStatsConfig,
    analyze_digital_system,
    make_step_fn_from_map,
)


# ----------------------------------------------------------------------
# 绘图辅助
# ----------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_time_and_phase_all_dims(
    samples: np.ndarray,
    system_name: str,
    cfg: OrbitStatsConfig,
    out_single_dir: Path,
    out_pdf_path: Path,
) -> None:
    """
    时间序列 + 相图：
    - 单图 EPS：每个维度一张时间序列 + 一张相图（无标题）
    - 总图 PDF：2 x n 子图，带标题
    """
    n = cfg.n_dims
    T = min(200, samples.shape[0])
    t_axis = np.arange(T)

    _ensure_dir(out_single_dir)
    _ensure_dir(out_pdf_path.parent)

    # --------- 单图 EPS：时间序列 ---------
    for j in range(n):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(t_axis, samples[:T, j], linewidth=0.8)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x_{{{}}}$".format(j + 1))
        # 不加标题
        fig.tight_layout()
        eps_path = out_single_dir / f"{system_name}_ts_dim{j+1}.eps"
        fig.savefig(eps_path, format="eps", bbox_inches="tight")
        plt.close(fig)

    # --------- 单图 EPS：相图 (x_j vs x_{j+1}) ---------
    for j in range(n):
        k = (j + 1) % n
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(samples[:T, j], samples[:T, k], ".", markersize=1)
        ax.set_xlabel(r"$x_{{{}}}$".format(j + 1))
        ax.set_ylabel(r"$x_{{{}}}$".format(k + 1))
        # 不加标题
        fig.tight_layout()
        eps_path = out_single_dir / f"{system_name}_phase_dim{j+1}_vs_{k+1}.eps"
        fig.savefig(eps_path, format="eps", bbox_inches="tight")
        plt.close(fig)

    # --------- 总图 PDF：2 x n 子图 ---------
    fig, axes = plt.subplots(
        2, n, figsize=(3.0 * n, 6.0), squeeze=False, sharex="row"
    )
    fig.suptitle(f"{system_name}: time series and phase portraits")

    # 上排：时间序列
    for j in range(n):
        ax = axes[0, j]
        ax.plot(t_axis, samples[:T, j], linewidth=0.8)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x_{{{}}}$".format(j + 1))
        ax.set_title(r"$x_{{{}}}(t)$".format(j + 1))

    # 下排：相图
    for j in range(n):
        k = (j + 1) % n
        ax = axes[1, j]
        ax.plot(samples[:T, j], samples[:T, k], ".", markersize=1)
        ax.set_xlabel(r"$x_{{{}}}$".format(j + 1))
        ax.set_ylabel(r"$x_{{{}}}$".format(k + 1))
        ax.set_title(r"$x_{{{}}}$ vs. $x_{{{}}}$".format(j + 1, k + 1))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_hist_all_dims(
    samples: np.ndarray,
    system_name: str,
    cfg: OrbitStatsConfig,
    out_single_dir: Path,
    out_pdf_path: Path,
) -> None:
    """
    边缘分布直方图：
    - 单图 EPS：每个维度一张直方图（无标题）
    - 总图 PDF：1 x n 子图，带标题
    """
    n = cfg.n_dims
    bins = cfg.hist_bins

    _ensure_dir(out_single_dir)
    _ensure_dir(out_pdf_path.parent)

    # --------- 单图 EPS ---------
    for j in range(n):
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.hist(
            samples[:, j],
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            histtype="stepfilled",
            alpha=0.7,
        )
        ax.set_xlabel(r"$x_{{{}}}$".format(j + 1))
        ax.set_ylabel("density")
        # 不加标题
        fig.tight_layout()
        eps_path = out_single_dir / f"{system_name}_hist_dim{j+1}.eps"
        fig.savefig(eps_path, format="eps", bbox_inches="tight")
        plt.close(fig)

    # --------- 总图 PDF：1 x n ---------
    fig, axes = plt.subplots(
        1, n, figsize=(3.0 * n, 3.0), squeeze=False, sharey=True
    )
    fig.suptitle(f"{system_name}: marginal distributions")

    for j in range(n):
        ax = axes[0, j]
        ax.hist(
            samples[:, j],
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            histtype="stepfilled",
            alpha=0.7,
        )
        ax.set_xlabel(r"$x_{{{}}}$".format(j + 1))
        if j == 0:
            ax.set_ylabel("density")
        ax.set_title(r"$x_{{{}}}$".format(j + 1))

    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(out_pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)



def _print_marginal_stats(label: str, stats: Dict[str, Any]) -> None:
    """
    打印每一维的均值/方差，便于比较 base / enhanced 在所有坐标上的差异。
    """
    print(f"=== {label} ===")
    print(f"  estimated (mu, lambda) = ({stats['period_mu']}, {stats['period_lambda']})")
    per_dim = stats["per_dim_stats"]
    for j, st in enumerate(per_dim, start=1):
        print(f"  x{j} mean/var = {st['mean']:.4f}, {st['var']:.4f}")
    print()  # 空行分隔






# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    # ---------------- 参数设置（你可以按需改） ----------------
    n = 4
    word_size = 16
    A_odd = 371          # 你现在关心的参数
    B_even = 50
    seed = 42
    max_steps = 50_0000   # 用于统计分布的步数
    transient = 1_000
    hist_bins = 256

    out_root = Path("results") / "period_stats"
    _ensure_dir(out_root)

    # ---------------- 合成参数 ----------------
    print("=== parameter synthesis ===")
    synth = ComplementaryChaosParams(n=n, A_odd=A_odd, B_even=B_even)
    params = synth.synthesize()

    print("  a_j1:", params["a_j1"])
    print("  a_jn:", params["a_jn"])
    print("  b   :", params["b"])
    print()

    # ---------------- 构造原始 & 强化系统 ----------------
    base_map = HighDimChaosMap(params, word_size=word_size, seed=seed)
    enh_map = HighDimChaosMapEnhanced(params, word_size=word_size, seed=seed)

    # 用当前物理状态作为 x0（只是给诊断函数一个 shape 正确的起点）
    x0_base = base_map.get_physical_state()
    x0_enh = enh_map.get_physical_state()

    # 将 .next() 封装成 step_fn
    step_base = make_step_fn_from_map(base_map)
    step_enh = make_step_fn_from_map(enh_map)

    # 数字诊断配置
    cfg = OrbitStatsConfig(
        n_dims=n,
        word_size=word_size,
        max_steps=max_steps,
        transient=transient,
        hist_bins=hist_bins,
        sample_stride=1,
    )

    # ---------------- 诊断：原始系统 ----------------
    print("=== Base system ===")
    stats_base = analyze_digital_system(step_base, x0_base, cfg)
    mu_b = stats_base["period_mu"]
    lam_b = stats_base["period_lambda"]
    print(f"  estimated (mu, lambda) = ({mu_b}, {lam_b})")
    _print_marginal_stats("Base system", stats_base)
    samples_base = stats_base["samples"]

    # ---------------- 诊断：bit-rotation 强化系统 ----------------
    print("\n=== Bit-rotation enhanced system ===")
    stats_enh = analyze_digital_system(step_enh, x0_enh, cfg)
    mu_e = stats_enh["period_mu"]
    lam_e = stats_enh["period_lambda"]
    print(f"  estimated (mu, lambda) = ({mu_e}, {lam_e})")
    _print_marginal_stats("Bit-rotation enhanced system", stats_enh)
    samples_enh = stats_enh["samples"]

    # ---------------- 绘图：原始系统 ----------------
    base_single_dir = out_root / "base_single"
    base_ts_phase_pdf = out_root / "base_ts_phase.pdf"
    base_hist_pdf = out_root / "base_hist.pdf"

    plot_time_and_phase_all_dims(
        samples_base,
        system_name="base",
        cfg=cfg,
        out_single_dir=base_single_dir,
        out_pdf_path=base_ts_phase_pdf,
    )

    plot_hist_all_dims(
        samples_base,
        system_name="base",
        cfg=cfg,
        out_single_dir=base_single_dir,
        out_pdf_path=base_hist_pdf,
    )

    # ---------------- 绘图：强化系统 ----------------
    enh_single_dir = out_root / "enhanced_single"
    enh_ts_phase_pdf = out_root / "enhanced_ts_phase.pdf"
    enh_hist_pdf = out_root / "enhanced_hist.pdf"

    plot_time_and_phase_all_dims(
        samples_enh,
        system_name="enhanced",
        cfg=cfg,
        out_single_dir=enh_single_dir,
        out_pdf_path=enh_ts_phase_pdf,
    )

    plot_hist_all_dims(
        samples_enh,
        system_name="enhanced",
        cfg=cfg,
        out_single_dir=enh_single_dir,
        out_pdf_path=enh_hist_pdf,
    )

    print("\n[Output]")
    print(f"  Base single EPS  : {base_single_dir}")
    print(f"  Base PDF (TS+PH) : {base_ts_phase_pdf}")
    print(f"  Base PDF (HIST)  : {base_hist_pdf}")
    print(f"  Enh. single EPS  : {enh_single_dir}")
    print(f"  Enh. PDF (TS+PH) : {enh_ts_phase_pdf}")
    print(f"  Enh. PDF (HIST)  : {enh_hist_pdf}")


if __name__ == "__main__":
    main()
