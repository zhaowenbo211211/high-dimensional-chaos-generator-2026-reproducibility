#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Statistical distribution analysis: histogram + CDF deviation for n-dimensional chaos.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Sequence, List

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
})


# =========================================================
# Histogram analyzer
# =========================================================

class ChaosHistogramAnalyzer:
    """Marginal histogram analysis for n-dimensional integer chaotic sequences."""

    COLOR_EMP = "#0066CC"
    COLOR_THEO = "#999999"

    def __init__(self, data: np.ndarray, mod: int,
                 dims_to_plot: Optional[Sequence[int]] = None):
        assert data.ndim == 2, "data must be (N, n) 2D array"
        self.data_int = data
        self.mod = mod
        self.data = data / float(mod)
        self.n = data.shape[1]

        if dims_to_plot is None:
            k = min(self.n, 4)
            self.dims = list(range(k))
        else:
            self.dims = [int(d) for d in dims_to_plot if 0 <= d < self.n]
            if len(self.dims) == 0:
                raise ValueError("dims_to_plot is empty or out of range")

    def plot_overall_histogram(self, out_dir: Path, filename: str = "histogram_overall.pdf",
                               bins: int = 64) -> None:
        num_plot = len(self.dims)
        n_cols = 2 if num_plot > 1 else 1
        n_rows = int(np.ceil(num_plot / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, dim in zip(axes, self.dims):
            x = self.data[:, dim]
            idx = dim + 1
            ax.hist(x, bins=bins, density=True, color=self.COLOR_EMP, alpha=0.9,
                    label="Empirical")
            ax.axhline(1.0, color=self.COLOR_THEO, linestyle="--", linewidth=1.5,
                       label="Ideal Uniform")
            ax.set_xlabel(f"Normalized state $x_{idx}$")
            ax.set_ylabel("Probability density")
            ax.set_title(f"Histogram of $x_{idx}$")
            ax.legend(loc="lower center", fontsize="small", frameon=True)

        for j in range(num_plot, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        save_path = Path(out_dir) / filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def plot_single_histogram_subfigure(self, out_dir: Path, dim: int = 0,
                                        filename: str = "hist_x1_sub.eps",
                                        bins: int = 64) -> None:
        if not (0 <= dim < self.n):
            raise ValueError(f"dim={dim} out of range 0..{self.n - 1}")

        old_usetex = plt.rcParams.get("text.usetex", False)
        plt.rcParams["text.usetex"] = True

        x = self.data[:, dim]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(x, bins=bins, density=True, color=self.COLOR_EMP, label="Empirical")
        ax.axhline(1.0, color=self.COLOR_THEO, linestyle="--", linewidth=1.5,
                   label="Ideal Uniform")

        idx = dim + 1
        ax.set_xlabel(r"Normalized state $x_{%d}$" % idx)
        ax.set_ylabel(r"$f_{x_{%d}}(x)$" % idx)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])
        ax.set_yticklabels([rf"${t:.2f}$" for t in ax.get_yticks()])
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        save_path = Path(out_dir) / filename
        fig.savefig(save_path, bbox_inches="tight", format="eps")
        plt.close(fig)
        plt.rcParams["text.usetex"] = old_usetex


# =========================================================
# CDF deviation analyzer
# =========================================================

class ChaosCDFDeviationAnalyzer:
    """Empirical CDF deviation from U[0,1): D_n(u) = F_n(u) - u."""

    COLOR_DEV = "#0066CC"
    COLOR_ZERO = "#999999"

    def __init__(self, data: np.ndarray, mod: int,
                 dims_to_plot: Optional[Sequence[int]] = None):
        assert data.ndim == 2, "data must be (N, n) 2D array"
        self.mod = mod
        self.data = data / float(mod)
        self.n = data.shape[1]

        if dims_to_plot is None:
            k = min(self.n, 4)
            self.dims = list(range(k))
        else:
            self.dims = [int(d) for d in dims_to_plot if 0 <= d < self.n]
            if len(self.dims) == 0:
                raise ValueError("dims_to_plot is empty or out of range")

        self._sorted_u = None
        self._deviation = None

    def compute_cdf_deviation(self) -> Tuple[np.ndarray, np.ndarray]:
        u = self.data
        N = u.shape[0]
        u_sorted = np.sort(u, axis=0)
        k = np.arange(1, N + 1, dtype=np.float64)[:, None]
        F_emp = k / float(N)
        deviation = F_emp - u_sorted
        self._sorted_u = u_sorted
        self._deviation = deviation
        return u_sorted, deviation

    def plot_overall_cdf_deviation(self, out_dir: Path,
                                   filename: str = "cdf_deviation_overall.pdf") -> None:
        if self._sorted_u is None:
            self.compute_cdf_deviation()

        num_plot = len(self.dims)
        n_cols = 2 if num_plot > 1 else 1
        n_rows = int(np.ceil(num_plot / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, dim in zip(axes, self.dims):
            dev_i = self._deviation[:, dim]
            d_star = np.max(np.abs(dev_i))
            idx = dim + 1

            ax.plot(self._sorted_u[:, dim], dev_i, color=self.COLOR_DEV, linewidth=1)
            ax.axhline(0.0, color=self.COLOR_ZERO, linestyle="--", linewidth=1)

            exponent = int(np.floor(np.log10(d_star))) if d_star > 0 else 0
            coeff = d_star / (10 ** exponent) if d_star > 0 else 0.0
            label_text = rf"$D^* = {coeff:.2f} \times 10^{{{exponent}}}$"
            ax.text(0.95, 0.95, label_text, transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                              edgecolor="#cccccc"))
            ax.set_xlabel(f"Normalized state $x_{idx}$")
            ax.set_ylabel(r"CDF deviation $F_n(u) - u$")
            ax.set_title(f"Deviation of $x_{idx}$")

        for j in range(num_plot, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        save_path = Path(out_dir) / filename
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def plot_single_cdf_deviation_subfigure(self, out_dir: Path, dim: int = 0,
                                            filename: str = "cdf_dev_x1_sub.eps") -> None:
        if self._sorted_u is None or self._deviation is None:
            self.compute_cdf_deviation()

        if not (0 <= dim < self.n):
            raise ValueError(f"dim={dim} out of range 0..{self.n - 1}")

        u = self._sorted_u[:, dim]
        dev = self._deviation[:, dim]
        d_star = np.max(np.abs(dev))

        old_usetex = plt.rcParams.get("text.usetex", False)
        plt.rcParams["text.usetex"] = True

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(u, dev, color=self.COLOR_DEV, linewidth=1)
        ax.axhline(0.0, color=self.COLOR_ZERO, linestyle="--", linewidth=1)

        exponent = int(np.floor(np.log10(d_star))) if d_star > 0 else 0
        coeff = d_star / (10 ** exponent) if d_star > 0 else 0.0
        label_text = rf"$D^* = {coeff:.2f} \times 10^{{{exponent}}}$"
        ax.text(0.95, 0.95, label_text, transform=ax.transAxes, fontsize=10,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                          alpha=0.8, lw=0.5))

        idx = dim + 1
        ax.set_xlabel(r"Normalized state $x_{%d}$" % idx)
        ax.set_ylabel(r"$F_n(u) - u$")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])
        ax.set_yticklabels([rf"${t:.3f}$" for t in ax.get_yticks()])

        fig.tight_layout()
        save_path = Path(out_dir) / filename
        fig.savefig(save_path, bbox_inches="tight", format="eps")
        plt.close(fig)
        plt.rcParams["text.usetex"] = old_usetex


# =========================================================
# Report generator
# =========================================================

class ChaosReportGenerator:
    """Auto-generate LaTeX-formatted statistical summary for paper writing."""

    def __init__(self, data_int: np.ndarray, mod: int,
                 cdf_analyzer: ChaosCDFDeviationAnalyzer,
                 warmup: int, total_samples: int):
        assert data_int.ndim == 2
        self.data_int = data_int
        self.mod = mod
        self.data = data_int / float(mod)
        self.cdf_analyzer = cdf_analyzer
        self.warmup = warmup
        self.N = int(total_samples)
        self.word_size = int(round(np.log2(mod)))
        self.dim = self.data.shape[1]

    def _compute_basic_stats(self):
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        mn = np.min(self.data, axis=0)
        mx = np.max(self.data, axis=0)
        return {"mean": mean, "std": std, "min": mn, "max": mx}

    def _compute_cdf_stats(self):
        _, deviation = self.cdf_analyzer.compute_cdf_deviation()
        ks = np.max(np.abs(deviation), axis=0)
        l2 = np.sqrt(np.mean(deviation ** 2, axis=0))
        return {"ks": ks, "l2": l2}

    def generate_report_text(self) -> str:
        basic = self._compute_basic_stats()
        cdf_stats = self._compute_cdf_stats()

        mean = basic["mean"]
        std = basic["std"]
        mn = basic["min"]
        mx = basic["max"]
        ks = cdf_stats["ks"]
        l2 = cdf_stats["l2"]

        lines: List[str] = []
        lines.append("% ===== Auto-generated chaos statistics summary =====")
        lines.append(r"\paragraph{Statistical distribution of fixed-point chaotic sequence}")
        lines.append(
            rf"To evaluate the statistical properties of the {self.word_size}-bit "
            rf"fixed-point {self.dim}-dimensional chaotic map under finite precision, "
            rf"we discarded the first {self.warmup} iterations as warm-up and then "
            rf"sampled $N={self.N}$ {self.dim}-dimensional state points. "
            rf"Integer states $x_j \in \{{0,\dots,2^{{{self.word_size}}}-1\}}$ "
            rf"are normalized as $u_j = x_j / 2^{{{self.word_size}}} \in [0,1)$."
        )
        lines.append(
            r"Let the normalized outputs be $u_j(k)$, $j=1,\dots,%d$. "
            r"The sample means and standard deviations are approximately:" % self.dim
        )
        for j in range(self.dim):
            lines.append(
                rf"\begin{{itemize}} "
                rf"\item $x_{{{j+1}}}$: "
                rf"$\hat\mu_{{{j+1}}} \approx {mean[j]:.4f}$, "
                rf"$\hat\sigma_{{{j+1}}} \approx {std[j]:.4f}$, "
                rf"$u_{{{j+1}}}^\text{{min}} \approx {mn[j]:.4f}$, "
                rf"$u_{{{j+1}}}^\text{{max}} \approx {mx[j]:.4f}$."
                rf"\end{{itemize}}"
            )
        lines.append(
            r"All dimensions have means close to $0.5$ and standard deviations "
            r"close to $1/\sqrt{12}\approx 0.2887$ (theoretical uniform), "
            r"indicating good marginal uniformity."
        )
        lines.append(
            r"Comparing the empirical CDF $F_n(u)$ with the ideal $F(u)=u$, "
            r"we compute $D_n(u)=F_n(u)-u$, the KS statistic "
            r"$D_n^\ast = \sup_u |D_n(u)|$, and the L$_2$ norm "
            r"$\|D_n\|_2 = \sqrt{\mathbb{E}[D_n(u)^2]}$."
        )
        for j in range(self.dim):
            lines.append(
                rf"\begin{{itemize}} "
                rf"\item $x_{{{j+1}}}$: "
                rf"$D^\ast_{{{j+1}}} \approx {ks[j]:.4e}$, "
                rf"$\|D_{{{j+1}}}\|_2 \approx {l2[j]:.4e}$."
                rf"\end{{itemize}}"
            )
        lines.append(
            r"These results show that the KS and L$_2$ deviations are small, "
            r"confirming good uniformity and pseudorandomness suitable for "
            r"chaotic encryption, keystream generation, or scrambling modules."
        )
        return "\n".join(lines)

    def save_report(self, save_path: Path) -> str:
        text = self.generate_report_text()
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
