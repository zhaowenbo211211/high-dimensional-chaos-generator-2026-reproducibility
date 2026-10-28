#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Verification visualization: time series, phase portraits, Lyapunov spectrum,
and chaos overview figures (EPS single + PDF overview).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from chaos.core.params import ComplementaryChaosParams
from chaos.core.base_map import HighDimChaosMap
from chaos.analysis.lyapunov import verify_chaos
from chaos.utils.sequence import generate_binary_sequence


def run_complete_verification(n: int = 8, word_size: int = 16):
    """Run complete verification and generate EPS/PDF figures."""
    print("=" * 70)
    print(f"R1-LY-Complementary verification (n={n}, {word_size}-bit fixed-point)")
    print("=" * 70)

    print("\n[1] Parameter synthesis...")
    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    assert all(params['a_j1'] % 2 == 1), "a_j1 must be odd"
    assert all(params['a_jn'] % 2 == 0), "a_jn must be even"

    print(f"   a_j1 (odd): {params['a_j1']}")
    print(f"   a_jn (even): {params['a_jn']}")
    print(f"   b = {params['b']}, Delta = {params['discriminant']:.3f}")

    print("\n[2] Initialize chaotic map...")
    chaos_map = HighDimChaosMap(params, word_size=word_size, seed=42)

    print("\n[3] Chaos verification...")
    result = verify_chaos(chaos_map, steps=3000)

    print(f"   Lyapunov exponents: {result['lyap_exps']}")
    print(f"   min LE = {result['min_le']:.6f}")
    print(f"   max LE = {result['max_le']:.6f}")
    print(f"   state variance: {result['variance']}")
    print(f"   chaotic: {'YES' if result['is_chaotic'] else 'NO'}")

    if not result['is_chaotic']:
        return result

    print("\n[4] Generating figures (EPS + PDF)...")

    base_dir = os.path.join("figures", "chaos_verification")
    out_dir = os.path.join(base_dir, f"n{n}_w{word_size}")
    os.makedirs(out_dir, exist_ok=True)

    chaos_map = HighDimChaosMap(params, word_size=word_size, seed=42)
    steps_plot = 201
    traj = chaos_map.iterate(steps_plot)
    traj_norm = traj.astype(np.float64) / chaos_map.mod

    t_axis = np.arange(200)
    x_t = traj_norm[:200, :]
    x_tp1 = traj_norm[1:201, :]

    # Single EPS: time series + phase portrait per dimension
    for j in range(n):
        fig_ts, ax_ts = plt.subplots(figsize=(4, 3))
        ax_ts.plot(t_axis, x_t[:, j], linewidth=0.8)
        ax_ts.set_xlabel(r"$t$")
        ax_ts.set_ylabel(r"$x_{%d}$" % (j + 1))
        fig_ts.tight_layout()
        ts_path = os.path.join(out_dir, f"time_series_x{j+1}_n{n}_w{word_size}.eps")
        fig_ts.savefig(ts_path, format="eps", dpi=300, bbox_inches="tight")
        plt.close(fig_ts)

        fig_ph, ax_ph = plt.subplots(figsize=(4, 3))
        ax_ph.plot(x_t[:, j], x_tp1[:, j], ",", alpha=0.6)
        ax_ph.set_xlabel(r"$x_{%d}(t)$" % (j + 1))
        ax_ph.set_ylabel(r"$x_{%d}(t+1)$" % (j + 1))
        fig_ph.tight_layout()
        ph_path = os.path.join(out_dir, f"phase_x{j+1}_n{n}_w{word_size}.eps")
        fig_ph.savefig(ph_path, format="eps", dpi=300, bbox_inches="tight")
        plt.close(fig_ph)

    le_vals = np.asarray(result['lyap_exps'], dtype=np.float64)
    fig_le, ax_le = plt.subplots(figsize=(4, 3))
    idx = np.arange(1, len(le_vals) + 1)
    ax_le.bar(idx, le_vals)
    ax_le.set_xlabel(r"$i$")
    ax_le.set_ylabel(r"$\lambda_i$")
    fig_le.tight_layout()
    le_path = os.path.join(out_dir, f"lyapunov_spectrum_n{n}_w{word_size}.eps")
    fig_le.savefig(le_path, format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig_le)

    # PDF overview: 3 rows x n cols
    rows, cols = 3, n
    fig_all, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for j in range(n):
        axes[0, j].plot(t_axis, x_t[:, j], linewidth=0.8)
        axes[0, j].set_xlabel(r"$t$")
        axes[0, j].set_ylabel(r"$x_{%d}$" % (j + 1))

    for j in range(n):
        axes[1, j].plot(x_t[:, j], x_tp1[:, j], ",", alpha=0.6)
        axes[1, j].set_xlabel(r"$x_{%d}(t)$" % (j + 1))
        axes[1, j].set_ylabel(r"$x_{%d}(t+1)$" % (j + 1))

    axes[2, 0].bar(idx, le_vals)
    axes[2, 0].set_xlabel(r"$i$")
    axes[2, 0].set_ylabel(r"$\lambda_i$")
    for j in range(1, cols):
        axes[2, j].axis("off")

    fig_all.tight_layout()
    pdf_path = os.path.join(out_dir, f"chaos_overview_n{n}_w{word_size}.pdf")
    fig_all.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_all)

    print(f"   Single EPS figures and overview PDF saved in: {out_dir}")
    return result
