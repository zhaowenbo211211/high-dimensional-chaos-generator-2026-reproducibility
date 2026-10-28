#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
R1-LY-Complementary demo: parameter synthesis, chaos verification,
LaTeX output, and initial-value sensitivity analysis (butterfly effect).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tkinter as tk
from tkinter import messagebox, filedialog
import os

from chaos.core import ComplementaryChaosParams, HighDimChaosMap
from chaos.analysis import verify_chaos
from chaos.utils import make_F, latex_system_like_figure_full

# LaTeX rendering for publication-quality labels
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern', 'DejaVu Serif', 'Times New Roman']

if __name__ == "__main__":
    n = 4
    word_size = 16

    print("=" * 70)
    print(f"R1-LY-Complementary verification (n={n}, {word_size}-bit fixed-point)")
    print("=" * 70)

    # ---- 1. Parameter synthesis ----
    print("\n[1] Parameter synthesis...")
    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    assert all(params['a_j1'] % 2 == 1), "a_j1 must all be odd"
    assert all(params['a_jn'] % 2 == 0), "a_jn must all be even"

    print(f"   a_j1 (odd):  {params['a_j1']}")
    print(f"   a_jn (even): {params['a_jn']}")
    print(f"   b = {params['b']}, Delta = {params['discriminant']:.3f}")
    print(f"   c = {params['c']:.3f}")
    print(f"   c > 1: {params['c'] > 1}")

    # ---- 2. Initialize map ----
    print("\n[2] Initialize chaotic map...")
    chaos_map = HighDimChaosMap(params, word_size=word_size, seed=42)

    # ---- 3. Chaos verification ----
    print("\n[3] Chaos verification...")
    result = verify_chaos(chaos_map, steps=5000)

    print(f"   Lyapunov exponents: {result['lyap_exps']}")
    print(f"   LE_min = {result['min_le']:.6f}")
    print(f"   LE_max = {result['max_le']:.6f}")
    print(f"   State variance: {result['variance']}")
    print(f"   Chaotic: {'PASS' if result['is_chaotic'] else 'FAIL'}")

    # ---- LaTeX system equations ----
    F = make_F(n, params['a_j1'], params['a_jn'], params['b'])
    print("\n" + latex_system_like_figure_full(F, with_mod=False))

    # ---- 4. Initial-value sensitivity analysis (butterfly effect) ----
    print("\n[4] Initial-value sensitivity analysis (base map, no enhancement)...\n")

    print("=== Chaotic system parameters ===")
    print(f"Dimension: {n}D")
    print(f"a_j1: {params['a_j1']}")
    print(f"a_jn: {params['a_jn']}")
    print(f"b: {params['b']}")
    print()

    chaos1 = HighDimChaosMap(params, word_size=16)
    chaos2 = HighDimChaosMap(params, word_size=16)

    initial_state = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    perturbed_state = initial_state.copy()
    perturbed_state[0] += 1

    print("=== Initial conditions ===")
    print(f"System 1: {initial_state}")
    print(f"System 2: {perturbed_state}")
    print(f"Initial difference: Delta x_1 = 1  (mod 65536)")
    print()

    chaos1.set_state_from_physical(initial_state)
    chaos2.set_state_from_physical(perturbed_state)

    max_steps = 100
    states1, states2, distances, log_distances = [], [], [], []

    print("=== Iteration process ===")
    for i in range(max_steps):
        s1 = chaos1.next()
        s2 = chaos2.next()
        states1.append(s1.copy())
        states2.append(s2.copy())

        d = np.linalg.norm(s1 - s2) / 65536.0
        distances.append(d)
        log_distances.append(np.log10(d + 1e-16))

        if i % 10 == 0 or i < 10:
            print(f"Step {i+1:3d}: distance = {d:.6e}, log10 = {log_distances[-1]:.3f}")

    states1 = np.array(states1)
    states2 = np.array(states2)
    distances = np.array(distances)
    log_distances = np.array(log_distances)
    steps = np.arange(1, max_steps + 1)

    # ---- 5. Visualization ----
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Distance evolution (linear)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(steps, distances, 'b-', linewidth=2, marker='o', markersize=3,
             label='Distance')
    ax1.axhline(y=1.0, color='r', linestyle='--', linewidth=2,
                label=r'Saturation threshold ($d=1$)')
    ax1.set_xlabel(r'Iteration step $t$', fontsize=14)
    ax1.set_ylabel(r'Normalized Euclidean distance $d(t)$', fontsize=14)
    ax1.set_title('Butterfly Effect: Trajectory Separation (Linear Scale)',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, r'$\Delta x_1(0) = 1$', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: Distance evolution (semi-log)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(steps, log_distances, 'g-', linewidth=2, marker='s', markersize=3,
             label=r'$\log_{10}(d(t))$')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5,
                label=r'$\log_{10}(1) = 0$')
    ax2.axhline(y=-4, color='orange', linestyle=':', linewidth=1.5,
                label=r'$\log_{10}(10^{-4})$')
    ax2.set_xlabel(r'Iteration step $t$', fontsize=14)
    ax2.set_ylabel(r'$\log_{10}(d)$', fontsize=14)
    ax2.set_title('Exponential Growth Characteristic (Semi-log Scale)',
                  fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3)

    if len(steps) > 10:
        z = np.polyfit(steps[:30], log_distances[:30], 1)
        p = np.poly1d(z)
        ax2.plot(steps[:30], p(steps[:30]), 'r--', linewidth=2,
                 label=rf'Fit: $\lambda \approx {z[0]:.3f}$')
        ax2.legend(fontsize=10, frameon=True)

    # Subplot 3: First 20 steps dimension-wise comparison
    ax3 = fig.add_subplot(2, 2, 3)
    colors = ['blue', 'green', 'red', 'purple']
    dim_labels = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']

    for dim in range(n):
        ax3.plot(steps[:20], states1[:20, dim] / 65536.0,
                 color=colors[dim], linestyle='-', linewidth=2,
                 label=f'Sys1: {dim_labels[dim]}', alpha=0.8)
        ax3.plot(steps[:20], states2[:20, dim] / 65536.0,
                 color=colors[dim], linestyle='--', linewidth=2,
                 label=f'Sys2: {dim_labels[dim]}', alpha=0.5)

    ax3.set_xlabel(r'Iteration step $t$', fontsize=14)
    ax3.set_ylabel(r'Normalized state $x_i/2^{16}$', fontsize=14)
    ax3.set_title('First 20 Steps: Dimension-wise Trajectory Comparison',
                  fontsize=16, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Growth rate evolution
    ax4 = fig.add_subplot(2, 2, 4)
    growth_rates = np.diff(log_distances)
    ax4.bar(steps[:-1], growth_rates, color='coral', edgecolor='darkred',
            alpha=0.7, label='Growth rate')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)

    if len(growth_rates) > 10:
        lyap_est = np.mean(growth_rates[:20])
        ax4.axhline(y=lyap_est, color='blue', linestyle='--', linewidth=2,
                    label=rf'$\langle\lambda\rangle \approx {lyap_est:.3f}$')

    ax4.set_xlabel(r'Iteration step $t$', fontsize=14)
    ax4.set_ylabel(r'Growth rate per step $\lambda_t$', fontsize=14)
    ax4.set_title('Distance Growth Rate Evolution', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=11, frameon=True)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # ---- 6. Statistics ----
    print("\n=== Statistical analysis ===")
    print(f"Initial distance:      d_0 = {distances[0]:.6e}")
    print(f"Final distance:        d_final = {distances[-1]:.6e}")
    print(f"Maximum distance:      d_max = {distances.max():.6e}")
    print(f"Total growth factor:   {distances[-1] / (distances[0] + 1e-16):.1e}x")

    if len(growth_rates) > 10:
        lyap_mean = np.mean(growth_rates[:20])
        lyap_std = np.std(growth_rates[:20])
        print(f"Lyapunov exponent estimate: lambda = {lyap_mean:.4f} +/- {lyap_std:.4f}")
        print(f"(Positive lambda indicates chaotic behavior)")

    # ---- 7. Interactive save dialog ----
    root = tk.Tk()
    root.withdraw()

    response = messagebox.askyesno(
        "Save Figure",
        "Save the sensitivity analysis figure?\n\n"
        "Format: EPS (vector graphics, publication quality)\n"
        "Resolution: 300 DPI",
        icon=messagebox.QUESTION)

    if response:
        default_filename = "sensitivity_analysis_result.eps"
        file_path = filedialog.asksaveasfilename(
            title="Save Figure As",
            defaultextension=".eps",
            filetypes=[
                ("EPS files", "*.eps"),
                ("PDF files", "*.pdf"),
                ("PNG files", "*.png"),
                ("All files", "*.*")],
            initialfile=default_filename)

        if file_path:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.eps':
                    fig.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
                elif ext == '.pdf':
                    fig.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
                elif ext == '.png':
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                else:
                    fig.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')

                messagebox.showinfo(
                    "Success",
                    f"Figure saved successfully!\n\n"
                    f"File: {file_path}\n"
                    f"Format: {ext.upper()[1:]}\n"
                    f"Size: {os.path.getsize(file_path) / 1024:.1f} KB")
                print(f"\nFigure saved to: {file_path}")
                print(f"  Format: {ext.upper()[1:]}")
                print(f"  Size: {os.path.getsize(file_path) / 1024:.1f} KB")
            except Exception as e:
                messagebox.showerror("Error",
                                     f"Failed to save figure:\n{str(e)}")
                print(f"\nError saving figure: {e}")
        else:
            print("\nSave cancelled by user")
            messagebox.showinfo("Cancelled", "Figure not saved.")
    else:
        print("\nSave declined by user")
        messagebox.showinfo("Cancelled", "Figure not saved.")

    root.destroy()
    plt.show()
