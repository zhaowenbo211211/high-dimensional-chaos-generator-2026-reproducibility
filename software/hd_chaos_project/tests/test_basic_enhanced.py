#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
HighDimChaosMapEnhanced experiments: basic iteration, chaos verification,
initial-value sensitivity, bit-rotation mechanism, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tkinter as tk
from tkinter import messagebox, filedialog
import os
from copy import deepcopy

from chaos.core import ComplementaryChaosParams, HighDimChaosMap, HighDimChaosMapEnhanced
from chaos.analysis import verify_chaos, wolf_lyapunov_full, wolf_lyapunov_embedding

# LaTeX rendering
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern', 'DejaVu Serif', 'Times New Roman']


# ========================================================================
# Test 1: Basic iteration
# ========================================================================

def test_enhanced_basic_iteration(n=4, word_size=16):
    print("\n" + "=" * 70)
    print(f"[Test 1] HighDimChaosMapEnhanced basic iteration (n={n}, {word_size}-bit)")
    print("=" * 70)

    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()
    enhanced_map = HighDimChaosMapEnhanced(params, word_size=word_size, seed=42)

    # Init checks
    assert hasattr(enhanced_map, 'iter_count'), "missing iter_count"
    assert enhanced_map.iter_count == 0, \
        f"initial iter_count should be 0, got {enhanced_map.iter_count}"

    # k-sequence check (save/restore to avoid side effects)
    backup = enhanced_map.get_rotation_schedule_state()
    k_seq = enhanced_map.debug_k_sequence(16)
    enhanced_map.set_rotation_schedule_state(backup)

    assert len(k_seq) == 16, "k-sequence length error"
    assert all(1 <= k <= (word_size - 1) for k in k_seq), \
        f"k-sequence out of range: {k_seq}"

    expected_seq = list(range(1, word_size)) + [1]
    assert k_seq == expected_seq, f"k-sequence mismatch: {k_seq}"

    # Iteration test
    steps = 100
    trajectory = enhanced_map.iterate(steps)
    assert trajectory.shape == (steps, n), f"wrong trajectory shape: {trajectory.shape}"
    assert trajectory.min() >= 0, f"min out of range: {trajectory.min()}"
    assert trajectory.max() < (1 << word_size), f"max out of range: {trajectory.max()}"
    assert enhanced_map.iter_count == steps, \
        f"iter_count error: should be {steps}, got {enhanced_map.iter_count}"

    # k-sequence after iteration
    backup2 = enhanced_map.get_rotation_schedule_state()
    k_seq_after = enhanced_map.debug_k_sequence(16)
    enhanced_map.set_rotation_schedule_state(backup2)

    assert all(1 <= k <= (word_size - 1) for k in k_seq_after), \
        f"post-iteration k-sequence out of range: {k_seq_after}"

    start_base = steps % (word_size - 1)
    expected_after = [1 + ((start_base + i) % (word_size - 1)) for i in range(16)]
    assert k_seq_after == expected_after, \
        f"post-iteration k-sequence mismatch: {k_seq_after}"

    print(f"  OK trajectory shape: {trajectory.shape}")
    print(f"  OK state range: [{trajectory.min()}, {trajectory.max()}]")
    print(f"  OK iter_count: {enhanced_map.iter_count}")
    print(f"  OK initial k-sequence: {k_seq}")
    print(f"  OK post-iteration k-sequence: {k_seq_after}")
    print(f"  OK normalized range: [{trajectory.min()/(1<<word_size):.4f}, "
          f"{trajectory.max()/(1<<word_size):.4f}]")

    return trajectory, params


# ========================================================================
# Test 2: Chaos verification (3 methods)
# ========================================================================

def test_enhanced_chaos_verification(n=4, word_size=16):
    print("\n" + "=" * 70)
    print(f"[Test 2] HighDimChaosMapEnhanced chaos verification (n={n}, {word_size}-bit)")
    print("Wolf (full-state + embedding) + Jacobian-QR")
    print("=" * 70)

    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    enhanced_map = HighDimChaosMapEnhanced(params, word_size=word_size, seed=42)
    basic_map = HighDimChaosMap(params, word_size=word_size, seed=42)

    # ---- Wolf full-state ----
    print("\n[2.1] Wolf full-state method...")
    enh_wolf = wolf_lyapunov_full(enhanced_map, max_steps=5000, transient=500)
    basic_wolf = wolf_lyapunov_full(basic_map, max_steps=5000, transient=500)

    print(f"\n{'='*50}")
    print("Enhanced (Wolf full-state):")
    print(f"  max LE = {enh_wolf['lyap_max']:.6f}")
    print(f"  mean growth = {enh_wolf['lyap_mean']:.6f}")
    print(f"  std = {enh_wolf['lyap_std']:.6f}")
    print(f"  renormalizations = {enh_wolf['renormalizations']}")
    print(f"  final distance = {enh_wolf['final_distance']:.6e}")
    print(f"  chaotic: {'PASS' if enh_wolf['is_chaotic'] else 'FAIL'}")

    print(f"\n{'='*50}")
    print("Basic (Wolf full-state):")
    print(f"  max LE = {basic_wolf['lyap_max']:.6f}")
    print(f"  mean growth = {basic_wolf['lyap_mean']:.6f}")
    print(f"  std = {basic_wolf['lyap_std']:.6f}")
    print(f"  renormalizations = {basic_wolf['renormalizations']}")
    print(f"  final distance = {basic_wolf['final_distance']:.6e}")
    print(f"  chaotic: {'PASS' if basic_wolf['is_chaotic'] else 'FAIL'}")

    # ---- Wolf delay embedding ----
    print("\n[2.2] Wolf delay-embedding method (x_n time series)...")
    enh_embed = wolf_lyapunov_embedding(
        enhanced_map, max_steps=5000, transient=500, embed_dim=n, delay=1)
    basic_embed = wolf_lyapunov_embedding(
        basic_map, max_steps=5000, transient=500, embed_dim=n, delay=1)

    print(f"\n{'='*50}")
    print("Enhanced (Wolf embedding):")
    print(f"  max LE = {enh_embed['lyap_max']:.6f}")
    print(f"  mean growth = {enh_embed['lyap_mean']:.6f}")
    print(f"  std = {enh_embed['lyap_std']:.6f}")
    print(f"  renormalizations = {enh_embed['renormalizations']}")
    print(f"  embed dim = {enh_embed['embed_dim']}")
    print(f"  chaotic: {'PASS' if enh_embed['is_chaotic'] else 'FAIL'}")

    print(f"\n{'='*50}")
    print("Basic (Wolf embedding):")
    print(f"  max LE = {basic_embed['lyap_max']:.6f}")
    print(f"  mean growth = {basic_embed['lyap_mean']:.6f}")
    print(f"  std = {basic_embed['lyap_std']:.6f}")
    print(f"  renormalizations = {basic_embed['renormalizations']}")
    print(f"  embed dim = {basic_embed['embed_dim']}")
    print(f"  chaotic: {'PASS' if basic_embed['is_chaotic'] else 'FAIL'}")

    # ---- Jacobian-QR ----
    print("\n[2.3] Jacobian-QR method (Lyapunov spectrum)...")
    enhanced_result = verify_chaos(enhanced_map, steps=3000)
    basic_result = verify_chaos(basic_map, steps=3000)

    print(f"\n{'='*50}")
    print("Enhanced (Jacobian-QR):")
    print(f"  LE spectrum: {enhanced_result['lyap_exps']}")
    print(f"  min LE = {enhanced_result['min_le']:.6f}")
    print(f"  max LE = {enhanced_result['max_le']:.6f}")
    print(f"  mean variance = {np.mean(enhanced_result['variance']):.6f}")
    print(f"  chaotic: {'PASS' if enhanced_result['is_chaotic'] else 'FAIL'}")

    print(f"\n{'='*50}")
    print("Basic (Jacobian-QR):")
    print(f"  LE spectrum: {basic_result['lyap_exps']}")
    print(f"  min LE = {basic_result['min_le']:.6f}")
    print(f"  max LE = {basic_result['max_le']:.6f}")
    print(f"  mean variance = {np.mean(basic_result['variance']):.6f}")
    print(f"  chaotic: {'PASS' if basic_result['is_chaotic'] else 'FAIL'}")

    # ---- Comparison ----
    print(f"\n{'='*50}")
    print("Three-method comparison:")
    print(f"\nEnhanced:")
    print(f"  Wolf full-state:  max LE = {enh_wolf['lyap_max']:.6f}")
    print(f"  Wolf embedding:   max LE = {enh_embed['lyap_max']:.6f}")
    print(f"  Jacobian-QR:      max LE = {enhanced_result['max_le']:.6f}")
    print(f"\nBasic:")
    print(f"  Wolf full-state:  max LE = {basic_wolf['lyap_max']:.6f}")
    print(f"  Wolf embedding:   max LE = {basic_embed['lyap_max']:.6f}")
    print(f"  Jacobian-QR:      max LE = {basic_result['max_le']:.6f}")

    print(f"\n{'='*50}")
    print("Enhancement effect:")
    d_wolf_full = enh_wolf['lyap_max'] - basic_wolf['lyap_max']
    d_wolf_embed = enh_embed['lyap_max'] - basic_embed['lyap_max']
    d_jacobian = enhanced_result['max_le'] - basic_result['max_le']
    print(f"  Wolf full-state:  {d_wolf_full:+.6f} ({'up' if d_wolf_full > 0 else 'down'})")
    print(f"  Wolf embedding:   {d_wolf_embed:+.6f} ({'up' if d_wolf_embed > 0 else 'down'})")
    print(f"  Jacobian:         {d_jacobian:+.6f} ({'up' if d_jacobian > 0 else 'down'})")

    is_chaotic_any = (enh_wolf['is_chaotic'] or enh_embed['is_chaotic']
                      or enhanced_result['is_chaotic'])
    assert is_chaotic_any, "Enhanced map should be chaotic (at least one method)"

    return {
        'wolf_full': enh_wolf, 'wolf_embed': enh_embed, 'jacobian': enhanced_result,
        'basic_wolf_full': basic_wolf, 'basic_wolf_embed': basic_embed,
        'basic_jacobian': basic_result,
    }


# ========================================================================
# Test 3: Initial-value sensitivity
# ========================================================================

def test_enhanced_sensitivity_analysis(n=4, word_size=16, max_steps=100):
    print("\n" + "=" * 70)
    print(f"[Test 3] HighDimChaosMapEnhanced sensitivity analysis (n={n})")
    print("=" * 70)

    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    enhanced1 = HighDimChaosMapEnhanced(params, word_size=word_size, seed=None)
    enhanced2 = HighDimChaosMapEnhanced(params, word_size=word_size, seed=None)

    initial_state = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    perturbed_state = initial_state.copy()
    perturbed_state[0] += 1

    print("=== Initial conditions ===")
    print(f"System 1: {initial_state}")
    print(f"System 2: {perturbed_state}")
    print(f"Initial difference: Delta x_1 = 1  (mod 65536)")

    enhanced1.set_state_from_physical(initial_state)
    enhanced2.set_state_from_physical(perturbed_state)

    distances = []
    log_distances = []

    for step in range(max_steps):
        s1 = enhanced1.next()
        s2 = enhanced2.next()
        d = np.linalg.norm(s1 - s2) / (1 << word_size)
        distances.append(d)
        log_distances.append(np.log10(d + 1e-16))
        if step % 10 == 0 or step < 10:
            print(f"Step {step+1:3d}: distance = {d:.6e}, log10 = {log_distances[-1]:.3f}")

    distances = np.array(distances)
    log_distances = np.array(log_distances)

    print("\n=== Statistical results ===")
    print(f"Initial distance:  d_0 = {distances[0]:.6e}")
    print(f"Final distance:    d_final = {distances[-1]:.6e}")
    print(f"Maximum distance:  d_max = {distances.max():.6e}")
    print(f"Growth factor:     {distances[-1] / (distances[0] + 1e-16):.1e}x")

    if len(log_distances) > 20:
        lyap_est = np.mean(np.diff(log_distances)[:20])
        print(f"Lyapunov estimate: lambda ~ {lyap_est:.4f}")

    return distances, log_distances


# ========================================================================
# Test 4: Bit-rotation mechanism
# ========================================================================

def test_bit_rotation_mechanism(n=4, word_size=16):
    print("\n" + "=" * 70)
    print(f"[Test 4] Bit-rotation mechanism (n={n}, {word_size}-bit)")
    print("=" * 70)

    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()
    enhanced_map = HighDimChaosMapEnhanced(params, word_size=word_size, seed=42)

    # ---- pack/unpack roundtrip ----
    test_states = [
        np.array([0, 0, 0, 0], dtype=np.int64),
        np.array([65535, 65535, 65535, 65535], dtype=np.int64),
        np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        np.random.randint(0, 1 << word_size, size=n, dtype=np.int64),
    ]

    print("\npack/unpack roundtrip:")
    for i, state in enumerate(test_states):
        packed = enhanced_map._pack_state_bits(state)
        unpacked = enhanced_map._unpack_state_bits(packed)
        assert np.array_equal(state, unpacked), f"test {i+1} failed: pack/unpack not inverse"
        print(f"  OK test {i+1}")

    # ---- ROL roundtrip ----
    print("\nROL roundtrip:")
    Z = 0b101100111000
    for k in [1, 2, 5, 10]:
        Z_rot = enhanced_map._rol_total(Z, k)
        Z_back = enhanced_map._rol_total(Z_rot, enhanced_map.total_bits - k)
        assert Z_back == Z, f"ROL roundtrip failed: k={k}"
        print(f"  OK k={k}")

    # ---- k-sequence ----
    print("\nk-sequence generation:")
    assert hasattr(enhanced_map, "iter_count"), "missing iter_count"
    assert enhanced_map.iter_count == 0, \
        f"initial iter_count should be 0, got {enhanced_map.iter_count}"

    backup = enhanced_map.get_rotation_schedule_state()
    k_sequence = enhanced_map.debug_k_sequence(16)
    enhanced_map.set_rotation_schedule_state(backup)

    print(f"  k-sequence: {k_sequence}")
    print(f"  k range: [{min(k_sequence)}, {max(k_sequence)}]")

    assert min(k_sequence) >= 1, "k min should be >= 1"
    assert max(k_sequence) <= word_size - 1, f"k max should be <= {word_size - 1}"
    print("  OK k values in valid range")

    expected_seq = list(range(1, word_size)) + [1]
    assert k_sequence == expected_seq, f"k-sequence mismatch: {k_sequence}"
    print("  OK k-sequence matches periodic pattern [1,2,...,15,1]")

    return True


# ========================================================================
# Test 5: Visualization (base vs enhanced comparison)
# ========================================================================

def test_enhanced_visualization(n=4, word_size=16, save_fig=True):
    print("\n" + "=" * 70)
    print(f"[Test 5] Visualization comparison (n={n}, {word_size}-bit)")
    print("=" * 70)

    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    enhanced_map = HighDimChaosMapEnhanced(params, word_size=word_size, seed=42)
    basic_map = HighDimChaosMap(params, word_size=word_size, seed=42)

    steps = 200
    enhanced_norm = enhanced_map.iterate(steps).astype(np.float64) / (1 << word_size)
    basic_norm = basic_map.iterate(steps).astype(np.float64) / (1 << word_size)

    t_axis = np.arange(steps)

    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Time series comparison (dimension 1, first 100 steps)
    ax1 = fig.add_subplot(2, 2, 1)
    plot_steps = 100
    ax1.plot(t_axis[:plot_steps], basic_norm[:plot_steps, 0], 'b-',
             linewidth=1.5, label='Basic', alpha=0.7)
    ax1.plot(t_axis[:plot_steps], enhanced_norm[:plot_steps, 0], 'r--',
             linewidth=1.5, label='Enhanced', alpha=0.7)
    ax1.set_xlabel(r'Iteration step $t$', fontsize=14)
    ax1.set_ylabel(r'Normalized state $x_1/2^{16}$', fontsize=14)
    ax1.set_title('Time Series Comparison (Dimension 1)', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Phase portrait (dimension 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(basic_norm[:-1, 0], basic_norm[1:, 0], 'b.', markersize=2,
             label='Basic', alpha=0.5)
    ax2.plot(enhanced_norm[:-1, 0], enhanced_norm[1:, 0], 'r,', markersize=2,
             label='Enhanced', alpha=0.5)
    ax2.set_xlabel(r'$x_1(t)$', fontsize=14)
    ax2.set_ylabel(r'$x_1(t+1)$', fontsize=14)
    ax2.set_title('Phase Portrait Comparison (Dimension 1)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, frameon=True)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Lyapunov spectrum comparison
    ax3 = fig.add_subplot(2, 2, 3)
    enhanced_result = verify_chaos(enhanced_map, steps=2000)
    basic_result = verify_chaos(basic_map, steps=2000)

    idx = np.arange(1, n + 1)
    width = 0.35
    ax3.bar(idx - width/2, basic_result['lyap_exps'], width,
            label='Basic', color='blue', alpha=0.7)
    ax3.bar(idx + width/2, enhanced_result['lyap_exps'], width,
            label='Enhanced', color='red', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel(r'Dimension $i$', fontsize=14)
    ax3.set_ylabel(r'Lyapunov exponent $\lambda_i$', fontsize=14)
    ax3.set_title('Lyapunov Spectrum Comparison', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12, frameon=True)
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: Sensitivity comparison
    ax4 = fig.add_subplot(2, 2, 4)

    init_state = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    pert_state = init_state.copy()
    pert_state[0] += 1

    # Basic sensitivity
    basic1 = HighDimChaosMap(params, word_size=word_size, seed=None)
    basic2 = HighDimChaosMap(params, word_size=word_size, seed=None)
    basic1.set_state_from_physical(init_state)
    basic2.set_state_from_physical(pert_state)
    basic_dist = []
    for _ in range(50):
        basic_dist.append(
            np.linalg.norm(basic1.next() - basic2.next()) / (1 << word_size))

    # Enhanced sensitivity
    enh1 = HighDimChaosMapEnhanced(params, word_size=word_size, seed=None)
    enh2 = HighDimChaosMapEnhanced(params, word_size=word_size, seed=None)
    enh1.set_state_from_physical(init_state)
    enh2.set_state_from_physical(pert_state)
    enh_dist = []
    for _ in range(50):
        enh_dist.append(
            np.linalg.norm(enh1.next() - enh2.next()) / (1 << word_size))

    steps_short = np.arange(1, 51)
    ax4.semilogy(steps_short, basic_dist, 'b-o', linewidth=1.5,
                 label='Basic', markersize=4)
    ax4.semilogy(steps_short, enh_dist, 'r-s', linewidth=1.5,
                 label='Enhanced', markersize=4)
    ax4.set_xlabel(r'Iteration step $t$', fontsize=14)
    ax4.set_ylabel(r'Normalized distance (log scale)', fontsize=14)
    ax4.set_title('Sensitivity Analysis Comparison', fontsize=16, fontweight='bold')
    ax4.legend(fontsize=12, frameon=True)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # ---- Interactive save ----
    if save_fig:
        root = tk.Tk()
        root.withdraw()

        response = messagebox.askyesno(
            "Save Figure",
            "Save enhanced visualization?\n\nFormat: EPS (300 DPI)")

        if response:
            file_path = filedialog.asksaveasfilename(
                title="Save Figure As",
                defaultextension=".eps",
                filetypes=[("EPS files", "*.eps"), ("PDF files", "*.pdf"),
                           ("All files", "*.*")],
                initialfile=f"enhanced_comparison_n{n}_w{word_size}.eps")

            if file_path:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.eps':
                    fig.savefig(file_path, format='eps', dpi=300, bbox_inches='tight')
                elif ext == '.pdf':
                    fig.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
                else:
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                print(f"  Figure saved: {file_path}")
                print(f"  Format: {ext.upper()[1:]}, "
                      f"Size: {os.path.getsize(file_path)/1024:.1f} KB")

        root.destroy()

    plt.show()
    return fig


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    n = 4
    word_size = 16

    print("=" * 70)
    print("HighDimChaosMapEnhanced complete test suite")
    print("=" * 70)

    try:
        test_enhanced_basic_iteration(n=n, word_size=word_size)
        test_enhanced_chaos_verification(n=n, word_size=word_size)
        test_enhanced_sensitivity_analysis(n=n, word_size=word_size)
        test_bit_rotation_mechanism(n=n, word_size=word_size)
        test_enhanced_visualization(n=n, word_size=word_size, save_fig=True)

        print("\n" + "=" * 70)
        print("All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
