#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Initial value sensitivity analysis for 16-bit fixed-point chaotic systems.
"""

import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def analyze_initial_value_sensitivity_16bit(
    params: Dict,
    map_cls,
    initial_state: Optional[np.ndarray] = None,
    perturbation_vector: Optional[np.ndarray] = None,
    min_iterations: int = 1000,
    max_iterations: int = 5000
) -> Dict:
    """Analyze initial value sensitivity (butterfly effect).

    Args:
        params: Chaotic map parameters {'a_j1': [...], 'a_jn': [...], 'b': ...}
        map_cls: Map class (HighDimChaosMap or HighDimChaosMapEnhanced)
        initial_state: If None, random initial state is generated
        perturbation_vector: If None, minimal perturbation [1, 0, 0, ...] is used
        min_iterations: Minimum number of iterations
        max_iterations: Maximum number of iterations
    """
    chaos1 = map_cls(params, word_size=16)
    chaos2 = map_cls(params, word_size=16)

    if initial_state is None:
        np.random.seed(42)
        initial_state = np.random.randint(0, 65536, size=chaos1.n, dtype=np.int64)

    if perturbation_vector is None:
        perturbation_vector = np.zeros(chaos1.n, dtype=np.int64)
        perturbation_vector[0] = 1

    perturbed_state = (initial_state + perturbation_vector) % 65536

    chaos1.set_state_from_physical(initial_state)
    chaos2.set_state_from_physical(perturbed_state)

    distances = []
    states1 = []
    states2 = []
    growth_rates = []

    print("=== 16-bit Fixed-Point Chaotic System: Initial Value Sensitivity Analysis ===")
    print(f"System dimension: {chaos1.n}D")
    print(f"Initial state: {initial_state}")
    print(f"Perturbation vector: {perturbation_vector}")
    print(f"Initial perturbation magnitude: {np.linalg.norm(perturbation_vector / 65536.0):.2e}")
    print("-" * 60)

    for i in range(max_iterations):
        state1 = chaos1.next()
        state2 = chaos2.next()

        states1.append(state1.copy())
        states2.append(state2.copy())

        diff = (state1 - state2) / 65536.0
        distance = np.linalg.norm(diff)
        distances.append(distance)

        if i > 0 and distances[i - 1] > 0:
            growth = distance / distances[i - 1]
            growth_rates.append(growth)
        elif i > 0:
            growth_rates.append(1.0)
        else:
            growth_rates.append(0.0)

        if distance > 0.8 and i >= min_iterations:
            if i < 10 or i == min_iterations:
                print(f"Step {i}: distance={distance:.6f} (saturated)")
            break

        if i < 50 and i % 10 == 0:
            print(f"Step {i:3d}: distance={distance:.6f}, growth={growth_rates[-1]:.3f}")
        elif i >= 50 and i % 50 == 0:
            print(f"Step {i:3d}: distance={distance:.6f}, growth={growth_rates[-1]:.3f}")

    while len(distances) < min_iterations:
        state1 = chaos1.next()
        state2 = chaos2.next()
        states1.append(state1.copy())
        states2.append(state2.copy())
        diff = (state1 - state2) / 65536.0
        distance = np.linalg.norm(diff)
        distances.append(distance)
        if len(distances) > 1 and distances[-2] > 0:
            growth_rates.append(distance / distances[-2])
        elif len(distances) > 1:
            growth_rates.append(1.0)
        else:
            growth_rates.append(0.0)

    sensitivity_results = _analyze_sensitivity_characteristics(
        distances, growth_rates, initial_state, perturbation_vector)

    sensitivity_results.update({
        'system_dimension': chaos1.n,
        'word_size': 16,
        'modulus': 65536,
        'initial_state': initial_state,
        'perturbation_vector': perturbation_vector,
        'distances': np.array(distances),
        'growth_rates': np.array(growth_rates),
        'states1': np.array(states1),
        'states2': np.array(states2),
        'min_iterations': min_iterations,
        'max_iterations': max_iterations
    })

    return sensitivity_results


def _analyze_sensitivity_characteristics(
    distances: List[float],
    growth_rates: List[float],
    initial_state: np.ndarray,
    perturbation_vector: np.ndarray
) -> Dict:
    distances = np.array(distances)
    growth_rates = np.array(growth_rates)

    initial_distance = np.linalg.norm(perturbation_vector / 65536.0)
    final_distance = distances[-1] if len(distances) > 0 else 0
    total_growth = final_distance / initial_distance if initial_distance > 0 else 0

    if len(growth_rates) > 2:
        mid_start = max(1, len(growth_rates) // 4)
        mid_end = min(len(growth_rates), 3 * len(growth_rates) // 4)
        avg_growth_rate = np.mean(growth_rates[mid_start:mid_end])
    else:
        avg_growth_rate = np.mean(growth_rates) if len(growth_rates) > 0 else 0

    saturation_step_05 = None
    saturation_step_08 = None
    for i, d in enumerate(distances):
        if d > 0.5 and saturation_step_05 is None:
            saturation_step_05 = i
        if d > 0.8 and saturation_step_08 is None:
            saturation_step_08 = i

    first_growth = distances[1] / distances[0] if len(distances) >= 2 and distances[0] > 0 else 0

    return {
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'total_growth_factor': total_growth,
        'average_growth_rate': avg_growth_rate,
        'saturation_step_05': saturation_step_05,
        'saturation_step_08': saturation_step_08,
        'max_distance': np.max(distances) if len(distances) > 0 else 0,
        'min_distance': np.min(distances) if len(distances) > 0 else 0,
        'mean_distance': np.mean(distances) if len(distances) > 0 else 0,
        'std_distance': np.std(distances) if len(distances) > 0 else 0,
        'first_growth': first_growth,
        'num_iterations': len(distances)
    }


def plot_sensitivity_analysis(analysis_results: Dict,
                              save_path: Optional[str] = None,
                              show_first_n: Optional[int] = 200):
    """Plot comprehensive sensitivity analysis with 9 subplots."""
    distances = analysis_results['distances']
    growth_rates = analysis_results['growth_rates']

    if show_first_n is not None and show_first_n < len(distances):
        display_distances = distances[:show_first_n]
        display_growth_rates = growth_rates[:show_first_n]
        display_label = f" (First {show_first_n} steps)"
    else:
        display_distances = distances
        display_growth_rates = growth_rates
        display_label = f" (Total {len(distances)} steps)"

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig = plt.figure(figsize=(16, 12))
    _plot_sensitivity_subplots(fig, analysis_results, display_distances,
                               display_growth_rates, display_label)

    if save_path:
        _save_figure_and_report(fig, analysis_results, save_path, len(distances))

    plt.show()
    return analysis_results


def _plot_sensitivity_subplots(fig, results, distances, growth_rates, label):
    # Subplot 1: Linear distance
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(distances, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax1.axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='Threshold (0.8)')
    if results['saturation_step_05'] is not None:
        ax1.axvline(x=results['saturation_step_05'], color='g', linestyle=':',
                    alpha=0.7, label=f"Saturation ({results['saturation_step_05']})")
    ax1.set_xlabel('Iteration Step t')
    ax1.set_ylabel('Normalized Distance d(t)')
    ax1.set_title(f'Distance Evolution{label}')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Semi-log
    ax2 = plt.subplot(3, 3, 2)
    ax2.semilogy(distances, 'g-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Iteration Step t')
    ax2.set_ylabel('Distance d(t) (log scale)')
    ax2.set_title(f'Exponential Growth Test{label}')
    ax2.grid(True, alpha=0.3, which='both')

    # Subplot 3: First 50 steps
    ax3 = plt.subplot(3, 3, 3)
    n_detail = min(50, len(distances))
    ax3.plot(distances[:n_detail], 'b-o', linewidth=1.5, markersize=3, alpha=0.7)
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    if results['saturation_step_05'] is not None and results['saturation_step_05'] < n_detail:
        ax3.axvline(x=results['saturation_step_05'], color='g', linestyle=':',
                    alpha=0.7)
    ax3.set_xlabel('Iteration Step t')
    ax3.set_ylabel('Normalized Distance d(t)')
    ax3.set_title(f'Initial Response (First {n_detail} steps)')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Growth rate distribution
    ax4 = plt.subplot(3, 3, 4)
    if len(growth_rates) > 2:
        filtered = growth_rates[1:]
        filtered = filtered[np.isfinite(filtered)]
        if len(filtered) > 0:
            log_rates = np.log10(filtered[filtered > 0])
            ax4.hist(log_rates, bins=30, alpha=0.7, color='orange', edgecolor='black')
            mean_log = np.mean(log_rates) if len(log_rates) > 0 else 0
            ax4.axvline(x=mean_log, color='b', linestyle='-',
                        label=f'Mean log: {mean_log:.2f}')
            ax4.axvline(x=0, color='r', linestyle=':', label='Growth = 1')
            ax4.set_xlabel('log10(Growth Rate)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Growth Rate Distribution')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)

    # Subplot 5: Cumulative growth
    ax5 = plt.subplot(3, 3, 5)
    if len(distances) > 0 and distances[0] > 0:
        cumulative = distances / distances[0]
        ax5.semilogy(cumulative, 'purple', linewidth=1, alpha=0.7)
        ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax5.axhline(y=10.0, color='orange', linestyle=':', alpha=0.5, label='10x growth')
        ax5.axhline(y=100.0, color='red', linestyle=':', alpha=0.5, label='100x growth')
        ax5.set_xlabel('Iteration Step t')
        ax5.set_ylabel('Cumulative Growth d(t)/d(0) (log scale)')
        ax5.set_title(f'Total Growth Factor: {cumulative[-1]:.1e}')
        ax5.legend(fontsize=7, loc='lower right')
        ax5.grid(True, alpha=0.3, which='both')

    # Subplot 6: Post-saturation
    ax6 = plt.subplot(3, 3, 6)
    if results['saturation_step_05'] is not None:
        sat_idx = results['saturation_step_05']
        if sat_idx < len(distances) - 10:
            post_sat = distances[sat_idx:]
            ax6.plot(post_sat, 'b-', linewidth=1, alpha=0.7)
            ax6.axhline(y=np.mean(post_sat), color='r', linestyle='--',
                        label=f'Mean: {np.mean(post_sat):.4f}')
            ax6.set_xlabel('Steps after saturation')
            ax6.set_ylabel('Distance')
            ax6.set_title(f'Post-saturation ({len(post_sat)} steps)')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)

    # Subplot 7: Growth rate evolution
    ax7 = plt.subplot(3, 3, 7)
    if len(growth_rates) > 1:
        steps = np.arange(1, len(growth_rates))
        ax7.plot(steps, growth_rates[1:], 'b-', linewidth=1, alpha=0.7)
        ax7.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Growth = 1')
        ax7.set_xlabel('Iteration Step')
        ax7.set_ylabel('Growth Rate r_t')
        ax7.set_title('Growth Rate Evolution')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)

    # Subplot 8: Distance distribution
    ax8 = plt.subplot(3, 3, 8)
    if len(distances) > 1:
        ax8.hist(distances, bins=30, alpha=0.7, color='teal', edgecolor='black')
        ax8.axvline(x=np.mean(distances), color='r', linestyle='--',
                    label=f'Mean: {np.mean(distances):.4f}')
        ax8.set_xlabel('Normalized Distance')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Distance Distribution')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3, axis='y')

    # Subplot 9: Info panel
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    info_text = (
        "SYSTEM INFORMATION\n"
        f"Dimension: {results['system_dimension']}D\n"
        f"Word size: {results['word_size']}-bit\n"
        f"Modulus: 2^{results['word_size']} = {results['modulus']:,}\n\n"
        "SENSITIVITY METRICS\n"
        f"Initial distance: {results['initial_distance']:.2e}\n"
        f"Final distance: {results['final_distance']:.4f}\n"
        f"Total growth: {results['total_growth_factor']:.1e}x\n"
        f"First step growth: {results['first_growth']:.1f}x\n"
        f"Avg growth rate: {results['average_growth_rate']:.3f}\n\n"
        "DIVERGENCE SPEED\n"
        f"Saturation @0.5: {results['saturation_step_05'] if results['saturation_step_05'] is not None else 'Not reached'}\n"
        f"Saturation @0.8: {results['saturation_step_08'] if results['saturation_step_08'] is not None else 'Not reached'}\n\n"
        "STATISTICS\n"
        f"Mean distance: {results['mean_distance']:.4f}\n"
        f"Std distance: {results['std_distance']:.4f}\n"
        f"Min distance: {results['min_distance']:.4f}\n"
        f"Max distance: {results['max_distance']:.4f}\n"
        f"Total iterations: {len(distances)}"
    )
    ax9.text(0.05, 0.95, info_text, fontfamily='monospace', fontsize=7.5,
             verticalalignment='top', transform=ax9.transAxes)

    plt.suptitle('Initial Value Sensitivity Analysis: 16-bit Fixed-Point Chaotic System',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()


def _save_figure_and_report(fig, results, save_path, n_iterations):
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sensitivity analysis figure saved to: {save_path}")

    report_path = save_path.rsplit('.', 1)[0] + '_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== 16-bit Chaotic System Sensitivity Analysis Report ===\n\n")
        f.write(f"Total iterations: {n_iterations}\n")
        f.write(f"Initial distance: {results['initial_distance']:.2e}\n")
        f.write(f"Final distance: {results['final_distance']:.4f}\n")
        f.write(f"Total growth factor: {results['total_growth_factor']:.1e}\n")
        f.write(f"First step growth: {results['first_growth']:.1f}\n")
        f.write(f"Average growth rate: {results['average_growth_rate']:.3f}\n")
    print(f"Analysis report saved to: {report_path}")
