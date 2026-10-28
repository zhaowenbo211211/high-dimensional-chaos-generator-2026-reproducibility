#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Individual sensitivity analysis plots for publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


def setup_latex_rendering():
    """Setup LaTeX rendering for matplotlib."""
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "mathtext.fontset": "cm",
        })
        return True
    except Exception as e:
        print(f"LaTeX rendering not available: {e}")
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        })
        return False


def plot_distance_evolution_linear(analysis_results: Dict,
                                   save_path: Optional[str] = None,
                                   show_first_n: Optional[int] = None,
                                   figsize: tuple = (8, 6)):
    setup_latex_rendering()
    distances = analysis_results['distances']
    saturation_step = analysis_results['saturation_step_05']

    if show_first_n is not None and show_first_n < len(distances):
        display_distances = distances[:show_first_n]
    else:
        display_distances = distances

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(display_distances, 'b-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.6, label=r'Threshold $(0.5)$')

    if saturation_step is not None and saturation_step < len(display_distances):
        ax.axvline(x=saturation_step, color='g', linestyle=':',
                   alpha=0.7, label=rf'Saturation $(t={saturation_step})$')

    ax.set_xlabel(r'Iteration Step $t$')
    ax.set_ylabel(r'Normalized Distance $d(t)$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_distance_evolution_semilog(analysis_results: Dict,
                                    save_path: Optional[str] = None,
                                    show_first_n: Optional[int] = None,
                                    figsize: tuple = (8, 6)):
    setup_latex_rendering()
    distances = analysis_results['distances']

    if show_first_n is not None and show_first_n < len(distances):
        display_distances = distances[:show_first_n]
    else:
        display_distances = distances

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(display_distances, 'g-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'Iteration Step $t$')
    ax.set_ylabel(r'Distance $d(t)$ (log scale)')
    ax.grid(True, alpha=0.3, which='both')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_initial_response(analysis_results: Dict, n_steps: int = 50,
                          save_path: Optional[str] = None,
                          figsize: tuple = (8, 6)):
    setup_latex_rendering()
    distances = analysis_results['distances']
    saturation_step = analysis_results['saturation_step_05']

    n_detail = min(n_steps, len(distances))
    display_distances = distances[:n_detail]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(display_distances, 'b-o', linewidth=1.5, markersize=4, alpha=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.6, label=r'Threshold $(0.5)$')

    if saturation_step is not None and saturation_step < n_detail:
        ax.axvline(x=saturation_step, color='g', linestyle=':',
                   alpha=0.7, label=rf'Saturation $(t={saturation_step})$')

    ax.set_xlabel(r'Iteration Step $t$')
    ax.set_ylabel(r'Normalized Distance $d(t)$')
    ax.set_title(rf'First ${n_detail}$ Steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_growth_rate_distribution(analysis_results: Dict,
                                  save_path: Optional[str] = None,
                                  figsize: tuple = (8, 6)):
    setup_latex_rendering()
    growth_rates = analysis_results['growth_rates']

    fig, ax = plt.subplots(figsize=figsize)
    if len(growth_rates) > 2:
        filtered_rates = growth_rates[1:]
        filtered_rates = filtered_rates[np.isfinite(filtered_rates)]
        if len(filtered_rates) > 0:
            log_rates = np.log10(filtered_rates[filtered_rates > 0])
            ax.hist(log_rates, bins=30, alpha=0.7, color='orange', edgecolor='black')

            mean_log = np.mean(log_rates) if len(log_rates) > 0 else 0
            median_log = np.median(log_rates) if len(log_rates) > 0 else 0
            ax.axvline(x=mean_log, color='b', linestyle='-',
                       label=rf'Mean: ${10**mean_log:.1e}$' if mean_log < 3
                       else rf'Mean $\log_{{10}}$: ${mean_log:.2f}$')
            ax.axvline(x=median_log, color='g', linestyle='--',
                       label=rf'Median: ${10**median_log:.1e}$' if median_log < 3
                       else rf'Median $\log_{{10}}$: ${median_log:.2f}$')
            ax.axvline(x=0, color='r', linestyle=':', label=r'Growth $= 1$')

            ax.set_xlabel(r'$\log_{10}(r_t)$ where $r_t = \frac{d(t+1)}{d(t)}$')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_cumulative_growth(analysis_results: Dict,
                           save_path: Optional[str] = None,
                           show_first_n: Optional[int] = None,
                           figsize: tuple = (8, 6)):
    setup_latex_rendering()
    distances = analysis_results['distances']

    if show_first_n is not None and show_first_n < len(distances):
        display_distances = distances[:show_first_n]
    else:
        display_distances = distances

    fig, ax = plt.subplots(figsize=figsize)
    if len(display_distances) > 0 and distances[0] > 0:
        cumulative_growth = display_distances / distances[0]
        ax.semilogy(cumulative_growth, 'purple', linewidth=1.5, alpha=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=2.0, color='green', linestyle=':', alpha=0.5, label=r'$2\times$ growth')
        ax.axhline(y=10.0, color='orange', linestyle=':', alpha=0.5, label=r'$10\times$ growth')
        ax.axhline(y=100.0, color='red', linestyle=':', alpha=0.5, label=r'$100\times$ growth')

        total_growth = cumulative_growth[-1]
        ax.text(0.05, 0.95, rf'Total growth: ${total_growth:.1e}\times$',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(r'Iteration Step $t$')
        ax.set_ylabel(r'Cumulative Growth $\frac{d(t)}{d(0)}$')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, which='both')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_post_saturation_behavior(analysis_results: Dict,
                                  save_path: Optional[str] = None,
                                  figsize: tuple = (8, 6)):
    setup_latex_rendering()
    distances = analysis_results['distances']
    saturation_step = analysis_results['saturation_step_05']

    fig, ax = plt.subplots(figsize=figsize)
    if saturation_step is not None and saturation_step < len(distances) - 10:
        post_sat = distances[saturation_step:]
        ax.plot(post_sat, 'b-', linewidth=1.5, alpha=0.8)

        mean_val = np.mean(post_sat)
        std_val = np.std(post_sat)
        ax.axhline(y=mean_val, color='r', linestyle='--', label=rf'Mean: ${mean_val:.4f}$')
        ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', label=r'$\pm 1\sigma$')
        ax.axhline(y=mean_val - std_val, color='orange', linestyle=':')

        stats_text = (rf'$\mu = {mean_val:.4f}$' + '\n' +
                      rf'$\sigma = {std_val:.4f}$' + '\n' +
                      rf'$n = {len(post_sat)}$')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        ax.set_xlabel(r'Steps after Saturation')
        ax.set_ylabel(r'Distance $d(t)$')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_growth_rate_evolution(analysis_results: Dict,
                               save_path: Optional[str] = None,
                               show_first_n: Optional[int] = None,
                               figsize: tuple = (8, 6)):
    setup_latex_rendering()
    growth_rates = analysis_results['growth_rates']

    if len(growth_rates) > 1:
        if show_first_n is not None and show_first_n < len(growth_rates):
            display_rates = growth_rates[:show_first_n]
        else:
            display_rates = growth_rates

        fig, ax = plt.subplots(figsize=figsize)
        steps = np.arange(1, len(display_rates))
        ax.plot(steps, display_rates[1:], 'b-', linewidth=1.5, alpha=0.8)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.6, label=r'Growth $= 1$')

        if len(display_rates) > 10:
            window = min(20, len(display_rates) // 10)
            rolling_avg = np.convolve(display_rates[1:], np.ones(window) / window,
                                      mode='valid')
            ax.plot(steps[window - 1:], rolling_avg, 'orange', linewidth=2, alpha=0.9,
                    label=rf'${window}$-step average')

        ax.set_xlabel(r'Iteration Step $t$')
        ax.set_ylabel(r'Growth Rate $r_t = \frac{d(t+1)}{d(t)}$')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        plt.show()


def plot_distance_distribution(analysis_results: Dict,
                               save_path: Optional[str] = None,
                               figsize: tuple = (8, 6)):
    setup_latex_rendering()
    distances = analysis_results['distances']

    fig, ax = plt.subplots(figsize=figsize)
    if len(distances) > 1:
        ax.hist(distances, bins=30, alpha=0.7, color='teal', edgecolor='black')

        mean_val = np.mean(distances)
        median_val = np.median(distances)
        std_val = np.std(distances)
        ax.axvline(x=mean_val, color='r', linestyle='--', label=rf'Mean: $\mu = {mean_val:.4f}$')
        ax.axvline(x=median_val, color='g', linestyle=':', label=rf'Median: ${median_val:.4f}$')

        stats_text = (rf'$\mu = {mean_val:.4f}$' + '\n' +
                      rf'$\sigma = {std_val:.4f}$' + '\n' +
                      rf'$n = {len(distances)}$' + '\n' +
                      rf'Range: $[{np.min(distances):.4f}, {np.max(distances):.4f}]$')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.set_xlabel(r'Normalized Distance $d(t)$')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def plot_sensitivity_statistics_panel(analysis_results: Dict,
                                      save_path: Optional[str] = None,
                                      figsize: tuple = (8, 10)):
    setup_latex_rendering()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    stats = {
        'System': {
            'Dimension': f"{analysis_results['system_dimension']}D",
            'Word size': f"{analysis_results['word_size']}-bit",
            'Modulus': f"$2^{{{analysis_results['word_size']}}} = {analysis_results['modulus']:,}$",
        },
        'Sensitivity Metrics': {
            'Initial distance': rf"${analysis_results['initial_distance']:.2e}$",
            'Final distance': rf"${analysis_results['final_distance']:.4f}$",
            'Total growth': rf"${analysis_results['total_growth_factor']:.1e}\times$",
            'First-step growth': rf"${analysis_results.get('first_growth', 0):.1f}\times$",
            'Average growth rate': rf"${analysis_results['average_growth_rate']:.3f}$",
        },
        'Divergence Speed': {
            'Saturation @0.5': f"{analysis_results['saturation_step_05'] if analysis_results['saturation_step_05'] is not None else 'Not reached'}",
            'Saturation @0.8': f"{analysis_results['saturation_step_08'] if analysis_results['saturation_step_08'] is not None else 'Not reached'}",
        },
        'Statistics': {
            'Mean distance': rf"${analysis_results['mean_distance']:.4f}$",
            'Std distance': rf"${analysis_results['std_distance']:.4f}$",
            'Min distance': rf"${analysis_results['min_distance']:.4f}$",
            'Max distance': rf"${analysis_results['max_distance']:.4f}$",
            'Total iterations': f"{len(analysis_results['distances'])}",
        }
    }

    text_lines = []
    for category, items in stats.items():
        text_lines.append(r'\textbf{' + category + r'}')
        for key, value in items.items():
            text_lines.append(f'{key}: {value}')
        text_lines.append('')

    text = '\n'.join(text_lines)
    if 'sensitivity_class' in analysis_results:
        text += f"\n\\textbf{{Sensitivity Classification}}\n"
        text += f"{analysis_results['sensitivity_class']}\n"

    ax.text(0.05, 0.95, text, fontfamily='monospace', fontsize=10,
            verticalalignment='top', transform=ax.transAxes)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()


def export_all_individual_plots(analysis_results: Dict,
                                output_dir: str = "sensitivity_plots",
                                prefix: str = "plot",
                                show_first_n: int = 200):
    """Export all 9 individual plots to separate files in one call."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting all individual plots to: {output_dir}/")

    plot_distance_evolution_linear(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_01_linear_distance.eps"),
        show_first_n=show_first_n)
    plot_distance_evolution_semilog(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_02_semilog_distance.eps"),
        show_first_n=show_first_n)
    plot_initial_response(
        analysis_results, n_steps=50,
        save_path=os.path.join(output_dir, f"{prefix}_03_initial_response.eps"))
    plot_growth_rate_distribution(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_04_growth_distribution.eps"))
    plot_cumulative_growth(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_05_cumulative_growth.eps"),
        show_first_n=show_first_n)
    plot_post_saturation_behavior(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_06_post_saturation.eps"))
    plot_growth_rate_evolution(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_07_growth_evolution.eps"),
        show_first_n=show_first_n)
    plot_distance_distribution(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_08_distance_distribution.eps"))
    plot_sensitivity_statistics_panel(
        analysis_results,
        save_path=os.path.join(output_dir, f"{prefix}_09_statistics_panel.pdf"))

    print(f"All plots exported to {output_dir}/")

    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("Sensitivity Analysis Individual Plots\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"System: {analysis_results['system_dimension']}D, "
                f"{analysis_results['word_size']}-bit\n")
        f.write(f"Total iterations: {len(analysis_results['distances'])}\n")
        f.write(f"Initial distance: {analysis_results['initial_distance']:.2e}\n")
        f.write(f"Total growth: {analysis_results['total_growth_factor']:.1e}x\n")
    print(f"README file created: {readme_path}")
