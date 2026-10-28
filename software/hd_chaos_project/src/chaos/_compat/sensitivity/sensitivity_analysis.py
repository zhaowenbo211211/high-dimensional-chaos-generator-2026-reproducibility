#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Backward-compatible wrapper for sensitivity analysis.
See chaos.analysis.sensitivity and chaos.visualization.sensitivity_plots for canonical implementations.
"""

from chaos.analysis.sensitivity import (
    analyze_initial_value_sensitivity_16bit,
    plot_sensitivity_analysis,
)
from chaos.visualization.sensitivity_plots import export_all_individual_plots
from chaos.core.params import ComplementaryChaosParams
from chaos.core.base_map import HighDimChaosMap
import numpy as np
import matplotlib.pyplot as plt


# Re-export the helper functions for backward compatibility
def test_multiple_perturbation_directions(*args, **kwargs):
    """See chaos.analysis.sensitivity for updated version."""
    from chaos.analysis.sensitivity import analyze_initial_value_sensitivity_16bit
    from chaos.core.base_map import HighDimChaosMap

    params = kwargs.get('params', args[0] if args else None)
    initial_state = kwargs.get('initial_state', args[1] if len(args) > 1 else None)
    perturbation_magnitude = kwargs.get('perturbation_magnitude', args[2] if len(args) > 2 else 1)
    min_iterations = kwargs.get('min_iterations', args[3] if len(args) > 3 else 1000)
    max_iterations = kwargs.get('max_iterations', args[4] if len(args) > 4 else 5000)

    test_chaos = HighDimChaosMap(params, word_size=16)
    n = test_chaos.n

    if initial_state is None:
        np.random.seed(42)
        initial_state = np.random.randint(0, 65536, size=n, dtype=np.int64)

    directions = []
    direction_names = []
    for i in range(n):
        vec = np.zeros(n, dtype=np.int64)
        vec[i] = perturbation_magnitude
        directions.append(vec)
        direction_names.append(f"x{i+1} component")

    results = {}
    for vec, name in zip(directions, direction_names):
        result = analyze_initial_value_sensitivity_16bit(
            params=params, map_cls=HighDimChaosMap,
            initial_state=initial_state, perturbation_vector=vec,
            min_iterations=min_iterations, max_iterations=max_iterations)
        results[name] = {
            'saturation_step_05': result['saturation_step_05'],
            'saturation_step_08': result['saturation_step_08'],
            'total_growth': result['total_growth_factor'],
            'average_growth': result['average_growth_rate'],
            'first_growth': result['first_growth'],
            'final_distance': result['final_distance'],
            'num_iterations': len(result['distances'])
        }
    return results


def compare_perturbation_magnitudes(*args, **kwargs):
    """See chaos.analysis.sensitivity for updated version."""
    from chaos.analysis.sensitivity import analyze_initial_value_sensitivity_16bit
    from chaos.core.base_map import HighDimChaosMap

    params = kwargs.get('params', args[0] if args else None)
    initial_state = kwargs.get('initial_state', args[1] if len(args) > 1 else None)
    magnitudes = kwargs.get('magnitudes', args[2] if len(args) > 2 else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    min_iterations = kwargs.get('min_iterations', args[3] if len(args) > 3 else 1000)
    max_iterations = kwargs.get('max_iterations', args[4] if len(args) > 4 else 5000)

    test_chaos = HighDimChaosMap(params, word_size=16)
    n = test_chaos.n

    if initial_state is None:
        np.random.seed(42)
        initial_state = np.random.randint(0, 65536, size=n, dtype=np.int64)

    results = {}
    for mag in magnitudes:
        perturbation = np.zeros(n, dtype=np.int64)
        perturbation[0] = mag
        result = analyze_initial_value_sensitivity_16bit(
            params=params, map_cls=HighDimChaosMap,
            initial_state=initial_state, perturbation_vector=perturbation,
            min_iterations=min_iterations, max_iterations=max_iterations)
        results[mag] = {
            'saturation_step_05': result['saturation_step_05'],
            'final_distance': result['final_distance'],
            'total_growth': result['total_growth_factor'],
            'first_growth': result['first_growth'],
            'num_iterations': len(result['distances'])
        }
    return results


if __name__ == "__main__":
    n = 4
    word_size = 16
    print("=" * 70)
    print(f"R1-LY-Complementary verification (n={n}, {word_size}-bit fixed-point)")
    print("=" * 70)

    print("\n[1] Parameter synthesis...")
    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    result = analyze_initial_value_sensitivity_16bit(
        params=params, map_cls=HighDimChaosMap, max_iterations=5000)
    plot_sensitivity_analysis(result, save_path="sensitivity_analysis.eps")

    analysis_results = analyze_initial_value_sensitivity_16bit(
        params=params, map_cls=HighDimChaosMap,
        min_iterations=1000, max_iterations=5000)

    export_all_individual_plots(
        analysis_results,
        output_dir="4dchaos_instance_sensitivity_analysis_plots",
        prefix="chaos_sensitivity",
        show_first_n=200)
