#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Statistical distribution analysis for n-dimensional chaos (standalone script).

See chaos.analysis.distribution for the canonical class implementations.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from chaos.core.params import ComplementaryChaosParams
from chaos.core.base_map import HighDimChaosMap
from chaos.analysis.distribution import (
    ChaosHistogramAnalyzer,
    ChaosCDFDeviationAnalyzer,
    ChaosReportGenerator,
)

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_SCRIPT_DIR / "nd_chaos_statistical_distribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_nd_chaos_map(n: int = 4, params: Optional[dict] = None,
                        seed: int = 123, word_size: int = 16) -> HighDimChaosMap:
    if params is None:
        if n != 4:
            raise ValueError(
                "Default params are only for n=4. "
                "For other dimensions, provide params={'a_j1', 'a_jn', 'b'}.")
        params = {
            "a_j1": np.array([371, 373, 375], dtype=np.int64),
            "a_jn": np.array([50, 48, 46], dtype=np.int64),
            "b": 111,
        }
    else:
        a_j1 = np.asarray(params["a_j1"])
        a_jn = np.asarray(params["a_jn"])
        if a_j1.shape[0] != n - 1 or a_jn.shape[0] != n - 1:
            raise ValueError(
                f"Parameter dimension mismatch: len(a_j1)={a_j1.shape[0]}, "
                f"len(a_jn)={a_jn.shape[0]}, n={n}")
    return HighDimChaosMap(params=params, word_size=word_size, seed=seed)


def generate_nd_sequence(chaos_map: HighDimChaosMap, warmup: int = 5000,
                         length: int = 100_000,
                         random_state_seed: int = 2025) -> Tuple[np.ndarray, int]:
    chaos_map.set_random_state(seed=random_state_seed)
    if warmup > 0:
        chaos_map.iterate(warmup)
    data = chaos_map.iterate(length)
    return data.astype(np.int64), chaos_map.mod


def main():
    n = 5
    word_size = 16

    synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
    params = synth.synthesize()

    print(f"   a_j1 : {params['a_j1']}")
    print(f"   a_jn : {params['a_jn']}")
    print(f"   b = {params['b']}")

    chaos_map = create_nd_chaos_map(n=n, params=params, seed=123, word_size=16)

    warmup = 5000
    length = 100_000
    data, mod = generate_nd_sequence(chaos_map, warmup=warmup, length=length,
                                     random_state_seed=2025)

    dims_to_plot = list(range(min(n, 5)))

    hist_analyzer = ChaosHistogramAnalyzer(data, mod, dims_to_plot=dims_to_plot)
    hist_bins = 128

    hist_analyzer.plot_overall_histogram(OUTPUT_DIR, filename="histogram_overall.pdf",
                                         bins=hist_bins)
    for dim in dims_to_plot:
        hist_analyzer.plot_single_histogram_subfigure(
            OUTPUT_DIR, dim=dim, filename=f"hist_x{dim+1}_sub.eps", bins=hist_bins)

    cdf_analyzer = ChaosCDFDeviationAnalyzer(data, mod, dims_to_plot=dims_to_plot)
    cdf_analyzer.plot_overall_cdf_deviation(OUTPUT_DIR, filename="cdf_deviation_overall.pdf")
    for dim in dims_to_plot:
        cdf_analyzer.plot_single_cdf_deviation_subfigure(
            OUTPUT_DIR, dim=dim, filename=f"cdf_dev_x{dim+1}_sub.eps")

    report_generator = ChaosReportGenerator(
        data_int=data, mod=mod, cdf_analyzer=cdf_analyzer,
        warmup=warmup, total_samples=length)
    report_path = OUTPUT_DIR / "chaos_stats_report.txt"
    report_text = report_generator.save_report(report_path)

    print(f"All figures saved in: {OUTPUT_DIR}")
    print(f"Text report saved to: {report_path}")
    print()
    print(report_text)


if __name__ == "__main__":
    main()
