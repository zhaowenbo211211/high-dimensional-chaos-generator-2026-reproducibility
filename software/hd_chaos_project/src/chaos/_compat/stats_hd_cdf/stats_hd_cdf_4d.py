#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Backward-compatible wrapper for 4D statistical distribution analysis.
See chaos.analysis.distribution and chaos.stats_hd_cdf.stats_hd_cdf_nd for canonical implementations.
"""

# This module is superseded by stats_hd_cdf_nd.py which handles arbitrary n.
# Kept for backward compatibility only.
from chaos.stats_hd_cdf.stats_hd_cdf_nd import (
    create_nd_chaos_map as create_4d_chaos_map,
    generate_nd_sequence as generate_4d_sequence,
    ChaosHistogramAnalyzer,
    ChaosCDFDeviationAnalyzer,
    ChaosReportGenerator,
    OUTPUT_DIR,
)
from pathlib import Path
import numpy as np


def main():
    import os
    from pathlib import Path

    # Use the 4D-specific output directory
    output_dir = Path(__file__).resolve().parent / "4dchaos_instance_Statistical_Distribution"
    os.makedirs(output_dir, exist_ok=True)

    chaos_map = create_4d_chaos_map(seed=123)
    warmup = 5000
    length = 100_000
    data, mod = generate_4d_sequence(chaos_map, warmup=warmup, length=length,
                                     random_state_seed=2025)

    hist_analyzer = ChaosHistogramAnalyzer(data, mod)
    hist_bins = 128

    hist_analyzer.plot_overall_histogram(output_dir, filename="histogram_overall.pdf",
                                         bins=hist_bins)
    for dim in range(4):
        hist_analyzer.plot_single_histogram_subfigure(
            output_dir, dim=dim, filename=f"hist_x{dim+1}_sub.eps", bins=hist_bins)

    cdf_analyzer = ChaosCDFDeviationAnalyzer(data, mod)
    cdf_analyzer.plot_overall_cdf_deviation(output_dir, filename="cdf_deviation_overall.pdf")
    for dim in range(4):
        cdf_analyzer.plot_single_cdf_deviation_subfigure(
            output_dir, dim=dim, filename=f"cdf_dev_x{dim+1}_sub.eps")

    report_generator = ChaosReportGenerator(
        data_int=data, mod=mod, cdf_analyzer=cdf_analyzer,
        warmup=warmup, total_samples=length)
    report_path = Path(output_dir) / "chaos_stats_report.tex"
    report_text = report_generator.save_report(report_path)

    print(f"All figures saved in: {output_dir}")
    print(f"LaTeX report saved to: {report_path}")
    print(report_text)


if __name__ == "__main__":
    main()
