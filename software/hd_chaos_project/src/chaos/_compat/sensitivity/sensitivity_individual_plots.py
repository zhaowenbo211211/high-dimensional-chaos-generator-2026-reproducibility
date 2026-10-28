#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Backward-compatible wrapper. See chaos.visualization.sensitivity_plots for canonical implementation.
"""

from chaos.visualization.sensitivity_plots import (
    setup_latex_rendering,
    plot_distance_evolution_linear,
    plot_distance_evolution_semilog,
    plot_initial_response,
    plot_growth_rate_distribution,
    plot_cumulative_growth,
    plot_post_saturation_behavior,
    plot_growth_rate_evolution,
    plot_distance_distribution,
    plot_sensitivity_statistics_panel,
    export_all_individual_plots,
)
