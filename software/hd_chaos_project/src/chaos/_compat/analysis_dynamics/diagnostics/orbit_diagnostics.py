#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""Backward-compatible wrapper. See chaos.analysis.orbit for the canonical implementation."""

from chaos.analysis.orbit import (
    OrbitStatsConfig,
    estimate_period,
    sample_orbit,
    compute_marginal_histograms,
    analyze_digital_system,
    make_step_fn_from_map,
    SupportsNext,
    State,
    StepFn,
)
