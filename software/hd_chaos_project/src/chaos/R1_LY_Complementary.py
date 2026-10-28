#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
R1-LY-Complementary: backward-compatible re-export module.

This file is kept for backward compatibility with existing scripts and tests.
All functionality has been refactored into:
  - chaos.core.params          (ComplementaryChaosParams)
  - chaos.core.base_map        (HighDimChaosMap)
  - chaos.core.enhanced_map    (HighDimChaosMapEnhanced)
  - chaos.analysis.lyapunov    (compute_lyapunov_exponents, verify_chaos)
  - chaos.utils.sequence       (generate_binary_sequence)
  - chaos.visualization.verification (run_complete_verification)
"""

# Re-export everything from new module locations
from chaos.core.params import ComplementaryChaosParams
from chaos.core.base_map import HighDimChaosMap
from chaos.core.enhanced_map import HighDimChaosMapEnhanced
from chaos.analysis.lyapunov import (
    compute_lyapunov_exponents,
    _jacobian_normalized,
    verify_chaos,
)
from chaos.utils.sequence import generate_binary_sequence
from chaos.visualization.verification import run_complete_verification
