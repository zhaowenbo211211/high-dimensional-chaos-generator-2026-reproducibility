# Maintainer: zhaowenbo
"""Numerically safe scalar/array math helpers.

These helpers are designed for later chaotic systems that may involve
potentially unstable nonlinear functions such as `tan`, `exp`, or `sinh`.
Each function supports both NumPy arrays and scalar inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_nan_to_num(x: Any) -> Any:
    """Replace NaN and infinities with finite values."""

    return np.nan_to_num(x, nan=0.0, posinf=1e12, neginf=-1e12)


def safe_div(a: Any, b: Any, eps: float = 1e-12) -> Any:
    """Safely divide `a / b` by preventing division by near-zero denominators."""

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    denom = np.where(np.abs(b_arr) < eps, np.sign(b_arr) * eps + (b_arr == 0) * eps, b_arr)
    return safe_nan_to_num(a_arr / denom)


def safe_tan(x: Any, clip: float = 1.55) -> Any:
    """Apply tangent after clipping the input away from asymptotes."""

    return safe_nan_to_num(np.tan(np.clip(x, -clip, clip)))


def safe_tanh(x: Any) -> Any:
    """Stable hyperbolic tangent wrapper."""

    return safe_nan_to_num(np.tanh(x))


def safe_sin(x: Any) -> Any:
    """Stable sine wrapper with NaN cleanup."""

    return safe_nan_to_num(np.sin(x))


def safe_cos(x: Any) -> Any:
    """Stable cosine wrapper with NaN cleanup."""

    return safe_nan_to_num(np.cos(x))


def safe_exp(x: Any, clip: float = 20.0) -> Any:
    """Exponentiate after clipping to avoid overflow."""

    return safe_nan_to_num(np.exp(np.clip(x, -clip, clip)))


def safe_sinh(x: Any, clip: float = 10.0) -> Any:
    """Hyperbolic sine after clipping to limit explosive growth."""

    return safe_nan_to_num(np.sinh(np.clip(x, -clip, clip)))
