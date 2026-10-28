# Maintainer: zhaowenbo
"""Common initialization utilities for 4D and high-dimensional maps."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from utils.fixed16 import float_vec_to_u16

COMMON_INIT_FLOAT = (0.1, 0.2, 0.3, 0.4)


def get_common_initial_state_float() -> np.ndarray:
    """Return the common floating-point initial state as float64."""

    return np.asarray(COMMON_INIT_FLOAT, dtype=np.float64)


def get_common_initial_state_u16() -> np.ndarray:
    """Return the common initial state quantized into 16-bit fixed-point."""

    return float_vec_to_u16(COMMON_INIT_FLOAT)


def get_dimension_initial_state_float(n: int) -> np.ndarray:
    """Return a deterministic dimension-aware initial state.

    The 4D default remains `(0.1, 0.2, 0.3, 0.4)`. For `n > 4`, the sequence is
    extended in a simple non-singular deterministic pattern:

    `(0.1, 0.2, ..., 0.9, 0.15, 0.25, 0.35, ...)`
    """

    if int(n) < 1:
        raise ValueError("n must be positive.")
    if int(n) == 4:
        return get_common_initial_state_float()

    values: list[float] = []
    for index in range(int(n)):
        block = index // 9
        offset = index % 9
        value = 0.1 * float(offset + 1) + 0.05 * float(block)
        values.append(float(np.mod(value, 1.0)))
    return np.asarray(values, dtype=np.float64)


def get_dimension_initial_state_u16(n: int) -> np.ndarray:
    """Return the dimension-aware initial state quantized into 16-bit integers."""

    return float_vec_to_u16(get_dimension_initial_state_float(int(n)))


def validate_non_singular_init(state: Iterable[float] | None = None) -> bool:
    """Placeholder hook for future non-singular initialization rules."""

    _ = state
    return True
