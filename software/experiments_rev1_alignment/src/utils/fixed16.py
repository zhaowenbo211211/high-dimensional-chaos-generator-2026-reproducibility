# Maintainer: zhaowenbo
"""Utilities for consistent 16-bit fixed-point state handling.

All normalized values are interpreted on `[0, 1)`. Conversion from float to
unsigned 16-bit integers uses `floor(x * 65536) mod 65536` after wrapping into
the unit interval. This keeps the numeric semantics identical across all maps.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

U16_MOD = 65536
U16_MAX = 65535


def wrap_mod1(x: float) -> float:
    """Wrap a real value into the normalized interval `[0, 1)`.

    Parameters
    ----------
    x:
        Input real value.

    Returns
    -------
    float
        The wrapped value `x mod 1.0`, guaranteed to lie in `[0, 1)`.
    """

    return float(np.mod(x, 1.0))


def quantize_mod1_to_u16(x: float) -> int:
    """Quantize a normalized real value into 16-bit unsigned fixed-point.

    The input is first wrapped into `[0, 1)`, then quantized by
    `floor(x * 65536) mod 65536`.
    """

    wrapped = wrap_mod1(x)
    return int(np.floor(wrapped * U16_MOD)) % U16_MOD


def float_to_u16(x: float) -> int:
    """Convert a real scalar to a wrapped 16-bit fixed-point integer."""

    return quantize_mod1_to_u16(x)


def u16_to_float(x: int) -> float:
    """Convert an unsigned 16-bit fixed-point integer to a float in `[0, 1)`."""

    return (int(x) % U16_MOD) / U16_MOD


def u16_vec_to_float(vec: Iterable[int]) -> np.ndarray:
    """Convert an iterable of unsigned 16-bit values to a float64 vector."""

    arr = np.asarray(list(vec), dtype=np.int64) % U16_MOD
    return arr.astype(np.float64) / U16_MOD


def float_vec_to_u16(vec: Iterable[float]) -> np.ndarray:
    """Convert an iterable of reals to a `uint16`-semantic integer vector."""

    arr = np.asarray(list(vec), dtype=np.float64)
    wrapped = np.mod(arr, 1.0)
    quantized = np.floor(wrapped * U16_MOD).astype(np.int64) % U16_MOD
    return quantized


def sequence_to_u16_array(seq: Iterable[float | int]) -> np.ndarray:
    """Convert a one-dimensional numeric sequence to the shared u16 domain."""

    raw = np.asarray(list(seq))
    if raw.ndim != 1:
        raise ValueError("Input sequence must be one-dimensional.")
    if np.issubdtype(raw.dtype, np.integer):
        return np.asarray(raw, dtype=np.int64) % U16_MOD
    return float_vec_to_u16(np.asarray(raw, dtype=np.float64))
