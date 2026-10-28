# Maintainer: zhaowenbo
"""4D-HCCDM under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float, wrap_mod1
from utils.safe_math import safe_exp, safe_nan_to_num, safe_sin, safe_sinh


class HCCDM4DMap(ChaosMap4DBase):
    """High-complexity 4D discrete system with Tent, ``sinh`` and ``exp`` terms.

    This is a high-complexity 4D discrete system containing a Tent piecewise
    map, together with ``sinh`` and ``exp`` nonlinearities. Its numerical risk
    is relatively high, so the implementation uses shared safe nonlinear
    helpers, explicit finite-value cleanup, and final ``mod 1`` wrapping before
    quantization into the common 16-bit representation.

    The focus here is the project's unified digital experiment semantics rather
    than exact fidelity to an unconstrained continuous-state realization.
    """

    name = "HCCDM4DMap"

    def __init__(
        self,
        c1: float = 30.0,
        c2: float = 13.0,
        c3: float = 6.0,
        c4: float = 20.0,
        m: float = 1.0,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)
        self.c4 = float(c4)
        self.m = float(m)
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the HCCDM state using the shared 16-bit initialization rules."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

    @staticmethod
    def _tent_map(x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the Tent-like piecewise map used by HCCDM."""

        x_arr = np.asarray(x, dtype=np.float64)
        sin_term = np.asarray(safe_sin(np.pi * x_arr), dtype=np.float64)
        lower = (5.0 * x_arr + 4.0 * sin_term) / 2.0
        upper = (5.0 * (1.0 - x_arr) + 4.0 * sin_term) / 2.0
        result = np.where((x_arr > 0.0) & (x_arr < 0.5), lower, upper)
        result = np.asarray(safe_nan_to_num(result), dtype=np.float64)
        if np.isscalar(x):
            return float(result)
        return result

    def step(self) -> np.ndarray:
        """Advance one step of the high-risk HCCDM recurrence."""

        x1, x2, x3, x4 = (float(value) for value in u16_vec_to_float(self.state_u16))
        tent_x4 = float(self._tent_map(x4))
        sin_pi_x2 = float(safe_sin(np.pi * x2))
        sin_pi_x3 = float(safe_sin(np.pi * x3))
        sinh_term = float(safe_sinh(x3 + 4.0))
        exp_term = float(safe_exp(x4))

        next_state_float = np.asarray(
            [
                self.c1 * x1 + (1.0 - 7.0 * (sin_pi_x2**2)) + x3 + tent_x4,
                self.c2 * ((x2 + 10.0) ** 3) + sin_pi_x3 + tent_x4,
                self.c3 * sinh_term + tent_x4,
                self.c4 * ((exp_term + 8.0) ** 2),
            ],
            dtype=np.float64,
        )
        next_state_float = np.asarray(safe_nan_to_num(next_state_float), dtype=np.float64)

        if self.m == 1.0:
            wrapped_state = np.asarray([wrap_mod1(value) for value in next_state_float], dtype=np.float64)
        else:
            wrapped_state = np.mod(next_state_float, self.m)
            wrapped_state = np.asarray(safe_nan_to_num(wrapped_state), dtype=np.float64)
            wrapped_state = np.asarray([wrap_mod1(value / self.m) for value in wrapped_state], dtype=np.float64)

        self.state_u16 = float_vec_to_u16(wrapped_state)
        return u16_vec_to_float(self.state_u16)
