# Maintainer: zhaowenbo
"""4D-IDHNN under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float, wrap_mod1
from utils.safe_math import safe_div, safe_nan_to_num, safe_sin, safe_tanh


class IDHNN4DMap(ChaosMap4DBase):
    """Tanh/neural-coupled 4D discrete system with reciprocal-driven sine term.

    This system contains tanh-based neural coupling together with the
    reciprocal-driven sine term ``sin(C / x2)``. To adapt it to the unified
    16-bit multi-system comparison framework, the implementation uses safe
    division, safe hyperbolic functions, safe sine evaluation, then applies the
    strategy ``float update + mod1 wrapping + u16 quantization``.

    The focus is a stable and shared numerical semantics for comparison
    experiments, not exact continuous-state fidelity.
    """

    name = "IDHNN4DMap"

    def __init__(
        self,
        C: float = 0.5,
        E: float = 1.0,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.C = float(C)
        self.E = float(E)
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the IDHNN state using the shared 16-bit initialization rules."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

    def step(self) -> np.ndarray:
        """Advance one step of the IDHNN recurrence under unified digital rules."""

        x1, x2, x3, x4 = (float(value) for value in u16_vec_to_float(self.state_u16))

        tanh_x1 = float(safe_tanh(x1))
        tanh_x2 = float(safe_tanh(x2))
        tanh_x3 = float(safe_tanh(x3))
        tanh_x4 = float(safe_tanh(x4))
        reciprocal_term = float(safe_div(self.C, x2, eps=1e-12))
        reciprocal_sine = float(safe_sin(reciprocal_term))

        next_state_float = np.asarray(
            [
                -x1 - 1.5 * tanh_x1 - 6.0 * tanh_x2 + self.C * reciprocal_sine * tanh_x3 + tanh_x4,
                -x2 + self.E * tanh_x1 + 4.0 * tanh_x2 - 3.0 * tanh_x3,
                -x3 - tanh_x1 - 3.0 * tanh_x2 + 1.5 * tanh_x3 + 10.0 * tanh_x4,
                -x4 + 4.0 * tanh_x2 - 5.0 * tanh_x3 + 2.0 * tanh_x4,
            ],
            dtype=np.float64,
        )
        next_state_float = np.asarray(safe_nan_to_num(next_state_float), dtype=np.float64)
        wrapped_state = np.asarray([wrap_mod1(value) for value in next_state_float], dtype=np.float64)
        self.state_u16 = float_vec_to_u16(wrapped_state)
        return u16_vec_to_float(self.state_u16)
