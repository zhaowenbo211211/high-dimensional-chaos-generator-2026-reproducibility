# Maintainer: zhaowenbo
"""URUK-4D-DCM under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float, wrap_mod1
from utils.safe_math import safe_nan_to_num, safe_tan


class URUK4DDCMMap(ChaosMap4DBase):
    """URUK-4D-DCM with safe tangent evaluation and mod-1 wrapping.

    This system contains strongly nonlinear ``tan`` terms. To adapt it to the
    project's unified 16-bit comparison framework, the implementation evaluates
    the original update in float form, applies safe trigonometric functions and
    finite-value cleanup, wraps the results back into ``[0, 1)``, and then
    quantizes to the shared u16 representation.

    The goal of this implementation is a consistent comparison semantics under
    the common digital framework, rather than exact continuous-state fidelity.
    """

    name = "URUK4DDCMMap"

    def __init__(
        self,
        a: float = 0.05,
        b: float = 0.05,
        c: float = 0.05,
        d: float = 0.05,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the URUK state using the shared 16-bit initialization rules."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

    def step(self) -> np.ndarray:
        """Advance one step of URUK-4D-DCM under unified digital semantics.

        The original system contains ``tan`` terms, so this implementation uses
        ``utils.safe_math.safe_tan`` and finite-value cleanup before mapping the
        updated float state back into ``[0, 1)`` with ``wrap_mod1`` and then
        quantizing it into the common 16-bit representation.
        """

        x, y, z, w = (float(value) for value in u16_vec_to_float(self.state_u16))

        tan_z = float(safe_tan(z))
        tan_z_sq = float(safe_tan(z * z))
        next_state_float = np.asarray(
            [
                1.0 - (x * y * z * w) - x * x - y * y - self.a * tan_z_sq - w * w,
                x - self.b * tan_z,
                y - self.c * tan_z,
                x - self.d * w,
            ],
            dtype=np.float64,
        )
        next_state_float = np.asarray(safe_nan_to_num(next_state_float), dtype=np.float64)
        wrapped_state = np.asarray([wrap_mod1(value) for value in next_state_float], dtype=np.float64)
        self.state_u16 = float_vec_to_u16(wrapped_state)
        return u16_vec_to_float(self.state_u16)
