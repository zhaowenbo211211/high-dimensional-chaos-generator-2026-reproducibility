# Maintainer: zhaowenbo
"""4D-MPRC under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float, wrap_mod1
from utils.safe_math import safe_cos, safe_nan_to_num, safe_sin


class MPRC4DMap(ChaosMap4DBase):
    """Multi-trigonometric cross-coupled 4D discrete chaotic system.

    This is a multi-trigonometric cross-coupled 4D discrete system containing
    several ``sin(pi * ...)`` and ``cos(pi * ...)`` interaction terms. For
    unified comparison, the implementation evaluates the map in float form,
    applies shared safe trigonometric functions, wraps each component with
    ``mod 1``, and then quantizes into the common 16-bit representation.

    The focus is a fair digital-comparison semantics, rather than extreme
    fidelity to an unconstrained continuous-state realization.
    """

    name = "MPRC4DMap"

    def __init__(
        self,
        a: float = 1.9,
        b: float = 5.0,
        c: float = 23.0,
        d: float = 6.7,
        e: float = 2.0,
        f: float = 4.0,
        g: float = 9.6,
        h: float = 3.5,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.e = float(e)
        self.f = float(f)
        self.g = float(g)
        self.h = float(h)
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the MPRC state using the shared 16-bit initialization rules."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

    def step(self) -> np.ndarray:
        """Advance one step of the mod-1 MPRC recurrence."""

        x, y, z, w = (float(value) for value in u16_vec_to_float(self.state_u16))

        next_state_float = np.asarray(
            [
                self.a * float(safe_sin(np.pi * x))
                + self.b * float(safe_cos(np.pi * y))
                + self.c * z * z
                + self.d * w
                + self.e * float(safe_sin(np.pi * y * z)),
                self.e * float(safe_sin(np.pi * y))
                + self.f * float(safe_cos(np.pi * z))
                + self.g * x * x
                + self.h * w
                + self.a * float(safe_cos(np.pi * x * w)),
                self.a * float(safe_cos(np.pi * z))
                + self.b * float(safe_sin(np.pi * w))
                + self.c * x * x
                + self.d * y
                + self.f * float(safe_sin(np.pi * x * y)),
                self.e * float(safe_cos(np.pi * w))
                + self.f * float(safe_sin(np.pi * x))
                + self.g * y * y
                + self.h * z
                + self.b * float(safe_cos(np.pi * y * z)),
            ],
            dtype=np.float64,
        )
        next_state_float = np.asarray(safe_nan_to_num(next_state_float), dtype=np.float64)
        wrapped_state = np.asarray([wrap_mod1(value) for value in next_state_float], dtype=np.float64)
        self.state_u16 = float_vec_to_u16(wrapped_state)
        return u16_vec_to_float(self.state_u16)
