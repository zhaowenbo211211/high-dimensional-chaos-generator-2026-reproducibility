# Maintainer: zhaowenbo
"""4D-MHM under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float, wrap_mod1
from utils.safe_math import safe_cos, safe_nan_to_num, safe_sin


class MHM4DMap(ChaosMap4DBase):
    """Memristive/trigonometric 4D discrete system under unified digital rules.

    This is a memristive/trigonometric 4D discrete system. To adapt it to the
    unified 16-bit multi-system comparison framework, the implementation uses
    safe trigonometric function evaluation, then applies ``mod 1`` wrapping and
    u16 quantization after each float-domain update.

    The focus here is the shared comparison semantics, rather than exact
    continuous-state fidelity.
    """

    name = "MHM4DMap"

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        k1: float = 0.1,
        k2: float = 0.1,
        k3: float = 0.1,
        c1: float = 0.0,
        c2: float = 0.0,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.c1 = float(c1)
        self.c2 = float(c2)
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the MHM state using the shared 16-bit initialization rules."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

    def step(self) -> np.ndarray:
        """Advance one step of the memristive/trigonometric 4D recurrence.

        To fit the shared 16-bit comparison framework, this implementation uses
        the strategy ``float update + mod1 wrapping + u16 quantization``. The
        map contains memristive/trigonometric couplings, so ``sin`` and ``cos``
        are evaluated via the shared safe-math helpers rather than raw NumPy.
        """

        x, y, z, q = (float(value) for value in u16_vec_to_float(self.state_u16))
        next_state_float = np.asarray(
            [
                self.a * float(safe_cos(z)) * float(safe_sin(y)),
                self.k1 * x * float(safe_cos(q)) + self.c1,
                self.b * y * float(safe_sin(x)) + self.c2,
                self.k2 * q + self.k3 * x,
            ],
            dtype=np.float64,
        )
        next_state_float = np.asarray(safe_nan_to_num(next_state_float), dtype=np.float64)
        wrapped_state = np.asarray([wrap_mod1(value) for value in next_state_float], dtype=np.float64)
        self.state_u16 = float_vec_to_u16(wrapped_state)
        return u16_vec_to_float(self.state_u16)
