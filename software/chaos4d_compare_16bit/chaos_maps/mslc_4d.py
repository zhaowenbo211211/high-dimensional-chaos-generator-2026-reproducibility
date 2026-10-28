# Maintainer: zhaowenbo
"""4D-MSLC under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float, wrap_mod1
from utils.safe_math import safe_cos, safe_nan_to_num, safe_sin


class MSLC4DMap(ChaosMap4DBase):
    """4D discrete system with explicit time-driving term ``sin(n)``.

    This is a non-autonomous 4D discrete system because it contains the explicit
    time-driving term ``sin(n)``. For unified comparison, the implementation
    uses the strategy ``float update + mod1 wrapping + u16 quantization`` with
    shared safe trigonometric helpers.

    The time-driving term is a key feature that distinguishes this system from
    autonomous maps, so an internal time index is maintained explicitly.
    """

    name = "MSLC4DMap"

    def __init__(
        self,
        a: float = 0.1,
        b: float = 0.1,
        c: float = 10.0,
        k1: float = 1.9,
        k2: float = 0.1,
        l1: float = 1.0,
        l2: float = 1.0,
        c1: float = 0.0,
        c2: float = 0.0,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.time_index = 0
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the MSLC state and explicit time index."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)
        self.time_index = 0

    def _current_time_drive(self) -> float:
        """Return the current explicit driving term ``sin(n)``."""

        return float(safe_sin(float(self.time_index)))

    def step(self) -> np.ndarray:
        """Advance one step of the time-driven MSLC recurrence."""

        x, y, z, q = (float(value) for value in u16_vec_to_float(self.state_u16))
        time_drive = self._current_time_drive()
        next_state_float = np.asarray(
            [
                self.a * float(safe_sin(y * (1.0 - y))) + self.k1 * float(safe_cos(q)) * x,
                self.b * z + self.l1 * time_drive + self.c1,
                self.c * float(safe_sin(x * (1.0 - x * x))) + self.l2 * time_drive + self.c2,
                q + self.k2 * x,
            ],
            dtype=np.float64,
        )
        next_state_float = np.asarray(safe_nan_to_num(next_state_float), dtype=np.float64)
        wrapped_state = np.asarray([wrap_mod1(value) for value in next_state_float], dtype=np.float64)
        self.state_u16 = float_vec_to_u16(wrapped_state)
        self.time_index += 1
        return u16_vec_to_float(self.state_u16)
