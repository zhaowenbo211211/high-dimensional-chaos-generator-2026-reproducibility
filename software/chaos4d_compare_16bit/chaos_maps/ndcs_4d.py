# Maintainer: zhaowenbo
"""4D-NDCS map under the project's unified 16-bit fixed-point framework."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMap4DBase
from utils.fixed16 import float_vec_to_u16, u16_vec_to_float


class NDCS4DMap(ChaosMap4DBase):
    """Natural mod-1 4D digital chaotic system.

    This map is a naturally mod-1 4D digital chaotic system, which makes it a
    good fit for 16-bit digital simulation comparisons. Its coefficients are
    integers, and that integer-friendliness can be revisited later in dedicated
    implementation-complexity analysis.
    """

    name = "NDCS4DMap"

    def __init__(
        self,
        P1: int = 199,
        P2: int = 33,
        P3: int = 99,
        P4: int = 43,
        P5: int = 193,
        P6: int = 2,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        self.P1 = int(P1)
        self.P2 = int(P2)
        self.P3 = int(P3)
        self.P4 = int(P4)
        self.P5 = int(P5)
        self.P6 = int(P6)
        super().__init__(initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the NDCS state using the shared 16-bit initialization rules."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

    def step(self) -> np.ndarray:
        """Advance one step of the natural mod-1 NDCS recurrence.

        This system is naturally defined on a mod-1 state space. We therefore
        convert the current u16 state to normalized floats in ``[0, 1)``,
        evaluate the original equations in float form, apply mod-1 wrapping via
        the shared fixed16 utilities, and quantize back to the unified 16-bit
        representation.
        """

        x1, x2, x3, x4 = (float(value) for value in u16_vec_to_float(self.state_u16))
        next_state_float = np.asarray(
            [
                self.P1 * x1 + self.P2 * x4,
                self.P3 * x2 + self.P4 * x4,
                self.P5 * x3 + self.P6 * x4,
                self.P6 * x3 + self.P5 * x4,
            ],
            dtype=np.float64,
        )
        self.state_u16 = float_vec_to_u16(next_state_float)
        return u16_vec_to_float(self.state_u16)
