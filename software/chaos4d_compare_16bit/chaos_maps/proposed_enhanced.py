# Maintainer: zhaowenbo
"""Enhanced proposed maps with bit-rotation post-processing."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.proposed_base import Proposed4DBaseMap, ProposedNDBaseMap
from utils.fixed16 import U16_MOD, u16_vec_to_float
from utils.r1ly import DEFAULT_R1LY_ETA


class ProposedNDEnhancedMap(ProposedNDBaseMap):
    """Enhanced n-dimensional version of the proposed delayed system.

    One base-map update is computed first. The resulting `n` 16-bit states are
    then packed into one `16n`-bit integer, rotated left by

    `k_t = 1 + (t mod 15)`,

    and unpacked back into `n` 16-bit states.
    """

    name = "ProposedNDEnhancedMap"

    def __init__(
        self,
        n: int = 4,
        A_odd: int = 371,
        B_even: int = 50,
        Delta: int = 2,
        eta: int = DEFAULT_R1LY_ETA,
        kappa: int = 4,
        tau: int = 1,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        xn_delay_u16: Iterable[int] | None = None,
    ) -> None:
        self.rotation_step_index = 0
        super().__init__(
            n=n,
            A_odd=A_odd,
            B_even=B_even,
            Delta=Delta,
            eta=eta,
            kappa=kappa,
            tau=tau,
            initial_state_u16=initial_state_u16,
            initial_state_float=initial_state_float,
            xn_delay_u16=xn_delay_u16,
        )

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        xn_delay_u16: Iterable[int] | None = None,
    ) -> None:
        """Reset state, delay chain, and cyclic rotation counter."""

        super().reset(
            initial_state_u16=initial_state_u16,
            initial_state_float=initial_state_float,
            xn_delay_u16=xn_delay_u16,
        )
        self.rotation_step_index = 0

    def _current_rotation_step(self) -> int:
        """Return the periodic rotate-left step `k_t = 1 + (t mod 15)`."""

        return 1 + (self.rotation_step_index % 15)

    @staticmethod
    def _rol_bits(z: int, k: int, width: int) -> int:
        """Rotate an unsigned integer left by `k` bits over the given width."""

        mask = (1 << width) - 1
        shift = int(k) % int(width)
        if shift == 0:
            return int(z) & mask
        value = int(z) & mask
        return ((value << shift) & mask) | (value >> (width - shift))

    def _pack_state_to_uint(self, state: Iterable[int]) -> int:
        """Pack `n` 16-bit state values into one big integer."""

        packed = 0
        for value in state:
            packed = (packed << 16) | (int(value) & 0xFFFF)
        return packed

    def _unpack_uint_to_state(self, value: int) -> np.ndarray:
        """Unpack one big integer into `n` 16-bit state values."""

        unpacked = np.empty(self.n, dtype=np.int64)
        working = int(value)
        for index in range(self.n - 1, -1, -1):
            unpacked[index] = working & 0xFFFF
            working >>= 16
        return unpacked

    def step(self) -> np.ndarray:
        """Advance the base map once, then apply cyclic bit rotation."""

        base_next_state = self._compute_base_next_state()
        packed = self._pack_state_to_uint(base_next_state)
        rotated = self._rol_bits(packed, self._current_rotation_step(), width=16 * self.n)
        enhanced_state = self._unpack_uint_to_state(rotated)

        self.state_u16 = np.asarray(enhanced_state, dtype=np.int64) % U16_MOD
        if self.n > 2:
            self.xn_delay_line = np.roll(self.xn_delay_line, -1)
            self.xn_delay_line[-1] = int(self.state_u16[-1])
        self.rotation_step_index += 1
        return u16_vec_to_float(self.state_u16)


class Proposed4DEnhancedMap(ProposedNDEnhancedMap):
    """4D-compatible wrapper of the generalized enhanced proposed map."""

    name = "Proposed4DEnhancedMap"

    def __init__(
        self,
        A_odd: int = 371,
        B_even: int = 50,
        Delta: int = 2,
        eta: int = DEFAULT_R1LY_ETA,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        x4_delay_u16: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            n=4,
            A_odd=A_odd,
            B_even=B_even,
            Delta=Delta,
            eta=eta,
            initial_state_u16=initial_state_u16,
            initial_state_float=initial_state_float,
            xn_delay_u16=None if x4_delay_u16 is None else (x4_delay_u16[1], x4_delay_u16[0]),
        )

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        x4_delay_u16: tuple[int, int] | None = None,
        xn_delay_u16: Iterable[int] | None = None,
    ) -> None:
        """Reset the 4D enhanced wrapper while preserving legacy delay naming."""

        if x4_delay_u16 is not None:
            xn_delay_u16 = (x4_delay_u16[1], x4_delay_u16[0])
        super().reset(
            initial_state_u16=initial_state_u16,
            initial_state_float=initial_state_float,
            xn_delay_u16=xn_delay_u16,
        )

    @property
    def x4_z1(self) -> int:
        return int(self.xn_delay_line[-1]) if len(self.xn_delay_line) >= 1 else 0

    @x4_z1.setter
    def x4_z1(self, value: int) -> None:
        if len(self.xn_delay_line) >= 1:
            self.xn_delay_line[-1] = int(value) % U16_MOD

    @property
    def x4_z2(self) -> int:
        return int(self.xn_delay_line[0]) if len(self.xn_delay_line) >= 2 else 0

    @x4_z2.setter
    def x4_z2(self, value: int) -> None:
        if len(self.xn_delay_line) >= 2:
            self.xn_delay_line[0] = int(value) % U16_MOD

    @staticmethod
    def _pack_state_to_u64(x1: int, x2: int, x3: int, x4: int) -> int:
        """Legacy 4D-specific pack helper retained for compatibility tests."""

        return (
            ((int(x1) & 0xFFFF) << 48)
            | ((int(x2) & 0xFFFF) << 32)
            | ((int(x3) & 0xFFFF) << 16)
            | (int(x4) & 0xFFFF)
        )

    @staticmethod
    def _rol64(z: int, k: int) -> int:
        """Legacy 64-bit rotate helper retained for compatibility tests."""

        return ProposedNDEnhancedMap._rol_bits(z, k, 64)

    @staticmethod
    def _unpack_u64_to_state(z: int) -> tuple[int, int, int, int]:
        """Legacy 4D-specific unpack helper retained for compatibility tests."""

        value = int(z) & ((1 << 64) - 1)
        return (
            (value >> 48) & 0xFFFF,
            (value >> 32) & 0xFFFF,
            (value >> 16) & 0xFFFF,
            value & 0xFFFF,
        )
