#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Bit-rotation based dynamics enhancer (Algorithm 2).

After each base map iteration, packs n w-bit coordinates into one nw-bit word,
applies a left cyclic rotation by k_t bits, then unpacks back to n coordinates.
The rotation step k_t = 1 + (t mod (w-1)) cycles through {1, ..., w-1}.
"""

from typing import Dict
import numpy as np

from chaos.core.base_map import HighDimChaosMap


class HighDimChaosMapEnhanced(HighDimChaosMap):
    """n-dimensional chaotic map with bit-rotation dynamics enhancement."""

    def __init__(self, params: Dict, word_size: int = 16, seed: int = None):
        super().__init__(params, word_size=word_size, seed=seed)

        self.total_bits: int = self.n * self.word_size
        self.word_mask: int = (1 << self.word_size) - 1
        self.total_mask: int = (1 << self.total_bits) - 1
        self.iter_count: int = 0

    # ---- bit-level utilities ----

    def _pack_state_bits(self, x_phys: np.ndarray) -> int:
        """Pi: (Z_{2^w})^n -> Z_{2^{nw}}, little-endian packing."""
        Z = 0
        for j in range(self.n):
            v = int(x_phys[j]) & self.word_mask
            Z |= v << (j * self.word_size)
        return Z

    def _unpack_state_bits(self, Z: int) -> np.ndarray:
        """Pi^{-1}: Z_{2^{nw}} -> (Z_{2^w})^n."""
        xs = np.zeros(self.n, dtype=np.int64)
        for j in range(self.n):
            xs[j] = (Z >> (j * self.word_size)) & self.word_mask
        return xs

    def _rol_total(self, Z: int, k: int) -> int:
        """ROL_{nw}(Z, k): left cyclic shift on nw-bit word."""
        k = int(k) % self.total_bits
        if k == 0:
            return Z & self.total_mask
        return ((Z << k) & self.total_mask) | (Z >> (self.total_bits - k))

    # ---- rotation schedule ----

    def _next_rotation_k(self) -> int:
        """Periodic rotation step: k_t = 1 + (iter_count mod (w-1))."""
        return 1 + (self.iter_count % (self.word_size - 1))

    # ---- main iteration ----

    def iterate(self, steps: int = 1) -> np.ndarray:
        """Iterate with bit-rotation enhancement: x^{(t+1)} = H_{k_t}(G(x^{(t)}))."""
        outputs = np.zeros((steps, self.n), dtype=np.int64)

        for _t in range(steps):
            super().iterate(1)
            x_phys = self.get_physical_state()

            k = self._next_rotation_k()
            Z = self._pack_state_bits(x_phys)
            Z_rot = self._rol_total(Z, k)
            x_rot = self._unpack_state_bits(Z_rot)

            self.state[self.x_idx] = x_rot[:-1]
            self.state[self.n - 1] = x_rot[-1]

            outputs[_t, :] = x_rot
            self.iter_count += 1

        return outputs

    def next(self) -> np.ndarray:
        return self.iterate(1)[0]

    # ---- debug / state management ----

    def debug_k_sequence(self, length: int = 32):
        """Print and return the first `length` k values (advances iter_count)."""
        ks = []
        for _ in range(length):
            ks.append(self._next_rotation_k())
            self.iter_count += 1
        print("k sequence:", ks)
        return ks

    def get_rotation_schedule_state(self) -> Dict:
        return {"iter_count": self.iter_count}

    def set_rotation_schedule_state(self, state: Dict):
        self.iter_count = int(state["iter_count"])

    def reset(self, seed: int = None) -> None:
        """Reset map state and rotation counter."""
        super().reset(seed=seed)
        self.iter_count = 0
