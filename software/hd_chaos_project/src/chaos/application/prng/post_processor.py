#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Lightweight word-level mixing stage for the chaos-based PRNG.

Given four 16-bit state outputs (x1..x4), this module:
  1. Concatenates into a 64-bit word: u_t = x1 || x2 || x3 || x4
  2. Applies inter-cycle mixing:
       y_t = u_t ⊕ ROTL64(u_{t-1}, 17) ⊕ ROTL64(u_{t-2}, 43)
  where ⊕ is bitwise XOR and ROTL64(·, r) is a 64-bit left circular rotation.

Hardware cost: two 64-bit delay registers, two fixed rotations, a few XOR gates.
"""

import numpy as np


class PostProcessor:
    """64-bit word-level post-processor with delay-line ROTL-XOR mixing.

    Parameters
    ----------
    word_bits : int
        Bit-width per chaotic state channel (default 16).
    """

    def __init__(self, word_bits: int = 16):
        self.word_bits = word_bits
        self.word_mask = (1 << word_bits) - 1   # per-channel mask
        self.total_bits = 4 * word_bits          # 64
        self.total_mask = (1 << self.total_bits) - 1

        # Delay registers: u_{t-1} and u_{t-2}
        self._u_prev1 = 0
        self._u_prev2 = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Core mixing
    # ------------------------------------------------------------------

    def mix(self, state_4d: np.ndarray) -> int:
        """Mix one 4D state vector into a 64-bit output word.

        Parameters
        ----------
        state_4d : np.ndarray  shape (4,), integer values in [0, 2^word_bits)

        Returns
        -------
        y_t : int  64-bit mixed output
        """
        # 1) Concatenate: u_t = x1 || x2 || x3 || x4  (little-endian: x1 at LSB)
        u_t = self._concat(state_4d)

        # 2) Mix: y_t = u_t ⊕ ROTL64(u_{t-1}, 17) ⊕ ROTL64(u_{t-2}, 43)
        if not self._initialized:
            # First output: no history yet, use u_t directly
            y_t = u_t
        else:
            y_t = (u_t
                   ^ self._rotl64(self._u_prev1, 17)
                   ^ self._rotl64(self._u_prev2, 43))

        # 3) Shift delay line
        self._u_prev2 = self._u_prev1
        self._u_prev1 = u_t
        self._initialized = True

        return y_t & self.total_mask

    def mix_batch(self, states: np.ndarray) -> np.ndarray:
        """Mix a batch of 4D states.

        Parameters
        ----------
        states : np.ndarray  shape (N, 4)

        Returns
        -------
        np.ndarray  shape (N,), dtype=np.uint64
        """
        outputs = np.zeros(states.shape[0], dtype=np.uint64)
        for i in range(states.shape[0]):
            outputs[i] = self.mix(states[i])
        return outputs

    def reset(self) -> None:
        """Clear delay registers."""
        self._u_prev1 = 0
        self._u_prev2 = 0
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _concat(self, state_4d: np.ndarray) -> int:
        """u_t = x1 || x2 || x3 || x4  (x1 at LSB, x4 at MSB)."""
        u = 0
        for j in range(4):
            v = int(state_4d[j]) & self.word_mask
            u |= v << (j * self.word_bits)
        return u

    @staticmethod
    def _rotl64(x: int, r: int) -> int:
        """64-bit left circular rotation."""
        r = r % 64
        if r == 0:
            return x & 0xFFFFFFFFFFFFFFFF
        return ((x << r) | (x >> (64 - r))) & 0xFFFFFFFFFFFFFFFF
