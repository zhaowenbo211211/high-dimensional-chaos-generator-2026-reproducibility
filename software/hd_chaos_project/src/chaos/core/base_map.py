#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
R1-LY fixed-point n-dimensional chaotic map (base version).

Implements strict fixed-point arithmetic on Z_{2^w} with
logistic nonlinear core and delay-chain state management.
"""

from typing import Dict
import numpy as np


class HighDimChaosMap:
    """Strict fixed-point n-dimensional chaotic map."""

    def __init__(self, params: Dict, word_size: int = 16, seed: int = None):
        self.params = params
        self.n = len(params['a_j1']) + 1
        self.word_size = word_size
        self.mod = 1 << word_size

        self.a_j1 = params['a_j1'].astype(np.int64)
        self.a_jn = params['a_jn'].astype(np.int64)
        self.b = params['b']

        self.state_dim = 2 * self.n - 2
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.randint(0, self.mod, size=self.state_dim,
                                       dtype=np.int64)

        self._setup_indices()

    def _setup_indices(self):
        self.x_idx = np.arange(self.n - 1)
        self.s_idx = np.arange(self.n - 1, self.state_dim)

    def set_random_state(self, seed: int = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.randint(0, self.mod, size=self.state_dim,
                                       dtype=np.int64)

    def reset(self, seed: int = None) -> None:
        self.set_random_state(seed=seed)

    def set_state_from_physical(self, physical_state: np.ndarray) -> None:
        physical_state = np.asarray(physical_state, dtype=np.int64)
        if physical_state.shape[0] != self.n:
            raise ValueError(
                f"physical state dim should be {self.n}, got {physical_state.shape[0]}")
        self.state[self.x_idx] = physical_state[:-1]
        self.state[self.n - 1] = physical_state[-1]

    def next(self) -> np.ndarray:
        out = self.iterate(1)
        return out[0]

    def _mod1(self, x: np.ndarray) -> np.ndarray:
        return x & (self.mod - 1)

    def _logistic_core(self, x_nm1: int, s0: int) -> int:
        """Fixed-point logistic core: 4 * z * (1-z) / 2^w on Z_{2^w}."""
        mask = self.mod - 1
        z = (x_nm1 + s0) & mask
        inv_z = (~z) & mask
        temp = z * inv_z
        shift = self.word_size - 2
        if shift >= 0:
            result = temp >> shift
        else:
            result = (temp << (-shift)) & ((1 << self.word_size) - 1)
        return result & mask

    def iterate(self, steps: int = 1) -> np.ndarray:
        """Iterate the map for `steps` iterations, return trajectory."""
        outputs = np.zeros((steps, self.n), dtype=np.int64)

        for t in range(steps):
            x = self.state[self.x_idx]
            s = self.state[self.s_idx]

            x_n_new = (self.b * self.state[self.n - 3] +
                       self._logistic_core(x[-1], s[0])) & (self.mod - 1)

            new_state = np.zeros_like(self.state)
            for j in range(self.n - 1):
                delay_idx = self.s_idx[self.n - 2 - j]
                new_state[j] = (self.a_j1[j] * x[j] +
                                self.a_jn[j] * self.state[delay_idx]) & (self.mod - 1)

            new_state[self.n - 1] = x_n_new
            # Shift delay chain: s_{j} <- s_{j-1}
            new_state[self.s_idx[1:]] = self.state[self.s_idx[:-1]]

            self.state = new_state

            outputs[t, :-1] = self.state[self.x_idx]
            outputs[t, -1] = self.state[self.n - 1]

        return outputs

    def get_physical_state(self) -> np.ndarray:
        state = np.zeros(self.n, dtype=np.int64)
        state[:-1] = self.state[self.x_idx]
        state[-1] = self.state[self.n - 1]
        return state
