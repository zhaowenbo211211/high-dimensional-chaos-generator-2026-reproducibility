# Maintainer: zhaowenbo
"""Proposed n-D digital chaotic core with Mode-II bit-rotation enhancer.

Implements the current Section 3 semantics of ``pre_art_rev1.tex``:

- Q = 2**16. Full state X = (x_1, ..., x_{n-1}, s_0, ..., s_{n-2}), 2n-2 words.
- All right-hand sides use the same old state simultaneously:
    x_j^+  = (a_{j,1} * x_j + a_{j,n} * s_{n-1-j}) mod Q,  j = 1..n-1
    S      = x_{n-1} + s_0            (NOT reduced mod Q first; range 0..2Q-2)
    s_0^+  = (b * x_{n-2} + floor(4*S*(Q-S)/Q)) mod Q     (floor, not truncation)
    s_k^+  = s_{k-1},                 k = 1..n-2
- Mode II: with U = G(X), the current n words (U_x1..U_x{n-1}, U_s0) are packed
  little-endian, Z = sum(current[j] << (16*j)), rotated left by
  k_t = 1 + (t mod 15) over the total width 16*n, and unpacked over the first n
  words of U; the remaining n-2 delay words keep the base-update values.
  This is exactly tilde_H_k ∘ G.

The public state exposed to the metric pipeline is the n current words
(x_1..x_{n-1}, s_0); the delay words and the rotation phase are hidden state,
kept as instance attributes so that ``copy.deepcopy`` (used by the Wolf LE
estimator at every renormalization) synchronizes them.

This module deliberately does NOT reuse the old ``proposed_base.py`` /
``proposed_enhanced.py`` semantics (big-endian packing, rotated x_n written into
the delay line, float logistic nonlinearity, b*x_2 for all n).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from utils.fixed16 import U16_MOD, float_vec_to_u16, u16_vec_to_float

Q = U16_MOD  # 2**16
ROTATION_PERIOD = 15


def reference_base_update(
    state_full: Iterable[int],
    n: int,
    a_j1: Iterable[int],
    a_jn: Iterable[int],
    b: int,
) -> list[int]:
    """Scalar reference implementation of one base-map step U = G(X).

    ``state_full`` is the full 2n-2 word state
    ``[x_1, ..., x_{n-1}, s_0, ..., s_{n-2}]``. Plain Python integers only.
    """

    n = int(n)
    state = [int(v) for v in state_full]
    if len(state) != 2 * n - 2:
        raise ValueError("state_full must have 2n-2 words.")
    a1 = [int(v) for v in a_j1]
    an = [int(v) for v in a_jn]
    if len(a1) != n - 1 or len(an) != n - 1:
        raise ValueError("coefficient lists must have n-1 entries.")
    b = int(b)

    x = state[: n - 1]
    s = state[n - 1 :]  # s[0] = s_0, ..., s[n-2] = s_{n-2}

    u_x = [(a1[j] * x[j] + an[j] * s[n - 2 - j]) % Q for j in range(n - 1)]
    big_s = x[n - 2] + s[0]  # x_{n-1} + s_0, no pre-mod
    nonlinear = (4 * big_s * (Q - big_s)) // Q  # floor division
    u_s0 = (b * x[n - 3] + nonlinear) % Q  # b * x_{n-2}
    u_delay = s[:-1]  # s_k^+ = s_{k-1}, k = 1..n-2
    return u_x + [u_s0] + u_delay


def reference_rotate_pack(words: Iterable[int], k: int) -> list[int]:
    """Pack n words little-endian, rotate left by k over 16*n bits, unpack."""

    words = [int(v) for v in words]
    n = len(words)
    width = 16 * n
    mask = (1 << width) - 1
    z = 0
    for j, value in enumerate(words):
        z |= (value & 0xFFFF) << (16 * j)
    k = int(k) % width
    zr = ((z << k) | (z >> (width - k))) & mask
    return [(zr >> (16 * j)) & 0xFFFF for j in range(n)]


class ProposedNDRev1Map:
    """Proposed n-D digital chaotic map (new core) with Mode-II enhancer."""

    word_size: int = 16
    modulus: int = Q
    name: str = "ProposedNDRev1Map"

    def __init__(
        self,
        n: int,
        a_j1: Iterable[int],
        a_jn: Iterable[int],
        b: int,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        n = int(n)
        if n < 3:
            raise ValueError("n must be at least 3.")
        self.n = n
        self.a_j1 = [int(v) for v in a_j1]
        self.a_jn = [int(v) for v in a_jn]
        if len(self.a_j1) != n - 1 or len(self.a_jn) != n - 1:
            raise ValueError("coefficient lists must have n-1 entries.")
        self.b = int(b)
        self.rotation_step_index = 0
        self.state_u16 = np.zeros(self.n, dtype=np.int64)
        self.delay_u16 = np.zeros(self.n - 2, dtype=np.int64)
        self.reset(
            initial_state_u16=initial_state_u16,
            initial_state_float=initial_state_float,
        )

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        initial_history_u16: Iterable[int] | None = None,
    ) -> None:
        """Reset current words, delay history (zero by default), and phase."""

        if initial_state_float is not None:
            state = float_vec_to_u16(initial_state_float)
        elif initial_state_u16 is not None:
            state = np.asarray(list(initial_state_u16), dtype=np.int64) % self.modulus
        else:
            state = np.zeros(self.n, dtype=np.int64)
        if state.shape != (self.n,):
            raise ValueError(f"Initial state must be a {self.n}-element vector.")
        if initial_history_u16 is None:
            history = np.zeros(self.n - 2, dtype=np.int64)
        else:
            history = np.asarray(list(initial_history_u16), dtype=np.int64) % self.modulus
            if history.shape != (self.n - 2,):
                raise ValueError(f"Initial history must be a {self.n - 2}-element vector.")
        self.state_u16 = state.astype(np.int64)
        self.delay_u16 = history.astype(np.int64)
        self.rotation_step_index = 0

    def _full_state(self) -> list[int]:
        return [int(v) for v in self.state_u16] + [int(v) for v in self.delay_u16]

    def step(self) -> np.ndarray:
        """Advance by one Mode-II step and return the current words as float."""

        u_full = reference_base_update(self._full_state(), self.n, self.a_j1, self.a_jn, self.b)
        k = 1 + (self.rotation_step_index % ROTATION_PERIOD)
        rotated_current = reference_rotate_pack(u_full[: self.n], k)
        self.state_u16 = np.asarray(rotated_current, dtype=np.int64)
        self.delay_u16 = np.asarray(u_full[self.n :], dtype=np.int64)
        self.rotation_step_index += 1
        return u16_vec_to_float(self.state_u16)

    def iterate(self, num_steps: int, discard_transient: int = 0) -> np.ndarray:
        """Iterate the map and return a trajectory array with shape (T, n)."""

        if num_steps < 0 or discard_transient < 0:
            raise ValueError("num_steps and discard_transient must be non-negative.")
        for _ in range(discard_transient):
            self.step()
        traj = np.empty((num_steps, self.n), dtype=np.float64)
        for idx in range(num_steps):
            traj[idx] = self.step()
        return traj

    def get_state_u16(self) -> np.ndarray:
        return self.state_u16.copy()

    def get_state_float(self) -> np.ndarray:
        return u16_vec_to_float(self.state_u16)


def make_proposed_rev1_map(
    n: int,
    params: dict,
    initial_state_u16: Iterable[int] | None = None,
    initial_state_float: Iterable[float] | None = None,
) -> ProposedNDRev1Map:
    """Build a map instance from an R1-LY parameter dictionary."""

    chaos_map = ProposedNDRev1Map(
        n=n,
        a_j1=params["a_j1"],
        a_jn=params["a_jn"],
        b=params["b"],
    )
    chaos_map.name = "The proposed method"
    chaos_map.reset(
        initial_state_u16=initial_state_u16,
        initial_state_float=initial_state_float,
    )
    return chaos_map
