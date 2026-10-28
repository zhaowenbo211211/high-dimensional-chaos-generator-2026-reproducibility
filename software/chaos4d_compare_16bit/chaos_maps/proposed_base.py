# Maintainer: zhaowenbo
"""Base maps specialized from the paper's original n-dimensional system."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMapNDBase
from utils.fixed16 import U16_MOD, float_to_u16, u16_to_float, u16_vec_to_float
from utils.r1ly import DEFAULT_R1LY_ETA, synthesize_r1_ly_params


class ProposedNDBaseMap(ChaosMapNDBase):
    """Paper-consistent n-dimensional base map under unified 16-bit semantics.

    The implemented recurrence extends the existing 4D specialization into the
    paper-facing delayed n-dimensional form:

    `x_j(i+1) = a_{j,1} x_j(i) + a_{j,n} x_n(i-(n-1-j)),  j = 1,...,n-1`

    `x_n(i+1) = b x_2(i) + 4 * (x_{n-1}(i) + x_n(i)) * (1 - (x_{n-1}(i) + x_n(i)))`
    """

    name = "ProposedNDBaseMap"

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
        params = synthesize_r1_ly_params(
            n=n,
            A_odd=A_odd,
            B_even=B_even,
            Delta=Delta,
            eta=eta,
            kappa=kappa,
            tau=tau,
        )

        self.n = int(params["n"])
        self.A_odd = int(params["A_odd_final"])
        self.B_even = int(params["B_even_final"])
        self.Delta = int(params["Delta"])
        self.eta = int(params["eta_final"])
        self.kappa = int(params["kappa"])
        self.tau = int(params["tau_final"])
        self.a_linear = [int(value) for value in params["a_linear"]]
        self.a_coupling = [int(value) for value in params["a_coupling"]]
        self.b = int(params["b"])
        self.r1_ly_metadata = params
        self.xn_delay_line = np.zeros(max(self.n - 2, 0), dtype=np.int64)

        super().__init__(n=self.n, initial_state_u16=initial_state_u16)

        if initial_state_float is not None or xn_delay_u16 is not None:
            self.reset(
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
        """Reset state and delayed `x_n` chain."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)

        if xn_delay_u16 is None:
            self.xn_delay_line = np.zeros(max(self.n - 2, 0), dtype=np.int64)
            return

        delay_values = np.asarray(list(xn_delay_u16), dtype=np.int64) % U16_MOD
        expected_length = max(self.n - 2, 0)
        if delay_values.shape != (expected_length,):
            raise ValueError(f"xn_delay_u16 must provide exactly {expected_length} delay values.")
        self.xn_delay_line = delay_values.astype(np.int64)

    def _delayed_xn_for_channel(self, channel_index: int, current_xn: int) -> int:
        """Return the required delayed `x_n` sample for channel `channel_index`."""

        if channel_index == self.n - 2:
            return int(current_xn)
        return int(self.xn_delay_line[channel_index])

    def _compute_base_next_state(self) -> np.ndarray:
        """Compute one base-map update without writing the result back."""

        current_state = np.asarray(self.state_u16, dtype=np.int64)
        x_n = int(current_state[-1])
        next_state = np.empty(self.n, dtype=np.int64)

        for channel_index in range(self.n - 1):
            delayed_xn = self._delayed_xn_for_channel(channel_index, x_n)
            next_state[channel_index] = (
                self.a_linear[channel_index] * int(current_state[channel_index])
                + self.a_coupling[channel_index] * delayed_xn
            ) % U16_MOD

        x2_float = u16_to_float(int(current_state[1]))
        x_prev_float = u16_to_float(int(current_state[-2]))
        x_n_float = u16_to_float(x_n)
        logistic_sum = x_prev_float + x_n_float
        next_state[-1] = float_to_u16(self.b * x2_float + 4.0 * logistic_sum * (1.0 - logistic_sum))
        return next_state

    def step(self) -> np.ndarray:
        """Advance one step using the strict n-dimensional base recurrence."""

        old_xn = int(self.state_u16[-1])
        next_state = self._compute_base_next_state()
        self.state_u16 = np.asarray(next_state, dtype=np.int64) % U16_MOD
        if self.n > 2:
            self.xn_delay_line = np.roll(self.xn_delay_line, -1)
            self.xn_delay_line[-1] = old_xn
        return u16_vec_to_float(self.state_u16)


class Proposed4DBaseMap(ProposedNDBaseMap):
    """4D-compatible wrapper of the generalized proposed base map."""

    name = "Proposed4DBaseMap"

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

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        x4_delay_u16: tuple[int, int] | None = None,
        xn_delay_u16: Iterable[int] | None = None,
    ) -> None:
        """Reset the 4D wrapper while preserving the legacy delay argument."""

        if x4_delay_u16 is not None:
            xn_delay_u16 = (x4_delay_u16[1], x4_delay_u16[0])
        super().reset(
            initial_state_u16=initial_state_u16,
            initial_state_float=initial_state_float,
            xn_delay_u16=xn_delay_u16,
        )

    @property
    def a11(self) -> int:
        return self.a_linear[0]

    @property
    def a21(self) -> int:
        return self.a_linear[1]

    @property
    def a31(self) -> int:
        return self.a_linear[2]

    @property
    def a14(self) -> int:
        return self.a_coupling[0]

    @property
    def a24(self) -> int:
        return self.a_coupling[1]

    @property
    def a34(self) -> int:
        return self.a_coupling[2]
