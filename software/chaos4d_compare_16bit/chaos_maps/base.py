# Maintainer: zhaowenbo
"""Abstract base classes for chaotic maps with 16-bit state semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from utils.fixed16 import U16_MOD, float_vec_to_u16, u16_vec_to_float
from utils.initializers import get_common_initial_state_u16, get_dimension_initial_state_u16


class ChaosMapNDBase(ABC):
    """Abstract base class for n-dimensional maps under shared 16-bit semantics."""

    word_size: int = 16
    modulus: int = U16_MOD
    name: str = "ChaosMapNDBase"

    def __init__(self, n: int, initial_state_u16: Iterable[int] | None = None) -> None:
        if int(n) < 2:
            raise ValueError("n must be at least 2.")
        self.n = int(n)
        self.state_u16 = np.zeros(self.n, dtype=np.int64)
        self.reset(initial_state_u16=initial_state_u16)

    @property
    def state_float(self) -> np.ndarray:
        """Current state represented as normalized float values in `[0, 1)`."""

        return u16_vec_to_float(self.state_u16)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the internal state using either float or u16 initialization."""

        if initial_state_float is not None:
            state = float_vec_to_u16(initial_state_float)
        elif initial_state_u16 is not None:
            state = np.asarray(list(initial_state_u16), dtype=np.int64) % self.modulus
        else:
            state = get_dimension_initial_state_u16(self.n)
        if state.shape != (self.n,):
            raise ValueError(f"Initial state must be a {self.n}-element vector.")
        self.state_u16 = state.astype(np.int64)

    @abstractmethod
    def step(self) -> np.ndarray:
        """Advance the map by one step and return the new state as float64."""

    def iterate(self, num_steps: int, discard_transient: int = 0) -> np.ndarray:
        """Iterate the map and return a trajectory array with shape `(T, n)`."""

        if num_steps < 0 or discard_transient < 0:
            raise ValueError("num_steps and discard_transient must be non-negative.")
        for _ in range(discard_transient):
            self.step()
        traj = np.empty((num_steps, self.n), dtype=np.float64)
        for idx in range(num_steps):
            traj[idx] = self.step()
        return traj

    def get_state_u16(self) -> np.ndarray:
        """Return a copy of the current integer state."""

        return self.state_u16.copy()

    def get_state_float(self) -> np.ndarray:
        """Return a copy of the current normalized float state."""

        return self.state_float.copy()


class ChaosMap4DBase(ChaosMapNDBase):
    """Abstract base class for all 4D chaotic maps in this project.

    Subclasses must implement :meth:`step` using the shared 16-bit state
    semantics. Public getters expose both integer and normalized float views.
    """

    name: str = "ChaosMap4DBase"

    def __init__(self, initial_state_u16: Iterable[int] | None = None) -> None:
        super().__init__(n=4, initial_state_u16=initial_state_u16)

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
    ) -> None:
        """Reset the internal 4D state using project-default initialization."""

        if initial_state_u16 is None and initial_state_float is None:
            initial_state_u16 = get_common_initial_state_u16()
        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)
