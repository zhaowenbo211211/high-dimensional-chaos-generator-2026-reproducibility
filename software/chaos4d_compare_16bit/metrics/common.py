# Maintainer: zhaowenbo
"""Common metric helpers."""

from __future__ import annotations

import copy
from typing import Protocol

import numpy as np

from utils.fixed16 import U16_MOD


def ensure_trajectory_2d(traj: np.ndarray) -> np.ndarray:
    """Validate and normalize a trajectory array of shape `(T, n)`."""

    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError("Trajectory must have shape (T, n) with n >= 1.")
    return arr


def ensure_sequence_1d(seq: np.ndarray) -> np.ndarray:
    """Validate and normalize a one-dimensional float sequence."""

    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    return arr


def clone_map_instance(map_instance):
    """Deep-copy a chaotic-map instance for metric-side parallel evolution."""

    return copy.deepcopy(map_instance)


class ChaosMapNDProtocol(Protocol):
    """Minimal protocol required by metric-side map estimators."""

    name: str
    n: int
    state_u16: np.ndarray

    def step(self) -> np.ndarray:
        """Advance the map by one step."""

    def get_state_u16(self) -> np.ndarray:
        """Return the current u16 state."""

    def get_state_float(self) -> np.ndarray:
        """Return the current float state."""


def euclidean_distance_nd(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Return the Euclidean distance between two float states."""

    arr_a = np.asarray(state_a, dtype=np.float64)
    arr_b = np.asarray(state_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1:
        raise ValueError("Both states must be one-dimensional and have the same shape.")
    return float(np.linalg.norm(arr_a - arr_b))


def validate_nd_map_interface(map_instance: ChaosMapNDProtocol) -> None:
    """Validate that a map instance exposes the expected 16-bit map interface."""

    state_u16 = np.asarray(map_instance.get_state_u16(), dtype=np.int64)
    state_float = np.asarray(map_instance.get_state_float(), dtype=np.float64)
    if state_u16.ndim != 1 or len(state_u16) < 1:
        raise ValueError("Map must expose a one-dimensional u16 state.")
    if state_float.shape != state_u16.shape:
        raise ValueError("Map float and u16 states must have the same shape.")
    if np.any(state_u16 < 0) or np.any(state_u16 >= U16_MOD):
        raise ValueError("Map u16 state must stay within [0, 65535].")


def build_directional_u16_perturbation(direction: np.ndarray, perturb_lsb: int = 1) -> np.ndarray:
    """Build a non-zero u16-domain perturbation aligned with a float direction.

    The output is an integer offset vector that preserves the dominant
    separation direction as much as the 16-bit grid allows, while guaranteeing
    at least one component changes by ``perturb_lsb``.
    """

    if perturb_lsb <= 0:
        raise ValueError("perturb_lsb must be positive.")

    direction_arr = np.asarray(direction, dtype=np.float64)
    if direction_arr.ndim != 1 or len(direction_arr) < 1:
        raise ValueError("direction must be a one-dimensional vector.")

    norm = float(np.linalg.norm(direction_arr))
    if norm <= 1e-15:
        perturb = np.zeros(len(direction_arr), dtype=np.int64)
        perturb[0] = int(perturb_lsb)
        return perturb

    unit_direction = direction_arr / norm
    dominant_scale = np.max(np.abs(unit_direction))
    scaled = unit_direction / max(dominant_scale, 1e-15)
    perturb = np.rint(scaled * float(perturb_lsb)).astype(np.int64)

    if not np.any(perturb):
        dominant_dim = int(np.argmax(np.abs(unit_direction)))
        perturb[dominant_dim] = int(perturb_lsb)

    return perturb


class ChaosMap4DProtocol(ChaosMapNDProtocol, Protocol):
    """Backward-compatible alias for existing 4D metric type annotations."""


def euclidean_distance_4d(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Backward-compatible 4D alias of the Euclidean distance helper."""

    return euclidean_distance_nd(state_a, state_b)


def validate_4d_map_interface(map_instance: ChaosMap4DProtocol) -> None:
    """Backward-compatible 4D alias of the map interface validator."""

    validate_nd_map_interface(map_instance)
