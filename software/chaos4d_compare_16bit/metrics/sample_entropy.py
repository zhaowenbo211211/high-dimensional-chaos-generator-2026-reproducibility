# Maintainer: zhaowenbo
"""Smoke-level sample entropy placeholder."""

from __future__ import annotations

import numpy as np

from metrics.common import ensure_trajectory_2d


def sample_entropy_smoke(traj: np.ndarray) -> np.ndarray:
    """Return a simple variability proxy as a future interface placeholder."""

    arr = ensure_trajectory_2d(traj)
    return np.std(np.diff(arr, axis=0), axis=0) if len(arr) > 1 else np.zeros(4)
