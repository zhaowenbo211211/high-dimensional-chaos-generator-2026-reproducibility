# Maintainer: zhaowenbo
"""Trajectory helpers for experiments."""

from __future__ import annotations

import numpy as np


def summarize_trajectory(traj: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-dimension mean and variance for a trajectory."""

    return {
        "mean": np.mean(traj, axis=0),
        "var": np.var(traj, axis=0),
    }
