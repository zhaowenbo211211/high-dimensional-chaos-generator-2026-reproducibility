#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Orbit diagnostics: Floyd cycle detection, orbit sampling, and digital system analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Protocol

import numpy as np

# ---- type aliases ----
State = np.ndarray
StepFn = Callable[[State], State]


class SupportsNext(Protocol):
    """Any object with a next() -> np.ndarray method can be diagnosed."""
    def next(self) -> np.ndarray: ...


def make_step_fn_from_map(chaos_map: SupportsNext) -> StepFn:
    """Wrap a chaos map with .next() into a step_fn(x) signature."""
    def step_fn(_: State) -> State:
        return np.asarray(chaos_map.next(), dtype=np.int64)
    return step_fn


@dataclass
class OrbitStatsConfig:
    """Configuration for orbit diagnostics."""
    n_dims: int
    word_size: int
    max_steps: int = 100_000
    transient: int = 1_000
    hist_bins: int = 256
    sample_stride: int = 1


# ---- Floyd cycle detection ----

def estimate_period(step_fn: StepFn, x0: State, max_steps: int,
                    ) -> Tuple[int | None, int | None]:
    """Floyd's tortoise-and-hare cycle detection.

    Returns (mu, lambda):
      mu  - transient length before cycle entry
      lambda - cycle length; None if not found within max_steps
    """
    tortoise = np.array(x0, copy=True)
    hare = np.array(x0, copy=True)

    for _step in range(1, max_steps + 1):
        tortoise = step_fn(tortoise)
        hare = step_fn(hare)
        hare = step_fn(hare)
        if np.array_equal(tortoise, hare):
            break
    else:
        return None, None

    mu = 0
    tortoise = np.array(x0, copy=True)
    while not np.array_equal(tortoise, hare) and mu <= max_steps:
        tortoise = step_fn(tortoise)
        hare = step_fn(hare)
        mu += 1

    if mu > max_steps:
        return None, None

    lam = 1
    hare = step_fn(tortoise)
    while not np.array_equal(tortoise, hare) and lam <= max_steps:
        hare = step_fn(hare)
        lam += 1

    if lam > max_steps:
        return mu, None
    return mu, lam


# ---- Orbit sampling ----

def sample_orbit(step_fn: StepFn, x0: State, config: OrbitStatsConfig) -> np.ndarray:
    """Sample orbit after discarding transient steps. Returns normalized [0,1) array."""
    n, w = config.n_dims, config.word_size
    mask = (1 << w) - 1

    x = np.array(x0, copy=True)
    for _ in range(config.transient):
        x = step_fn(x)

    samples = []
    for t in range(config.max_steps):
        x = step_fn(x)
        if t % config.sample_stride == 0:
            xx = np.asarray(x, dtype=np.int64) & mask
            samples.append(xx.astype(np.float64) / (1 << w))

    return np.vstack(samples)


def compute_marginal_histograms(samples: np.ndarray,
                                config: OrbitStatsConfig) -> Dict[str, Any]:
    """Compute per-dimension histograms and basic statistics."""
    n, bins = config.n_dims, config.hist_bins
    hist_list = []
    stats_list = []

    for j in range(n):
        data = samples[:, j]
        hist, edges = np.histogram(data, bins=bins, range=(0.0, 1.0), density=True)
        hist_list.append((hist, edges))
        stats_list.append({
            "mean": float(np.mean(data)),
            "var": float(np.var(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        })

    return {"histograms": hist_list, "per_dim_stats": stats_list}


# ---- Unified diagnosis ----

def analyze_digital_system(step_fn: StepFn, x0: State, config: OrbitStatsConfig,
                           ) -> Dict[str, Any]:
    """Unified diagnosis: cycle detection + marginal distribution analysis."""
    mu, lam = estimate_period(step_fn, np.array(x0, copy=True), config.max_steps)

    samples = sample_orbit(step_fn, x0, config)
    dist_info = compute_marginal_histograms(samples, config)

    return {
        "period_mu": mu,
        "period_lambda": lam,
        "num_samples": samples.shape[0],
        "samples": samples,
        **dist_info,
    }
