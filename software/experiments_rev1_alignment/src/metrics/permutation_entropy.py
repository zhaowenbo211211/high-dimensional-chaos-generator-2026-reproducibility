# Maintainer: zhaowenbo
"""Permutation entropy (PE / PEn) for 16-bit digital 4D chaotic trajectories.

This implementation follows the standard Bandt-Pompe permutation entropy
definition:

1. Build ordinal patterns from delay-embedding vectors of dimension ``m`` and
   delay ``tau``.
2. Estimate the frequency distribution of observed ordinal patterns.
3. Compute the Shannon entropy of that distribution.
4. Optionally normalize by ``log(m!)``.

Paper-facing numeric semantics:
Permutation entropy is computed on the 16-bit quantized integer sequence rather
than directly on the raw floating-point trajectory. This keeps the metric
aligned with the project's fixed-point digital realization.

Tie handling:
Because the analysis sequence is quantized to 16-bit integers, ties are
possible and must be handled deterministically. The current implementation uses
stable ranking through ``np.argsort(..., kind="mergesort")``. Equal values are
therefore ordered by their original time position inside each embedding vector.
This gives a deterministic and reproducible tie-breaking rule.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TypedDict

import numpy as np

from metrics.common import ensure_sequence_1d, ensure_trajectory_2d
from utils.fixed16 import sequence_to_u16_array


DEFAULT_PE_RAW_LENGTH = 5000
DEFAULT_PE_TRANSIENT = 1000
DEFAULT_PE_EMBEDDING_DIM = 5
DEFAULT_PE_TAU = 1
DEFAULT_PE_NORMALIZED = True


class PermutationEntropyResult(TypedDict):
    """Structured result for paper-facing permutation-entropy experiments."""

    pe_components: list[float]
    pe_mean: float
    raw_length: int
    test_length: int
    mode: str
    m: int
    tau: int
    normalized: bool


def prepare_u16_sequence_for_pe(seq: np.ndarray) -> np.ndarray:
    """Convert one numeric sequence into the shared 16-bit integer domain.

    Permutation entropy is computed on the 16-bit quantized integer sequence
    rather than directly on the raw floating-point trajectory.
    """

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    return sequence_to_u16_array(raw)


def _ordinal_pattern(window: np.ndarray) -> tuple[int, ...]:
    """Return the ordinal pattern for one embedding window.

    Stable sorting makes the tie rule deterministic: equal values keep their
    original temporal order.
    """

    return tuple(int(index) for index in np.argsort(window, kind="mergesort"))


def _permutation_entropy_1d(
    seq: np.ndarray,
    m: int = DEFAULT_PE_EMBEDDING_DIM,
    tau: int = DEFAULT_PE_TAU,
    normalized: bool = DEFAULT_PE_NORMALIZED,
    eps: float = 1e-12,
) -> tuple[float, int, int]:
    """Compute one-dimensional permutation entropy on a u16-quantized sequence."""

    if m < 2:
        raise ValueError("m must be at least 2.")
    if tau <= 0:
        raise ValueError("tau must be positive.")

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")

    seq_u16 = prepare_u16_sequence_for_pe(raw)
    raw_length = int(len(seq_u16))
    window_count = raw_length - (m - 1) * tau
    if window_count <= 0:
        raise ValueError("Sequence is too short for the requested m and tau.")

    offsets = np.arange(m, dtype=np.int64) * tau
    counts: Counter[tuple[int, ...]] = Counter()
    for start in range(window_count):
        window = seq_u16[start + offsets]
        counts[_ordinal_pattern(window)] += 1

    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / float(window_count)
    pe_value = -float(np.sum(probabilities * np.log(probabilities + eps)))
    if normalized:
        pe_value /= float(np.log(math.factorial(m)))
    return pe_value, raw_length, raw_length


def permutation_entropy(
    signal: np.ndarray,
    component: int | None = None,
    mode: str = "direct",
    m: int = DEFAULT_PE_EMBEDDING_DIM,
    tau: int = DEFAULT_PE_TAU,
    normalized: bool = DEFAULT_PE_NORMALIZED,
    eps: float = 1e-12,
) -> PermutationEntropyResult:
    """Compute standard Bandt-Pompe permutation entropy.

    The paper-facing default is direct numeric-sequence permutation entropy.
    Before ordinal-pattern counting starts, each trajectory component is
    quantized into the shared 16-bit integer domain. For a 4D trajectory, the
    metric is computed on each component independently and then averaged:

        pe_mean = (pe_x1 + pe_x2 + pe_x3 + pe_x4) / 4
    """

    if mode != "direct":
        raise ValueError("Only direct mode is supported for permutation entropy in this project.")

    arr = np.asarray(signal)
    if arr.ndim == 1:
        pe_value, raw_length, test_length = _permutation_entropy_1d(
            arr,
            m=m,
            tau=tau,
            normalized=normalized,
            eps=eps,
        )
        return {
            "pe_components": [float(pe_value)],
            "pe_mean": float(pe_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": mode,
            "m": int(m),
            "tau": int(tau),
            "normalized": bool(normalized),
        }

    traj = ensure_trajectory_2d(np.asarray(signal, dtype=np.float64))
    dimension = int(traj.shape[1])
    if component is not None:
        if component < 0 or component >= dimension:
            raise ValueError(f"component must be in [0, {dimension - 1}].")
        pe_value, raw_length, test_length = _permutation_entropy_1d(
            traj[:, component],
            m=m,
            tau=tau,
            normalized=normalized,
            eps=eps,
        )
        return {
            "pe_components": [float(pe_value)],
            "pe_mean": float(pe_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": mode,
            "m": int(m),
            "tau": int(tau),
            "normalized": bool(normalized),
        }

    component_values: list[float] = []
    raw_length = 0
    test_length = 0
    for component_index in range(dimension):
        pe_value, raw_length, test_length = _permutation_entropy_1d(
            traj[:, component_index],
            m=m,
            tau=tau,
            normalized=normalized,
            eps=eps,
        )
        component_values.append(float(pe_value))

    return {
        "pe_components": component_values,
        "pe_mean": float(np.mean(component_values)),
        "raw_length": int(raw_length),
        "test_length": int(test_length),
        "mode": mode,
        "m": int(m),
        "tau": int(tau),
        "normalized": bool(normalized),
    }


def permutation_entropy_smoke(traj: np.ndarray) -> np.ndarray:
    """Return four direct-mode PE values for smoke testing."""

    arr = ensure_trajectory_2d(np.asarray(traj, dtype=np.float64))
    smoke_window = min(len(arr), 256)
    result = permutation_entropy(arr[:smoke_window])
    return np.asarray(result["pe_components"], dtype=np.float64)
