# Maintainer: zhaowenbo
"""Fuzzy entropy (FuzzyEn / FE) for 16-bit digital 4D chaotic trajectories.

The implementation here follows the common FuzzyEn workflow used in the time
series literature:

1. Build delay-embedding templates of dimension ``m`` and ``m + 1``.
2. Remove the local mean from each template.
3. Measure template distance with the Chebyshev / max norm.
4. Convert distance to a fuzzy similarity using a generalized Gaussian
   membership function.
5. Average fuzzy similarities for dimensions ``m`` and ``m + 1`` and take the
   logarithmic ratio.

This makes FuzzyEn a smooth alternative to ApEn / SampEn: instead of a hard
match / non-match threshold, it uses a fuzzy membership value between 0 and 1.

Paper-facing numeric semantics:
FuzzyEn is computed on the 16-bit quantized integer sequence rather than
directly on the raw floating-point trajectory. This keeps the metric aligned
with the project's fixed-point digital realization.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from metrics.common import ensure_sequence_1d, ensure_trajectory_2d
from utils.fixed16 import sequence_to_u16_array


DEFAULT_FE_RAW_LENGTH = 5000
DEFAULT_FE_TRANSIENT = 1000
DEFAULT_FE_EMBEDDING_DIM = 2
DEFAULT_FE_TAU = 1
DEFAULT_FE_FUZZY_EXPONENT = 2
DEFAULT_FE_R_RATIO = 0.2


class FuzzyEntropyResult(TypedDict):
    """Structured result for paper-facing FuzzyEn experiments."""

    fe_components: list[float]
    fe_mean: float
    raw_length: int
    test_length: int
    mode: str
    m: int
    r: float | list[float]
    tau: int
    n: int


def prepare_u16_sequence_for_fe(seq: np.ndarray) -> np.ndarray:
    """Convert one numeric sequence into the shared 16-bit integer domain.

    FuzzyEn is computed on the 16-bit quantized integer sequence rather than
    directly on the raw floating-point trajectory.
    """

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    return sequence_to_u16_array(raw)


def _resolve_tolerance(seq_u16: np.ndarray, r: float | None, eps: float) -> float:
    """Resolve the effective fuzzy tolerance for one quantized sequence."""

    std_value = float(np.std(seq_u16.astype(np.float64)))
    if r is None:
        return max(DEFAULT_FE_R_RATIO * std_value, eps)
    if r < 0:
        raise ValueError("r must be non-negative.")
    return max(float(r), eps)


def _build_centered_templates(seq_u16: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Build locally mean-centered delay-embedding templates."""

    if m <= 0:
        raise ValueError("m must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")

    length = len(seq_u16) - (m - 1) * tau
    if length <= 1:
        raise ValueError("Sequence is too short for the requested embedding setup.")

    base_indices = np.arange(length, dtype=np.int64)[:, None]
    delay_indices = (np.arange(m, dtype=np.int64) * tau)[None, :]
    templates = seq_u16[base_indices + delay_indices].astype(np.float64)
    templates -= np.mean(templates, axis=1, keepdims=True)
    return templates


def _average_fuzzy_similarity(
    templates: np.ndarray,
    tolerance: float,
    fuzzy_exponent: int,
    block_size: int = 128,
) -> float:
    """Average fuzzy similarities while excluding self matches.

    The membership function is the common generalized Gaussian form:

        mu(d) = exp(- (d / r) ** n)

    where ``d`` is the Chebyshev distance between two locally mean-centered
    embedding vectors, ``r`` is the tolerance, and ``n`` controls how sharply
    similarity decays.
    """

    count = int(templates.shape[0])
    if count <= 1:
        raise ValueError("At least two templates are required for FuzzyEn.")

    denominator = float(count * (count - 1))
    similarity_sum = 0.0
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        block = templates[start:stop]
        distances = np.max(np.abs(block[:, None, :] - templates[None, :, :]), axis=2)
        similarities = np.exp(-np.power(distances / tolerance, fuzzy_exponent))

        row_indices = np.arange(stop - start, dtype=np.int64)
        similarities[row_indices, start + row_indices] = 0.0
        similarity_sum += float(np.sum(similarities))

    return similarity_sum / denominator


def _fuzzy_entropy_1d(
    seq: np.ndarray,
    m: int = DEFAULT_FE_EMBEDDING_DIM,
    r: float | None = None,
    tau: int = DEFAULT_FE_TAU,
    n: int = DEFAULT_FE_FUZZY_EXPONENT,
    eps: float = 1e-12,
) -> tuple[float, int, int, float]:
    """Compute one-dimensional FuzzyEn on a 16-bit quantized sequence."""

    if n <= 0:
        raise ValueError("n must be positive.")

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    seq_u16 = prepare_u16_sequence_for_fe(raw)
    raw_length = int(len(seq_u16))
    templates_m = _build_centered_templates(seq_u16, m=m, tau=tau)
    templates_m1 = _build_centered_templates(seq_u16, m=m + 1, tau=tau)
    tolerance = _resolve_tolerance(seq_u16, r=r, eps=eps)

    phi_m = _average_fuzzy_similarity(templates_m, tolerance=tolerance, fuzzy_exponent=n)
    phi_m1 = _average_fuzzy_similarity(templates_m1, tolerance=tolerance, fuzzy_exponent=n)

    phi_m = max(phi_m, eps)
    phi_m1 = max(phi_m1, eps)
    fe_value = float(np.log(phi_m) - np.log(phi_m1))
    return fe_value, raw_length, raw_length, tolerance


def fuzzy_entropy(
    signal: np.ndarray,
    component: int | None = None,
    mode: str = "direct",
    m: int = DEFAULT_FE_EMBEDDING_DIM,
    r: float | None = None,
    tau: int = DEFAULT_FE_TAU,
    n: int = DEFAULT_FE_FUZZY_EXPONENT,
    eps: float = 1e-12,
) -> FuzzyEntropyResult:
    """Compute direct-mode FuzzyEn for 1D or 4D signals.

    The paper-facing default is direct numeric-sequence FuzzyEn. Before the
    entropy computation starts, each trajectory component is quantized into the
    common 16-bit integer domain. For a 4D trajectory, the metric is computed
    on each component independently and then averaged:

        fe_mean = (fe_x1 + fe_x2 + fe_x3 + fe_x4) / 4

    Parameters
    ----------
    signal:
        One-dimensional sequence or ``(T, 4)`` trajectory.
    component:
        Optional component selector for 4D input.
    mode:
        Only ``"direct"`` is supported in the current paper-facing workflow.
    m, r, tau, n:
        Standard FuzzyEn parameters. When ``r`` is ``None``, the effective
        tolerance is resolved per component as ``0.2 * std(u16_sequence)``.
    """

    if mode != "direct":
        raise ValueError("Only direct mode is supported for FuzzyEn in this project.")

    arr = np.asarray(signal)
    if arr.ndim == 1:
        fe_value, raw_length, test_length, tolerance = _fuzzy_entropy_1d(
            arr,
            m=m,
            r=r,
            tau=tau,
            n=n,
            eps=eps,
        )
        return {
            "fe_components": [float(fe_value)],
            "fe_mean": float(fe_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": mode,
            "m": int(m),
            "r": float(tolerance),
            "tau": int(tau),
            "n": int(n),
        }

    traj = ensure_trajectory_2d(np.asarray(signal, dtype=np.float64))
    dimension = int(traj.shape[1])
    if component is not None:
        if component < 0 or component >= dimension:
            raise ValueError(f"component must be in [0, {dimension - 1}].")
        fe_value, raw_length, test_length, tolerance = _fuzzy_entropy_1d(
            traj[:, component],
            m=m,
            r=r,
            tau=tau,
            n=n,
            eps=eps,
        )
        return {
            "fe_components": [float(fe_value)],
            "fe_mean": float(fe_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": mode,
            "m": int(m),
            "r": float(tolerance),
            "tau": int(tau),
            "n": int(n),
        }

    component_values: list[float] = []
    tolerance_values: list[float] = []
    raw_length = 0
    test_length = 0
    for component_index in range(dimension):
        fe_value, raw_length, test_length, tolerance = _fuzzy_entropy_1d(
            traj[:, component_index],
            m=m,
            r=r,
            tau=tau,
            n=n,
            eps=eps,
        )
        component_values.append(float(fe_value))
        tolerance_values.append(float(tolerance))

    return {
        "fe_components": component_values,
        "fe_mean": float(np.mean(component_values)),
        "raw_length": int(raw_length),
        "test_length": int(test_length),
        "mode": mode,
        "m": int(m),
        "r": tolerance_values if r is None else float(r),
        "tau": int(tau),
        "n": int(n),
    }


def fuzzy_entropy_smoke(traj: np.ndarray) -> np.ndarray:
    """Return four direct-mode FE values for smoke testing.

    Smoke tests only need an interface-level finite result, so the helper runs
    FuzzyEn on a short leading window instead of the full default 5000-point
    formal experiment.
    """

    arr = ensure_trajectory_2d(np.asarray(traj, dtype=np.float64))
    smoke_window = min(len(arr), 256)
    result = fuzzy_entropy(arr[:smoke_window])
    return np.asarray(result["fe_components"], dtype=np.float64)
