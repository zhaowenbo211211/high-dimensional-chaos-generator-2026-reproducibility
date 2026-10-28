# Maintainer: zhaowenbo
"""Wolf-style Lyapunov exponent estimation under the unified map framework."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from metrics.common import (
    ChaosMap4DProtocol,
    ChaosMapNDProtocol,
    build_directional_u16_perturbation,
    clone_map_instance,
    ensure_trajectory_2d,
    euclidean_distance_4d,
    euclidean_distance_nd,
    validate_4d_map_interface,
    validate_nd_map_interface,
)
from utils.fixed16 import U16_MOD, u16_to_float


class WolfLyapunovResult(TypedDict):
    """Result container for the full-state Wolf maximum LE estimator."""

    lyap_max: float
    lyap_mean: float
    lyap_std: float
    lyap_components: np.ndarray
    renormalizations: int
    final_distance: float
    is_chaotic: bool
    growth_rates: list[float]
    distances: list[float]


def estimate_lyapunov_smoke(traj: np.ndarray) -> np.ndarray:
    """Return a simple per-dimension log-difference proxy for smoke testing."""

    arr = ensure_trajectory_2d(traj)
    diffs = np.diff(arr, axis=0)
    return np.mean(np.log1p(np.abs(diffs)), axis=0) if len(arr) > 1 else np.zeros(4)


def _perturb_map_u16_state(map_instance, perturb_dim: int, perturb_lsb: int) -> None:
    """Apply a u16-domain LSB perturbation to one state dimension."""

    state_u16 = map_instance.get_state_u16()
    state_u16[perturb_dim] = (int(state_u16[perturb_dim]) + int(perturb_lsb)) % U16_MOD
    map_instance.state_u16 = state_u16.astype(np.int64)


def _renormalize_perturbed_clone(
    sys1: ChaosMapNDProtocol,
    sys2: ChaosMapNDProtocol,
    perturb_lsb: int,
    eps: float,
) -> tuple[ChaosMapNDProtocol, float, np.ndarray]:
    """Rebuild the perturbed system around the current main trajectory state.

    The perturbed system is re-cloned from the main system so that hidden
    internal states such as delay chains, explicit time counters, or enhancer
    counters remain synchronized. Only the public physical state is then displaced
    by a small 16-bit-compatible perturbation aligned with the current
    separation direction.
    """

    state1 = sys1.get_state_float()
    state2 = sys2.get_state_float()
    diff = np.asarray(state2 - state1, dtype=np.float64)
    norm = float(np.linalg.norm(diff))

    if norm <= 1e-15:
        direction = np.zeros(len(diff), dtype=np.float64)
        direction[0] = 1.0
    else:
        direction = diff / norm

    sys2_new = clone_map_instance(sys1)
    perturb_u16 = build_directional_u16_perturbation(direction=direction, perturb_lsb=perturb_lsb)
    sys2_new.state_u16 = (sys1.get_state_u16() + perturb_u16) % U16_MOD
    actual_diff = np.asarray(sys2_new.get_state_float() - sys1.get_state_float(), dtype=np.float64)
    actual_distance = euclidean_distance_nd(sys1.get_state_float(), sys2_new.get_state_float())
    return sys2_new, max(actual_distance, eps), actual_diff


def wolf_lyapunov_full_state_nd(
    map_instance: ChaosMapNDProtocol,
    max_steps: int = 5000,
    transient: int = 1000,
    renorm_threshold: float = 5e-2,
    eps: float | None = None,
    perturb_dim: int = 0,
    perturb_lsb: int = 1,
) -> WolfLyapunovResult:
    """Generalized Wolf-style maximum LE estimator for n-dimensional maps."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if transient < 0:
        raise ValueError("transient must be non-negative.")
    if perturb_lsb <= 0:
        raise ValueError("perturb_lsb must be positive.")
    if renorm_threshold <= 0.0:
        raise ValueError("renorm_threshold must be positive.")

    validate_nd_map_interface(map_instance)
    dimension = len(np.asarray(map_instance.get_state_u16(), dtype=np.int64))
    if perturb_dim < 0 or perturb_dim >= dimension:
        raise ValueError(f"perturb_dim must be in [0, {dimension - 1}].")

    lsb_float = u16_to_float(perturb_lsb)
    eps_value = float(lsb_float if eps is None else eps)
    eps_value = max(eps_value, lsb_float)

    sys1 = clone_map_instance(map_instance)
    sys2 = clone_map_instance(map_instance)
    _perturb_map_u16_state(sys2, perturb_dim=perturb_dim, perturb_lsb=perturb_lsb)

    for _ in range(transient):
        sys1.step()
        sys2.step()

    initial_distance = euclidean_distance_nd(sys1.get_state_float(), sys2.get_state_float())
    if initial_distance <= 1e-15 or initial_distance > renorm_threshold:
        sys2, segment_start_distance, _ = _renormalize_perturbed_clone(
            sys1,
            sys2,
            perturb_lsb=perturb_lsb,
            eps=eps_value,
        )
    else:
        segment_start_distance = initial_distance

    distances: list[float] = []
    growth_rates: list[float] = []
    component_growth_rates: list[np.ndarray] = []
    renormalizations = 0
    total_log_growth = 0.0
    total_growth_steps = 0
    segment_steps = 0
    final_distance = segment_start_distance

    for _ in range(max_steps):
        sys1.step()
        sys2.step()
        segment_steps += 1
        current_diff = np.asarray(sys2.get_state_float() - sys1.get_state_float(), dtype=np.float64)
        distance = euclidean_distance_nd(sys1.get_state_float(), sys2.get_state_float())
        final_distance = distance
        distances.append(distance)

        if distance <= 1e-15:
            sys2, segment_start_distance, _ = _renormalize_perturbed_clone(
                sys1,
                sys2,
                perturb_lsb=perturb_lsb,
                eps=eps_value,
            )
            segment_steps = 0
            final_distance = segment_start_distance
            continue

        if distance >= renorm_threshold:
            log_growth = float(np.log(max(distance, eps_value) / max(segment_start_distance, eps_value)))
            rate = log_growth / max(segment_steps, 1)
            direction_weights = np.abs(current_diff) / max(np.sum(np.abs(current_diff)), eps_value)
            component_rate = rate * direction_weights * float(dimension)
            growth_rates.append(rate)
            component_growth_rates.append(np.asarray(component_rate, dtype=np.float64))
            total_log_growth += log_growth
            total_growth_steps += segment_steps
            renormalizations += 1
            sys2, segment_start_distance, _ = _renormalize_perturbed_clone(
                sys1,
                sys2,
                perturb_lsb=perturb_lsb,
                eps=eps_value,
            )
            segment_steps = 0
            final_distance = segment_start_distance

    if segment_steps > 0:
        log_growth = float(np.log(max(final_distance, eps_value) / max(segment_start_distance, eps_value)))
        rate = log_growth / max(segment_steps, 1)
        final_diff = np.asarray(sys2.get_state_float() - sys1.get_state_float(), dtype=np.float64)
        direction_weights = np.abs(final_diff) / max(np.sum(np.abs(final_diff)), eps_value)
        component_rate = rate * direction_weights * float(dimension)
        growth_rates.append(rate)
        component_growth_rates.append(np.asarray(component_rate, dtype=np.float64))
        total_log_growth += log_growth
        total_growth_steps += segment_steps

    lyap_max = total_log_growth / max(total_growth_steps, 1)
    lyap_mean = float(np.mean(growth_rates)) if growth_rates else 0.0
    lyap_std = float(np.std(growth_rates)) if growth_rates else 0.0
    lyap_components = (
        np.mean(np.asarray(component_growth_rates, dtype=np.float64), axis=0)
        if component_growth_rates
        else np.zeros(dimension, dtype=np.float64)
    )

    return {
        "lyap_max": float(lyap_max),
        "lyap_mean": float(lyap_mean),
        "lyap_std": float(lyap_std),
        "lyap_components": lyap_components.astype(np.float64),
        "renormalizations": int(renormalizations),
        "final_distance": float(final_distance),
        "is_chaotic": bool(lyap_max > 1e-3),
        "growth_rates": growth_rates,
        "distances": distances,
    }


def wolf_lyapunov_full_state(
    map_instance: ChaosMap4DProtocol,
    max_steps: int = 5000,
    transient: int = 1000,
    renorm_threshold: float = 5e-2,
    eps: float | None = None,
    perturb_dim: int = 0,
    perturb_lsb: int = 1,
) -> WolfLyapunovResult:
    """Estimate the maximum Lyapunov exponent using a Wolf-style full-state method.

    The estimator works on two synchronized map instances:

    - a main trajectory
    - a perturbed trajectory

    The perturbation is applied in the u16 state domain by adding one or more
    LSBs to one selected component. Distances are always measured in the 4D
    float state space defined by the existing map interface. When the trajectory
    separation exceeds the threshold, a Wolf-style logarithmic growth is
    accumulated and the perturbed trajectory is renormalized along the current
    separation direction. This function is intentionally limited to the
    full-state Wolf workflow for the current round. It does not implement
    delay-embedding Wolf estimation or Jacobian-QR methods.

    The default length ``transient = 1000`` and ``max_steps = 5000`` follows the
    current paper-facing experiment setting for unified 16-bit comparison across
    different 4D systems. The same interface is intended to remain usable for
    later proposed-family scans over the golden interval parameter family,
    especially for ``Proposed4DBaseMap`` and ``Proposed4DEnhancedMap`` under
    fixed coupling gains ``[50, 48, 46]`` and ``271 <= a11 <= 507``.

    ``lyap_components`` is not a strict orthogonal Lyapunov spectrum. It is a
    direction-aware finite-time component estimate derived within the full-state
    Wolf framework, intended for comparing per-state sensitivity tendencies.
    """

    validate_4d_map_interface(map_instance)
    return wolf_lyapunov_full_state_nd(
        map_instance=map_instance,
        max_steps=max_steps,
        transient=transient,
        renorm_threshold=renorm_threshold,
        eps=eps,
        perturb_dim=perturb_dim,
        perturb_lsb=perturb_lsb,
    )


def estimate_max_lyapunov_wolf_full_state(
    map_instance: ChaosMap4DProtocol,
    max_steps: int = 5000,
    transient: int = 1000,
    renorm_threshold: float = 5e-2,
    eps: float | None = None,
    perturb_dim: int = 0,
    perturb_lsb: int = 1,
) -> WolfLyapunovResult:
    """Backward-compatible wrapper for the full-state Wolf LE estimator."""

    return wolf_lyapunov_full_state(
        map_instance=map_instance,
        max_steps=max_steps,
        transient=transient,
        renorm_threshold=renorm_threshold,
        eps=eps,
        perturb_dim=perturb_dim,
        perturb_lsb=perturb_lsb,
    )
