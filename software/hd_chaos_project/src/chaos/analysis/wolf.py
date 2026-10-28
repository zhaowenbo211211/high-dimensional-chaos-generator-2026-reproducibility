#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Wolf-style maximum Lyapunov exponent estimation for integer digital chaotic maps.

Two methods:
  - wolf_lyapunov_full: full-state space method with renormalization
  - wolf_lyapunov_embedding: delay-embedding method (Rosenstein-style)
"""

from copy import deepcopy
import numpy as np


def wolf_lyapunov_full(map_instance, max_steps=50000, transient=5000,
                       renorm_threshold=0.1, min_segment_len=5):
    """Wolf-style max LE estimation in full state space (integer map version).

    Tracks divergence of two nearby trajectories with periodic 1-LSB renormalization.
    Suitable for finite-precision integer state spaces and enhanced digital chaotic maps.
    """
    n = map_instance.n
    mod = map_instance.mod

    initial_state = map_instance.get_physical_state().astype(np.int64)

    sys1 = deepcopy(map_instance)
    sys2 = deepcopy(map_instance)

    sys1.set_state_from_physical(initial_state)

    perturbed_state = initial_state.copy()
    perturbed_state[0] = min(perturbed_state[0] + 1, mod - 1)
    sys2.set_state_from_physical(perturbed_state)

    for _ in range(transient):
        sys1.next()
        sys2.next()

    state1 = sys1.get_physical_state().astype(np.float64)
    state2 = sys2.get_physical_state().astype(np.float64)

    segment_start_distance = np.linalg.norm(state1 - state2) / mod
    if segment_start_distance < 1e-16:
        segment_start_distance = 1.0 / mod

    segment_len = 0
    growth_rates = []
    distances = []
    renormalizations = 0

    for _ in range(max_steps):
        state1 = sys1.next().astype(np.float64)
        state2 = sys2.next().astype(np.float64)

        d = np.linalg.norm(state1 - state2) / mod
        distances.append(d)
        segment_len += 1

        if d >= renorm_threshold and segment_len >= min_segment_len:
            if segment_start_distance > 1e-16 and d > 1e-16:
                growth_rate = np.log(d / segment_start_distance) / segment_len
                growth_rates.append(growth_rate)
                renormalizations += 1

            diff = state2 - state1
            norm_diff = np.linalg.norm(diff)

            if norm_diff < 1e-16:
                new_state2 = state1.astype(np.int64).copy()
                new_state2[0] = min(new_state2[0] + 1, mod - 1)
            else:
                idx = np.argmax(np.abs(diff))
                sign = 1 if diff[idx] >= 0 else -1
                new_state2 = state1.astype(np.int64).copy()
                new_state2[idx] = np.clip(new_state2[idx] + sign, 0, mod - 1)
                if np.array_equal(new_state2, state1.astype(np.int64)):
                    alt_idx = 0 if idx != 0 else min(1, n - 1)
                    new_state2[alt_idx] = min(new_state2[alt_idx] + 1, mod - 1)

            sys2.set_state_from_physical(new_state2)

            state1_now = sys1.get_physical_state().astype(np.float64)
            state2_now = sys2.get_physical_state().astype(np.float64)
            segment_start_distance = np.linalg.norm(state1_now - state2_now) / mod
            if segment_start_distance < 1e-16:
                segment_start_distance = 1.0 / mod
            segment_len = 0

    if len(growth_rates) == 0:
        if len(distances) > 20 and distances[0] > 1e-16:
            log_dist = np.log(np.array(distances) + 1e-16)
            slope = np.polyfit(np.arange(len(log_dist)), log_dist, 1)[0]
            lyap_max = slope
            growth_rates = [slope]
        else:
            lyap_max = 0.0
            growth_rates = [0.0]
    else:
        lyap_max = float(np.mean(growth_rates))

    return {
        "lyap_max": lyap_max,
        "lyap_mean": float(np.mean(growth_rates)),
        "lyap_std": float(np.std(growth_rates)),
        "final_distance": float(distances[-1]) if distances else 0.0,
        "is_chaotic": lyap_max > 0.0,
        "growth_rates": growth_rates,
        "distances": distances,
        "renormalizations": renormalizations,
    }


def wolf_lyapunov_embedding(map_instance, max_steps=20000, transient=5000,
                            embed_dim=None, delay=1, theiler_window=50,
                            fit_horizon=20):
    """Rosenstein-style max LE via delay embedding of the x_n time series.

    Uses Theiler window to exclude temporally correlated neighbours,
    then fits the average log-divergence slope.
    """
    n = map_instance.n
    mod = map_instance.mod

    if embed_dim is None:
        embed_dim = n

    for _ in range(transient):
        map_instance.next()

    series = []
    total_length = max_steps + (embed_dim - 1) * delay + fit_horizon + 100
    for _ in range(total_length):
        state = map_instance.next()
        series.append(float(state[-1]) / mod)
    series = np.array(series, dtype=np.float64)

    embedded = []
    for i in range((embed_dim - 1) * delay, len(series) - fit_horizon):
        vec = [series[i - j * delay] for j in range(embed_dim)]
        embedded.append(vec)
    embedded = np.array(embedded, dtype=np.float64)

    n_embed = len(embedded)
    if n_embed < 100:
        return _empty_wolf_result(embed_dim, delay, embedded.shape)

    log_divergence = []
    valid_pairs = 0
    max_ref = n_embed - fit_horizon - 1

    for i in range(max_ref):
        ref = embedded[i]
        candidate_idx = np.arange(n_embed)
        mask = np.abs(candidate_idx - i) > theiler_window
        mask &= (candidate_idx < max_ref)
        candidates = candidate_idx[mask]

        if len(candidates) == 0:
            continue

        dists = np.linalg.norm(embedded[candidates] - ref, axis=1)
        d0 = np.min(dists)
        nn = candidates[np.argmin(dists)]

        if d0 < 1e-12:
            continue

        local_logs = []
        for k in range(fit_horizon):
            dk = np.linalg.norm(embedded[i + k] - embedded[nn + k])
            local_logs.append(np.log(dk + 1e-16))
        log_divergence.append(local_logs)
        valid_pairs += 1

    if valid_pairs < 10:
        return _empty_wolf_result(embed_dim, delay, embedded.shape)

    log_divergence = np.array(log_divergence)
    mean_log_div = np.mean(log_divergence, axis=0)
    t = np.arange(fit_horizon, dtype=np.float64)
    slope = np.polyfit(t, mean_log_div, 1)[0]

    return {
        "lyap_max": float(slope),
        "lyap_mean": float(slope),
        "lyap_std": float(np.std(log_divergence[:, 1] - log_divergence[:, 0])),
        "final_distance": float(np.exp(mean_log_div[-1])),
        "is_chaotic": slope > 0.0,
        "growth_rates": [float(slope)],
        "distances": list(np.exp(mean_log_div)),
        "renormalizations": valid_pairs,
        "embed_dim": embed_dim,
        "delay": delay,
        "embedded_traj_shape": embedded.shape,
    }


def _empty_wolf_result(embed_dim, delay, shape):
    return {
        "lyap_max": 0.0, "lyap_mean": 0.0, "lyap_std": 0.0,
        "final_distance": 0.0, "is_chaotic": False,
        "growth_rates": [], "distances": [], "renormalizations": 0,
        "embed_dim": embed_dim, "delay": delay,
        "embedded_traj_shape": shape,
    }
