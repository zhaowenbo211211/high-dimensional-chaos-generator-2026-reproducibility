#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Lyapunov exponent computation via Jacobian-QR method and chaos verification.
"""

from typing import Dict
import warnings
import numpy as np

from chaos.core.base_map import HighDimChaosMap


def _jacobian_normalized(x: np.ndarray, params: Dict, n: int) -> np.ndarray:
    """Compute Jacobian in normalized space [0,1)^n."""
    J = np.zeros((n, n), dtype=np.float64)

    for j in range(n - 1):
        J[j, j] = params['a_j1'][j]
        J[j, -1] = params['a_jn'][j]

    z = x[-2] + x[-1]
    if 0 < z < 1:
        d_logistic = 4.0 * (1.0 - 2.0 * z)
    else:
        d_logistic = 0.0

    J[-1, -2] = params['b']
    J[-1, -1] = d_logistic

    return J


def compute_lyapunov_exponents(map_instance: HighDimChaosMap,
                               steps: int = 2000,
                               transient: int = 500,
                               ortho_interval: int = 5) -> np.ndarray:
    """Compute Lyapunov spectrum via QR decomposition."""
    n = map_instance.n
    mod = map_instance.mod

    map_instance.iterate(transient)

    Q = np.eye(n, dtype=np.float64)
    lyap_sum = np.zeros(n, dtype=np.float64)
    count = 0

    for t in range(steps):
        x_int = map_instance.get_physical_state()
        x_norm = x_int.astype(np.float64) / mod

        J = _jacobian_normalized(x_norm, map_instance.params, n)
        Q = J @ Q

        if t % ortho_interval == 0:
            Q, R = np.linalg.qr(Q)
            diag_r = np.abs(np.diag(R))
            diag_r = np.maximum(diag_r, 1e-20)
            lyap_sum += np.log(diag_r)
            count += 1

            if np.max(diag_r) > 1e15:
                warnings.warn(f"Step {t}: tangent vectors too large")

    return lyap_sum / (count * ortho_interval)


def verify_chaos(map_instance: HighDimChaosMap, steps: int = 2000) -> Dict:
    """Verify chaotic property: positive Lyapunov exponents + state variance."""
    trajectory = map_instance.iterate(steps)
    traj_norm = trajectory.astype(np.float64) / map_instance.mod

    les = compute_lyapunov_exponents(map_instance, steps=steps, transient=100)
    variance = np.var(traj_norm, axis=0)

    x_n = traj_norm[:, -1]
    autocorr = np.correlate(x_n - x_n.mean(), x_n - x_n.mean(),
                            mode='full')[len(x_n) - 1:]
    autocorr /= autocorr[0]

    is_chaotic = np.all(les > 0.01) and np.mean(variance) > 0.05

    return {
        'params': map_instance.params,
        'lyap_exps': les,
        'is_chaotic': is_chaotic,
        'variance': variance,
        'min_le': les.min(),
        'max_le': les.max()
    }
