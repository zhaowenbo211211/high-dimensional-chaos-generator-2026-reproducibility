#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
R1-LY-Complementary parameter synthesis (Algorithm R1-LY).

Given dimension n, odd base gain A_odd, even base gain B_even,
generates complementary odd-gain/even-coupling integer parameter sets.
Returned candidates satisfy the strict algebraic cone-field condition (S);
measure hypotheses and two-band covering require separate verification.
"""

from typing import Dict
import numpy as np


class ComplementaryChaosParams:
    """Adaptive integer synthesis with the strict algebraic Condition (S)."""

    def __init__(self, n: int, A_odd: int, B_even: int,
                 eta: float = 0.3, kappa: float = 0.8, tau: float = 0.5):
        assert n >= 3, "dimension n must be >= 3"
        assert A_odd % 2 == 1 and A_odd >= 5, "A_odd must be odd and >= 5"
        assert B_even % 2 == 0 and B_even > 2 * (n - 1), \
            "B_even must be even and > 2(n-1)"

        self.n = n
        self.A_odd = A_odd
        self.B_even = B_even
        self.eta = eta
        self.kappa = kappa
        self.tau = tau
        self.mod = 1 << 16

    def synthesize(self, max_iter: int = 100) -> Dict:
        """Return parameters and strict-condition diagnostics, or raise on failure.

        The returned c and discriminant use the theoretical Jacobian bounds.
        Candidate widths retain their separate heuristic constant.
        """
        rho = np.sqrt(145.0)
        for _iteration in range(max_iter):
            try:
                a_j1 = np.array([self.A_odd + 2 * j for j in range(self.n - 1)],
                                dtype=np.int64)
                a_jn = np.array([self.B_even - 2 * j for j in range(self.n - 1)],
                                dtype=np.int64)

                if a_jn[-1] <= 0:
                    self.B_even += 2 * (1 - a_jn[-1])
                    a_jn = np.array([self.B_even - 2 * j for j in range(self.n - 1)],
                                    dtype=np.int64)

                A_min = a_j1.min()
                beta = np.sqrt(np.sum(a_jn.astype(np.float64) ** 2))

                b = max(2, int(self.eta * A_min))
                c = np.sqrt(b ** 2 + 144)
                c_width = np.sqrt(b ** 2 + 32)

                # Both tests are needed: a positive discriminant alone can
                # correspond to the wrong branch of the cone inequality.
                discriminant = (A_min - rho) ** 2 - 4 * beta * c
                if A_min <= rho or discriminant <= 0:
                    if self.A_odd < 100:
                        self.A_odd += 2
                    elif self.eta > 0.1:
                        self.eta *= 0.9
                    elif self.B_even > 2 * (self.n - 1) + 2:
                        self.B_even -= 2
                    continue

                Delta_n2 = self.tau * c_width / (2 * b * (A_min - 1))
                Delta_n1 = self.tau * c_width / (2 * a_j1[-1] * (A_min - 1))
                Delta_s0 = self.tau * c_width / (2 * a_jn[-1] * (A_min - 1))

                delta = min(max(np.sqrt(3 * self.tau * c_width / (4 * (A_min - 1))),
                                3 * self.tau * c_width / (2 * (A_min - 1))), 0.25)

                wrap_safe = True
                for j in range(self.n - 1):
                    wrap_val = a_j1[j] * Delta_n1 + a_jn[j] * Delta_s0
                    if wrap_val >= 1 - delta:
                        wrap_safe = False
                        break

                if not wrap_safe:
                    self.tau *= 0.8
                    if self.tau < 0.1:
                        raise ValueError("Cannot satisfy no-wrap condition")
                    continue

                return {
                    'a_j1': a_j1,
                    'a_jn': a_jn,
                    'b': b,
                    'c': c,
                    'A_min': A_min,
                    'beta': beta,
                    'discriminant': discriminant,
                }

            except Exception as e:
                if _iteration == max_iter - 1:
                    raise RuntimeError(f"Synthesis failed: {e}")
                continue

        raise RuntimeError(f"Cannot synthesize parameters within {max_iter} iterations")
