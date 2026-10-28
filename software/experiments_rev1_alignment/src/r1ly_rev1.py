# Maintainer: zhaowenbo
"""R1-LY parameter synthesis implementing the current Section 3 algorithm.

This module implements Algorithm R1-LY of ``pre_art_rev1.tex`` exactly as
printed (Steps 1-8, attempt limit M=100, tau retry factor 0.8, tau floor 0.1).
It does NOT reuse the old ``utils/r1ly.py`` semantics (eta=61, different beta,
different condition check), which are incompatible with the current paper.

Strong check (Step 4):  A_min > rho  AND  Dcone = (A_min - rho)^2 - 4*beta*c > 0
Width budget (Steps 5-7): uses chat = sqrt(b^2 + 32) only (never c).
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_ETA = 0.30
DEFAULT_TAU = 0.5
DEFAULT_MAX_ATTEMPTS = 100
RHO = math.sqrt(145.0)


def _b_from_eta(eta: float, a_min: int) -> int:
    """b = max(2, floor(eta * A_min)); integer-exact for the default eta=0.30."""

    if eta == DEFAULT_ETA:
        return max(2, (3 * int(a_min)) // 10)
    return max(2, int(math.floor(eta * int(a_min))))


def synthesize_r1ly(
    n: int,
    A_odd: int,
    B_even: int,
    eta: float = DEFAULT_ETA,
    tau: float = DEFAULT_TAU,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Run Algorithm R1-LY and return the parameter dictionary with diagnostics.

    Returned keys (superset of the paper's Step-8 dictionary):
    ``a_j1``, ``a_jn``, ``b``, ``c``, ``A_min``, ``beta``, ``discriminant``,
    plus diagnostics: ``chat``, ``delta``, ``width_W``, ``strong_check_passed``,
    ``width_check_passed``, ``attempts``, ``eta_final``, ``tau_final``,
    ``A_odd_final``, ``B_even_final``, ``auto_adjustments``.
    Raises ``RuntimeError`` on synthesis failure after ``max_attempts``.
    """

    if int(n) < 3:
        raise ValueError("n must be at least 3.")
    n = int(n)
    A_odd = int(A_odd)
    B_even = int(B_even)
    if A_odd < 5 or A_odd % 2 == 0:
        raise ValueError("A_odd must be odd and >= 5.")
    if B_even <= 2 * (n - 1) or B_even % 2 != 0:
        raise ValueError("B_even must be even and > 2(n-1).")

    auto_adjustments: list[str] = []
    for attempt in range(1, int(max_attempts) + 1):
        try:
            # Step 1: coefficient construction (+ positivity repair of a_{n-1,n})
            a_j1 = [A_odd + 2 * (j - 1) for j in range(1, n)]
            a_jn = [B_even - 2 * (j - 1) for j in range(1, n)]
            if a_jn[-1] <= 0:
                B_even_new = B_even + 2 * (1 - a_jn[-1])
                auto_adjustments.append(
                    f"attempt {attempt}: B_even {B_even} -> {B_even_new} (a_(n-1,n) positivity)"
                )
                B_even = B_even_new
                a_jn = [B_even - 2 * (j - 1) for j in range(1, n)]

            # Step 2
            A_min = min(a_j1)
            beta = math.sqrt(sum(value * value for value in a_jn))

            # Step 3
            b = _b_from_eta(eta, A_min)
            chat = math.sqrt(b * b + 32.0)

            # Step 4 (strong check, equivalent to Condition (S))
            c = math.sqrt(b * b + 144.0)
            dcone = (A_min - RHO) ** 2 - 4.0 * beta * c
            if A_min <= RHO or dcone <= 0.0:
                if A_odd < 100:
                    A_odd += 2
                    auto_adjustments.append(f"attempt {attempt}: A_odd += 2 (strong check)")
                elif eta > 0.1:
                    eta = 0.9 * eta
                    auto_adjustments.append(f"attempt {attempt}: eta *= 0.9 (strong check)")
                elif B_even > 2 * (n - 1) + 2:
                    B_even -= 2
                    auto_adjustments.append(f"attempt {attempt}: B_even -= 2 (strong check)")
                continue

            # Step 5
            delta_n_2 = tau * chat / (2.0 * b * (A_min - 1))
            delta_n_1 = tau * chat / (2.0 * a_j1[-1] * (A_min - 1))
            delta_s0 = tau * chat / (2.0 * a_jn[-1] * (A_min - 1))

            # Step 6
            delta = min(
                0.25,
                max(
                    math.sqrt(3.0 * tau * chat / (4.0 * (A_min - 1))),
                    3.0 * tau * chat / (2.0 * (A_min - 1)),
                ),
            )

            # Step 7
            width_W = max(
                a_j1[j] * delta_n_1 + a_jn[j] * delta_s0 for j in range(n - 1)
            )
            if width_W >= 1.0 - delta:
                tau = 0.8 * tau
                auto_adjustments.append(f"attempt {attempt}: tau *= 0.8 (width budget)")
                if tau < 0.1:
                    raise ArithmeticError("width-budget exception: tau < 0.1")
                continue

            # Step 8
            return {
                "a_j1": [int(v) for v in a_j1],
                "a_jn": [int(v) for v in a_jn],
                "b": int(b),
                "c": float(c),
                "A_min": int(A_min),
                "beta": float(beta),
                "discriminant": float(dcone),
                "chat": float(chat),
                "delta": float(delta),
                "delta_n_2": float(delta_n_2),
                "delta_n_1": float(delta_n_1),
                "delta_s0": float(delta_s0),
                "width_W": float(width_W),
                "strong_check_passed": True,
                "width_check_passed": True,
                "attempts": int(attempt),
                "eta_final": float(eta),
                "tau_final": float(tau),
                "A_odd_final": int(A_odd),
                "B_even_final": int(B_even),
                "auto_adjustments": auto_adjustments,
            }
        except ArithmeticError:
            if attempt < max_attempts:
                continue
            raise RuntimeError("R1-LY synthesis failed: width-budget exception at attempt limit.")
    raise RuntimeError("R1-LY synthesis failed after exhausting the attempt limit.")


# Recommended B_even table of Section 3 (subsec:nd-theory).
RECOMMENDED_B_EVEN = {4: 50, 5: 44, 6: 40, 7: 38, 8: 36, 9: 36, 10: 34, 11: 34, 12: 34}
RECOMMENDED_A_ODD = 371


def golden_range_a_values() -> list[int]:
    """Golden-Range scan set for n=4: odd A in [271, 515-2*4] = [271, 507]."""

    return list(range(271, 515 - 2 * 4 + 1, 2))
