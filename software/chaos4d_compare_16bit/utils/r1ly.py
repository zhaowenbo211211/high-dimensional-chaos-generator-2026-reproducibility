# Maintainer: zhaowenbo
"""R1--LY parameter synthesis for the paper's proposed n-dimensional family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


DEFAULT_R1LY_ETA = 61
DEFAULT_R1LY_KAPPA = 4
DEFAULT_R1LY_TAU = 1
DEFAULT_R1LY_MAX_ITERATIONS = 256


@dataclass(frozen=True)
class R1LYSynthesisResult:
    """Structured validated result for one n-dimensional R1--LY synthesis."""

    n: int
    A_odd_init: int
    B_even_init: int
    A_odd_final: int
    B_even_final: int
    Delta: int
    eta_init: int
    eta_final: int
    kappa: int
    tau_init: int
    tau_final: int
    a_linear: list[int]
    a_coupling: list[int]
    b: int
    A_min: int
    beta: float
    smallest_coupling: int
    band_widths: list[float]
    slab_thickness: float
    no_wrap_budget: float
    condition_s_passed: bool
    no_wrap_passed: bool
    auto_adjustments: list[str]
    iterations: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_inputs(n: int, A_odd: int, B_even: int, Delta: int, eta: int, kappa: int, tau: int) -> None:
    if n < 4:
        raise ValueError("n must be at least 4.")
    if A_odd <= 0 or A_odd % 2 == 0:
        raise ValueError("A_odd must be a positive odd integer.")
    if B_even <= 0 or B_even % 2 != 0:
        raise ValueError("B_even must be a positive even integer.")
    if Delta <= 0 or Delta % 2 != 0:
        raise ValueError("Delta must be a positive even integer.")
    if eta < 0 or kappa < 0 or tau <= 0:
        raise ValueError("eta and kappa must be non-negative, and tau must be positive.")


def _build_sequences(n: int, A_odd: int, B_even: int, Delta: int) -> tuple[list[int], list[int]]:
    state_count = n - 1
    a_linear = [A_odd + j * Delta for j in range(state_count)]
    a_coupling = [B_even - j * Delta for j in range(state_count)]
    return a_linear, a_coupling


def _compute_beta(a_linear: list[int], a_coupling: list[int], tau: int) -> tuple[int, float]:
    A_min = min(a_linear)
    coupling_load = sum(a_coupling) + tau * max(a_coupling)
    beta = float(coupling_load) / float(A_min)
    return int(A_min), float(beta)


def _compute_b(B_even: int, eta: int, kappa: int, tau: int, n: int) -> int:
    return int(B_even + eta + kappa * max(n - 4, 0) * tau)


def _compute_band_widths(a_linear: list[int], a_coupling: list[int], A_min: int, beta: float) -> list[float]:
    denominator = max(float(A_min), 1.0)
    stabilizer = max(1.0 - beta / 2.0, 1e-6)
    return [float((linear - coupling) / denominator) * stabilizer for linear, coupling in zip(a_linear, a_coupling)]


def _compute_no_wrap_budget(a_linear: list[int], a_coupling: list[int], b: int, eta: int, tau: int) -> tuple[float, float]:
    slab_thickness = float(min(linear - coupling for linear, coupling in zip(a_linear, a_coupling)) + eta)
    drive_scale = float(max(a_coupling) + tau * max(1, len(a_coupling)) + ceil(b / 8.0))
    no_wrap_budget = slab_thickness - drive_scale
    return slab_thickness, no_wrap_budget


def _record_adjustment(adjustments: list[str], message: str) -> None:
    """Append one adjustment message, compacting repeated consecutive entries."""

    if adjustments and adjustments[-1].startswith(message):
        previous = adjustments[-1]
        if " x" in previous:
            base, count_str = previous.rsplit(" x", maxsplit=1)
            adjustments[-1] = f"{base} x{int(count_str) + 1}"
        else:
            adjustments[-1] = f"{previous} x2"
        return
    adjustments.append(message)


def synthesize_r1_ly_params(
    n: int,
    A_odd: int = 371,
    B_even: int = 50,
    Delta: int = 2,
    eta: int = DEFAULT_R1LY_ETA,
    kappa: int = DEFAULT_R1LY_KAPPA,
    tau: int = DEFAULT_R1LY_TAU,
    max_iterations: int = DEFAULT_R1LY_MAX_ITERATIONS,
) -> dict[str, object]:
    """Synthesize validated R1--LY parameters for dimensions `4 <= n <= 12`.

    The synthesis follows a deterministic validation loop aligned with the
    requested paper-facing steps:

    1. build complementary parity-alternating sequences;
    1a. enforce positivity of the smallest coupling coefficient;
    2. compute `A_min` and `beta`;
    3. compute the feedback parameter `b`;
    4. verify cone-field Condition (S);
    5. compute band widths;
    6. compute the default slab thickness / no-wrap budget;
    7. verify the no-wrap budget;
    8. return the validated parameter set.

    When validation fails, the loop automatically applies:
    - Option A: increase `A_odd` by 2;
    - Option B: reduce `eta`;
    - Option C: reduce `B_even` by 2.
    """

    _validate_inputs(n=n, A_odd=A_odd, B_even=B_even, Delta=Delta, eta=eta, kappa=kappa, tau=tau)

    A_odd_init = int(A_odd)
    B_even_init = int(B_even)
    eta_init = int(eta)
    tau_init = int(tau)
    adjustments: list[str] = []

    for iteration in range(1, max_iterations + 1):
        a_linear, a_coupling = _build_sequences(n=n, A_odd=A_odd, B_even=B_even, Delta=Delta)
        smallest_coupling = int(a_coupling[-1])

        if smallest_coupling <= 0:
            B_even += 2 * (1 - smallest_coupling)
            if B_even % 2 != 0:
                B_even += 1
            _record_adjustment(adjustments, "Step1a: increased B_even to restore positive smallest coupling")
            continue

        A_min, beta = _compute_beta(a_linear=a_linear, a_coupling=a_coupling, tau=tau)
        b = _compute_b(B_even=B_even, eta=eta, kappa=kappa, tau=tau, n=n)

        condition_s_limit = 1.0 - 1.0 / float(n + tau + 1)
        condition_s_passed = bool(beta < condition_s_limit and A_min > max(a_coupling))
        if not condition_s_passed:
            A_odd += 2
            _record_adjustment(adjustments, "Step4: increased A_odd by 2 to satisfy Condition (S)")
            continue

        band_widths = _compute_band_widths(a_linear=a_linear, a_coupling=a_coupling, A_min=A_min, beta=beta)
        slab_thickness, no_wrap_budget = _compute_no_wrap_budget(
            a_linear=a_linear,
            a_coupling=a_coupling,
            b=b,
            eta=eta,
            tau=tau,
        )
        no_wrap_passed = bool(no_wrap_budget > 0.0 and min(band_widths) > 0.0)
        if not no_wrap_passed:
            if eta > 8:
                eta = max(8, eta - 2)
                _record_adjustment(adjustments, "Step7: reduced eta by 2 to recover no-wrap budget")
                continue
            if B_even > 2 * (n - 1):
                B_even -= 2
                _record_adjustment(adjustments, "Step7: reduced B_even by 2 to recover no-wrap budget")
                continue
            A_odd += 2
            _record_adjustment(adjustments, "Step7: increased A_odd by 2 after eta/B_even exhaustion")
            continue

        result = R1LYSynthesisResult(
            n=int(n),
            A_odd_init=A_odd_init,
            B_even_init=B_even_init,
            A_odd_final=int(A_odd),
            B_even_final=int(B_even),
            Delta=int(Delta),
            eta_init=eta_init,
            eta_final=int(eta),
            kappa=int(kappa),
            tau_init=tau_init,
            tau_final=int(tau),
            a_linear=[int(value) for value in a_linear],
            a_coupling=[int(value) for value in a_coupling],
            b=int(b),
            A_min=int(A_min),
            beta=float(beta),
            smallest_coupling=int(smallest_coupling),
            band_widths=[float(value) for value in band_widths],
            slab_thickness=float(slab_thickness),
            no_wrap_budget=float(no_wrap_budget),
            condition_s_passed=True,
            no_wrap_passed=True,
            auto_adjustments=list(adjustments),
            iterations=int(iteration),
        )
        return result.as_dict()

    raise RuntimeError("R1--LY synthesis did not converge within max_iterations.")


def generate_r1ly_params_4d(
    A_odd: int = 371,
    B_even: int = 50,
    Delta: int = 2,
    eta: int = DEFAULT_R1LY_ETA,
    n: int = 4,
) -> dict[str, object]:
    """Backward-compatible 4D wrapper around the generalized R1--LY synthesis."""

    if n != 4:
        raise ValueError("generate_r1ly_params_4d only supports n=4.")
    params = synthesize_r1_ly_params(n=n, A_odd=A_odd, B_even=B_even, Delta=Delta, eta=eta)
    return {
        "n": int(params["n"]),
        "A_odd": int(params["A_odd_final"]),
        "B_even": int(params["B_even_final"]),
        "Delta": int(params["Delta"]),
        "eta": int(params["eta_final"]),
        "a_linear": [int(value) for value in params["a_linear"]],
        "a_coupling": [int(value) for value in params["a_coupling"]],
        "b": int(params["b"]),
    }
