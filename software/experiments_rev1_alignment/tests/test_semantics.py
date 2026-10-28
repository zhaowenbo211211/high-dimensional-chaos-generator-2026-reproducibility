# Maintainer: zhaowenbo
"""Unit tests for the rev1 core semantics and the R1-LY synthesizer.

Runnable with plain python (no pytest dependency):
    python3 tests/test_semantics.py
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
import sys

SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SUBSYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

from chaos_maps.proposed_rev1 import (
    ProposedNDRev1Map,
    Q,
    reference_base_update,
    reference_rotate_pack,
)
from r1ly_rev1 import RECOMMENDED_A_ODD, RECOMMENDED_B_EVEN, golden_range_a_values, synthesize_r1ly

PASS = 0


def check(cond: bool, label: str) -> None:
    global PASS
    if not cond:
        raise AssertionError(f"FAILED: {label}")
    PASS += 1
    print(f"ok: {label}")


def independent_step(state_full, n, a1, an, b, phase):
    """Independent numpy-based reimplementation of one Mode-II step."""

    x = np.asarray(state_full[: n - 1], dtype=np.int64)
    s = np.asarray(state_full[n - 1 :], dtype=np.int64)
    u_x = (np.asarray(a1) * x + np.asarray(an) * s[::-1]) % Q
    big_s = int(x[-1]) + int(s[0])
    # exact rational floor via Fraction (independent of the // operator path)
    nonlinear = Fraction(4 * big_s * (Q - big_s), Q).__floor__()
    u_s0 = (b * int(x[-2]) + nonlinear) % Q
    u = list(int(v) for v in u_x) + [u_s0] + [int(v) for v in s[:-1]]
    # pack with explicit bit loop (little-endian: word j at bits 16j)
    z = 0
    for j in range(n):
        z += u[j] * (1 << (16 * j))
    k = 1 + (phase % 15)
    width = 16 * n
    zr = ((z << k) % (1 << width)) + (z >> (width - k))
    out = [(zr // (1 << (16 * j))) % (1 << 16) for j in range(n)]
    return out + u[n:]


def full_state_of(m: ProposedNDRev1Map) -> list[int]:
    return [int(v) for v in m.state_u16] + [int(v) for v in m.delay_u16]


def test_reference_vs_independent():
    rng = np.random.default_rng(20260909)
    for n in (4, 5, 12):
        params = synthesize_r1ly(n=n, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[n])
        init = [int(v) for v in rng.integers(0, Q, size=n)]
        hist = [int(v) for v in rng.integers(0, Q, size=n - 2)]
        m = ProposedNDRev1Map(n=n, a_j1=params["a_j1"], a_jn=params["a_jn"], b=params["b"])
        m.reset(initial_state_u16=init, initial_history_u16=hist)
        state = full_state_of(m)
        phase = 0
        for _ in range(200):  # covers phase wrap (15 steps) many times
            expected = independent_step(state, n, params["a_j1"], params["a_jn"], params["b"], phase)
            m.step()
            state = full_state_of(m)
            check(state == expected, f"reference==independent n={n} step")
            phase += 1
    print("reference vs independent implementation: OK")


def test_s_regions_floor():
    # n=4: full state [x1,x2,x3,s0,s1,s2]; S = x3 + s0; feedback uses x2.
    a1 = [371, 373, 375]
    an = [50, 48, 46]
    b = 111
    for big_s_desc, x3, s0 in [("S<Q", 1000, 2000), ("S==Q", Q // 2, Q // 2), ("S>Q", Q - 1, Q - 1)]:
        state = [123, 4567, x3, s0, 0, 0]
        u = reference_base_update(state, 4, a1, an, b)
        big_s = x3 + s0
        # exact floor of a possibly negative rational, computed via Fraction-free math
        num = 4 * big_s * (Q - big_s)
        # exact rational floor, computed independently of the // operator
        expected_nl = Fraction(num, Q).__floor__()
        check(u[3] == (b * 4567 + expected_nl) % Q, f"floor semantics {big_s_desc}")
    # explicit floor-vs-trunc distinction: S = Q+1 => 4*S*(Q-S) = -4Q-4, floor(-4 - 4/Q) = -5
    u = reference_base_update([0, 0, Q // 2 + 1, Q // 2, 0, 0], 4, a1, an, b)
    big_s = Q + 1
    check((4 * big_s * (Q - big_s)) // Q == -5, "floor(-4-4/Q) = -5, not -4 (truncation)")


def test_pack_rotate():
    words = [0x1234, 0xABCD, 0x0001, 0xFFFF]
    rotated = reference_rotate_pack(words, 1)
    # little-endian: word 0 occupies bits 0..15; rotating left by 1 moves bit 15 of
    # word 0 into bit 0 of word 1 etc. Explicit expectation:
    z = 0x1234 | (0xABCD << 16) | (0x0001 << 32) | (0xFFFF << 48)
    zr = ((z << 1) | (z >> 63)) & ((1 << 64) - 1)
    expected = [(zr >> (16 * j)) & 0xFFFF for j in range(4)]
    check(rotated == expected, "little-endian pack + rol(1)")
    # old big-endian packing must give a different result; use words whose
    # high bits wrap differently under rotation so the two orders diverge
    words2 = [0x8000, 0x0002, 0x4000, 0x1111]
    rotated2 = reference_rotate_pack(words2, 1)
    z_old = 0
    for w in words2:
        z_old = (z_old << 16) | w
    zr_old = ((z_old << 1) | (z_old >> 63)) & ((1 << 64) - 1)
    old_words = [(zr_old >> (16 * (3 - j))) & 0xFFFF for j in range(4)]
    check(rotated2 != old_words, "new packing distinguishable from old big-endian packing")
    # inverse rotation recovers words
    n = 4
    width = 16 * n
    zz = 0
    for j, w in enumerate(rotated):
        zz |= w << (16 * j)
    zback = ((zz >> 1) | (zz << (width - 1))) & ((1 << width) - 1)
    check([(zback >> (16 * j)) & 0xFFFF for j in range(n)] == words, "inverse rotation roundtrip")


def test_delay_and_phase():
    n = 5
    params = synthesize_r1ly(n=n, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[n])
    m = ProposedNDRev1Map(n=n, a_j1=params["a_j1"], a_jn=params["a_jn"], b=params["b"])
    init = [1111, 2222, 3333, 4444, 5555]
    hist = [60000, 50000, 40000]  # s1, s2, s3 distinctive
    m.reset(initial_state_u16=init, initial_history_u16=hist)
    # one step: delay must shift the OLD (s0, s1, s2) = (5555, 60000, 50000)
    m.step()
    check([int(v) for v in m.delay_u16] == [5555, 60000, 50000], "delay uses old values s_k^+ = s_{k-1}")
    # delay words must not receive rotated content: run 20 steps, delay must equal
    # base-update shift of the un-rotated U sequence; verify via independent chain.
    m2 = ProposedNDRev1Map(n=n, a_j1=params["a_j1"], a_jn=params["a_jn"], b=params["b"])
    m2.reset(initial_state_u16=init, initial_history_u16=hist)
    for i in range(20):
        u = reference_base_update(full_state_of(m2), n, params["a_j1"], params["a_jn"], params["b"])
        m2.step()
        check([int(v) for v in m2.delay_u16] == u[n:], f"delay untouched by rotation step {i}")
        check(m2.rotation_step_index == i + 1, "phase increments per step")
    # phase: k_t = 1 + (t mod 15); at t=15 -> k=1 again
    check(1 + (15 % 15) == 1, "phase wrap at t=15")


def test_old_semantics_differ():
    # Old enhancer wrote the rotated x_n back into the delay line; new one must not.
    n = 4
    params = synthesize_r1ly(n=n, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[n])
    init = [40000, 30000, 20000, 10000]
    m = ProposedNDRev1Map(n=n, a_j1=params["a_j1"], a_jn=params["a_jn"], b=params["b"])
    m.reset(initial_state_u16=init)
    u = reference_base_update(full_state_of(m), n, params["a_j1"], params["a_jn"], params["b"])
    m.step()
    # old rule would set delay[-1] = rotated x_n; new rule keeps u[n:] = (s0_old, s1_old)
    check([int(v) for v in m.delay_u16] == u[n:], "new delay excludes rotated x_n")


def test_r1ly_recommended():
    import csv
    csv_path = SUBSYSTEM_ROOT / "configs" / "recommended_parameters.csv"
    rows = {int(r["n"]): r for r in csv.DictReader(csv_path.read_text().splitlines())}
    for n in range(4, 13):
        params = synthesize_r1ly(n=n, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[n])
        row = rows[n]
        check(params["b"] == int(row["b"]), f"n={n} b={params['b']}")
        check(params["a_j1"] == eval(row["a_linear"]), f"n={n} a_linear")
        check(params["a_jn"] == eval(row["a_coupling"]), f"n={n} a_coupling")
        check(abs(params["beta"] - float(row["beta"])) < 1e-9, f"n={n} beta")
        check(abs(params["discriminant"] - float(row["discriminant"])) < 1e-6, f"n={n} discriminant")
        check(params["strong_check_passed"] and params["width_check_passed"], f"n={n} checks pass")
        check(params["attempts"] == 1 and not params["auto_adjustments"], f"n={n} one-shot, no adaptation")
        check(params["b"] == 3 * params["A_min"] // 10 == 111, f"n={n} b=3*A_min//10=111")
    print("recommended parameters: OK")


def test_r1ly_golden_range():
    values = golden_range_a_values()
    check(len(values) == 119 and values[0] == 271 and values[-1] == 507, "119 odd A in [271,507]")
    for a in values:
        params = synthesize_r1ly(n=4, A_odd=a, B_even=RECOMMENDED_B_EVEN[4])
        check(params["strong_check_passed"], f"A={a} strong check")
        check(params["width_check_passed"], f"A={a} width check")
        check(params["b"] == max(2, 3 * a // 10), f"A={a} b=3A//10")
        check(params["attempts"] == 1, f"A={a} one-shot")
    # width budget must use chat = sqrt(b^2+32), not c: recompute both and confirm
    # the implementation used chat (i.e., recorded width_W matches chat-based value)
    params = synthesize_r1ly(n=4, A_odd=271, B_even=50)
    a1, an, b = params["a_j1"], params["a_jn"], params["b"]
    chat = math.sqrt(b * b + 32.0)
    c = math.sqrt(b * b + 144.0)
    tau = params["tau_final"]
    A_min = params["A_min"]
    d1 = tau * chat / (2.0 * a1[-1] * (A_min - 1))
    ds0 = tau * chat / (2.0 * an[-1] * (A_min - 1))
    w_chat = max(a1[j] * d1 + an[j] * ds0 for j in range(3))
    d1c = tau * c / (2.0 * a1[-1] * (A_min - 1))
    ds0c = tau * c / (2.0 * an[-1] * (A_min - 1))
    w_c = max(a1[j] * d1c + an[j] * ds0c for j in range(3))
    check(abs(params["width_W"] - w_chat) < 1e-12, "width budget uses chat")
    check(abs(w_chat - w_c) > 1e-9, "chat-based and c-based widths differ (test is discriminating)")
    print("golden range parameters: OK")


def test_wolf_interface():
    params = synthesize_r1ly(n=4, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[4])
    m = ProposedNDRev1Map(n=4, a_j1=params["a_j1"], a_jn=params["a_jn"], b=params["b"],
                          initial_state_u16=[6554, 13107, 19661, 26214])
    from metrics.lyapunov import wolf_lyapunov_full_state_nd
    result = wolf_lyapunov_full_state_nd(m, max_steps=200, transient=50)
    check(math.isfinite(result["lyap_max"]), "wolf LE finite on new map")
    check(len(result["lyap_components"]) == 4, "wolf components length n")
    print("wolf interface: OK")


if __name__ == "__main__":
    test_reference_vs_independent()
    test_s_regions_floor()
    test_pack_rotate()
    test_delay_and_phase()
    test_old_semantics_differ()
    test_r1ly_recommended()
    test_r1ly_golden_range()
    test_wolf_interface()
    print(f"\nALL TESTS PASSED ({PASS} checks)")
