#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
R1-LY generation test: 4D, 16-bit, A_odd=271, B_even=50.

Verifies:
  1. Parameter synthesis produces valid parity-alternating sequences
  2. Cone-field condition is satisfied
  3. Chaotic map iterates correctly
  4. Lyapunov exponents are all positive (chaos confirmed)
  5. LaTeX system equations are generated correctly
"""

import numpy as np

from chaos.core import ComplementaryChaosParams, HighDimChaosMap
from chaos.analysis import verify_chaos
from chaos.utils import make_F, latex_system_like_figure_full


def test_generation():
    """Generate 4D 16-bit chaos with R1-LY and verify all constraints."""
    n = 4
    word_size = 16

    # ---- Step 1: Parameter synthesis ----
    synth = ComplementaryChaosParams(n=n, A_odd=271, B_even=50)
    params = synth.synthesize()

    # Parity constraints
    assert all(params['a_j1'] % 2 == 1), "a_j1 must all be odd (expansion)"
    assert all(params['a_jn'] % 2 == 0), "a_jn must all be even (coupling)"
    assert all(params['a_jn'] > 0),      "a_jn must be strictly positive"

    # Cone-field condition
    assert params['discriminant'] > 0, \
        f"Cone-field condition failed: Delta = {params['discriminant']:.3f} <= 0"

    # b constraint
    assert params['b'] >= 2, f"Feedback b = {params['b']} must be >= 2"

    print("=" * 70)
    print(f"R1-LY Generation: n={n}, {word_size}-bit, A_odd={synth.A_odd}, B_even={synth.B_even}")
    print("=" * 70)
    print(f"  a_j1 (odd expansion):  {params['a_j1']}")
    print(f"  a_jn (even coupling):  {params['a_jn']}")
    print(f"  b (feedback):          {params['b']}")
    print(f"  c (Lipschitz bound):   {params['c']:.3f}")
    print(f"  beta (coupling norm):  {params['beta']:.3f}")
    print(f"  A_min:                 {params['A_min']}")
    print(f"  Delta (cone-field):    {params['discriminant']:.3f}  (> 0: PASS)")
    print()

    # ---- Step 2: Map iteration ----
    chaos_map = HighDimChaosMap(params, word_size=word_size, seed=42)
    traj = chaos_map.iterate(500)
    assert traj.shape == (500, n), f"Wrong trajectory shape: {traj.shape}"
    assert traj.min() >= 0 and traj.max() < (1 << word_size), \
        "Trajectory values out of [0, 2^w) range"

    print(f"[Iteration] 500 steps, range: [{traj.min()}, {traj.max()}]")
    print(f"             normalized: [{traj.min()/65536:.4f}, {traj.max()/65536:.4f}]")
    print()

    # ---- Step 3: Chaos verification ----
    result = verify_chaos(chaos_map, steps=2000)
    assert result['is_chaotic'], \
        f"Chaos verification failed: min LE = {result['min_le']:.6f}"
    assert np.all(result['lyap_exps'] > 0.01), \
        f"Not all LEs > 0.01: {result['lyap_exps']}"

    print(f"[Chaos] Lyapunov exponents: {result['lyap_exps']}")
    print(f"        min LE = {result['min_le']:.6f}")
    print(f"        max LE = {result['max_le']:.6f}")
    print(f"        variance = {result['variance']}")
    print(f"        chaotic: PASS")
    print()

    # ---- Step 4: LaTeX system equations ----
    F = make_F(n, params['a_j1'], params['a_jn'], params['b'])
    tex_symbolic = latex_system_like_figure_full(F, with_mod=False)

    print("[LaTeX] System equations:")
    print(tex_symbolic)

    return params, tex_symbolic


if __name__ == "__main__":
    try:
        params, tex = test_generation()
        print("\n" + "=" * 70)
        print("R1-LY generation test: ALL CHECKS PASSED")
        print("=" * 70)
    except Exception as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
