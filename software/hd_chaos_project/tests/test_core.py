#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Unit tests for core chaotic map modules:
  - chaos.core.params:  ComplementaryChaosParams (R1-LY synthesis)
  - chaos.core.base_map: HighDimChaosMap (fixed-point map)
  - chaos.core.enhanced_map: HighDimChaosMapEnhanced (bit-rotation enhancer)
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Parameter synthesis tests
# ---------------------------------------------------------------------------

def test_params_constraints():
    """Parameter synthesis obeys parity and positivity constraints."""
    from chaos.core.params import ComplementaryChaosParams

    for n in [3, 4, 8]:
        synth = ComplementaryChaosParams(n=n, A_odd=371, B_even=50)
        p = synth.synthesize()
        assert all(p['a_j1'] % 2 == 1), f"n={n}: a_j1 must be all odd"
        assert all(p['a_jn'] % 2 == 0), f"n={n}: a_jn must be all even"
        assert all(p['a_jn'] > 0), f"n={n}: a_jn must be strictly positive"
        assert p['discriminant'] > 0, f"n={n}: cone-field discriminant must be > 0"


def test_params_deterministic():
    """Same inputs produce identical parameter sets."""
    from chaos.core.params import ComplementaryChaosParams

    s1 = ComplementaryChaosParams(n=4, A_odd=371, B_even=50)
    s2 = ComplementaryChaosParams(n=4, A_odd=371, B_even=50)
    p1 = s1.synthesize()
    p2 = s2.synthesize()
    assert np.array_equal(p1['a_j1'], p2['a_j1'])
    assert np.array_equal(p1['a_jn'], p2['a_jn'])
    assert p1['b'] == p2['b']


def test_params_different_n():
    """Different dimensions produce correct-length sequences."""
    from chaos.core.params import ComplementaryChaosParams

    for n in [3, 5, 7]:
        p = ComplementaryChaosParams(n=n, A_odd=371, B_even=50).synthesize()
        assert len(p['a_j1']) == n - 1
        assert len(p['a_jn']) == n - 1


def test_params_adaptive_adjustment():
    """Synthesis auto-adjusts when constraints are tight."""
    from chaos.core.params import ComplementaryChaosParams

    synth = ComplementaryChaosParams(n=4, A_odd=5, B_even=10, eta=0.05, tau=0.1)
    p = synth.synthesize(max_iter=200)
    assert p['discriminant'] > 0
    assert p['b'] >= 2


# ---------------------------------------------------------------------------
# Base map tests
# ---------------------------------------------------------------------------

def _make_params(n=4):
    from chaos.core.params import ComplementaryChaosParams
    return ComplementaryChaosParams(n=n, A_odd=371, B_even=50).synthesize()


def test_base_map_initialization():
    """Base map initializes with correct dimensions."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    m = HighDimChaosMap(p, word_size=16, seed=42)
    assert m.n == 4
    assert m.mod == 65536
    assert m.state_dim == 2 * m.n - 2
    assert m.state.shape == (m.state_dim,)


def test_base_map_iteration_shape():
    """iterate() returns correct shape."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    m = HighDimChaosMap(p, word_size=16, seed=42)
    traj = m.iterate(100)
    assert traj.shape == (100, 4)


def test_base_map_value_range():
    """All outputs stay in [0, 2^w)."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    m = HighDimChaosMap(p, word_size=16, seed=42)
    traj = m.iterate(200)
    assert traj.min() >= 0
    assert traj.max() < 65536


def test_base_map_deterministic():
    """Same seed produces identical trajectories."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    t1 = HighDimChaosMap(p, word_size=16, seed=42).iterate(50)
    t2 = HighDimChaosMap(p, word_size=16, seed=42).iterate(50)
    assert np.array_equal(t1, t2)


def test_base_map_seed_differs():
    """Different seeds produce different trajectories."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    t1 = HighDimChaosMap(p, word_size=16, seed=42).iterate(50)
    t2 = HighDimChaosMap(p, word_size=16, seed=123).iterate(50)
    assert not np.array_equal(t1, t2)


def test_base_map_set_state_from_physical():
    """set_state_from_physical sets correct internal state."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    m = HighDimChaosMap(p, word_size=16)
    init = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    m.set_state_from_physical(init)
    phys = m.get_physical_state()
    assert np.array_equal(phys, init)


def test_base_map_next():
    """next() returns shape (n,) and values in range."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(4)
    m = HighDimChaosMap(p, word_size=16, seed=42)
    s = m.next()
    assert s.shape == (4,)
    assert s.min() >= 0 and s.max() < 65536


def test_base_map_n_dimensional():
    """Base map works for n > 4."""
    from chaos.core.base_map import HighDimChaosMap

    p = _make_params(8)
    m = HighDimChaosMap(p, word_size=16, seed=42)
    traj = m.iterate(50)
    assert traj.shape == (50, 8)
    assert traj.min() >= 0 and traj.max() < 65536


# ---------------------------------------------------------------------------
# Enhanced map tests (bit-rotation enhancer)
# ---------------------------------------------------------------------------

def test_enhanced_initialization():
    """Enhanced map inherits base map and adds rotation state."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)
    assert m.n == 4
    assert m.total_bits == 4 * 16
    assert m.iter_count == 0
    assert hasattr(m, '_pack_state_bits')
    assert hasattr(m, '_unpack_state_bits')
    assert hasattr(m, '_rol_total')


def test_enhanced_iteration_shape():
    """Enhanced map iterate() returns correct shape."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)
    traj = m.iterate(100)
    assert traj.shape == (100, 4)


def test_enhanced_value_range():
    """Enhanced map outputs stay in [0, 2^w)."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)
    traj = m.iterate(200)
    assert traj.min() >= 0
    assert traj.max() < 65536


def test_enhanced_differs_from_base():
    """Enhanced trajectory differs from base trajectory (enhancement works)."""
    from chaos.core.base_map import HighDimChaosMap
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    base = HighDimChaosMap(p, word_size=16, seed=42).iterate(50)
    enh = HighDimChaosMapEnhanced(p, word_size=16, seed=42).iterate(50)
    assert not np.array_equal(base, enh)


def test_enhanced_iter_count_advances():
    """iter_count increments correctly with each step."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)
    assert m.iter_count == 0
    m.iterate(10)
    assert m.iter_count == 10
    m.iterate(5)
    assert m.iter_count == 15


def test_bit_pack_unpack_roundtrip():
    """Pi^{-1}(Pi(x)) == x for all test vectors."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)

    test_states = [
        np.array([0, 0, 0, 0], dtype=np.int64),
        np.array([65535, 65535, 65535, 65535], dtype=np.int64),
        np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        np.random.randint(0, 65536, size=4, dtype=np.int64),
    ]
    for state in test_states:
        packed = m._pack_state_bits(state)
        unpacked = m._unpack_state_bits(packed)
        assert np.array_equal(state, unpacked)


def test_rol_roundtrip():
    """ROL_k followed by ROL_{nw-k} recovers original value."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)

    Z = 0x123456789ABC
    for k in [1, 3, 7, 15, 31]:
        Z_rot = m._rol_total(Z, k)
        Z_back = m._rol_total(Z_rot, m.total_bits - k)
        assert Z_back == Z, f"ROL roundtrip failed for k={k}"


def test_k_sequence_periodic():
    """k_t = 1 + (t mod (w-1)) produces correct periodic sequence."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)

    backup = m.get_rotation_schedule_state()
    k_seq = m.debug_k_sequence(32)
    m.set_rotation_schedule_state(backup)

    assert min(k_seq) >= 1
    assert max(k_seq) <= 15

    expected_cycle = list(range(1, 16))
    for i, k in enumerate(k_seq):
        assert k == expected_cycle[i % 15]


def test_k_sequence_after_iteration():
    """k-sequence is continuous across iterate() calls."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)
    m.iterate(20)

    backup = m.get_rotation_schedule_state()
    k_seq = m.debug_k_sequence(16)
    m.set_rotation_schedule_state(backup)

    expected = [1 + ((20 + i) % 15) for i in range(16)]
    assert k_seq == expected, f"After 20 iters: got {k_seq}, expected {expected}"


def test_rotation_state_save_restore():
    """get/set_rotation_schedule_state roundtrip preserves iter_count."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    m = HighDimChaosMapEnhanced(p, word_size=16, seed=42)
    m.iterate(42)

    state = m.get_rotation_schedule_state()
    assert state['iter_count'] == 42

    m.iterate(10)
    assert m.iter_count == 52

    m.set_rotation_schedule_state(state)
    assert m.iter_count == 42


def test_enhanced_deterministic():
    """Same seed + same iters = same trajectory for enhanced map."""
    from chaos.core.enhanced_map import HighDimChaosMapEnhanced

    p = _make_params(4)
    t1 = HighDimChaosMapEnhanced(p, word_size=16, seed=42).iterate(50)
    t2 = HighDimChaosMapEnhanced(p, word_size=16, seed=42).iterate(50)
    assert np.array_equal(t1, t2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed != len(tests):
        sys.exit(1)
