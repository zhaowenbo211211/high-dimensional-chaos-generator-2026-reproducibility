#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Tests for the chaos-based PRNG: post-processor mixing and full PRNG engine.
"""

import sys
import numpy as np

from chaos.application.prng import PostProcessor, ChaosPRNG


# ========================================================================
# PostProcessor tests
# ========================================================================

def test_concat_packing():
    """Concatenation packs 4×16-bit into correct 64-bit word."""
    pp = PostProcessor(word_bits=16)
    state = np.array([0x1111, 0x2222, 0x3333, 0x4444], dtype=np.int64)
    u = pp._concat(state)
    assert u == 0x4444333322221111, f"concat error: {u:#018x}"


def test_rotl64_basic():
    """ROTL64 produces correct rotations."""
    x = 0x8000000000000001  # MSB + LSB set
    r1 = PostProcessor._rotl64(x, 1)
    assert r1 == 0x0000000000000003, f"ROTL64(x,1) = {r1:#018x}"
    r63 = PostProcessor._rotl64(x, 63)
    assert r63 == 0xC000000000000000, f"ROTL64(x,63) = {r63:#018x}"


def test_rotl64_roundtrip():
    """ROTL64(x, r) followed by ROTL64(·, 64-r) recovers x."""
    x = 0xDEADBEEFCAFEBABE
    for r in [1, 7, 17, 43, 63]:
        assert PostProcessor._rotl64(PostProcessor._rotl64(x, r), 64 - r) == x


def test_post_processor_mix_output_range():
    """Mixed output fits in 64 bits."""
    pp = PostProcessor(word_bits=16)
    for _ in range(100):
        state = np.random.randint(0, 65536, size=4, dtype=np.int64)
        y = pp.mix(state)
        assert 0 <= y < (1 << 64), f"output out of 64-bit range: {y}"


def test_post_processor_deterministic():
    """Same sequence of inputs → same outputs."""
    pp1 = PostProcessor()
    pp2 = PostProcessor()
    rng = np.random.RandomState(42)
    for _ in range(50):
        s = rng.randint(0, 65536, size=4, dtype=np.int64)
        assert pp1.mix(s.copy()) == pp2.mix(s.copy())


def test_post_processor_reset():
    """reset() clears delay line."""
    pp = PostProcessor()
    pp.mix(np.array([1, 2, 3, 4], dtype=np.int64))
    pp.mix(np.array([5, 6, 7, 8], dtype=np.int64))
    pp.reset()
    # After reset, first output uses no history (same as fresh pp)
    pp2 = PostProcessor()
    s = np.array([100, 200, 300, 400], dtype=np.int64)
    assert pp.mix(s) == pp2.mix(s)


def test_mix_batch():
    """mix_batch produces same results as repeated mix()."""
    pp1 = PostProcessor()
    pp2 = PostProcessor()
    states = np.random.randint(0, 65536, size=(20, 4), dtype=np.int64)

    batch_out = pp1.mix_batch(states)
    single_out = np.array([pp2.mix(s) for s in states], dtype=np.uint64)
    assert np.array_equal(batch_out, single_out)


# ========================================================================
# ChaosPRNG tests
# ========================================================================

def test_prng_factory():
    """ChaosPRNG.from_r1_ly creates a working PRNG."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    assert prng.word_size == 16
    state = prng.get_state()
    assert state.shape == (4,)
    assert state.min() >= 0 and state.max() < 65536


def test_prng_next_output():
    """next() returns a 64-bit value in range."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    for _ in range(100):
        y = prng.next()
        assert isinstance(y, int) or isinstance(y, np.integer)
        assert 0 <= y < (1 << 64)


def test_prng_generate_shape():
    """generate(n) returns n uint64 values."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    out = prng.generate(500)
    assert out.shape == (500,)
    assert out.dtype == np.uint64
    assert out.min() >= 0


def test_prng_generate_bytes():
    """generate_bytes returns correct number of bytes."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    b = prng.generate_bytes(1000)
    assert isinstance(b, bytes)
    assert len(b) == 1000


def test_prng_deterministic():
    """Same seed → identical output stream."""
    p1 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=123)
    p2 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=123)
    out1 = p1.generate(200)
    out2 = p2.generate(200)
    assert np.array_equal(out1, out2)


def test_prng_different_seed():
    """Different seeds → different streams."""
    p1 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    p2 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=123)
    out1 = p1.generate(100)
    out2 = p2.generate(100)
    assert not np.array_equal(out1, out2)


def test_prng_set_get_state():
    """set_state + warmup → get_state returns valid range."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    prng.set_state(1000, 2000, 3000, 4000)
    s = prng.get_state()
    # After warmup, state has evolved — but must be in valid range
    assert s.shape == (4,)
    assert s.min() >= 0 and s.max() < 65536


def test_prng_reset():
    """reset() + set_state → deterministic restart."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    prng.set_state(1000, 2000, 3000, 4000)
    out1 = prng.generate(50)
    prng.reset(seed=42)
    prng.set_state(1000, 2000, 3000, 4000)
    out2 = prng.generate(50)
    assert np.array_equal(out1, out2)


def test_seed_from_key_deterministic():
    """Same key → identical output stream."""
    p1 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    p2 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    p1.seed_from_key(0xDEADBEEF)
    p2.seed_from_key(0xDEADBEEF)
    assert np.array_equal(p1.generate(100), p2.generate(100))


def test_seed_from_key_differs():
    """Different keys → different streams."""
    p1 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    p2 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    p1.seed_from_key(12345)
    p2.seed_from_key(67890)
    assert not np.array_equal(p1.generate(50), p2.generate(50))


def test_seed_from_key_resets_processor():
    """seed_from_key resets post-processor (no history leakage)."""
    p1 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    p2 = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    # Advance p1, then re-seed
    p1.generate(50)
    p1.seed_from_key(0xCAFE)
    p2.seed_from_key(0xCAFE)
    assert np.array_equal(p1.generate(50), p2.generate(50))


def test_export_raw_creates_file():
    """export_binary_sequence (raw) creates expected file."""
    import tempfile, os
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        prng.export_binary_sequence(tmp_path, length=10000, mode="raw")
        size = os.path.getsize(tmp_path)
        expected = (10000 + 7) // 8
        assert size == expected, f"raw file size {size} != expected {expected}"
    finally:
        os.unlink(tmp_path)


def test_export_ascii_creates_file():
    """export_binary_sequence (ascii) creates '0'/'1' text file."""
    import tempfile, os
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        prng.export_binary_sequence(tmp_path, length=1000, mode="ascii")
        with open(tmp_path, "r") as f:
            content = f.read()
        assert len(content) == 1000
        assert all(c in "01" for c in content)
    finally:
        os.unlink(tmp_path)


def test_export_mode_validation():
    """Invalid mode raises ValueError."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    try:
        prng.export_binary_sequence("/tmp/test.bin", length=100, mode="hex")
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_prng_output_not_constant():
    """PRNG output varies (not stuck at a single value)."""
    prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
    out = prng.generate(500)
    assert np.std(out.astype(np.float64)) > 0, "output is constant"


# ========================================================================
# Runner
# ========================================================================

if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
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
