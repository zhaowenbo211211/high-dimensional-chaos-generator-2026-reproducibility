#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Stream cipher test — Python golden model vs HLS CSim reference.

Verifies bit-exact keystream alignment and encrypt/decrypt roundtrip.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from hw.src_371.golden_prng import ChaosPRNG371

# HLS CSim reference (from tb_prng_371.cpp, Test 4)
PLAINTEXT = b"ChaosPRNG-371: Lightweight stream cipher for IoT security."
HLS_CIPHER = ("5d43d6ee815cd11f94a9c0d4548111ebdc23d51d94de33ce6059c179"
              "7ed5e75c6d3c2e7204d3d61beda0f4188369c0f259a9b378dc9a71d41084")
INIT = (0, 0, 0, 0)  # all-zero → defaults [1000,2000,3000,4000]
WARMUP = 2000


def test_keystream_alignment():
    """Python golden model keystream matches HLS CSim output."""
    prng = ChaosPRNG371(init_x=INIT)
    keystream = prng.generate_bytes(len(PLAINTEXT))
    ciphertext = bytes(p ^ k for p, k in zip(PLAINTEXT, keystream))
    assert ciphertext.hex() == HLS_CIPHER, \
        f"Cipher mismatch:\n  HLS: {HLS_CIPHER}\n  Py:  {ciphertext.hex()}"
    print("  PASS — Python ciphertext == HLS CSim ciphertext")


def test_encrypt_decrypt_roundtrip():
    """XOR encrypt + decrypt recovers original plaintext."""
    prng = ChaosPRNG371(init_x=INIT)
    keystream = prng.generate_bytes(len(PLAINTEXT))
    ciphertext = bytes(p ^ k for p, k in zip(PLAINTEXT, keystream))

    prng2 = ChaosPRNG371(init_x=INIT)
    keystream2 = prng2.generate_bytes(len(PLAINTEXT))
    recovered = bytes(c ^ k for c, k in zip(ciphertext, keystream2))
    assert recovered == PLAINTEXT, f"Roundtrip failed: {recovered}"
    print("  PASS — decrypt(encrypt(msg)) == msg")


def test_different_init_yields_different_keystream():
    """Different init values produce different keystreams."""
    prng1 = ChaosPRNG371(init_x=(1000, 2000, 3000, 4000))
    prng2 = ChaosPRNG371(init_x=(1001, 2000, 3000, 4000))
    ks1 = prng1.generate_bytes(64)
    ks2 = prng2.generate_bytes(64)
    assert ks1 != ks2, "Different inits should yield different keystreams"
    print("  PASS — different init → different keystream")


def test_ciphertext_looks_random():
    """Ciphertext is not all zeros and not equal to plaintext."""
    prng = ChaosPRNG371(init_x=INIT)
    keystream = prng.generate_bytes(len(PLAINTEXT))
    ciphertext = bytes(p ^ k for p, k in zip(PLAINTEXT, keystream))
    assert ciphertext != PLAINTEXT, "Ciphertext should differ from plaintext"
    assert any(b != 0 for b in ciphertext), "Ciphertext should not be all zeros"
    print("  PASS — ciphertext looks random")


def test_large_transfer():
    """1 MB stream cipher transfer (performance sanity check)."""
    prng = ChaosPRNG371(init_x=INIT)
    size = 1_000_000
    keystream = prng.generate_bytes(size)
    assert len(keystream) == size
    # Quick check: first and last 8 bytes of keystream differ
    assert keystream[:8] != keystream[-8:], "Keystream should be non-repeating"
    print(f"  PASS — {size:,} bytes keystream generated, non-repeating")


if __name__ == "__main__":
    print("=" * 60)
    print("Stream Cipher Test — golden_prng (A_odd=371)")
    print("=" * 60)
    print(f"  Init:       {INIT} (all-zero → defaults)")
    print(f"  Warmup:     {WARMUP}")
    print(f"  Plaintext:  {PLAINTEXT.decode()}")
    print(f"  Ref cipher: {HLS_CIPHER[:32]}...")
    print()

    tests = [f for f in sorted(globals()) if f.startswith("test_")]
    passed = 0
    for name in tests:
        try:
            globals()[name]()
            passed += 1
        except Exception as e:
            print(f"  FAIL — {name}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
