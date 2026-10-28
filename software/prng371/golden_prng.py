#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Python wrapper for the C golden model — 100% bit-exact with HLS (A_odd=371)."""

import ctypes
import os

_SO_PATH = os.path.join(os.path.dirname(__file__), "golden_371.so")
_lib = ctypes.CDLL(_SO_PATH)


class GoldenState(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_uint16), ("x2", ctypes.c_uint16), ("x3", ctypes.c_uint16),
        ("s0", ctypes.c_uint16), ("s1", ctypes.c_uint16), ("s2", ctypes.c_uint16),
        ("x1nr", ctypes.c_uint16), ("x2nr", ctypes.c_uint16),
        ("x3nr", ctypes.c_uint16), ("x4nr", ctypes.c_uint16),
        ("temp_reg", ctypes.c_uint32),
        ("x1r", ctypes.c_uint16), ("x2r", ctypes.c_uint16),
        ("x3r", ctypes.c_uint16), ("s0r", ctypes.c_uint16),
        ("rot_cnt", ctypes.c_uint16),
        ("u_prev1", ctypes.c_uint64), ("u_prev2", ctypes.c_uint64),
        ("pp_init", ctypes.c_int),
    ]

_lib.golden_init.argtypes = [ctypes.POINTER(GoldenState),
    ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16]
_lib.golden_next.argtypes = [ctypes.POINTER(GoldenState)]
_lib.golden_next.restype = ctypes.c_uint64


class ChaosPRNG371:
    """Bit-exact Python interface to the HLS-verified C golden model (A_odd=371)."""

    def __init__(self, init_x=(1000, 2000, 3000, 4000), warmup: int = 2000):
        self._st = GoldenState()
        x1, x2, x3, x4 = init_x
        _lib.golden_init(self._st,
            ctypes.c_uint16(x1), ctypes.c_uint16(x2),
            ctypes.c_uint16(x3), ctypes.c_uint16(x4))
        self._do_warmup(warmup)

    def _do_warmup(self, n: int):
        for _ in range(n):
            _lib.golden_next(self._st)

    def next(self) -> int:
        return _lib.golden_next(self._st)

    def generate(self, n: int):
        return [self.next() for _ in range(n)]

    def generate_bytes(self, n_bytes: int) -> bytes:
        words = (n_bytes + 7) // 8
        data = bytearray()
        for _ in range(words):
            w = self.next()
            data.extend(w.to_bytes(8, 'little'))
        return bytes(data[:n_bytes])


if __name__ == "__main__":
    prng = ChaosPRNG371()
    w = prng.next()
    print(f"HLS CSim:  0x7bc4f78f3593f82a")
    print(f"C golden:  0x{int(w):016x}")
    match = f"{int(w):016x}" == "7bc4f78f3593f82a"
    print(f"Match: {'YES' if match else 'NO'}")

    # First 5 outputs without warmup
    prng2 = ChaosPRNG371(warmup=0)
    print("\nFirst 5 (no warmup):")
    for i in range(5):
        print(f"  [{i}] 0x{prng2.next():016x}")
