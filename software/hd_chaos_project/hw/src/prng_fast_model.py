#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Python reference model — bit-exact with hw/src/hls_src/prng_top.cpp (3-stage pipeline).

Matches the C golden model in hw/testbench/tb_prng.cpp exactly.
Verified: HLS CSim PASS (1000/1000 match, warmup=1000).
"""

import numpy as np

MASK16 = 0xFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# Shift-and-add multipliers — MUST match auto-generated hw/src_371/shift_mul.h EXACTLY
def _mul(a, x):
    """Bit-exact with the auto-generated shift_mul.h (ap_uint<16> semantics)."""
    x = int(x) & MASK16
    if a == 371:
        t = (x << 9) & MASK16; t = (t - (x << 7)) & MASK16; t = (t - (x << 3)) & MASK16
        t = (t - (x << 2)) & MASK16; t = (t - x) & MASK16; return t & MASK16
    if a == 373:
        t = (x << 9) & MASK16; t = (t - (x << 7)) & MASK16; t = (t - (x << 3)) & MASK16
        t = (t - (x << 1)) & MASK16; t = (t - x) & MASK16; return t & MASK16
    if a == 375:
        t = (x << 9) & MASK16; t = (t - (x << 7)) & MASK16; t = (t - (x << 3)) & MASK16
        t = (t - x) & MASK16; return t & MASK16
    if a == 50:
        t = (x << 5) & MASK16; t = (t + (x << 4)) & MASK16
        t = (t + (x << 1)) & MASK16; return t & MASK16
    if a == 48:
        t = (x << 5) & MASK16; t = (t + (x << 4)) & MASK16; return t & MASK16
    if a == 46:
        t = (x << 6) & MASK16; t = (t - (x << 4)) & MASK16
        t = (t - (x << 1)) & MASK16; return t & MASK16
    if a == 111:
        t = (x << 7) & MASK16; t = (t - (x << 4)) & MASK16
        t = (t - x) & MASK16; return t & MASK16
    # Fallback for other constants (271, 273, 275, 81)
    if a == 271:
        t = (x << 8) & MASK16; t = (t + (x << 3)) & MASK16; t = (t + (x << 2)) & MASK16
        t = (t + (x << 1)) & MASK16; t = (t + x) & MASK16; return t & MASK16
    if a == 273:
        t = (x << 8) & MASK16; t = (t + (x << 4)) & MASK16
        t = (t + x) & MASK16; return t & MASK16
    if a == 275:
        t = (x << 8) & MASK16; t = (t + (x << 4)) & MASK16; t = (t + (x << 1)) & MASK16
        t = (t + x) & MASK16; return t & MASK16
    if a == 81:
        t = (x << 6) & MASK16; t = (t + (x << 4)) & MASK16
        t = (t + x) & MASK16; return t & MASK16
    return (a * x) & MASK16


class ChaosPRNGFast:
    """Bit-exact Python reference for the 3-stage pipelined HLS PRNG."""

    def __init__(self, a_j1, a_jn, b_val, init_x=(1000, 2000, 3000, 4000)):
        self.a_j1 = np.array(a_j1, dtype=np.int64)
        self.a_jn = np.array(a_jn, dtype=np.int64)
        self.b_val = int(b_val)

        # Stage 2: current state
        self.x1, self.x2, self.x3 = int(init_x[0]), int(init_x[1]), int(init_x[2])
        self.s0, self.s1, self.s2 = int(init_x[3]), 0, 0

        # Stage 1 registers
        self.x1nr = self.x2nr = self.x3nr = self.x4nr = 0

        # DSP pipeline register
        self.temp_reg = 0

        # Stage 2 rotation output registers
        self.x1r, self.x2r = int(init_x[0]), int(init_x[1])
        self.x3r, self.s0r = int(init_x[2]), int(init_x[3])

        self.rot_cnt = 0

        # Stage 3 post-processor
        self.u_prev1 = self.u_prev2 = 0
        self.pp_init = False

    def next(self) -> int:
        # ---- Stage 1: shift-add (all ops mod 2^16 via intermediate masking) ----
        x1n = (_mul(self.a_j1[0], self.x1) + _mul(self.a_jn[0], self.s2)) & MASK16
        x2n = (_mul(self.a_j1[1], self.x2) + _mul(self.a_jn[1], self.s1)) & MASK16
        x3n = (_mul(self.a_j1[2], self.x3) + _mul(self.a_jn[2], self.s0)) & MASK16

        # Logistic core with DSP pipeline
        z = (self.x3 + self.s0) & MASK16
        x4n = (_mul(self.b_val, self.x2) + (self.temp_reg >> 14)) & MASK16
        inv_z = (~z) & MASK16
        self.temp_reg = (z * inv_z) & 0xFFFFFFFF

        # Register stage 1
        self.x1nr, self.x2nr, self.x3nr, self.x4nr = x1n, x2n, x3n, x4n

        # ---- Stage 2: state update + bit-rotation ----
        self.s2, self.s1 = self.s1, self.s0
        self.s0 = self.x4nr
        self.x1, self.x2, self.x3 = self.x1nr, self.x2nr, self.x3nr

        # Bit-rotation on updated state
        k = 1 + (self.rot_cnt % 15)
        s0i, x3i, x2i, x1i = int(self.s0), int(self.x3), int(self.x2), int(self.x1)
        Z = ((s0i << 48) | (x3i << 32) | (x2i << 16) | x1i) & MASK64
        Z_rot = ((Z << k) | (Z >> (64 - k))) & MASK64 if k else Z
        r1 = Z_rot & 0xFFFF
        r2 = (Z_rot >> 16) & 0xFFFF
        r3 = (Z_rot >> 32) & 0xFFFF
        r0 = (Z_rot >> 48) & 0xFFFF
        self.rot_cnt = 0 if self.rot_cnt == 14 else self.rot_cnt + 1

        # Register stage 2
        self.x1r, self.x2r, self.x3r, self.s0r = r1, r2, r3, r0

        # ---- Stage 3: post-processor ----
        s0ri, x3ri, x2ri, x1ri = int(self.s0r), int(self.x3r), int(self.x2r), int(self.x1r)
        u_t = ((s0ri << 48) | (x3ri << 32) | (x2ri << 16) | x1ri) & MASK64
        if not self.pp_init:
            y_t = u_t
            self.pp_init = True
        else:
            rt1 = ((self.u_prev1 << 17) | (self.u_prev1 >> 47)) & MASK64
            rt2 = ((self.u_prev2 << 43) | (self.u_prev2 >> 21)) & MASK64
            y_t = u_t ^ rt1 ^ rt2
        self.u_prev2, self.u_prev1 = self.u_prev1, u_t
        return y_t & MASK64

    def generate(self, n):
        return [self.next() for _ in range(n)]


if __name__ == "__main__":
    prng = ChaosPRNGFast([271, 273, 275], [50, 48, 46], 81)
    for _ in range(1000):
        prng.next()
    w = prng.next()
    print(f"HLS CSim:  0xa30b60890048de4f")
    print(f"Python:    0x{int(w):016x}")
    print(f"Match: {'YES' if f'{int(w):016x}' == 'a30b60890048de4f' else 'NO'}")
