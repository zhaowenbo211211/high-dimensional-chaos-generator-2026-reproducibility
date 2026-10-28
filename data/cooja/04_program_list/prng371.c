// Maintainer: zhaowenbo
#include "prng371.h"

/* ---- Shift-and-add multipliers (zero DSP, bit-exact with HLS) ---- */
static inline uint16_t mul_A_J1_0(uint16_t x) {
    return (uint16_t)((x << 9) - (x << 7) - (x << 3) - (x << 2) - x);
}
static inline uint16_t mul_A_J1_1(uint16_t x) {
    return (uint16_t)((x << 9) - (x << 7) - (x << 3) - (x << 1) - x);
}
static inline uint16_t mul_A_J1_2(uint16_t x) {
    return (uint16_t)((x << 9) - (x << 7) - (x << 3) - x);
}
static inline uint16_t mul_A_JN_0(uint16_t x) {
    return (uint16_t)((x << 5) + (x << 4) + (x << 1));
}
static inline uint16_t mul_A_JN_1(uint16_t x) {
    return (uint16_t)((x << 5) + (x << 4));
}
static inline uint16_t mul_A_JN_2(uint16_t x) {
    return (uint16_t)((x << 6) - (x << 4) - (x << 1));
}
static inline uint16_t mul_B_VAL(uint16_t x) {
    return (uint16_t)((x << 7) - (x << 4) - x);
}

static inline uint64_t rotl64(uint64_t v, uint8_t k) {
    return (v << k) | (v >> (64 - k));
}

/* ---- Internal state (matches prng_top.cpp pipeline registers) ---- */
static uint16_t x1_reg, x2_reg, x3_reg;
static uint16_t s0_reg, s1_reg, s2_reg;
static uint16_t x1_next_reg, x2_next_reg, x3_next_reg, x4_next_reg;
static uint32_t temp_reg;
static uint8_t  rot_cnt;   /* 0..14 */
static uint64_t u_prev1, u_prev2;
static uint8_t  pp_init;   /* 0 or 1 */
static uint8_t  init_done; /* 0 or 1 */

void prng371_init(uint16_t x1, uint16_t x2, uint16_t x3, uint16_t x4)
{
    if (x1 == 0 && x2 == 0 && x3 == 0 && x4 == 0) {
        x1 = 1000; x2 = 2000; x3 = 3000; x4 = 4000;
    }

    x1_reg = x1;  x2_reg = x2;  x3_reg = x3;
    s0_reg = x4;  s1_reg = 0;   s2_reg = 0;

    rot_cnt = 0;
    x1_next_reg = x2_next_reg = x3_next_reg = x4_next_reg = 0;
    temp_reg = 0;
    u_prev1 = u_prev2 = 0;
    pp_init = 0;
    init_done = 1;

    /* Warmup: 2000 iterations (discard outputs) */
    for (int i = 0; i < 2000; i++) {
        (void)prng371_next();
    }
}

uint64_t prng371_next(void)
{
    if (!init_done) {
        prng371_init(0, 0, 0, 0);
    }

    /* ---- Stage 1: shift-add + logistic pipeline ---- */
    uint16_t x1n = mul_A_J1_0(x1_reg) + mul_A_JN_0(s2_reg);
    uint16_t x2n = mul_A_J1_1(x2_reg) + mul_A_JN_1(s1_reg);
    uint16_t x3n = mul_A_J1_2(x3_reg) + mul_A_JN_2(s0_reg);

    uint16_t z   = x3_reg + s0_reg;
    uint16_t x4n = mul_B_VAL(x2_reg) + (uint16_t)(temp_reg >> 14);
    uint16_t inv_z = (~z) & 0xFFFF;
    temp_reg = (uint32_t)z * (uint32_t)inv_z;

    x1_next_reg = x1n;
    x2_next_reg = x2n;
    x3_next_reg = x3n;
    x4_next_reg = x4n;

    /* ---- Stage 2: state update + delay chain ---- */
    s2_reg = s1_reg;
    s1_reg = s0_reg;
    s0_reg = x4_next_reg;
    x1_reg = x1_next_reg;
    x2_reg = x2_next_reg;
    x3_reg = x3_next_reg;

    /* ---- Stage 2: bit-rotation enhancement ---- */
    uint16_t r1 = x1_reg, r2 = x2_reg, r3 = x3_reg, r0 = s0_reg;
    uint8_t k = 1 + rot_cnt;

    uint64_t Z = ((uint64_t)r0 << 48) |
                 ((uint64_t)r3 << 32) |
                 ((uint64_t)r2 << 16) |
                 ((uint64_t)r1);

    uint64_t Z_rot;
    if (k == 0) {
        Z_rot = Z;
    } else {
        Z_rot = (Z << k) | (Z >> (64 - k));
    }

    r1 = (uint16_t)(Z_rot & 0xFFFF);
    r2 = (uint16_t)((Z_rot >> 16) & 0xFFFF);
    r3 = (uint16_t)((Z_rot >> 32) & 0xFFFF);
    r0 = (uint16_t)((Z_rot >> 48) & 0xFFFF);

    rot_cnt = (rot_cnt == 14) ? 0 : (rot_cnt + 1);

    /* ---- Stage 3: output mixer ---- */
    uint64_t u_t = ((uint64_t)r0 << 48) |
                   ((uint64_t)r3 << 32) |
                   ((uint64_t)r2 << 16) |
                   ((uint64_t)r1);

    uint64_t y_t;
    if (!pp_init) {
        y_t = u_t;
        pp_init = 1;
    } else {
        uint64_t rot1 = rotl64(u_prev1, 17);
        uint64_t rot2 = rotl64(u_prev2, 43);
        y_t = u_t ^ rot1 ^ rot2;
    }

    u_prev2 = u_prev1;
    u_prev1 = u_t;

    return y_t;
}

void prng371_generate_bytes(uint8_t *buf, uint16_t n_bytes)
{
    uint16_t i = 0;
    while (i < n_bytes) {
        uint64_t w = prng371_next();
        /* little-endian per 64-bit word */
        uint8_t tmp[8];
        tmp[0] = (uint8_t)(w & 0xFF);
        tmp[1] = (uint8_t)((w >> 8) & 0xFF);
        tmp[2] = (uint8_t)((w >> 16) & 0xFF);
        tmp[3] = (uint8_t)((w >> 24) & 0xFF);
        tmp[4] = (uint8_t)((w >> 32) & 0xFF);
        tmp[5] = (uint8_t)((w >> 40) & 0xFF);
        tmp[6] = (uint8_t)((w >> 48) & 0xFF);
        tmp[7] = (uint8_t)((w >> 56) & 0xFF);

        for (int j = 0; j < 8 && i < n_bytes; j++, i++) {
            buf[i] = tmp[j];
        }
    }
}

void prng371_crypt(uint8_t *data, uint16_t len)
{
    uint16_t i = 0;
    while (i < len) {
        uint64_t w = prng371_next();
        uint8_t tmp[8];
        tmp[0] = (uint8_t)(w & 0xFF);
        tmp[1] = (uint8_t)((w >> 8) & 0xFF);
        tmp[2] = (uint8_t)((w >> 16) & 0xFF);
        tmp[3] = (uint8_t)((w >> 24) & 0xFF);
        tmp[4] = (uint8_t)((w >> 32) & 0xFF);
        tmp[5] = (uint8_t)((w >> 40) & 0xFF);
        tmp[6] = (uint8_t)((w >> 48) & 0xFF);
        tmp[7] = (uint8_t)((w >> 56) & 0xFF);

        for (int j = 0; j < 8 && i < len; j++, i++) {
            data[i] ^= tmp[j];
        }
    }
}
