# hw/src/org_src/ — Original HLS Sources (chaos_prng v1)

Reference copy of the first-generation PRNG design. Kept for comparison and archival.

| File | Description |
|---|---|
| `prng_top.cpp` | Original single-cycle PRNG (no pipeline registers) |
| `prng_top.h` | Interface + register map |
| `chaos_core.h` | Type defs + parameter includes |
| `shift_mul.h` | Shift-and-add constant multipliers |
| `chaos_params.h` | #define A_J1_0=271, A_JN_0=50, B_VAL=81 |
| `tb_prng.cpp` | Original testbench with golden model |

## Key Differences vs hls_src/ (fast version)

| | org_src (v1) | hls_src (v2, canonical) |
|---|---|---|
| Pipeline | Single-cycle, all combinational | 3-stage (shift-add → rotation → post-process) |
| DSP register | None | `temp_reg` registers z*inv_z output |
| State feedback | Immediate | 1-cycle delayed (adds complexity) |
| Critical path | 15.4 ns | 13.5 ns |
| Achieved clock | 62.5 MHz | 70.4 MHz |
| Throughput | ~100 MB/s (II=5) | 563 MB/s |
| LUT/FF | 903/699 | 784/904 |

## HLS Project

Built in `~/verilog-learning/chaos_prng/`. No longer the canonical build target.
