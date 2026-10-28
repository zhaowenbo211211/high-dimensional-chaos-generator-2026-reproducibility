# hw/src_371/ — A_odd=371 Variant (Top Performer)

Same 3-stage pipeline architecture as `hw/src/hls_src/`, different R1-LY parameters.

| Parameter | 271 (fast) | 371 |
|---|---|---|
| A_odd | 271 | **371** |
| a_j1 | [271, 273, 275] | **[371, 373, 375]** |
| a_jn | [50, 48, 46] | [50, 48, 46] |
| b | 81 | **111** |
| Δ | 45882 | **99917** |
| Clock | 70.4 MHz | **100 MHz** |
| Throughput | 563 MB/s | **800 MB/s** |
| LUT / FF / DSP | 784 / 904 / 1 | **1042 / 707 / 1** |
| Power | 0.11 W | **0.13 W** |
| Status | Verified | **Verified (top performer)** |

## Files

| File | Description |
|---|---|
| `prng_top.cpp` | 3-stage pipeline HLS source |
| `prng_top.h` | Interface + register map |
| `chaos_core.h` | Type definitions |
| `shift_mul.h` | Auto-generated shift-and-add (pure addition) |
| `chaos_params.h` | #define A_J1_0=371, B_VAL=111... |
| `tb_prng_371.cpp` | C testbench with golden model |
| `golden_371.so` | **C shared library — 100% HLS bit-exact** |
| `golden_prng.py` | Python ctypes wrapper for golden_371.so |
| `golden_ref_371.txt` | HLS CSim reference data (100 words) |

## HLS Project

Built in `~/verilog-learning/chaos_prng_371/`.

```bash
cp hw/src_371/*.cpp hw/src_371/*.h ~/verilog-learning/chaos_prng_371/src/
cp hw/src_371/tb_prng_371.cpp ~/verilog-learning/chaos_prng_371/testbench/
vivado-docker-bash
cd /workspace/chaos_prng_371
make csim && make csynth && make ip && make impl
```

## Software PRNG (100% HLS Bit-Exact)

```python
from hw.src_371.golden_prng import ChaosPRNG371

prng = ChaosPRNG371()                    # A_odd=371, warmup=1000
w = prng.next()                          # 0x515c697507ec42e9 — matches HLS
words = prng.generate(1000)              # 1000 words
data = prng.generate_bytes(1024)         # 1024 bytes

# Verified: 100/100 match with HLS golden_ref_371.txt
```

Under the hood: `golden_prng.py` → ctypes → `golden_371.so` (C shared lib) → same algorithm as HLS `prng_top`.

## Export Script

```bash
python tests/prng_export_371.py
# GUI: set init values + length → generates HLS-verified binary sequence
```

## Why C Shared Library?

The Python `ChaosPRNG` class uses native ints with `& 0xFFFF` masking, which diverges from `ap_uint<16>` subtraction semantics after ~1000 iterations. The C shared library eliminates this by using `uint16_t` with pure-addition decompositions — proven 100/100 match against HLS reference data.
