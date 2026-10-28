# Step 1 Warmup Alignment Trace — 100 MHz ACZ702 Clean Rerun

**Capture file**: `prng_experiment/fpga_physical_validation/step1_clean_rerun_acz702_100mhz/capture/ila_capture_100mhz_clean_rerun.csv`  
**Capture SHA-256**: `a8a702832075e9a545771746c7fc13aa618e57d7d1d837b97d486f8c11663c42`  
**Total data rows**: 16384  
**PRNG output column**: `ila_probe2_diag[63:0]`  
**cfg_done column**: `cfg_done`  
**dbg_state column**: `dbg_state[4:0]`  
**ILA clock**: 100 MHz (same as internal PRNG clock)  
**Golden reference**: `prng_experiment/randomness/prng371_warmup2000/reference/golden_ref_371_warmup2000.txt`  
**Golden SHA-256**: `e21ce13d872f8bf93f0770b4e0c16c3224c7cf035b261673790d51618f53b803`

---

## 1. Fixed warmup alignment rule

The post-warmup FPGA output stream is obtained by the following fixed rule (no skip-parameter search):

1. Locate the `cfg_done` rising edge in the raw ILA CSV.
2. Deduplicate consecutive identical values in `ila_probe2_diag[63:0]`.
3. Discard the first unique value if it equals `0x0000000000000000`.
4. Skip the next 2000 valid PRNG outputs (the documented warmup interval).
5. The remaining sequence is the post-warmup stream used for golden comparison.

---

## 2. cfg_done rising edge

`cfg_done` is defined as `assign cfg_done = (cfg_state >= 5'd24);`.

- `cfg_done` rising edge at **buffer sample 8215**.
- At sample 8215: `dbg_state = 0x18`, `cfg_done = 1`, `ila_probe2_diag = 0x0000000000000000`.

---

## 3. First unique value after cfg_done

The first unique value after `cfg_done` is `0x0000000000000000`.

**Why it is not a valid PRNG output**: `ila_probe2_diag` is driven by
`assign ila_probe2_diag = (cfg_state < 5'd24) ? {axi_wdata, axi_awaddr, 26'd0} : prng_out;`

At the exact cycle when `cfg_done` first asserts (`cfg_state` transitions from below 24 to 24 or above), the PRNG pipeline has not yet produced its first valid output. The bus carries a transient `0x0000000000000000` for one cycle before the first valid `prng_out` appears. This is confirmed by the fact that the next unique value, `0xc661f290c4215270`, matches the C golden model with `warmup=0`.

---

## 4. First valid PRNG output after discarding leading zero

After discarding the leading zero:

| Index | FPGA output | C golden (warmup=0) | Match |
|---|---|---|---|
| 0 | `c661f290c4215270` | `c661f290c4215270` | ✅ |
| 1 | `7fa2642199a17c60` | `7fa2642199a17c60` | ✅ |
| 2 | `adf7fbf7aa526ae0` | `adf7fbf7aa526ae0` | ✅ |
| 3 | `1a888edad1ecfe26` | `1a888edad1ecfe26` | ✅ |
| 4 | `7d0cd5f6db306630` | `7d0cd5f6db306630` | ✅ |

---

## 5. Warmup=2000 skip interval

- Raw unique outputs after `cfg_done`: 4085 (including leading zero).
- After discarding leading zero: 4084 valid outputs.
- Skip outputs 0 through 1999 as warmup.
- First post-warmup output is at raw index **2000** (after leading-zero removal).

---

## 6. Post-warmup alignment verification

| Post-warmup index | FPGA output | Golden output | Match |
|---|---|---|---|
| 0 | `51830cf281b72b1e` | `51830cf281b72b1e` | ✅ |
| 1 | `a731bb65e3f384d3` | `a731bb65e3f384d3` | ✅ |
| 2 | `a95abbe369bd44b5` | `a95abbe369bd44b5` | ✅ |
| 3 | `3d82a70a0ae12d08` | `3d82a70a0ae12d08` | ✅ |
| 4 | `69b3bb741b4d1c00` | `69b3bb741b4d1c00` | ✅ |

Total post-warmup outputs available: 2084.  
First 1000 compared against golden: all match.

---

## 7. No skip-parameter search

The alignment offset is determined entirely by RTL behavior (`cfg_done` assertion, one-cycle output latency, and the documented `warmup=2000` parameter). No arbitrary skip value was searched to obtain PASS.

---

## 8. Reproducibility

```bash
cd __REMOTE_PRNG_PROJECT__
__AUTHOR_HOME__/.local/share/uv/python/cpython-3.13.13-linux-x86_64-gnu/bin/python3.13 \
  prng_experiment/fpga_physical_validation/step1_clean_rerun_acz702_100mhz/tcl/extract_and_compare_100mhz.py \
  --capture prng_experiment/fpga_physical_validation/step1_clean_rerun_acz702_100mhz/capture/ila_capture_100mhz_clean_rerun.csv \
  --golden prng_experiment/randomness/prng371_warmup2000/reference/golden_ref_371_warmup2000.txt \
  --warmup 2000 \
  --output-txt prng_experiment/fpga_physical_validation/step1_clean_rerun_acz702_100mhz/postprocess/fpga_post_warmup_100mhz_clean_rerun.txt \
  --output-report prng_experiment/fpga_physical_validation/step1_clean_rerun_acz702_100mhz/reports/fpga_vs_golden_compare_report_100mhz.md
```
