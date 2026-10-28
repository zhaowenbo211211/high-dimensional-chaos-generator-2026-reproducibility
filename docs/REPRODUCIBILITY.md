# Reproducibility and limits

This is a local candidate, not evidence that all results were reproduced end-to-end. The stored-data check tests CSV identity, counts and archived task-key integrity without recomputing the metrics. Historical experiment records, FPGA reports and statistical tests remain historical; no large experiment, HLS/Vivado, FPGA, NIST or Cooja run occurred during packaging.

| Item | Current scope |
|---|---|
| R1–CS and fixed-precision source | Provided; distinct generic and current experiment semantics documented. |
| PRNG software | Original wrapper plus Linux x86-64 prebuilt shared library; C-source rebuild unavailable and not required for selected scope. |
| Fixed 4D PRNG HLS | Functional source and local headers provided, vendor tools external; exact paper netlist not recreated. |
| Figure numeric data | B/E CSV and mappings provided; Fig. 1 inset columns supported by historical EPS/PDF and data audit chain, not newly rendered. |
| Cooja figures | Image assets and header withheld for rights; complete image reproduction unavailable. Recorded Cooja `prng371_*` and Golden ctypes interface linkage not established. |
| Golden reference | Legacy 1000-warmup and current 2000-warmup references distinguished from saved FPGA ILA capture. |
| License | Not selected; public upload HOLD. |

`docs/PRNG_SOFTWARE_BACKEND.md`, `docs/PRNG_HARDWARE_SOURCE_MAP.md`, `docs/PRNG_REPRODUCIBILITY_SCOPE.md` preserve the detailed boundary analysis.
