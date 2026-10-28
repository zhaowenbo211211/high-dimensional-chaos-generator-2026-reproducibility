# PRNG-371 reproducibility scope

| Material | Status | Boundary |
|---|---|---|
| Fixed 4D HLS functional source | Present | Source and local headers retained; vendor `ap_int.h` required; not freshly compiled. |
| Original software reference | Provided as Python ctypes wrapper plus Linux x86-64 prebuilt .so | Original C source unavailable; author does not require finding it for this release. Library not newly executed, and rebuilding it from source is not claimed. |
| Historical software–FPGA check | Evidence included for a paper-specific 2000-output warmup / 1000-word window | Saved capture, reference and reports are historical. No fresh FPGA validation; do not generalize to other seeds or windows. |
| Earlier 1000-warmup reference and external FPGA project report | Historical context | The external project's reported 8180-word comparison has no raw ILA CSV in that project. Do not conflate it with the paper's 2000-warmup comparison or with an offline test summary. |
| Cooja | Source/configuration and some metrics present | The archived programs use a `prng371_*` C interface and separate `libprng371.so` offline; link/equivalence to the author-described Golden backend is not established merely from API names. Image/header/payload rights pending. |
| HLS/RTL/Vivado rebuild | Partial materials, not rerun | Vendor tools and IP dependencies, source-to-final-netlist identity and exact bit-for-bit final netlist regeneration are not established. |

No numerical, HLS/Vivado, FPGA, NIST or Cooja experiment is repeated in this update. Estimated maximum throughput remains the paper's reported convention; sustained-output behavior was not investigated. Source ownership, third-party rights, binary redistribution, license and publication timing require author decisions; public GitHub release remains HOLD.
