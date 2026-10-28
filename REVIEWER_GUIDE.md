# Reviewer guide

Project identifier: `high-dimensional-chaos-generator-2026-reproducibility`.

This local submission candidate accompanies the manuscript *A High-Dimensional Nonlinear Chaos Generator with Integer Coefficients for Lightweight Cryptography*. The paper itself is not included. Section, algorithm and object references below follow manuscript source SHA-256 `5927a817097c448a00838763184a2519e1b40c6caa95f6f9408148286cba2b8e`.

| Manuscript location | Material and entry point |
|---|---|
| §2.1–2.2: delayed integer model and expansion criterion | Generic implementation in [`software/hd_chaos_project/src/chaos/core/`](software/hd_chaos_project/src/chaos/core/); the paper's finite-precision experiment implementation is [`proposed_rev1.py`](software/experiments_rev1_alignment/src/chaos_maps/proposed_rev1.py). The two digital implementations have different quantization semantics; see [software scope](docs/SOFTWARE.md). |
| §2.3, Algorithm R1–CS, Table 1 | [`r1ly_rev1.py`](software/experiments_rev1_alignment/src/r1ly_rev1.py), [`params.py`](software/hd_chaos_project/src/chaos/core/params.py), and [`tab_r1cs-recommended-parameters.tex`](data/tables/tab_r1cs-recommended-parameters.tex). Historical source identifiers use R1–LY. The table is analytical; the archived A=371 configuration is not its direct export. |
| §2.4, bit-rotation algorithm | [`enhanced_map.py`](software/hd_chaos_project/src/chaos/core/enhanced_map.py), current [Mode II implementation](software/experiments_rev1_alignment/src/chaos_maps/proposed_rev1.py), and [fixed-4D HLS source](hardware/prng371_hls/prng_top.cpp) for Mode I output rotation. |
| §3.2–3.4, Figures 1–4 and Table 2 | [Figure-to-CSV and column map](data/figures/README.md); [archived B/D/E records](data/experiments_rev1_alignment/results/); [metric code](software/experiments_rev1_alignment/src/metrics/). Figure 1 inset mapping uses an earlier hash/plotting audit, not a new render. |
| §3.5, Figure 5 and Tables 3–4 | [Table extracts and scope](data/tables/README.md). The runtime-configurable core is conceptual/analytical; this package does not claim a built 4D–12D switching FPGA. |
| §4.1–4.3, Figure 6 and Tables 5–7 | [PRNG-371 HLS source](hardware/prng371_hls/) and [hardware source map](docs/PRNG_HARDWARE_SOURCE_MAP.md). The selected source is the fixed 4D functional datapath; vendor tools and the paper's exact final FPGA netlists are not included. |
| §4.4, software–hardware consistency | [Golden references, saved ILA capture and comparison reports](data/prng371_validation/README.md); [software reference](software/prng371/). Historical 1000-warmup and paper 2000-warmup configurations are distinguished. The precompiled `.so` requires Linux x86-64; it was not newly executed here. |
| §5, Figures 7–8 and Table 8 | [Data and table scope](docs/DATA.md), [Cooja source/config](data/cooja/cooja_sim/), and [NIST table summaries](data/tables/nist/). Cooja image pixels and payloads are absent pending rights; Cooja's stored C ABI and the Golden shared-library ABI are distinct and their exact linkage is not independently established. |

The archival data check below reads stored CSV/task records and checks their hashes and identities; it does not recompute scientific metrics:

```bash
python3 -B software/stored_data.py validate
```

Saved FPGA, NIST and Cooja results are historical evidence. No new large numerical experiment, HLS/Vivado, board run, NIST or Cooja experiment was conducted for this package. Read [reproducibility limits](docs/REPRODUCIBILITY.md), [known issues](docs/KNOWN_ISSUES.md), and [license status](LICENSE_STATUS.md) before use.
