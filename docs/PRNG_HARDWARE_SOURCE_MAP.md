# PRNG-371 hardware source map

Primary author-identified hardware project: `PRNG371_FPGA/`. Selected HLS source with maintainer comments added is retained at `hardware/prng371_hls/`; the separate R2 archive contains additional historical snapshots. Paths below are relative to that hls_src directory.

| File | Actual role |
|---|---|
| `prng_top.cpp` | HLS `prng_top`: six 16-bit state words and delay chain; delayed nonlinear product; state update; Mode I rotate of packed output; two-word history and ROTL–XOR mixer. The rotated words are used for output and not fed back into core state. |
| `prng_top.h` | `<ap_int.h>` dependency, 16-bit initialization inputs, 64-bit output and valid flag, control and default seeds. |
| `chaos_core.h`, `chaos_params.h` | 16-bit word type, 4D/six-word state definition and fixed 371/373/375, 50/48/46, b=111 parameter macros. The legacy `chaos_iterate` declaration/comment is not the top-level implementation. |
| `shift_mul.h` | `ap_uint<16>` shift/add/sub constant multiplier helpers. |
| `tb_prng_371.cpp` | HLS C++ testbench with local static Golden functions; 1000-step warmup. Not the shared-library C source. |

The packed word places x1 in bits 0–15, x2 in 16–31, x3 in 32–47 and x4/s0 in 48–63. The whole 64-bit word rotates by 1 through 15; the mixer XORs it with previous two packed words rotated 17 and 43 bits, with a first-output initialization case. The exact implementation is the source; descriptive historical comments are not a separate hardware proof.

The board project also contains `fpga/prng_wrapper.v`, `fpga/acz702_top.v`, constraints, Vivado `.xpr`, generated HLS Verilog, component/IP metadata and ILA artifacts. This public candidate includes functional HLS source and historical capture evidence, not board RTL, generated RTL, the external Vivado project, vendor SDK/IP caches, bitstream or a freshly executed HLS build. The project's root Makefile runs verification tools; it is not an HLS synthesis build recipe. HLS compilation requires compatible Vitis/Vivado tools and `ap_int.h`/vendor support. The original `.xpr` references generated Verilog, wrapper, top, XDC and ILA resources; it is board/debug context, not automatically the exact source of the paper's separate resource netlists.

## Separate assessments

1. **Functional HLS source: PRESENT for the fixed 4D PRNG datapath**, based on static source and included local header closure. Vendor header `ap_int.h` is external. This is not a runtime 4D–12D hardware implementation.
2. **HLS/RTL build materials: PARTIAL for a self-contained release build.** Source, testbench and selected historical records exist; generated RTL/wrapper are excluded from this candidate; toolchain, full portable HLS project and vendor/IP dependencies are not fully packaged or rebuilt.
3. **Exact regeneration of the paper-specific final netlist: NOT VERIFIED.** Independent netlist boundaries and source-to-final-build lineage require their own evidence; static correspondence and saved reports do not establish bit-for-bit netlist reproduction.

No HLS/Vivado run, board operation, ongoing-throughput study or new experiment was undertaken. The software .so C-source status does not determine HLS functional source availability.
