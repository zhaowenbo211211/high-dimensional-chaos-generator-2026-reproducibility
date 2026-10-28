# Hardware scope

`hardware/prng371_hls/` contains original `prng_top.cpp`, `prng_top.h`, `chaos_core.h`, `chaos_params.h`, `shift_mul.h`, and `tb_prng_371.cpp` from the author-identified PRNG-371 FPGA project with maintainer comments added; functional code and HLS pragmas are unchanged. Includes use vendor `ap_int.h`; Vitis HLS or equivalent vendor tools/headers are not included. Existing pipeline and other HLS pragmas are unchanged.

The top implements the 16-bit fixed 4D core, Mode I output rotation, 64-bit packing and two-history ROTL17/43 XOR mixer. HLS functional source is provided; the full vendor HLS/Vivado build, generated RTL/netlists, DCP/bitstreams, proprietary IP and the paper's exact final circuit are outside this public candidate. No fresh HLS compile, FPGA run or throughput investigation was performed. See `PRNG_HARDWARE_SOURCE_MAP.md` and `PRNG_REPRODUCIBILITY_SCOPE.md`.
