# Fixed-4D PRNG-371 HLS functional source

The `.cpp` and `.h` files are copied from `PRNG371_FPGA/hls_src/` with a maintainer comment added and original HLS pragmas intact. `prng_top.cpp` implements the 4D core, Mode I rotation, 64-bit packing and two-history ROTL–XOR mixer. `prng_top.h`, `chaos_core.h`, `chaos_params.h` and `shift_mul.h` provide local interfaces/types/constants; `tb_prng_371.cpp` is a 1000-warmup testbench. The vendor `ap_int.h` and HLS toolchain are external. No HLS build, final netlist regeneration or hardware test was conducted here.

Source/binary redistribution rights are pending; no license is assigned. See `../../docs/PRNG_HARDWARE_SOURCE_MAP.md`.
