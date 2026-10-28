# PRNG-371 HLS source candidate

Rights and redistribution permissions for all files are pending confirmation; no license is assigned here. `prng371_hls/` contains the top-level `prng_top.cpp` and header, all local headers referenced by the source, and the historical testbench. Each is copied from the author-identified `hls_src` with a maintainer comment added; functional source and HLS pragmas are unchanged. `SOURCE_MANIFEST.json` records each file's exact source and both SHA-256 values. Quoted local include closure was checked statically.

The source uses vendor `ap_int.h`, which is not included and requires a compatible HLS toolchain. Generated RTL, netlists, bitstreams, and vendor projects are excluded. No synthesis, simulation, binary execution, or experiment was performed. Saved source and historical comments alone do not establish hardware performance or equivalence.
