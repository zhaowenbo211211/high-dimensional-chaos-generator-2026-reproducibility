# Software scope

- Generic high-dimensional code and parameter synthesis: `software/hd_chaos_project/src/chaos/core/`; historical source names use R1-LY. Generic and paper's digital map have different modular/complement semantics and must not be silently interchanged.
- Legacy comparison/metrics: `software/chaos4d_compare_16bit/`; its original research package layout is preserved.
- Current B/D/E experiment implementation: `software/experiments_rev1_alignment/`; B = Golden Range, D = 4–12 dimension family, E = five methods × 4–12. The old A comparison was retired. Numerical scripts can recompute/write results and were not run for this package.
- Author-selected software PRNG: `software/prng371/golden_prng.py` plus adjacent `golden_371.so`, Linux x86-64 prebuilt; original C source rebuilding is not provided or required in this scope. `hardware/prng371_hls` is HLS source, not this binary. Rights and binary redistribution remain pending.

Preserve local package layouts and read each README before running. Historical environment inventories are not verified lockfiles. Official STS, Cooja and vendor tools are external. The stored-data verifier has been run for integrity only; no scientific metrics were recomputed.
