# high-dimensional-chaos-generator-2026-reproducibility

Reproducibility materials for the high-dimensional chaos generator.

**Reviewers:** start with the [Reviewer Guide](REVIEWER_GUIDE.md).

**Local release candidate; public release not authorized. No license has been selected.** Author, third-party and binary-distribution permissions remain to be confirmed. The paper and reference PDFs are not included.

The package covers the integer high-dimensional generator and R1–CS parameter synthesis; finite-precision comparison software; fixed-4D PRNG-371 HLS functional source; software reference and historical Figure/Table/validation evidence. R1–LY is the historical code name for R1–CS. See [software](docs/SOFTWARE.md), [hardware](docs/HARDWARE.md), [data](docs/DATA.md) and [reproducibility](docs/REPRODUCIBILITY.md).

- `software/chaos4d_compare_16bit/` and `software/hd_chaos_project/` retain original package layout; `software/experiments_rev1_alignment/` contains the current Mode II experiment implementation. Core and synthesis entry points are indexed in `docs/SOFTWARE.md`.
- `software/prng371/golden_prng.py` loads adjacent `golden_371.so`; the selected backend is prebuilt Linux x86-64. Rebuilding it from C source is outside the authorized scope.
- `hardware/prng371_hls/` provides original C/HLS top and local headers. Vendor `ap_int.h` and tools are external; post-synthesis RTL, netlists and project caches are not part of this selected release.
- `data/figures/README.md` gives Figure→CSV→columns→script mapping; `data/tables/` supplies exact current table fragments and applicable summaries. `data/prng371_validation/` distinguishes 1000-warmup historical reference and paper's 2000-warmup 1000-word comparison.

Stored data integrity can be checked without recomputing scientific metrics from the repository root using `python3 -B software/stored_data.py validate`. This check is not an independent reproduction of research or a license grant. Historical trajectories, Cooja image rights and original STS environment are incomplete. The included Cooja result evidence uses a distinct `prng371_*` ABI; exact identity to the author-selected Golden backend is not independently established. No new experiment, HLS/Vivado or FPGA run has been performed in packaging.
