# PRNG-371 consistency evidence

| File | Source | Contents | Configuration / role |
|---|---|---|---|
| `golden_ref_371.txt` | Author-identified `PRNG371_FPGA/hls_src/golden_ref_371.txt` | 100 lines of 16 hex digits, one 64-bit word per line; SHA-256 `d6fd8c7dc2752d67e5f2614e4723728bb85628e5f8222f03f753be30e25a320c` | Legacy 1000-output warmup reference, default seed [1000,2000,3000,4000] per existing lineage/HLS setup. Software output, not FPGA capture. |
| `golden_ref_371_warmup2000.txt` | `PRNG371_FPGA/golden_ref_371_warmup2000.txt` | 1000 lines of 16 hex digits; SHA-256 `e21ce13d872f8bf93f0770b4e0c16c3224c7cf035b261673790d51618f53b803` | Current paper comparison reference: seed [1000,2000,3000,4000], discard 2000 outputs, then retain 1000 64-bit words. Software output, not FPGA capture. |

`historical_board/capture/` contains the separately saved ILA CSV; `historical_board/postprocess/` contains extracted output; `historical_board/reports/` preserves the alignment and 1000-word comparison report. The historical report records 1000 matched words and zero bit errors for that selected configuration/window; it does not establish behavior outside that window. The saved capture and report were not regenerated or revalidated during this package update. The separate historical 8180-word project report is not represented as the same experiment.

`scripts/prng371/extract_and_compare_100mhz.py` is original historical source and can write output or invoke other commands; examine its dependencies and work on isolated copies. No HLS rebuild is required merely to distribute these data.

The extraction script defaults to a historical relative comparison path. `scripts/prng371/compare_fpga_capture.py` is supplied; any separately authorized replay must pass its path explicitly via `--compare-script` and write output in an isolated location.

The historical project validation report is copied unchanged for evidence: its relative links to `../fpga/prng_wrapper.v` and `../fpga/ENGINEERING_STATUS.md` refer to files not included beside that archival report.
