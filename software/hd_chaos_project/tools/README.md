# Tools

Utility scripts for the Chaos PRNG project.

## analyze_reports.py

Parses HLS synthesis + Vivado implementation reports and prints a unified summary table with PASS/FAIL verdict.

```bash
python tools/analyze_reports.py

# Specify custom paths
python tools/analyze_reports.py --hls-dir ~/verilog-learning/chaos_prng

# Output example:
#   Performance:  clock, latency, interval, pipeline, data path
#   Timing:       WNS, TNS, failing endpoints, PASS/FAIL
#   Resource:     LUT/FF/DSP/BRAM (Vivado vs HLS estimate)
#   Power:        total, dynamic, static, junction temp
#   Verdict:      green PASS or red FAIL with suggested clock
```

The script scans:
- `output/synthesis/report.txt` — HLS synthesis
- `output/implementation/utilization_impl.rpt` — Vivado utilization
- `output/implementation/timing_impl.rpt` — Vivado timing
- `output/implementation/power_impl.rpt` — Vivado power

## Adding New Tools

Place scripts here if they are reusable across the project. Each tool should:
- Have a `--help` flag
- Work from the project root directory
- Not require the Vitis/Vivado environment (pure Python)
