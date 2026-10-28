# experiments

This folder contains runnable experiment entry points.

Main groups:

- representative single-map and cross-map metric runs
- golden-range family scans and plotting helpers
- high-dimensional proposed-family evaluation
- official NIST/PRNG entry points under `nist_prng/`

## Core Metric Entry Points

- `run_single_map.py`: run one map trajectory preview or one metric.
- `run_cross_compare.py`: compare representative maps under shared metrics.
- `run_family_scan.py`: scan the proposed enhanced golden range.
- `run_dimension_scalability.py`: evaluate the proposed enhanced family from 4D to 12D.
- `export_tables.py`: export paper-facing tables from generated results.

## Plotting Helpers

- `plot_golden_range_le.py`
- `plot_golden_range_se.py`
- `plot_golden_range_fe.py`
- `plot_golden_range_pe.py`
- `plot_golden_range_si.py`
- `plot_dimension_scalability.py`
- `plot_si_diagnostics.py`

## NIST/PRNG Entry Points

NIST and PRNG scripts now live in `experiments/nist_prng/`. Only the final
system-level PRNG block rule is retained there:

```text
u_t = x1(t) || x2(t) || x3(t) || x4(t)
y_t = u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43)
```

See `experiments/nist_prng/README.md` for the runnable commands.
