# Scripts

Standalone analysis and comparison scripts.

## compare_period_stats.py

Compares period length and statistical distribution between the base map and the bit-rotation enhanced map.

```bash
PYTHONPATH="src:$PYTHONPATH" python scripts/compare_period_stats.py
```

**Output** (in `results/period_stats/`):
- `base_single/` — per-dimension EPS: time series + phase portraits for base map
- `enhanced_single/` — per-dimension EPS: time series + phase portraits for enhanced map
- `base_ts_phase.pdf`, `base_hist.pdf` — base map overview PDFs
- `enhanced_ts_phase.pdf`, `enhanced_hist.pdf` — enhanced map overview PDFs

**Configuration**: Edit `A_odd`, `B_even`, `n`, `word_size`, `max_steps` in the `main()` function.

**Documentation constraint**: If you change output paths, figure formats, or add new comparison scripts, update this README.
