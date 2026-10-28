# chaos.analysis — Analysis & Verification

Quantitative analysis tools for chaotic map evaluation.

## Modules

### lyapunov.py — Lyapunov Spectrum & Chaos Verification

QR-decomposition method for the full Lyapunov exponent spectrum.

```python
from chaos.analysis import compute_lyapunov_exponents, verify_chaos

les = compute_lyapunov_exponents(map_instance, steps=2000, transient=500)
result = verify_chaos(map_instance, steps=3000)
# result: {'lyap_exps', 'is_chaotic', 'variance', 'min_le', 'max_le', 'params'}
```

| Function | Description |
|---|---|
| `compute_lyapunov_exponents(map, steps, transient, ortho_interval)` | Full LE spectrum via Jacobian-QR |
| `verify_chaos(map, steps)` | LE > 0.01 + variance > 0.05 → chaotic |
| `_jacobian_normalized(x, params, n)` | Jacobian in normalized space |

### wolf.py — Wolf-Style Maximum LE Estimation

Two black-box max LE estimators suitable for discrete digital maps (no Jacobian required).

```python
from chaos.analysis import wolf_lyapunov_full, wolf_lyapunov_embedding

r = wolf_lyapunov_full(map, max_steps=5000, transient=500)
# Returns: {'lyap_max', 'lyap_mean', 'lyap_std', 'is_chaotic', 'renormalizations', ...}

r = wolf_lyapunov_embedding(map, max_steps=5000, embed_dim=4, delay=1)
# Returns: {'lyap_max', 'embed_dim', 'delay', ...}
```

| Function | Method | Best for |
|---|---|---|
| `wolf_lyapunov_full` | Full-state, 1-LSB renormalization | Digital map with known state space |
| `wolf_lyapunov_embedding` | Delay embedding, Theiler window | Black-box with only output time series |

### sensitivity.py — Initial-Value Sensitivity

Butterfly effect analysis: how fast do nearby trajectories diverge?

```python
from chaos.analysis import analyze_initial_value_sensitivity_16bit
from chaos.core import HighDimChaosMap

result = analyze_initial_value_sensitivity_16bit(
    params, HighDimChaosMap, min_iterations=200, max_iterations=5000)
# Returns: distances, growth_rates, saturation steps, trajectory data
```

### distribution.py — Statistical Distribution

Marginal histogram, empirical CDF deviation from $U[0,1)$, and LaTeX report generation.

```python
from chaos.analysis import (
    ChaosHistogramAnalyzer, ChaosCDFDeviationAnalyzer, ChaosReportGenerator)

hist = ChaosHistogramAnalyzer(data, mod=65536, dims_to_plot=[0, 1, 2, 3])
hist.plot_overall_histogram(out_dir, bins=128)

cdf = ChaosCDFDeviationAnalyzer(data, mod=65536)
cdf.plot_overall_cdf_deviation(out_dir)

report = ChaosReportGenerator(data, mod, cdf, warmup=5000, total_samples=100000)
report.save_report(output_path)
```

### orbit.py — Orbit Diagnostics

Floyd's tortoise-and-hare cycle detection, orbit sampling, and unified digital system analysis.

```python
from chaos.analysis import OrbitStatsConfig, analyze_digital_system, make_step_fn_from_map

step_fn = make_step_fn_from_map(map_instance)
cfg = OrbitStatsConfig(n_dims=4, word_size=16, max_steps=100000)
result = analyze_digital_system(step_fn, x0, cfg)
# Returns: period, samples, per-dimension stats, histograms
```

---

**Dependencies**: `chaos.core`, `numpy`, `matplotlib`

**Documentation constraint**: If you add a new analysis function or change a return signature, update this README.

**Previous layer**: [chaos.core](../core/README.md) — the maps being analyzed.
**Next layer**: [chaos.visualization](../visualization/README.md) — figure generation for these analyses.
