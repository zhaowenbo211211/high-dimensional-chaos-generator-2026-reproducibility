# chaos.\_compat — Backward Compatibility Wrappers

This directory contains thin wrapper modules that re-export from the new canonical locations. They exist solely so that old code using the original import paths continues to work.

## Contents

| Old path | Now re-exports from |
|---|---|
| `chaos._compat.make_latex.make_F_and_latex` | `chaos.utils.latex` |
| `chaos._compat.make_latex.print_math_based_on_tex` | (standalone, unchanged) |
| `chaos._compat.analysis_dynamics.diagnostics.orbit_diagnostics` | `chaos.analysis.orbit` |
| `chaos._compat.sensitivity.sensitivity_analysis` | `chaos.analysis.sensitivity` + `chaos.visualization.sensitivity_plots` |
| `chaos._compat.sensitivity.sensitivity_individual_plots` | `chaos.visualization.sensitivity_plots` |
| `chaos._compat.stats_hd_cdf.stats_hd_cdf_nd` | `chaos.analysis.distribution` |
| `chaos._compat.stats_hd_cdf.stats_hd_cdf_4d` | `chaos.analysis.distribution` |
| `chaos._compat.plot_4d.plot_4d` | `chaos.visualization.plot_4d` |

## Usage

**Do NOT add new code that imports from `_compat`.** Use the canonical paths instead:

```python
# Correct (new code)
from chaos.utils.latex import make_F
from chaos.analysis.orbit import OrbitStatsConfig

# Only for running legacy unmodified scripts
from chaos._compat.make_latex.make_F_and_latex import make_F
```

## Removal Policy

These wrappers will be kept until all dependent scripts and notebooks have been migrated to the new import paths. No new functionality will be added here.

**Documentation constraint**: When migrating a legacy import to the canonical path, check if the old wrapper can be deleted. If a wrapper directory becomes empty, remove it and update this README.
