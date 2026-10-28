# chaos.visualization — Plotting & Figure Generation

Publication-quality figure generation for chaotic system analysis.

## Modules

### plot_4d.py — 4D Glass-Sphere 3D Rendering

Visualizes 4D chaotic data: first 3 dimensions as spatial coordinates, 4th dimension as color.

```python
from chaos.visualization import plot_4d_chaos_glass_spheres, plot_4d_chaos_simple_spheres

# High quality: true 3D sphere surfaces
fig, ax = plot_4d_chaos_glass_spheres(
    data_4d, title="4D Chaos",
    color_map="plasma", alpha=0.85, sphere_quality=15)

# Fast: scatter-based rendering
fig, ax = plot_4d_chaos_simple_spheres(
    data_4d, title="4D Chaos (scatter)",
    color_map="viridis", sphere_size_factor=1.5)
```

| Function | Rendering | Performance |
|---|---|---|
| `plot_4d_chaos_glass_spheres` | 3D sphere surfaces, semi-transparent panes | Slower, publication quality |
| `plot_4d_chaos_simple_spheres` | Scatter with depth shading | Faster, good for exploration |

### verification.py — Complete Verification Visualization

Generates per-dimension time series + phase portrait EPS figures and a combined PDF overview including Lyapunov spectrum.

```python
from chaos.visualization import run_complete_verification

result = run_complete_verification(n=4, word_size=16)
```

**Output** (in `figures/chaos_verification/n4_w16/`):
- `time_series_x{j}_n4_w16.eps` — time series per dimension
- `phase_x{j}_n4_w16.eps` — $(x_j(t), x_j(t+1))$ phase portrait per dimension
- `lyapunov_spectrum_n4_w16.eps` — Lyapunov exponent bar chart
- `chaos_overview_n4_w16.pdf` — all subplots combined

### sensitivity_plots.py — Sensitivity Analysis Plots

Nine individual publication-quality plots for initial-value sensitivity analysis.

```python
from chaos.visualization import export_all_individual_plots

export_all_individual_plots(
    sensitivity_results,
    output_dir="sensitivity_plots",
    prefix="chaos",
    show_first_n=200)
```

| Plot | Description |
|---|---|
| `01_linear_distance` | Distance evolution (linear scale) |
| `02_semilog_distance` | Distance evolution (log scale) |
| `03_initial_response` | First 50 steps detailed view |
| `04_growth_distribution` | Growth rate histogram |
| `05_cumulative_growth` | $d(t)/d(0)$ cumulative growth |
| `06_post_saturation` | Behavior after saturation |
| `07_growth_evolution` | Per-step growth rate over time |
| `08_distance_distribution` | Distance histogram |
| `09_statistics_panel` | Full statistics summary |

---

**Dependencies**: `matplotlib`, `numpy`, `chaos.core`, `chaos.analysis`

**LaTeX rendering**: Enable with `setup_latex_rendering()` for publication-quality math labels. Falls back to standard rendering if LaTeX is unavailable.

**Documentation constraint**: If you add/remove plots or change output paths, update this README.

**Previous layer**: [chaos.analysis](../analysis/README.md) — the data being visualized.
