# Maintainer: zhaowenbo
"""
High-dimensional chaotic map generator for lightweight cryptography.

Core modules:
  - chaos.core: parameter synthesis (R1-LY), base map, enhanced map
  - chaos.analysis: Lyapunov exponents, sensitivity, distribution, orbit diagnostics
  - chaos.visualization: 4D glass-sphere plots, verification figures, sensitivity plots
  - chaos.utils: LaTeX equation generation, NIST binary sequence export
"""

from chaos.core import (
    ComplementaryChaosParams,
    HighDimChaosMap,
    HighDimChaosMapEnhanced,
)
from chaos.analysis import (
    compute_lyapunov_exponents,
    verify_chaos,
    analyze_initial_value_sensitivity_16bit,
    plot_sensitivity_analysis,
    ChaosHistogramAnalyzer,
    ChaosCDFDeviationAnalyzer,
    ChaosReportGenerator,
    OrbitStatsConfig,
    analyze_digital_system,
    make_step_fn_from_map,
    wolf_lyapunov_full,
    wolf_lyapunov_embedding,
)
from chaos.utils import (
    make_F,
    latex_system_like_figure,
    latex_system_like_figure_full,
    latex_from_F,
    generate_binary_sequence,
)
from chaos.visualization import (
    plot_4d_chaos_glass_spheres,
    plot_4d_chaos_simple_spheres,
    run_complete_verification,
    export_all_individual_plots,
)

__all__ = [
    # core
    "ComplementaryChaosParams",
    "HighDimChaosMap",
    "HighDimChaosMapEnhanced",
    # analysis
    "compute_lyapunov_exponents",
    "verify_chaos",
    "analyze_initial_value_sensitivity_16bit",
    "plot_sensitivity_analysis",
    "ChaosHistogramAnalyzer",
    "ChaosCDFDeviationAnalyzer",
    "ChaosReportGenerator",
    "OrbitStatsConfig",
    "analyze_digital_system",
    "make_step_fn_from_map",
    "wolf_lyapunov_full",
    "wolf_lyapunov_embedding",
    # utils
    "make_F",
    "latex_system_like_figure",
    "latex_system_like_figure_full",
    "latex_from_F",
    "generate_binary_sequence",
    # visualization
    "plot_4d_chaos_glass_spheres",
    "plot_4d_chaos_simple_spheres",
    "run_complete_verification",
    "export_all_individual_plots",
]
