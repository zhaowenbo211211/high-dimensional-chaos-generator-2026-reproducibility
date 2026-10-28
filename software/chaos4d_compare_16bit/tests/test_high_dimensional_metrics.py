# Maintainer: zhaowenbo
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import ProposedNDEnhancedMap
from metrics.fuzzy_entropy import fuzzy_entropy
from metrics.lyapunov import wolf_lyapunov_full_state_nd
from metrics.nonperiodicity import scale_index_cwt
from metrics.permutation_entropy import permutation_entropy
from metrics.spectral_entropy import spectral_entropy_complexity


def test_high_dimensional_metrics_support_6d_trajectory() -> None:
    se_trajectory = ProposedNDEnhancedMap(n=6).iterate(5000, discard_transient=32)
    short_trajectory = ProposedNDEnhancedMap(n=6).iterate(512, discard_transient=32)

    se_result = spectral_entropy_complexity(se_trajectory)
    fe_result = fuzzy_entropy(short_trajectory)
    si_result = scale_index_cwt(short_trajectory[:256])
    pe_result = permutation_entropy(short_trajectory)

    assert len(se_result["se_components"]) == 6
    assert len(fe_result["fe_components"]) == 6
    assert len(si_result["si_components"]) == 6
    assert len(pe_result["pe_components"]) == 6
    assert np.isfinite(se_result["se_mean"])
    assert np.isfinite(fe_result["fe_mean"])
    assert np.isfinite(si_result["si_mean"])
    assert np.isfinite(pe_result["pe_mean"])


def test_high_dimensional_lyapunov_supports_8d_map() -> None:
    result = wolf_lyapunov_full_state_nd(ProposedNDEnhancedMap(n=8), max_steps=128, transient=32)
    assert np.isfinite(result["lyap_max"])
    assert len(result["lyap_components"]) == 8


def test_dimension_scalability_cli_generates_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = root / "temp" / "tmp_test_high_dimensional"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_dimension_scalability.py"),
            "--dimensions",
            "4",
            "5",
            "--se-length",
            "5000",
            "--fe-length",
            "512",
            "--pe-length",
            "512",
            "--si-length",
            "512",
            "--le-steps",
            "128",
            "--se-transient",
            "16",
            "--fe-transient",
            "16",
            "--pe-transient",
            "16",
            "--si-transient",
            "16",
            "--le-transient",
            "16",
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert (output_root / "raw" / "r1_ly_dimension_params.csv").exists()
    assert (output_root / "raw" / "dimension_metrics_summary.csv").exists()
    assert (output_root / "raw" / "dimension_metrics_detailed.csv").exists()
    assert (output_root / "tables" / "dimension_metrics_summary.tex").exists()
    assert (output_root / "figures" / "dimension_metrics_summary.pdf").exists()
    assert (output_root / "figures" / "dimension_metrics_summary.eps").exists()
