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

from chaos_maps import NDCS4DMap, Proposed4DBaseMap, Proposed4DEnhancedMap
from metrics.fuzzy_entropy import fuzzy_entropy_smoke
from metrics.lyapunov import estimate_lyapunov_smoke, estimate_max_lyapunov_wolf_full_state
from metrics.nonperiodicity import nonperiodicity_smoke
from metrics.permutation_entropy import permutation_entropy_smoke
from metrics.sample_entropy import sample_entropy_smoke
from metrics.spectral_entropy import (
    DEFAULT_SE_RAW_LENGTH,
    MIN_SE_RAW_LENGTH,
    spectral_entropy_complexity,
    spectral_entropy_smoke,
)


def test_metric_smoke_interfaces() -> None:
    traj = Proposed4DBaseMap().iterate(DEFAULT_SE_RAW_LENGTH)

    outputs = [
        estimate_lyapunov_smoke(traj),
        spectral_entropy_smoke(traj),
        sample_entropy_smoke(traj),
        fuzzy_entropy_smoke(traj),
        nonperiodicity_smoke(traj),
        permutation_entropy_smoke(traj),
    ]

    for out in outputs:
        assert isinstance(out, np.ndarray)
        assert out.shape == (4,)
        assert np.all(np.isfinite(out))


def test_wolf_lyapunov_full_state_smoke() -> None:
    results = [
        estimate_max_lyapunov_wolf_full_state(Proposed4DBaseMap(), max_steps=128, transient=16),
        estimate_max_lyapunov_wolf_full_state(Proposed4DEnhancedMap(), max_steps=128, transient=16),
    ]

    required_fields = {
        "lyap_max",
        "lyap_mean",
        "lyap_std",
        "lyap_components",
        "renormalizations",
        "final_distance",
        "is_chaotic",
        "growth_rates",
        "distances",
    }

    for result in results:
        assert required_fields.issubset(result.keys())
        assert np.isfinite(result["lyap_max"])
        assert np.isfinite(result["lyap_mean"])
        assert np.isfinite(result["lyap_std"])
        assert len(result["lyap_components"]) == 4
        assert isinstance(result["growth_rates"], list)
        assert isinstance(result["distances"], list)
        assert len(result["distances"]) > 0


def test_wolf_lyapunov_default_length_runs_on_ndcs() -> None:
    result = estimate_max_lyapunov_wolf_full_state(NDCS4DMap())
    assert np.isfinite(result["lyap_max"])
    assert len(result["lyap_components"]) == 4
    assert result["growth_rates"] or result["distances"]
    assert np.isfinite(result["final_distance"])


def test_run_cross_compare_outputs_le_table() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "experiments" / "run_cross_compare.py"), "--metric", "le", "--steps", "64", "--discard", "16"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "LE Max" in completed.stdout
    assert "Proposed4DBaseMap" in completed.stdout
    assert "NDCS4DMap" in completed.stdout


def test_spectral_entropy_complexity_smoke() -> None:
    trajectory = Proposed4DEnhancedMap().iterate(DEFAULT_SE_RAW_LENGTH, discard_transient=16)
    result = spectral_entropy_complexity(trajectory)
    assert len(result["se_components"]) == 4
    assert np.isfinite(result["se_mean"])
    assert result["raw_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["test_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["mode"] == "direct"


def test_default_se_length_is_5000() -> None:
    assert DEFAULT_SE_RAW_LENGTH == 5000
    assert MIN_SE_RAW_LENGTH == 5000


def test_run_cross_compare_outputs_se_table() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_se_compare"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_cross_compare.py"),
            "--metric",
            "se",
            "--raw-output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "SE Mean" in completed.stdout
    assert "Proposed4DEnhancedMap" in completed.stdout
    assert "Saved CSV:" in completed.stdout
    assert (output_dir / "se_comparison.csv").exists()


def test_run_cross_compare_outputs_fe_table() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_fe_compare"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_cross_compare.py"),
            "--metric",
            "fe",
            "--steps",
            "128",
            "--discard",
            "16",
            "--raw-output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "FE Mean" in completed.stdout
    assert "Proposed4DEnhancedMap" in completed.stdout
    assert "Saved CSV:" in completed.stdout
    assert (output_dir / "fe_comparison.csv").exists()


def test_run_family_scan_outputs_se_results() -> None:
    root = Path(__file__).resolve().parents[1]
    pdf_path = root / "results" / "figures" / "enhanced_golden_range_se_components.pdf"
    eps_path = root / "results" / "figures" / "enhanced_golden_range_se_components.eps"
    csv_path = root / "results" / "raw" / "enhanced_golden_range_se.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_family_scan.py"),
            "--map",
            "proposed_enhanced",
            "--metric",
            "se",
            "--a11-start",
            "271",
            "--a11-stop",
            "275",
            "--a11-step",
            "2",
            "--transient",
            "8",
            "--steps",
            "64",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert ">= 5000" in completed.stderr or ">= 5000" in completed.stdout


def test_run_family_scan_outputs_fe_results() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_fe_family"
    figure_dir = root / "temp" / "tmp_test_fe_figure"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_family_scan.py"),
            "--map",
            "proposed_enhanced",
            "--metric",
            "fe",
            "--a11-start",
            "271",
            "--a11-stop",
            "275",
            "--a11-step",
            "2",
            "--transient",
            "8",
            "--steps",
            "128",
            "--raw-output-dir",
            str(output_dir),
            "--figure-output-dir",
            str(figure_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Completed" in completed.stdout
    assert (output_dir / "enhanced_golden_range_fe.csv").exists()
    assert (figure_dir / "enhanced_golden_range_fe.pdf").exists()
    assert (figure_dir / "enhanced_golden_range_fe.eps").exists()


def test_run_cross_compare_outputs_si_table() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_si_compare"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_cross_compare.py"),
            "--metric",
            "si",
            "--steps",
            "128",
            "--discard",
            "16",
            "--raw-output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "SI Mean" in completed.stdout
    assert "Proposed4DEnhancedMap" in completed.stdout
    assert "Saved CSV:" in completed.stdout
    assert (output_dir / "scale_index_comparison.csv").exists()


def test_run_cross_compare_outputs_pe_table() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_pe_compare"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_cross_compare.py"),
            "--metric",
            "pe",
            "--steps",
            "128",
            "--discard",
            "16",
            "--raw-output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "PE Mean" in completed.stdout
    assert "Proposed4DEnhancedMap" in completed.stdout
    assert "Saved CSV:" in completed.stdout
    assert (output_dir / "pe_comparison.csv").exists()


def test_run_family_scan_outputs_si_results() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_si_family"
    figure_dir = root / "temp" / "tmp_test_si_figure"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_family_scan.py"),
            "--map",
            "proposed_enhanced",
            "--metric",
            "si",
            "--a11-start",
            "271",
            "--a11-stop",
            "275",
            "--a11-step",
            "2",
            "--transient",
            "8",
            "--steps",
            "128",
            "--raw-output-dir",
            str(output_dir),
            "--figure-output-dir",
            str(figure_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Completed" in completed.stdout
    assert (output_dir / "enhanced_golden_range_scale_index.csv").exists()
    assert (figure_dir / "enhanced_golden_range_scale_index.pdf").exists()
    assert (figure_dir / "enhanced_golden_range_scale_index.eps").exists()


def test_run_family_scan_outputs_pe_results() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_pe_family"
    figure_dir = root / "temp" / "tmp_test_pe_figure"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "run_family_scan.py"),
            "--map",
            "proposed_enhanced",
            "--metric",
            "pe",
            "--a11-start",
            "271",
            "--a11-stop",
            "275",
            "--a11-step",
            "2",
            "--transient",
            "8",
            "--steps",
            "128",
            "--raw-output-dir",
            str(output_dir),
            "--figure-output-dir",
            str(figure_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Completed" in completed.stdout
    assert (output_dir / "enhanced_golden_range_pe.csv").exists()
    assert (figure_dir / "enhanced_golden_range_pe.pdf").exists()
    assert (figure_dir / "enhanced_golden_range_pe.eps").exists()


def test_plot_si_diagnostics_outputs_files() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "temp" / "tmp_test_si_diag_raw"
    figure_dir = root / "temp" / "tmp_test_si_diag_fig"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "plot_si_diagnostics.py"),
            "--steps",
            "512",
            "--discard",
            "32",
            "--raw-output-dir",
            str(output_dir),
            "--figure-output-dir",
            str(figure_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Scalogram overview of the enhanced map." in completed.stdout
    assert "Inner scalogram curve for x1." in completed.stdout
    assert (output_dir / "enhanced_scale_index_diagnostics.json").exists()
    assert (figure_dir / "enhanced_scale_index_scalogram.pdf").exists()
    assert (figure_dir / "enhanced_scale_index_scalogram.eps").exists()
    assert (figure_dir / "enhanced_scale_index_si_x1.pdf").exists()
    assert (figure_dir / "enhanced_scale_index_si_x1.eps").exists()
    assert (figure_dir / "enhanced_scale_index_si_x2.pdf").exists()
    assert (figure_dir / "enhanced_scale_index_si_x2.eps").exists()
    assert (figure_dir / "enhanced_scale_index_si_x3.pdf").exists()
    assert (figure_dir / "enhanced_scale_index_si_x3.eps").exists()
    assert (figure_dir / "enhanced_scale_index_si_x4.pdf").exists()
    assert (figure_dir / "enhanced_scale_index_si_x4.eps").exists()


def main() -> None:
    tests = [
        ("metric_smoke_interfaces", test_metric_smoke_interfaces),
        ("wolf_lyapunov_full_state_smoke", test_wolf_lyapunov_full_state_smoke),
        ("wolf_lyapunov_default_length_runs_on_ndcs", test_wolf_lyapunov_default_length_runs_on_ndcs),
        ("run_cross_compare_outputs_le_table", test_run_cross_compare_outputs_le_table),
        ("spectral_entropy_complexity_smoke", test_spectral_entropy_complexity_smoke),
        ("default_se_length_is_5000", test_default_se_length_is_5000),
        ("run_cross_compare_outputs_se_table", test_run_cross_compare_outputs_se_table),
        ("run_cross_compare_outputs_fe_table", test_run_cross_compare_outputs_fe_table),
        ("run_family_scan_outputs_se_results", test_run_family_scan_outputs_se_results),
        ("run_family_scan_outputs_fe_results", test_run_family_scan_outputs_fe_results),
        ("run_cross_compare_outputs_si_table", test_run_cross_compare_outputs_si_table),
        ("run_cross_compare_outputs_pe_table", test_run_cross_compare_outputs_pe_table),
        ("run_family_scan_outputs_si_results", test_run_family_scan_outputs_si_results),
        ("run_family_scan_outputs_pe_results", test_run_family_scan_outputs_pe_results),
        ("plot_si_diagnostics_outputs_files", test_plot_si_diagnostics_outputs_files),
    ]
    print("运行 test_metrics_smoke.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_metrics_smoke.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
