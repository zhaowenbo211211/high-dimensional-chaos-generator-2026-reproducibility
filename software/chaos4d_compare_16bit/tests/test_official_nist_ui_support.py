# Maintainer: zhaowenbo
from pathlib import Path
import subprocess
import sys

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from utils.official_nist_runner import _build_assess_interaction, export_builtin_dataset_to_ascii, list_official_nist_tests


def test_export_builtin_dataset_to_ascii() -> None:
    output_path = Path("/tmp/test_official_nist_prng_best.txt")
    exported = export_builtin_dataset_to_ascii(
        dataset_key="prng_best",
        output_path=output_path,
        sequence_bit_length=1024,
        sequence_count=2,
        warmup_steps=1000,
    )
    assert exported.exists()
    assert exported.stat().st_size == 2048


def test_official_nist_ui_lists_datasets() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "nist_prng" / "run_official_nist_ui.py"),
            "--list-datasets",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    for test_id, test_name in list_official_nist_tests():
        assert f"{test_id:02d}: {test_name}" in completed.stdout


def test_build_assess_interaction_for_partial_binary_selection() -> None:
    interaction = _build_assess_interaction(
        input_path_for_assess="ui_inputs/sample.bin",
        sequence_count=100,
        input_mode="binary",
        selected_tests=[1, 3, 15],
    )
    assert "ui_inputs/sample.bin" in interaction
    assert "\n0\n101000000000001\n0\n100\n1\n" in interaction
