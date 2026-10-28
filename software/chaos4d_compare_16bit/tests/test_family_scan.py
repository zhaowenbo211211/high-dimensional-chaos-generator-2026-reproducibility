# Maintainer: zhaowenbo
from pathlib import Path
import sys

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from experiments.plot_golden_range_le import (
    golden_range_a11_values,
    plot_proposed_golden_range_le,
    scan_proposed_golden_range_le,
)


def test_golden_range_a11_values_are_odd_and_bounded() -> None:
    values = golden_range_a11_values()
    assert values[0] == 271
    assert values[-1] == 507
    assert all(value % 2 == 1 for value in values)


def test_scan_proposed_golden_range_le_small_window() -> None:
    records = scan_proposed_golden_range_le(
        variant="proposed_base",
        a11_start=271,
        a11_stop=275,
        a11_step=2,
        transient=8,
        max_steps=32,
        progress=False,
    )
    assert len(records) == 3
    assert [record["group_index"] for record in records] == [1, 2, 3]
    assert [record["a11"] for record in records] == [271, 273, 275]
    assert all(record["variant"] == "proposed_base" for record in records)
    assert all("lyap_max" in record for record in records)
    assert all(len(record["lyap_components"]) == 4 for record in records)
    assert any(np.isfinite(float(record["lyap_max"])) for record in records)


def test_plot_proposed_golden_range_le_outputs_files(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return

    records = scan_proposed_golden_range_le(
        variant="proposed_base",
        a11_start=271,
        a11_stop=273,
        a11_step=2,
        transient=8,
        max_steps=32,
        progress=False,
    )
    paths = plot_proposed_golden_range_le(records, output_dir=tmp_path)
    assert paths["pdf"].exists()
    assert paths["png"].exists()
    assert paths["pdf"].name == "proposed_golden_range_le.pdf"
    assert paths["png"].name == "proposed_golden_range_le.png"


def main() -> None:
    tests = [
        ("golden_range_a11_values_are_odd_and_bounded", test_golden_range_a11_values_are_odd_and_bounded),
        ("scan_proposed_golden_range_le_small_window", test_scan_proposed_golden_range_le_small_window),
        ("plot_proposed_golden_range_le_outputs_files", lambda: test_plot_proposed_golden_range_le_outputs_files(Path("temp/tmp_test_figures"))),
    ]
    print("运行 test_family_scan.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_family_scan.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
