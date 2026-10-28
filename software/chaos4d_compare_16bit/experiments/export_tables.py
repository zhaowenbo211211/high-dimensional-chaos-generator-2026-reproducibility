# Maintainer: zhaowenbo
"""Helpers for exporting experiment tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.io_utils import save_csv_rows, save_json


def export_proposed_golden_range_le_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "proposed_golden_range_le",
) -> dict[str, Path]:
    """Export proposed-family LE scan records to JSON and CSV.

    Complex fields such as ``lyap_components`` are preserved directly in JSON
    and serialized as compact JSON strings in CSV.
    """

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")

    csv_rows: list[dict[str, Any]] = []
    fieldnames = [
        "group_index",
        "a11",
        "a21",
        "a31",
        "b",
        "lyap_max",
        "renormalizations",
        "lyap_components",
        "error",
    ]
    for record in records:
        csv_row = {field: record.get(field) for field in fieldnames}
        csv_row["lyap_components"] = json.dumps(record.get("lyap_components", []), ensure_ascii=False)
        csv_rows.append(csv_row)

    csv_path = save_csv_rows(
        csv_rows,
        output_root / f"{stem}.csv",
        fieldnames=fieldnames,
    )
    return {"json": json_path, "csv": csv_path}


def export_le_comparison_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "le_comparison",
) -> dict[str, Path]:
    """Export cross-map Wolf full-state LE comparison records."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "map_name",
        "lyap_max",
        "lyap_mean",
        "lyap_std",
        "renormalizations",
        "final_distance",
        "is_chaotic",
        "max_steps",
        "transient",
        "initial_state_float",
        "initial_state_u16",
        "fixed_point",
        "a_linear",
        "a_coupling",
        "b",
        "run_command",
    ]
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        csv_row = {field: record.get(field) for field in fieldnames}
        csv_row["initial_state_float"] = json.dumps(record.get("initial_state_float"), ensure_ascii=False)
        csv_row["initial_state_u16"] = json.dumps(record.get("initial_state_u16"), ensure_ascii=False)
        csv_row["a_linear"] = json.dumps(record.get("a_linear"), ensure_ascii=False)
        csv_row["a_coupling"] = json.dumps(record.get("a_coupling"), ensure_ascii=False)
        csv_rows.append(csv_row)
    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_enhanced_golden_range_se_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "enhanced_golden_range_se",
) -> dict[str, Path]:
    """Export enhanced-map golden-range SE records to CSV and JSON."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    csv_rows: list[dict[str, Any]] = []
    fieldnames = [
        "group_index",
        "a11",
        "a21",
        "a31",
        "b",
        "se_x1",
        "se_x2",
        "se_x3",
        "se_x4",
        "se_mean",
        "raw_length",
        "test_length",
        "mode",
        "error",
    ]
    for record in records:
        csv_rows.append({field: record.get(field) for field in fieldnames})

    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_se_comparison_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "se_comparison",
) -> dict[str, Path]:
    """Export cross-map SE comparison records to CSV and JSON."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "map_name",
        "se_x1",
        "se_x2",
        "se_x3",
        "se_x4",
        "se_mean",
        "raw_length",
        "test_length",
        "mode",
    ]
    csv_rows = [{field: record.get(field) for field in fieldnames} for record in records]
    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_enhanced_golden_range_fe_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "enhanced_golden_range_fe",
) -> dict[str, Path]:
    """Export enhanced-map golden-range FE records to CSV and JSON."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    csv_rows: list[dict[str, Any]] = []
    fieldnames = [
        "group_index",
        "a11",
        "a21",
        "a31",
        "b",
        "fe_x1",
        "fe_x2",
        "fe_x3",
        "fe_x4",
        "fe_mean",
        "raw_length",
        "test_length",
        "mode",
        "m",
        "r",
        "tau",
        "n",
        "error",
    ]
    for record in records:
        csv_row = {field: record.get(field) for field in fieldnames}
        csv_row["r"] = json.dumps(record.get("r"), ensure_ascii=False)
        csv_rows.append(csv_row)

    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_fe_comparison_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "fe_comparison",
) -> dict[str, Path]:
    """Export cross-map FE comparison records to CSV and JSON."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "map_name",
        "fe_x1",
        "fe_x2",
        "fe_x3",
        "fe_x4",
        "fe_mean",
        "raw_length",
        "test_length",
        "mode",
        "m",
        "r",
        "tau",
        "n",
        "initial_state_float",
        "initial_state_u16",
        "a_linear",
        "a_coupling",
        "b",
        "run_command",
    ]
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        csv_row = {field: record.get(field) for field in fieldnames}
        csv_row["r"] = json.dumps(record.get("r"), ensure_ascii=False)
        csv_row["initial_state_float"] = json.dumps(record.get("initial_state_float"), ensure_ascii=False)
        csv_row["initial_state_u16"] = json.dumps(record.get("initial_state_u16"), ensure_ascii=False)
        csv_row["a_linear"] = json.dumps(record.get("a_linear"), ensure_ascii=False)
        csv_row["a_coupling"] = json.dumps(record.get("a_coupling"), ensure_ascii=False)
        csv_rows.append(csv_row)
    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_enhanced_golden_range_scale_index_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "enhanced_golden_range_scale_index",
) -> dict[str, Path]:
    """Export enhanced-map golden-range scale-index records to CSV and JSON."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "group_index",
        "a11",
        "a21",
        "a31",
        "b",
        "si_x1",
        "si_x2",
        "si_x3",
        "si_x4",
        "si_mean",
        "raw_length",
        "test_length",
        "mode",
        "wavelet",
        "scales",
        "error",
    ]
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        csv_row = {field: record.get(field) for field in fieldnames}
        csv_row["scales"] = json.dumps(record.get("scales"), ensure_ascii=False)
        csv_rows.append(csv_row)

    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_scale_index_comparison_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/raw",
    stem: str = "scale_index_comparison",
) -> dict[str, Path]:
    """Export cross-map scale-index comparison records to CSV and JSON."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "map_name",
        "si_x1",
        "si_x2",
        "si_x3",
        "si_x4",
        "si_mean",
        "raw_length",
        "test_length",
        "mode",
        "wavelet",
        "scales",
    ]
    csv_rows: list[dict[str, Any]] = []
    for record in records:
        csv_row = {field: record.get(field) for field in fieldnames}
        csv_row["scales"] = json.dumps(record.get("scales"), ensure_ascii=False)
        csv_rows.append(csv_row)

    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_enhanced_golden_range_pe_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/permutation_entropy/raw",
    stem: str = "enhanced_golden_range_pe",
) -> dict[str, Path]:
    """Export enhanced-map golden-range permutation-entropy records."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "group_index",
        "a11",
        "a21",
        "a31",
        "b",
        "pe_x1",
        "pe_x2",
        "pe_x3",
        "pe_x4",
        "pe_mean",
        "raw_length",
        "test_length",
        "mode",
        "m",
        "tau",
        "normalized",
        "error",
    ]
    csv_rows = [{field: record.get(field) for field in fieldnames} for record in records]
    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def export_pe_comparison_records(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/permutation_entropy/raw",
    stem: str = "pe_comparison",
) -> dict[str, Path]:
    """Export cross-map permutation-entropy comparison records."""

    output_root = Path(output_dir)
    json_path = save_json(records, output_root / f"{stem}.json")
    fieldnames = [
        "map_name",
        "pe_x1",
        "pe_x2",
        "pe_x3",
        "pe_x4",
        "pe_mean",
        "raw_length",
        "test_length",
        "mode",
        "m",
        "tau",
        "normalized",
    ]
    csv_rows = [{field: record.get(field) for field in fieldnames} for record in records]
    csv_path = save_csv_rows(csv_rows, output_root / f"{stem}.csv", fieldnames=fieldnames)
    return {"json": json_path, "csv": csv_path}


def main() -> None:
    """Placeholder main function for direct CLI extension later."""

    print("Use export_proposed_golden_range_le_records() from experiment scripts.")


if __name__ == "__main__":
    main()
