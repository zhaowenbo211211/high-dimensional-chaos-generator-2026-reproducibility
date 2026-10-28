# Maintainer: zhaowenbo
"""Minimal IO helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return the normalized path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_default_results_subdir(
    requested_dir: str | Path,
    default_root: str | Path,
    *subdirs: str,
) -> Path:
    """Resolve a default results directory into a more specific subdirectory.

    If the caller passes the shared default root such as ``results/raw`` or
    ``results/figures``, the returned path is expanded with ``subdirs`` so
    different experiment families do not all write into the same flat folder.
    Custom caller-provided directories are preserved as-is.
    """

    requested_path = Path(requested_dir)
    default_path = Path(default_root)
    if requested_path == default_path:
        requested_path = requested_path.joinpath(*subdirs)
    return ensure_directory(requested_path)


def ensure_metric_output_dirs(metric_name: str, root: str | Path = "results") -> dict[str, Path]:
    """Create and return a standard output tree for one metric family."""

    base_dir = ensure_directory(Path(root) / metric_name)
    return {
        "base": base_dir,
        "raw": ensure_directory(base_dir / "raw"),
        "tables": ensure_directory(base_dir / "tables"),
        "figures": ensure_directory(base_dir / "figures"),
        "diagnostics": ensure_directory(base_dir / "diagnostics"),
    }


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data using UTF-8 encoding."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def save_csv_rows(rows: Iterable[Mapping[str, Any]], path: str | Path, fieldnames: list[str]) -> Path:
    """Save row dictionaries to CSV with explicit field ordering."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output_path
