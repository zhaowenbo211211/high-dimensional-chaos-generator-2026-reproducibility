# Maintainer: zhaowenbo
"""Helpers for making project-local imports work from subdirectory entry points."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_project_root(start_file: str) -> Path:
    """Insert the repository root into ``sys.path`` based on a file path."""

    start_path = Path(start_file).resolve().parent
    project_root = start_path
    for candidate in (start_path, *start_path.parents):
        if (candidate / "project_bootstrap.py").exists() and (candidate / "chaos_maps").is_dir():
            project_root = candidate
            break

    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    return project_root
