# Maintainer: zhaowenbo
"""Verbatim copies of the original paper-figure plotting functions.

These functions were copied from the legacy experiment project (read-only source):
  $OLD = paper2025/../可行性报告/python_program/chaos4d_compare_16bit/experiments/
See audit/plot_style_rework/source_map.md for per-figure provenance and SHA-256.

Only the publication plotting functions are copied; the scan/experiment drivers and
the project bootstrap are intentionally NOT copied. Two helpers from
``utils.io_utils`` are replaced by the local ``_ensure_directory``. No function in
this package loads legacy CSVs or regenerates trajectories: callers must pass the
already-validated rev1 data records.
"""
