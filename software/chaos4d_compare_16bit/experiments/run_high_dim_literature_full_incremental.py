# Maintainer: zhaowenbo
"""Incremental full run for the all-method high-dimensional comparison."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from experiments.run_high_dim_literature_comparison import (
    DEFAULT_ZANG_DIAGONAL,
    DEFAULT_ZANG_ROW_COUPLING_BUDGET,
    METHODS,
    build_comparison_initial_state_u16,
    build_zang_linear_modular_matrix,
    evaluate_method_dimension,
    export_algorithm_analysis,
    export_comparison_table,
    get_method_dimension_support,
    plot_metric_comparisons,
    serialize_int_list,
)
from metrics.fuzzy_entropy import DEFAULT_FE_RAW_LENGTH, DEFAULT_FE_TRANSIENT
from metrics.nonperiodicity import DEFAULT_SI_RAW_LENGTH, DEFAULT_SI_TRANSIENT
from metrics.permutation_entropy import DEFAULT_PE_RAW_LENGTH, DEFAULT_PE_TRANSIENT
from metrics.spectral_entropy import DEFAULT_SE_RAW_LENGTH, DEFAULT_SE_TRANSIENT
from utils.io_utils import ensure_directory, save_csv_rows


SUMMARY_FIELDS = [
    "dimension",
    "method",
    "method_slug",
    "display_name",
    "initial_state_u16",
    "lambda_max",
    "lyap_mean",
    "lyap_components",
    "se_mean",
    "fe_mean",
    "si_mean",
    "pe_mean",
    "se_components",
    "fe_components",
    "si_components",
    "pe_components",
    "raw_length_se",
    "raw_length_fe",
    "raw_length_si",
    "raw_length_pe",
    "le_renormalizations",
]
DETAILED_FIELDS = [
    "dimension",
    "method",
    "method_slug",
    "display_name",
    "component",
    "lyap_component",
    "se",
    "fe",
    "si",
    "pe",
]
SUPPORT_FIELDS = [
    "method",
    "method_slug",
    "dimension",
    "supported",
    "public_dimension",
    "internal_dimension",
    "initialization_rule",
    "support_note",
]
CONFIG_FIELDS = [
    "dimension",
    "initial_state_u16",
    "zang_diagonal",
    "zang_row_coupling_budget",
    "zang_min_over_one_margin",
    "se_length",
    "se_transient",
    "fe_length",
    "fe_transient",
    "pe_length",
    "pe_transient",
    "si_length",
    "si_transient",
    "le_steps",
    "le_transient",
]


def flush_outputs(
    *,
    raw_dir: Path,
    summary_rows: list[dict[str, Any]],
    detailed_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
) -> None:
    """Write current CSV outputs after each completed method/dimension pair."""

    save_csv_rows(summary_rows, raw_dir / "high_dim_all_methods_summary.csv", SUMMARY_FIELDS)
    save_csv_rows(detailed_rows, raw_dir / "high_dim_all_methods_detailed.csv", DETAILED_FIELDS)
    save_csv_rows(support_rows, raw_dir / "method_dimension_support.csv", SUPPORT_FIELDS)
    save_csv_rows(config_rows, raw_dir / "experiment_config.csv", CONFIG_FIELDS)


def main() -> None:
    dimensions = list(range(4, 13))
    base_dir = ensure_directory("results/high_dim_literature_comparison")
    raw_dir = ensure_directory(base_dir / "raw")
    table_dir = ensure_directory(base_dir / "tables")
    figure_dir = ensure_directory(base_dir / "figures")

    se_length = DEFAULT_SE_RAW_LENGTH
    se_transient = DEFAULT_SE_TRANSIENT
    fe_length = DEFAULT_FE_RAW_LENGTH
    fe_transient = DEFAULT_FE_TRANSIENT
    pe_length = DEFAULT_PE_RAW_LENGTH
    pe_transient = DEFAULT_PE_TRANSIENT
    si_length = DEFAULT_SI_RAW_LENGTH
    si_transient = DEFAULT_SI_TRANSIENT
    le_steps = 5000
    le_transient = 1000

    export_algorithm_analysis(base_dir / "algorithm_analysis_all.md")
    support_rows = get_method_dimension_support(dimensions)
    summary_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []

    for dimension in dimensions:
        initial_state = build_comparison_initial_state_u16(dimension)
        zang_matrix = build_zang_linear_modular_matrix(dimension)
        margins = np.abs(np.diag(zang_matrix)) - (
            np.sum(np.abs(zang_matrix), axis=1) - np.abs(np.diag(zang_matrix))
        )
        config_rows.append(
            {
                "dimension": int(dimension),
                "initial_state_u16": serialize_int_list(initial_state),
                "zang_diagonal": float(DEFAULT_ZANG_DIAGONAL),
                "zang_row_coupling_budget": float(DEFAULT_ZANG_ROW_COUPLING_BUDGET),
                "zang_min_over_one_margin": float(np.min(margins)),
                "se_length": int(se_length),
                "se_transient": int(se_transient),
                "fe_length": int(fe_length),
                "fe_transient": int(fe_transient),
                "pe_length": int(pe_length),
                "pe_transient": int(pe_transient),
                "si_length": int(si_length),
                "si_transient": int(si_transient),
                "le_steps": int(le_steps),
                "le_transient": int(le_transient),
            }
        )

        for method in METHODS:
            print(f"START n={dimension}, method={method}", flush=True)
            summary, details = evaluate_method_dimension(
                method,
                dimension,
                initial_state_u16=initial_state,
                se_length=se_length,
                se_transient=se_transient,
                fe_length=fe_length,
                fe_transient=fe_transient,
                pe_length=pe_length,
                pe_transient=pe_transient,
                si_length=si_length,
                si_transient=si_transient,
                le_steps=le_steps,
                le_transient=le_transient,
            )
            summary_rows.append(summary)
            detailed_rows.extend(details)
            flush_outputs(
                raw_dir=raw_dir,
                summary_rows=summary_rows,
                detailed_rows=detailed_rows,
                support_rows=support_rows,
                config_rows=config_rows,
            )
            print(
                f"DONE n={dimension}, method={method}, "
                f"lambda_max={summary['lambda_max']:.6f}, "
                f"SE={summary['se_mean']:.6f}, FE={summary['fe_mean']:.6f}, "
                f"SI={summary['si_mean']:.6f}, PE={summary['pe_mean']:.6f}",
                flush=True,
            )

    table_path = export_comparison_table(summary_rows, table_dir / "high_dim_all_methods_comparison_table.tex")
    figure_paths = plot_metric_comparisons(summary_rows, figure_dir)
    print(f"SAVED table={table_path}", flush=True)
    for stem, pair in figure_paths.items():
        print(f"SAVED {stem}: {pair['pdf']} {pair['eps']}", flush=True)


if __name__ == "__main__":
    main()
