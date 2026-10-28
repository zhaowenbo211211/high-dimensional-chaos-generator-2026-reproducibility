# Maintainer: zhaowenbo
"""Generate a reproducible framework for all high-dimensional methods.

This entry point is intentionally conservative.  The default ``smoke`` mode
creates the requested directory tree, writes CSV/LaTeX/figure templates, and
performs a one-step structure check for the five registered high-dimensional
maps.  The ``full`` mode is accepted as a future execution hook, but it does
not run the long metric experiment in this framework generator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import (  # noqa: E402
    CyclicSymmetricLogisticMap,
    ProposedNDEnhancedMap,
    SineMapLatticeMemristiveMap,
    YaoComplexPascalMap,
    ZangLinearModularMap,
)
from chaos_maps.literature_high_dim import get_linear_modular_initial_state_u16  # noqa: E402
from utils.io_utils import ensure_directory, save_csv_rows, save_json  # noqa: E402


OUTPUT_ROOT = Path("results/high_dim_literature_comparison")
DIMENSIONS = list(range(4, 13))
METRICS = ["lambda_max", "SE_mean", "FE_mean", "SI_mean", "PE_mean"]
FIGURE_STEMS = {
    "lambda_max": "lambda_max",
    "SE_mean": "se_mean",
    "FE_mean": "fe_mean",
    "SI_mean": "si_mean",
    "PE_mean": "pe_mean",
}


MapFactory = Callable[[int, np.ndarray], Any]


METHOD_REGISTRY: list[dict[str, Any]] = [
    {
        "method": "Proposed Enhanced n-D Map",
        "method_slug": "proposed_enhanced",
        "display_name": "The proposed method",
        "class_name": "ProposedNDEnhancedMap",
        "implementation_file": "chaos_maps/proposed_enhanced.py",
        "factory": lambda n, initial: ProposedNDEnhancedMap(n=n, initial_state_u16=initial),
        "internal_multiplier": 1,
        "support_note": "Enhanced proposed n-D map only; base-only variant is excluded.",
    },
    {
        "method": "Yao et al. 2025 n-D complex map",
        "method_slug": "yao_complex_pascal",
        "display_name": "Yao et al.'s method",
        "class_name": "YaoComplexPascalMap",
        "implementation_file": "chaos_maps/literature_high_dim.py",
        "factory": lambda n, initial: YaoComplexPascalMap(n=n, initial_state_u16=initial),
        "internal_multiplier": 2,
        "support_note": "Complex internal state; real initial state is x_i(0)=100*i and imaginary initial state is fixed to zero.",
    },
    {
        "method": "Sine-map lattice memristive",
        "method_slug": "sine_lattice_memristive",
        "display_name": "He et al.'s method",
        "class_name": "SineMapLatticeMemristiveMap",
        "implementation_file": "chaos_maps/literature_high_dim.py",
        "factory": lambda n, initial: SineMapLatticeMemristiveMap(n=n, initial_state_u16=initial),
        "internal_multiplier": 3,
        "support_note": "Public dimension uses n lattice sites; memristor and delay auxiliary states start from zero.",
    },
    {
        "method": "High-dimensional cyclic symmetric map",
        "method_slug": "cyclic_symmetric_logistic",
        "display_name": "Zhang et al.'s method",
        "class_name": "CyclicSymmetricLogisticMap",
        "implementation_file": "chaos_maps/literature_high_dim.py",
        "factory": lambda n, initial: CyclicSymmetricLogisticMap(n=n, initial_state_u16=initial),
        "internal_multiplier": 1,
        "support_note": "Cyclic symmetric logistic construction generalized directly to requested n.",
    },
    {
        "method": "Zang et al. 2021 linear modular system",
        "method_slug": "zang_linear_modular",
        "display_name": "Zang et al.'s method",
        "class_name": "ZangLinearModularMap",
        "implementation_file": "chaos_maps/literature_high_dim.py",
        "factory": lambda n, initial: ZangLinearModularMap(n=n, initial_state_u16=initial),
        "internal_multiplier": 1,
        "support_note": "Existing linear modular implementation with deterministic generated matrices for n=4,...,12.",
    },
]


def create_output_dirs(output_root: Path = OUTPUT_ROOT) -> dict[str, Path]:
    """Create the requested output directory tree."""

    return {
        "base": ensure_directory(output_root),
        "raw": ensure_directory(output_root / "raw"),
        "tables": ensure_directory(output_root / "tables"),
        "figures": ensure_directory(output_root / "figures"),
        "diagnostics": ensure_directory(output_root / "diagnostics"),
    }


def build_initial_state(n: int) -> np.ndarray:
    """Return the shared comparison initial state x_i(0)=100*i."""

    return get_linear_modular_initial_state_u16(int(n))


def validate_u16_state(state: np.ndarray, n: int) -> None:
    """Raise if a returned state is not a valid public n-D uint16 vector."""

    state_array = np.asarray(state)
    if state_array.shape != (int(n),):
        raise ValueError(f"state shape must be ({int(n)},), got {state_array.shape}")
    if not np.all(np.isfinite(state_array)):
        raise ValueError("state contains non-finite values")
    if np.any((state_array < 0) | (state_array > 65535)):
        raise ValueError("state contains values outside [0,65535]")


def run_smoke_check() -> list[dict[str, Any]]:
    """Instantiate every method at n=4 and record one quantized step."""

    smoke_rows: list[dict[str, Any]] = []
    n = 4
    initial_state = build_initial_state(n)
    for entry in METHOD_REGISTRY:
        map_obj = entry["factory"](n, initial_state)
        initial_u16 = np.asarray(map_obj.get_state_u16(), dtype=np.int64)
        validate_u16_state(initial_u16, n)
        map_obj.step()
        next_u16 = np.asarray(map_obj.get_state_u16(), dtype=np.int64)
        validate_u16_state(next_u16, n)
        if entry["class_name"] == "YaoComplexPascalMap":
            imag_u16 = np.asarray(map_obj.get_imag_u16(), dtype=np.int64)
            validate_u16_state(imag_u16, n)
        for component, value in enumerate(next_u16, start=1):
            smoke_rows.append(
                {
                    "mode": "smoke",
                    "status": "computed_one_step",
                    "method": entry["method"],
                    "method_slug": entry["method_slug"],
                    "display_name": entry["display_name"],
                    "n": n,
                    "step": 1,
                    "component": component,
                    "state_u16": int(value),
                    "note": "one-step smoke trajectory only; full metric trajectory pending",
                }
            )
    return smoke_rows


def build_support_rows() -> list[dict[str, Any]]:
    """Return support metadata for every registered method and n=4,...,12."""

    rows: list[dict[str, Any]] = []
    for entry in METHOD_REGISTRY:
        for n in DIMENSIONS:
            rows.append(
                {
                    "method": entry["method"],
                    "method_slug": entry["method_slug"],
                    "display_name": entry["display_name"],
                    "class_name": entry["class_name"],
                    "implementation_file": entry["implementation_file"],
                    "n": n,
                    "supported": "yes",
                    "public_dimension": n,
                    "internal_dimension": n * int(entry["internal_multiplier"]),
                    "initial_state_rule": "x_i(0)=100*i",
                    "complex_imaginary_initial_state": "fixed_zero" if entry["class_name"] == "YaoComplexPascalMap" else "not_applicable",
                    "auxiliary_state_rule": "zero unless specified by literature implementation",
                    "support_note": entry["support_note"],
                }
            )
    return rows


def build_summary_template_rows(mode: str) -> list[dict[str, Any]]:
    """Create one pending summary row per method/dimension pair."""

    rows: list[dict[str, Any]] = []
    for entry in METHOD_REGISTRY:
        for n in DIMENSIONS:
            row = {
                "mode": mode,
                "status": "pending_full_metrics",
                "method": entry["method"],
                "method_slug": entry["method_slug"],
                "display_name": entry["display_name"],
                "n": n,
                "initial_state_u16": json.dumps([int(v) for v in build_initial_state(n)]),
                "lambda_max": "",
                "SE_mean": "",
                "FE_mean": "",
                "SI_mean": "",
                "PE_mean": "",
                "note": "Template row. Run the future full branch to compute metrics.",
            }
            rows.append(row)
    return rows


def write_detailed_csv(rows: list[dict[str, Any]], raw_dir: Path) -> Path:
    """Write smoke trajectory rows plus template headers for future full data."""

    return save_csv_rows(
        rows,
        raw_dir / "high_dim_all_methods_detailed.csv",
        fieldnames=[
            "mode",
            "status",
            "method",
            "method_slug",
            "display_name",
            "n",
            "step",
            "component",
            "state_u16",
            "note",
        ],
    )


def write_summary_csv(rows: list[dict[str, Any]], raw_dir: Path) -> Path:
    """Write the pending metric summary template."""

    return save_csv_rows(
        rows,
        raw_dir / "high_dim_all_methods_summary.csv",
        fieldnames=[
            "mode",
            "status",
            "method",
            "method_slug",
            "display_name",
            "n",
            "initial_state_u16",
            "lambda_max",
            "SE_mean",
            "FE_mean",
            "SI_mean",
            "PE_mean",
            "note",
        ],
    )


def write_support_csv(rows: list[dict[str, Any]], raw_dir: Path) -> Path:
    """Write dimension-support metadata."""

    return save_csv_rows(
        rows,
        raw_dir / "method_dimension_support.csv",
        fieldnames=[
            "method",
            "method_slug",
            "display_name",
            "class_name",
            "implementation_file",
            "n",
            "supported",
            "public_dimension",
            "internal_dimension",
            "initial_state_rule",
            "complex_imaginary_initial_state",
            "auxiliary_state_rule",
            "support_note",
        ],
    )


def write_latex_template(summary_rows: list[dict[str, Any]], table_dir: Path) -> Path:
    """Write a paper-ready table skeleton with pending metric cells."""

    table_path = table_dir / "high_dim_all_methods_comparison_table.tex"
    lines = [
        r"% Auto-generated template by experiments/run_high_dim_all_methods_full.py.",
        r"% Metric cells marked -- are placeholders for future full-run data.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{High-dimensional all-method comparison under unified 16-bit arithmetic.}",
        r"\label{tab:high-dimensional-all-methods-comparison}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{clccccc}",
        r"\hline",
        r"$n$ & Method & $\lambda_{\max}$ & $\mathrm{SE}_{\mathrm{mean}}$ & $\mathrm{FE}_{\mathrm{mean}}$ & $\mathrm{SI}_{\mathrm{mean}}$ & $\mathrm{PE}_{\mathrm{mean}}$ \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{int(row['n'])} & {row['display_name']} & -- & -- & -- & -- & -- \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    table_path.write_text("\n".join(lines), encoding="utf-8")
    return table_path


def write_placeholder_figures(figure_dir: Path) -> dict[str, dict[str, str]]:
    """Create one placeholder PDF/EPS pair per requested metric."""

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    import matplotlib.pyplot as plt

    figure_paths: dict[str, dict[str, str]] = {}
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(5.8, 3.6), dpi=300)
        ax.set_title(f"{metric} template")
        ax.set_xlabel("n")
        ax.set_ylabel(metric)
        ax.set_xlim(4, 12)
        ax.set_xticks(DIMENSIONS)
        ax.text(
            0.5,
            0.5,
            "Full experiment not executed\nmetric values pending",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.tick_params(direction="in")
        fig.tight_layout()
        stem = FIGURE_STEMS[metric]
        pdf_path = figure_dir / f"high_dim_comparison_{stem}.pdf"
        eps_path = figure_dir / f"high_dim_comparison_{stem}.eps"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(eps_path, bbox_inches="tight")
        plt.close(fig)
        figure_paths[metric] = {"pdf": str(pdf_path), "eps": str(eps_path)}
    return figure_paths


def write_diagnostics(mode: str, paths: dict[str, Any], diagnostics_dir: Path) -> Path:
    """Record framework-generation metadata."""

    return save_json(
        {
            "script": "experiments/run_high_dim_all_methods_full.py",
            "mode": mode,
            "full_experiment_executed": False,
            "dimensions": DIMENSIONS,
            "methods": [
                {
                    key: value
                    for key, value in entry.items()
                    if key not in {"factory"}
                }
                for entry in METHOD_REGISTRY
            ],
            "metric_scripts": {
                "lambda_max": "metrics/lyapunov.py",
                "SE_mean": "metrics/spectral_entropy.py",
                "FE_mean": "metrics/fuzzy_entropy.py",
                "SI_mean": "metrics/nonperiodicity.py",
                "PE_mean": "metrics/permutation_entropy.py",
            },
            "outputs": paths,
            "note": "This framework generator writes templates and smoke data only. Full metric computation is intentionally not executed.",
        },
        diagnostics_dir / "high_dim_all_methods_framework_metadata.json",
    )


def generate_framework(mode: str) -> dict[str, Any]:
    """Generate directories and template artifacts for the all-method study."""

    dirs = create_output_dirs()
    detailed_rows = run_smoke_check() if mode == "smoke" else []
    summary_rows = build_summary_template_rows(mode)
    support_rows = build_support_rows()

    detailed_csv = write_detailed_csv(detailed_rows, dirs["raw"])
    summary_csv = write_summary_csv(summary_rows, dirs["raw"])
    support_csv = write_support_csv(support_rows, dirs["raw"])
    table_path = write_latex_template(summary_rows, dirs["tables"])
    figure_paths = write_placeholder_figures(dirs["figures"])

    output_paths: dict[str, Any] = {
        "raw_detailed_csv": str(detailed_csv),
        "raw_summary_csv": str(summary_csv),
        "method_dimension_support_csv": str(support_csv),
        "latex_table": str(table_path),
        "figures": figure_paths,
    }
    diagnostics_json = write_diagnostics(mode, output_paths, dirs["diagnostics"])
    output_paths["diagnostics_json"] = str(diagnostics_json)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate templates for the high-dimensional all-method comparison."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke writes templates and one-step checks; full is a future hook and does not run the long experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_framework(args.mode)
    print(f"Generated high-dimensional all-method framework in {OUTPUT_ROOT}")
    print(f"Mode: {args.mode}; full experiment executed: no")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
