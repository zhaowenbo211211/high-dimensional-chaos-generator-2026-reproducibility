# Maintainer: zhaowenbo
"""Run the enhanced high-dimensional family average dynamical evaluation."""

from __future__ import annotations

import argparse
import json
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

from chaos_maps import ProposedNDEnhancedMap
from experiments.plot_dimension_scalability import plot_dimension_scalability
from metrics.fuzzy_entropy import (
    DEFAULT_FE_EMBEDDING_DIM,
    DEFAULT_FE_FUZZY_EXPONENT,
    DEFAULT_FE_RAW_LENGTH,
    DEFAULT_FE_TAU,
    DEFAULT_FE_TRANSIENT,
    fuzzy_entropy,
)
from metrics.lyapunov import wolf_lyapunov_full_state_nd
from metrics.nonperiodicity import (
    DEFAULT_SI_RAW_LENGTH,
    DEFAULT_SI_TRANSIENT,
    DEFAULT_SI_WAVELET,
    scale_index_cwt,
)
from metrics.permutation_entropy import (
    DEFAULT_PE_EMBEDDING_DIM,
    DEFAULT_PE_NORMALIZED,
    DEFAULT_PE_RAW_LENGTH,
    DEFAULT_PE_TAU,
    DEFAULT_PE_TRANSIENT,
    permutation_entropy,
)
from metrics.spectral_entropy import DEFAULT_SE_RAW_LENGTH, DEFAULT_SE_TRANSIENT, spectral_entropy_complexity
from utils.initializers import get_dimension_initial_state_float
from utils.io_utils import ensure_directory, save_csv_rows
from utils.r1ly import synthesize_r1_ly_params


def _serialize_float_list(values: list[float]) -> str:
    return json.dumps([float(value) for value in values], ensure_ascii=False)


def _serialize_int_list(values: list[int]) -> str:
    return json.dumps([int(value) for value in values], ensure_ascii=False)


def export_dimension_metrics_table(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Export a compact LaTeX summary table."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    lines = [
        r"\begin{tabular}{cccccc}",
        r"\hline",
        r"$n$ & $\lambda_{\max}$ & $\mathrm{SE}_{\mathrm{mean}}$ & $\mathrm{FE}_{\mathrm{mean}}$ & $\mathrm{SI}_{\mathrm{mean}}$ & $\mathrm{PE}_{\mathrm{mean}}$ \\",
        r"\hline",
    ]
    for record in sorted(records, key=lambda item: int(item["dimension"])):
        lines.append(
            f"{int(record['dimension'])} & "
            f"{float(record['lyap_max']):.6f} & "
            f"{float(record['se_mean']):.6f} & "
            f"{float(record['fe_mean']):.6f} & "
            f"{float(record['si_mean']):.6f} & "
            f"{float(record['pe_mean']):.6f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_dimension_scalability(
    dimensions: list[int] | None = None,
    se_length: int = DEFAULT_SE_RAW_LENGTH,
    se_transient: int = DEFAULT_SE_TRANSIENT,
    fe_length: int = DEFAULT_FE_RAW_LENGTH,
    fe_transient: int = DEFAULT_FE_TRANSIENT,
    pe_length: int = DEFAULT_PE_RAW_LENGTH,
    pe_transient: int = DEFAULT_PE_TRANSIENT,
    si_length: int = DEFAULT_SI_RAW_LENGTH,
    si_transient: int = DEFAULT_SI_TRANSIENT,
    le_steps: int = 5000,
    le_transient: int = 1000,
    output_root: str | Path = "results/high_dimensional",
    progress: bool = True,
) -> dict[str, Any]:
    """Run the enhanced-family dimension scalability experiment for `n=4..12`."""

    dims = list(range(4, 13)) if dimensions is None else [int(value) for value in dimensions]
    base_dir = ensure_directory(output_root)
    raw_dir = ensure_directory(base_dir / "raw")
    table_dir = ensure_directory(base_dir / "tables")
    figure_dir = ensure_directory(base_dir / "figures")
    diagnostics_dir = ensure_directory(base_dir / "diagnostics")

    parameter_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    for dimension in dims:
        if progress:
            print(f"Running enhanced high-dimensional evaluation for n={dimension}")

        params = synthesize_r1_ly_params(n=dimension)
        parameter_rows.append(
            {
                "dimension": int(dimension),
                "A_odd_init": int(params["A_odd_init"]),
                "B_even_init": int(params["B_even_init"]),
                "A_odd_final": int(params["A_odd_final"]),
                "B_even_final": int(params["B_even_final"]),
                "eta_final": int(params["eta_final"]),
                "tau_final": int(params["tau_final"]),
                "a_linear": _serialize_int_list(list(params["a_linear"])),
                "a_coupling": _serialize_int_list(list(params["a_coupling"])),
                "b": int(params["b"]),
                "beta": float(params["beta"]),
                "condition_s_passed": bool(params["condition_s_passed"]),
                "no_wrap_passed": bool(params["no_wrap_passed"]),
                "auto_adjustments": json.dumps(list(params["auto_adjustments"]), ensure_ascii=False),
            }
        )

        initial_state = get_dimension_initial_state_float(dimension)
        se_map = ProposedNDEnhancedMap(n=dimension, initial_state_float=initial_state)
        fe_map = ProposedNDEnhancedMap(n=dimension, initial_state_float=initial_state)
        pe_map = ProposedNDEnhancedMap(n=dimension, initial_state_float=initial_state)
        si_map = ProposedNDEnhancedMap(n=dimension, initial_state_float=initial_state)
        le_map = ProposedNDEnhancedMap(n=dimension, initial_state_float=initial_state)

        se_traj = se_map.iterate(num_steps=se_length, discard_transient=se_transient)
        fe_traj = fe_map.iterate(num_steps=fe_length, discard_transient=fe_transient)
        pe_traj = pe_map.iterate(num_steps=pe_length, discard_transient=pe_transient)
        si_traj = si_map.iterate(num_steps=si_length, discard_transient=si_transient)
        se_result = spectral_entropy_complexity(se_traj, mode="direct")
        fe_result = fuzzy_entropy(
            fe_traj,
            mode="direct",
            m=DEFAULT_FE_EMBEDDING_DIM,
            r=None,
            tau=DEFAULT_FE_TAU,
            n=DEFAULT_FE_FUZZY_EXPONENT,
        )
        pe_result = permutation_entropy(
            pe_traj,
            mode="direct",
            m=DEFAULT_PE_EMBEDDING_DIM,
            tau=DEFAULT_PE_TAU,
            normalized=DEFAULT_PE_NORMALIZED,
        )
        si_result = scale_index_cwt(si_traj, mode="direct", wavelet=DEFAULT_SI_WAVELET)
        le_result = wolf_lyapunov_full_state_nd(le_map, max_steps=le_steps, transient=le_transient)

        detailed_rows.extend(
            {
                "dimension": int(dimension),
                "component": int(component_index + 1),
                "se": float(se_result["se_components"][component_index]),
                "fe": float(fe_result["fe_components"][component_index]),
                "si": float(si_result["si_components"][component_index]),
                "pe": float(pe_result["pe_components"][component_index]),
            }
            for component_index in range(dimension)
        )

        summary_rows.append(
            {
                "dimension": int(dimension),
                "lyap_max": float(le_result["lyap_max"]),
                "lyap_mean": float(le_result["lyap_mean"]),
                "lyap_components": _serialize_float_list(list(np.asarray(le_result["lyap_components"], dtype=np.float64))),
                "se_mean": float(se_result["se_mean"]),
                "fe_mean": float(fe_result["fe_mean"]),
                "si_mean": float(si_result["si_mean"]),
                "pe_mean": float(pe_result["pe_mean"]),
                "se_components": _serialize_float_list(list(se_result["se_components"])),
                "fe_components": _serialize_float_list(list(fe_result["fe_components"])),
                "si_components": _serialize_float_list(list(si_result["si_components"])),
                "pe_components": _serialize_float_list(list(pe_result["pe_components"])),
                "raw_length_se": int(se_result["raw_length"]),
                "raw_length_fe": int(fe_result["raw_length"]),
                "raw_length_si": int(si_result["raw_length"]),
                "raw_length_pe": int(pe_result["raw_length"]),
                "le_renormalizations": int(le_result["renormalizations"]),
            }
        )

    parameter_csv = save_csv_rows(
        parameter_rows,
        raw_dir / "r1_ly_dimension_params.csv",
        fieldnames=[
            "dimension",
            "A_odd_init",
            "B_even_init",
            "A_odd_final",
            "B_even_final",
            "eta_final",
            "tau_final",
            "a_linear",
            "a_coupling",
            "b",
            "beta",
            "condition_s_passed",
            "no_wrap_passed",
            "auto_adjustments",
        ],
    )
    summary_csv = save_csv_rows(
        summary_rows,
        raw_dir / "dimension_metrics_summary.csv",
        fieldnames=[
            "dimension",
            "lyap_max",
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
        ],
    )
    detailed_csv = save_csv_rows(
        detailed_rows,
        raw_dir / "dimension_metrics_detailed.csv",
        fieldnames=["dimension", "component", "se", "fe", "si", "pe"],
    )
    figure_paths = plot_dimension_scalability(summary_rows, output_dir=figure_dir)
    latex_table = export_dimension_metrics_table(summary_rows, table_dir / "dimension_metrics_summary.tex")

    diagnostics_path = diagnostics_dir / "dimension_experiment_metadata.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "dimensions": dims,
                "se_length": se_length,
                "se_transient": se_transient,
                "fe_length": fe_length,
                "fe_transient": fe_transient,
                "pe_length": pe_length,
                "pe_transient": pe_transient,
                "si_length": si_length,
                "si_transient": si_transient,
                "le_steps": le_steps,
                "le_transient": le_transient,
                "map": "ProposedNDEnhancedMap",
                "initialization_rule": "dimension-aware extension of (0.1, 0.2, 0.3, 0.4)",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "parameter_csv": parameter_csv,
        "summary_csv": summary_csv,
        "detailed_csv": detailed_csv,
        "latex_table": latex_table,
        "figure_paths": figure_paths,
        "diagnostics_json": diagnostics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the enhanced high-dimensional family experiment.")
    parser.add_argument("--dimensions", nargs="*", type=int, default=list(range(4, 13)))
    parser.add_argument("--output-root", default="results/high_dimensional")
    parser.add_argument("--se-length", type=int, default=DEFAULT_SE_RAW_LENGTH)
    parser.add_argument("--se-transient", type=int, default=DEFAULT_SE_TRANSIENT)
    parser.add_argument("--fe-length", type=int, default=DEFAULT_FE_RAW_LENGTH)
    parser.add_argument("--fe-transient", type=int, default=DEFAULT_FE_TRANSIENT)
    parser.add_argument("--pe-length", type=int, default=DEFAULT_PE_RAW_LENGTH)
    parser.add_argument("--pe-transient", type=int, default=DEFAULT_PE_TRANSIENT)
    parser.add_argument("--si-length", type=int, default=DEFAULT_SI_RAW_LENGTH)
    parser.add_argument("--si-transient", type=int, default=DEFAULT_SI_TRANSIENT)
    parser.add_argument("--le-steps", type=int, default=5000)
    parser.add_argument("--le-transient", type=int, default=1000)
    args = parser.parse_args()

    result = run_dimension_scalability(
        dimensions=[int(value) for value in args.dimensions],
        se_length=args.se_length,
        se_transient=args.se_transient,
        fe_length=args.fe_length,
        fe_transient=args.fe_transient,
        pe_length=args.pe_length,
        pe_transient=args.pe_transient,
        si_length=args.si_length,
        si_transient=args.si_transient,
        le_steps=args.le_steps,
        le_transient=args.le_transient,
        output_root=args.output_root,
        progress=True,
    )
    print(f"Saved parameter CSV: {result['parameter_csv']}")
    print(f"Saved summary CSV: {result['summary_csv']}")
    print(f"Saved detailed CSV: {result['detailed_csv']}")
    print(f"Saved LaTeX table: {result['latex_table']}")
    print(f"Saved PDF: {result['figure_paths']['pdf']}")
    print(f"Saved EPS: {result['figure_paths']['eps']}")


if __name__ == "__main__":
    main()
