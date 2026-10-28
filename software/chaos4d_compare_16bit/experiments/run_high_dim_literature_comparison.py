# Maintainer: zhaowenbo
"""Compare proposed enhanced n-D family with reproduced literature generators."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import (
    CyclicSymmetricLogisticMap,
    ProposedNDEnhancedMap,
    SineMapLatticeMemristiveMap,
    YaoComplexPascalMap,
    ZangLinearModularMap,
)
from chaos_maps.literature_high_dim import (
    DEFAULT_ZANG_DIAGONAL,
    DEFAULT_ZANG_ROW_COUPLING_BUDGET,
    build_zang_linear_modular_matrix,
    get_linear_modular_initial_state_u16,
)
from metrics.fuzzy_entropy import (
    DEFAULT_FE_EMBEDDING_DIM,
    DEFAULT_FE_FUZZY_EXPONENT,
    DEFAULT_FE_RAW_LENGTH,
    DEFAULT_FE_TAU,
    DEFAULT_FE_TRANSIENT,
    fuzzy_entropy,
)
from metrics.lyapunov import wolf_lyapunov_full_state_nd
from metrics.nonperiodicity import DEFAULT_SI_RAW_LENGTH, DEFAULT_SI_TRANSIENT, DEFAULT_SI_WAVELET, scale_index_cwt
from metrics.permutation_entropy import (
    DEFAULT_PE_EMBEDDING_DIM,
    DEFAULT_PE_NORMALIZED,
    DEFAULT_PE_RAW_LENGTH,
    DEFAULT_PE_TAU,
    DEFAULT_PE_TRANSIENT,
    permutation_entropy,
)
from metrics.spectral_entropy import DEFAULT_SE_RAW_LENGTH, DEFAULT_SE_TRANSIENT, spectral_entropy_complexity
from utils.io_utils import ensure_directory, save_csv_rows


METHOD_PROPOSED = "Proposed enhanced n-D chaotic family"
METHOD_YAO = "Yao et al. 2025 complex Pascal n-D map"
METHOD_SINE_MEMRISTIVE = "Sine-map lattice with discrete memristor coupling"
METHOD_CYCLIC = "High-dimensional cyclic symmetric logistic map"
METHOD_ZANG = "Zang et al. 2021 linear modular system"
METHOD_SLUGS = {
    METHOD_PROPOSED: "proposed_enhanced",
    METHOD_YAO: "yao_complex_pascal",
    METHOD_SINE_MEMRISTIVE: "sine_lattice_memristive",
    METHOD_CYCLIC: "cyclic_symmetric_logistic",
    METHOD_ZANG: "zang_linear_modular",
}
METHODS = [METHOD_PROPOSED, METHOD_YAO, METHOD_SINE_MEMRISTIVE, METHOD_CYCLIC, METHOD_ZANG]
METHOD_SHORT_LABELS = {
    METHOD_PROPOSED: "The proposed method",
    METHOD_YAO: "Yao et al.'s method",
    METHOD_SINE_MEMRISTIVE: "He et al.'s method",
    METHOD_CYCLIC: "Zhang et al.'s method",
    METHOD_ZANG: "Zang et al.'s method",
}
METHOD_IMPLEMENTATION_NOTES = {
    METHOD_PROPOSED: "Existing enhanced n-D digital map; base-only method excluded.",
    METHOD_YAO: "Complex Pascal matrix map; real state is public n-D output, imaginary initial state is fixed to zero.",
    METHOD_SINE_MEMRISTIVE: "Requested n maps to m=n lattice sites; q_i and y_i are internal auxiliary states initialized to zero.",
    METHOD_CYCLIC: "General cyclic construction with logistic seed maps extended directly to requested n.",
    METHOD_ZANG: "Existing linear modular matrix reproduction with deterministic 4D-12D matrix generator.",
}
METRIC_SPECS = [
    ("lambda_max", r"$\lambda_{\max}$", "lambda_max"),
    ("se_mean", r"$\mathrm{SE}_{\mathrm{mean}}$", "se_mean"),
    ("fe_mean", r"$\mathrm{FE}_{\mathrm{mean}}$", "fe_mean"),
    ("si_mean", r"$\mathrm{SI}_{\mathrm{mean}}$", "si_mean"),
    ("pe_mean", r"$\mathrm{PE}_{\mathrm{mean}}$", "pe_mean"),
]


def serialize_float_list(values: list[float] | np.ndarray) -> str:
    return json.dumps([float(value) for value in list(values)], ensure_ascii=False)


def serialize_int_list(values: list[int] | np.ndarray) -> str:
    return json.dumps([int(value) for value in list(values)], ensure_ascii=False)


def build_comparison_initial_state_u16(n: int) -> np.ndarray:
    """Return `[100, 200, ..., 100*n]` in the shared 16-bit domain."""

    return get_linear_modular_initial_state_u16(int(n))


def _make_map(method: str, dimension: int, initial_state_u16: np.ndarray):
    if method == METHOD_PROPOSED:
        return ProposedNDEnhancedMap(n=dimension, initial_state_u16=initial_state_u16)
    if method == METHOD_YAO:
        return YaoComplexPascalMap(n=dimension, initial_state_u16=initial_state_u16)
    if method == METHOD_SINE_MEMRISTIVE:
        return SineMapLatticeMemristiveMap(n=dimension, initial_state_u16=initial_state_u16)
    if method == METHOD_CYCLIC:
        return CyclicSymmetricLogisticMap(n=dimension, initial_state_u16=initial_state_u16)
    if method == METHOD_ZANG:
        return ZangLinearModularMap(n=dimension, initial_state_u16=initial_state_u16)
    raise ValueError(f"unsupported method: {method}")


def get_method_dimension_support(dimensions: list[int] | None = None) -> list[dict[str, Any]]:
    """Return support metadata for all methods and requested dimensions."""

    dims = list(range(4, 13)) if dimensions is None else [int(value) for value in dimensions]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for dimension in dims:
            rows.append(
                {
                    "method": method,
                    "method_slug": METHOD_SLUGS[method],
                    "dimension": int(dimension),
                    "supported": "yes",
                    "public_dimension": int(dimension),
                    "internal_dimension": int(2 * dimension if method == METHOD_YAO else 3 * dimension if method == METHOD_SINE_MEMRISTIVE else dimension),
                    "initialization_rule": "x_i(0)=100*i; complex imaginary initial states are fixed to zero; other auxiliary states are zero where applicable",
                    "support_note": METHOD_IMPLEMENTATION_NOTES[method],
                }
            )
    return rows


def _finite_or_nan(value: float) -> float:
    value_float = float(value)
    return value_float if np.isfinite(value_float) else float("nan")


def evaluate_method_dimension(
    method: str,
    dimension: int,
    *,
    initial_state_u16: np.ndarray,
    se_length: int,
    se_transient: int,
    fe_length: int,
    fe_transient: int,
    pe_length: int,
    pe_transient: int,
    si_length: int,
    si_transient: int,
    le_steps: int,
    le_transient: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all requested metrics for one method and one dimension."""

    se_map = _make_map(method, dimension, initial_state_u16)
    fe_map = _make_map(method, dimension, initial_state_u16)
    pe_map = _make_map(method, dimension, initial_state_u16)
    si_map = _make_map(method, dimension, initial_state_u16)
    le_map = _make_map(method, dimension, initial_state_u16)

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

    detailed_rows: list[dict[str, Any]] = []
    lyap_components = np.asarray(le_result["lyap_components"], dtype=np.float64)
    for component_index in range(dimension):
        detailed_rows.append(
            {
                "dimension": int(dimension),
                "method": method,
                "method_slug": METHOD_SLUGS[method],
                "display_name": METHOD_SHORT_LABELS[method],
                "component": int(component_index + 1),
                "lyap_component": _finite_or_nan(lyap_components[component_index]),
                "se": _finite_or_nan(se_result["se_components"][component_index]),
                "fe": _finite_or_nan(fe_result["fe_components"][component_index]),
                "si": _finite_or_nan(si_result["si_components"][component_index]),
                "pe": _finite_or_nan(pe_result["pe_components"][component_index]),
            }
        )

    summary = {
        "dimension": int(dimension),
        "method": method,
        "method_slug": METHOD_SLUGS[method],
        "display_name": METHOD_SHORT_LABELS[method],
        "initial_state_u16": serialize_int_list(initial_state_u16),
        "lambda_max": _finite_or_nan(le_result["lyap_max"]),
        "lyap_mean": _finite_or_nan(le_result["lyap_mean"]),
        "lyap_components": serialize_float_list(lyap_components),
        "se_mean": _finite_or_nan(se_result["se_mean"]),
        "fe_mean": _finite_or_nan(fe_result["fe_mean"]),
        "si_mean": _finite_or_nan(si_result["si_mean"]),
        "pe_mean": _finite_or_nan(pe_result["pe_mean"]),
        "se_components": serialize_float_list(se_result["se_components"]),
        "fe_components": serialize_float_list(fe_result["fe_components"]),
        "si_components": serialize_float_list(si_result["si_components"]),
        "pe_components": serialize_float_list(pe_result["pe_components"]),
        "raw_length_se": int(se_result["raw_length"]),
        "raw_length_fe": int(fe_result["raw_length"]),
        "raw_length_si": int(si_result["raw_length"]),
        "raw_length_pe": int(pe_result["raw_length"]),
        "le_renormalizations": int(le_result["renormalizations"]),
    }
    return summary, detailed_rows


def export_comparison_table(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Export the paper-facing high-dimensional comparison table."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{High-dimensional literature comparison under unified 16-bit digital arithmetic.}",
        r"\label{tab:high-dimensional-all-literature-comparison}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{clccccc}",
        r"\hline",
        r"$n$ & Method & $\lambda_{\max}$ & $\mathrm{SE}_{\mathrm{mean}}$ & $\mathrm{FE}_{\mathrm{mean}}$ & $\mathrm{SI}_{\mathrm{mean}}$ & $\mathrm{PE}_{\mathrm{mean}}$ \\",
        r"\hline",
    ]
    for record in sorted(records, key=lambda item: (int(item["dimension"]), str(item["method_slug"]))):
        method_label = METHOD_SHORT_LABELS[str(record["method"])]
        lines.append(
            f"{int(record['dimension'])} & {method_label} & "
            f"{float(record['lambda_max']):.6f} & "
            f"{float(record['se_mean']):.6f} & "
            f"{float(record['fe_mean']):.6f} & "
            f"{float(record['si_mean']):.6f} & "
            f"{float(record['pe_mean']):.6f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def plot_metric_comparisons(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Create one PDF/EPS figure per metric."""

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    import matplotlib.pyplot as plt

    figure_dir = ensure_directory(output_dir)
    paths: dict[str, dict[str, Path]] = {}
    methods = METHODS
    markers = {
        METHOD_PROPOSED: "o",
        METHOD_YAO: "^",
        METHOD_SINE_MEMRISTIVE: "D",
        METHOD_CYCLIC: "v",
        METHOD_ZANG: "s",
    }
    labels = METHOD_SHORT_LABELS

    for metric_key, ylabel, stem in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=300)
        for method in methods:
            method_records = sorted(
                [record for record in records if record["method"] == method],
                key=lambda item: int(item["dimension"]),
            )
            ax.plot(
                [int(record["dimension"]) for record in method_records],
                [float(record[metric_key]) for record in method_records],
                marker=markers[method],
                linewidth=1.4,
                markersize=4.0,
                label=labels[method],
            )
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(4, 13)))
        ax.tick_params(direction="in")
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf_path = figure_dir / f"high_dim_comparison_{stem}.pdf"
        eps_path = figure_dir / f"high_dim_comparison_{stem}.eps"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(eps_path, bbox_inches="tight")
        plt.close(fig)
        paths[stem] = {"pdf": pdf_path, "eps": eps_path}
    return paths


def export_algorithm_analysis(path: str | Path) -> Path:
    """Write the required algorithm-analysis document."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(
        """# Algorithm Analysis for All High-Dimensional Literature Reproductions

## Unified Comparison Semantics

- Compared dimensions: `n=4,...,12`.
- Public initial state: `x_i(0)=100*i` in the 16-bit integer domain.
- Complex imaginary initial states are fixed to zero. Internal auxiliary states are initialized to zero unless a paper-specific rule is required.
- Every step maps the public state back to uint16 `[0,65535]`.
- The proposed comparator is only the enhanced n-D method; base-only results are excluded.

## Implemented Methods

1. Proposed enhanced n-D chaotic family
   - Source: existing project method `ProposedNDEnhancedMap`.
   - Rule: base n-D update followed by pack, rotate-left, unpack, and uint16 storage.

2. Yao et al. 2025 complex Pascal n-D map
   - Source file: `new-materials/Generation of n-dimensional complex chaotic system via parameter matrixconfiguration.pdf`
   - Paper form: `z(k+1)=CFOLD_1(A z(k)+v)` with a complex Pascal parameter matrix.
   - Digital mapping: real parts use `x_i(0)=100*i`; imaginary initial parts are fixed to zero and cannot be randomized. The public n-D trajectory is the real part, while the imaginary part is retained internally during iteration.
   - Reproduction decision: use deterministic positive Pascal borders and a deterministic fixed complex vector so 4D--12D runs are reproducible.

3. Sine-map lattice with discrete memristor coupling
   - Source file: `new-materials/Spatiotemporal_Chaos_in_a_Sine_Map_Lattice_With_Discrete_Memristor_Coupling.pdf`
   - Paper form: sine-map coupled lattice with memristor charge `q_i` and delay `y_i`.
   - Digital mapping: requested `n` maps to `m=n` public lattice sites. The internal state has `x_i`, `q_i`, and `y_i`; only `x_i` is exported to the metric pipeline.
   - Parameters: `a=1`, `psi=0.99`, `k=1.5`, `w=pi`, matching the paper's lattice example.

4. High-dimensional cyclic symmetric logistic map
   - Source file: `new-materials/Construction of high-dimensional cyclic symmetric chaotic map with one-dimensional chaotic map and its security application.pdf`
   - Paper form: `F(X)=(f(x_2), f(x_3), ..., f(x_1))`.
   - Digital mapping: use logistic seed maps with deterministic parameters near the chaotic range and quantize every step.

5. Zang et al. 2021 linear modular system
   - Source file: `new-materials/Construction of a Class of High-Dimensional Discrete Chaotic Systems.pdf`
   - Paper form: `x(k+1)=A x(k) mod 1`.
   - Digital mapping: keep the existing `ZangLinearModularMap`, using deterministic generated matrices for 4D--12D because the paper does not provide a full matrix table.

## Output Metrics

The runner computes `lambda_max`, `SE_mean`, `FE_mean`, `SI_mean`, and `PE_mean` for every supported method/dimension pair. Component-level values are saved in the detailed CSV.
""",
        encoding="utf-8",
    )
    return output_path


def copy_figures_to_out_paper(figure_paths: dict[str, dict[str, Path]], out_dir: str | Path) -> list[Path]:
    """Copy generated figures into the paper figure tree."""

    target_dir = ensure_directory(out_dir)
    copied: list[Path] = []
    for pair in figure_paths.values():
        for path in pair.values():
            destination = target_dir / path.name
            shutil.copy2(path, destination)
            copied.append(destination)
    return copied


def run_high_dim_literature_comparison(
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
    output_root: str | Path = "results/high_dim_literature_comparison",
    copy_to_out_paper: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    dims = list(range(4, 13)) if dimensions is None else [int(value) for value in dimensions]
    base_dir = ensure_directory(output_root)
    raw_dir = ensure_directory(base_dir / "raw")
    table_dir = ensure_directory(base_dir / "tables")
    figure_dir = ensure_directory(base_dir / "figures")
    diagnostics_dir = ensure_directory(base_dir / "diagnostics")

    algorithm_analysis = export_algorithm_analysis(base_dir / "algorithm_analysis_all.md")
    summary_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    support_rows = get_method_dimension_support(dims)

    for dimension in dims:
        initial_state = build_comparison_initial_state_u16(dimension)
        zang_matrix = build_zang_linear_modular_matrix(dimension)
        margins = np.abs(np.diag(zang_matrix)) - (np.sum(np.abs(zang_matrix), axis=1) - np.abs(np.diag(zang_matrix)))
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
            if progress:
                print(f"Evaluating n={dimension}, method={METHOD_SLUGS[method]}")
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

    config_csv = save_csv_rows(
        config_rows,
        raw_dir / "experiment_config.csv",
        fieldnames=[
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
        ],
    )
    support_csv = save_csv_rows(
        support_rows,
        raw_dir / "method_dimension_support.csv",
        fieldnames=[
            "method",
            "method_slug",
            "dimension",
            "supported",
            "public_dimension",
            "internal_dimension",
            "initialization_rule",
            "support_note",
        ],
    )
    summary_csv = save_csv_rows(
        summary_rows,
        raw_dir / "high_dim_all_methods_summary.csv",
        fieldnames=[
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
        ],
    )
    detailed_csv = save_csv_rows(
        detailed_rows,
        raw_dir / "high_dim_all_methods_detailed.csv",
        fieldnames=[
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
        ],
    )
    latex_table = export_comparison_table(summary_rows, table_dir / "high_dim_all_methods_comparison_table.tex")
    figure_paths = plot_metric_comparisons(summary_rows, figure_dir)
    copied_figures: list[Path] = []
    if copy_to_out_paper:
        copied_figures = copy_figures_to_out_paper(figure_paths, "out_paper/figures/high_dim_literature_comparison")

    diagnostics_path = diagnostics_dir / "high_dim_literature_comparison_metadata.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "dimensions": dims,
                "methods": METHODS,
                "initialization_rule": "x_i(0)=100*i in the 16-bit integer domain",
                "literature_algorithm": "All four high-dimensional literature methods plus proposed enhanced n-D map",
                "zang_diagonal": DEFAULT_ZANG_DIAGONAL,
                "zang_row_coupling_budget": DEFAULT_ZANG_ROW_COUPLING_BUDGET,
                "method_notes": METHOD_IMPLEMENTATION_NOTES,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "algorithm_analysis": algorithm_analysis,
        "config_csv": config_csv,
        "support_csv": support_csv,
        "summary_csv": summary_csv,
        "detailed_csv": detailed_csv,
        "latex_table": latex_table,
        "figure_paths": figure_paths,
        "copied_figures": copied_figures,
        "diagnostics_json": diagnostics_path,
        "summary_rows": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-dimensional literature comparison.")
    parser.add_argument("--dimensions", nargs="*", type=int, default=list(range(4, 13)))
    parser.add_argument("--output-root", default="results/high_dim_literature_comparison")
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
    parser.add_argument("--no-copy-to-out-paper", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_high_dim_literature_comparison(
        dimensions=args.dimensions,
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
        copy_to_out_paper=not args.no_copy_to_out_paper,
        progress=not args.quiet,
    )
    print(f"Saved algorithm analysis: {result['algorithm_analysis']}")
    print(f"Saved config CSV: {result['config_csv']}")
    print(f"Saved support CSV: {result['support_csv']}")
    print(f"Saved summary CSV: {result['summary_csv']}")
    print(f"Saved detailed CSV: {result['detailed_csv']}")
    print(f"Saved LaTeX table: {result['latex_table']}")
    for stem, pair in result["figure_paths"].items():
        print(f"Saved {stem}: {pair['pdf']} and {pair['eps']}")


if __name__ == "__main__":
    main()
