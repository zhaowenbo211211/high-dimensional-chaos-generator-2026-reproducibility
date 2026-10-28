# Maintainer: zhaowenbo
"""Export LaTeX table fragments from aggregated rev1 results.

Usage: python3 scripts/export_tables.py
Outputs land in staging/tables/ as .tex fragments (inspected before splicing).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1]
RESULTS = SUBSYSTEM_ROOT / "results"
STAGING = SUBSYSTEM_ROOT / "staging"

A_ORDER = ["The proposed method", "4D-NDCS", "URUK-4D-DCM", "4D-MHM",
           "4D-IDHNN", "4D-MPRC", "4D-MSLC", "4D-HCCDM"]
E_ORDER = ["proposed_rev1", "yao_complex_pascal", "sine_lattice_memristive",
           "cyclic_symmetric_logistic", "zang_linear_modular"]
E_DISPLAY = {
    "proposed_rev1": "The proposed method",
    "yao_complex_pascal": "Yao et al.",
    "sine_lattice_memristive": "He et al.",
    "cyclic_symmetric_logistic": "Zhang et al.",
    "zang_linear_modular": "Zang et al.",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt(value: str, digits: int) -> str:
    v = float(value)
    if abs(v) < 0.5 * 10 ** (-digits):  # avoid printing "-0.000000"
        v = 0.0
    return f"{v:.{digits}f}"


def _fmt_sci(value: str) -> str:
    v = float(value)
    if v == 0.0:
        return "0"
    mantissa, exponent = f"{v:.6e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def export_le_table(rows: list[dict]) -> str:
    lines = []
    by_name = {r["map_name"]: r for r in rows}
    for name in A_ORDER:
        r = by_name[name]
        lines.append(
            f"\t\t{name} & {_fmt(r['lyap_max'], 6)} & {_fmt(r['lyap_mean'], 6)} & "
            f"{_fmt(r['lyap_std'], 6)} & {int(r['renormalizations'])} & "
            f"{_fmt_sci(r['final_distance'])} & {r['is_chaotic']} \\\\"
        )
    return "\n".join(lines)


def export_metric_table(metric: str, digits: int, rows: list[dict]) -> str:
    lines = []
    by_name = {r["map_name"]: r for r in rows}
    for name in A_ORDER:
        r = by_name[name]
        comps = " & ".join(_fmt(r[f"{metric}_x{i}"], digits) for i in range(1, 5))
        lines.append(f"\t\t{name} & {comps} & {_fmt(r[f'{metric}_mean'], digits)} \\\\")
    return "\n".join(lines)


def export_dimension_table(rows: list[dict]) -> str:
    lines = []
    for r in sorted(rows, key=lambda r: int(r["dimension"])):
        lines.append(
            f"\t\t{int(r['dimension'])} & {_fmt(r['lyap_max'], 6)} & {_fmt(r['se_mean'], 6)} & "
            f"{_fmt(r['fe_mean'], 6)} & {_fmt(r['si_mean'], 6)} & {_fmt(r['pe_mean'], 6)} \\\\"
        )
    return "\n".join(lines)


def export_high_dim_archive_table(rows: list[dict]) -> str:
    """Archival full high-dimensional comparison table (NOT inserted into the paper)."""

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{High-dimensional literature comparison under unified 16-bit digital arithmetic (archival full table, not inserted into the main text).}",
        r"\label{tab:high-dimensional-all-literature-comparison-archive}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{clccccc}",
        r"\hline",
        r"$n$ & Method & $\lambda_{\max}$ & $\mathrm{SE}_{\mathrm{mean}}$ & $\mathrm{FE}_{\mathrm{mean}}$ & $\mathrm{SI}_{\mathrm{mean}}$ & $\mathrm{PE}_{\mathrm{mean}}$ \\",
        r"\hline",
    ]
    ordered = sorted(rows, key=lambda r: (int(r["dimension"]), E_ORDER.index(r["method_slug"])))
    for r in ordered:
        lines.append(
            f"{int(r['dimension'])} & {E_DISPLAY[r['method_slug']]} & "
            f"{_fmt(r['lambda_max'], 6)} & {_fmt(r['se_mean'], 6)} & {_fmt(r['fe_mean'], 6)} & "
            f"{_fmt(r['si_mean'], 6)} & {_fmt(r['pe_mean'], 6)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def main() -> None:
    out = STAGING / "tables"
    out.mkdir(parents=True, exist_ok=True)

    a_dir = RESULTS / "expA_4d_comparison"
    le_rows = _read_csv(a_dir / "le_comparison.csv")
    (out / "tab_le_comparison_rows.tex").write_text(export_le_table(le_rows) + "\n", encoding="utf-8")
    for metric, digits in (("se", 9), ("fe", 6), ("si", 6), ("pe", 6)):
        rows = _read_csv(a_dir / f"{metric}_comparison.csv")
        (out / f"tab_{metric}_comparison_rows.tex").write_text(
            export_metric_table(metric, digits, rows) + "\n", encoding="utf-8")

    d_rows = _read_csv(RESULTS / "expD_dimension_scalability" / "dimension_metrics_summary.csv")
    (out / "tab_dyna_per_high_dimensional_family_rows.tex").write_text(
        export_dimension_table(d_rows) + "\n", encoding="utf-8")

    e_rows = _read_csv(RESULTS / "expE_high_dim_literature" / "high_dim_all_methods_summary.csv")
    (out / "tab_high_dim_all_literature_archive.tex").write_text(
        export_high_dim_archive_table(e_rows) + "\n", encoding="utf-8")

    print("tables written to", out)


if __name__ == "__main__":
    main()
