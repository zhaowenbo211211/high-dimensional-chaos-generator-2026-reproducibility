# Maintainer: zhaowenbo
"""Run official NIST STS evaluation for system-level 4-direction PRNG outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from metrics.nist_randomness import (
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
)
from metrics.prng_system_level_official import (
    OFFICIAL_STS_CATEGORY_ORDER,
    build_system_level_schemes,
    export_payload_to_ascii,
    generate_system_level_payload,
    parse_official_summary_rows_by_category,
)
from utils.io_utils import ensure_directory, save_csv_rows
from utils.official_nist_runner import run_official_nist_assess


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&")


def _is_category_level_pass(row: dict[str, object]) -> bool | None:
    passed = row.get("passed")
    if passed is True:
        return True
    if passed is False:
        total_rows = int(row.get("total_report_rows", 0) or 0)
        flagged_rows = int(row.get("flagged_subtests", 0) or 0)
        if total_rows > 1 and flagged_rows < total_rows:
            return True
        return False
    return None


def _format_table_status(row: dict[str, object]) -> str:
    category_pass = _is_category_level_pass(row)
    strict_pass = row.get("passed")
    if category_pass is True and strict_pass is True:
        return "Pass"
    if category_pass is True and strict_pass is False:
        return "Pass*"
    if category_pass is False:
        return "Fail*"
    return "Error"


def _error_category_rows(message: str) -> list[dict[str, object]]:
    return [
        {
            "test_name": test_name,
            "passed": None,
            "flagged_subtests": "",
            "total_report_rows": 0,
            "notes": message,
        }
        for test_name in OFFICIAL_STS_CATEGORY_ORDER
    ]


def _scheme_rank_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["passed_tests"]),
        -int(row["failed_tests"]),
        1 if str(row["scheme_name"]) == "plain_concat_linear_mixing" else 0,
    )


def export_system_level_summary_table(
    category_rows: list[dict[str, object]],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    lookup = {str(row["test_name"]): row for row in category_rows}
    lines = [
        r"\begin{tabular}{cc}",
        r"\hline",
        r"Test & Plain concat + ROTL/XOR mixing \\",
        r"\hline",
    ]
    for test_name in OFFICIAL_STS_CATEGORY_ORDER:
        status = _format_table_status(lookup[test_name])
        lines.append(f"{_latex_escape(test_name)} & {status} \\\\")
    note = (
        "Official STS configuration: 100 sequences of length $10^6$, warm-up = 1000; "
        "the PRNG block is $y_t=u_t\\oplus\\mathrm{ROTL}_{64}(u_{t-1},17)"
        "\\oplus\\mathrm{ROTL}_{64}(u_{t-2},43)$, where "
        "$u_t=x_1(t)\\|x_2(t)\\|x_3(t)\\|x_4(t)$; "
        "`Pass*' means the high-level category is acceptable while one or more subtests are flagged."
    )
    lines.extend([r"\hline", rf"\multicolumn{{2}}{{l}}{{{note}}} \\", r"\hline", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_nist_prng_system_level(
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    nist_root: str | Path = "nist_sp_800_22",
    output_root: str | Path = "results/nist_prng_system_level",
    progress: bool = True,
) -> dict[str, Any]:
    base_dir = ensure_directory(output_root)
    raw_dir = ensure_directory(base_dir / "raw")
    official_reports_dir = ensure_directory(base_dir / "official_reports")
    tables_dir = ensure_directory(base_dir / "tables")
    diagnostics_dir = ensure_directory(base_dir / "diagnostics")
    total_bit_length = int(sequence_bit_length) * int(sequence_count)

    summary_rows: list[dict[str, Any]] = []
    report_dirs: dict[str, str] = {}
    category_rows_for_table: list[dict[str, object]] = []

    for scheme in build_system_level_schemes():
        if progress:
            print(f"Official system-level PRNG STS: {scheme.scheme_name}")
        payload = generate_system_level_payload(
            scheme=scheme,
            sequence_bit_length=sequence_bit_length,
            sequence_count=sequence_count,
            warmup_steps=warmup_steps,
            diagnostics_dir=diagnostics_dir / scheme.scheme_name,
        )
        input_ascii_path = export_payload_to_ascii(
            payload=payload,
            bit_length=total_bit_length,
            output_path=diagnostics_dir / scheme.scheme_name / f"{scheme.scheme_name}_{total_bit_length}bit_ascii.txt",
        )
        try:
            artifacts = run_official_nist_assess(
                nist_root=nist_root,
                input_file=input_ascii_path,
                sequence_bit_length=sequence_bit_length,
                sequence_count=sequence_count,
                label=scheme.scheme_name,
                output_root=official_reports_dir / scheme.scheme_name,
                input_mode="ascii",
                selected_tests=None,
                timeout_seconds=180.0,
            )
            report_dirs[scheme.scheme_name] = artifacts.run_dir
            summary_data = __import__("json").loads(Path(artifacts.summary_json).read_text(encoding="utf-8"))
            category_rows = parse_official_summary_rows_by_category(summary_data.get("tests", []))
            passed_tests = sum(1 for row in category_rows if row["passed"] is True)
            failed_names = [str(row["test_name"]) for row in category_rows if row["passed"] is False]
            error_message = ""
            official_report_path = artifacts.final_report
        except Exception as exc:
            category_rows = _error_category_rows(str(exc))
            passed_tests = 0
            failed_names = list(OFFICIAL_STS_CATEGORY_ORDER)
            error_message = str(exc)
            official_report_path = str(official_reports_dir / scheme.scheme_name)

        category_rows_for_table = category_rows
        summary_rows.append(
            {
                "scheme_name": scheme.scheme_name,
                "preprocessing_description": scheme.preprocessing_description,
                "total_tests": len(OFFICIAL_STS_CATEGORY_ORDER),
                "passed_tests": passed_tests,
                "failed_tests": len(OFFICIAL_STS_CATEGORY_ORDER) - passed_tests,
                "failed_test_names": "; ".join(failed_names),
                "official_sts_report_path": official_report_path,
                "error_message": error_message,
                "is_best": False,
            }
        )
        save_csv_rows(
            category_rows,
            raw_dir / f"{scheme.scheme_name}_official_category_summary.csv",
            fieldnames=["test_name", "passed", "flagged_subtests", "total_report_rows", "notes"],
        )

    best_row = max(summary_rows, key=_scheme_rank_key)
    for row in summary_rows:
        row["is_best"] = row["scheme_name"] == best_row["scheme_name"]

    summary_csv = save_csv_rows(
        summary_rows,
        raw_dir / "system_level_summary.csv",
        fieldnames=[
            "scheme_name",
            "preprocessing_description",
            "total_tests",
            "passed_tests",
            "failed_tests",
            "failed_test_names",
            "official_sts_report_path",
            "error_message",
            "is_best",
        ],
    )
    latex_table = export_system_level_summary_table(category_rows_for_table, tables_dir / "system_level_nist_summary.tex")
    metadata_path = diagnostics_dir / "system_level_prng_metadata.txt"
    metadata_path.write_text(
        "\n".join(
            [
                "Target: representative enhanced 4D instance, system-level PRNG output only",
                "Retained scheme: plain_concat_linear_mixing",
                "Stage 0: u_t = x1(t) || x2(t) || x3(t) || x4(t)",
                "Stage 1: y_t = u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43)",
                f"Sequence bit length: {sequence_bit_length}",
                f"Sequence count: {sequence_count}",
                f"Total bit length: {total_bit_length}",
                f"Warm-up discard: {warmup_steps}",
                "Tail post-processing: none",
                "Official suite: bundled official NIST SP 800-22 STS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_csv,
        "latex_table": latex_table,
        "official_report_dirs": report_dirs,
        "summary_rows": summary_rows,
        "metadata_path": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official STS evaluation for system-level 4-direction PRNG outputs.")
    parser.add_argument("--sequence-bit-length", type=int, default=DEFAULT_NIST_SEQUENCE_BIT_LENGTH)
    parser.add_argument("--sequence-count", type=int, default=DEFAULT_NIST_SEQUENCE_COUNT)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_NIST_WARMUP_STEPS)
    parser.add_argument("--nist-root", default="nist_sp_800_22")
    parser.add_argument("--output-root", default="results/nist_prng_system_level")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_nist_prng_system_level(
        sequence_bit_length=args.sequence_bit_length,
        sequence_count=args.sequence_count,
        warmup_steps=args.warmup_steps,
        nist_root=args.nist_root,
        output_root=args.output_root,
        progress=not args.quiet,
    )
    print(f"Saved summary CSV: {result['summary_csv']}")
    print(f"Saved LaTeX table: {result['latex_table']}")
    for scheme_name, report_dir in result["official_report_dirs"].items():
        print(f"{scheme_name}: {report_dir}")
    for row in result["summary_rows"]:
        print(
            f"{row['scheme_name']}: passed {row['passed_tests']}/{row['total_tests']} "
            f"(failed: {row['failed_test_names'] or 'none'})"
        )


if __name__ == "__main__":
    main()
