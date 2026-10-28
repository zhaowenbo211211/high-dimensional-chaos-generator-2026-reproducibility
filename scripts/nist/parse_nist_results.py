#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Parse NIST SP 800-22 finalAnalysisReport.txt conservatively."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_report(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.match(r"^\s*\d", line):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        official_result = "no_marker"
        test_name_idx = 12
        if parts[12] == "*":
            official_result = "official_marker"
            test_name_idx = 13
        if len(parts) <= test_name_idx:
            continue
        rows.append(
            {
                "test_name": parts[test_name_idx],
                "uniformity_p_value": parts[10],
                "pass_proportion": parts[11],
                "official_result": official_result,
                "notes": line.rstrip(),
            }
        )
    return rows


def write_outputs(rows: list[dict[str, str]], output_dir: Path, report_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nist_summary.csv"
    md_path = output_dir / "nist_summary.md"
    tex_path = output_dir / "nist_summary.tex"
    interp_path = output_dir / "NIST_RESULT_INTERPRETATION.md"

    fields = ["test_name", "uniformity_p_value", "pass_proportion", "official_result", "notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# NIST Summary\n\n")
        f.write(f"Source report: `{report_path}`\n\n")
        marker_rows = [row for row in rows if row["official_result"] != "no_marker"]
        f.write(f"Parsed rows: {len(rows)}\n\n")
        f.write(f"Official marker rows: {len(marker_rows)}\n\n")
        f.write("| Test name | Uniformity p-value | Pass proportion | Official result | Notes |\n")
        f.write("|---|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['test_name']} | {row['uniformity_p_value']} | {row['pass_proportion']} | "
                f"{row['official_result']} | `{row['notes']}` |\n"
            )

    with tex_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated conservative NIST summary table\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{NIST SP 800-22 summary for PRNG-371 generated with warmup=2000. Verify against the raw official report before final submission.}\n")
        f.write("\\label{tab:prng371_nist_warmup2000}\n")
        f.write("\\begin{tabular}{lll}\n\\toprule\n")
        f.write("Test & Uniformity $p$-value & Pass proportion \\\\\n\\midrule\n")
        for row in rows:
            name = row["test_name"].replace("_", "\\_")
            marker = " $^{*}$" if row["official_result"] != "no_marker" else ""
            f.write(f"{name}{marker} & {row['uniformity_p_value']} & {row['pass_proportion']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    marker_rows = [row for row in rows if row["official_result"] != "no_marker"]
    marker_tests = sorted({row["test_name"] for row in marker_rows})
    status = "official_markers_present" if marker_rows else "no_official_markers_found"
    interp_path.write_text(
        "\n".join(
            [
                "# NIST Result Interpretation",
                "",
                f"Source report: `{report_path}`",
                f"Parsed rows: {len(rows)}",
                f"Status: `{status}`",
                f"Official marker rows: {len(marker_rows)}",
                f"Tests with official markers: `{', '.join(marker_tests) if marker_tests else 'none'}`",
                "",
                "Parsing policy:",
                "",
                "- This parser preserves the official finalAnalysisReport lines as notes.",
                "- Rows marked with `*` by the official report are preserved as official markers.",
                "- A marker in a multi-row family such as Non-overlapping Template should not automatically be interpreted as a whole-family failure.",
                "- Rows without `*` are reported as having no official marker; this parser does not add stronger claims.",
                "- If rows are missing or the report format differs, inspect the raw report manually.",
                "- Non-overlapping Template and Random Excursions families contain multiple subtests and should be discussed conservatively.",
                "- Passing NIST statistical tests is not a proof of cryptographic security.",
                "- These results are PC golden PRNG-371 randomness evidence, not FPGA physical validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_report(args.report)
    write_outputs(rows, args.output_dir, args.report)
    print(f"parsed_rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
