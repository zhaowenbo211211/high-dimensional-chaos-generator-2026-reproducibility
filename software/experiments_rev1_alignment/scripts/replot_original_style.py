# Maintainer: zhaowenbo
"""Re-plot the 15 paper figure groups with the copied original plotting functions.

Data policy: the only numeric sources are the validated rev1 result files under
results/ (expB_golden_range, expC_si_diagnostics, expE_high_dim_literature).
Nothing is recomputed, resampled, smoothed, or dropped; the adapters below only
rename/select fields so the copied original plotting functions receive exactly
the record structure they were written for.

Outputs are written to results/figures/ in the same layout the paper uses;
deployment into ../figures/ (incl. epstopdf regeneration of the
*-eps-converted-to.pdf files) is done by the deploy step at the end.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # experiments_rev1_alignment/
PAPER = REPO.parents[0]  # paper2025/
sys.path.insert(0, str(REPO / "scripts"))

from original_plotting.golden_range import (  # noqa: E402
    plot_enhanced_golden_range_fe,
    plot_enhanced_golden_range_le,
    plot_enhanced_golden_range_pe,
    plot_enhanced_golden_range_scale_index,
    plot_enhanced_golden_range_se,
)
from original_plotting.high_dim import (  # noqa: E402
    METHOD_CYCLIC,
    METHOD_PROPOSED,
    METHOD_SINE_MEMRISTIVE,
    METHOD_YAO,
    METHOD_ZANG,
    plot_metric_comparisons,
)
from original_plotting.si_diagnostics import (  # noqa: E402
    plot_enhanced_scalogram_overview,
    plot_enhanced_si_component_curve,
)

RESULTS = REPO / "results"
FIG_OUT = RESULTS / "figures"

# New-data method_slug -> the legacy full method name the original plot function
# filters on. Legend text stays the paper-facing one (METHOD_SHORT_LABELS).
SLUG_TO_LEGACY_METHOD = {
    "proposed_rev1": METHOD_PROPOSED,
    "yao_complex_pascal": METHOD_YAO,
    "sine_lattice_memristive": METHOD_SINE_MEMRISTIVE,
    "cyclic_symmetric_logistic": METHOD_CYCLIC,
    "zang_linear_modular": METHOD_ZANG,
}


def load_golden_range_csv(metric: str) -> list[dict[str, str]]:
    path = RESULTS / "expB_golden_range" / f"golden_range_{metric}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 119, f"{path}: expected 119 golden-range rows, got {len(records)}"
    return records


def golden_range_figures() -> dict[str, dict[str, Path]]:
    out = FIG_OUT / "camp_dynamic"
    paths = {
        "le": plot_enhanced_golden_range_le(load_golden_range_csv("le"), output_dir=out),
        "se": plot_enhanced_golden_range_se(load_golden_range_csv("se"), output_dir=out),
        "fe": plot_enhanced_golden_range_fe(load_golden_range_csv("fe"), output_dir=out),
        "pe": plot_enhanced_golden_range_pe(load_golden_range_csv("pe"), output_dir=out),
        "si": plot_enhanced_golden_range_scale_index(
            load_golden_range_csv("si"), output_dir=out / "scale_index"
        ),
    }
    return paths


def si_diagnostics_figures() -> dict[str, dict[str, Path]]:
    json_path = RESULTS / "expC_si_diagnostics" / "enhanced_scale_index_diagnostics.json"
    records = json.loads(json_path.read_text(encoding="utf-8"))
    out = FIG_OUT / "camp_dynamic" / "scale_index" / "diagnostics"
    paths = {"scalogram": plot_enhanced_scalogram_overview(records, output_dir=out)}
    for component_index in (1, 2, 3, 4):
        paths[f"si_x{component_index}"] = plot_enhanced_si_component_curve(
            records, component_index=component_index, output_dir=out
        )
    return paths


def high_dim_figures() -> dict[str, dict[str, Path]]:
    path = RESULTS / "expE_high_dim_literature" / "high_dim_all_methods_summary.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 45, f"{path}: expected 45 summary rows, got {len(rows)}"
    records: list[dict[str, str]] = []
    for row in rows:
        slug = row["method_slug"]
        assert slug in SLUG_TO_LEGACY_METHOD, f"unexpected method_slug {slug!r}"
        record = dict(row)
        record["method"] = SLUG_TO_LEGACY_METHOD[slug]
        records.append(record)
    return plot_metric_comparisons(records, output_dir=FIG_OUT / "high_dim_literature_comparison")


def deploy_to_paper() -> list[Path]:
    """Copy the new figures into paper2025/figures/ and refresh epstopdf outputs."""

    deployed: list[Path] = []
    camp_src = FIG_OUT / "camp_dynamic"
    camp_dst = PAPER / "figures" / "camp_dynamic"
    for eps in sorted(camp_src.rglob("*.eps")):
        rel = eps.relative_to(camp_src)
        dst_eps = camp_dst / rel
        dst_eps.parent.mkdir(parents=True, exist_ok=True)
        dst_eps.write_bytes(eps.read_bytes())
        # Refresh the file LaTeX actually loads for .eps includes.
        converted = dst_eps.with_name(dst_eps.name.replace(".eps", "-eps-converted-to.pdf"))
        subprocess.run(["epstopdf", str(dst_eps), "--outfile", str(converted)], check=True)
        deployed.extend([dst_eps, converted])

    hd_src = FIG_OUT / "high_dim_literature_comparison"
    hd_dst = PAPER / "figures" / "high_dim_literature_comparison"
    for src in sorted(hd_src.iterdir()):
        if src.suffix in (".pdf", ".eps"):
            dst = hd_dst / src.name
            dst.write_bytes(src.read_bytes())
            deployed.append(dst)
    return deployed


def main() -> None:
    paths: dict[str, object] = {}
    paths.update(golden_range_figures())
    paths.update(si_diagnostics_figures())
    paths.update(high_dim_figures())
    print("rendered figure groups:")
    for key, value in sorted(paths.items()):
        print(f"  {key}: {value}")
    deployed = deploy_to_paper()
    print(f"deployed {len(deployed)} files into {PAPER / 'figures'}")
    for path in deployed:
        print(f"  {path.relative_to(PAPER)}")


if __name__ == "__main__":
    main()
