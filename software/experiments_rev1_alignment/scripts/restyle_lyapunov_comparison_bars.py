# Maintainer: zhaowenbo
"""Publication-only update of Lyapunov panel (b): high-contrast grouped bars.

Reads the six Phase A frozen CSVs through the verified Phase B selectors;
compares their complete selected dictionaries with the retained B2 manifest.
Only expE lambda_max values are drawn. No experiment, metric calculation,
aggregation, smoothing, fitting, or Table 7 data is invoked.

This targeted renderer supersedes only figure3_le.pdf from the B2 renderer.
Its companion lyapunov_combined.pdf retains the unchanged left vector panel.
Use a staging output directory; deployment and the official LaTeX build are
separate steps so verification precedes replacement of publication assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import restyle_publication_figures as phase_b

METHOD_STYLES = (
    ("proposed_rev1", "Proposed", "#0072B2", ""),
    ("yao_complex_pascal", "Yao et al.", "#E69F00", "//"),
    ("sine_lattice_memristive", "He et al.", "#009E73", "\\\\"),
    ("cyclic_symmetric_logistic", "Zhang et al.", "#AA3377", "xx"),
    ("zang_linear_modular", "Zang et al.", "#999999", ".."),
)
FULL_WIDTH_MM = 159.2
PANEL_WIDTH_MM = FULL_WIDTH_MM * 0.48
PANEL_HEIGHT_MM = 54.0
BAR_WIDTH = 0.14
METHOD_SPACING = 0.18
EDGE_COLOR = "#333333"
MM_TO_BP = 72 / 25.4


def draw_panel(panel, output, plt, np):
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4))
    ax = fig.add_axes([14 / PANEL_WIDTH_MM, 9 / PANEL_HEIGHT_MM,
                       (PANEL_WIDTH_MM - 16.5) / PANEL_WIDTH_MM, 30.5 / PANEL_HEIGHT_MM])
    handles, checks = [], []
    for i, (series, style) in enumerate(zip(panel["series"], METHOD_STYLES, strict=True)):
        slug, label, color, hatch = style
        assert series["method_slug"] == slug and series["display_label"] == label
        n, y = np.asarray(series["x"]), np.asarray(series["y"])
        assert np.array_equal(n, np.arange(4, 13))
        offset = (i - 2) * METHOD_SPACING
        bars = ax.bar(n + offset, y, width=BAR_WIDTH, color=color,
                      edgecolor=EDGE_COLOR, linewidth=0.4, hatch=hatch)
        actual_y = np.array([bar.get_height() for bar in bars])
        actual_centers = np.array([bar.get_x() + bar.get_width() / 2 for bar in bars])
        assert np.array_equal(actual_y, y)
        assert np.allclose(actual_centers, n + offset, atol=1e-14, rtol=0)
        assert len(bars) == 9
        handles.append(Patch(facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.4,
                             hatch=hatch, label=label))
        checks.append({
            "method_slug": slug, "series_name": series["name"], "display_label": label,
            "field": series["field"], "x": n.tolist(), "y": y.tolist(),
            "x_exactly_equal_to_b2": True, "y_exactly_equal_to_b2": True,
            "rendered_bar_heights_exactly_equal": True, "point_count": len(bars),
            "xy_sha256": hashlib.sha256(np.column_stack([n, y]).astype("<f8").tobytes()).hexdigest(),
            "color": color, "hatch": hatch, "display_offset_within_integer_n_group": offset,
        })
    # Group centres are the original integer dimensions. Offsets distinguish
    # methods within a dimension; they do not encode fractional dimensions.
    ax.set_xlim(3.45, 12.55)
    ax.set_ylim(0, 12)
    ax.set_xticks(np.arange(4, 13))
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_xlabel(r"$n$", fontsize=8.5, labelpad=2)
    ax.set_ylabel(r"$\lambda_{\max}$", fontsize=8.5, labelpad=3)
    ax.tick_params(direction="out", labelsize=7.5, length=2.6, width=0.6, pad=2.1)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#555B60")
    ax.grid(False)
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 47 / PANEL_HEIGHT_MM),
                        ncol=3, frameon=False, fontsize=7.5, handlelength=2.4,
                        handletextpad=0.45, columnspacing=0.9, borderpad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in [legend, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        box = item.get_window_extent(renderer)
        assert box.x0 >= -0.5 and box.y0 >= -0.5
        assert box.x1 <= fig.bbox.width + 0.5 and box.y1 <= fig.bbox.height + 0.5
    path = output / "figure3_le.pdf"
    fig.savefig(path, metadata={"Title": "High-dimensional Lyapunov comparison: grouped bars",
                               "Creator": "paper2025 targeted frozen-data publication renderer",
                               "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return checks


def compose(left, right, output, plt):
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=FULL_WIDTH_MM * MM_TO_BP, height=59 * MM_TO_BP)
    labels = ["(a) Golden-Range parameter family", "(b) High-dimensional generator comparison"]
    for i, panel in enumerate([left, right]):
        x = 0 if i == 0 else FULL_WIDTH_MM - PANEL_WIDTH_MM
        with fitz.open(panel) as source:
            page.show_pdf_page(fitz.Rect(x * MM_TO_BP, 0, (x + PANEL_WIDTH_MM) * MM_TO_BP,
                                         PANEL_HEIGHT_MM * MM_TO_BP), source, 0)
        with tempfile.TemporaryDirectory(prefix="paper2025_bar_label_") as temp:
            fig = plt.figure(figsize=(PANEL_WIDTH_MM / 25.4, 5 / 25.4))
            fig.text(0.5, 0.5, labels[i], fontsize=8.5, ha="center", va="center")
            strip = Path(temp) / "label.pdf"
            fig.savefig(strip, metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)
            with fitz.open(strip) as source:
                page.show_pdf_page(fitz.Rect(x * MM_TO_BP, 54 * MM_TO_BP,
                                             (x + PANEL_WIDTH_MM) * MM_TO_BP, 59 * MM_TO_BP), source, 0)
    doc.set_metadata({"title": "Lyapunov evaluation: 1x2, frozen-data grouped comparison bars",
                      "creator": "paper2025 targeted publication vector composition"})
    doc.save(output / "lyapunov_combined.pdf", garbage=4, deflate=True)
    doc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper2025_bars_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import fitz

    matplotlib.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#25292D", "axes.labelcolor": "#25292D", "xtick.color": "#25292D",
        "ytick.color": "#25292D", "hatch.color": EDGE_COLOR, "hatch.linewidth": 0.4,
        "path.simplify": False, "figure.dpi": 160,
    })
    rows = phase_b.load_frozen_rows()
    panels = phase_b.select_series(rows)
    previous_path = phase_b.ROOT / "submission_csf/review/phase_b2_lyapunov_restructuring/data_integrity.json"
    previous = json.loads(previous_path.read_text())
    assert panels == previous["data"], "SOURCE_PROVENANCE_BLOCKED: selected arrays/mappings differ"
    left = phase_b.ROOT / "submission_csf/figures/scientific_publication_b2/figure2_le.pdf"
    left_hash = phase_b.sha256(left)
    checks = draw_panel(panels["figure3_le"], output, plt, np)
    compose(left, output / "figure3_le.pdf", output, plt)
    assert phase_b.sha256(left) == left_hash
    for name in ("figure3_le", "lyapunov_combined"):
        with fitz.open(output / f"{name}.pdf") as pdf:
            assert not pdf[0].get_images(), "Raster content in publication output"
            pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(output / f"{name}.png")
    manifest = {
        "source_hashes": phase_b.SOURCE_HASHES, "array_reference": str(previous_path.relative_to(phase_b.ROOT)),
        "all_ten_panel_selected_arrays_exactly_equal_to_b2": True,
        "target_source": panels["figure3_le"]["source"], "target_field": "lambda_max",
        "series_count": len(checks), "data_point_count": sum(s["point_count"] for s in checks),
        "series": checks, "left_native_sha256_unchanged": left_hash,
        "physical_geometry_mm": {"combined": [FULL_WIDTH_MM, 59],
                                  "native_panel": [PANEL_WIDTH_MM, PANEL_HEIGHT_MM],
                                  "axes_from_bottom_left": [14, 9, PANEL_WIDTH_MM - 16.5, 30.5]},
        "axis_font_bp": 8.5, "tick_legend_font_bp": 7.5, "axis_y_range": [0, 12],
        "display_encoding": "Original integer n group centres; method offsets only within each group",
        "DATA VALUES CHANGED": "NO", "SCIENTIFIC METRICS RECOMPUTED": "NO", "EXPERIMENTS RERUN": "NO",
    }
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2))
    print("PASS: five methods, 45 bar heights equal frozen values; six hashes match; left panel unchanged.")


if __name__ == "__main__":
    main()
