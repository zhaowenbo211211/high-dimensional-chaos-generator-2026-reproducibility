# Maintainer: zhaowenbo
"""Phase B2: regroup ten frozen scientific panels into three vector figures.

Presentation only. Reuses the frozen Phase B selectors, array checks and visual
styles; no generator, experiment, scientific-metric or aggregation imports.
Standalone composites are composed from freshly redrawn native vector pages,
never from historical plots or screenshots. TeX uses the same native panels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import restyle_publication_figures as phase_b

ASSIGNMENTS = {
    "figure2_le": ("lyapunov_combined", "a"),
    "figure3_le": ("lyapunov_combined", "b"),
    **{f"figure2_{metric}": ("golden_range_complexity", letter)
       for metric, letter in zip(("se", "fe", "pe", "si"), "abcd")},
    **{f"figure3_{metric}": ("high_dim_complexity", letter)
       for metric, letter in zip(("se", "fe", "pe", "si"), "abcd")},
}
LE_HEIGHT_MM = 54.0
GOLDEN_LEGEND_HEIGHT_MM = 10.0


def check_text(fig, ax, legend=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    objects = [ax.xaxis.label, ax.yaxis.label]
    for axis in (ax.xaxis, ax.yaxis):
        low, high = axis.get_view_interval()
        objects += [label for value, label in zip(axis.get_ticklocs(), axis.get_ticklabels())
                    if low <= value <= high]
    if legend is not None:
        objects.append(legend)
    for obj in objects:
        box = obj.get_window_extent(renderer)
        assert box.x0 >= -0.5 and box.y0 >= -0.5
        assert box.x1 <= fig.bbox.width + 0.5 and box.y1 <= fig.bbox.height + 0.5


def draw_lyapunov(key, panel, width_mm, output, plt, np):
    """Both axes have precisely equal physical rectangles, independent scales."""
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(width_mm / 25.4, LE_HEIGHT_MM / 25.4))
    ax = fig.add_axes([14 / width_mm, 9 / LE_HEIGHT_MM,
                       (width_mm - 16.5) / width_mm, 30.5 / LE_HEIGHT_MM])
    group = key.split("_")[0]
    actual = []
    for index, series in enumerate(panel["series"]):
        style = ({"color": phase_b.COLORS[0], "linewidth": 1.1,
                  "label": "Proposed Mode II family"} if group == "figure2"
                 else phase_b.method_style(index))
        line, = ax.plot(series["x"], series["y"], **style)
        actual.append({"x": line.get_xdata(), "y": line.get_ydata()})
    check = phase_b.verify_arrays(panel["series"], actual, np)
    ax.set_xlabel(r"$k$" if group == "figure2" else r"$n$", fontsize=8.5, labelpad=2)
    ax.set_ylabel(r"$\lambda_{\max}$", fontsize=8.5, labelpad=3)
    ax.tick_params(direction="out", labelsize=7.5, length=2.6, width=0.6, pad=2.1)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#555B60")
    ax.grid(False)
    ax.set_xmargin(0.05 if group == "figure2" else 0.035)
    ax.set_xticks(list(range(0, 121, 20)) if group == "figure2" else list(range(4, 13)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    legend = fig.legend(*ax.get_legend_handles_labels(), loc="center",
                        bbox_to_anchor=(0.5, 47 / LE_HEIGHT_MM),
                        ncol=1 if group == "figure2" else 3, frameon=False,
                        fontsize=7.5, handlelength=2.4, handletextpad=0.45,
                        columnspacing=0.9, borderpad=0)
    check_text(fig, ax, legend)
    path = output / f"{key}.pdf"
    fig.savefig(path, metadata={"Title": key, "Creator": "paper2025 Phase B2 publication-only renderer",
                                "CreationDate": None, "ModDate": None})
    check.update({"path": str(path), "width_mm": width_mm, "height_mm": LE_HEIGHT_MM,
                  "axis_rectangle_mm": [14, 9, width_mm - 16.5, 30.5],
                  "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                  "axis_font_bp": 8.5, "tick_font_bp": 7.5, "legend_font_bp": 7.5})
    plt.close(fig)
    return check


def draw_golden_legend(width_mm, output, plt):
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(width_mm / 25.4, GOLDEN_LEGEND_HEIGHT_MM / 25.4))
    handles = [Line2D([], [], **{k: v for k, v in phase_b.component_style(i).items()
                               if k != "markevery"}) for i in range(5)]
    title = fig.text(0.5, 0.70, "Components and stored mean (a-d)", ha="center", va="center", fontsize=7.5)
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.26),
                        ncol=5, frameon=False, fontsize=7.5, handlelength=2.4,
                        handletextpad=0.45, columnspacing=1.1, borderpad=0)
    fig.canvas.draw()
    for text in (title, legend):
        box = text.get_window_extent(fig.canvas.get_renderer())
        assert box.x0 >= 0 and box.x1 <= fig.bbox.width and box.y0 >= 0 and box.y1 <= fig.bbox.height
    fig.savefig(output / "figure2_legend.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def compose_vector_pages(output, width_mm, plt):
    """Preserve all freshly rendered vector/text resources at physical scale 1."""
    import fitz
    mm = 72 / 25.4
    panel_width = 0.48 * width_mm
    groups = {
        "lyapunov_combined": (["figure2_le", "figure3_le"], LE_HEIGHT_MM, None,
                              ["Golden-Range parameter family", "High-dimensional comparison"]),
        "golden_range_complexity": ([f"figure2_{m}" for m in ("se", "fe", "pe", "si")], 43,
                                    ("figure2_legend", GOLDEN_LEGEND_HEIGHT_MM), ["SE", "FE", "PE", "SI"]),
        "high_dim_complexity": ([f"figure3_{m}" for m in ("se", "fe", "pe", "si")], 43,
                                ("figure3_legend", 6), ["SE mean", "FE mean", "PE mean", "SI mean"]),
    }
    geometry = {}
    for name, (keys, height, legend, captions) in groups.items():
        label_height, row_gap = 5.0, 2.0
        top = legend[1] + 1 if legend else 0
        rows = len(keys) // 2
        total_height = top + rows * (height + label_height) + (rows - 1) * row_gap
        composite = fitz.open()
        page = composite.new_page(width=width_mm * mm, height=total_height * mm)
        placements = []
        if legend:
            with fitz.open(output / f"{legend[0]}.pdf") as source:
                page.show_pdf_page(fitz.Rect(0, 0, width_mm * mm, legend[1] * mm), source, 0)
        for index, key in enumerate(keys):
            x = 0 if index % 2 == 0 else width_mm - panel_width
            y = top + (index // 2) * (height + label_height + row_gap)
            with fitz.open(output / f"{key}.pdf") as source:
                page.show_pdf_page(fitz.Rect(x * mm, y * mm, (x + panel_width) * mm,
                                             (y + height) * mm), source, 0)
            # A native Matplotlib text strip supplies embedded STIX panel labels.
            with tempfile.TemporaryDirectory(prefix="paper2025_b2_label_") as tmp:
                fig = plt.figure(figsize=(panel_width / 25.4, label_height / 25.4))
                fig.text(0.5, 0.5, f"({chr(97 + index)}) {captions[index]}",
                         ha="center", va="center", fontsize=8.5)
                strip = Path(tmp) / "label.pdf"
                fig.savefig(strip, metadata={"CreationDate": None, "ModDate": None})
                plt.close(fig)
                with fitz.open(strip) as source:
                    page.show_pdf_page(fitz.Rect(x * mm, (y + height) * mm,
                                                 (x + panel_width) * mm,
                                                 (y + height + label_height) * mm), source, 0)
            placements.append({"key": key, "panel": chr(97 + index),
                               "rectangle_mm": [x, y, panel_width, height]})
        composite.set_metadata({"title": name, "creator": "paper2025 Phase B2 native-vector composition"})
        composite.save(output / f"{name}.pdf", garbage=4, deflate=True)
        composite.close()
        geometry[name] = {"width_mm": width_mm, "height_mm": total_height,
                          "layout": "1x2" if rows == 1 else "2x2", "panels": placements}
    return geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--width-mm", type=float, default=159.2)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper2025_b2_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#25292D", "axes.labelcolor": "#25292D",
        "xtick.color": "#25292D", "ytick.color": "#25292D",
        "path.simplify": False, "agg.path.chunksize": 0, "figure.dpi": 150,
    })
    rows = phase_b.load_frozen_rows()
    panels = phase_b.select_series(rows)
    old = json.loads((phase_b.ROOT / "submission_csf/review/phase_b_scientific_plot_restyling/data_integrity.json").read_text())
    # Exact dictionary equality checks names, fields, slugs, order, hashes and x/y.
    assert panels == old["data"], "Phase B stored arrays/mapping differ"
    legacy = phase_b.compare_legacy_arrays(rows, panels, plt, np)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    for key, panel in panels.items():
        if key.endswith("_le"):
            check = draw_lyapunov(key, panel, args.width_mm * 0.48, output, plt, np)
        else:
            check = phase_b.make_panel(key, panel, args.width_mm * 0.48, output, plt, np)
        check["assignment"] = ASSIGNMENTS[key]
        checks[key] = check
    draw_golden_legend(args.width_mm, output, plt)
    phase_b.make_legend("figure3", args.width_mm, output, plt)
    geometry = compose_vector_pages(output, args.width_mm, plt)
    manifest = {"data": panels, "sources": phase_b.SOURCE_HASHES,
                "phase_b_data_dictionary_identical": True, "legacy_array_checks": legacy,
                "rendered_array_checks": checks, "assignments": ASSIGNMENTS,
                "composites": geometry, "DATA VALUES CHANGED": "NO",
                "SCIENTIFIC METRICS RECOMPUTED": "NO", "EXPERIMENTS RERUN": "NO",
                "series_count": 46, "point_count": 2724,
                "environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                                "font": "STIXGeneral", "mathtext": "stix", "pdf_fonttype": 42}}
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest, indent=2))
    print("B2: 10 panels / 46 curves / 2724 points unchanged; 1x2 + 2x2 + 2x2, 15 vector PDFs.")


if __name__ == "__main__":
    main()
