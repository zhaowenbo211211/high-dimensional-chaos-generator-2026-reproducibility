# Maintainer: zhaowenbo
"""Phase B: presentation-only Figure 2/3 renderer for the frozen rev1 CSVs.

Provenance: PHASE_A_SCIENTIFIC_DISCLOSURE_AND_READABILITY.md, section 6.
The original publication plotting functions are run in an isolated temporary
working directory to capture their *main-axis* x/y arrays. No experiment,
generator, aggregator, parameter synthesis, or metric module is imported.
Stored component means are read directly; they are never recomputed.

The output directory is mandatory. This program never deploys to the paper,
modifies results, or renders the unrelated SI diagnostics. PDF page boxes are
fixed physical sizes: no tight-crop followed by uncontrolled LaTeX scaling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments_rev1_alignment/results"
LEGACY = ROOT / "experiments_rev1_alignment/scripts/original_plotting"
SOURCE_HASHES = {
    "expB_golden_range/golden_range_le.csv": "a9603d8bf3a3d4d8eb8f02cbd5cdd756133a5219c879fb4c8cbaea4066ed0694",
    "expB_golden_range/golden_range_se.csv": "0d7aa60a7b3c25c95c70584929edfd2b26bf2e7d1c3109ebae53acd7c1bf542a",
    "expB_golden_range/golden_range_fe.csv": "4e1433f472173cdd470d0bcedc9e462983bef755589f48724d2b23fb73d52458",
    "expB_golden_range/golden_range_pe.csv": "579f6ded60d711237fba3e68b1b1eb5437e8231658126769c912eb408a1377eb",
    "expB_golden_range/golden_range_si.csv": "18ac5c82836596d229129c4409773f74c205a67ffff7e0857101337a47288091",
    "expE_high_dim_literature/high_dim_all_methods_summary.csv": "4915633f7a590b455f86004e58d18976a7da2b5716c5557a73c282fcf0aa8744",
}
METRICS = ("le", "se", "fe", "pe", "si")
METHODS = (
    ("proposed_rev1", "Proposed", "The proposed method", "o", "-"),
    ("yao_complex_pascal", "Yao et al.", "Yao et al.'s method", "^", "--"),
    ("sine_lattice_memristive", "He et al.", "He et al.'s method", "D", "-."),
    ("cyclic_symmetric_logistic", "Zhang et al.", "Zhang et al.'s method", "v", ":"),
    ("zang_linear_modular", "Zang et al.", "Zang et al.'s method", "s", (0, (5, 2))),
)
COLORS = ("#3B6F91", "#B07837", "#3B8876", "#9A6488", "#6F7276")
COMPONENTS = (("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"))
HEIGHT_MM = 43.0
LABEL_BP = 8.5
TICK_BP = 7.5
LEGEND_BP = 7.5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_rows():
    rows = {}
    for rel, expected in SOURCE_HASHES.items():
        path = RESULTS / rel
        if sha256(path) != expected:
            raise RuntimeError(f"SOURCE_PROVENANCE_BLOCKED: frozen hash differs: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows[rel] = list(csv.DictReader(handle))
    for metric in METRICS:
        records = rows[f"expB_golden_range/golden_range_{metric}.csv"]
        assert len(records) == 119
        assert sorted(int(r["group_index"]) for r in records) == list(range(1, 120))
    records = rows["expE_high_dim_literature/high_dim_all_methods_summary.csv"]
    assert len(records) == 45
    assert set(r["method_slug"] for r in records) == set(m[0] for m in METHODS)
    for slug, *_ in METHODS:
        assert sorted(int(r["dimension"]) for r in records if r["method_slug"] == slug) == list(range(4, 13))
    return rows


def select_series(rows):
    panels = {}
    for metric in METRICS:
        source = f"expB_golden_range/golden_range_{metric}.csv"
        records = sorted(rows[source], key=lambda r: int(r["group_index"]))
        fields = ["lyap_max"] if metric == "le" else [f"{metric}_x{i}" for i in range(1, 5)] + [f"{metric}_mean"]
        panels[f"figure2_{metric}"] = {
            "source": source, "source_sha256": SOURCE_HASHES[source],
            "series": [{"name": field, "x": [int(r["group_index"]) for r in records],
                        "y": [float(r[field]) for r in records]} for field in fields],
        }
    source = "expE_high_dim_literature/high_dim_all_methods_summary.csv"
    for metric in METRICS:
        field = "lambda_max" if metric == "le" else f"{metric}_mean"
        series = []
        for slug, short, original, *_ in METHODS:
            records = sorted([r for r in rows[source] if r["method_slug"] == slug], key=lambda r: int(r["dimension"]))
            series.append({"name": original, "method_slug": slug, "display_label": short,
                           "field": field, "x": [int(r["dimension"]) for r in records],
                           "y": [float(r[field]) for r in records]})
        panels[f"figure3_{metric}"] = {"source": source, "source_sha256": SOURCE_HASHES[source], "series": series}
    return panels


def load_legacy_module(name):
    spec = importlib.util.spec_from_file_location(f"phase_b_legacy_{name}", LEGACY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_legacy_arrays(rows, panels, plt, np):
    """Capture actual legacy main-axis plot calls, without saving legacy assets."""
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    golden = load_legacy_module("golden_range")
    high = load_legacy_module("high_dim")
    original_plot = Axes.plot
    original_save = Figure.savefig
    captured = []

    def record_plot(ax, *args, **kwargs):
        lines = original_plot(ax, *args, **kwargs)
        if ax is ax.figure.axes[0]:
            for line in lines:
                captured.append({"x": np.asarray(line.get_xdata()).tolist(),
                                 "y": np.asarray(line.get_ydata()).tolist(), "label": line.get_label()})
        return lines

    checks = {}
    old_cwd = Path.cwd()
    try:
        Axes.plot = record_plot
        Figure.savefig = lambda *args, **kwargs: None
        with tempfile.TemporaryDirectory(prefix="paper2025_phase_b_legacy_") as temp:
            os.chdir(temp)  # Legacy _ensure_directory also touches results/.mplconfig.
            functions = {
                "le": golden.plot_enhanced_golden_range_le,
                "se": golden.plot_enhanced_golden_range_se,
                "fe": golden.plot_enhanced_golden_range_fe,
                "pe": golden.plot_enhanced_golden_range_pe,
                "si": golden.plot_enhanced_golden_range_scale_index,
            }
            for metric in METRICS:
                captured.clear()
                functions[metric](rows[f"expB_golden_range/golden_range_{metric}.csv"], output_dir=Path(temp) / metric)
                key = f"figure2_{metric}"
                checks[key] = verify_arrays(panels[key]["series"], captured, np)
                if metric != "le":
                    assert [s["label"] for s in captured] == [rf"$\mathrm{{{metric.upper()}}}_{i}$" for i in range(1, 5)] + [rf"$\mathrm{{{metric.upper()}}}_{{\mathrm{{mean}}}}$"]
                plt.close("all")
            slug_map = dict(zip([m[0] for m in METHODS], high.METHODS, strict=True))
            records = [dict(row, method=slug_map[row["method_slug"]]) for row in rows["expE_high_dim_literature/high_dim_all_methods_summary.csv"]]
            captured.clear()
            high.plot_metric_comparisons(records, output_dir=Path(temp) / "high_dim")
            # Legacy output order is LE/SE/FE/SI/PE; manuscript order is LE/SE/FE/PE/SI.
            for i, metric in enumerate(("le", "se", "fe", "si", "pe")):
                original = captured[5 * i:5 * i + 5]
                key = f"figure3_{metric}"
                assert [s["label"] for s in original] == [m[2] for m in METHODS]
                checks[key] = verify_arrays(panels[key]["series"], original, np)
            plt.close("all")
    finally:
        os.chdir(old_cwd)
        Axes.plot = original_plot
        Figure.savefig = original_save
    return checks


def verify_arrays(expected, actual, np):
    assert len(expected) == len(actual), "series count mismatch"
    result = []
    for source, plotted in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(source["x"]), np.asarray(plotted["x"])), "x mismatch"
        assert np.array_equal(np.asarray(source["y"]), np.asarray(plotted["y"])), "y mismatch"
        result.append({"series_name": source["name"], "point_count": len(source["x"]),
                       "x_equal": True, "y_equal": True,
                       "xy_sha256": hashlib.sha256(np.column_stack([source["x"], source["y"]]).astype("<f8").tobytes()).hexdigest()})
    return {"series_count": len(expected), "series": result, "status": "PASS"}


def component_style(index):
    if index == 4:
        return {"color": "#25292D", "linestyle": "-", "linewidth": 1.25,
                "marker": None, "label": "Stored mean"}
    marker, linestyle = COMPONENTS[index]
    return {"color": COLORS[index], "linestyle": linestyle, "linewidth": 0.8,
            "marker": marker, "markersize": 2.7, "markerfacecolor": "white",
            "markeredgewidth": 0.65, "markevery": 8, "label": f"Component {index + 1}"}


def method_style(index):
    _, short, _, marker, linestyle = METHODS[index]
    return {"color": COLORS[index], "linestyle": linestyle, "linewidth": 1.1,
            "marker": marker, "markersize": 4.0 if index == 0 else 3.2,
            "markerfacecolor": "white", "markeredgewidth": 0.8, "label": short}


def make_panel(key, panel, width_mm, output, plt, np):
    group, metric = key.split("_")
    fig = plt.figure(figsize=(width_mm / 25.4, HEIGHT_MM / 25.4))
    # Fixed physical margins; fonts do not depend on a large native canvas.
    ax = fig.add_axes([14.0 / width_mm, 9.0 / HEIGHT_MM,
                       (width_mm - 16.5) / width_mm, 30.5 / HEIGHT_MM])
    plotted = []
    for index, series in enumerate(panel["series"]):
        if group == "figure2":
            style = {"color": COLORS[0], "linewidth": 1.1} if metric == "le" else component_style(index)
        else:
            style = method_style(index)
        line, = ax.plot(series["x"], series["y"], **style)
        plotted.append({"x": line.get_xdata(), "y": line.get_ydata()})
    check = verify_arrays(panel["series"], plotted, np)
    xlabel = r"$k$" if group == "figure2" else r"$n$"
    ylabel = r"$\lambda_{\max}$" if metric == "le" else (rf"$\mathrm{{{metric.upper()}}}$" if group == "figure2" else rf"$\mathrm{{{metric.upper()}}}_{{\mathrm{{mean}}}}$")
    ax.set_xlabel(xlabel, fontsize=LABEL_BP, labelpad=2.0)
    ax.set_ylabel(ylabel, fontsize=LABEL_BP, labelpad=3.0)
    ax.tick_params(direction="out", labelsize=TICK_BP, length=2.6, width=0.6, pad=2.1)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#555B60")
    ax.grid(False)
    ax.set_xmargin(0.035 if group == "figure3" else 0.05)
    ax.set_xticks(list(range(4, 13)) if group == "figure3" else list(range(0, 121, 20)))
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    if group == "figure2" and metric != "le":
        values = [v for s in panel["series"] for v in s["y"]]
        low, high = min(values), max(values)
        pad = max(0.05 * (high - low), 1e-3)
        ax.set_ylim(low - pad, high + pad)  # Same legacy main-axis rule.
        decimals = {"se": 3, "fe": 2, "pe": 4, "si": 2}[metric]
        ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
        if metric in ("se", "pe"):
            ax.text(1, 1.028, "Local scale; upper bound = 1", transform=ax.transAxes,
                    fontsize=7.0, ha="right", va="bottom", color="#555B60")
    if metric in ("se", "pe", "si") and (group == "figure3" or metric == "si"):
        low, high = ax.get_ylim()
        ax.set_ylim(low, max(high, 1.04 if group == "figure3" else high))
        ax.axhline(1.0, color="#A4A8AB", linestyle=(0, (2, 3)), linewidth=0.55, zorder=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Detect clipping in the native physical page, including tick/axis labels.
    labels = [label for tick, label in zip(ax.get_xticks(), ax.get_xticklabels())
              if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]]
    labels += [label for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())
               if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]]
    labels += [ax.xaxis.label, ax.yaxis.label] + ax.texts
    for label in labels:
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            raise RuntimeError(f"Text clipping: {key}: {label.get_text()}")
    path = output / f"{key}.pdf"
    fig.savefig(path, metadata={"Title": key, "Creator": "paper2025 Phase B publication-only renderer", "CreationDate": None, "ModDate": None})
    check.update({"path": str(path), "width_mm": width_mm, "height_mm": HEIGHT_MM,
                  "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                  "axis_font_bp": LABEL_BP, "tick_font_bp": TICK_BP, "grid": False})
    plt.close(fig)
    return check


def make_legend(group, width_mm, output, plt):
    from matplotlib.lines import Line2D
    height = 8.0 if group == "figure2" else 6.0
    fig = plt.figure(figsize=(width_mm / 25.4, height / 25.4))
    if group == "figure2":
        handles = [Line2D([], [], **{k: v for k, v in component_style(i).items() if k != "markevery"}) for i in range(5)]
        fig.text(0.5, 0.88, "Components and stored mean (b-e)", ha="center", va="center", fontsize=LEGEND_BP)
        anchor = (0.5, 0.32)
    else:
        handles = [Line2D([], [], **method_style(i)) for i in range(5)]
        anchor = (0.5, 0.5)
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=anchor, ncol=5,
                        frameon=False, fontsize=LEGEND_BP, handlelength=2.4,
                        handletextpad=0.45, columnspacing=1.1, borderpad=0)
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    assert box.x0 >= 0 and box.x1 <= fig.bbox.width and box.y0 >= 0 and box.y1 <= fig.bbox.height, "Legend clipping"
    fig.savefig(output / f"{group}_legend.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width-mm", type=float, default=159.2)
    parser.add_argument("--layout", choices=("3+2", "2+2+1"), default="2+2+1")
    parser.add_argument("--manifest-path", type=Path, help="Optional evidence location outside the image directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "paper2025_phase_b_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": TICK_BP,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "#25292D", "axes.labelcolor": "#25292D", "xtick.color": "#25292D", "ytick.color": "#25292D",
        "path.simplify": False, "agg.path.chunksize": 0, "figure.dpi": 150,
    })
    rows = load_frozen_rows()
    panels = select_series(rows)
    legacy_checks = compare_legacy_arrays(rows, panels, plt, np)
    panel_width = args.width_mm * (0.32 if args.layout == "3+2" else 0.48)
    checks = {key: make_panel(key, panel, panel_width, output, plt, np) for key, panel in panels.items()}
    for group in ("figure2", "figure3"):
        make_legend(group, args.width_mm, output, plt)
    manifest = {
        "layout": args.layout, "composite_width_mm": args.width_mm, "panel_order": list(METRICS),
        "sources": SOURCE_HASHES, "data": panels, "legacy_array_checks": legacy_checks,
        "rendered_array_checks": checks, "DATA VALUES CHANGED": "NO",
        "SCIENTIFIC METRICS RECOMPUTED": "NO", "EXPERIMENTS RERUN": "NO",
        "insets": "Redundant k=51..69 duplicate views omitted; every main-curve point retained.",
        "path_simplify": False,
        "rendering_environment": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                                  "font_family": "STIXGeneral", "mathtext_fontset": "stix", "pdf_fonttype": 42},
    }
    manifest_path = args.manifest_path.resolve() if args.manifest_path else output / "data_integrity.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{args.layout}: 10 panels, 12 PDFs; 46 series verified against legacy plot calls; no data changes.")


if __name__ == "__main__":
    main()
