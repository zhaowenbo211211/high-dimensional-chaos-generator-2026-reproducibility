# Maintainer: zhaowenbo
"""Golden-range fuzzy entropy scan for ``Proposed4DEnhancedMap``."""

from __future__ import annotations

import os
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

from chaos_maps import Proposed4DEnhancedMap
from experiments.export_tables import export_enhanced_golden_range_fe_records
from metrics.fuzzy_entropy import (
    DEFAULT_FE_EMBEDDING_DIM,
    DEFAULT_FE_FUZZY_EXPONENT,
    DEFAULT_FE_RAW_LENGTH,
    DEFAULT_FE_TAU,
    DEFAULT_FE_TRANSIENT,
    fuzzy_entropy,
)
from utils.initializers import get_common_initial_state_float
from utils.io_utils import ensure_directory

GOLDEN_RANGE_START = 271
GOLDEN_RANGE_STOP = 507
GOLDEN_RANGE_STEP = 2


def golden_range_a11_values(
    start: int = GOLDEN_RANGE_START,
    stop: int = GOLDEN_RANGE_STOP,
    step: int = GOLDEN_RANGE_STEP,
) -> list[int]:
    """Return valid odd ``a11`` values inside the paper's golden interval."""

    if step <= 0:
        raise ValueError("step must be positive.")
    if start > stop:
        raise ValueError("start must not exceed stop.")
    first_value = start if start % 2 == 1 else start + 1
    return list(range(first_value, stop + 1, step))


def scan_enhanced_golden_range_fe(
    a11_start: int = GOLDEN_RANGE_START,
    a11_stop: int = GOLDEN_RANGE_STOP,
    a11_step: int = GOLDEN_RANGE_STEP,
    transient: int = DEFAULT_FE_TRANSIENT,
    seq_length: int = DEFAULT_FE_RAW_LENGTH,
    m: int = DEFAULT_FE_EMBEDDING_DIM,
    r: float | None = None,
    tau: int = DEFAULT_FE_TAU,
    n: int = DEFAULT_FE_FUZZY_EXPONENT,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Scan the enhanced-map golden range and compute direct-mode FuzzyEn.

    The paper-facing workflow quantizes each trajectory component into the
    shared 16-bit integer domain before computing direct numeric-sequence
    FuzzyEn. The horizontal plotting coordinate ``k`` is the group index after
    sorting the golden-range parameter family by increasing ``a11``.
    """

    records: list[dict[str, Any]] = []
    a11_values = golden_range_a11_values(start=a11_start, stop=a11_stop, step=a11_step)
    for group_index, a11 in enumerate(a11_values, start=1):
        if progress:
            print(f"[{group_index}/{len(a11_values)}] scanning a11={a11}")
        try:
            chaos_map = Proposed4DEnhancedMap(
                A_odd=a11,
                B_even=50,
                Delta=2,
                initial_state_float=get_common_initial_state_float(),
            )
            trajectory = chaos_map.iterate(num_steps=seq_length, discard_transient=transient)
            result = fuzzy_entropy(trajectory, mode="direct", m=m, r=r, tau=tau, n=n)
            fe_x1, fe_x2, fe_x3, fe_x4 = result["fe_components"]
            records.append(
                {
                    "group_index": group_index,
                    "a11": a11,
                    "a21": a11 + 2,
                    "a31": a11 + 4,
                    "b": int(chaos_map.b),
                    "fe_x1": float(fe_x1),
                    "fe_x2": float(fe_x2),
                    "fe_x3": float(fe_x3),
                    "fe_x4": float(fe_x4),
                    "fe_mean": float(result["fe_mean"]),
                    "raw_length": int(result["raw_length"]),
                    "test_length": int(result["test_length"]),
                    "mode": str(result["mode"]),
                    "m": int(result["m"]),
                    "r": result["r"],
                    "tau": int(result["tau"]),
                    "n": int(result["n"]),
                    "error": "",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "group_index": group_index,
                    "a11": a11,
                    "a21": a11 + 2,
                    "a31": a11 + 4,
                    "b": None,
                    "fe_x1": float("nan"),
                    "fe_x2": float("nan"),
                    "fe_x3": float("nan"),
                    "fe_x4": float("nan"),
                    "fe_mean": float("nan"),
                    "raw_length": int(seq_length),
                    "test_length": int(seq_length),
                    "mode": "direct",
                    "m": int(m),
                    "r": r if r is not None else "0.2*std(component_u16_sequence)",
                    "tau": int(tau),
                    "n": int(n),
                    "error": str(exc),
                }
            )
    return records


def plot_enhanced_golden_range_fe(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_golden_range_fe",
) -> dict[str, Path]:
    """Plot component-wise FE curves and the mean curve versus group index.

    The layout follows the current SE plotting style: a single main panel with
    five curves, tight y-axis scaling, an outside legend, and an optional
    outside zoom inset when the curves cluster tightly.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render the enhanced golden-range FE figure.") from exc

    figure_dir = ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["group_index"]))
    x_values = np.asarray([int(record["group_index"]) for record in sorted_records], dtype=np.int64)
    component_curves = {
        r"$\mathrm{FE}_1$": np.asarray([float(record["fe_x1"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{FE}_2$": np.asarray([float(record["fe_x2"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{FE}_3$": np.asarray([float(record["fe_x3"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{FE}_4$": np.asarray([float(record["fe_x4"]) for record in sorted_records], dtype=np.float64),
    }
    mean_curve = np.asarray([float(record["fe_mean"]) for record in sorted_records], dtype=np.float64)
    stacked_curves = np.vstack([*component_curves.values(), mean_curve])
    finite_values = stacked_curves[np.isfinite(stacked_curves)]
    if finite_values.size == 0:
        raise ValueError("No valid FE values are available for plotting.")

    fig, ax = plt.subplots(figsize=(9.4, 4.4), dpi=300)
    component_styles = [
        {"linestyle": "-", "marker": "o"},
        {"linestyle": "--", "marker": "s"},
        {"linestyle": "-.", "marker": "^"},
        {"linestyle": ":", "marker": "D"},
    ]
    mark_every = max(len(x_values) // 18, 1)
    for (label, y_values), style in zip(component_curves.items(), component_styles, strict=True):
        ax.plot(
            x_values,
            y_values,
            linewidth=1.1,
            markersize=3.0,
            markevery=mark_every,
            label=label,
            **style,
        )
    ax.plot(
        x_values,
        mean_curve,
        linewidth=2.2,
        linestyle="-",
        marker="o",
        markersize=4.2,
        markevery=mark_every,
        label=r"$\mathrm{FE}_{\mathrm{mean}}$",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{FE}$")
    ax.tick_params(direction="in")
    y_min_data = float(np.min(finite_values))
    y_max_data = float(np.max(finite_values))
    delta = y_max_data - y_min_data
    pad = max(0.05 * delta, 1e-3)
    ax.set_ylim(y_min_data - pad, y_max_data + pad)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.2,
        labelspacing=0.45,
    )

    overlap_ratio = delta / max(abs(float(np.mean(finite_values))), 1e-12)
    should_add_inset = len(x_values) >= 10 and overlap_ratio < 0.08
    if should_add_inset:
        zoom_window = max(len(x_values) // 6, 12)
        start_index = max((len(x_values) - zoom_window) // 2, 0)
        end_index = min(start_index + zoom_window, len(x_values))
        x_zoom = x_values[start_index:end_index]
        zoom_stack = stacked_curves[:, start_index:end_index]
        zoom_finite = zoom_stack[np.isfinite(zoom_stack)]
        if x_zoom.size > 1 and zoom_finite.size > 0:
            inset = inset_axes(
                ax,
                width="100%",
                height="100%",
                bbox_to_anchor=(1.01, 0.06, 0.32, 0.28),
                bbox_transform=ax.transAxes,
                loc="lower left",
                borderpad=0.0,
            )
            for (_, y_values), style in zip(component_curves.items(), component_styles, strict=True):
                inset.plot(
                    x_values,
                    y_values,
                    linewidth=1.0,
                    markersize=2.2,
                    markevery=mark_every,
                    **style,
                )
            inset.plot(
                x_values,
                mean_curve,
                linewidth=1.8,
                linestyle="-",
                marker="o",
                markersize=3.0,
                markevery=mark_every,
            )
            zoom_y_min = float(np.min(zoom_finite))
            zoom_y_max = float(np.max(zoom_finite))
            zoom_delta = zoom_y_max - zoom_y_min
            zoom_pad = max(0.08 * zoom_delta, 5e-4)
            inset.set_xlim(float(x_zoom[0]), float(x_zoom[-1]))
            inset.set_ylim(zoom_y_min - zoom_pad, zoom_y_max + zoom_pad)
            inset.tick_params(direction="in", labelsize=7)
            inset.set_xticks([int(x_zoom[0]), int(x_zoom[len(x_zoom) // 2]), int(x_zoom[-1])])
            inset.set_facecolor("white")
            for spine in inset.spines.values():
                spine.set_linewidth(0.8)
            mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.55", lw=0.7)

    fig.subplots_adjust(left=0.10, right=0.60, bottom=0.16, top=0.96)

    pdf_path = figure_dir / f"{stem}.pdf"
    eps_path = figure_dir / f"{stem}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "eps": eps_path}


def run_enhanced_golden_range_fe_scan(
    a11_start: int = GOLDEN_RANGE_START,
    a11_stop: int = GOLDEN_RANGE_STOP,
    a11_step: int = GOLDEN_RANGE_STEP,
    transient: int = DEFAULT_FE_TRANSIENT,
    seq_length: int = DEFAULT_FE_RAW_LENGTH,
    m: int = DEFAULT_FE_EMBEDDING_DIM,
    r: float | None = None,
    tau: int = DEFAULT_FE_TAU,
    n: int = DEFAULT_FE_FUZZY_EXPONENT,
    raw_output_dir: str | Path = "results/raw",
    figure_output_dir: str | Path = "results/figures",
    progress: bool = True,
) -> dict[str, Any]:
    """Run the full enhanced-map golden-range FE scan, export, and plot."""

    records = scan_enhanced_golden_range_fe(
        a11_start=a11_start,
        a11_stop=a11_stop,
        a11_step=a11_step,
        transient=transient,
        seq_length=seq_length,
        m=m,
        r=r,
        tau=tau,
        n=n,
        progress=progress,
    )
    stem = "enhanced_golden_range_fe"
    data_paths = export_enhanced_golden_range_fe_records(records, output_dir=raw_output_dir, stem=stem)
    figure_paths = plot_enhanced_golden_range_fe(records, output_dir=figure_output_dir, stem=stem)
    valid_count = sum(np.isfinite(float(record["fe_mean"])) for record in records)
    return {
        "records": records,
        "data_paths": data_paths,
        "figure_paths": figure_paths,
        "valid_count": int(valid_count),
        "total_count": len(records),
    }


def main() -> None:
    """Run the default enhanced golden-range FE scan directly."""

    result = run_enhanced_golden_range_fe_scan()
    print(f"Completed {result['valid_count']}/{result['total_count']} valid FE points.")
    print(f"Saved CSV: {result['data_paths']['csv']}")
    print(f"Saved PDF: {result['figure_paths']['pdf']}")
    print(f"Saved EPS: {result['figure_paths']['eps']}")


if __name__ == "__main__":
    main()
