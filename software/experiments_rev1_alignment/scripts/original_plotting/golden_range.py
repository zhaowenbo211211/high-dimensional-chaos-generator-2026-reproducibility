# Maintainer: zhaowenbo
"""Golden-range publication figures, copied from the legacy plotting scripts.

Provenance (see audit/plot_style_rework/source_map.md):
  - plot_enhanced_golden_range_se   <- plot_golden_range_se.py  :: plot_enhanced_golden_range_se   (L126-265), sha256 a52bef70...
  - plot_enhanced_golden_range_fe   <- plot_golden_range_fe.py  :: plot_enhanced_golden_range_fe   (L136-272), sha256 a4209d01...
  - plot_enhanced_golden_range_pe   <- plot_golden_range_pe.py  :: plot_enhanced_golden_range_pe   (L127-257), sha256 02505992...
  - plot_enhanced_golden_range_scale_index <- plot_golden_range_si.py :: plot_enhanced_golden_range_scale_index (L122-258), sha256 f327839a...
  - plot_enhanced_golden_range_le   <- plot_golden_range_le.py  :: plot_proposed_golden_range_le   (L136-174), sha256 c475750c...
    The published LE figure predates the surviving script version; the copy keeps the
    surviving function body and applies only the three evidence-based adaptations
    documented in source_map.md: x = group_index, xlabel = $k$, eps export.

Functional adaptations relative to the legacy files (no style changes):
  - utils.io_utils.ensure_directory / resolve_default_results_subdir are replaced by
    the local _ensure_directory; output directories are passed explicitly by the caller.
  - MPLCONFIGDIR defaults to results/.mplconfig under the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_enhanced_golden_range_le(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_golden_range_le",
) -> dict[str, Path]:
    """Plot group index ``k`` against ``lyap_max`` for the proposed golden-range family.

    The plotted ordinate is the Wolf full-state maximum Lyapunov exponent. The
    abscissa uses the group index ``k`` in ascending order (the published figure;
    see source_map.md for why this differs from the surviving legacy script).
    """

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render the golden-range LE figure. Install it before plotting."
        ) from exc

    figure_dir = _ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["group_index"]))
    x_values = np.asarray([int(record["group_index"]) for record in sorted_records], dtype=np.int64)
    y_values = np.asarray([float(record["lyap_max"]) for record in sorted_records], dtype=np.float64)
    if not np.any(np.isfinite(y_values)):
        raise ValueError("No valid lyap_max values are available for plotting.")

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    ax.plot(x_values, y_values, linewidth=1.1)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.tick_params(direction="in")
    fig.tight_layout()

    pdf_path = figure_dir / f"{stem}.pdf"
    eps_path = figure_dir / f"{stem}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "eps": eps_path}


def plot_enhanced_golden_range_se(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_golden_range_se_components",
) -> dict[str, Path]:
    """Plot component-wise SE curves and the mean curve versus group index.

    The main figure stays a single-panel multi-line plot. When the five curves
    are numerically very close to each other, a small zoom inset is added
    outside the main axes so that local detail is enlarged without covering
    any part of the primary data region.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render the enhanced golden-range SE figure.") from exc

    figure_dir = _ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["group_index"]))
    x_values = np.asarray([int(record["group_index"]) for record in sorted_records], dtype=np.int64)
    component_curves = {
        r"$\mathrm{SE}_1$": np.asarray([float(record["se_x1"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{SE}_2$": np.asarray([float(record["se_x2"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{SE}_3$": np.asarray([float(record["se_x3"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{SE}_4$": np.asarray([float(record["se_x4"]) for record in sorted_records], dtype=np.float64),
    }
    mean_curve = np.asarray([float(record["se_mean"]) for record in sorted_records], dtype=np.float64)
    stacked_curves = np.vstack([*component_curves.values(), mean_curve])
    finite_values = stacked_curves[np.isfinite(stacked_curves)]
    if finite_values.size == 0:
        raise ValueError("No valid SE values are available for plotting.")

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
        label=r"$\mathrm{SE}_{\mathrm{mean}}$",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{SE}$")
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

    # Add an outside inset automatically when the five curves cluster tightly
    # so readers can inspect subtle differences without covering the main plot.
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
            for (label, y_values), style in zip(component_curves.items(), component_styles, strict=True):
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


def plot_enhanced_golden_range_fe(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_golden_range_fe",
) -> dict[str, Path]:
    """Plot component-wise FE curves and the mean curve versus group index.

    Structural copy of plot_enhanced_golden_range_se with FE fields/labels,
    exactly as in the legacy plot_golden_range_fe.py.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render the enhanced golden-range FE figure.") from exc

    figure_dir = _ensure_directory(output_dir)
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
            for (label, y_values), style in zip(component_curves.items(), component_styles, strict=True):
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


def plot_enhanced_golden_range_pe(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_golden_range_pe",
) -> dict[str, Path]:
    """Plot component-wise PE curves and the mean curve versus group index.

    Structural copy with PE fields/labels, as in the legacy plot_golden_range_pe.py.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render the enhanced golden-range PE figure.") from exc

    figure_dir = _ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["group_index"]))
    x_values = np.asarray([int(record["group_index"]) for record in sorted_records], dtype=np.int64)
    component_curves = {
        r"$\mathrm{PE}_1$": np.asarray([float(record["pe_x1"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{PE}_2$": np.asarray([float(record["pe_x2"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{PE}_3$": np.asarray([float(record["pe_x3"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{PE}_4$": np.asarray([float(record["pe_x4"]) for record in sorted_records], dtype=np.float64),
    }
    mean_curve = np.asarray([float(record["pe_mean"]) for record in sorted_records], dtype=np.float64)
    stacked_curves = np.vstack([*component_curves.values(), mean_curve])
    finite_values = stacked_curves[np.isfinite(stacked_curves)]
    if finite_values.size == 0:
        raise ValueError("No valid PE values are available for plotting.")

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
        label=r"$\mathrm{PE}_{\mathrm{mean}}$",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{PE}$")
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
            for (label, y_values), style in zip(component_curves.items(), component_styles, strict=True):
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


def plot_enhanced_golden_range_scale_index(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_golden_range_scale_index",
) -> dict[str, Path]:
    """Plot component-wise SI curves and the mean curve versus group index.

    Structural copy with SI fields/labels, as in the legacy plot_golden_range_si.py.
    The legacy version resolves output_dir "results/figures" into a "scale_index"
    subdirectory; here the caller passes the target directory explicitly.
    Unlike SE/FE/PE, the zoom inset is added unconditionally.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render the enhanced golden-range SI figure.") from exc

    figure_dir = _ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["group_index"]))
    x_values = np.asarray([int(record["group_index"]) for record in sorted_records], dtype=np.int64)
    component_curves = {
        r"$\mathrm{SI}_1$": np.asarray([float(record["si_x1"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{SI}_2$": np.asarray([float(record["si_x2"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{SI}_3$": np.asarray([float(record["si_x3"]) for record in sorted_records], dtype=np.float64),
        r"$\mathrm{SI}_4$": np.asarray([float(record["si_x4"]) for record in sorted_records], dtype=np.float64),
    }
    mean_curve = np.asarray([float(record["si_mean"]) for record in sorted_records], dtype=np.float64)
    stacked_curves = np.vstack([*component_curves.values(), mean_curve])
    finite_values = stacked_curves[np.isfinite(stacked_curves)]
    if finite_values.size == 0:
        raise ValueError("No valid SI values are available for plotting.")

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
        label=r"$\mathrm{SI}_{\mathrm{mean}}$",
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{SI}$")
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

    should_add_inset = len(x_values) >= 10
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
            for (label, y_values), style in zip(component_curves.items(), component_styles, strict=True):
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
