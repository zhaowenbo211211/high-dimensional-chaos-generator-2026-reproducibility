# Maintainer: zhaowenbo
"""SI diagnostic publication figures, copied from legacy plot_si_diagnostics.py.

Provenance (see audit/plot_style_rework/source_map.md):
  plot_si_diagnostics.py :: plot_enhanced_scalogram_overview   (L60-104)
  plot_si_diagnostics.py :: plot_enhanced_si_component_curve   (L107-163)
  source sha256 d8d6f4c6ff5a190803591290b1054f0a7673453f79707270b2b73e8acde4d115

Only the two plotting functions are copied; build_si_diagnostics_records (which
iterates the map) is NOT copied. The caller must supply the validated diagnostics
JSON from results/expC_si_diagnostics/, which has the identical structure.
The legacy resolve_default_results_subdir indirection is replaced by an explicit
output directory argument.
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


def plot_enhanced_scalogram_overview(
    records: dict[str, Any],
    output_dir: str | Path,
    stem: str = "enhanced_scale_index_scalogram",
) -> dict[str, Path]:
    """Render one four-panel scalogram overview figure for the enhanced system."""

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render SI diagnostics.") from exc

    figure_dir = _ensure_directory(output_dir)
    component_diagnostics = records["system"]["component_diagnostics"]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), dpi=300)
    for component_index, ax in enumerate(axes.ravel(), start=1):
        diag = component_diagnostics[f"x{component_index}"]
        scales = np.asarray(diag["scales"], dtype=np.float64)
        scalogram = np.asarray(diag["scalogram"], dtype=np.float64)
        heat_image = np.tile(scalogram[:, None], (1, 32))
        ax.imshow(
            heat_image,
            aspect="auto",
            origin="lower",
            extent=(0.0, 1.0, float(scales[0]), float(scales[-1])),
            cmap="viridis",
        )
        ax.axhline(float(diag["study_scale_max"]), color="white", linestyle="--", linewidth=0.8)
        ax.axhline(float(diag["s_max_scale"]), color="red", linestyle="-", linewidth=0.9)
        ax.axhline(float(diag["s_min_scale"]), color="cyan", linestyle="-", linewidth=0.9)
        ax.set_xticks([])
        if component_index in (1, 3):
            ax.set_ylabel(r"$s$")
        ax.tick_params(direction="in")
        ax.text(0.03, 0.95, rf"$x_{{{component_index}}}$", transform=ax.transAxes, ha="left", va="top", color="white")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, wspace=0.14, hspace=0.16)
    pdf_path = figure_dir / f"{stem}.pdf"
    eps_path = figure_dir / f"{stem}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "eps": eps_path}


def plot_enhanced_si_component_curve(
    records: dict[str, Any],
    component_index: int,
    output_dir: str | Path,
    stem_prefix: str = "enhanced_scale_index_si",
) -> dict[str, Path]:
    """Render one SI inner-scalogram figure for a single enhanced-system component."""

    if component_index not in (1, 2, 3, 4):
        raise ValueError("component_index must be one of 1, 2, 3, 4.")

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render SI diagnostics.") from exc

    figure_dir = _ensure_directory(output_dir)
    diag = records["system"]["component_diagnostics"][f"x{component_index}"]
    scales = np.asarray(diag["scales"], dtype=np.float64)
    scalogram = np.asarray(diag["scalogram"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
    ax.plot(scales, scalogram, linewidth=1.7, color="C0")
    ax.axvline(float(diag["study_scale_max"]), color="0.5", linestyle="--", linewidth=0.8, label=r"$s_1$")
    ax.scatter([float(diag["s_max_scale"])], [float(diag["s_max_value"])], s=28, marker="o", label=r"$s_{\max}$")
    ax.scatter([float(diag["s_min_scale"])], [float(diag["s_min_value"])], s=28, marker="s", label=r"$s_{\min}$")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$S(s)$")
    ax.tick_params(direction="in")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.text(0.03, 0.95, rf"$x_{{{component_index}}}$", transform=ax.transAxes, ha="left", va="top")
    ax.text(
        0.03,
        0.85,
        rf"$\mathrm{{SI}}={float(diag['si_value']):.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    finite_values = scalogram[np.isfinite(scalogram)]
    if finite_values.size > 0:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        delta = y_max - y_min
        pad = max(0.05 * delta, 1e-4)
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.96)
    stem = f"{stem_prefix}_x{component_index}"
    pdf_path = figure_dir / f"{stem}.pdf"
    eps_path = figure_dir / f"{stem}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "eps": eps_path}
