# Maintainer: zhaowenbo
"""Diagnostic SI figures for ``Proposed4DEnhancedMap`` only."""

from __future__ import annotations

import argparse
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
from metrics.nonperiodicity import (
    DEFAULT_SI_RAW_LENGTH,
    DEFAULT_SI_TRANSIENT,
    DEFAULT_SI_WAVELET,
    scale_index_cwt,
    scale_index_cwt_diagnostics,
)
from utils.initializers import get_common_initial_state_float
from utils.io_utils import ensure_directory, resolve_default_results_subdir, save_json


def build_si_diagnostics_records(
    seq_length: int = DEFAULT_SI_RAW_LENGTH,
    transient: int = DEFAULT_SI_TRANSIENT,
    wavelet: str = DEFAULT_SI_WAVELET,
    a11: int = 371,
) -> dict[str, Any]:
    """Build SI diagnostics for the enhanced proposed system only."""

    chaos_map = Proposed4DEnhancedMap(A_odd=a11, initial_state_float=get_common_initial_state_float())
    trajectory = chaos_map.iterate(num_steps=seq_length, discard_transient=transient)
    summary = scale_index_cwt(trajectory, wavelet=wavelet)
    component_diagnostics = {
        f"x{component_index + 1}": scale_index_cwt_diagnostics(trajectory[:, component_index], wavelet=wavelet)
        for component_index in range(4)
    }
    return {
        "wavelet": wavelet,
        "a11": int(a11),
        "system": {
            "name": chaos_map.name,
            "si_components": summary["si_components"],
            "si_mean": summary["si_mean"],
            "component_diagnostics": component_diagnostics,
        },
    }


def plot_enhanced_scalogram_overview(
    records: dict[str, Any],
    output_dir: str | Path = "results/figures",
    stem: str = "enhanced_scale_index_scalogram",
) -> dict[str, Path]:
    """Render one four-panel scalogram overview figure for the enhanced system."""

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render SI diagnostics.") from exc

    figure_dir = resolve_default_results_subdir(output_dir, "results/figures", "scale_index", "diagnostics")
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
    output_dir: str | Path = "results/figures",
    stem_prefix: str = "enhanced_scale_index_si",
) -> dict[str, Path]:
    """Render one SI inner-scalogram figure for a single enhanced-system component."""

    if component_index not in (1, 2, 3, 4):
        raise ValueError("component_index must be one of 1, 2, 3, 4.")

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render SI diagnostics.") from exc

    figure_dir = resolve_default_results_subdir(output_dir, "results/figures", "scale_index", "diagnostics")
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


def run_si_diagnostics(
    seq_length: int = DEFAULT_SI_RAW_LENGTH,
    transient: int = DEFAULT_SI_TRANSIENT,
    wavelet: str = DEFAULT_SI_WAVELET,
    a11: int = 371,
    raw_output_dir: str | Path = "results/raw",
    figure_output_dir: str | Path = "results/figures",
) -> dict[str, Any]:
    """Run enhanced-only SI diagnostics and save five independent figures."""

    records = build_si_diagnostics_records(
        seq_length=seq_length,
        transient=transient,
        wavelet=wavelet,
        a11=a11,
    )
    raw_dir = resolve_default_results_subdir(raw_output_dir, "results/raw", "scale_index", "diagnostics")
    json_path = save_json(records, raw_dir / "enhanced_scale_index_diagnostics.json")

    figure_paths: dict[str, dict[str, Path]] = {
        "scalogram": plot_enhanced_scalogram_overview(records, output_dir=figure_output_dir)
    }
    for component_index in (1, 2, 3, 4):
        figure_paths[f"si_x{component_index}"] = plot_enhanced_si_component_curve(
            records,
            component_index=component_index,
            output_dir=figure_output_dir,
        )

    explanations = {
        "scalogram": "Scalogram overview of the enhanced map. Each panel shows how wavelet energy changes across scales for one state component.",
        "si_x1": "Inner scalogram curve for x1. The marked scales indicate the representative scale and the post-maximum minimum used in the scale-index computation.",
        "si_x2": "Inner scalogram curve for x2. This figure highlights how the non-periodicity score is extracted from the scale-wise energy profile.",
        "si_x3": "Inner scalogram curve for x3. It shows the scale interval used by the standard scale-index workflow for the enhanced map.",
        "si_x4": "Inner scalogram curve for x4. It visualizes the component-level scale-index structure behind the final SI value.",
    }
    return {
        "records": records,
        "json_path": json_path,
        "figure_paths": figure_paths,
        "explanations": explanations,
    }


def main() -> None:
    """Run enhanced-only SI diagnostics with command-line overrides."""

    parser = argparse.ArgumentParser(description="Plot SI diagnostics for the enhanced proposed map.")
    parser.add_argument("--steps", type=int, default=DEFAULT_SI_RAW_LENGTH)
    parser.add_argument("--discard", type=int, default=DEFAULT_SI_TRANSIENT)
    parser.add_argument("--wavelet", default=DEFAULT_SI_WAVELET)
    parser.add_argument("--a11", type=int, default=371)
    parser.add_argument("--raw-output-dir", default="results/raw")
    parser.add_argument("--figure-output-dir", default="results/figures")
    args = parser.parse_args()

    result = run_si_diagnostics(
        seq_length=args.steps,
        transient=args.discard,
        wavelet=args.wavelet,
        a11=args.a11,
        raw_output_dir=args.raw_output_dir,
        figure_output_dir=args.figure_output_dir,
    )
    print(f"Saved JSON: {result['json_path']}")
    for key in ("scalogram", "si_x1", "si_x2", "si_x3", "si_x4"):
        paths = result["figure_paths"][key]
        print(f"Saved PDF: {paths['pdf']}")
        print(f"Saved EPS: {paths['eps']}")
        print(f"{key}: {result['explanations'][key]}")


if __name__ == "__main__":
    main()
