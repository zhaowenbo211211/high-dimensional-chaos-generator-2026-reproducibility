# Maintainer: zhaowenbo
"""Golden-range Wolf full-state LE scan for the proposed 4D family."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import Proposed4DBaseMap, Proposed4DEnhancedMap
from experiments.export_tables import export_proposed_golden_range_le_records
from metrics.lyapunov import estimate_max_lyapunov_wolf_full_state
from utils.io_utils import ensure_directory
from utils.initializers import get_common_initial_state_float

ProposedVariant = Literal["proposed_base", "proposed_enhanced"]

GOLDEN_RANGE_START = 271
GOLDEN_RANGE_STOP = 507
GOLDEN_RANGE_STEP = 2

 
def golden_range_a11_values(
    start: int = GOLDEN_RANGE_START,
    stop: int = GOLDEN_RANGE_STOP,
    step: int = GOLDEN_RANGE_STEP,
) -> list[int]:
    """Return the odd ``a11`` values in the paper's golden interval.

    The R1--LY construction requires ``A_odd = a11`` to be odd, so the default
    scan follows the valid odd subsequence inside ``271 <= a11 <= 507``.
    """

    if step <= 0:
        raise ValueError("step must be positive.")
    if start > stop:
        raise ValueError("start must not exceed stop.")

    first_value = start if start % 2 == 1 else start + 1
    return list(range(first_value, stop + 1, step))


def build_proposed_variant_map(variant: ProposedVariant, a11: int):
    """Build one proposed-family map instance for a golden-range parameter group."""

    common_kwargs = {
        "A_odd": a11,
        "B_even": 50,
        "Delta": 2,
        "initial_state_float": get_common_initial_state_float(),
    }
    if variant == "proposed_base":
        return Proposed4DBaseMap(**common_kwargs)
    if variant == "proposed_enhanced":
        return Proposed4DEnhancedMap(**common_kwargs)
    raise ValueError(f"Unsupported proposed variant: {variant}")


def scan_proposed_golden_range_le(
    variant: ProposedVariant = "proposed_base",
    a11_start: int = GOLDEN_RANGE_START,
    a11_stop: int = GOLDEN_RANGE_STOP,
    a11_step: int = GOLDEN_RANGE_STEP,
    transient: int = 1000,
    max_steps: int = 5000,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Scan the proposed-family golden range and estimate Wolf full-state LE.

    This workflow supports the paper's proposed 4D family under fixed coupling
    gains ``[50, 48, 46]`` and linear gains ``[a11, a11 + 2, a11 + 4]``.
    The LE estimator is the existing Wolf full-state method. Every group starts
    from the common project initialization ``(0.1, 0.2, 0.3, 0.4)``. The raw
    output preserves both the real coefficients and the scan order.
    """

    records: list[dict[str, Any]] = []
    a11_values = golden_range_a11_values(start=a11_start, stop=a11_stop, step=a11_step)

    for index, a11 in enumerate(a11_values, start=1):
        if progress:
            print(f"[{index}/{len(a11_values)}] scanning a11={a11}")

        try:
            chaos_map = build_proposed_variant_map(variant=variant, a11=a11)
            result = estimate_max_lyapunov_wolf_full_state(
                chaos_map,
                transient=transient,
                max_steps=max_steps,
            )
            records.append(
                {
                    "variant": variant,
                    "group_index": index,
                    "a11": a11,
                    "a21": a11 + 2,
                    "a31": a11 + 4,
                    "b": int(chaos_map.b),
                    "lyap_max": float(result["lyap_max"]),
                    "lyap_components": np.asarray(result["lyap_components"], dtype=np.float64).tolist(),
                    "renormalizations": int(result["renormalizations"]),
                    "is_chaotic": bool(result["is_chaotic"]),
                    "error": "",
                }
            )
        except Exception as exc:
            records.append(
                {
                    "variant": variant,
                    "group_index": index,
                    "a11": a11,
                    "a21": a11 + 2,
                    "a31": a11 + 4,
                    "b": None,
                    "lyap_max": float("nan"),
                    "lyap_components": [float("nan")] * 4,
                    "renormalizations": 0,
                    "is_chaotic": False,
                    "error": str(exc),
                }
            )

    return records


def plot_proposed_golden_range_le(
    records: list[dict[str, Any]],
    output_dir: str | Path = "results/figures",
    stem: str = "proposed_golden_range_le",
) -> dict[str, Path]:
    """Plot ``a11`` against ``lyap_max`` for the proposed golden-range family.

    The plotted ordinate is the Wolf full-state maximum Lyapunov exponent. The
    abscissa uses the real parameter value ``a11`` in ascending order.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render the golden-range LE figure. Install it before plotting."
        ) from exc

    figure_dir = ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["a11"]))
    x_values = np.asarray([int(record["a11"]) for record in sorted_records], dtype=np.int64)
    y_values = np.asarray([float(record["lyap_max"]) for record in sorted_records], dtype=np.float64)
    if not np.any(np.isfinite(y_values)):
        raise ValueError("No valid lyap_max values are available for plotting.")

    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    ax.plot(x_values, y_values, linewidth=1.1)
    ax.set_xlabel(r"$a_{1,1}$")
    ax.set_ylabel(r"$\lambda_{\max}$")
    ax.tick_params(direction="in")
    fig.tight_layout()

    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def run_proposed_golden_range_le_scan(
    variant: ProposedVariant = "proposed_base",
    a11_start: int = GOLDEN_RANGE_START,
    a11_stop: int = GOLDEN_RANGE_STOP,
    a11_step: int = GOLDEN_RANGE_STEP,
    transient: int = 1000,
    max_steps: int = 5000,
    raw_output_dir: str | Path = "results/raw",
    figure_output_dir: str | Path = "results/figures",
    progress: bool = True,
) -> dict[str, Any]:
    """Run the proposed-family golden-range Wolf full-state LE scan, export, and plot."""

    records = scan_proposed_golden_range_le(
        variant=variant,
        a11_start=a11_start,
        a11_stop=a11_stop,
        a11_step=a11_step,
        transient=transient,
        max_steps=max_steps,
        progress=progress,
    )
    stem = f"{variant}_golden_range_le"
    data_paths = export_proposed_golden_range_le_records(records, output_dir=raw_output_dir, stem=stem)
    figure_paths = plot_proposed_golden_range_le(records, output_dir=figure_output_dir, stem=stem)
    valid_count = sum(np.isfinite(float(record["lyap_max"])) for record in records)

    return {
        "records": records,
        "data_paths": data_paths,
        "figure_paths": figure_paths,
        "valid_count": int(valid_count),
        "total_count": len(records),
    }
