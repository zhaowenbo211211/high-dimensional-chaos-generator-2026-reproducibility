# Maintainer: zhaowenbo
"""Plot helpers for the enhanced high-dimensional family experiment."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from utils.io_utils import ensure_directory


def plot_dimension_scalability(
    records: list[dict[str, object]],
    output_dir: str | Path = "results/high_dimensional/figures",
    stem: str = "dimension_metrics_summary",
) -> dict[str, Path]:
    """Plot metric means versus system dimension."""

    os.environ.setdefault("MPLCONFIGDIR", str(ensure_directory("results/.mplconfig")))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render dimension-scalability figures.") from exc

    figure_dir = ensure_directory(output_dir)
    sorted_records = sorted(records, key=lambda record: int(record["dimension"]))
    dimensions = np.asarray([int(record["dimension"]) for record in sorted_records], dtype=np.int64)

    curves = [
        (r"$\lambda_{\max}$", np.asarray([float(record["lyap_max"]) for record in sorted_records], dtype=np.float64), "-"),
        (r"$\mathrm{SE}_{\mathrm{mean}}$", np.asarray([float(record["se_mean"]) for record in sorted_records], dtype=np.float64), "--"),
        (r"$\mathrm{FE}_{\mathrm{mean}}$", np.asarray([float(record["fe_mean"]) for record in sorted_records], dtype=np.float64), "-."),
        (r"$\mathrm{SI}_{\mathrm{mean}}$", np.asarray([float(record["si_mean"]) for record in sorted_records], dtype=np.float64), ":"),
        (r"$\mathrm{PE}_{\mathrm{mean}}$", np.asarray([float(record["pe_mean"]) for record in sorted_records], dtype=np.float64), (0, (3, 1, 1, 1))),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=300)
    for label, values, linestyle in curves:
        ax.plot(
            dimensions,
            values,
            linewidth=1.6,
            linestyle=linestyle,
            marker="o",
            markersize=4.0,
            label=label,
        )

    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathrm{Metric\ Mean}$")
    ax.set_xticks(dimensions.tolist())
    ax.tick_params(direction="in")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()

    pdf_path = figure_dir / f"{stem}.pdf"
    eps_path = figure_dir / f"{stem}.eps"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "eps": eps_path}
