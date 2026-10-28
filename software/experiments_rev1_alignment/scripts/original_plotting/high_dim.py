# Maintainer: zhaowenbo
"""High-dimension literature-comparison figures, copied from the legacy script.

Provenance (see audit/plot_style_rework/source_map.md):
  run_high_dim_literature_comparison.py :: plot_metric_comparisons (L273-318)
  plus the METHODS / METHOD_SLUGS / METHOD_SHORT_LABELS / METRIC_SPECS tables (L58-91)
  source sha256 6ecf7272da612be3b41facefaa4cabec8dcb27af7fe157fc8137c6d263a10840

NOTE: run_high_dim_all_methods_full.py is NOT the publication entry point — its
write_placeholder_figures only emits placeholder figures. The published five-panel
comparison figures come from plot_metric_comparisons below (verified byte-identical
against the backup figures).

Only the plotting function and its style tables are copied; evaluate_method_dimension
and the scan drivers are NOT copied. The caller must pass the validated summary rows
from results/expE_high_dim_literature/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


METHOD_PROPOSED = "Proposed enhanced n-D chaotic family"
METHOD_YAO = "Yao et al. 2025 complex Pascal n-D map"
METHOD_SINE_MEMRISTIVE = "Sine-map lattice with discrete memristor coupling"
METHOD_CYCLIC = "High-dimensional cyclic symmetric logistic map"
METHOD_ZANG = "Zang et al. 2021 linear modular system"
METHODS = [METHOD_PROPOSED, METHOD_YAO, METHOD_SINE_MEMRISTIVE, METHOD_CYCLIC, METHOD_ZANG]
METHOD_SHORT_LABELS = {
    METHOD_PROPOSED: "The proposed method",
    METHOD_YAO: "Yao et al.'s method",
    METHOD_SINE_MEMRISTIVE: "He et al.'s method",
    METHOD_CYCLIC: "Zhang et al.'s method",
    METHOD_ZANG: "Zang et al.'s method",
}
METRIC_SPECS = [
    ("lambda_max", r"$\lambda_{\max}$", "lambda_max"),
    ("se_mean", r"$\mathrm{SE}_{\mathrm{mean}}$", "se_mean"),
    ("fe_mean", r"$\mathrm{FE}_{\mathrm{mean}}$", "fe_mean"),
    ("si_mean", r"$\mathrm{SI}_{\mathrm{mean}}$", "si_mean"),
    ("pe_mean", r"$\mathrm{PE}_{\mathrm{mean}}$", "pe_mean"),
]


def plot_metric_comparisons(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Create one PDF/EPS figure per metric."""

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_directory("results/.mplconfig")))
    import matplotlib.pyplot as plt

    figure_dir = _ensure_directory(output_dir)
    paths: dict[str, dict[str, Path]] = {}
    methods = METHODS
    markers = {
        METHOD_PROPOSED: "o",
        METHOD_YAO: "^",
        METHOD_SINE_MEMRISTIVE: "D",
        METHOD_CYCLIC: "v",
        METHOD_ZANG: "s",
    }
    labels = METHOD_SHORT_LABELS

    for metric_key, ylabel, stem in METRIC_SPECS:
        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=300)
        for method in methods:
            method_records = sorted(
                [record for record in records if record["method"] == method],
                key=lambda item: int(item["dimension"]),
            )
            ax.plot(
                [int(record["dimension"]) for record in method_records],
                [float(record[metric_key]) for record in method_records],
                marker=markers[method],
                linewidth=1.4,
                markersize=4.0,
                label=labels[method],
            )
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(4, 13)))
        ax.tick_params(direction="in")
        ax.legend(frameon=False)
        fig.tight_layout()
        pdf_path = figure_dir / f"high_dim_comparison_{stem}.pdf"
        eps_path = figure_dir / f"high_dim_comparison_{stem}.eps"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(eps_path, bbox_inches="tight")
        plt.close(fig)
        paths[stem] = {"pdf": pdf_path, "eps": eps_path}
    return paths
