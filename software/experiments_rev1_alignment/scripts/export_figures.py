# Maintainer: zhaowenbo
"""Generate all rev1 figures (PDF + EPS) from aggregated results.

Usage: python3 scripts/export_figures.py
Outputs land in results/figures/<paper-relative-dir>/ as <stem>.pdf and <stem>.eps.
The staging step converts EPS to the *-eps-converted-to.pdf form where the paper
loads extensionless paths.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SUBSYSTEM_ROOT / "src"
for path in (str(SRC_ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)

RESULTS = SUBSYSTEM_ROOT / "results"
FIGURES = RESULTS / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(fig, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    eps = out_dir / f"{stem}.eps"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(eps, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "eps": str(eps)}


def plot_golden_range() -> dict:
    base = FIGURES / "camp_dynamic"
    paths = {}
    metric_specs = {
        "le": ("le", ["lyap_max"], [r"$\lambda_{\max}$"], "enhanced_golden_range_le"),
        "se": ("se", [f"se_x{i}" for i in range(1, 5)] + ["se_mean"],
               [r"$\mathrm{SE}_1$", r"$\mathrm{SE}_2$", r"$\mathrm{SE}_3$", r"$\mathrm{SE}_4$",
                r"$\mathrm{SE}_{\mathrm{mean}}$"], "enhanced_golden_range_se_components"),
        "fe": ("fe", [f"fe_x{i}" for i in range(1, 5)] + ["fe_mean"],
               [r"$\mathrm{FE}_1$", r"$\mathrm{FE}_2$", r"$\mathrm{FE}_3$", r"$\mathrm{FE}_4$",
                r"$\mathrm{FE}_{\mathrm{mean}}$"], "enhanced_golden_range_fe"),
        "si": ("si", [f"si_x{i}" for i in range(1, 5)] + ["si_mean"],
               [r"$\mathrm{SI}_1$", r"$\mathrm{SI}_2$", r"$\mathrm{SI}_3$", r"$\mathrm{SI}_4$",
                r"$\mathrm{SI}_{\mathrm{mean}}$"], "enhanced_golden_range_scale_index"),
        "pe": ("pe", [f"pe_x{i}" for i in range(1, 5)] + ["pe_mean"],
               [r"$\mathrm{PE}_1$", r"$\mathrm{PE}_2$", r"$\mathrm{PE}_3$", r"$\mathrm{PE}_4$",
                r"$\mathrm{PE}_{\mathrm{mean}}$"], "enhanced_golden_range_pe"),
    }
    for metric, (key, cols, labels, stem) in metric_specs.items():
        rows = _read_csv(RESULTS / "expB_golden_range" / f"golden_range_{key}.csv")
        x = [int(r["group_index"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
        for col, label in zip(cols, labels):
            linewidth = 1.6 if col.endswith("mean") or col == "lyap_max" else 1.0
            ax.plot(x, [float(r[col]) for r in rows], linewidth=linewidth, label=label)
        ax.set_xlabel("Parameter-group index")
        ax.set_ylabel(labels[0] if len(labels) == 1 else "metric value")
        ax.tick_params(direction="in")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out_dir = base if metric != "si" else base / "scale_index"
        paths[stem] = _save(fig, out_dir, stem)
    return paths


def plot_si_diagnostics() -> dict:
    """Experiment C: scalogram overview + four inner-scalogram curves."""

    from metrics.nonperiodicity import (
        DEFAULT_SI_RAW_LENGTH,
        DEFAULT_SI_TRANSIENT,
        DEFAULT_SI_WAVELET,
        scale_index_cwt,
        scale_index_cwt_diagnostics,
    )
    from r1ly_rev1 import RECOMMENDED_A_ODD, RECOMMENDED_B_EVEN, synthesize_r1ly
    from chaos_maps.proposed_rev1 import ProposedNDRev1Map
    from utils.initializers import get_common_initial_state_float

    params = synthesize_r1ly(n=4, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[4])
    chaos_map = ProposedNDRev1Map(n=4, a_j1=params["a_j1"], a_jn=params["a_jn"], b=params["b"],
                                  initial_state_float=get_common_initial_state_float())
    trajectory = chaos_map.iterate(num_steps=DEFAULT_SI_RAW_LENGTH,
                                   discard_transient=DEFAULT_SI_TRANSIENT)
    summary = scale_index_cwt(trajectory, wavelet=DEFAULT_SI_WAVELET)
    component_diagnostics = {
        f"x{i}": scale_index_cwt_diagnostics(trajectory[:, i - 1], wavelet=DEFAULT_SI_WAVELET)
        for i in range(1, 5)
    }
    records = {
        "wavelet": DEFAULT_SI_WAVELET,
        "a11": RECOMMENDED_A_ODD,
        "b": params["b"],
        "system": {
            "name": "ProposedNDRev1Map",
            "si_components": [float(v) for v in summary["si_components"]],
            "si_mean": float(summary["si_mean"]),
            "component_diagnostics": {
                k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                    for kk, vv in v.items() if kk != "scalogram_full"}
                for k, v in component_diagnostics.items()
            },
        },
    }
    out_json = RESULTS / "expC_si_diagnostics" / "enhanced_scale_index_diagnostics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, ensure_ascii=False, default=float), encoding="utf-8")

    diag_dir = FIGURES / "camp_dynamic" / "scale_index" / "diagnostics"
    paths = {}

    # scalogram overview (2x2 heat panels)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), dpi=300)
    for component_index, ax in enumerate(axes.ravel(), start=1):
        diag = component_diagnostics[f"x{component_index}"]
        scales = np.asarray(diag["scales"], dtype=np.float64)
        scalogram = np.asarray(diag["scalogram"], dtype=np.float64)
        heat_image = np.tile(scalogram[:, None], (1, 32))
        ax.imshow(heat_image, aspect="auto", origin="lower",
                  extent=(0.0, 1.0, float(scales[0]), float(scales[-1])), cmap="viridis")
        ax.axhline(float(diag["study_scale_max"]), color="white", linestyle="--", linewidth=0.8)
        ax.axhline(float(diag["s_max_scale"]), color="red", linestyle="-", linewidth=0.9)
        ax.axhline(float(diag["s_min_scale"]), color="cyan", linestyle="-", linewidth=0.9)
        ax.set_xticks([])
        if component_index in (1, 3):
            ax.set_ylabel(r"$s$")
        ax.tick_params(direction="in")
        ax.text(0.03, 0.95, rf"$x_{{{component_index}}}$", transform=ax.transAxes,
                ha="left", va="top", color="white")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, wspace=0.14, hspace=0.16)
    paths["scalogram"] = _save(fig, diag_dir, "enhanced_scale_index_scalogram")

    # inner-scalogram curves
    for component_index in (1, 2, 3, 4):
        diag = component_diagnostics[f"x{component_index}"]
        scales = np.asarray(diag["scales"], dtype=np.float64)
        scalogram = np.asarray(diag["scalogram"], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)
        ax.plot(scales, scalogram, linewidth=1.7, color="C0")
        ax.axvline(float(diag["study_scale_max"]), color="0.5", linestyle="--",
                   linewidth=0.8, label=r"$s_1$")
        ax.scatter([float(diag["s_max_scale"])], [float(diag["s_max_value"])], s=28,
                   marker="o", label=r"$s_{\max}$")
        ax.scatter([float(diag["s_min_scale"])], [float(diag["s_min_value"])], s=28,
                   marker="s", label=r"$s_{\min}$")
        ax.set_xlabel(r"$s$")
        ax.set_ylabel(r"$S(s)$")
        ax.tick_params(direction="in")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.text(0.03, 0.95, rf"$x_{{{component_index}}}$", transform=ax.transAxes,
                ha="left", va="top")
        ax.text(0.03, 0.85, rf"$\mathrm{{SI}}={float(diag['si_value']):.4f}$",
                transform=ax.transAxes, ha="left", va="top")
        finite_values = scalogram[np.isfinite(scalogram)]
        if finite_values.size > 0:
            y_min = float(np.min(finite_values))
            y_max = float(np.max(finite_values))
            pad = max(0.05 * (y_max - y_min), 1e-4)
            ax.set_ylim(y_min - pad, y_max + pad)
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.96)
        paths[f"si_x{component_index}"] = _save(fig, diag_dir,
                                                f"enhanced_scale_index_si_x{component_index}")
    return paths


def plot_high_dim_comparison() -> dict:
    rows = _read_csv(RESULTS / "expE_high_dim_literature" / "high_dim_all_methods_summary.csv")
    order = ["proposed_rev1", "yao_complex_pascal", "sine_lattice_memristive",
             "cyclic_symmetric_logistic", "zang_linear_modular"]
    labels = {
        "proposed_rev1": "The proposed method",
        "yao_complex_pascal": "Yao et al.'s method",
        "sine_lattice_memristive": "He et al.'s method",
        "cyclic_symmetric_logistic": "Zhang et al.'s method",
        "zang_linear_modular": "Zang et al.'s method",
    }
    markers = {
        "proposed_rev1": "o",
        "yao_complex_pascal": "^",
        "sine_lattice_memristive": "D",
        "cyclic_symmetric_logistic": "v",
        "zang_linear_modular": "s",
    }
    metric_specs = [
        ("lambda_max", r"$\lambda_{\max}$", "high_dim_comparison_lambda_max"),
        ("se_mean", r"$\mathrm{SE}_{\mathrm{mean}}$", "high_dim_comparison_se_mean"),
        ("fe_mean", r"$\mathrm{FE}_{\mathrm{mean}}$", "high_dim_comparison_fe_mean"),
        ("si_mean", r"$\mathrm{SI}_{\mathrm{mean}}$", "high_dim_comparison_si_mean"),
        ("pe_mean", r"$\mathrm{PE}_{\mathrm{mean}}$", "high_dim_comparison_pe_mean"),
    ]
    out_dir = FIGURES / "high_dim_literature_comparison"
    paths = {}
    for metric_key, ylabel, stem in metric_specs:
        fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=300)
        for slug in order:
            method_rows = sorted([r for r in rows if r["method_slug"] == slug],
                                 key=lambda r: int(r["dimension"]))
            ax.plot([int(r["dimension"]) for r in method_rows],
                    [float(r[metric_key]) for r in method_rows],
                    marker=markers[slug], linewidth=1.4, markersize=4.0, label=labels[slug])
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(4, 13)))
        ax.tick_params(direction="in")
        ax.legend(frameon=False)
        fig.tight_layout()
        paths[stem] = _save(fig, out_dir, stem)
    return paths


def main() -> None:
    paths = {}
    paths.update(plot_golden_range())
    print("golden-range figures done")
    paths.update(plot_si_diagnostics())
    print("SI diagnostic figures done")
    paths.update(plot_high_dim_comparison())
    print("high-dim comparison figures done")
    (RESULTS / "figure_manifest.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
