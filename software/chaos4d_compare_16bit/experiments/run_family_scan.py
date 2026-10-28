# Maintainer: zhaowenbo
"""CLI entry for proposed-family golden-range scans."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from experiments.plot_golden_range_le import run_proposed_golden_range_le_scan
from experiments.plot_golden_range_fe import run_enhanced_golden_range_fe_scan
from experiments.plot_golden_range_pe import run_enhanced_golden_range_pe_scan
from experiments.plot_golden_range_se import run_enhanced_golden_range_se_scan
from experiments.plot_golden_range_si import run_enhanced_golden_range_scale_index_scan
from metrics.fuzzy_entropy import (
    DEFAULT_FE_EMBEDDING_DIM,
    DEFAULT_FE_FUZZY_EXPONENT,
    DEFAULT_FE_RAW_LENGTH,
    DEFAULT_FE_TAU,
    DEFAULT_FE_TRANSIENT,
)
from metrics.nonperiodicity import DEFAULT_SI_RAW_LENGTH, DEFAULT_SI_TRANSIENT, DEFAULT_SI_WAVELET
from metrics.permutation_entropy import (
    DEFAULT_PE_EMBEDDING_DIM,
    DEFAULT_PE_NORMALIZED,
    DEFAULT_PE_RAW_LENGTH,
    DEFAULT_PE_TAU,
    DEFAULT_PE_TRANSIENT,
)
from metrics.spectral_entropy import MIN_SE_RAW_LENGTH
from utils.io_utils import ensure_metric_output_dirs


def main() -> None:
    """Run golden-range LE, SE, or FE scans from the command line.

    The current default no-argument workflow is the enhanced-map FuzzyEn scan.
    """

    parser = argparse.ArgumentParser(description="Run golden-range scans for the proposed 4D family.")
    parser.add_argument("--map", choices=["proposed_base", "proposed_enhanced"], default="proposed_enhanced")
    parser.add_argument("--metric", choices=["le", "se", "fe", "si", "pe"], default="fe")
    parser.add_argument("--a11-start", type=int, default=271)
    parser.add_argument("--a11-stop", type=int, default=507)
    parser.add_argument("--a11-step", type=int, default=2)
    parser.add_argument("--transient", type=int, default=DEFAULT_FE_TRANSIENT)
    parser.add_argument("--steps", type=int, default=DEFAULT_FE_RAW_LENGTH)
    parser.add_argument("--se-mode", choices=["direct", "byte_test", "character"], default="direct")
    parser.add_argument("--fe-m", type=int, default=DEFAULT_FE_EMBEDDING_DIM)
    parser.add_argument("--fe-r", type=float, default=None)
    parser.add_argument("--fe-tau", type=int, default=DEFAULT_FE_TAU)
    parser.add_argument("--fe-n", type=int, default=DEFAULT_FE_FUZZY_EXPONENT)
    parser.add_argument("--si-wavelet", default=DEFAULT_SI_WAVELET)
    parser.add_argument("--pe-m", type=int, default=DEFAULT_PE_EMBEDDING_DIM)
    parser.add_argument("--pe-tau", type=int, default=DEFAULT_PE_TAU)
    parser.add_argument("--pe-unnormalized", action="store_true")
    parser.add_argument("--raw-output-dir", default="results/raw")
    parser.add_argument("--figure-output-dir", default="results/figures")
    args = parser.parse_args()
    if args.metric == "se" and args.steps < MIN_SE_RAW_LENGTH:
        raise ValueError(f"SE complexity requires --steps >= {MIN_SE_RAW_LENGTH}.")

    if args.metric == "se":
        if args.map != "proposed_enhanced":
            raise ValueError("Golden-range SE scan only supports proposed_enhanced.")
        result = run_enhanced_golden_range_se_scan(
            a11_start=args.a11_start,
            a11_stop=args.a11_stop,
            a11_step=args.a11_step,
            transient=args.transient,
            seq_length=args.steps,
            mode=args.se_mode,
            raw_output_dir=args.raw_output_dir,
            figure_output_dir=args.figure_output_dir,
            progress=True,
        )
        print(f"Completed {result['valid_count']}/{result['total_count']} valid SE points.")
        print(f"Saved CSV: {result['data_paths']['csv']}")
        print(f"Saved PDF: {result['figure_paths']['pdf']}")
        print(f"Saved EPS: {result['figure_paths']['eps']}")
        return
    if args.metric == "fe":
        if args.map != "proposed_enhanced":
            raise ValueError("Golden-range FE scan only supports proposed_enhanced.")
        result = run_enhanced_golden_range_fe_scan(
            a11_start=args.a11_start,
            a11_stop=args.a11_stop,
            a11_step=args.a11_step,
            transient=args.transient,
            seq_length=args.steps,
            m=args.fe_m,
            r=args.fe_r,
            tau=args.fe_tau,
            n=args.fe_n,
            raw_output_dir=args.raw_output_dir,
            figure_output_dir=args.figure_output_dir,
            progress=True,
        )
        print(f"Completed {result['valid_count']}/{result['total_count']} valid FE points.")
        print(f"Saved CSV: {result['data_paths']['csv']}")
        print(f"Saved PDF: {result['figure_paths']['pdf']}")
        print(f"Saved EPS: {result['figure_paths']['eps']}")
        return
    if args.metric == "si":
        if args.map != "proposed_enhanced":
            raise ValueError("Golden-range SI scan only supports proposed_enhanced.")
        result = run_enhanced_golden_range_scale_index_scan(
            a11_start=args.a11_start,
            a11_stop=args.a11_stop,
            a11_step=args.a11_step,
            transient=DEFAULT_SI_TRANSIENT if args.transient == DEFAULT_FE_TRANSIENT else args.transient,
            seq_length=DEFAULT_SI_RAW_LENGTH if args.steps == DEFAULT_FE_RAW_LENGTH else args.steps,
            wavelet=args.si_wavelet,
            raw_output_dir=args.raw_output_dir,
            figure_output_dir=args.figure_output_dir,
            progress=True,
        )
        print(f"Completed {result['valid_count']}/{result['total_count']} valid SI points.")
        print(f"Saved CSV: {result['data_paths']['csv']}")
        print(f"Saved PDF: {result['figure_paths']['pdf']}")
        print(f"Saved EPS: {result['figure_paths']['eps']}")
        return
    if args.metric == "pe":
        if args.map != "proposed_enhanced":
            raise ValueError("Golden-range PE scan only supports proposed_enhanced.")
        pe_dirs = ensure_metric_output_dirs("permutation_entropy")
        pe_raw_dir = str(pe_dirs["raw"]) if args.raw_output_dir == "results/raw" else args.raw_output_dir
        pe_figure_dir = str(pe_dirs["figures"]) if args.figure_output_dir == "results/figures" else args.figure_output_dir
        result = run_enhanced_golden_range_pe_scan(
            a11_start=args.a11_start,
            a11_stop=args.a11_stop,
            a11_step=args.a11_step,
            transient=DEFAULT_PE_TRANSIENT if args.transient == DEFAULT_FE_TRANSIENT else args.transient,
            seq_length=DEFAULT_PE_RAW_LENGTH if args.steps == DEFAULT_FE_RAW_LENGTH else args.steps,
            m=args.pe_m,
            tau=args.pe_tau,
            normalized=DEFAULT_PE_NORMALIZED if not args.pe_unnormalized else False,
            raw_output_dir=pe_raw_dir,
            figure_output_dir=pe_figure_dir,
            progress=True,
        )
        print(f"Completed {result['valid_count']}/{result['total_count']} valid PE points.")
        print(f"Saved CSV: {result['data_paths']['csv']}")
        print(f"Saved PDF: {result['figure_paths']['pdf']}")
        print(f"Saved EPS: {result['figure_paths']['eps']}")
        return

    result = run_proposed_golden_range_le_scan(
        variant=args.map,
        a11_start=args.a11_start,
        a11_stop=args.a11_stop,
        a11_step=args.a11_step,
        transient=args.transient,
        max_steps=args.steps,
        raw_output_dir=args.raw_output_dir,
        figure_output_dir=args.figure_output_dir,
        progress=True,
    )
    print(f"Completed {result['valid_count']}/{result['total_count']} valid LE points.")
    print(f"Saved JSON: {result['data_paths']['json']}")
    print(f"Saved CSV: {result['data_paths']['csv']}")
    print(f"Saved PDF: {result['figure_paths']['pdf']}")
    print(f"Saved PNG: {result['figure_paths']['png']}")


if __name__ == "__main__":
    main()
