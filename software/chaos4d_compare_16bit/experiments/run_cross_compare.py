# Maintainer: zhaowenbo
"""Run a minimal cross comparison between current implemented maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
from typing import Callable

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import (
    HCCDM4DMap,
    IDHNN4DMap,
    MHM4DMap,
    MPRC4DMap,
    MSLC4DMap,
    NDCS4DMap,
    Proposed4DBaseMap,
    Proposed4DEnhancedMap,
    URUK4DDCMMap,
)
from experiments.export_tables import (
    export_fe_comparison_records,
    export_le_comparison_records,
    export_pe_comparison_records,
    export_scale_index_comparison_records,
    export_se_comparison_records,
)
from metrics.fuzzy_entropy import (
    DEFAULT_FE_EMBEDDING_DIM,
    DEFAULT_FE_FUZZY_EXPONENT,
    DEFAULT_FE_RAW_LENGTH,
    DEFAULT_FE_TAU,
    DEFAULT_FE_TRANSIENT,
    fuzzy_entropy,
)
from metrics.lyapunov import estimate_max_lyapunov_wolf_full_state
from metrics.nonperiodicity import (
    DEFAULT_SI_RAW_LENGTH,
    DEFAULT_SI_TRANSIENT,
    DEFAULT_SI_WAVELET,
    scale_index_cwt,
)
from metrics.permutation_entropy import (
    DEFAULT_PE_EMBEDDING_DIM,
    DEFAULT_PE_NORMALIZED,
    DEFAULT_PE_RAW_LENGTH,
    DEFAULT_PE_TAU,
    DEFAULT_PE_TRANSIENT,
    permutation_entropy,
)
from metrics.spectral_entropy import (
    DEFAULT_SE_RAW_LENGTH,
    DEFAULT_SE_TRANSIENT,
    MIN_SE_RAW_LENGTH,
    spectral_entropy_complexity,
)
from utils.initializers import get_common_initial_state_float, get_common_initial_state_u16
from utils.io_utils import ensure_metric_output_dirs
from utils.trajectory import summarize_trajectory


MapFactory = Callable[[], object]


def build_map_factories(a11: int) -> dict[str, MapFactory]:
    """Build map factories for batch comparison.

    The proposed-family factories keep ``a11`` explicit so this script can later
    grow from one representative instance into a golden-interval family scan.
    The fixed coupling configuration remains the paper-facing default
    ``[50, 48, 46]`` through the existing map classes.
    """

    return {
        "Proposed4DBaseMap": lambda: Proposed4DBaseMap(A_odd=a11),
        "Proposed4DEnhancedMap": lambda: Proposed4DEnhancedMap(A_odd=a11),
        "NDCS4DMap": NDCS4DMap,
        "URUK4DDCMMap": URUK4DDCMMap,
        "MHM4DMap": MHM4DMap,
        "IDHNN4DMap": IDHNN4DMap,
        "MPRC4DMap": MPRC4DMap,
        "MSLC4DMap": MSLC4DMap,
        "HCCDM4DMap": HCCDM4DMap,
    }


def build_initialized_maps(a11: int) -> list[object]:
    """Build fresh map instances and reset each one to the common 4D state."""

    maps = [factory() for factory in build_map_factories(a11=a11).values()]
    initial_state = get_common_initial_state_float()
    for chaos_map in maps:
        chaos_map.reset(initial_state_float=initial_state)
    return maps


def build_le_comparison_factories(a11: int) -> dict[str, MapFactory]:
    """Build the paper-facing LE direct-comparison factory set."""

    factories = build_map_factories(a11=a11)
    return {
        "Proposed4DEnhancedMap": factories["Proposed4DEnhancedMap"],
        "NDCS4DMap": factories["NDCS4DMap"],
        "URUK4DDCMMap": factories["URUK4DDCMMap"],
        "MHM4DMap": factories["MHM4DMap"],
        "IDHNN4DMap": factories["IDHNN4DMap"],
        "MPRC4DMap": factories["MPRC4DMap"],
        "MSLC4DMap": factories["MSLC4DMap"],
        "HCCDM4DMap": factories["HCCDM4DMap"],
    }


def format_current_run_command() -> str:
    """Return the Python command used for this run."""

    return shlex.join([sys.executable, *sys.argv])


def list_common_initial_state_float() -> list[float]:
    """Return the shared 4D initial state as a JSON-serializable list."""

    return [float(value) for value in get_common_initial_state_float()]


def list_common_initial_state_u16() -> list[int]:
    """Return the shared 4D initial state in the quantized 16-bit domain."""

    return [int(value) for value in get_common_initial_state_u16()]


def fe_record_metadata(chaos_map: object, run_command: str) -> dict[str, object]:
    """Collect reproducibility metadata for one FE comparison row."""

    a_linear = getattr(chaos_map, "a_linear", None)
    a_coupling = getattr(chaos_map, "a_coupling", None)
    return {
        "initial_state_float": list_common_initial_state_float(),
        "initial_state_u16": list_common_initial_state_u16(),
        "a_linear": None if a_linear is None else [int(value) for value in a_linear],
        "a_coupling": None if a_coupling is None else [int(value) for value in a_coupling],
        "b": None if not hasattr(chaos_map, "b") else int(getattr(chaos_map, "b")),
        "run_command": run_command,
    }


def le_record_metadata(chaos_map: object, run_command: str) -> dict[str, object]:
    """Collect reproducibility metadata for one LE comparison row."""

    a_linear = getattr(chaos_map, "a_linear", None)
    a_coupling = getattr(chaos_map, "a_coupling", None)
    return {
        "initial_state_float": list_common_initial_state_float(),
        "initial_state_u16": list_common_initial_state_u16(),
        "fixed_point": "u16 / 16-bit",
        "a_linear": None if a_linear is None else [int(value) for value in a_linear],
        "a_coupling": None if a_coupling is None else [int(value) for value in a_coupling],
        "b": None if not hasattr(chaos_map, "b") else int(getattr(chaos_map, "b")),
        "run_command": run_command,
    }


def print_and_export_le_comparison(
    map_factories: dict[str, MapFactory],
    steps: int | None,
    discard: int | None,
    raw_output_dir: str,
) -> None:
    """Print and export the Wolf full-state LE comparison table."""

    records: list[dict[str, object]] = []
    max_steps = 5000 if steps is None else steps
    transient = 1000 if discard is None else discard
    run_command = format_current_run_command()
    print("Map                  LE Max       LE Mean      LE Std       Renorms    Final Dist     Chaotic")
    print("-" * 106)
    for chaos_map_factory in map_factories.values():
        chaos_map = chaos_map_factory()
        chaos_map.reset(initial_state_float=get_common_initial_state_float())
        metadata = le_record_metadata(chaos_map, run_command=run_command)
        result = estimate_max_lyapunov_wolf_full_state(
            chaos_map,
            max_steps=max_steps,
            transient=transient,
        )
        print(
            f"{chaos_map.name:<20} "
            f"{result['lyap_max']:<12.6f} "
            f"{result['lyap_mean']:<12.6f} "
            f"{result['lyap_std']:<12.6f} "
            f"{result['renormalizations']:<10d} "
            f"{result['final_distance']:<14.6e} "
            f"{str(result['is_chaotic'])}"
        )
        records.append(
            {
                "map_name": chaos_map.name,
                "lyap_max": float(result["lyap_max"]),
                "lyap_mean": float(result["lyap_mean"]),
                "lyap_std": float(result["lyap_std"]),
                "renormalizations": int(result["renormalizations"]),
                "final_distance": float(result["final_distance"]),
                "is_chaotic": bool(result["is_chaotic"]),
                "max_steps": int(max_steps),
                "transient": int(transient),
                **metadata,
            }
        )
    paths = export_le_comparison_records(records, output_dir=raw_output_dir)
    print(f"Saved CSV: {paths['csv']}")


def print_and_export_se_comparison(
    maps: list[object],
    steps: int | None,
    discard: int | None,
    se_mode: str,
    raw_output_dir: str,
) -> None:
    """Print the SE comparison table and export the CSV result."""

    records: list[dict[str, object]] = []
    print(
        "Map                  "
        "SE x1       SE x2       SE x3       SE x4       "
        "SE Mean     Raw Len   Test Len  Mode"
    )
    print("-" * 132)
    for chaos_map in maps:
        try:
            trajectory = chaos_map.iterate(
                num_steps=DEFAULT_SE_RAW_LENGTH if steps is None else steps,
                discard_transient=DEFAULT_SE_TRANSIENT if discard is None else discard,
            )
            result = spectral_entropy_complexity(trajectory, mode=se_mode)
            se_x1, se_x2, se_x3, se_x4 = result["se_components"]
        except Exception:
            se_x1 = se_x2 = se_x3 = se_x4 = float("nan")
            result = {"se_mean": float("nan"), "raw_length": 0, "test_length": 0, "mode": se_mode}
        print(
            f"{chaos_map.name:<20} "
            f"{se_x1:<11.6f} "
            f"{se_x2:<11.6f} "
            f"{se_x3:<11.6f} "
            f"{se_x4:<11.6f} "
            f"{result['se_mean']:<11.6f} "
            f"{result['raw_length']:<9d} "
            f"{result['test_length']:<9d} "
            f"{result['mode']}"
        )
        records.append(
            {
                "map_name": chaos_map.name,
                "se_x1": float(se_x1),
                "se_x2": float(se_x2),
                "se_x3": float(se_x3),
                "se_x4": float(se_x4),
                "se_mean": float(result["se_mean"]),
                "raw_length": int(result["raw_length"]),
                "test_length": int(result["test_length"]),
                "mode": str(result["mode"]),
            }
        )
    paths = export_se_comparison_records(records, output_dir=raw_output_dir)
    print(f"Saved CSV: {paths['csv']}")


def print_and_export_fe_comparison(
    maps: list[object],
    steps: int | None,
    discard: int | None,
    raw_output_dir: str,
    fe_m: int,
    fe_r: float | None,
    fe_tau: int,
    fe_n: int,
) -> None:
    """Print the FE comparison table and export the CSV result."""

    records: list[dict[str, object]] = []
    run_command = format_current_run_command()
    print(
        "Map                  "
        "FE x1       FE x2       FE x3       FE x4       "
        "FE Mean     Raw Len   Test Len  m   tau n   Mode"
    )
    print("-" * 140)
    for chaos_map in maps:
        metadata = fe_record_metadata(chaos_map, run_command=run_command)
        try:
            chaos_map.reset(initial_state_float=get_common_initial_state_float())
            trajectory = chaos_map.iterate(
                num_steps=DEFAULT_FE_RAW_LENGTH if steps is None else steps,
                discard_transient=DEFAULT_FE_TRANSIENT if discard is None else discard,
            )
            result = fuzzy_entropy(
                trajectory,
                mode="direct",
                m=fe_m,
                r=fe_r,
                tau=fe_tau,
                n=fe_n,
            )
            fe_x1, fe_x2, fe_x3, fe_x4 = result["fe_components"]
        except Exception:
            fe_x1 = fe_x2 = fe_x3 = fe_x4 = float("nan")
            result = {
                "fe_mean": float("nan"),
                "raw_length": 0,
                "test_length": 0,
                "mode": "direct",
                "m": fe_m,
                "r": fe_r if fe_r is not None else "0.2*std(component_u16_sequence)",
                "tau": fe_tau,
                "n": fe_n,
            }
        print(
            f"{chaos_map.name:<20} "
            f"{fe_x1:<11.6f} "
            f"{fe_x2:<11.6f} "
            f"{fe_x3:<11.6f} "
            f"{fe_x4:<11.6f} "
            f"{result['fe_mean']:<11.6f} "
            f"{result['raw_length']:<9d} "
            f"{result['test_length']:<9d} "
            f"{result['m']:<3d} "
            f"{result['tau']:<3d} "
            f"{result['n']:<3d} "
            f"{result['mode']}"
        )
        records.append(
            {
                "map_name": chaos_map.name,
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
                **metadata,
            }
        )
    paths = export_fe_comparison_records(records, output_dir=raw_output_dir)
    print(f"Saved CSV: {paths['csv']}")


def print_and_export_scale_index_comparison(
    maps: list[object],
    steps: int | None,
    discard: int | None,
    raw_output_dir: str,
    si_wavelet: str,
) -> None:
    """Print the scale-index comparison table and export the CSV result."""

    records: list[dict[str, object]] = []
    print(
        "Map                  "
        "SI x1       SI x2       SI x3       SI x4       "
        "SI Mean     Raw Len   Test Len  Wavelet"
    )
    print("-" * 136)
    for chaos_map in maps:
        try:
            trajectory = chaos_map.iterate(
                num_steps=DEFAULT_SI_RAW_LENGTH if steps is None else steps,
                discard_transient=DEFAULT_SI_TRANSIENT if discard is None else discard,
            )
            result = scale_index_cwt(trajectory, mode="direct", wavelet=si_wavelet)
            si_x1, si_x2, si_x3, si_x4 = result["si_components"]
        except Exception:
            si_x1 = si_x2 = si_x3 = si_x4 = float("nan")
            result = {
                "si_mean": float("nan"),
                "raw_length": 0,
                "test_length": 0,
                "mode": "direct",
                "wavelet": si_wavelet,
                "scales": [],
            }
        print(
            f"{chaos_map.name:<20} "
            f"{si_x1:<11.6f} "
            f"{si_x2:<11.6f} "
            f"{si_x3:<11.6f} "
            f"{si_x4:<11.6f} "
            f"{result['si_mean']:<11.6f} "
            f"{result['raw_length']:<9d} "
            f"{result['test_length']:<9d} "
            f"{result['wavelet']}"
        )
        records.append(
            {
                "map_name": chaos_map.name,
                "si_x1": float(si_x1),
                "si_x2": float(si_x2),
                "si_x3": float(si_x3),
                "si_x4": float(si_x4),
                "si_mean": float(result["si_mean"]),
                "raw_length": int(result["raw_length"]),
                "test_length": int(result["test_length"]),
                "mode": str(result["mode"]),
                "wavelet": str(result["wavelet"]),
                "scales": result["scales"],
            }
        )
    paths = export_scale_index_comparison_records(records, output_dir=raw_output_dir)
    print(f"Saved CSV: {paths['csv']}")


def print_and_export_pe_comparison(
    maps: list[object],
    steps: int | None,
    discard: int | None,
    raw_output_dir: str,
    pe_m: int,
    pe_tau: int,
    pe_normalized: bool,
) -> None:
    """Print the permutation-entropy comparison table and export the CSV result."""

    records: list[dict[str, object]] = []
    print(
        "Map                  "
        "PE x1       PE x2       PE x3       PE x4       "
        "PE Mean     Raw Len   Test Len  m   tau normalized"
    )
    print("-" * 144)
    for chaos_map in maps:
        try:
            trajectory = chaos_map.iterate(
                num_steps=DEFAULT_PE_RAW_LENGTH if steps is None else steps,
                discard_transient=DEFAULT_PE_TRANSIENT if discard is None else discard,
            )
            result = permutation_entropy(
                trajectory,
                mode="direct",
                m=pe_m,
                tau=pe_tau,
                normalized=pe_normalized,
            )
            pe_x1, pe_x2, pe_x3, pe_x4 = result["pe_components"]
        except Exception:
            pe_x1 = pe_x2 = pe_x3 = pe_x4 = float("nan")
            result = {
                "pe_mean": float("nan"),
                "raw_length": 0,
                "test_length": 0,
                "mode": "direct",
                "m": pe_m,
                "tau": pe_tau,
                "normalized": pe_normalized,
            }
        print(
            f"{chaos_map.name:<20} "
            f"{pe_x1:<11.6f} "
            f"{pe_x2:<11.6f} "
            f"{pe_x3:<11.6f} "
            f"{pe_x4:<11.6f} "
            f"{result['pe_mean']:<11.6f} "
            f"{result['raw_length']:<9d} "
            f"{result['test_length']:<9d} "
            f"{result['m']:<3d} "
            f"{result['tau']:<3d} "
            f"{str(result['normalized'])}"
        )
        records.append(
            {
                "map_name": chaos_map.name,
                "pe_x1": float(pe_x1),
                "pe_x2": float(pe_x2),
                "pe_x3": float(pe_x3),
                "pe_x4": float(pe_x4),
                "pe_mean": float(result["pe_mean"]),
                "raw_length": int(result["raw_length"]),
                "test_length": int(result["test_length"]),
                "mode": str(result["mode"]),
                "m": int(result["m"]),
                "tau": int(result["tau"]),
                "normalized": bool(result["normalized"]),
            }
        )
    paths = export_pe_comparison_records(records, output_dir=raw_output_dir)
    print(f"Saved CSV: {paths['csv']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current implemented maps.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--discard", type=int, default=None)
    parser.add_argument("--metric", choices=["summary", "le", "se", "fe", "si", "pe", "all"], default="all")
    parser.add_argument(
        "--a11",
        type=int,
        default=371,
        help="Representative proposed-family a11 value. Golden-interval family scan is reserved for a later round.",
    )
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
    args = parser.parse_args()
    if args.metric in ("se", "all") and args.steps is not None and args.steps < MIN_SE_RAW_LENGTH:
        raise ValueError(f"SE complexity requires --steps >= {MIN_SE_RAW_LENGTH}.")

    if args.metric == "le":
        # TODO: extend this entry to scan the proposed-family golden interval
        # 271 <= a11 <= 507 with fixed coupling gains [50, 48, 46].
        print_and_export_le_comparison(
            build_le_comparison_factories(args.a11),
            args.steps,
            args.discard,
            args.raw_output_dir,
        )
        return
    if args.metric == "se":
        print_and_export_se_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            args.se_mode,
            args.raw_output_dir,
        )
        return
    if args.metric == "fe":
        print_and_export_fe_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            args.raw_output_dir,
            args.fe_m,
            args.fe_r,
            args.fe_tau,
            args.fe_n,
        )
        return
    if args.metric == "si":
        print_and_export_scale_index_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            args.raw_output_dir,
            args.si_wavelet,
        )
        return
    if args.metric == "pe":
        pe_dirs = ensure_metric_output_dirs("permutation_entropy")
        pe_raw_dir = str(pe_dirs["raw"]) if args.raw_output_dir == "results/raw" else args.raw_output_dir
        print_and_export_pe_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            pe_raw_dir,
            args.pe_m,
            args.pe_tau,
            not args.pe_unnormalized,
        )
        return
    if args.metric == "all":
        print("Scale Index Comparison")
        print()
        print_and_export_scale_index_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            args.raw_output_dir,
            args.si_wavelet,
        )
        print()
        print("FE Comparison")
        print()
        print_and_export_fe_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            args.raw_output_dir,
            args.fe_m,
            args.fe_r,
            args.fe_tau,
            args.fe_n,
        )
        print()
        print("PE Comparison")
        print()
        pe_dirs = ensure_metric_output_dirs("permutation_entropy")
        pe_raw_dir = str(pe_dirs["raw"]) if args.raw_output_dir == "results/raw" else args.raw_output_dir
        print_and_export_pe_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            pe_raw_dir,
            args.pe_m,
            args.pe_tau,
            not args.pe_unnormalized,
        )
        print()
        print("LE Comparison")
        print()
        print_and_export_le_comparison(
            build_le_comparison_factories(args.a11),
            args.steps,
            args.discard,
            args.raw_output_dir,
        )
        print()
        print("SE Comparison")
        print()
        print_and_export_se_comparison(
            build_initialized_maps(args.a11),
            args.steps,
            args.discard,
            args.se_mode,
            args.raw_output_dir,
        )
        return

    maps = build_initialized_maps(args.a11)
    print("Map                  Mean                                      Variance")
    print("-" * 78)
    for chaos_map in maps:
        traj = chaos_map.iterate(
            num_steps=1000 if args.steps is None else args.steps,
            discard_transient=0 if args.discard is None else args.discard,
        )
        summary = summarize_trajectory(traj)
        print(f"{chaos_map.name:<20} {summary['mean']} {summary['var']}")


if __name__ == "__main__":
    main()
