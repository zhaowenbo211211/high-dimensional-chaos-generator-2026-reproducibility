# Maintainer: zhaowenbo
"""Run a single 4D chaotic map experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

import numpy as np

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
from utils.initializers import get_common_initial_state_float
from utils.trajectory import summarize_trajectory


MapFactory = Callable[[], object]


def build_map_factories(a11: int) -> dict[str, MapFactory]:
    """Build supported map factories for representative-instance experiments.

    The proposed-family entries keep ``a11`` configurable so the same CLI can
    later grow into golden-interval scans without redesigning the LE interface.
    The current round still runs only one representative instance at a time.
    """

    return {
        "proposed_base": lambda: Proposed4DBaseMap(A_odd=a11),
        "proposed_enhanced": lambda: Proposed4DEnhancedMap(A_odd=a11),
        "ndcs_4d": NDCS4DMap,
        "uruk_4d_dcm": URUK4DDCMMap,
        "mhm_4d": MHM4DMap,
        "idhnn_4d": IDHNN4DMap,
        "mprc_4d": MPRC4DMap,
        "mslc_4d": MSLC4DMap,
        "hccdm_4d": HCCDM4DMap,
    }


def build_map(map_name: str, a11: int = 371):
    """Construct a supported map from CLI name."""

    factories = build_map_factories(a11=a11)
    try:
        chaos_map = factories[map_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported map: {map_name}") from exc
    chaos_map.reset(initial_state_float=get_common_initial_state_float())
    return chaos_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single 4D chaotic map.")
    parser.add_argument(
        "--map",
        default="proposed_base",
        choices=[
            "proposed_base",
            "proposed_enhanced",
            "ndcs_4d",
            "uruk_4d_dcm",
            "mhm_4d",
            "idhnn_4d",
            "mprc_4d",
            "mslc_4d",
            "hccdm_4d",
        ],
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--discard", type=int, default=None)
    parser.add_argument("--metric", choices=["trajectory", "le", "se", "fe", "si", "pe"], default="trajectory")
    parser.add_argument(
        "--a11",
        type=int,
        default=371,
        help="Representative proposed-family a11 value. Later golden-interval scans use 271..507.",
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
    args = parser.parse_args()
    if args.metric == "se" and args.steps is not None and args.steps < MIN_SE_RAW_LENGTH:
        raise ValueError(f"SE complexity requires --steps >= {MIN_SE_RAW_LENGTH}.")

    chaos_map = build_map(args.map, a11=args.a11)
    if args.metric == "le":
        result = estimate_max_lyapunov_wolf_full_state(
            chaos_map,
            max_steps=5000 if args.steps is None else args.steps,
            transient=1000 if args.discard is None else args.discard,
        )
        print(f"Map: {chaos_map.name}")
        print(f"lyap_max: {result['lyap_max']}")
        print(f"lyap_mean: {result['lyap_mean']}")
        print(f"lyap_std: {result['lyap_std']}")
        print(f"lyap_components: {np.asarray(result['lyap_components'])}")
        print(f"renormalizations: {result['renormalizations']}")
        print(f"final_distance: {result['final_distance']}")
        print(f"is_chaotic: {result['is_chaotic']}")
        return
    if args.metric == "se":
        trajectory = chaos_map.iterate(
            num_steps=DEFAULT_SE_RAW_LENGTH if args.steps is None else args.steps,
            discard_transient=DEFAULT_SE_TRANSIENT if args.discard is None else args.discard,
        )
        result = spectral_entropy_complexity(trajectory, mode=args.se_mode)
        print(f"Map: {chaos_map.name}")
        print(f"se_components: {np.asarray(result['se_components'])}")
        print(f"se_mean: {result['se_mean']}")
        print(f"raw_length: {result['raw_length']}")
        print(f"test_length: {result['test_length']}")
        print(f"mode: {result['mode']}")
        return
    if args.metric == "fe":
        trajectory = chaos_map.iterate(
            num_steps=DEFAULT_FE_RAW_LENGTH if args.steps is None else args.steps,
            discard_transient=DEFAULT_FE_TRANSIENT if args.discard is None else args.discard,
        )
        result = fuzzy_entropy(
            trajectory,
            mode="direct",
            m=args.fe_m,
            r=args.fe_r,
            tau=args.fe_tau,
            n=args.fe_n,
        )
        print(f"Map: {chaos_map.name}")
        print(f"fe_components: {np.asarray(result['fe_components'])}")
        print(f"fe_mean: {result['fe_mean']}")
        print(f"raw_length: {result['raw_length']}")
        print(f"test_length: {result['test_length']}")
        print(f"mode: {result['mode']}")
        print(f"m: {result['m']}")
        print(f"r: {result['r']}")
        print(f"tau: {result['tau']}")
        print(f"n: {result['n']}")
        return
    if args.metric == "si":
        trajectory = chaos_map.iterate(
            num_steps=DEFAULT_SI_RAW_LENGTH if args.steps is None else args.steps,
            discard_transient=DEFAULT_SI_TRANSIENT if args.discard is None else args.discard,
        )
        result = scale_index_cwt(trajectory, mode="direct", wavelet=args.si_wavelet)
        print(f"Map: {chaos_map.name}")
        print(f"si_components: {np.asarray(result['si_components'])}")
        print(f"si_mean: {result['si_mean']}")
        print(f"raw_length: {result['raw_length']}")
        print(f"test_length: {result['test_length']}")
        print(f"mode: {result['mode']}")
        print(f"wavelet: {result['wavelet']}")
        print(f"scales: {np.asarray(result['scales'])}")
        return
    if args.metric == "pe":
        trajectory = chaos_map.iterate(
            num_steps=DEFAULT_PE_RAW_LENGTH if args.steps is None else args.steps,
            discard_transient=DEFAULT_PE_TRANSIENT if args.discard is None else args.discard,
        )
        result = permutation_entropy(
            trajectory,
            mode="direct",
            m=args.pe_m,
            tau=args.pe_tau,
            normalized=not args.pe_unnormalized,
        )
        print(f"Map: {chaos_map.name}")
        print(f"pe_components: {np.asarray(result['pe_components'])}")
        print(f"pe_mean: {result['pe_mean']}")
        print(f"raw_length: {result['raw_length']}")
        print(f"test_length: {result['test_length']}")
        print(f"mode: {result['mode']}")
        print(f"m: {result['m']}")
        print(f"tau: {result['tau']}")
        print(f"normalized: {result['normalized']}")
        return

    traj = chaos_map.iterate(
        num_steps=1000 if args.steps is None else args.steps,
        discard_transient=0 if args.discard is None else args.discard,
    )
    summary = summarize_trajectory(traj)
    preview_count = min(5, len(traj))

    print(f"Map: {chaos_map.name}")
    print(f"Trajectory shape: {traj.shape}")
    print(f"First {preview_count} states:")
    print(traj[:preview_count])
    print("Last state:")
    print(traj[-1])
    print("Per-dimension mean:")
    print(np.asarray(summary["mean"]))
    print("Per-dimension variance:")
    print(np.asarray(summary["var"]))


if __name__ == "__main__":
    main()
