# Maintainer: zhaowenbo
"""Task builders for experiments A-E of the rev1 alignment subsystem."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from r1ly_rev1 import RECOMMENDED_A_ODD, RECOMMENDED_B_EVEN, golden_range_a_values, synthesize_r1ly
from utils.initializers import get_common_initial_state_u16, get_dimension_initial_state_u16
from chaos_maps.literature_high_dim import get_linear_modular_initial_state_u16

METRICS = ["le", "se", "fe", "si", "pe"]

BASELINE_4D = [
    ("4D-NDCS", "chaos_maps.ndcs_4d.NDCS4DMap"),
    ("URUK-4D-DCM", "chaos_maps.uruk_4d_dcm.URUK4DDCMMap"),
    ("4D-MHM", "chaos_maps.mhm_4d.MHM4DMap"),
    ("4D-IDHNN", "chaos_maps.idhnn_4d.IDHNN4DMap"),
    ("4D-MPRC", "chaos_maps.mprc_4d.MPRC4DMap"),
    ("4D-MSLC", "chaos_maps.mslc_4d.MSLC4DMap"),
    ("4D-HCCDM", "chaos_maps.hccdm_4d.HCCDM4DMap"),
]

LITERATURE_METHODS = [
    ("Yao et al.'s method", "yao_complex_pascal", "chaos_maps.literature_high_dim.YaoComplexPascalMap"),
    ("He et al.'s method", "sine_lattice_memristive", "chaos_maps.literature_high_dim.SineMapLatticeMemristiveMap"),
    ("Zhang et al.'s method", "cyclic_symmetric_logistic", "chaos_maps.literature_high_dim.CyclicSymmetricLogisticMap"),
    ("Zang et al.'s method", "zang_linear_modular", "chaos_maps.literature_high_dim.ZangLinearModularMap"),
]


def _init_u16(values) -> dict[str, Any]:
    return {"kind": "u16", "values": [int(v) for v in values]}


def proposed_params_4d(a_odd: int = RECOMMENDED_A_ODD) -> dict[str, Any]:
    return synthesize_r1ly(n=4, A_odd=a_odd, B_even=RECOMMENDED_B_EVEN[4])


def proposed_map_spec(n: int, params: dict[str, Any], init: dict[str, Any]) -> dict[str, Any]:
    return {
        "display_name": "The proposed method",
        "slug": "proposed_rev1",
        "class_path": "chaos_maps.proposed_rev1.ProposedNDRev1Map",
        "kwargs": {
            "n": int(n),
            "a_j1": [int(v) for v in params["a_j1"]],
            "a_jn": [int(v) for v in params["a_jn"]],
            "b": int(params["b"]),
        },
        "init": init,
        "r1ly_params": params,
    }


def baseline_map_spec(display: str, class_path: str, init: dict[str, Any]) -> dict[str, Any]:
    slug = class_path.rsplit(".", 1)[1].lower()
    return {
        "display_name": display,
        "slug": slug,
        "class_path": class_path,
        "kwargs": {},
        "init": init,
    }


def literature_map_spec(display: str, slug: str, class_path: str, n: int, init: dict[str, Any]) -> dict[str, Any]:
    return {
        "display_name": display,
        "slug": slug,
        "class_path": class_path,
        "kwargs": {"n": int(n)},
        "init": init,
    }


def make_tasks(experiment: str, map_spec: dict[str, Any], metrics: list[str] | None = None) -> list[dict[str, Any]]:
    """Expand one map spec into one task per metric."""

    tasks = []
    for metric in (metrics or METRICS):
        tasks.append({
            "experiment": experiment,
            "metric": metric,
            "map": map_spec,
        })
    return tasks


def tasks_experiment_a() -> list[dict[str, Any]]:
    """4D direct comparison: proposed (new core + Mode II) vs 7 baselines."""

    init = _init_u16(get_common_initial_state_u16())
    params = proposed_params_4d()
    tasks = make_tasks("A_4d_comparison", proposed_map_spec(4, params, init))
    for display, class_path in BASELINE_4D:
        tasks += make_tasks("A_4d_comparison", baseline_map_spec(display, class_path, init))
    return tasks


def tasks_experiment_b() -> list[dict[str, Any]]:
    """Golden-Range scan: 119 odd A in [271, 507], gains [A,A+2,A+4],
    coupling [50,48,46], per-group b = 3*A//10."""

    init = _init_u16(get_common_initial_state_u16())
    tasks = []
    for a_odd in golden_range_a_values():
        params = synthesize_r1ly(n=4, A_odd=a_odd, B_even=RECOMMENDED_B_EVEN[4])
        spec = proposed_map_spec(4, params, init)
        spec["slug"] = f"proposed_rev1_A{a_odd}"
        spec["golden_group"] = {"A_odd": int(a_odd), "b": int(params["b"])}
        tasks += make_tasks("B_golden_range", spec)
    return tasks


def tasks_experiment_d() -> list[dict[str, Any]]:
    """Single-family scalability n=4..12 with the recommended B_even table."""

    tasks = []
    for n in range(4, 13):
        init = _init_u16(get_dimension_initial_state_u16(n))
        params = synthesize_r1ly(n=n, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[n])
        tasks += make_tasks("D_dimension_scalability", proposed_map_spec(n, params, init))
    return tasks


def tasks_experiment_e() -> list[dict[str, Any]]:
    """High-dimensional literature comparison: 5 methods x n=4..12."""

    tasks = []
    for n in range(4, 13):
        init = _init_u16(get_linear_modular_initial_state_u16(n))  # x_i(0) = 100*i
        params = synthesize_r1ly(n=n, A_odd=RECOMMENDED_A_ODD, B_even=RECOMMENDED_B_EVEN[n])
        spec = proposed_map_spec(n, params, init)
        spec["slug"] = "proposed_rev1"
        tasks += make_tasks("E_high_dim_literature", spec)
        for display, slug, class_path in LITERATURE_METHODS:
            tasks += make_tasks(
                "E_high_dim_literature",
                literature_map_spec(display, slug, class_path, n, init),
            )
    return tasks


def all_tasks() -> list[dict[str, Any]]:
    return (
        tasks_experiment_a()
        + tasks_experiment_b()
        + tasks_experiment_d()
        + tasks_experiment_e()
    )
