# Maintainer: zhaowenbo
from pathlib import Path
import sys

import numpy as np
import pytest

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import (
    CyclicSymmetricLogisticMap,
    ProposedNDEnhancedMap,
    SineMapLatticeMemristiveMap,
    YaoComplexPascalMap,
    ZangLinearModularMap,
)
from experiments.run_high_dim_literature_comparison import (
    METHODS,
    _make_map,
    build_comparison_initial_state_u16,
    get_method_dimension_support,
)
from metrics.fuzzy_entropy import fuzzy_entropy
from metrics.lyapunov import wolf_lyapunov_full_state_nd
from metrics.nonperiodicity import scale_index_cwt
from metrics.permutation_entropy import permutation_entropy
from metrics.spectral_entropy import spectral_entropy_complexity


LITERATURE_MAP_CLASSES = [
    ProposedNDEnhancedMap,
    YaoComplexPascalMap,
    SineMapLatticeMemristiveMap,
    CyclicSymmetricLogisticMap,
    ZangLinearModularMap,
]


def assert_u16_state_is_valid(state: np.ndarray, n: int) -> None:
    assert state.shape == (n,)
    assert np.issubdtype(state.dtype, np.integer)
    assert np.all((state >= 0) & (state <= 65535))


def test_all_high_dim_literature_maps_initialize_with_comparison_state() -> None:
    initial_state = build_comparison_initial_state_u16(4)
    assert initial_state.tolist() == [100, 200, 300, 400]

    for map_class in LITERATURE_MAP_CLASSES:
        chaos_map = map_class(n=4, initial_state_u16=initial_state)
        assert_u16_state_is_valid(chaos_map.get_state_u16(), 4)
        assert chaos_map.get_state_u16().tolist() == [100, 200, 300, 400]


def test_complex_literature_map_imaginary_initial_state_is_fixed_zero() -> None:
    initial_state = build_comparison_initial_state_u16(4)
    chaos_map = YaoComplexPascalMap(n=4, initial_state_u16=initial_state)

    assert chaos_map.get_state_u16().tolist() == [100, 200, 300, 400]
    assert chaos_map.get_imag_u16().tolist() == [0, 0, 0, 0]

    with pytest.raises(ValueError, match="imaginary initial state must be fixed to zero"):
        YaoComplexPascalMap(n=4, initial_state_u16=initial_state, initial_imag_u16=[1, 0, 0, 0])


def test_all_high_dim_literature_maps_step_to_valid_u16_state() -> None:
    initial_state = build_comparison_initial_state_u16(4)

    for map_class in LITERATURE_MAP_CLASSES:
        chaos_map = map_class(n=4, initial_state_u16=initial_state)
        state_float = chaos_map.step()
        assert state_float.shape == (4,)
        assert np.all(np.isfinite(state_float))
        assert np.all((state_float >= 0.0) & (state_float < 1.0))
        assert_u16_state_is_valid(chaos_map.get_state_u16(), 4)


def test_high_dim_literature_method_registry_supports_n4() -> None:
    initial_state = build_comparison_initial_state_u16(4)
    support_rows = get_method_dimension_support([4])

    assert len(METHODS) == 5
    assert len(support_rows) == 5
    assert all(row["supported"] == "yes" for row in support_rows)

    for method in METHODS:
        chaos_map = _make_map(method, 4, initial_state)
        assert_u16_state_is_valid(chaos_map.get_state_u16(), 4)
        assert np.all(np.isfinite(chaos_map.step()))


def test_high_dim_literature_smoke_metrics_can_run_on_n4() -> None:
    initial_state = build_comparison_initial_state_u16(4)

    for method in METHODS:
        se_traj = _make_map(method, 4, initial_state).iterate(5000, discard_transient=8)
        short_traj = _make_map(method, 4, initial_state).iterate(64, discard_transient=8)

        se_result = spectral_entropy_complexity(se_traj, mode="direct")
        fe_result = fuzzy_entropy(short_traj, mode="direct")
        si_result = scale_index_cwt(short_traj, mode="direct")
        pe_result = permutation_entropy(short_traj, mode="direct")
        le_result = wolf_lyapunov_full_state_nd(_make_map(method, 4, initial_state), max_steps=32, transient=8)

        assert np.isfinite(se_result["se_mean"])
        assert np.isfinite(fe_result["fe_mean"])
        assert np.isfinite(si_result["si_mean"])
        assert np.isfinite(pe_result["pe_mean"])
        assert np.isfinite(le_result["lyap_max"])
