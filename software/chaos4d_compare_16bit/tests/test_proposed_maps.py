# Maintainer: zhaowenbo
from pathlib import Path
import sys

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import IDHNN4DMap, MHM4DMap, NDCS4DMap, Proposed4DBaseMap, Proposed4DEnhancedMap, URUK4DDCMMap
from utils.fixed16 import U16_MOD, float_to_u16, float_vec_to_u16


def test_proposed_base_initializes_with_r1ly_defaults() -> None:
    chaos_map = Proposed4DBaseMap()

    assert chaos_map.a_linear == [371, 373, 375]
    assert chaos_map.a_coupling == [50, 48, 46]
    assert chaos_map.b == 111
    assert chaos_map.x4_z1 == 0
    assert chaos_map.x4_z2 == 0


def test_proposed_base_step_runs() -> None:
    chaos_map = Proposed4DBaseMap()
    state = chaos_map.step()
    assert state.shape == (4,)
    assert np.all((state >= 0.0) & (state < 1.0))


def test_proposed_base_delay_chain_updates_correctly() -> None:
    initial_state_u16 = float_vec_to_u16((0.1, 0.2, 0.3, 0.4))
    chaos_map = Proposed4DBaseMap(initial_state_u16=initial_state_u16)

    old_x4 = int(initial_state_u16[3])
    chaos_map.step()
    assert chaos_map.x4_z1 == old_x4
    assert chaos_map.x4_z2 == 0

    first_step_x4 = int(chaos_map.get_state_u16()[3])
    chaos_map.step()
    assert chaos_map.x4_z1 == first_step_x4
    assert chaos_map.x4_z2 == old_x4


def test_proposed_base_step_matches_4d_specialized_formula() -> None:
    initial_state_u16 = float_vec_to_u16((0.1, 0.2, 0.3, 0.4))
    chaos_map = Proposed4DBaseMap(initial_state_u16=initial_state_u16)

    x1, x2, x3, x4 = (int(value) for value in initial_state_u16)
    expected_x1 = (371 * x1 + 50 * 0) % U16_MOD
    expected_x2 = (373 * x2 + 48 * 0) % U16_MOD
    expected_x3 = (375 * x3 + 46 * x4) % U16_MOD
    x2_float = x2 / U16_MOD
    logistic_sum = (x3 / U16_MOD) + (x4 / U16_MOD)
    expected_x4 = float_to_u16(111 * x2_float + 4.0 * logistic_sum * (1.0 - logistic_sum))

    chaos_map.step()

    assert np.array_equal(
        chaos_map.get_state_u16(),
        np.asarray([expected_x1, expected_x2, expected_x3, expected_x4], dtype=np.int64),
    )


def test_proposed_enhanced_initializes_successfully() -> None:
    chaos_map = Proposed4DEnhancedMap()
    assert chaos_map.rotation_step_index == 0
    assert chaos_map.x4_z1 == 0
    assert chaos_map.x4_z2 == 0


def test_proposed_enhanced_step_runs() -> None:
    chaos_map = Proposed4DEnhancedMap()
    state = chaos_map.step()
    assert state.shape == (4,)
    assert np.all((state >= 0.0) & (state < 1.0))


def test_proposed_enhanced_first_rotation_step_is_one() -> None:
    chaos_map = Proposed4DEnhancedMap()
    assert chaos_map._current_rotation_step() == 1
    chaos_map.step()
    assert chaos_map.rotation_step_index == 1


def test_proposed_enhanced_rotation_sequence_wraps_after_fifteen() -> None:
    chaos_map = Proposed4DEnhancedMap()
    steps = []
    for _ in range(16):
        steps.append(chaos_map._current_rotation_step())
        chaos_map.step()

    assert steps == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1]


def test_pack_and_unpack_are_inverse() -> None:
    values = (0x1234, 0x5678, 0x9ABC, 0xDEF0)
    packed = Proposed4DEnhancedMap._pack_state_to_u64(*values)
    unpacked = Proposed4DEnhancedMap._unpack_u64_to_state(packed)
    assert unpacked == values


def test_rol64_basic_case() -> None:
    rotated = Proposed4DEnhancedMap._rol64(0x8000000000000000, 1)
    assert rotated == 0x0000000000000001


def test_enhanced_trajectory_differs_from_base() -> None:
    base_map = Proposed4DBaseMap()
    enhanced_map = Proposed4DEnhancedMap()

    traj_base = base_map.iterate(32)
    traj_enh = enhanced_map.iterate(32)

    assert traj_base.shape == (32, 4)
    assert traj_enh.shape == (32, 4)
    assert not np.allclose(traj_base, traj_enh)


def test_enhanced_delay_chain_uses_enhanced_x4() -> None:
    initial_state_u16 = float_vec_to_u16((0.1, 0.2, 0.3, 0.4))
    chaos_map = Proposed4DEnhancedMap(initial_state_u16=initial_state_u16)

    base_next_state = chaos_map._compute_base_next_state()
    packed = chaos_map._pack_state_to_u64(*base_next_state)
    rotated = chaos_map._rol64(packed, chaos_map._current_rotation_step())
    expected_state = chaos_map._unpack_u64_to_state(rotated)

    chaos_map.step()

    assert tuple(int(value) for value in chaos_map.get_state_u16()) == expected_state
    assert chaos_map.x4_z1 == expected_state[3]
    assert chaos_map.x4_z2 == 0


def test_iterate_shape_and_difference() -> None:
    base_map = Proposed4DBaseMap()
    enhanced_map = Proposed4DEnhancedMap()

    traj_base = base_map.iterate(32)
    traj_enh = enhanced_map.iterate(32)

    assert traj_base.shape == (32, 4)
    assert traj_enh.shape == (32, 4)
    assert not np.allclose(traj_base, traj_enh)


def test_new_maps_remain_finite_under_smoke_iteration() -> None:
    ndcs_traj = NDCS4DMap().iterate(24)
    uruk_traj = URUK4DDCMMap().iterate(24)
    mhm_traj = MHM4DMap().iterate(24)
    idhnn_traj = IDHNN4DMap().iterate(24)

    assert ndcs_traj.shape == (24, 4)
    assert uruk_traj.shape == (24, 4)
    assert mhm_traj.shape == (24, 4)
    assert idhnn_traj.shape == (24, 4)
    assert np.all(np.isfinite(ndcs_traj))
    assert np.all(np.isfinite(uruk_traj))
    assert np.all(np.isfinite(mhm_traj))
    assert np.all(np.isfinite(idhnn_traj))


def main() -> None:
    tests = [
        ("proposed_base_initializes_with_r1ly_defaults", test_proposed_base_initializes_with_r1ly_defaults),
        ("proposed_base_step_runs", test_proposed_base_step_runs),
        ("proposed_base_delay_chain_updates_correctly", test_proposed_base_delay_chain_updates_correctly),
        ("proposed_base_step_matches_4d_specialized_formula", test_proposed_base_step_matches_4d_specialized_formula),
        ("proposed_enhanced_initializes_successfully", test_proposed_enhanced_initializes_successfully),
        ("proposed_enhanced_step_runs", test_proposed_enhanced_step_runs),
        ("proposed_enhanced_first_rotation_step_is_one", test_proposed_enhanced_first_rotation_step_is_one),
        ("proposed_enhanced_rotation_sequence_wraps_after_fifteen", test_proposed_enhanced_rotation_sequence_wraps_after_fifteen),
        ("pack_and_unpack_are_inverse", test_pack_and_unpack_are_inverse),
        ("rol64_basic_case", test_rol64_basic_case),
        ("enhanced_trajectory_differs_from_base", test_enhanced_trajectory_differs_from_base),
        ("enhanced_delay_chain_uses_enhanced_x4", test_enhanced_delay_chain_uses_enhanced_x4),
        ("iterate_shape_and_difference", test_iterate_shape_and_difference),
        ("new_maps_remain_finite_under_smoke_iteration", test_new_maps_remain_finite_under_smoke_iteration),
    ]
    print("运行 test_proposed_maps.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_proposed_maps.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
