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

from chaos_maps import HCCDM4DMap, IDHNN4DMap, MHM4DMap, MPRC4DMap, MSLC4DMap, NDCS4DMap, URUK4DDCMMap


def test_ndcs4d_initializes_successfully() -> None:
    chaos_map = NDCS4DMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))


def test_ndcs4d_step_runs() -> None:
    state = NDCS4DMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_ndcs4d_iterate_shape_is_correct() -> None:
    traj = NDCS4DMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_uruk4ddcm_initializes_successfully() -> None:
    chaos_map = URUK4DDCMMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))


def test_uruk4ddcm_step_runs() -> None:
    state = URUK4DDCMMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_uruk4ddcm_iterate_shape_is_correct() -> None:
    traj = URUK4DDCMMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_ndcs4d_trajectory_is_not_constant() -> None:
    traj = NDCS4DMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_uruk4ddcm_trajectory_is_not_constant() -> None:
    traj = URUK4DDCMMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_no_nan_or_inf_leaks_into_outputs() -> None:
    ndcs_traj = NDCS4DMap().iterate(20)
    uruk_traj = URUK4DDCMMap().iterate(20)
    mhm_traj = MHM4DMap().iterate(20)
    idhnn_traj = IDHNN4DMap().iterate(20)
    mprc_traj = MPRC4DMap().iterate(20)
    mslc_traj = MSLC4DMap().iterate(20)
    hccdm_traj = HCCDM4DMap().iterate(20)
    assert np.all(np.isfinite(ndcs_traj))
    assert np.all(np.isfinite(uruk_traj))
    assert np.all(np.isfinite(mhm_traj))
    assert np.all(np.isfinite(idhnn_traj))
    assert np.all(np.isfinite(mprc_traj))
    assert np.all(np.isfinite(mslc_traj))
    assert np.all(np.isfinite(hccdm_traj))


def test_ndcs4d_and_uruk4ddcm_trajectories_differ() -> None:
    ndcs_traj = NDCS4DMap().iterate(20)
    uruk_traj = URUK4DDCMMap().iterate(20)
    assert ndcs_traj.shape == (20, 4)
    assert uruk_traj.shape == (20, 4)
    assert not np.allclose(ndcs_traj, uruk_traj)


def test_mhm4d_initializes_successfully() -> None:
    chaos_map = MHM4DMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))


def test_mhm4d_step_runs() -> None:
    state = MHM4DMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_mhm4d_iterate_shape_is_correct() -> None:
    traj = MHM4DMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_idhnn4d_initializes_successfully() -> None:
    chaos_map = IDHNN4DMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))


def test_idhnn4d_step_runs() -> None:
    state = IDHNN4DMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_idhnn4d_iterate_shape_is_correct() -> None:
    traj = IDHNN4DMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_mhm4d_trajectory_is_not_constant() -> None:
    traj = MHM4DMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_idhnn4d_trajectory_is_not_constant() -> None:
    traj = IDHNN4DMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_mhm4d_and_idhnn4d_trajectories_differ() -> None:
    mhm_traj = MHM4DMap().iterate(20)
    idhnn_traj = IDHNN4DMap().iterate(20)
    assert mhm_traj.shape == (20, 4)
    assert idhnn_traj.shape == (20, 4)
    assert not np.allclose(mhm_traj, idhnn_traj)


def test_mprc4d_initializes_successfully() -> None:
    chaos_map = MPRC4DMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))


def test_mprc4d_step_runs() -> None:
    state = MPRC4DMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_mprc4d_iterate_shape_is_correct() -> None:
    traj = MPRC4DMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_mslc4d_initializes_successfully() -> None:
    chaos_map = MSLC4DMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))
    assert chaos_map.time_index == 0


def test_mslc4d_step_runs() -> None:
    state = MSLC4DMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_mslc4d_iterate_shape_is_correct() -> None:
    traj = MSLC4DMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_hccdm4d_initializes_successfully() -> None:
    chaos_map = HCCDM4DMap()
    assert chaos_map.get_state_u16().shape == (4,)
    assert np.all((chaos_map.get_state_float() >= 0.0) & (chaos_map.get_state_float() < 1.0))


def test_hccdm4d_step_runs() -> None:
    state = HCCDM4DMap().step()
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))


def test_hccdm4d_iterate_shape_is_correct() -> None:
    traj = HCCDM4DMap().iterate(16)
    assert traj.shape == (16, 4)
    assert np.all(np.isfinite(traj))


def test_mprc4d_trajectory_is_not_constant() -> None:
    traj = MPRC4DMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_mslc4d_trajectory_is_not_constant() -> None:
    traj = MSLC4DMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_hccdm4d_trajectory_is_not_constant() -> None:
    traj = HCCDM4DMap().iterate(12)
    assert not np.allclose(traj, traj[0])


def test_mslc4d_time_drive_changes_with_steps() -> None:
    chaos_map = MSLC4DMap()
    first_drive = chaos_map._current_time_drive()
    chaos_map.step()
    second_drive = chaos_map._current_time_drive()
    assert chaos_map.time_index == 1
    assert not np.isclose(first_drive, second_drive)


def test_hccdm4d_tent_map_piecewise_behavior() -> None:
    lower_value = HCCDM4DMap._tent_map(0.25)
    upper_value = HCCDM4DMap._tent_map(0.75)
    expected_lower = (5.0 * 0.25 + 4.0 * np.sin(np.pi * 0.25)) / 2.0
    expected_upper = (5.0 * (1.0 - 0.75) + 4.0 * np.sin(np.pi * 0.75)) / 2.0
    assert np.isclose(lower_value, expected_lower)
    assert np.isclose(upper_value, expected_upper)


def main() -> None:
    tests = [
        ("ndcs4d_initializes_successfully", test_ndcs4d_initializes_successfully),
        ("ndcs4d_step_runs", test_ndcs4d_step_runs),
        ("ndcs4d_iterate_shape_is_correct", test_ndcs4d_iterate_shape_is_correct),
        ("uruk4ddcm_initializes_successfully", test_uruk4ddcm_initializes_successfully),
        ("uruk4ddcm_step_runs", test_uruk4ddcm_step_runs),
        ("uruk4ddcm_iterate_shape_is_correct", test_uruk4ddcm_iterate_shape_is_correct),
        ("ndcs4d_trajectory_is_not_constant", test_ndcs4d_trajectory_is_not_constant),
        ("uruk4ddcm_trajectory_is_not_constant", test_uruk4ddcm_trajectory_is_not_constant),
        ("no_nan_or_inf_leaks_into_outputs", test_no_nan_or_inf_leaks_into_outputs),
        ("ndcs4d_and_uruk4ddcm_trajectories_differ", test_ndcs4d_and_uruk4ddcm_trajectories_differ),
        ("mhm4d_initializes_successfully", test_mhm4d_initializes_successfully),
        ("mhm4d_step_runs", test_mhm4d_step_runs),
        ("mhm4d_iterate_shape_is_correct", test_mhm4d_iterate_shape_is_correct),
        ("idhnn4d_initializes_successfully", test_idhnn4d_initializes_successfully),
        ("idhnn4d_step_runs", test_idhnn4d_step_runs),
        ("idhnn4d_iterate_shape_is_correct", test_idhnn4d_iterate_shape_is_correct),
        ("mhm4d_trajectory_is_not_constant", test_mhm4d_trajectory_is_not_constant),
        ("idhnn4d_trajectory_is_not_constant", test_idhnn4d_trajectory_is_not_constant),
        ("mhm4d_and_idhnn4d_trajectories_differ", test_mhm4d_and_idhnn4d_trajectories_differ),
        ("mprc4d_initializes_successfully", test_mprc4d_initializes_successfully),
        ("mprc4d_step_runs", test_mprc4d_step_runs),
        ("mprc4d_iterate_shape_is_correct", test_mprc4d_iterate_shape_is_correct),
        ("mslc4d_initializes_successfully", test_mslc4d_initializes_successfully),
        ("mslc4d_step_runs", test_mslc4d_step_runs),
        ("mslc4d_iterate_shape_is_correct", test_mslc4d_iterate_shape_is_correct),
        ("hccdm4d_initializes_successfully", test_hccdm4d_initializes_successfully),
        ("hccdm4d_step_runs", test_hccdm4d_step_runs),
        ("hccdm4d_iterate_shape_is_correct", test_hccdm4d_iterate_shape_is_correct),
        ("mprc4d_trajectory_is_not_constant", test_mprc4d_trajectory_is_not_constant),
        ("mslc4d_trajectory_is_not_constant", test_mslc4d_trajectory_is_not_constant),
        ("hccdm4d_trajectory_is_not_constant", test_hccdm4d_trajectory_is_not_constant),
        ("mslc4d_time_drive_changes_with_steps", test_mslc4d_time_drive_changes_with_steps),
        ("hccdm4d_tent_map_piecewise_behavior", test_hccdm4d_tent_map_piecewise_behavior),
    ]
    print("运行 test_maps_basic.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_maps_basic.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
