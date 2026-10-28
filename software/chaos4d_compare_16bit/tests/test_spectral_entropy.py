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

from chaos_maps import Proposed4DEnhancedMap
from metrics.spectral_entropy import (
    DEFAULT_SE_RAW_LENGTH,
    MIN_SE_RAW_LENGTH,
    build_test_sequence_from_u16,
    encode_test_sequence_to_numeric,
    spectral_entropy_complexity,
)
from utils.fixed16 import sequence_to_u16_array


def test_build_test_sequence_doubles_length() -> None:
    u16_sequence = np.asarray([0x1234, 0xABCD, 0x0001], dtype=np.int64)
    test_sequence = build_test_sequence_from_u16(u16_sequence)
    assert isinstance(test_sequence, bytes)
    assert len(test_sequence) == 2 * len(u16_sequence)


def test_sequence_to_u16_array_quantizes_float_sequence() -> None:
    sequence = np.asarray([0.0, 1.0 / 65536.0, 2.0 / 65536.0, 0.5], dtype=np.float64)
    converted = sequence_to_u16_array(sequence)
    assert np.array_equal(converted, np.asarray([0, 1, 2, 32768], dtype=np.int64))


def test_encode_test_sequence_to_numeric_returns_byte_codes() -> None:
    encoded = encode_test_sequence_to_numeric(bytes([0x12, 0x34, 0xAB, 0xCD]))
    assert np.array_equal(encoded, np.asarray([0x12, 0x34, 0xAB, 0xCD], dtype=np.float64))


def test_spectral_entropy_complexity_runs_on_1d_sequence() -> None:
    sequence = np.linspace(0.0, 1.0, DEFAULT_SE_RAW_LENGTH, endpoint=False)
    result = spectral_entropy_complexity(sequence)
    assert set(result.keys()) == {"se_components", "se_mean", "raw_length", "test_length", "mode"}
    assert len(result["se_components"]) == 1
    assert np.isfinite(result["se_mean"])
    assert result["raw_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["test_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["mode"] == "direct"


def test_direct_mode_quantizes_float_input_before_se() -> None:
    float_sequence = np.asarray([0.0, 1.0 / 65536.0, 2.0 / 65536.0, 0.5] * 1250, dtype=np.float64)
    u16_sequence = sequence_to_u16_array(float_sequence)
    from_float = spectral_entropy_complexity(float_sequence, mode="direct")
    from_u16 = spectral_entropy_complexity(u16_sequence, mode="direct")
    assert np.isclose(from_float["se_mean"], from_u16["se_mean"], atol=1e-12)


def test_direct_mode_runs_on_u16_sequence() -> None:
    u16_sequence = np.asarray([0, 1, 2, 32768] * 1250, dtype=np.int64)
    result = spectral_entropy_complexity(u16_sequence, mode="direct")
    assert np.isfinite(result["se_mean"])
    assert result["raw_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["test_length"] == DEFAULT_SE_RAW_LENGTH


def test_spectral_entropy_complexity_byte_test_mode_doubles_length() -> None:
    sequence = np.linspace(0.0, 1.0, DEFAULT_SE_RAW_LENGTH, endpoint=False)
    result = spectral_entropy_complexity(sequence, mode="byte_test")
    assert np.isfinite(result["se_mean"])
    assert result["raw_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["test_length"] == 2 * DEFAULT_SE_RAW_LENGTH
    assert result["mode"] == "byte_test"


def test_spectral_entropy_complexity_runs_on_4d_trajectory() -> None:
    trajectory = Proposed4DEnhancedMap().iterate(DEFAULT_SE_RAW_LENGTH, discard_transient=16)
    result = spectral_entropy_complexity(trajectory)
    assert len(result["se_components"]) == 4
    assert np.all(np.isfinite(np.asarray(result["se_components"], dtype=np.float64)))
    assert np.isfinite(result["se_mean"])
    assert result["raw_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["test_length"] == DEFAULT_SE_RAW_LENGTH
    assert result["mode"] == "direct"


def test_direct_mode_is_invariant_to_constant_offset_after_mean_removal() -> None:
    sequence = np.linspace(0.0, 1.0, DEFAULT_SE_RAW_LENGTH, endpoint=False)
    base = spectral_entropy_complexity(sequence, mode="direct")
    shifted = spectral_entropy_complexity(sequence + 7.5, mode="direct")
    assert np.isclose(base["se_mean"], shifted["se_mean"], atol=1e-10)


def test_default_se_raw_length_constant_is_5000() -> None:
    assert DEFAULT_SE_RAW_LENGTH == 5000
    assert MIN_SE_RAW_LENGTH == 5000


def test_spectral_entropy_rejects_short_sequences() -> None:
    short_sequence = np.linspace(0.0, 1.0, 128, endpoint=False)
    try:
        spectral_entropy_complexity(short_sequence, mode="direct")
    except ValueError as exc:
        assert ">= 5000" in str(exc)
        return
    raise AssertionError("Expected short SE sequence to be rejected.")


def main() -> None:
    tests = [
        ("build_test_sequence_doubles_length", test_build_test_sequence_doubles_length),
        ("sequence_to_u16_array_quantizes_float_sequence", test_sequence_to_u16_array_quantizes_float_sequence),
        ("encode_test_sequence_to_numeric_returns_byte_codes", test_encode_test_sequence_to_numeric_returns_byte_codes),
        ("spectral_entropy_complexity_runs_on_1d_sequence", test_spectral_entropy_complexity_runs_on_1d_sequence),
        ("direct_mode_quantizes_float_input_before_se", test_direct_mode_quantizes_float_input_before_se),
        ("direct_mode_runs_on_u16_sequence", test_direct_mode_runs_on_u16_sequence),
        ("spectral_entropy_complexity_byte_test_mode_doubles_length", test_spectral_entropy_complexity_byte_test_mode_doubles_length),
        ("spectral_entropy_complexity_runs_on_4d_trajectory", test_spectral_entropy_complexity_runs_on_4d_trajectory),
        ("direct_mode_is_invariant_to_constant_offset_after_mean_removal", test_direct_mode_is_invariant_to_constant_offset_after_mean_removal),
        ("default_se_raw_length_constant_is_5000", test_default_se_raw_length_constant_is_5000),
        ("spectral_entropy_rejects_short_sequences", test_spectral_entropy_rejects_short_sequences),
    ]
    print("运行 test_spectral_entropy.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_spectral_entropy.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
