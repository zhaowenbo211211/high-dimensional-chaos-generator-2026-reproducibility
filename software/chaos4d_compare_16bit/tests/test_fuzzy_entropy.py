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
from metrics.fuzzy_entropy import (
    DEFAULT_FE_EMBEDDING_DIM,
    DEFAULT_FE_FUZZY_EXPONENT,
    DEFAULT_FE_RAW_LENGTH,
    DEFAULT_FE_TAU,
    fuzzy_entropy,
    prepare_u16_sequence_for_fe,
)
from utils.fixed16 import sequence_to_u16_array


def test_prepare_u16_sequence_for_fe_quantizes_float_sequence() -> None:
    sequence = np.asarray([0.0, 1.0 / 65536.0, 2.0 / 65536.0, 0.5], dtype=np.float64)
    converted = prepare_u16_sequence_for_fe(sequence)
    assert np.array_equal(converted, np.asarray([0, 1, 2, 32768], dtype=np.int64))


def test_prepare_u16_sequence_for_fe_accepts_integer_sequence() -> None:
    sequence = np.asarray([0, 1, 2, 32768], dtype=np.int64)
    converted = prepare_u16_sequence_for_fe(sequence)
    assert np.array_equal(converted, sequence)


def test_fuzzy_entropy_runs_on_1d_sequence() -> None:
    sequence = np.linspace(0.0, 1.0, 256, endpoint=False)
    result = fuzzy_entropy(sequence)
    assert set(result.keys()) == {
        "fe_components",
        "fe_mean",
        "raw_length",
        "test_length",
        "mode",
        "m",
        "r",
        "tau",
        "n",
    }
    assert len(result["fe_components"]) == 1
    assert np.isfinite(result["fe_mean"])
    assert result["mode"] == "direct"
    assert result["test_length"] == result["raw_length"]


def test_fuzzy_entropy_quantizes_float_input_before_fe() -> None:
    float_sequence = np.asarray([0.0, 1.0 / 65536.0, 2.0 / 65536.0, 0.5] * 64, dtype=np.float64)
    u16_sequence = sequence_to_u16_array(float_sequence)
    from_float = fuzzy_entropy(float_sequence, m=2, tau=1, n=2)
    from_u16 = fuzzy_entropy(u16_sequence, m=2, tau=1, n=2)
    assert np.isclose(from_float["fe_mean"], from_u16["fe_mean"], atol=1e-12)


def test_fuzzy_entropy_runs_on_4d_trajectory() -> None:
    trajectory = Proposed4DEnhancedMap().iterate(256, discard_transient=8)
    result = fuzzy_entropy(trajectory)
    assert len(result["fe_components"]) == 4
    assert np.all(np.isfinite(np.asarray(result["fe_components"], dtype=np.float64)))
    assert np.isclose(result["fe_mean"], np.mean(result["fe_components"]), atol=1e-12)
    assert result["mode"] == "direct"
    assert result["m"] == DEFAULT_FE_EMBEDDING_DIM
    assert result["tau"] == DEFAULT_FE_TAU
    assert result["n"] == DEFAULT_FE_FUZZY_EXPONENT
    assert result["test_length"] == result["raw_length"]


def test_default_fe_raw_length_constant_is_5000() -> None:
    assert DEFAULT_FE_RAW_LENGTH == 5000
