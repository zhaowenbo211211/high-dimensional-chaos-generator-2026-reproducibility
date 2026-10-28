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
from metrics.nonperiodicity import (
    DEFAULT_SI_RAW_LENGTH,
    DEFAULT_SI_MIN_SCALE,
    DEFAULT_SI_STUDY_MAX_SCALE,
    DEFAULT_SI_TRANSIENT,
    DEFAULT_SI_WAVELET,
    prepare_u16_sequence_for_scale_index,
    scale_index_cwt,
)


def test_prepare_u16_sequence_for_scale_index_quantizes_float_sequence() -> None:
    sequence = np.asarray([0.0, 1.0 / 65536.0, 2.0 / 65536.0, 0.5], dtype=np.float64)
    converted = prepare_u16_sequence_for_scale_index(sequence)
    assert np.array_equal(converted, np.asarray([0, 1, 2, 32768], dtype=np.int64))


def test_prepare_u16_sequence_for_scale_index_accepts_integer_sequence() -> None:
    sequence = np.asarray([0, 1, 2, 32768], dtype=np.int64)
    converted = prepare_u16_sequence_for_scale_index(sequence)
    assert np.array_equal(converted, sequence)


def test_scale_index_runs_on_1d_sequence() -> None:
    sequence = np.linspace(0.0, 1.0, 256, endpoint=False)
    result = scale_index_cwt(sequence)
    assert set(result.keys()) == {"si_components", "si_mean", "raw_length", "test_length", "mode", "wavelet", "scales"}
    assert len(result["si_components"]) == 1
    assert np.isfinite(result["si_mean"])
    assert result["mode"] == "direct"
    assert result["wavelet"] == DEFAULT_SI_WAVELET
    assert result["test_length"] == result["raw_length"]
    assert len(result["scales"]) > 1


def test_scale_index_runs_on_4d_trajectory() -> None:
    trajectory = Proposed4DEnhancedMap().iterate(256, discard_transient=8)
    result = scale_index_cwt(trajectory)
    assert len(result["si_components"]) == 4
    assert np.all(np.isfinite(np.asarray(result["si_components"], dtype=np.float64)))
    assert np.isclose(result["si_mean"], np.mean(result["si_components"]), atol=1e-12)
    assert result["test_length"] == result["raw_length"]


def test_default_si_configuration() -> None:
    assert DEFAULT_SI_RAW_LENGTH == 40000
    assert DEFAULT_SI_TRANSIENT == 3000
    assert DEFAULT_SI_WAVELET == "morlet"
    assert DEFAULT_SI_MIN_SCALE == 2.0
    assert DEFAULT_SI_STUDY_MAX_SCALE == 256.0
