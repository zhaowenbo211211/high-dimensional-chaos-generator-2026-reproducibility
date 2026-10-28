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
from metrics.permutation_entropy import (
    DEFAULT_PE_EMBEDDING_DIM,
    DEFAULT_PE_NORMALIZED,
    DEFAULT_PE_RAW_LENGTH,
    DEFAULT_PE_TAU,
    DEFAULT_PE_TRANSIENT,
    permutation_entropy,
    prepare_u16_sequence_for_pe,
)
from utils.io_utils import ensure_metric_output_dirs


def test_prepare_u16_sequence_for_pe_quantizes_float_sequence() -> None:
    sequence = np.asarray([0.0, 1.0 / 65536.0, 2.0 / 65536.0, 0.5], dtype=np.float64)
    converted = prepare_u16_sequence_for_pe(sequence)
    assert np.array_equal(converted, np.asarray([0, 1, 2, 32768], dtype=np.int64))


def test_prepare_u16_sequence_for_pe_accepts_integer_sequence() -> None:
    sequence = np.asarray([0, 1, 2, 32768], dtype=np.int64)
    converted = prepare_u16_sequence_for_pe(sequence)
    assert np.array_equal(converted, sequence)


def test_permutation_entropy_runs_on_1d_sequence() -> None:
    sequence = np.linspace(0.0, 1.0, 128, endpoint=False)
    result = permutation_entropy(sequence)
    assert set(result.keys()) == {"pe_components", "pe_mean", "raw_length", "test_length", "mode", "m", "tau", "normalized"}
    assert len(result["pe_components"]) == 1
    assert np.isfinite(result["pe_mean"])
    assert result["mode"] == "direct"
    assert result["test_length"] == result["raw_length"]


def test_permutation_entropy_runs_on_4d_trajectory() -> None:
    trajectory = Proposed4DEnhancedMap().iterate(256, discard_transient=8)
    result = permutation_entropy(trajectory)
    assert len(result["pe_components"]) == 4
    assert np.all(np.isfinite(np.asarray(result["pe_components"], dtype=np.float64)))
    assert np.isclose(result["pe_mean"], np.mean(result["pe_components"]), atol=1e-12)


def test_permutation_entropy_ties_are_deterministic() -> None:
    sequence = np.asarray([5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8], dtype=np.int64)
    result_a = permutation_entropy(sequence, m=3, tau=1, normalized=True)
    result_b = permutation_entropy(sequence, m=3, tau=1, normalized=True)
    assert np.isclose(result_a["pe_mean"], result_b["pe_mean"], atol=1e-12)


def test_default_pe_configuration_and_output_dirs() -> None:
    assert DEFAULT_PE_RAW_LENGTH == 5000
    assert DEFAULT_PE_TRANSIENT == 1000
    assert DEFAULT_PE_EMBEDDING_DIM == 5
    assert DEFAULT_PE_TAU == 1
    assert DEFAULT_PE_NORMALIZED is True
    dirs = ensure_metric_output_dirs("permutation_entropy")
    assert dirs["raw"].exists()
    assert dirs["tables"].exists()
    assert dirs["figures"].exists()
    assert dirs["diagnostics"].exists()
