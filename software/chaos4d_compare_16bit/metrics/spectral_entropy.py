# Maintainer: zhaowenbo
"""SE complexity for direct numeric sequences and optional byte-test sequences."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from metrics.common import ensure_sequence_1d, ensure_trajectory_2d
from utils.fixed16 import U16_MOD, sequence_to_u16_array


DEFAULT_SE_RAW_LENGTH = 5000
DEFAULT_SE_TRANSIENT = 1000
MIN_SE_RAW_LENGTH = 5000


class SpectralEntropyResult(TypedDict):
    """Structured result for the paper-facing SE complexity analysis."""

    se_components: list[float]
    se_mean: float
    raw_length: int
    test_length: int
    mode: str


SpectralEntropyMode = Literal["direct", "byte_test", "character"]


def validate_se_raw_length(length: int) -> None:
    """Enforce the paper-facing minimum raw sequence length for SE analysis."""

    if int(length) < MIN_SE_RAW_LENGTH:
        raise ValueError(f"SE complexity requires raw sequence length >= {MIN_SE_RAW_LENGTH}.")


def _sequence_to_u16(seq: np.ndarray) -> np.ndarray:
    """Convert one raw numeric sequence into project-consistent u16 samples.

    The SE workflow in this project is aligned with the paper's 16-bit digital
    realization. Therefore, even in direct numeric-sequence mode, the analyzed
    sequence is first quantized into the shared 16-bit integer domain before
    mean removal, FFT, and spectral-entropy evaluation.
    """

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    return sequence_to_u16_array(raw) % U16_MOD


def build_test_sequence_from_u16(u16_sequence: np.ndarray) -> bytes:
    """Build the paper-facing character test sequence from a u16 sequence.

    Each 16-bit sample is split into two 8-bit bytes, and each byte is treated
    as one character. Therefore a raw numeric sequence of length ``n`` becomes
    a symbolized test sequence of length ``2n``. The SE complexity implemented
    in this project is defined on that character-based test sequence, not on
    the original floating-point trajectory directly.
    """

    arr = np.asarray(u16_sequence, dtype=np.int64).reshape(-1) % U16_MOD
    if len(arr) <= 0:
        raise ValueError("u16_sequence must not be empty.")

    byte_values = np.empty(len(arr) * 2, dtype=np.uint8)
    byte_values[0::2] = ((arr >> 8) & 0xFF).astype(np.uint8)
    byte_values[1::2] = (arr & 0xFF).astype(np.uint8)
    return bytes(byte_values.tolist())


def encode_test_sequence_to_numeric(test_sequence: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Convert the character test sequence into numeric byte codes for FFT.

    The frequency analysis target is the numeric encoding of the characterized
    test sequence, rather than the original float trajectory itself.
    """

    if isinstance(test_sequence, (bytes, bytearray)):
        encoded = np.frombuffer(test_sequence, dtype=np.uint8).astype(np.float64)
    else:
        encoded = np.asarray(test_sequence, dtype=np.uint8).astype(np.float64).reshape(-1)
    if len(encoded) <= 0:
        raise ValueError("test_sequence must not be empty.")
    return encoded


def _spectral_entropy_from_encoded_sequence(encoded_sequence: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized SE complexity from the encoded test sequence."""

    encoded = ensure_sequence_1d(encoded_sequence)
    if len(encoded) < 4:
        raise ValueError("Encoded test sequence must contain at least 4 samples.")

    test_length = len(encoded)
    half_length = test_length // 2
    if half_length <= 1:
        raise ValueError("Encoded test sequence is too short for normalized SE computation.")

    spectrum = np.fft.fft(encoded)
    power = np.abs(spectrum[:half_length]) ** 2
    power_sum = float(np.sum(power))
    if power_sum <= eps:
        return 0.0

    probabilities = power / power_sum
    se = -float(np.sum(probabilities * np.log(probabilities + eps)))
    return se / float(np.log(half_length))


def _remove_mean(seq: np.ndarray) -> np.ndarray:
    """Return a float sequence after mean removal before FFT."""

    arr = ensure_sequence_1d(np.asarray(seq, dtype=np.float64))
    return arr - float(np.mean(arr))


def _direct_numeric_spectral_entropy_complexity(seq: np.ndarray, eps: float = 1e-12) -> tuple[float, int, int]:
    """Compute normalized SE directly from the raw numeric sequence.

    This mode applies FFT to the direct numeric sequence itself and is the
    default for literature-scale comparison runs in this project. However, the
    sequence is not taken straight from the raw float trajectory: it is first
    quantized into the shared 16-bit integer domain so the SE analysis remains
    consistent with the paper's fixed-point realization. The paper-facing
    default experiment length is ``raw_length = 5000``. The FFT length is the
    raw sequence length ``n`` after even-length trimming, so ``test_length``
    equals ``raw_length`` in this mode. Before FFT, the 16-bit integer sequence
    is explicitly demeaned via ``x(n) <- x(n) - mean(x)``.
    """

    u16_sequence = _sequence_to_u16(seq)
    validate_se_raw_length(len(u16_sequence))
    arr = _remove_mean(u16_sequence.astype(np.float64))
    if len(arr) % 2 == 1:
        arr = arr[:-1]
    if len(arr) < 4:
        raise ValueError("Direct numeric sequence must contain at least 4 samples.")

    se_value = _spectral_entropy_from_encoded_sequence(arr, eps=eps)
    return float(se_value), int(len(arr)), int(len(arr))


def _character_test_sequence_spectral_entropy_complexity(
    seq: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, int, int]:
    """Compute normalized SE from the paper's character-based test sequence.

    This byte-test mode does not analyze the raw floating-point trajectory directly.
    Instead, the 16-bit numeric sequence is symbolized into a character test
    sequence with two byte-characters per sample, so a raw length ``n`` becomes
    test length ``N = 2n``. FFT is applied to the numeric byte-code encoding of
    that test sequence. It is kept only as an auxiliary digital-analysis mode,
    not as the paper's default result mode.
    """

    u16_sequence = _sequence_to_u16(seq)
    validate_se_raw_length(len(u16_sequence))
    test_sequence = build_test_sequence_from_u16(u16_sequence)
    encoded_sequence = encode_test_sequence_to_numeric(test_sequence)
    se_value = _spectral_entropy_from_encoded_sequence(encoded_sequence, eps=eps)
    return float(se_value), int(len(u16_sequence)), int(len(encoded_sequence))


def spectral_entropy_complexity(
    signal: np.ndarray,
    component: int | None = None,
    mode: SpectralEntropyMode = "direct",
    eps: float = 1e-12,
) -> SpectralEntropyResult:
    """Compute SE complexity in either direct or optional byte-test mode.

    Two modes are supported:

    - ``mode="direct"``:
      FFT is applied to the direct numeric sequence. This is the default
      comparison mode for reproducing literature-scale results in this project.
      Before FFT, the trajectory component is first quantized into the shared
      16-bit integer domain, then mean removal is applied, and the formal
      experiment default uses ``raw_length = test_length = 5000``.
    - ``mode="byte_test"``:
      The 16-bit sequence is first converted into a symbolized character-based
      test sequence with two characters per sample. FFT is then applied to the
      numeric byte-code representation of that test sequence.
    - ``mode="character"``:
      Backward-compatible alias of ``byte_test``.

    For a 4D trajectory, each state component is processed independently and
    the overall comparison indicator is the arithmetic mean:

        se_mean = (se_x1 + se_x2 + se_x3 + se_x4) / 4
    """

    normalized_mode = "byte_test" if mode == "character" else mode
    if normalized_mode not in ("direct", "byte_test"):
        raise ValueError("mode must be one of 'direct', 'byte_test', or 'character'.")

    arr = np.asarray(signal)

    def compute_one_dimension(seq_1d: np.ndarray) -> tuple[float, int, int]:
        if normalized_mode == "direct":
            return _direct_numeric_spectral_entropy_complexity(seq_1d, eps=eps)
        return _character_test_sequence_spectral_entropy_complexity(seq_1d, eps=eps)

    if arr.ndim == 1:
        se_value, raw_length, test_length = compute_one_dimension(arr)
        return {
            "se_components": [float(se_value)],
            "se_mean": float(se_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": normalized_mode,
        }

    traj = ensure_trajectory_2d(np.asarray(signal, dtype=np.float64))
    dimension = int(traj.shape[1])
    if component is not None:
        if component < 0 or component >= dimension:
            raise ValueError(f"component must be in [0, {dimension - 1}].")
        se_value, raw_length, test_length = compute_one_dimension(traj[:, component])
        return {
            "se_components": [float(se_value)],
            "se_mean": float(se_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": normalized_mode,
        }

    component_values: list[float] = []
    raw_length = 0
    test_length = 0
    for component_index in range(dimension):
        se_value, raw_length, test_length = compute_one_dimension(traj[:, component_index])
        component_values.append(float(se_value))

    return {
        "se_components": component_values,
        "se_mean": float(np.mean(component_values)),
        "raw_length": int(raw_length),
        "test_length": int(test_length),
        "mode": normalized_mode,
    }


def spectral_entropy_smoke(traj: np.ndarray) -> np.ndarray:
    """Return four direct-mode SE values for smoke testing."""

    result = spectral_entropy_complexity(traj, mode="direct")
    return np.asarray(result["se_components"], dtype=np.float64)
