# Maintainer: zhaowenbo
"""Scale Index based non-periodicity analysis via continuous wavelet transform.

This module implements the standard global scale index, not the windowed scale
index. The implementation is guided by the standard scale-index definition
revisited by Bolos et al. (2020), which in turn builds on the original tool of
Benitez et al. (2010). In particular, the current implementation follows these
paper-facing ideas more closely than the earlier placeholder version:

- direct numeric-sequence mode;
- 16-bit quantized integer sequences as the analysis target;
- a Morlet mother wavelet as the default continuous wavelet for the standard
  CWT-based scale-index workflow;
- an optional ``sym4`` mother-wavelet approximation retained for digital-chaos
  experiments that want a compact-support wavelet variant;
- a normalized inner scalogram based on the L2 norm of the valid interior CWT
  coefficients at each scale;
- a standard scale-index ratio that searches the main scale in a study interval
  ``[s0, s1]`` and the minimum in the extended interval ``[s_max, 2*s1]``;
- the paper-facing default raw sequence length ``40000`` with transient
  discard ``3000`` and study interval ``[2, 256]``.

Scale Index is computed on the 16-bit quantized integer sequence rather than
directly on the raw floating-point trajectory.
"""

from __future__ import annotations

import json
from typing import TypedDict

import numpy as np

from metrics.common import ensure_trajectory_2d
from utils.fixed16 import sequence_to_u16_array


DEFAULT_SI_RAW_LENGTH = 40000
DEFAULT_SI_TRANSIENT = 3000
DEFAULT_SI_WAVELET = "morlet"
DEFAULT_SI_NUM_SCALES = 48
DEFAULT_SI_MIN_SCALE = 2.0
DEFAULT_SI_STUDY_MAX_SCALE = 256.0


class ScaleIndexResult(TypedDict):
    """Structured result for paper-facing scale-index experiments."""

    si_components: list[float]
    si_mean: float
    raw_length: int
    test_length: int
    mode: str
    wavelet: str
    scales: list[float]


class ScaleIndexDiagnostics(TypedDict):
    """Detailed one-dimensional SI diagnostics for scalogram inspection."""

    si_value: float
    raw_length: int
    test_length: int
    mode: str
    wavelet: str
    scales: list[float]
    scalogram: list[float]
    study_scale_max: float
    extended_scale_max: float
    s_max_index: int
    s_max_scale: float
    s_max_value: float
    s_min_index: int
    s_min_scale: float
    s_min_value: float


def prepare_u16_sequence_for_scale_index(seq: np.ndarray) -> np.ndarray:
    """Convert one numeric sequence into the shared 16-bit integer domain."""

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    return sequence_to_u16_array(raw)


def _default_scales(length: int, num_scales: int = DEFAULT_SI_NUM_SCALES) -> np.ndarray:
    """Build a practical total scale interval for the standard global scale index.

    Following the Benitez/Bolos scale-index setup, we compute the scalogram on
    an interval that reaches approximately ``2*s1`` so that the minimum after
    the most representative scale can be searched in the extended range.
    """

    if length < 16:
        raise ValueError("Sequence is too short for scale-index analysis.")
    # The paper-facing study interval is [2, 256]. We compute the CWT on an
    # extended interval up to 2*s1 so the post-maximum minimum can be searched
    # in [s_max, 2*s1], following the standard scale-index construction.
    max_scale = min(2.0 * DEFAULT_SI_STUDY_MAX_SCALE, length / 4.0)
    min_scale = min(DEFAULT_SI_MIN_SCALE, max_scale)
    if max_scale <= min_scale:
        max_scale = min_scale + 1.0
    return np.geomspace(min_scale, max_scale, num=num_scales).astype(np.float64)


def _morlet_kernel(scale: float, omega0: float = 6.0) -> tuple[np.ndarray, int]:
    """Return a scale-dilated complex Morlet kernel and half support."""

    half_support = max(int(np.ceil(6.0 * float(scale))), 1)
    t = np.arange(-half_support, half_support + 1, dtype=np.float64)
    ts = t / float(scale)
    kernel = (np.pi ** -0.25) * np.exp(1j * omega0 * ts) * np.exp(-(ts**2) / 2.0)
    kernel = kernel / np.sqrt(max(float(scale), 1.0))
    return kernel.astype(np.complex128), half_support


def _morlet_cwt(signal: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Morlet-based CWT approximation by direct convolution."""

    seq = np.asarray(signal, dtype=np.float64)
    coeffs = np.empty((len(scales), len(seq)), dtype=np.complex128)
    half_supports = np.empty(len(scales), dtype=np.int64)
    for index, scale in enumerate(scales):
        kernel, half_support = _morlet_kernel(float(scale))
        full = np.convolve(seq, np.conj(kernel[::-1]), mode="full")
        start = (len(full) - len(seq)) // 2
        coeffs[index] = full[start : start + len(seq)]
        half_supports[index] = half_support
    return coeffs, half_supports


_SYM4_DEC_LO = np.asarray(
    [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.8037387518059161,
        0.29785779560527736,
        -0.0992195435769354,
        -0.012603967262037833,
        0.0322231006040427,
    ],
    dtype=np.float64,
)

_SYM4_DEC_HI = np.asarray(
    [
        -0.0322231006040427,
        -0.0126039672620378,
        0.0992195435768472,
        0.297857795605277,
        -0.803738751805916,
        0.497618667632015,
        0.0296355276459999,
        -0.0757657147892733,
    ],
    dtype=np.float64,
)


def _upsample_by_two(arr: np.ndarray) -> np.ndarray:
    """Return a 2x upsampled array with zeros inserted between samples."""

    upsampled = np.zeros(len(arr) * 2, dtype=np.float64)
    upsampled[::2] = arr
    return upsampled


def _cascade_scaling_function(levels: int = 8) -> np.ndarray:
    """Approximate the ``sym4`` scaling function via the cascade algorithm."""

    phi = np.asarray([1.0], dtype=np.float64)
    for _ in range(levels):
        phi = np.sqrt(2.0) * np.convolve(_upsample_by_two(phi), _SYM4_DEC_LO, mode="full")
    phi = phi / max(np.linalg.norm(phi), 1e-12)
    return phi


def _sym4_wavelet_function(levels: int = 8) -> tuple[np.ndarray, float]:
    """Approximate the compact-support ``sym4`` mother wavelet function.

    This is closer to the compact-support wavelet-function interpretation used
    in the Benitez/Bolos framework than directly stretching the raw filter
    coefficients.
    """

    phi = _cascade_scaling_function(levels=levels)
    psi = np.sqrt(2.0) * np.convolve(_upsample_by_two(phi), _SYM4_DEC_HI, mode="full")
    dx = 2.0 ** (-levels)
    psi = psi - float(np.mean(psi))
    energy = float(np.sqrt(np.sum(psi**2) * dx))
    if energy > 0.0:
        psi = psi / energy
    return psi.astype(np.float64), dx


_SYM4_WAVELET_FUNCTION, _SYM4_DX = _sym4_wavelet_function(levels=8)


def _sym4_kernel(scale: float) -> tuple[np.ndarray, int]:
    """Return a scale-dilated ``sym4`` mother-wavelet kernel and half support."""

    base = _SYM4_WAVELET_FUNCTION
    dx = _SYM4_DX
    support = (np.arange(len(base), dtype=np.float64) - (len(base) - 1) / 2.0) * dx
    scaled_support = support * float(scale)
    min_t = int(np.floor(np.min(scaled_support)))
    max_t = int(np.ceil(np.max(scaled_support)))
    sample_points = np.arange(min_t, max_t + 1, dtype=np.float64)
    kernel = np.interp(sample_points / float(scale), support, base, left=0.0, right=0.0)
    kernel = kernel / np.sqrt(max(float(scale), 1.0))
    kernel = kernel - float(np.mean(kernel))
    norm = float(np.linalg.norm(kernel))
    if norm > 0.0:
        kernel = kernel / norm
    half_support = max(abs(min_t), abs(max_t))
    return kernel, int(half_support)


def _sym4_cwt(signal: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a sym4-based compact-support CWT approximation."""

    seq = np.asarray(signal, dtype=np.float64)
    coeffs = np.empty((len(scales), len(seq)), dtype=np.complex128)
    half_supports = np.empty(len(scales), dtype=np.int64)
    for index, scale in enumerate(scales):
        kernel, half_support = _sym4_kernel(float(scale))
        full = np.convolve(seq, kernel[::-1], mode="full")
        start = (len(full) - len(seq)) // 2
        coeffs[index] = full[start : start + len(seq)]
        half_supports[index] = half_support
    return coeffs, half_supports


def _normalized_inner_scalogram(coeffs: np.ndarray, half_supports: np.ndarray) -> np.ndarray:
    """Compute a normalized inner scalogram with basic edge protection.

    This follows the normalized inner-scalogram idea from Benitez/Bolos:
    coefficients whose wavelet support overlaps the signal boundary are removed,
    and the remaining CWT energy is normalized by the square root of the valid
    interior length so different scales remain comparable.
    """

    num_scales, length = coeffs.shape
    scalogram = np.zeros(num_scales, dtype=np.float64)
    for index, half_support in enumerate(half_supports):
        margin = int(half_support)
        start = min(max(margin, 0), length - 1)
        stop = max(length - margin, start + 1)
        inner_coeffs = coeffs[index, start:stop]
        inner_length = max(stop - start, 1)
        scalogram[index] = float(np.linalg.norm(inner_coeffs) / np.sqrt(inner_length))
    return scalogram


def _scale_index_from_scalogram(
    scalogram: np.ndarray,
    scales: np.ndarray,
    study_scale_count: int,
    eps: float = 1e-12,
) -> tuple[float, int, int]:
    """Compute the standard scale-index ratio from a discrete scalogram.

    The main scale is searched inside the study interval ``[s0, s1]`` and the
    minimum is searched afterwards in the extended interval ``[s_max, 2*s1]``.
    """

    if len(scalogram) < 2 or study_scale_count < 1:
        raise ValueError("At least two scales are required for scale-index analysis.")
    upper_index = min(int(study_scale_count), len(scalogram))
    study_scalogram = scalogram[:upper_index]
    max_index = int(np.argmax(study_scalogram))
    max_value = float(scalogram[max_index])
    if max_value <= eps:
        return 0.0, max_index, max_index
    min_index = max_index + int(np.argmin(scalogram[max_index:]))
    min_value = float(scalogram[min_index])
    return float(np.clip(min_value / max_value, 0.0, 1.0)), max_index, min_index


def _scale_index_1d(
    seq: np.ndarray,
    wavelet: str = DEFAULT_SI_WAVELET,
    scales: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[float, int, int, list[float], np.ndarray, int, int]:
    """Compute one-dimensional scale index on a 16-bit quantized sequence."""

    if wavelet not in ("morlet", "sym4"):
        raise ValueError("wavelet must be 'morlet' or 'sym4'.")

    raw = np.asarray(seq)
    if raw.ndim != 1:
        raise ValueError("Sequence must be one-dimensional.")
    seq_u16 = prepare_u16_sequence_for_scale_index(raw).astype(np.float64)
    raw_length = int(len(seq_u16))
    scale_array = _default_scales(raw_length) if scales is None else np.asarray(scales, dtype=np.float64)
    if wavelet == "morlet":
        coeffs, half_supports = _morlet_cwt(seq_u16, scale_array)
    else:
        coeffs, half_supports = _sym4_cwt(seq_u16, scale_array)
    scalogram = _normalized_inner_scalogram(coeffs, half_supports)
    study_scale_count = int(np.sum(scale_array <= DEFAULT_SI_STUDY_MAX_SCALE))
    study_scale_count = max(study_scale_count, 1)
    si_value, s_max_index, s_min_index = _scale_index_from_scalogram(
        scalogram,
        scales=scale_array,
        study_scale_count=study_scale_count,
        eps=eps,
    )
    return si_value, raw_length, raw_length, [float(scale) for scale in scale_array], scalogram, s_max_index, s_min_index


def scale_index_cwt(
    signal: np.ndarray,
    component: int | None = None,
    mode: str = "direct",
    wavelet: str = DEFAULT_SI_WAVELET,
    scales: np.ndarray | None = None,
    eps: float = 1e-12,
) -> ScaleIndexResult:
    """Compute the standard CWT-based Scale Index for 1D or 4D signals.

    The current implementation follows the standard global scale index rather
    than the windowed extension. For 4D trajectories, each state component is
    quantized into the shared 16-bit integer domain, analyzed independently,
    and then averaged:

        si_mean = (si_x1 + si_x2 + si_x3 + si_x4) / 4
    """

    if mode != "direct":
        raise ValueError("Only direct mode is supported for Scale Index in this project.")

    arr = np.asarray(signal)
    if arr.ndim == 1:
        si_value, raw_length, test_length, scale_values, _, _, _ = _scale_index_1d(
            arr,
            wavelet=wavelet,
            scales=scales,
            eps=eps,
        )
        return {
            "si_components": [float(si_value)],
            "si_mean": float(si_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": mode,
            "wavelet": wavelet,
            "scales": scale_values,
        }

    traj = ensure_trajectory_2d(np.asarray(signal, dtype=np.float64))
    dimension = int(traj.shape[1])
    if component is not None:
        if component < 0 or component >= dimension:
            raise ValueError(f"component must be in [0, {dimension - 1}].")
        si_value, raw_length, test_length, scale_values, _, _, _ = _scale_index_1d(
            traj[:, component],
            wavelet=wavelet,
            scales=scales,
            eps=eps,
        )
        return {
            "si_components": [float(si_value)],
            "si_mean": float(si_value),
            "raw_length": int(raw_length),
            "test_length": int(test_length),
            "mode": mode,
            "wavelet": wavelet,
            "scales": scale_values,
        }

    component_values: list[float] = []
    raw_length = 0
    test_length = 0
    scale_values: list[float] = []
    for component_index in range(dimension):
        si_value, raw_length, test_length, scale_values, _, _, _ = _scale_index_1d(
            traj[:, component_index],
            wavelet=wavelet,
            scales=scales,
            eps=eps,
        )
        component_values.append(float(si_value))

    return {
        "si_components": component_values,
        "si_mean": float(np.mean(component_values)),
        "raw_length": int(raw_length),
        "test_length": int(test_length),
        "mode": mode,
        "wavelet": wavelet,
        "scales": scale_values,
    }


def nonperiodicity_smoke(traj: np.ndarray) -> np.ndarray:
    """Return four SI values on a short leading window for smoke testing."""

    arr = ensure_trajectory_2d(np.asarray(traj, dtype=np.float64))
    smoke_window = min(len(arr), 256)
    result = scale_index_cwt(arr[:smoke_window])
    return np.asarray(result["si_components"], dtype=np.float64)


def scale_index_cwt_diagnostics(
    signal: np.ndarray,
    wavelet: str = DEFAULT_SI_WAVELET,
    scales: np.ndarray | None = None,
    eps: float = 1e-12,
) -> ScaleIndexDiagnostics:
    """Return detailed one-dimensional SI diagnostics for scalogram analysis."""

    arr = np.asarray(signal)
    if arr.ndim != 1:
        raise ValueError("Diagnostics require a one-dimensional sequence.")

    (
        si_value,
        raw_length,
        test_length,
        scale_values,
        scalogram,
        s_max_index,
        s_min_index,
    ) = _scale_index_1d(arr, wavelet=wavelet, scales=scales, eps=eps)
    scales_arr = np.asarray(scale_values, dtype=np.float64)
    study_scale_max = float(DEFAULT_SI_STUDY_MAX_SCALE)
    extended_scale_max = float(np.max(scales_arr))
    return {
        "si_value": float(si_value),
        "raw_length": int(raw_length),
        "test_length": int(test_length),
        "mode": "direct",
        "wavelet": wavelet,
        "scales": scale_values,
        "scalogram": [float(value) for value in np.asarray(scalogram, dtype=np.float64)],
        "study_scale_max": study_scale_max,
        "extended_scale_max": extended_scale_max,
        "s_max_index": int(s_max_index),
        "s_max_scale": float(scales_arr[s_max_index]),
        "s_max_value": float(np.asarray(scalogram, dtype=np.float64)[s_max_index]),
        "s_min_index": int(s_min_index),
        "s_min_scale": float(scales_arr[s_min_index]),
        "s_min_value": float(np.asarray(scalogram, dtype=np.float64)[s_min_index]),
    }


def scales_to_json(scales: list[float]) -> str:
    """Serialize scales for CSV-friendly storage."""

    return json.dumps(scales, ensure_ascii=False)
