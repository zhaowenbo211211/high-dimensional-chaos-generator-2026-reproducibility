# Maintainer: zhaowenbo
"""NIST SP 800-22 randomness evaluation for the representative enhanced 4D instance."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from math import erfc, exp, factorial, floor, lgamma, log, pi, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np

from utils.fixed16 import U16_MOD, float_vec_to_u16
from utils.initializers import get_common_initial_state_float
from utils.io_utils import ensure_directory


NIST_SIGNIFICANCE_LEVEL = 0.01
NIST_UNIFORMITY_THRESHOLD = 0.0001
DEFAULT_NIST_SEQUENCE_BIT_LENGTH = 1_000_000
DEFAULT_NIST_SEQUENCE_COUNT = 100
DEFAULT_NIST_WARMUP_STEPS = 1000
REPRESENTATIVE_A_LINEAR = (371, 373, 375)
REPRESENTATIVE_A_COUPLING = (50, 48, 46)
REPRESENTATIVE_B = 111
DEFAULT_NIST_BIT_LENGTH = DEFAULT_NIST_SEQUENCE_BIT_LENGTH * DEFAULT_NIST_SEQUENCE_COUNT
DEFAULT_BIT_ORDER = "msb"
DEFAULT_TEMPLATE_BITS = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
DEFAULT_OVERLAPPING_TEMPLATE_BITS = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
UNIVERSAL_EXPECTED = {
    6: 5.2177052,
    7: 6.1962507,
    8: 7.1836656,
    9: 8.1764248,
    10: 9.1723243,
    11: 10.170032,
    12: 11.168765,
    13: 12.16807,
    14: 13.167693,
    15: 14.167488,
    16: 15.167379,
}
UNIVERSAL_VARIANCE = {
    6: 2.954,
    7: 3.125,
    8: 3.238,
    9: 3.311,
    10: 3.356,
    11: 3.384,
    12: 3.401,
    13: 3.41,
    14: 3.416,
    15: 3.419,
    16: 3.421,
}
LINEAR_COMPLEXITY_PI = np.asarray([0.01047, 0.03125, 0.12500, 0.50000, 0.25000, 0.06250, 0.020833], dtype=np.float64)


@dataclass(frozen=True)
class NISTTestResult:
    """Structured result for one test on one direction."""

    direction: int
    test_name: str
    p_value: float
    passed: bool
    proportion: float
    bit_length: int
    significance_level: float
    sequence_index: int = 1
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NISTAggregateResult:
    """Two-level result across multiple sequences for one direction and one test."""

    direction: int
    test_name: str
    sequence_count: int
    sequence_bit_length: int
    total_bit_length: int
    pass_count: int
    proportion: float
    minimum_pass_proportion: float
    uniformity_p_value: float
    passed: bool
    significance_level: float
    min_p_value: float
    max_p_value: float
    mean_p_value: float
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _regularized_gamma_q(a: float, x: float, eps: float = 1e-12, max_iter: int = 200) -> float:
    """Return the regularized upper incomplete gamma function Q(a, x)."""

    if a <= 0.0:
        raise ValueError("a must be positive.")
    if x < 0.0:
        raise ValueError("x must be non-negative.")
    if x == 0.0:
        return 1.0

    gln = lgamma(a)
    if x < a + 1.0:
        ap = a
        delta = 1.0 / a
        series = delta
        for _ in range(max_iter):
            ap += 1.0
            delta *= x / ap
            series += delta
            if abs(delta) < abs(series) * eps:
                break
        p = series * exp(-x + a * log(x) - gln)
        return float(max(0.0, min(1.0, 1.0 - p)))

    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    q = exp(-x + a * log(x) - gln) * h
    return float(max(0.0, min(1.0, q)))


def _bits_from_packed_bytes(bitstream_bytes: bytes, bit_length: int, bit_order: str = DEFAULT_BIT_ORDER) -> np.ndarray:
    """Unpack packed bytes into a 0/1 bit array."""

    unpack_order = "big" if bit_order == "msb" else "little"
    bits = np.unpackbits(np.frombuffer(bitstream_bytes, dtype=np.uint8), bitorder=unpack_order)
    return bits[:bit_length].astype(np.uint8, copy=False)


def _phi_normal(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2.0))


def _sliding_windows(arr: np.ndarray, width: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(arr, width)


def _pattern_codes_circular(bits: np.ndarray, m: int) -> np.ndarray:
    extended = np.concatenate((bits, bits[: m - 1])) if m > 1 else bits
    windows = _sliding_windows(extended, m)
    weights = (1 << np.arange(m - 1, -1, -1, dtype=np.int64)).astype(np.int64)
    return (windows.astype(np.int64) @ weights).astype(np.int64)


def _pattern_counts_circular(bits: np.ndarray, m: int) -> np.ndarray:
    codes = _pattern_codes_circular(bits, m)
    return np.bincount(codes, minlength=1 << m).astype(np.int64)


def _template_match_counts_overlapping(bits: np.ndarray, template: np.ndarray) -> int:
    windows = _sliding_windows(bits, len(template))
    return int(np.sum(np.all(windows == template[None, :], axis=1)))


def _template_match_counts_non_overlapping(bits: np.ndarray, template: np.ndarray) -> int:
    if len(bits) < len(template):
        return 0
    windows = _sliding_windows(bits, len(template))
    matches = np.flatnonzero(np.all(windows == template[None, :], axis=1))
    if len(matches) == 0:
        return 0
    count = 1
    last = int(matches[0])
    step = len(template)
    for index in matches[1:]:
        index_int = int(index)
        if index_int >= last + step:
            count += 1
            last = index_int
    return count


def _berlekamp_massey(block: np.ndarray) -> int:
    n = len(block)
    c = 1
    b = 1
    l_value = 0
    m_value = -1
    history = 0
    for n_idx in range(n):
        history = (history << 1) | int(block[n_idx])
        discrepancy = (c & history).bit_count() & 1
        if discrepancy == 1:
            temp = c
            c ^= b << (n_idx - m_value)
            if l_value <= n_idx / 2:
                l_value = n_idx + 1 - l_value
                m_value = n_idx
                b = temp
    return int(l_value)


def _random_excursions_probabilities(state_abs: int) -> np.ndarray:
    probs = np.zeros(6, dtype=np.float64)
    probs[0] = 1.0 - 1.0 / (2.0 * state_abs)
    for k in range(1, 5):
        probs[k] = (1.0 / (4.0 * state_abs * state_abs)) * ((1.0 - 1.0 / (2.0 * state_abs)) ** (k - 1))
    probs[5] = (1.0 / (2.0 * state_abs)) * ((1.0 - 1.0 / (2.0 * state_abs)) ** 4)
    return probs


def _quantize_float_to_u16(x: float) -> int:
    wrapped = x % 1.0
    return int(floor(wrapped * U16_MOD)) % U16_MOD


def _next_representative_enhanced_4d_state(
    x1: int,
    x2: int,
    x3: int,
    x4: int,
    x4_z1: int,
    x4_z2: int,
    rotation_index: int,
) -> tuple[int, int, int, int, int, int]:
    x1_next = (REPRESENTATIVE_A_LINEAR[0] * x1 + REPRESENTATIVE_A_COUPLING[0] * x4_z2) % U16_MOD
    x2_next = (REPRESENTATIVE_A_LINEAR[1] * x2 + REPRESENTATIVE_A_COUPLING[1] * x4_z1) % U16_MOD
    x3_next = (REPRESENTATIVE_A_LINEAR[2] * x3 + REPRESENTATIVE_A_COUPLING[2] * x4) % U16_MOD
    x2_float = x2 / U16_MOD
    logistic_sum = (x3 / U16_MOD) + (x4 / U16_MOD)
    x4_next = _quantize_float_to_u16(REPRESENTATIVE_B * x2_float + 4.0 * logistic_sum * (1.0 - logistic_sum))

    packed = (
        ((x1_next & 0xFFFF) << 48)
        | ((x2_next & 0xFFFF) << 32)
        | ((x3_next & 0xFFFF) << 16)
        | (x4_next & 0xFFFF)
    )
    rotation = 1 + (rotation_index % 15)
    rotated = ((packed << rotation) & ((1 << 64) - 1)) | (packed >> (64 - rotation))
    x1_m = (rotated >> 48) & 0xFFFF
    x2_m = (rotated >> 32) & 0xFFFF
    x3_m = (rotated >> 16) & 0xFFFF
    x4_m = rotated & 0xFFFF
    return x1_m, x2_m, x3_m, x4_m, x4_m, x4_z1


def _generate_representative_enhanced_4d_words(
    word_count: int,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> np.ndarray:
    """Generate representative enhanced 4D state words with minimal overhead."""

    if word_count < 0:
        raise ValueError("word_count must be non-negative.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")

    states = np.empty((4, word_count), dtype=np.uint16)
    initial_state_u16 = float_vec_to_u16(get_common_initial_state_float())
    x1 = int(initial_state_u16[0])
    x2 = int(initial_state_u16[1])
    x3 = int(initial_state_u16[2])
    x4 = int(initial_state_u16[3])
    x4_z1 = 0
    x4_z2 = 0

    for index in range(warmup_steps):
        x1, x2, x3, x4, x4_z1, x4_z2 = _next_representative_enhanced_4d_state(
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x4_z1=x4_z1,
            x4_z2=x4_z2,
            rotation_index=index,
        )

    for index in range(word_count):
        x1_m, x2_m, x3_m, x4_m, x4_z1, x4_z2 = _next_representative_enhanced_4d_state(
            x1=x1,
            x2=x2,
            x3=x3,
            x4=x4,
            x4_z1=x4_z1,
            x4_z2=x4_z2,
            rotation_index=warmup_steps + index,
        )

        states[0, index] = x1_m
        states[1, index] = x2_m
        states[2, index] = x3_m
        states[3, index] = x4_m

        x1 = x1_m
        x2 = x2_m
        x3 = x3_m
        x4 = x4_m

    return states


def generate_representative_enhanced_4d_words(
    word_count: int,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> np.ndarray:
    """Return enhanced 4D state words after the configured warm-up discard."""

    return _generate_representative_enhanced_4d_words(word_count=word_count, warmup_steps=warmup_steps)


def pack_u16_words_to_bitstream(words: np.ndarray, bit_length: int, bit_order: str = DEFAULT_BIT_ORDER) -> bytes:
    """Pack 16-bit state words into a byte-aligned bitstream with optional truncation."""

    if bit_length <= 0:
        raise ValueError("bit_length must be positive.")
    if bit_order not in ("msb", "lsb"):
        raise ValueError("bit_order must be 'msb' or 'lsb'.")
    total_bytes = (bit_length + 7) // 8
    byte_order = "big" if bit_order == "msb" else "little"
    mask = 0xFF << (8 - (bit_length % 8)) if bit_length % 8 else 0xFF
    payload = words.astype(np.uint16).byteswap().tobytes() if byte_order == "big" else words.astype(np.uint16).tobytes()
    payload = payload[:total_bytes]
    if bit_length % 8 and bit_order == "msb":
        trimmed = bytearray(payload)
        trimmed[-1] &= mask
        payload = bytes(trimmed)
    return payload


def generate_representative_enhanced_4d_bitstreams(
    bit_length: int = DEFAULT_NIST_BIT_LENGTH,
    bit_order: str = DEFAULT_BIT_ORDER,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    diagnostics_dir: str | Path | None = None,
) -> dict[int, bytes]:
    """Generate one packed bitstream per state direction.

    Bit extraction rule:
    - the first `warmup_steps` enhanced states are discarded as transient warm-up;
    - one 16-bit state word is taken from one direction at every iteration;
    - the word is written MSB-first or LSB-first according to `bit_order`;
    - the stream is extended until exactly `bit_length` bits are obtained;
    - no additional tail post-processing is applied.
    """

    if bit_length <= 0:
        raise ValueError("bit_length must be positive.")
    if bit_order not in ("msb", "lsb"):
        raise ValueError("bit_order must be 'msb' or 'lsb'.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")

    word_count = (bit_length + 15) // 16
    states = _generate_representative_enhanced_4d_words(word_count=word_count, warmup_steps=warmup_steps)
    streams: dict[int, bytes] = {}
    diagnostics_path = ensure_directory(diagnostics_dir) if diagnostics_dir is not None else None
    for direction in range(4):
        payload = pack_u16_words_to_bitstream(states[direction], bit_length=bit_length, bit_order=bit_order)
        streams[direction + 1] = payload
        if diagnostics_path is not None:
            (diagnostics_path / f"direction_{direction + 1}_{bit_length}bit.bin").write_bytes(payload)
    return streams


def generate_representative_enhanced_4d_bitstream_sequences(
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    bit_order: str = DEFAULT_BIT_ORDER,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    diagnostics_dir: str | Path | None = None,
) -> dict[int, list[bytes]]:
    """Generate directional bitstreams as 100 independent `10^6`-bit sequences by default."""

    if sequence_bit_length <= 0:
        raise ValueError("sequence_bit_length must be positive.")
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive.")
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    combined = generate_representative_enhanced_4d_bitstreams(
        bit_length=total_bit_length,
        bit_order=bit_order,
        warmup_steps=warmup_steps,
        diagnostics_dir=diagnostics_dir,
    )
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    outputs: dict[int, list[bytes]] = {}
    for direction, payload in combined.items():
        outputs[direction] = [
            payload[index * bytes_per_sequence : (index + 1) * bytes_per_sequence]
            for index in range(sequence_count)
        ]
    return outputs


def monobit_frequency_test(bits: np.ndarray) -> tuple[float, str]:
    n = len(bits)
    s_n = int(np.sum(bits.astype(np.int64) * 2 - 1))
    s_obs = abs(s_n) / sqrt(n)
    return erfc(s_obs / sqrt(2.0)), ""


def block_frequency_test(bits: np.ndarray, block_size: int = 128) -> tuple[float, str]:
    n = len(bits)
    block_count = n // block_size
    if block_count == 0:
        return 0.0, "insufficient data"
    trimmed = bits[: block_count * block_size].reshape(block_count, block_size)
    proportions = np.mean(trimmed, axis=1)
    chi_square = 4.0 * block_size * float(np.sum((proportions - 0.5) ** 2))
    p_value = _regularized_gamma_q(block_count / 2.0, chi_square / 2.0)
    return p_value, f"M={block_size}, N={block_count}"


def runs_test(bits: np.ndarray) -> tuple[float, str]:
    n = len(bits)
    pi = float(np.mean(bits))
    tau = 2.0 / sqrt(n)
    if abs(pi - 0.5) >= tau:
        return 0.0, "prerequisite frequency condition failed"
    runs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    numerator = abs(runs - 2.0 * n * pi * (1.0 - pi))
    denominator = 2.0 * sqrt(2.0 * n) * pi * (1.0 - pi)
    return erfc(numerator / denominator), f"pi={pi:.6f}, Vn={runs}"


def _longest_run_params(n: int) -> tuple[int, int, list[float], list[int]]:
    if n < 128:
        raise ValueError("Longest-run test requires at least 128 bits.")
    if n < 6272:
        return 8, 3, [0.2148, 0.3672, 0.2305, 0.1875], [1, 2, 3]
    if n < 750000:
        return 128, 5, [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124], [4, 5, 6, 7, 8]
    return 10000, 6, [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727], [10, 11, 12, 13, 14, 15]


def _max_run_of_ones(block: np.ndarray) -> int:
    zeros = np.flatnonzero(block == 0)
    if len(zeros) == 0:
        return int(len(block))
    boundaries = np.concatenate(([-1], zeros, [len(block)]))
    return int(np.max(np.diff(boundaries) - 1))


def longest_run_of_ones_test(bits: np.ndarray) -> tuple[float, str]:
    n = len(bits)
    block_size, k_value, pi_values, thresholds = _longest_run_params(n)
    block_count = n // block_size
    trimmed = bits[: block_count * block_size].reshape(block_count, block_size)
    frequencies = np.zeros(len(pi_values), dtype=np.int64)
    for block in trimmed:
        run = _max_run_of_ones(block)
        if run <= thresholds[0]:
            frequencies[0] += 1
        elif run >= thresholds[-1] + 1:
            frequencies[-1] += 1
        else:
            for idx in range(1, len(pi_values) - 1):
                if run == thresholds[idx]:
                    frequencies[idx] += 1
                    break
    expected = block_count * np.asarray(pi_values, dtype=np.float64)
    chi_square = float(np.sum((frequencies - expected) ** 2 / expected))
    p_value = _regularized_gamma_q(k_value / 2.0, chi_square / 2.0)
    return p_value, f"M={block_size}, N={block_count}"


def _gf2_rank_32(matrix_rows: Iterable[int]) -> int:
    rows = [int(value) & 0xFFFFFFFF for value in matrix_rows]
    rank = 0
    for column in range(31, -1, -1):
        pivot = None
        for row_index in range(rank, 32):
            if (rows[row_index] >> column) & 1:
                pivot = row_index
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_row = rows[rank]
        for row_index in range(32):
            if row_index != rank and ((rows[row_index] >> column) & 1):
                rows[row_index] ^= pivot_row
        rank += 1
        if rank == 32:
            break
    return rank


def binary_matrix_rank_test(bits: np.ndarray) -> tuple[float, str]:
    matrix_size = 32
    block_bits = matrix_size * matrix_size
    matrix_count = len(bits) // block_bits
    if matrix_count == 0:
        return 0.0, "insufficient data"
    trimmed = bits[: matrix_count * block_bits].reshape(matrix_count, matrix_size, matrix_size)
    full_rank = 0
    rank_minus_one = 0
    for matrix in trimmed:
        row_values = []
        for row in matrix:
            value = 0
            for bit in row:
                value = (value << 1) | int(bit)
            row_values.append(value)
        rank = _gf2_rank_32(row_values)
        if rank == 32:
            full_rank += 1
        elif rank == 31:
            rank_minus_one += 1
    remainder = matrix_count - full_rank - rank_minus_one
    probabilities = np.asarray([0.2888, 0.5776, 0.1336], dtype=np.float64)
    observed = np.asarray([full_rank, rank_minus_one, remainder], dtype=np.float64)
    expected = matrix_count * probabilities
    chi_square = float(np.sum((observed - expected) ** 2 / expected))
    p_value = exp(-chi_square / 2.0)
    return p_value, f"N={matrix_count}"


def spectral_dft_test(bits: np.ndarray) -> tuple[float, str]:
    n = len(bits)
    sequence = bits.astype(np.float64) * 2.0 - 1.0
    spectrum = np.fft.fft(sequence)
    modulus = np.abs(spectrum[: n // 2])
    threshold = sqrt(log(1.0 / 0.05) * n)
    count_below = int(np.sum(modulus < threshold))
    expected = 0.95 * n / 2.0
    variance = n * 0.95 * 0.05 / 4.0
    d_value = (count_below - expected) / sqrt(variance)
    return erfc(abs(d_value) / sqrt(2.0)), f"N1={count_below}, T={threshold:.3f}"


def non_overlapping_template_matching_test(bits: np.ndarray, template: np.ndarray = DEFAULT_TEMPLATE_BITS) -> tuple[float, str]:
    n = len(bits)
    m = len(template)
    block_count = 8
    block_size = n // block_count
    if block_size < m:
        return 0.0, "insufficient data"
    counts = np.zeros(block_count, dtype=np.int64)
    for block_index in range(block_count):
        block = bits[block_index * block_size : (block_index + 1) * block_size]
        counts[block_index] = _template_match_counts_non_overlapping(block, template)
    mu = (block_size - m + 1) / float(1 << m)
    variance = block_size * ((1.0 / float(1 << m)) - ((2.0 * m - 1.0) / float(1 << (2 * m))))
    chi_square = float(np.sum((counts - mu) ** 2 / variance))
    p_value = _regularized_gamma_q(block_count / 2.0, chi_square / 2.0)
    template_string = "".join(str(int(bit)) for bit in template)
    return p_value, f"m={m}, B={template_string}, M={block_size}, N={block_count}"


def overlapping_template_matching_test(bits: np.ndarray, template: np.ndarray = DEFAULT_OVERLAPPING_TEMPLATE_BITS) -> tuple[float, str]:
    m = len(template)
    block_size = 1032
    block_count = len(bits) // block_size
    if block_count == 0:
        return 0.0, "insufficient data"
    eta = block_size / float(1 << m)
    trimmed = bits[: block_count * block_size].reshape(block_count, block_size)
    counts = np.zeros(6, dtype=np.int64)
    poisson = [exp(-eta)]
    for u in range(1, 5):
        poisson.append(exp(-eta) * (eta**u) / float(factorial(u)))
    poisson.append(1.0 - float(np.sum(poisson)))
    for block in trimmed:
        match_count = _template_match_counts_overlapping(block, template)
        counts[min(match_count, 5)] += 1
    expected = block_count * np.asarray(poisson, dtype=np.float64)
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    p_value = _regularized_gamma_q(5.0 / 2.0, chi_square / 2.0)
    return p_value, f"m={m}, M={block_size}, N={block_count}"


def universal_statistical_test(bits: np.ndarray) -> tuple[float, str]:
    n = len(bits)
    thresholds = [
        (1_059_061_760, 16),
        (496_435_200, 15),
        (231_669_760, 14),
        (107_560_960, 13),
        (49_643_520, 12),
        (22_753_280, 11),
        (10_342_400, 10),
        (4_654_080, 9),
        (2_068_480, 8),
        (904_960, 7),
        (387_840, 6),
    ]
    l_value = None
    for minimum_n, candidate_l in thresholds:
        if n >= minimum_n:
            l_value = candidate_l
            break
    if l_value is None:
        return 0.0, "insufficient data"
    q_value = 10 * (1 << l_value)
    block_count = n // l_value
    k_value = block_count - q_value
    if k_value <= 0:
        return 0.0, "insufficient data"
    trimmed = bits[: block_count * l_value].reshape(block_count, l_value)
    weights = (1 << np.arange(l_value - 1, -1, -1, dtype=np.int64)).astype(np.int64)
    pattern_values = (trimmed.astype(np.int64) @ weights).astype(np.int64)
    table = np.zeros(1 << l_value, dtype=np.int64)
    for idx in range(q_value):
        table[pattern_values[idx]] = idx + 1
    total = 0.0
    for idx in range(q_value, q_value + k_value):
        distance = idx + 1 - int(table[pattern_values[idx]])
        table[pattern_values[idx]] = idx + 1
        total += log(distance, 2)
    fn_value = total / float(k_value)
    c_value = 0.7 - 0.8 / float(l_value) + (4.0 + 32.0 / float(l_value)) * (k_value ** (-3.0 / float(l_value))) / 15.0
    sigma = c_value * sqrt(UNIVERSAL_VARIANCE[l_value] / float(k_value))
    p_value = erfc(abs(fn_value - UNIVERSAL_EXPECTED[l_value]) / (sqrt(2.0) * sigma))
    return p_value, f"L={l_value}, Q={q_value}, K={k_value}"


def linear_complexity_test(bits: np.ndarray, block_size: int = 500) -> tuple[float, str]:
    block_count = len(bits) // block_size
    if block_count == 0:
        return 0.0, "insufficient data"
    trimmed = bits[: block_count * block_size].reshape(block_count, block_size)
    mean = (
        block_size / 2.0
        + (9.0 + (-1.0) ** (block_size + 1)) / 36.0
        - (block_size / 3.0 + 2.0 / 9.0) / (2.0**block_size)
    )
    frequencies = np.zeros(7, dtype=np.int64)
    for block in trimmed:
        l_value = _berlekamp_massey(block)
        t_value = ((-1) ** block_size) * (l_value - mean) + 2.0 / 9.0
        if t_value <= -2.5:
            frequencies[0] += 1
        elif t_value <= -1.5:
            frequencies[1] += 1
        elif t_value <= -0.5:
            frequencies[2] += 1
        elif t_value <= 0.5:
            frequencies[3] += 1
        elif t_value <= 1.5:
            frequencies[4] += 1
        elif t_value <= 2.5:
            frequencies[5] += 1
        else:
            frequencies[6] += 1
    expected = block_count * LINEAR_COMPLEXITY_PI
    chi_square = float(np.sum((frequencies - expected) ** 2 / expected))
    p_value = _regularized_gamma_q(6.0 / 2.0, chi_square / 2.0)
    return p_value, f"M={block_size}, N={block_count}"


def serial_test(bits: np.ndarray, m: int = 16) -> tuple[float, str]:
    if m < 2:
        return 0.0, "m must be at least 2"

    def psi2(mm: int) -> float:
        counts = _pattern_counts_circular(bits, mm).astype(np.float64)
        return (float(np.sum(counts**2)) * (2.0**mm) / len(bits)) - len(bits)

    psi_m = psi2(m)
    psi_m1 = psi2(m - 1)
    psi_m2 = psi2(m - 2)
    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2.0 * psi_m1 + psi_m2
    p1 = _regularized_gamma_q((2 ** (m - 1)) / 2.0, delta1 / 2.0)
    p2 = _regularized_gamma_q((2 ** (m - 2)) / 2.0, delta2 / 2.0)
    p_value = min(p1, p2)
    return p_value, f"m={m}, p1={p1:.6f}, p2={p2:.6f}"


def approximate_entropy_test(bits: np.ndarray, m: int = 10) -> tuple[float, str]:
    if m < 1:
        return 0.0, "m must be positive"

    def phi(mm: int) -> float:
        counts = _pattern_counts_circular(bits, mm).astype(np.float64)
        probabilities = counts / float(len(bits))
        positive = probabilities[probabilities > 0.0]
        return float(np.sum(positive * np.log(positive)))

    ap_en = phi(m) - phi(m + 1)
    chi_square = 2.0 * len(bits) * (log(2.0) - ap_en)
    p_value = _regularized_gamma_q(2 ** (m - 1), chi_square / 2.0)
    return p_value, f"m={m}"


def cumulative_sums_test(bits: np.ndarray) -> tuple[float, str]:
    sequence = bits.astype(np.int8) * 2 - 1
    cumulative = np.cumsum(sequence, dtype=np.int32)
    z = int(np.max(np.abs(cumulative)))
    n = len(bits)
    if z == 0:
        return 1.0, "z=0"

    sum_a = 0.0
    start = int(floor((-n / z + 1.0) / 4.0))
    end = int(floor((n / z - 1.0) / 4.0))
    for k in range(start, end + 1):
        sum_a += _phi_normal((4 * k + 1) * z / sqrt(n)) - _phi_normal((4 * k - 1) * z / sqrt(n))

    sum_b = 0.0
    start = int(floor((-n / z - 3.0) / 4.0))
    end = int(floor((n / z - 1.0) / 4.0))
    for k in range(start, end + 1):
        sum_b += _phi_normal((4 * k + 3) * z / sqrt(n)) - _phi_normal((4 * k + 1) * z / sqrt(n))
    p_value = 1.0 - sum_a + sum_b
    return max(0.0, min(1.0, p_value)), f"z={z}"


def random_excursions_test(bits: np.ndarray) -> tuple[float, str]:
    walk = np.cumsum(bits.astype(np.int8) * 2 - 1, dtype=np.int32)
    walk = np.concatenate(([0], walk, [0]))
    zero_positions = np.flatnonzero(walk == 0)
    cycle_count = len(zero_positions) - 1
    if cycle_count < 500:
        return 0.0, f"insufficient excursions J={cycle_count}"

    state_p_values: list[float] = []
    failed_states = 0
    for state in [-4, -3, -2, -1, 1, 2, 3, 4]:
        visits = np.zeros(6, dtype=np.int64)
        for cycle_index in range(cycle_count):
            cycle = walk[zero_positions[cycle_index] : zero_positions[cycle_index + 1] + 1]
            count = int(np.sum(cycle == state))
            visits[min(count, 5)] += 1
        probs = _random_excursions_probabilities(abs(state))
        expected = cycle_count * probs
        chi_square = float(np.sum((visits - expected) ** 2 / expected))
        p_value = _regularized_gamma_q(5.0 / 2.0, chi_square / 2.0)
        state_p_values.append(p_value)
        if p_value < NIST_SIGNIFICANCE_LEVEL:
            failed_states += 1
    return min(state_p_values), f"J={cycle_count}, failed_states={failed_states}"


def random_excursions_variant_test(bits: np.ndarray) -> tuple[float, str]:
    walk = np.cumsum(bits.astype(np.int8) * 2 - 1, dtype=np.int32)
    walk = np.concatenate(([0], walk, [0]))
    zero_positions = np.flatnonzero(walk == 0)
    cycle_count = len(zero_positions) - 1
    if cycle_count < 500:
        return 0.0, f"insufficient excursions J={cycle_count}"

    state_p_values: list[float] = []
    failed_states = 0
    for state in list(range(-9, 0)) + list(range(1, 10)):
        count = int(np.sum(walk == state))
        denominator = sqrt(2.0 * cycle_count * (4.0 * abs(state) - 2.0))
        p_value = erfc(abs(count - cycle_count) / denominator)
        state_p_values.append(p_value)
        if p_value < NIST_SIGNIFICANCE_LEVEL:
            failed_states += 1
    return min(state_p_values), f"J={cycle_count}, failed_states={failed_states}"


CORE_NIST_TESTS: list[tuple[str, callable]] = [
    ("Frequency (Monobit)", monobit_frequency_test),
    ("Block Frequency", block_frequency_test),
    ("Runs", runs_test),
    ("Longest Run of Ones", longest_run_of_ones_test),
    ("Binary Matrix Rank", binary_matrix_rank_test),
    ("Spectral DFT", spectral_dft_test),
    ("Non-Overlapping Template Matching", non_overlapping_template_matching_test),
    ("Overlapping Template Matching", overlapping_template_matching_test),
    ("Maurer's Universal", universal_statistical_test),
    ("Linear Complexity", linear_complexity_test),
    ("Serial", serial_test),
    ("Approximate Entropy", approximate_entropy_test),
    ("Cumulative Sums", cumulative_sums_test),
    ("Random Excursions", random_excursions_test),
    ("Random Excursions Variant", random_excursions_variant_test),
]


def run_core_nist_sp800_22_suite(
    bitstream_bytes: bytes,
    direction: int,
    bit_length: int,
    bit_order: str = DEFAULT_BIT_ORDER,
    significance_level: float = NIST_SIGNIFICANCE_LEVEL,
    sequence_index: int = 1,
) -> list[NISTTestResult]:
    """Run the current core NIST SP 800-22 tests on one direction."""

    bits = _bits_from_packed_bytes(bitstream_bytes=bitstream_bytes, bit_length=bit_length, bit_order=bit_order)
    results: list[NISTTestResult] = []
    for test_name, test_fn in CORE_NIST_TESTS:
        p_value, notes = test_fn(bits)
        passed = bool(p_value >= significance_level)
        results.append(
            NISTTestResult(
                direction=int(direction),
                test_name=test_name,
                p_value=float(p_value),
                passed=passed,
                proportion=1.0 if passed else 0.0,
                bit_length=int(bit_length),
                significance_level=float(significance_level),
                sequence_index=int(sequence_index),
                notes=str(notes),
            )
        )
    return results


def _run_sequence_worker(
    *,
    bitstream_bytes: bytes,
    direction: int,
    bit_length: int,
    bit_order: str,
    significance_level: float,
    sequence_index: int,
) -> list[NISTTestResult]:
    return run_core_nist_sp800_22_suite(
        bitstream_bytes=bitstream_bytes,
        direction=direction,
        bit_length=bit_length,
        bit_order=bit_order,
        significance_level=significance_level,
        sequence_index=sequence_index,
    )


def _minimum_pass_proportion(significance_level: float, sequence_count: int) -> float:
    p_hat = 1.0 - significance_level
    margin = 3.0 * sqrt((p_hat * significance_level) / float(sequence_count))
    return p_hat - margin


def _uniformity_p_value(p_values: list[float]) -> float:
    if len(p_values) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(np.asarray(p_values, dtype=np.float64), bins=bins)
    expected = len(p_values) / 10.0
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    return _regularized_gamma_q(9.0 / 2.0, chi_square / 2.0)


def run_core_nist_sp800_22_suite_on_sequences(
    bitstream_sequences: list[bytes],
    direction: int,
    sequence_bit_length: int,
    bit_order: str = DEFAULT_BIT_ORDER,
    significance_level: float = NIST_SIGNIFICANCE_LEVEL,
    max_workers: int | None = None,
) -> tuple[list[NISTTestResult], list[NISTAggregateResult]]:
    """Run the current core suite on multiple sequences and aggregate proportion/uniformity."""

    detailed_results: list[NISTTestResult] = []
    aggregates: list[NISTAggregateResult] = []
    worker_count = max_workers if max_workers is not None else max(1, min(os.cpu_count() or 1, len(bitstream_sequences)))
    if worker_count <= 1:
        for sequence_index, bitstream in enumerate(bitstream_sequences, start=1):
            detailed_results.extend(
                run_core_nist_sp800_22_suite(
                    bitstream_bytes=bitstream,
                    direction=direction,
                    bit_length=sequence_bit_length,
                    bit_order=bit_order,
                    significance_level=significance_level,
                    sequence_index=sequence_index,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _run_sequence_worker,
                    bitstream_bytes=bitstream,
                    direction=direction,
                    bit_length=sequence_bit_length,
                    bit_order=bit_order,
                    significance_level=significance_level,
                    sequence_index=sequence_index,
                )
                for sequence_index, bitstream in enumerate(bitstream_sequences, start=1)
            ]
            for future in futures:
                detailed_results.extend(future.result())

    for test_name, _ in CORE_NIST_TESTS:
        matching = [result for result in detailed_results if result.direction == direction and result.test_name == test_name]
        p_values = [float(result.p_value) for result in matching]
        pass_count = sum(1 for result in matching if result.passed)
        proportion = pass_count / float(len(matching))
        min_prop = _minimum_pass_proportion(significance_level=significance_level, sequence_count=len(matching))
        uniformity = _uniformity_p_value(p_values)
        passed = bool(proportion >= min_prop and uniformity >= NIST_UNIFORMITY_THRESHOLD)
        notes = ""
        source_notes = {str(result.notes) for result in matching if str(result.notes)}
        if source_notes:
            sorted_notes = sorted(source_notes)
            preview = sorted_notes[:5]
            suffix = "" if len(sorted_notes) <= 5 else f" | ... ({len(sorted_notes)} unique notes)"
            notes = " | ".join(preview) + suffix
        aggregates.append(
            NISTAggregateResult(
                direction=int(direction),
                test_name=test_name,
                sequence_count=len(matching),
                sequence_bit_length=int(sequence_bit_length),
                total_bit_length=int(sequence_bit_length) * int(len(matching)),
                pass_count=int(pass_count),
                proportion=float(proportion),
                minimum_pass_proportion=float(min_prop),
                uniformity_p_value=float(uniformity),
                passed=passed,
                significance_level=float(significance_level),
                min_p_value=float(min(p_values)),
                max_p_value=float(max(p_values)),
                mean_p_value=float(np.mean(p_values)),
                notes=notes,
            )
        )
    return detailed_results, aggregates
