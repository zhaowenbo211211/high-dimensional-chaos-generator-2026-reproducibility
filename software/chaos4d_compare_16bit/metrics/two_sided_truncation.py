# Maintainer: zhaowenbo
"""Two-sided truncation helpers for NIST SP 800-22 experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from metrics.nist_randomness import (
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
    generate_representative_enhanced_4d_words,
)
from utils.io_utils import ensure_directory


DEFAULT_TRUNCATE_HIGH_BITS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_TRUNCATE_LOW_BITS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_MIN_KEPT_MID_BITS = 6
DEFAULT_MIDBIT_ORDER = "msb"


def valid_two_sided_truncation_pairs(
    high_candidates: list[int] | tuple[int, ...] = DEFAULT_TRUNCATE_HIGH_BITS,
    low_candidates: list[int] | tuple[int, ...] = DEFAULT_TRUNCATE_LOW_BITS,
    min_kept_mid_bits: int = DEFAULT_MIN_KEPT_MID_BITS,
) -> list[tuple[int, int]]:
    """Return valid `(h, l)` pairs satisfying the minimum retained-width constraint."""

    pairs: list[tuple[int, int]] = []
    for high_bits in high_candidates:
        for low_bits in low_candidates:
            kept_mid_bits = 16 - int(high_bits) - int(low_bits)
            if kept_mid_bits >= min_kept_mid_bits:
                pairs.append((int(high_bits), int(low_bits)))
    return pairs


def extract_mid_bits(words: np.ndarray, truncate_high_bits_h: int, truncate_low_bits_l: int) -> np.ndarray:
    """Extract the retained middle-bit field and compact it into the low bits."""

    kept_mid_bits = 16 - int(truncate_high_bits_h) - int(truncate_low_bits_l)
    if truncate_high_bits_h < 0 or truncate_low_bits_l < 0:
        raise ValueError("truncate_high_bits_h and truncate_low_bits_l must be non-negative.")
    if kept_mid_bits <= 0 or kept_mid_bits > 16:
        raise ValueError("the retained middle-bit width must be in [1, 16].")
    mask = (1 << kept_mid_bits) - 1 if kept_mid_bits < 16 else 0xFFFF
    return np.bitwise_and(words.astype(np.uint16, copy=False) >> int(truncate_low_bits_l), mask).astype(np.uint16, copy=False)


def pack_midbit_words_to_bitstream(words: np.ndarray, kept_mid_bits: int, total_bit_length: int) -> bytes:
    """Pack compacted middle-bit words from MSB to LSB within the retained field."""

    if kept_mid_bits <= 0 or kept_mid_bits > 16:
        raise ValueError("kept_mid_bits must be in [1, 16].")
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")
    required_words = (int(total_bit_length) + kept_mid_bits - 1) // kept_mid_bits
    selected_words = words[:required_words].astype(np.uint16, copy=False)
    shifts = np.arange(kept_mid_bits - 1, -1, -1, dtype=np.uint16)
    bits = ((selected_words[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(bits[:total_bit_length], bitorder="big")
    return packed.tobytes()


def precompute_raw_words_for_midbit_search(
    valid_pairs: list[tuple[int, int]],
    total_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH * DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    delay_d: int = 4,
) -> dict[int, np.ndarray]:
    """Precompute raw post-warmup state words long enough for stages A and B."""

    if delay_d < 0:
        raise ValueError("delay_d must be non-negative.")
    minimum_kept_mid_bits = min(16 - h_value - l_value for h_value, l_value in valid_pairs)
    total_words = (int(total_bit_length) + minimum_kept_mid_bits - 1) // minimum_kept_mid_bits + int(delay_d)
    raw_words = generate_representative_enhanced_4d_words(word_count=total_words, warmup_steps=warmup_steps)
    return {direction + 1: raw_words[direction].astype(np.uint16, copy=False) for direction in range(4)}


def save_midbit_payload(
    payload: bytes,
    output_name: str,
    diagnostics_dir: str | Path | None = None,
) -> None:
    """Optionally persist a generated payload for diagnostics."""

    if diagnostics_dir is None:
        return
    diagnostics_path = ensure_directory(diagnostics_dir)
    (diagnostics_path / output_name).write_bytes(payload)
