# Maintainer: zhaowenbo
"""High-bit truncation search built on top of single-direction delay-XOR whitening."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from metrics.delay_xor_whitening import generate_delay_xor_words
from metrics.direction_fusion_whitening import DEFAULT_BEST_DELAY_BY_DIRECTION
from metrics.nist_randomness import (
    CORE_NIST_TESTS,
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
    NISTAggregateResult,
    run_core_nist_sp800_22_suite_on_sequences,
)
from utils.io_utils import ensure_directory


DEFAULT_TRUNCATION_CANDIDATES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_TRUNCATION_BIT_ORDER = "msb"


@dataclass(frozen=True)
class TruncationCandidateResult:
    direction: int
    delay_d: int
    truncate_high_bits_k: int
    kept_low_bits: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    all_passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BestTruncationDirection:
    direction: int
    delay_d: int
    truncate_high_bits_k: int
    kept_low_bits: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    all_passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _truncation_rank_key(result: BestTruncationDirection) -> tuple[int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.truncate_high_bits_k),
        int(result.kept_low_bits),
    )


def pack_truncated_low_bits_to_bitstream(words: np.ndarray, kept_low_bits: int, total_bit_length: int) -> bytes:
    """Pack the low kept bits of each 16-bit word using MSB-to-LSB order within the kept block."""

    if kept_low_bits <= 0 or kept_low_bits > 16:
        raise ValueError("kept_low_bits must be in [1, 16].")
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")

    required_words = (int(total_bit_length) + kept_low_bits - 1) // kept_low_bits
    selected_words = words[:required_words].astype(np.uint16, copy=False)
    mask = (1 << kept_low_bits) - 1
    truncated = np.bitwise_and(selected_words, mask).astype(np.uint16, copy=False)
    shifts = np.arange(kept_low_bits - 1, -1, -1, dtype=np.uint16)
    bits = ((truncated[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(bits[:total_bit_length], bitorder="big")
    return packed.tobytes()


def generate_truncated_bitstream_sequences(
    direction: int,
    truncate_high_bits_k: int,
    delay_xor_words_by_direction: dict[int, np.ndarray],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    """Generate bitstream sequences for one direction after high-bit truncation."""

    if direction not in delay_xor_words_by_direction:
        raise ValueError("direction not available in delay_xor_words_by_direction.")
    if truncate_high_bits_k <= 0 or truncate_high_bits_k >= 16:
        raise ValueError("truncate_high_bits_k must be in [1, 15].")

    kept_low_bits = 16 - int(truncate_high_bits_k)
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    packed = pack_truncated_low_bits_to_bitstream(
        delay_xor_words_by_direction[direction],
        kept_low_bits=kept_low_bits,
        total_bit_length=total_bit_length,
    )
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    sequences = [packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]
    diagnostics_path = ensure_directory(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_path is not None:
        (diagnostics_path / f"direction_{direction}_k{truncate_high_bits_k}_{total_bit_length}bit.bin").write_bytes(packed)
    return sequences


def precompute_delay_xor_words_for_truncation(
    truncation_candidates: tuple[int, ...] | list[int] = DEFAULT_TRUNCATION_CANDIDATES,
    total_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH * DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> dict[int, np.ndarray]:
    """Precompute delay-XOR words long enough for every requested truncation width."""

    minimum_kept_bits = min(16 - int(k_value) for k_value in truncation_candidates)
    total_words = (int(total_bit_length) + minimum_kept_bits - 1) // minimum_kept_bits
    return generate_delay_xor_words(
        delay_by_direction=DEFAULT_BEST_DELAY_BY_DIRECTION,
        total_words=total_words,
        warmup_steps=warmup_steps,
    )


def evaluate_truncation_candidate(
    direction: int,
    truncate_high_bits_k: int,
    delay_xor_words_by_direction: dict[int, np.ndarray],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[TruncationCandidateResult, list[NISTAggregateResult]]:
    """Evaluate one truncation width for one direction."""

    sequences = generate_truncated_bitstream_sequences(
        direction=direction,
        truncate_high_bits_k=truncate_high_bits_k,
        delay_xor_words_by_direction=delay_xor_words_by_direction,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        diagnostics_dir=diagnostics_dir,
    )
    _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
        bitstream_sequences=sequences,
        direction=direction,
        sequence_bit_length=sequence_bit_length,
        bit_order="msb",
        max_workers=max_workers,
    )
    total_tests = len(CORE_NIST_TESTS)
    passed_tests = sum(1 for item in aggregates if item.passed)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = TruncationCandidateResult(
        direction=direction,
        delay_d=int(DEFAULT_BEST_DELAY_BY_DIRECTION[direction]),
        truncate_high_bits_k=int(truncate_high_bits_k),
        kept_low_bits=16 - int(truncate_high_bits_k),
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
    )
    return result, aggregates


def select_best_truncation_per_direction(candidate_rows: list[TruncationCandidateResult]) -> dict[int, BestTruncationDirection]:
    best: dict[int, BestTruncationDirection] = {}
    for row in candidate_rows:
        challenger = BestTruncationDirection(
            direction=row.direction,
            delay_d=row.delay_d,
            truncate_high_bits_k=row.truncate_high_bits_k,
            kept_low_bits=row.kept_low_bits,
            total_tests=row.total_tests,
            passed_tests=row.passed_tests,
            failed_tests=row.failed_tests,
            pass_ratio=row.pass_ratio,
            failed_test_names=row.failed_test_names,
            all_passed=row.all_passed,
        )
        current = best.get(row.direction)
        if current is None or _truncation_rank_key(challenger) > _truncation_rank_key(current):
            best[row.direction] = challenger
    return best
