# Maintainer: zhaowenbo
"""Two-stage post-processing search: raw truncation, then truncation plus delay-XOR."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from metrics.nist_randomness import (
    CORE_NIST_TESTS,
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
    NISTAggregateResult,
    generate_representative_enhanced_4d_words,
    run_core_nist_sp800_22_suite_on_sequences,
)
from utils.io_utils import ensure_directory


DEFAULT_STAGE_A_TRUNCATION_CANDIDATES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_STAGE_B_DELAY_CANDIDATES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48)
DEFAULT_STAGE_BIT_ORDER = "msb"


@dataclass(frozen=True)
class StageATruncationResult:
    direction: int
    truncate_high_bits_k: int
    kept_low_bits: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    all_passed: bool
    is_best: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageBDelayResult:
    direction: int
    truncate_high_bits_k: int
    kept_low_bits: int
    delay_d: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    all_passed: bool
    is_best: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _stage_a_rank_key(result: StageATruncationResult) -> tuple[int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.truncate_high_bits_k),
        int(result.kept_low_bits),
    )


def _stage_b_rank_key(result: StageBDelayResult) -> tuple[int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.delay_d),
        int(result.kept_low_bits),
    )


def _pack_words_with_kept_low_bits(words: np.ndarray, kept_low_bits: int, total_bit_length: int) -> bytes:
    if kept_low_bits <= 0 or kept_low_bits > 16:
        raise ValueError("kept_low_bits must be in [1, 16].")
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")
    required_words = (int(total_bit_length) + kept_low_bits - 1) // kept_low_bits
    selected_words = words[:required_words].astype(np.uint16, copy=False)
    mask = (1 << kept_low_bits) - 1 if kept_low_bits < 16 else 0xFFFF
    truncated = np.bitwise_and(selected_words, mask).astype(np.uint16, copy=False)
    shifts = np.arange(kept_low_bits - 1, -1, -1, dtype=np.uint16)
    bits = ((truncated[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(bits[:total_bit_length], bitorder="big")
    return packed.tobytes()


def precompute_raw_words_for_two_stage(
    truncation_candidates: list[int] | tuple[int, ...] = DEFAULT_STAGE_A_TRUNCATION_CANDIDATES,
    delay_candidates: list[int] | tuple[int, ...] = DEFAULT_STAGE_B_DELAY_CANDIDATES,
    total_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH * DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> dict[int, np.ndarray]:
    """Precompute raw post-warmup state words long enough for stages A and B."""

    minimum_kept_bits = min(16 - int(k_value) for k_value in truncation_candidates)
    max_delay = max(int(value) for value in delay_candidates) if delay_candidates else 0
    total_words = (int(total_bit_length) + minimum_kept_bits - 1) // minimum_kept_bits + max_delay
    raw_words = generate_representative_enhanced_4d_words(word_count=total_words, warmup_steps=warmup_steps)
    return {direction + 1: raw_words[direction].astype(np.uint16, copy=False) for direction in range(4)}


def generate_stage_a_bitstream_sequences(
    direction: int,
    truncate_high_bits_k: int,
    raw_words_by_direction: dict[int, np.ndarray],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    if direction not in raw_words_by_direction:
        raise ValueError("direction not available in raw_words_by_direction.")
    kept_low_bits = 16 - int(truncate_high_bits_k)
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    packed = _pack_words_with_kept_low_bits(
        raw_words_by_direction[direction],
        kept_low_bits=kept_low_bits,
        total_bit_length=total_bit_length,
    )
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    sequences = [packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]
    diagnostics_path = ensure_directory(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_path is not None:
        (diagnostics_path / f"stageA_direction_{direction}_k{truncate_high_bits_k}_{total_bit_length}bit.bin").write_bytes(packed)
    return sequences


def evaluate_stage_a_candidate(
    direction: int,
    truncate_high_bits_k: int,
    raw_words_by_direction: dict[int, np.ndarray],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[StageATruncationResult, list[NISTAggregateResult]]:
    sequences = generate_stage_a_bitstream_sequences(
        direction=direction,
        truncate_high_bits_k=truncate_high_bits_k,
        raw_words_by_direction=raw_words_by_direction,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        diagnostics_dir=diagnostics_dir,
    )
    _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
        bitstream_sequences=sequences,
        direction=direction,
        sequence_bit_length=sequence_bit_length,
        bit_order=DEFAULT_STAGE_BIT_ORDER,
        max_workers=max_workers,
    )
    total_tests = len(CORE_NIST_TESTS)
    passed_tests = sum(1 for item in aggregates if item.passed)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = StageATruncationResult(
        direction=direction,
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


def select_best_stage_a(results: list[StageATruncationResult]) -> dict[int, StageATruncationResult]:
    best: dict[int, StageATruncationResult] = {}
    for result in results:
        current = best.get(result.direction)
        if current is None or _stage_a_rank_key(result) > _stage_a_rank_key(current):
            best[result.direction] = result
    return best


def generate_stage_b_bitstream_sequences(
    direction: int,
    truncate_high_bits_k: int,
    delay_d: int,
    raw_words_by_direction: dict[int, np.ndarray],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    if direction not in raw_words_by_direction:
        raise ValueError("direction not available in raw_words_by_direction.")
    if delay_d <= 0:
        raise ValueError("delay_d must be positive.")
    kept_low_bits = 16 - int(truncate_high_bits_k)
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    required_words = (int(total_bit_length) + kept_low_bits - 1) // kept_low_bits + int(delay_d)
    raw_words = raw_words_by_direction[direction][:required_words]
    mask = (1 << kept_low_bits) - 1 if kept_low_bits < 16 else 0xFFFF
    truncated_words = np.bitwise_and(raw_words, mask).astype(np.uint16, copy=False)
    whitened_words = np.bitwise_xor(truncated_words[delay_d:], truncated_words[:-delay_d]).astype(np.uint16, copy=False)
    packed = _pack_words_with_kept_low_bits(
        whitened_words,
        kept_low_bits=kept_low_bits,
        total_bit_length=total_bit_length,
    )
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    sequences = [packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]
    diagnostics_path = ensure_directory(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_path is not None:
        (diagnostics_path / f"stageB_direction_{direction}_k{truncate_high_bits_k}_d{delay_d}_{total_bit_length}bit.bin").write_bytes(packed)
    return sequences


def evaluate_stage_b_candidate(
    direction: int,
    truncate_high_bits_k: int,
    delay_d: int,
    raw_words_by_direction: dict[int, np.ndarray],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[StageBDelayResult, list[NISTAggregateResult]]:
    sequences = generate_stage_b_bitstream_sequences(
        direction=direction,
        truncate_high_bits_k=truncate_high_bits_k,
        delay_d=delay_d,
        raw_words_by_direction=raw_words_by_direction,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        diagnostics_dir=diagnostics_dir,
    )
    _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
        bitstream_sequences=sequences,
        direction=direction,
        sequence_bit_length=sequence_bit_length,
        bit_order=DEFAULT_STAGE_BIT_ORDER,
        max_workers=max_workers,
    )
    total_tests = len(CORE_NIST_TESTS)
    passed_tests = sum(1 for item in aggregates if item.passed)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = StageBDelayResult(
        direction=direction,
        truncate_high_bits_k=int(truncate_high_bits_k),
        kept_low_bits=16 - int(truncate_high_bits_k),
        delay_d=int(delay_d),
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
    )
    return result, aggregates


def select_best_stage_b(results: list[StageBDelayResult]) -> dict[int, StageBDelayResult]:
    best: dict[int, StageBDelayResult] = {}
    for result in results:
        current = best.get(result.direction)
        if current is None or _stage_b_rank_key(result) > _stage_b_rank_key(current):
            best[result.direction] = result
    return best
