# Maintainer: zhaowenbo
"""Two-stage mid-bit truncation plus fixed-delay-XOR evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from metrics.nist_randomness import (
    CORE_NIST_TESTS,
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    NISTAggregateResult,
    run_core_nist_sp800_22_suite_on_sequences,
)
from metrics.two_sided_truncation import extract_mid_bits, pack_midbit_words_to_bitstream, save_midbit_payload


DEFAULT_FIXED_DELAY_D = 4


@dataclass(frozen=True)
class StageAMidbitResult:
    direction: int
    truncate_high_bits_h: int
    truncate_low_bits_l: int
    kept_mid_bits: int
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
class StageBMidbitDelayResult:
    direction: int
    truncate_high_bits_h: int
    truncate_low_bits_l: int
    kept_mid_bits: int
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


def _stage_a_rank_key(result: StageAMidbitResult) -> tuple[int, int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        int(result.kept_mid_bits),
        -(int(result.truncate_high_bits_h) + int(result.truncate_low_bits_l)),
        -int(result.truncate_high_bits_h),
    )


def _stage_b_rank_key(result: StageBMidbitDelayResult) -> tuple[int, int, int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.delay_d),
        int(result.kept_mid_bits),
        -(int(result.truncate_high_bits_h) + int(result.truncate_low_bits_l)),
        -int(result.truncate_high_bits_h),
    )


def generate_stage_a_midbit_sequences(
    direction: int,
    truncate_high_bits_h: int,
    truncate_low_bits_l: int,
    raw_words_by_direction: dict[int, object],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    if direction not in raw_words_by_direction:
        raise ValueError("direction not available in raw_words_by_direction.")
    kept_mid_bits = 16 - int(truncate_high_bits_h) - int(truncate_low_bits_l)
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    compact_mid_words = extract_mid_bits(raw_words_by_direction[direction], truncate_high_bits_h, truncate_low_bits_l)
    packed = pack_midbit_words_to_bitstream(compact_mid_words, kept_mid_bits=kept_mid_bits, total_bit_length=total_bit_length)
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    save_midbit_payload(
        packed,
        output_name=f"stageA_direction_{direction}_h{truncate_high_bits_h}_l{truncate_low_bits_l}_{total_bit_length}bit.bin",
        diagnostics_dir=diagnostics_dir,
    )
    return [packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]


def evaluate_stage_a_midbit_candidate(
    direction: int,
    truncate_high_bits_h: int,
    truncate_low_bits_l: int,
    raw_words_by_direction: dict[int, object],
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[StageAMidbitResult, list[NISTAggregateResult]]:
    sequences = generate_stage_a_midbit_sequences(
        direction=direction,
        truncate_high_bits_h=truncate_high_bits_h,
        truncate_low_bits_l=truncate_low_bits_l,
        raw_words_by_direction=raw_words_by_direction,
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
    result = StageAMidbitResult(
        direction=direction,
        truncate_high_bits_h=int(truncate_high_bits_h),
        truncate_low_bits_l=int(truncate_low_bits_l),
        kept_mid_bits=16 - int(truncate_high_bits_h) - int(truncate_low_bits_l),
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
    )
    return result, aggregates


def select_best_stage_a_midbit(results: list[StageAMidbitResult]) -> dict[int, StageAMidbitResult]:
    best: dict[int, StageAMidbitResult] = {}
    for result in results:
        current = best.get(result.direction)
        if current is None or _stage_a_rank_key(result) > _stage_a_rank_key(current):
            best[result.direction] = result
    return best


def generate_stage_b_midbit_delay_sequences(
    direction: int,
    truncate_high_bits_h: int,
    truncate_low_bits_l: int,
    raw_words_by_direction: dict[int, object],
    delay_d: int = DEFAULT_FIXED_DELAY_D,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    if direction not in raw_words_by_direction:
        raise ValueError("direction not available in raw_words_by_direction.")
    if delay_d <= 0:
        raise ValueError("delay_d must be positive.")
    kept_mid_bits = 16 - int(truncate_high_bits_h) - int(truncate_low_bits_l)
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    required_words = (int(total_bit_length) + kept_mid_bits - 1) // kept_mid_bits + int(delay_d)
    raw_words = raw_words_by_direction[direction][:required_words]
    compact_mid_words = extract_mid_bits(raw_words, truncate_high_bits_h, truncate_low_bits_l)
    whitened_words = (compact_mid_words[delay_d:] ^ compact_mid_words[:-delay_d]).astype("uint16", copy=False)
    packed = pack_midbit_words_to_bitstream(whitened_words, kept_mid_bits=kept_mid_bits, total_bit_length=total_bit_length)
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    save_midbit_payload(
        packed,
        output_name=f"stageB_direction_{direction}_h{truncate_high_bits_h}_l{truncate_low_bits_l}_d{delay_d}_{total_bit_length}bit.bin",
        diagnostics_dir=diagnostics_dir,
    )
    return [packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]


def evaluate_stage_b_midbit_delay(
    direction: int,
    truncate_high_bits_h: int,
    truncate_low_bits_l: int,
    raw_words_by_direction: dict[int, object],
    delay_d: int = DEFAULT_FIXED_DELAY_D,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[StageBMidbitDelayResult, list[NISTAggregateResult]]:
    sequences = generate_stage_b_midbit_delay_sequences(
        direction=direction,
        truncate_high_bits_h=truncate_high_bits_h,
        truncate_low_bits_l=truncate_low_bits_l,
        raw_words_by_direction=raw_words_by_direction,
        delay_d=delay_d,
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
    result = StageBMidbitDelayResult(
        direction=direction,
        truncate_high_bits_h=int(truncate_high_bits_h),
        truncate_low_bits_l=int(truncate_low_bits_l),
        kept_mid_bits=16 - int(truncate_high_bits_h) - int(truncate_low_bits_l),
        delay_d=int(delay_d),
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
    )
    return result, aggregates


def select_best_stage_b_midbit(results: list[StageBMidbitDelayResult]) -> dict[int, StageBMidbitDelayResult]:
    best: dict[int, StageBMidbitDelayResult] = {}
    for result in results:
        current = best.get(result.direction)
        if current is None or _stage_b_rank_key(result) > _stage_b_rank_key(current):
            best[result.direction] = result
    return best
