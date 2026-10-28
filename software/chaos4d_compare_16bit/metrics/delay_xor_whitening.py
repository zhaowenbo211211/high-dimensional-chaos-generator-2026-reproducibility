# Maintainer: zhaowenbo
"""Delay-XOR whitening search for the representative enhanced 4D instance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from metrics.nist_randomness import (
    CORE_NIST_TESTS,
    DEFAULT_BIT_ORDER,
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
    NISTAggregateResult,
    generate_representative_enhanced_4d_words,
    pack_u16_words_to_bitstream,
    run_core_nist_sp800_22_suite_on_sequences,
)
from utils.io_utils import ensure_directory


DEFAULT_DELAY_CANDIDATES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)


@dataclass(frozen=True)
class DelayXORCandidateResult:
    direction: int
    delay_d: int
    passed_tests: int
    failed_tests: int
    all_passed: bool
    failed_test_names: str
    total_tests: int
    pass_ratio: float
    total_passed_all_directions: int
    all_directions_passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DelayXORBestDirection:
    direction: int
    delay_d: int
    passed_tests: int
    failed_tests: int
    all_passed: bool
    failed_test_names: str
    total_tests: int
    pass_ratio: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate_key(passed_tests: int, pass_ratio: float, delay_d: int) -> tuple[int, float, int]:
    return (int(passed_tests), float(pass_ratio), -int(delay_d))


def generate_delay_xor_words(
    delay_by_direction: dict[int, int],
    total_words: int,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> dict[int, np.ndarray]:
    """Return state-level delay-XOR words for each direction."""

    if total_words <= 0:
        raise ValueError("total_words must be positive.")
    max_delay = max(int(value) for value in delay_by_direction.values())
    state_words = generate_representative_enhanced_4d_words(
        word_count=total_words + max_delay,
        warmup_steps=warmup_steps,
    )
    outputs: dict[int, np.ndarray] = {}
    for direction in range(1, 5):
        delay_d = int(delay_by_direction[direction])
        outputs[direction] = np.bitwise_xor(
            state_words[direction - 1, delay_d : delay_d + total_words],
            state_words[direction - 1, :total_words],
        ).astype(np.uint16, copy=False)
    return outputs


def generate_delay_xor_bitstream_sequences(
    delay_d: int,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    bit_order: str = DEFAULT_BIT_ORDER,
    diagnostics_dir: str | Path | None = None,
) -> dict[int, list[bytes]]:
    """Generate per-direction bitstreams using state-level delay-XOR whitening."""

    if delay_d <= 0:
        raise ValueError("delay_d must be positive.")
    if sequence_bit_length <= 0:
        raise ValueError("sequence_bit_length must be positive.")
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive.")

    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    total_words = (total_bit_length + 15) // 16
    delay_xor_words = generate_delay_xor_words(
        delay_by_direction={1: delay_d, 2: delay_d, 3: delay_d, 4: delay_d},
        total_words=total_words,
        warmup_steps=warmup_steps,
    )

    outputs: dict[int, list[bytes]] = {}
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    diagnostics_path = ensure_directory(diagnostics_dir) if diagnostics_dir is not None else None

    for direction in range(4):
        whitened_words = delay_xor_words[direction + 1]
        packed = pack_u16_words_to_bitstream(whitened_words, bit_length=total_bit_length, bit_order=bit_order)
        outputs[direction + 1] = [
            packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence]
            for index in range(sequence_count)
        ]
        if diagnostics_path is not None:
            (diagnostics_path / f"delay_{delay_d}_direction_{direction + 1}_{total_bit_length}bit.bin").write_bytes(packed)
    return outputs


def evaluate_delay_xor_candidate(
    delay_d: int,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    bit_order: str = DEFAULT_BIT_ORDER,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[list[DelayXORCandidateResult], dict[int, list[NISTAggregateResult]]]:
    """Evaluate one delay candidate across the four directions."""

    bitstreams = generate_delay_xor_bitstream_sequences(
        delay_d=delay_d,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        warmup_steps=warmup_steps,
        bit_order=bit_order,
        diagnostics_dir=diagnostics_dir,
    )

    per_direction_aggregates: dict[int, list[NISTAggregateResult]] = {}
    rows: list[DelayXORCandidateResult] = []
    total_tests = len(CORE_NIST_TESTS)
    total_passed_all_directions = 0
    per_direction_stats: list[tuple[int, int, str, float, bool]] = []

    for direction in range(1, 5):
        _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
            bitstream_sequences=bitstreams[direction],
            direction=direction,
            sequence_bit_length=sequence_bit_length,
            bit_order=bit_order,
            max_workers=max_workers,
        )
        per_direction_aggregates[direction] = aggregates
        passed_tests = sum(1 for item in aggregates if item.passed)
        failed_names = [item.test_name for item in aggregates if not item.passed]
        all_passed = passed_tests == total_tests
        pass_ratio = passed_tests / float(total_tests)
        total_passed_all_directions += passed_tests
        per_direction_stats.append((direction, passed_tests, "; ".join(failed_names), pass_ratio, all_passed))

    all_directions_passed = total_passed_all_directions == 4 * total_tests
    for direction, passed_tests, failed_names, pass_ratio, all_passed in per_direction_stats:
        rows.append(
            DelayXORCandidateResult(
                direction=direction,
                delay_d=delay_d,
                passed_tests=passed_tests,
                failed_tests=total_tests - passed_tests,
                all_passed=all_passed,
                failed_test_names=failed_names,
                total_tests=total_tests,
                pass_ratio=pass_ratio,
                total_passed_all_directions=total_passed_all_directions,
                all_directions_passed=all_directions_passed,
            )
        )
    return rows, per_direction_aggregates


def select_best_per_direction(candidate_rows: list[DelayXORCandidateResult]) -> dict[int, DelayXORBestDirection]:
    """Select the best candidate for each direction using the configured ranking."""

    best: dict[int, DelayXORBestDirection] = {}
    for row in candidate_rows:
        current = best.get(row.direction)
        challenger = DelayXORBestDirection(
            direction=row.direction,
            delay_d=row.delay_d,
            passed_tests=row.passed_tests,
            failed_tests=row.failed_tests,
            all_passed=row.all_passed,
            failed_test_names=row.failed_test_names,
            total_tests=row.total_tests,
            pass_ratio=row.pass_ratio,
        )
        if current is None or _candidate_key(challenger.passed_tests, challenger.pass_ratio, challenger.delay_d) > _candidate_key(
            current.passed_tests,
            current.pass_ratio,
            current.delay_d,
        ):
            best[row.direction] = challenger
    return best
