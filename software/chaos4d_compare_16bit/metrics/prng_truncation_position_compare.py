# Maintainer: zhaowenbo
"""System-level PRNG truncation-position comparison for NIST evaluation."""

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


SCHEME1_ROTATION_CANDIDATES: tuple[tuple[int, int], ...] = ((11, 29), (13, 31), (17, 37), (19, 41))
SCHEME2_ROTATION_CANDIDATES: tuple[tuple[int, int], ...] = ((11, 29), (13, 31), (17, 41), (19, 43))
SCHEME3_ROTATION: tuple[int, int] = (17, 43)
DEFAULT_TRUNCATION_COMPARE_BIT_ORDER = "msb"


@dataclass(frozen=True)
class PRNGTruncationSchemeSpec:
    scheme_id: str
    input_word_width: int
    output_word_width: int
    rotation_r1: int
    rotation_r2: int
    implementation_notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PRNGTruncationSchemeResult:
    scheme_id: str
    input_word_width: int
    output_word_width: int
    rotation_r1: int
    rotation_r2: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    all_passed: bool
    is_best: bool
    implementation_notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _width_mask(width: int) -> np.uint64:
    if width <= 0 or width > 64:
        raise ValueError("width must be in [1, 64].")
    if width == 64:
        return np.uint64(0xFFFFFFFFFFFFFFFF)
    return (np.uint64(1) << np.uint64(width)) - np.uint64(1)


def _rotl_width(words: np.ndarray, width: int, shift: int) -> np.ndarray:
    shift_value = int(shift) % int(width)
    mask = _width_mask(width)
    words_u64 = words.astype(np.uint64, copy=False) & mask
    if shift_value == 0:
        return words_u64
    return (((words_u64 << np.uint64(shift_value)) | (words_u64 >> np.uint64(width - shift_value))) & mask).astype(np.uint64, copy=False)


def _pack_words_to_bitstream(words: np.ndarray, width: int, total_bit_length: int) -> bytes:
    required_words = (int(total_bit_length) + width - 1) // width
    selected = words[:required_words].astype(np.uint64, copy=False)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    bits = ((selected[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(bits[:total_bit_length], bitorder="big")
    return packed.tobytes()


def _truncate_u16_mid14(words: np.ndarray) -> np.ndarray:
    return ((words.astype(np.uint16, copy=False) >> np.uint16(1)) & np.uint16((1 << 14) - 1)).astype(np.uint16, copy=False)


def _truncate_u64_mid62(words: np.ndarray) -> np.ndarray:
    return ((words.astype(np.uint64, copy=False) >> np.uint64(1)) & ((np.uint64(1) << np.uint64(62)) - np.uint64(1))).astype(np.uint64, copy=False)


def _assemble_plain64(raw_states: np.ndarray) -> np.ndarray:
    x1 = raw_states[0].astype(np.uint16, copy=False)
    x2 = raw_states[1].astype(np.uint16, copy=False)
    x3 = raw_states[2].astype(np.uint16, copy=False)
    x4 = raw_states[3].astype(np.uint16, copy=False)
    return (
        (x1.astype(np.uint64) << np.uint64(48))
        | (x2.astype(np.uint64) << np.uint64(32))
        | (x3.astype(np.uint64) << np.uint64(16))
        | x4.astype(np.uint64)
    ).astype(np.uint64, copy=False)


def _assemble_scheme1_word56(raw_states: np.ndarray) -> np.ndarray:
    x1 = _truncate_u16_mid14(raw_states[0])
    x2 = _truncate_u16_mid14(raw_states[1])
    x3 = _truncate_u16_mid14(raw_states[2])
    x4 = _truncate_u16_mid14(raw_states[3])
    return (
        (x1.astype(np.uint64) << np.uint64(42))
        | (x2.astype(np.uint64) << np.uint64(28))
        | (x3.astype(np.uint64) << np.uint64(14))
        | x4.astype(np.uint64)
    ).astype(np.uint64, copy=False)


def _apply_stage1(words: np.ndarray, width: int, r1: int, r2: int, block_count: int) -> np.ndarray:
    mask = _width_mask(width)
    words = (words[: block_count + 2].astype(np.uint64, copy=False) & mask).astype(np.uint64, copy=False)
    current = words[2 : 2 + block_count]
    lag1 = _rotl_width(words[1 : 1 + block_count], width=width, shift=r1)
    lag2 = _rotl_width(words[0:block_count], width=width, shift=r2)
    return (current ^ lag1 ^ lag2) & mask


def precompute_prng_truncation_compare_raw_states(
    total_bit_length: int,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> np.ndarray:
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")
    max_required_blocks = (int(total_bit_length) + 55) // 56
    required_words = max_required_blocks + 2
    return generate_representative_enhanced_4d_words(word_count=required_words, warmup_steps=warmup_steps)


def generate_scheme_output_payload(
    raw_states: np.ndarray,
    spec: PRNGTruncationSchemeSpec,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    block_count = (int(total_bit_length) + spec.output_word_width - 1) // spec.output_word_width

    if spec.scheme_id == "scheme1_per_direction_before_stage0":
        stage0_words = _assemble_scheme1_word56(raw_states)
        final_words = _apply_stage1(stage0_words, width=56, r1=spec.rotation_r1, r2=spec.rotation_r2, block_count=block_count)
    elif spec.scheme_id == "scheme2_after_stage0_before_stage1":
        stage0_words = _assemble_plain64(raw_states)
        truncated = _truncate_u64_mid62(stage0_words)
        final_words = _apply_stage1(truncated, width=62, r1=spec.rotation_r1, r2=spec.rotation_r2, block_count=block_count)
    elif spec.scheme_id == "scheme3_after_stage1":
        stage0_words = _assemble_plain64(raw_states)
        mixed64 = _apply_stage1(stage0_words, width=64, r1=spec.rotation_r1, r2=spec.rotation_r2, block_count=block_count)
        final_words = _truncate_u64_mid62(mixed64)
    else:
        raise ValueError(f"unsupported scheme_id: {spec.scheme_id}")

    payload = _pack_words_to_bitstream(final_words, width=spec.output_word_width, total_bit_length=total_bit_length)
    if diagnostics_dir is not None:
        diagnostics_path = ensure_directory(diagnostics_dir)
        (diagnostics_path / f"{spec.scheme_id}_r1_{spec.rotation_r1}_r2_{spec.rotation_r2}_{total_bit_length}bit.bin").write_bytes(payload)
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    return [payload[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]


def evaluate_scheme_spec(
    raw_states: np.ndarray,
    spec: PRNGTruncationSchemeSpec,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[PRNGTruncationSchemeResult, list[NISTAggregateResult]]:
    sequences = generate_scheme_output_payload(
        raw_states=raw_states,
        spec=spec,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        diagnostics_dir=diagnostics_dir,
    )
    _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
        bitstream_sequences=sequences,
        direction=0,
        sequence_bit_length=sequence_bit_length,
        bit_order=DEFAULT_TRUNCATION_COMPARE_BIT_ORDER,
        max_workers=max_workers,
    )
    total_tests = len(CORE_NIST_TESTS)
    passed_tests = sum(1 for item in aggregates if item.passed)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = PRNGTruncationSchemeResult(
        scheme_id=spec.scheme_id,
        input_word_width=spec.input_word_width,
        output_word_width=spec.output_word_width,
        rotation_r1=spec.rotation_r1,
        rotation_r2=spec.rotation_r2,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
        is_best=False,
        implementation_notes=spec.implementation_notes,
    )
    return result, aggregates


def _candidate_rank_key(result: PRNGTruncationSchemeResult) -> tuple[int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -(int(result.rotation_r1) + int(result.rotation_r2)),
        -int(result.rotation_r1),
    )


def select_best_candidate_for_scheme(results: list[PRNGTruncationSchemeResult]) -> PRNGTruncationSchemeResult:
    if not results:
        raise ValueError("results must not be empty.")
    return max(results, key=_candidate_rank_key)


def _scheme_priority(result: PRNGTruncationSchemeResult) -> tuple[int, int]:
    if result.scheme_id == "scheme3_after_stage1":
        return (2, int(result.output_word_width))
    if result.scheme_id == "scheme2_after_stage0_before_stage1":
        return (1, int(result.output_word_width))
    return (0, int(result.output_word_width))


def _overall_rank_key(result: PRNGTruncationSchemeResult) -> tuple[int, int, tuple[int, int]]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        _scheme_priority(result),
    )


def select_best_overall_scheme(results: list[PRNGTruncationSchemeResult]) -> PRNGTruncationSchemeResult:
    if not results:
        raise ValueError("results must not be empty.")
    return max(results, key=_overall_rank_key)
