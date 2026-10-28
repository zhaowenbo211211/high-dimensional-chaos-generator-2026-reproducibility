# Maintainer: zhaowenbo
"""Direction-4 lightweight GF(2) linear mixing for NIST experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

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


Direction4Variant = Literal["staggered_sparse", "interleaved_sparse"]

DEFAULT_LINEAR_MIXING_WINDOWS: tuple[int, ...] = (2, 3, 4)
DEFAULT_LINEAR_MIXING_OUTPUT_BITS: tuple[int, ...] = (16, 12, 8)
DEFAULT_LINEAR_MIXING_FANINS: tuple[int, ...] = (2, 3, 4)
DEFAULT_LINEAR_MIXING_VARIANTS: tuple[Direction4Variant, ...] = ("staggered_sparse", "interleaved_sparse")
DEFAULT_DIRECTION4_BIT_ORDER = "msb"


@dataclass(frozen=True)
class Direction4LinearMixingSpec:
    method_name: str
    variant: Direction4Variant
    window_W: int
    input_bits: int
    output_bits: int
    fanin: int
    mapping_description: str
    taps: tuple[tuple[tuple[int, int], ...], ...]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("taps", None)
        return data


@dataclass(frozen=True)
class Direction4LinearMixingResult:
    method_name: str
    window_W: int
    input_bits: int
    output_bits: int
    fanin: int
    mapping_description: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    all_passed: bool
    is_best: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _staggered_source(output_index: int, source_index: int, window_w: int) -> tuple[int, int]:
    lag = (output_index + source_index) % window_w
    bit = (output_index * (source_index + 1) + 3 * source_index + 5 * lag + 1) % 16
    return lag, bit


def _interleaved_source(output_index: int, source_index: int, window_w: int) -> tuple[int, int]:
    lag = (2 * source_index + output_index) % window_w
    bit = ((15 - output_index) + 5 * source_index + 3 * lag + output_index // 2) % 16
    return lag, bit


def _build_unique_taps(
    variant: Direction4Variant,
    output_bits: int,
    fanin: int,
    window_w: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    taps: list[tuple[tuple[int, int], ...]] = []
    for output_index in range(output_bits):
        used: set[tuple[int, int]] = set()
        output_taps: list[tuple[int, int]] = []
        attempt = 0
        while len(output_taps) < fanin:
            source_index = len(output_taps) + attempt
            if variant == "staggered_sparse":
                lag, bit = _staggered_source(output_index, source_index, window_w)
            else:
                lag, bit = _interleaved_source(output_index, source_index, window_w)
            while (lag, bit) in used:
                bit = (bit + 1) % 16
                if (lag, bit) not in used:
                    break
                lag = (lag + 1) % window_w
            used.add((lag, bit))
            output_taps.append((lag, bit))
            attempt += 1
        taps.append(tuple(output_taps))
    return tuple(taps)


def build_direction4_linear_mixing_spec(
    variant: Direction4Variant,
    window_w: int,
    output_bits: int,
    fanin: int,
) -> Direction4LinearMixingSpec:
    if window_w <= 0:
        raise ValueError("window_w must be positive.")
    if output_bits <= 0 or output_bits > 16:
        raise ValueError("output_bits must be in [1, 16].")
    if fanin <= 0:
        raise ValueError("fanin must be positive.")
    taps = _build_unique_taps(variant=variant, output_bits=output_bits, fanin=fanin, window_w=window_w)
    description = (
        f"{variant}: rule-based sparse GF(2) XOR mixing over W={window_w} states, "
        f"output width={output_bits}, fanin={fanin}"
    )
    return Direction4LinearMixingSpec(
        method_name=f"{variant}_W{window_w}_m{output_bits}_f{fanin}",
        variant=variant,
        window_W=window_w,
        input_bits=16 * window_w,
        output_bits=output_bits,
        fanin=fanin,
        mapping_description=description,
        taps=taps,
    )


def enumerate_direction4_linear_mixing_specs(
    windows: tuple[int, ...] | list[int] = DEFAULT_LINEAR_MIXING_WINDOWS,
    output_bits_candidates: tuple[int, ...] | list[int] = DEFAULT_LINEAR_MIXING_OUTPUT_BITS,
    fanins: tuple[int, ...] | list[int] = DEFAULT_LINEAR_MIXING_FANINS,
    variants: tuple[Direction4Variant, ...] | list[Direction4Variant] = DEFAULT_LINEAR_MIXING_VARIANTS,
) -> list[Direction4LinearMixingSpec]:
    specs: list[Direction4LinearMixingSpec] = []
    for variant in variants:
        for window_w in windows:
            for output_bits in output_bits_candidates:
                for fanin in fanins:
                    specs.append(
                        build_direction4_linear_mixing_spec(
                            variant=variant,
                            window_w=int(window_w),
                            output_bits=int(output_bits),
                            fanin=int(fanin),
                        )
                    )
    return specs


def precompute_direction4_raw_words(
    specs: list[Direction4LinearMixingSpec],
    total_bit_length: int,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> np.ndarray:
    if not specs:
        raise ValueError("specs must not be empty.")
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")
    minimum_output_bits = min(spec.output_bits for spec in specs)
    maximum_window_w = max(spec.window_W for spec in specs)
    required_words = (int(total_bit_length) + minimum_output_bits - 1) // minimum_output_bits + maximum_window_w - 1
    raw_states = generate_representative_enhanced_4d_words(word_count=required_words, warmup_steps=warmup_steps)
    return raw_states[3].astype(np.uint16, copy=False)


def apply_direction4_linear_mixing(
    raw_words: np.ndarray,
    spec: Direction4LinearMixingSpec,
    total_bit_length: int,
) -> bytes:
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")
    block_count = (int(total_bit_length) + spec.output_bits - 1) // spec.output_bits
    required_words = block_count + spec.window_W - 1
    if len(raw_words) < required_words:
        raise ValueError("raw_words are shorter than required for the requested output.")

    mixed_words = np.zeros(block_count, dtype=np.uint16)
    for output_index, output_taps in enumerate(spec.taps):
        bit_values = np.zeros(block_count, dtype=np.uint8)
        for lag, bit in output_taps:
            start = spec.window_W - 1 - lag
            source_slice = raw_words[start : start + block_count]
            bit_values ^= ((source_slice >> bit) & 1).astype(np.uint8, copy=False)
        mixed_words |= bit_values.astype(np.uint16) << np.uint16(spec.output_bits - 1 - output_index)

    shifts = np.arange(spec.output_bits - 1, -1, -1, dtype=np.uint16)
    bits = ((mixed_words[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(bits[:total_bit_length], bitorder="big")
    return packed.tobytes()


def generate_direction4_linear_mixing_sequences(
    raw_words: np.ndarray,
    spec: Direction4LinearMixingSpec,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    payload = apply_direction4_linear_mixing(raw_words=raw_words, spec=spec, total_bit_length=total_bit_length)
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    if diagnostics_dir is not None:
        diagnostics_path = ensure_directory(diagnostics_dir)
        (diagnostics_path / f"{spec.method_name}_{total_bit_length}bit.bin").write_bytes(payload)
    return [payload[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]


def evaluate_direction4_linear_mixing(
    raw_words: np.ndarray,
    spec: Direction4LinearMixingSpec,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[Direction4LinearMixingResult, list[NISTAggregateResult]]:
    sequences = generate_direction4_linear_mixing_sequences(
        raw_words=raw_words,
        spec=spec,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        diagnostics_dir=diagnostics_dir,
    )
    _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
        bitstream_sequences=sequences,
        direction=4,
        sequence_bit_length=sequence_bit_length,
        bit_order=DEFAULT_DIRECTION4_BIT_ORDER,
        max_workers=max_workers,
    )
    total_tests = len(CORE_NIST_TESTS)
    passed_tests = sum(1 for item in aggregates if item.passed)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = Direction4LinearMixingResult(
        method_name=spec.method_name,
        window_W=spec.window_W,
        input_bits=spec.input_bits,
        output_bits=spec.output_bits,
        fanin=spec.fanin,
        mapping_description=spec.mapping_description,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
    )
    return result, aggregates


def _result_rank_key(result: Direction4LinearMixingResult) -> tuple[int, int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.window_W),
        -int(result.fanin),
        int(result.output_bits),
    )


def select_best_direction4_linear_mixing(
    results: list[Direction4LinearMixingResult],
) -> Direction4LinearMixingResult:
    if not results:
        raise ValueError("results must not be empty.")
    return max(results, key=_result_rank_key)
