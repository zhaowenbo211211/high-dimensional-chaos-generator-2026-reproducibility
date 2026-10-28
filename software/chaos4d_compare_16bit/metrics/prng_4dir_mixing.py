# Maintainer: zhaowenbo
"""System-level 4-direction PRNG post-processing for NIST evaluation."""

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


ConcatMode = Literal["plain_concat", "rotated_concat"]

DEFAULT_PRNG_R1 = 17
DEFAULT_PRNG_R2 = 43
DEFAULT_PRNG_BIT_ORDER = "msb"
DEFAULT_PRNG_CONCAT_MODES: tuple[ConcatMode, ...] = ("plain_concat", "rotated_concat")
DEFAULT_PRNG_NONLINEAR_OPTIONS: tuple[bool, ...] = (False, True)


@dataclass(frozen=True)
class PRNG4DirMixingSpec:
    method_name: str
    concat_mode: ConcatMode
    use_nonlinear_scrambler: bool
    linear_r1: int
    linear_r2: int
    mapping_description: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PRNG4DirMixingResult:
    method_name: str
    concat_mode: ConcatMode
    use_nonlinear_scrambler: bool
    linear_r1: int
    linear_r2: int
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


def _rotl16(words: np.ndarray, shift: int) -> np.ndarray:
    shift_value = int(shift) % 16
    words_u16 = words.astype(np.uint16, copy=False)
    if shift_value == 0:
        return words_u16
    return ((words_u16 << shift_value) | (words_u16 >> (16 - shift_value))).astype(np.uint16, copy=False)


def _rotl64(words: np.ndarray, shift: int) -> np.ndarray:
    shift_value = int(shift) % 64
    words_u64 = words.astype(np.uint64, copy=False)
    if shift_value == 0:
        return words_u64
    return ((words_u64 << np.uint64(shift_value)) | (words_u64 >> np.uint64(64 - shift_value))).astype(np.uint64, copy=False)


def build_prng_4dir_mixing_specs(
    concat_modes: tuple[ConcatMode, ...] | list[ConcatMode] = DEFAULT_PRNG_CONCAT_MODES,
    nonlinear_options: tuple[bool, ...] | list[bool] = DEFAULT_PRNG_NONLINEAR_OPTIONS,
    linear_r1: int = DEFAULT_PRNG_R1,
    linear_r2: int = DEFAULT_PRNG_R2,
) -> list[PRNG4DirMixingSpec]:
    specs: list[PRNG4DirMixingSpec] = []
    for concat_mode in concat_modes:
        for use_nonlinear in nonlinear_options:
            nonlinear_tag = "linear_mixing" if not use_nonlinear else "linear_mixing_nonlinear_scrambler"
            description = (
                f"{concat_mode} -> word-level lag mixing "
                f"L_t = raw_t XOR ROTL64(raw_(t-1), {linear_r1}) XOR ROTL64(raw_(t-2), {linear_r2})"
            )
            if use_nonlinear:
                description += " -> Y_t = L_t XOR (ROTL64(L_t,5) AND ROTL64(L_t,29))"
            specs.append(
                PRNG4DirMixingSpec(
                    method_name=f"{concat_mode}_{nonlinear_tag}",
                    concat_mode=concat_mode,
                    use_nonlinear_scrambler=bool(use_nonlinear),
                    linear_r1=int(linear_r1),
                    linear_r2=int(linear_r2),
                    mapping_description=description,
                )
            )
    return specs


def precompute_prng_4dir_raw_states(
    total_bit_length: int,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> np.ndarray:
    if total_bit_length <= 0:
        raise ValueError("total_bit_length must be positive.")
    block_count = (int(total_bit_length) + 63) // 64
    required_words = block_count + 2
    return generate_representative_enhanced_4d_words(word_count=required_words, warmup_steps=warmup_steps)


def assemble_prng_4dir_raw_words(
    raw_states: np.ndarray,
    concat_mode: ConcatMode,
) -> np.ndarray:
    if raw_states.shape[0] != 4:
        raise ValueError("raw_states must contain four directions.")
    x1 = raw_states[0].astype(np.uint16, copy=False)
    x2 = raw_states[1].astype(np.uint16, copy=False)
    x3 = raw_states[2].astype(np.uint16, copy=False)
    x4 = raw_states[3].astype(np.uint16, copy=False)
    if concat_mode == "rotated_concat":
        x2 = _rotl16(x2, 3)
        x3 = _rotl16(x3, 7)
        x4 = _rotl16(x4, 11)
    return (
        (x1.astype(np.uint64) << np.uint64(48))
        | (x2.astype(np.uint64) << np.uint64(32))
        | (x3.astype(np.uint64) << np.uint64(16))
        | x4.astype(np.uint64)
    ).astype(np.uint64, copy=False)


def apply_prng_4dir_postprocessing(
    raw_states: np.ndarray,
    spec: PRNG4DirMixingSpec,
    total_bit_length: int,
) -> bytes:
    block_count = (int(total_bit_length) + 63) // 64
    raw_words = assemble_prng_4dir_raw_words(raw_states=raw_states, concat_mode=spec.concat_mode)
    raw_words = raw_words[: block_count + 2]
    current = raw_words[2 : 2 + block_count]
    lag1 = _rotl64(raw_words[1 : 1 + block_count], spec.linear_r1)
    lag2 = _rotl64(raw_words[0:block_count], spec.linear_r2)
    linear_words = (current ^ lag1 ^ lag2).astype(np.uint64, copy=False)
    if spec.use_nonlinear_scrambler:
        nonlinear_term = _rotl64(linear_words, 5) & _rotl64(linear_words, 29)
        output_words = (linear_words ^ nonlinear_term).astype(np.uint64, copy=False)
    else:
        output_words = linear_words
    payload = output_words.byteswap().tobytes()
    total_bytes = (int(total_bit_length) + 7) // 8
    payload = payload[:total_bytes]
    if total_bit_length % 8:
        trimmed = bytearray(payload)
        trimmed[-1] &= 0xFF << (8 - (int(total_bit_length) % 8))
        payload = bytes(trimmed)
    return payload


def generate_prng_4dir_sequences(
    raw_states: np.ndarray,
    spec: PRNG4DirMixingSpec,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    payload = apply_prng_4dir_postprocessing(raw_states=raw_states, spec=spec, total_bit_length=total_bit_length)
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    if diagnostics_dir is not None:
        diagnostics_path = ensure_directory(diagnostics_dir)
        (diagnostics_path / f"{spec.method_name}_{total_bit_length}bit.bin").write_bytes(payload)
    return [payload[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]


def evaluate_prng_4dir_mixing(
    raw_states: np.ndarray,
    spec: PRNG4DirMixingSpec,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[PRNG4DirMixingResult, list[NISTAggregateResult]]:
    sequences = generate_prng_4dir_sequences(
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
        bit_order=DEFAULT_PRNG_BIT_ORDER,
        max_workers=max_workers,
    )
    total_tests = len(CORE_NIST_TESTS)
    passed_tests = sum(1 for item in aggregates if item.passed)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = PRNG4DirMixingResult(
        method_name=spec.method_name,
        concat_mode=spec.concat_mode,
        use_nonlinear_scrambler=spec.use_nonlinear_scrambler,
        linear_r1=spec.linear_r1,
        linear_r2=spec.linear_r2,
        mapping_description=spec.mapping_description,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        all_passed=passed_tests == total_tests,
    )
    return result, aggregates


def _result_rank_key(result: PRNG4DirMixingResult) -> tuple[int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.use_nonlinear_scrambler),
        0 if result.concat_mode == "plain_concat" else -1,
    )


def select_best_prng_4dir_mixing(results: list[PRNG4DirMixingResult]) -> PRNG4DirMixingResult:
    if not results:
        raise ValueError("results must not be empty.")
    return max(results, key=_result_rank_key)
