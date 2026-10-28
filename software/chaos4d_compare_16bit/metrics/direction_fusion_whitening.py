# Maintainer: zhaowenbo
"""Direction-fusion whitening built on top of delay-XOR outputs."""

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


DEFAULT_BEST_DELAY_BY_DIRECTION: dict[int, int] = {1: 1, 2: 2, 3: 1, 4: 1}
DEFAULT_FUSION_SCHEMES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("y14", (1, 4)),
    ("y24", (2, 4)),
    ("y34", (3, 4)),
    ("y1234", (1, 2, 3, 4)),
    ("y12", (1, 2)),
    ("y23", (2, 3)),
    ("y34_alt", (3, 4)),
)


@dataclass(frozen=True)
class FusionSchemeDefinition:
    fusion_name: str
    source_directions: tuple[int, ...]

    @property
    def source_formula(self) -> str:
        return " XOR ".join(f"z{direction}" for direction in self.source_directions)

    def as_dict(self) -> dict[str, object]:
        return {
            "fusion_name": self.fusion_name,
            "source_formula": self.source_formula,
            "source_directions": ",".join(str(direction) for direction in self.source_directions),
        }


@dataclass(frozen=True)
class FusionSchemeResult:
    fusion_name: str
    source_formula: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    pass_ratio: float
    failed_test_names: str
    xor_inputs: int
    includes_direction_4: bool
    is_best: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _fusion_rank_key(result: FusionSchemeResult) -> tuple[int, int, int, int]:
    return (
        int(result.passed_tests),
        -int(result.failed_tests),
        -int(result.xor_inputs),
        1 if result.includes_direction_4 else 0,
    )


def _compute_delay_xor_words(
    delay_by_direction: dict[int, int],
    total_words: int,
    warmup_steps: int,
) -> dict[int, np.ndarray]:
    max_delay = max(delay_by_direction.values())
    base_words = generate_representative_enhanced_4d_words(
        word_count=total_words + max_delay,
        warmup_steps=warmup_steps,
    )
    outputs: dict[int, np.ndarray] = {}
    for direction in range(1, 5):
        delay_d = int(delay_by_direction[direction])
        outputs[direction] = np.bitwise_xor(
            base_words[direction - 1, delay_d : delay_d + total_words],
            base_words[direction - 1, :total_words],
        ).astype(np.uint16, copy=False)
    return outputs


def generate_fusion_bitstream_sequences(
    fusion: FusionSchemeDefinition,
    delay_by_direction: dict[int, int] | None = None,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    bit_order: str = DEFAULT_BIT_ORDER,
    diagnostics_dir: str | Path | None = None,
) -> list[bytes]:
    if sequence_bit_length <= 0:
        raise ValueError("sequence_bit_length must be positive.")
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive.")

    selected_delays = dict(DEFAULT_BEST_DELAY_BY_DIRECTION if delay_by_direction is None else delay_by_direction)
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    total_words = (total_bit_length + 15) // 16
    delay_xor_words = _compute_delay_xor_words(
        delay_by_direction=selected_delays,
        total_words=total_words,
        warmup_steps=warmup_steps,
    )

    fused_words = np.zeros(total_words, dtype=np.uint16)
    for direction in fusion.source_directions:
        fused_words = np.bitwise_xor(fused_words, delay_xor_words[direction]).astype(np.uint16, copy=False)

    packed = pack_u16_words_to_bitstream(fused_words, bit_length=total_bit_length, bit_order=bit_order)
    bytes_per_sequence = (sequence_bit_length + 7) // 8
    sequences = [packed[index * bytes_per_sequence : (index + 1) * bytes_per_sequence] for index in range(sequence_count)]

    diagnostics_path = ensure_directory(diagnostics_dir) if diagnostics_dir is not None else None
    if diagnostics_path is not None:
        (diagnostics_path / f"{fusion.fusion_name}_{total_bit_length}bit.bin").write_bytes(packed)
    return sequences


def evaluate_fusion_scheme(
    fusion: FusionSchemeDefinition,
    delay_by_direction: dict[int, int] | None = None,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    bit_order: str = DEFAULT_BIT_ORDER,
    max_workers: int | None = None,
    diagnostics_dir: str | Path | None = None,
) -> tuple[FusionSchemeResult, list[NISTAggregateResult]]:
    sequences = generate_fusion_bitstream_sequences(
        fusion=fusion,
        delay_by_direction=delay_by_direction,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        warmup_steps=warmup_steps,
        bit_order=bit_order,
        diagnostics_dir=diagnostics_dir,
    )
    _, aggregates = run_core_nist_sp800_22_suite_on_sequences(
        bitstream_sequences=sequences,
        direction=1,
        sequence_bit_length=sequence_bit_length,
        bit_order=bit_order,
        max_workers=max_workers,
    )
    passed_tests = sum(1 for item in aggregates if item.passed)
    total_tests = len(CORE_NIST_TESTS)
    failed_names = [item.test_name for item in aggregates if not item.passed]
    result = FusionSchemeResult(
        fusion_name=fusion.fusion_name,
        source_formula=fusion.source_formula,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=total_tests - passed_tests,
        pass_ratio=passed_tests / float(total_tests),
        failed_test_names="; ".join(failed_names),
        xor_inputs=len(fusion.source_directions),
        includes_direction_4=4 in fusion.source_directions,
        is_best=False,
    )
    return result, aggregates


def select_best_fusion(results: list[FusionSchemeResult]) -> FusionSchemeResult:
    if not results:
        raise ValueError("results must not be empty.")
    return max(results, key=_fusion_rank_key)

