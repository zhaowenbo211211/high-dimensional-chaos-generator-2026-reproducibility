# Maintainer: zhaowenbo
"""Official-STS-oriented system-level 4-direction PRNG helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from metrics.nist_randomness import (
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
    generate_representative_enhanced_4d_words,
)
from utils.io_utils import ensure_directory


DEFAULT_SYSTEM_LEVEL_R1 = 17
DEFAULT_SYSTEM_LEVEL_R2 = 43
OFFICIAL_STS_CATEGORY_ORDER: tuple[str, ...] = (
    "Frequency",
    "Block Frequency",
    "Cumulative Sums",
    "Runs",
    "Longest Run of Ones",
    "Rank",
    "Discrete Fourier Transform",
    "NonOverlapping Template Matchings",
    "Overlapping Template Matchings",
    "Universal Statistical",
    "Approximate Entropy",
    "Random Excursions",
    "Random Excursions Variant",
    "Serial",
    "Linear Complexity",
)
_OFFICIAL_REPORT_TEST_NAME_MAP: dict[str, str] = {
    "Frequency": "Frequency",
    "BlockFrequency": "Block Frequency",
    "CumulativeSums": "Cumulative Sums",
    "Runs": "Runs",
    "LongestRun": "Longest Run of Ones",
    "Rank": "Rank",
    "FFT": "Discrete Fourier Transform",
    "NonOverlappingTemplate": "NonOverlapping Template Matchings",
    "OverlappingTemplate": "Overlapping Template Matchings",
    "Universal": "Universal Statistical",
    "ApproximateEntropy": "Approximate Entropy",
    "RandomExcursions": "Random Excursions",
    "RandomExcursionsVariant": "Random Excursions Variant",
    "Serial": "Serial",
    "LinearComplexity": "Linear Complexity",
}


@dataclass(frozen=True)
class SystemLevelPRNGScheme:
    scheme_name: str
    preprocessing_description: str
    linear_r1: int = DEFAULT_SYSTEM_LEVEL_R1
    linear_r2: int = DEFAULT_SYSTEM_LEVEL_R2

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _rotl64(words: np.ndarray, shift: int) -> np.ndarray:
    shift_value = int(shift) % 64
    words_u64 = words.astype(np.uint64, copy=False)
    if shift_value == 0:
        return words_u64
    return ((words_u64 << np.uint64(shift_value)) | (words_u64 >> np.uint64(64 - shift_value))).astype(np.uint64, copy=False)


def build_system_level_schemes() -> list[SystemLevelPRNGScheme]:
    return [
        SystemLevelPRNGScheme(
            scheme_name="plain_concat_linear_mixing",
            preprocessing_description=(
                "raw_word_t = x1(t) || x2(t) || x3(t) || x4(t); "
                "L_t = raw_t XOR ROTL64(raw_(t-1),17) XOR ROTL64(raw_(t-2),43)"
            ),
        ),
    ]


def _assemble_plain_words(raw_states: np.ndarray) -> np.ndarray:
    return (
        (raw_states[0].astype(np.uint64) << np.uint64(48))
        | (raw_states[1].astype(np.uint64) << np.uint64(32))
        | (raw_states[2].astype(np.uint64) << np.uint64(16))
        | raw_states[3].astype(np.uint64)
    ).astype(np.uint64, copy=False)


def generate_system_level_payload(
    scheme: SystemLevelPRNGScheme,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    diagnostics_dir: str | Path | None = None,
) -> bytes:
    total_bit_length = int(sequence_bit_length) * int(sequence_count)
    block_count = (total_bit_length + 63) // 64
    lag_words = block_count + 2
    raw_states = generate_representative_enhanced_4d_words(
        word_count=lag_words,
        warmup_steps=warmup_steps,
    )
    source_states = raw_states[:, :lag_words].astype(np.uint16, copy=False)
    raw_words = _assemble_plain_words(source_states)
    current = raw_words[2 : 2 + block_count]
    lag1 = _rotl64(raw_words[1 : 1 + block_count], scheme.linear_r1)
    lag2 = _rotl64(raw_words[0:block_count], scheme.linear_r2)
    linear_words = (current ^ lag1 ^ lag2).astype(np.uint64, copy=False)

    payload = linear_words.byteswap().tobytes()
    total_bytes = (total_bit_length + 7) // 8
    payload = payload[:total_bytes]
    if total_bit_length % 8:
        trimmed = bytearray(payload)
        trimmed[-1] &= 0xFF << (8 - (total_bit_length % 8))
        payload = bytes(trimmed)

    if diagnostics_dir is not None:
        diagnostics_path = ensure_directory(diagnostics_dir)
        (diagnostics_path / f"{scheme.scheme_name}_{total_bit_length}bit.bin").write_bytes(payload)
    return payload


def export_payload_to_ascii(
    payload: bytes,
    bit_length: int,
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path)
    ensure_directory(output_file.parent)
    with output_file.open("w", encoding="ascii") as handle:
        handle.write("".join(f"{byte:08b}" for byte in payload)[:bit_length])
    return output_file


def parse_official_summary_rows_by_category(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in OFFICIAL_STS_CATEGORY_ORDER}
    for row in summary_rows:
        raw_name = str(row.get("test_name", "")).strip()
        category_name = _OFFICIAL_REPORT_TEST_NAME_MAP.get(raw_name)
        if category_name is None:
            continue
        grouped.setdefault(category_name, []).append(row)

    records: list[dict[str, object]] = []
    for category_name in OFFICIAL_STS_CATEGORY_ORDER:
        rows = grouped.get(category_name, [])
        flagged_rows = [row for row in rows if "*" in str(row.get("raw_line", ""))]
        records.append(
            {
                "test_name": category_name,
                "passed": len(flagged_rows) == 0,
                "flagged_subtests": len(flagged_rows),
                "total_report_rows": len(rows),
                "notes": " | ".join(str(row.get("raw_line", "")).strip() for row in flagged_rows[:4]),
            }
        )
    return records
