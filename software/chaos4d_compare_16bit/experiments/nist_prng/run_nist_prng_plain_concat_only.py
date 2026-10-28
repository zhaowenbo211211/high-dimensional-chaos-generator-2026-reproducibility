# Maintainer: zhaowenbo
"""Generate data for the final hardware-oriented plain system-level PRNG scheme only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from metrics.nist_randomness import (
    DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    DEFAULT_NIST_SEQUENCE_COUNT,
    DEFAULT_NIST_WARMUP_STEPS,
)
from metrics.prng_system_level_official import build_system_level_schemes, export_payload_to_ascii, generate_system_level_payload
from utils.io_utils import ensure_directory


def run_plain_concat_only(
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
    output_root: str | Path = "results/nist_prng_system_level_plain_only",
) -> dict[str, str]:
    base_dir = ensure_directory(output_root)
    diagnostics_dir = ensure_directory(base_dir / "diagnostics" / "plain_concat_linear_mixing")
    scheme = next(scheme for scheme in build_system_level_schemes() if scheme.scheme_name == "plain_concat_linear_mixing")
    total_bit_length = int(sequence_bit_length) * int(sequence_count)

    payload = generate_system_level_payload(
        scheme=scheme,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        warmup_steps=warmup_steps,
        diagnostics_dir=diagnostics_dir,
    )
    ascii_path = export_payload_to_ascii(
        payload=payload,
        bit_length=total_bit_length,
        output_path=diagnostics_dir / f"{scheme.scheme_name}_{total_bit_length}bit_ascii.txt",
    )
    metadata_path = diagnostics_dir / "plain_concat_linear_mixing_metadata.txt"
    metadata_path.write_text(
        "\n".join(
            [
                "Final hardware-oriented system-level PRNG source generation",
                "Selected scheme: plain_concat_linear_mixing",
                "Stage 0: u_t = x1(t) || x2(t) || x3(t) || x4(t)",
                "Stage 1: y_t = u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43)",
                f"Sequence bit length: {sequence_bit_length}",
                f"Sequence count: {sequence_count}",
                f"Total bit length: {total_bit_length}",
                f"Warm-up discard: {warmup_steps}",
                "Input mode target for official STS: ASCII 0/1",
                "Tail post-processing: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ascii_path": str(ascii_path),
        "binary_path": str(diagnostics_dir / f"{scheme.scheme_name}_{total_bit_length}bit.bin"),
        "metadata_path": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate only the final plain system-level PRNG source data.")
    parser.add_argument("--sequence-bit-length", type=int, default=DEFAULT_NIST_SEQUENCE_BIT_LENGTH)
    parser.add_argument("--sequence-count", type=int, default=DEFAULT_NIST_SEQUENCE_COUNT)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_NIST_WARMUP_STEPS)
    parser.add_argument("--output-root", default="results/nist_prng_system_level_plain_only")
    args = parser.parse_args()

    result = run_plain_concat_only(
        sequence_bit_length=args.sequence_bit_length,
        sequence_count=args.sequence_count,
        warmup_steps=args.warmup_steps,
        output_root=args.output_root,
    )
    print(f"Saved ASCII source: {result['ascii_path']}")
    print(f"Saved binary source: {result['binary_path']}")
    print(f"Saved metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
