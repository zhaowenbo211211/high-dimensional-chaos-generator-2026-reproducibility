# Maintainer: zhaowenbo
"""Helpers for exporting datasets and running the bundled official NIST STS."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable

from metrics.nist_randomness import DEFAULT_NIST_SEQUENCE_BIT_LENGTH, DEFAULT_NIST_SEQUENCE_COUNT, DEFAULT_NIST_WARMUP_STEPS
from metrics.prng_system_level_official import build_system_level_schemes, generate_system_level_payload
from utils.io_utils import ensure_directory


OFFICIAL_NIST_TESTS: tuple[tuple[int, str], ...] = (
    (1, "Frequency"),
    (2, "Block Frequency"),
    (3, "Cumulative Sums"),
    (4, "Runs"),
    (5, "Longest Run of Ones"),
    (6, "Rank"),
    (7, "Discrete Fourier Transform"),
    (8, "NonOverlapping Template Matchings"),
    (9, "Overlapping Template Matchings"),
    (10, "Universal Statistical"),
    (11, "Approximate Entropy"),
    (12, "Random Excursions"),
    (13, "Random Excursions Variant"),
    (14, "Serial"),
    (15, "Linear Complexity"),
)
OFFICIAL_NIST_TEST_IDS: frozenset[int] = frozenset(test_id for test_id, _ in OFFICIAL_NIST_TESTS)


@dataclass(frozen=True)
class OfficialNISTDatasetOption:
    key: str
    title: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialNISTRunArtifacts:
    run_dir: str
    input_file: str
    staged_input_file: str
    runtime_nist_root: str
    final_report: str
    algorithm_testing_dir: str
    summary_json: str
    stdout_log: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


DATASET_OPTIONS: tuple[OfficialNISTDatasetOption, ...] = (
    OfficialNISTDatasetOption(
        key="prng_best",
        title="Plain Concat ROTL/XOR PRNG",
        description="u_t=x1(t)||x2(t)||x3(t)||x4(t); y_t=u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43).",
    ),
)


def list_official_nist_dataset_options() -> list[OfficialNISTDatasetOption]:
    return list(DATASET_OPTIONS)


def list_official_nist_tests() -> list[tuple[int, str]]:
    return list(OFFICIAL_NIST_TESTS)


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned or "official_nist_run"


def _bytes_to_ascii_bits(payload: bytes, bit_length: int) -> str:
    return "".join(f"{byte:08b}" for byte in payload)[:bit_length]


def _build_prng_best_sequences(sequence_bit_length: int, sequence_count: int, warmup_steps: int) -> list[bytes]:
    scheme = next(
        scheme
        for scheme in build_system_level_schemes()
        if scheme.scheme_name == "plain_concat_linear_mixing"
    )
    payload = generate_system_level_payload(
        scheme=scheme,
        sequence_bit_length=sequence_bit_length,
        sequence_count=sequence_count,
        warmup_steps=warmup_steps,
        diagnostics_dir=None,
    )
    sequence_byte_length = (int(sequence_bit_length) + 7) // 8
    return [
        payload[index * sequence_byte_length : (index + 1) * sequence_byte_length]
        for index in range(int(sequence_count))
    ]


BUILTIN_SEQUENCE_BUILDERS: dict[str, Callable[[int, int, int], list[bytes]]] = {
    "prng_best": _build_prng_best_sequences,
}


def export_builtin_dataset_to_ascii(
    dataset_key: str,
    output_path: str | Path,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    warmup_steps: int = DEFAULT_NIST_WARMUP_STEPS,
) -> Path:
    if dataset_key not in BUILTIN_SEQUENCE_BUILDERS:
        raise ValueError(f"unsupported dataset key: {dataset_key}")
    sequences = BUILTIN_SEQUENCE_BUILDERS[dataset_key](sequence_bit_length, sequence_count, warmup_steps)
    output_file = Path(output_path)
    ensure_directory(output_file.parent)
    with output_file.open("w", encoding="ascii") as handle:
        for payload in sequences:
            handle.write(_bytes_to_ascii_bits(payload, sequence_bit_length))
    return output_file


def ensure_relative_input_path(input_file: str | Path, nist_root: str | Path) -> str:
    input_path = Path(input_file)
    nist_path = Path(nist_root)
    try:
        relative = input_path.resolve().relative_to(nist_path.resolve())
        relative_text = str(relative)
        if " " in relative_text:
            raise ValueError("relative input path contains spaces; the official assess parser is space-sensitive.")
        return relative_text
    except ValueError:
        raw_text = str(input_path)
        if " " in raw_text:
            raise ValueError(
                "official assess cannot reliably read input paths containing spaces; "
                "please place the input file under nist_sp_800_22 with a no-space relative path."
            )
        return raw_text


def _parse_final_analysis_report(report_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^\s*\d")
    for line in report_text.splitlines():
        if not pattern.match(line):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        rows.append(
            {
                "uniformity_p_value": parts[-3],
                "proportion": parts[-2],
                "test_name": parts[-1],
                "raw_line": line.rstrip(),
            }
        )
    return rows


def _copy_input_into_runtime(
    source_input_file: str | Path,
    runtime_nist_root: str | Path,
    input_mode: str,
) -> Path:
    source_path = Path(source_input_file).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"input file not found: {source_path}")

    runtime_root = Path(runtime_nist_root)
    runtime_input_dir = ensure_directory(runtime_root / "ui_inputs")
    suffix = ".txt" if input_mode.strip().lower() == "ascii" else ".bin"
    staged_input = runtime_input_dir / f"input{suffix}"
    shutil.copyfile(source_path, staged_input)
    return staged_input


def _create_runtime_sandbox(
    template_nist_root: str | Path,
    base_dir: str | Path,
) -> Path:
    template_root = Path(template_nist_root).resolve()
    runtime_parent = ensure_directory(Path(base_dir) / "runtime")
    runtime_nist_root = runtime_parent / template_root.name

    def _ignore(path: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        current = Path(path)
        if current == template_root:
            if "ui_inputs" in names:
                ignored.add("ui_inputs")
        if current == template_root / "experiments":
            if "AlgorithmTesting" in names:
                ignored.add("AlgorithmTesting")
        return ignored

    shutil.copytree(template_root, runtime_nist_root, ignore=_ignore)
    return runtime_nist_root


def _ensure_algorithm_testing_tree(nist_root: str | Path) -> Path:
    nist_path = Path(nist_root)
    experiments_dir = nist_path / "experiments"
    algorithm_testing_dir = experiments_dir / "AlgorithmTesting"
    template_dir = experiments_dir / "BBS"

    if algorithm_testing_dir.exists():
        return algorithm_testing_dir
    if not template_dir.exists():
        raise FileNotFoundError(
            f"cannot rebuild AlgorithmTesting because template directory is missing: {template_dir}"
        )

    shutil.copytree(template_dir, algorithm_testing_dir)
    return algorithm_testing_dir


def _normalize_selected_tests(selected_tests: list[int] | tuple[int, ...] | None) -> list[int]:
    if selected_tests is None:
        return [test_id for test_id, _ in OFFICIAL_NIST_TESTS]
    normalized = sorted({int(test_id) for test_id in selected_tests})
    if not normalized:
        raise ValueError("at least one NIST test category must be selected.")
    invalid = [test_id for test_id in normalized if test_id not in OFFICIAL_NIST_TEST_IDS]
    if invalid:
        raise ValueError(f"unsupported NIST test ids: {invalid}")
    return normalized


def _build_assess_interaction(
    input_path_for_assess: str,
    sequence_count: int,
    input_mode: str,
    selected_tests: list[int],
) -> str:
    normalized_mode = input_mode.strip().lower()
    if normalized_mode not in {"ascii", "binary"}:
        raise ValueError("input_mode must be 'ascii' or 'binary'.")
    mode_choice = "0" if normalized_mode == "ascii" else "1"

    all_test_ids = [test_id for test_id, _ in OFFICIAL_NIST_TESTS]
    if selected_tests == all_test_ids:
        test_selection_lines = ["1"]
    else:
        vector = "".join("1" if test_id in selected_tests else "0" for test_id in all_test_ids)
        test_selection_lines = ["0", vector]

    return "\n".join(
        [
            "0",
            str(input_path_for_assess),
            *test_selection_lines,
            "0",
            str(sequence_count),
            mode_choice,
            "",
        ]
    )


def run_official_nist_assess(
    nist_root: str | Path,
    input_file: str | Path,
    sequence_bit_length: int = DEFAULT_NIST_SEQUENCE_BIT_LENGTH,
    sequence_count: int = DEFAULT_NIST_SEQUENCE_COUNT,
    label: str = "official_nist_run",
    output_root: str | Path = "results/official_nist_ui",
    input_mode: str = "ascii",
    selected_tests: list[int] | tuple[int, ...] | None = None,
    timeout_seconds: float | None = 900.0,
) -> OfficialNISTRunArtifacts:
    nist_path = Path(nist_root).resolve()
    assess_path = (nist_path / "assess").resolve()
    if not assess_path.exists():
        raise FileNotFoundError(f"assess not found: {assess_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = _sanitize_label(label)
    base_dir = ensure_directory(Path(output_root) / f"{timestamp}_{safe_label}")
    runtime_nist_root = _create_runtime_sandbox(nist_path, base_dir)
    runtime_assess_path = (runtime_nist_root / "assess").resolve()
    _ensure_algorithm_testing_tree(runtime_nist_root)
    staged_input_file = _copy_input_into_runtime(
        source_input_file=input_file,
        runtime_nist_root=runtime_nist_root,
        input_mode=input_mode,
    )
    input_path_for_assess = ensure_relative_input_path(staged_input_file, runtime_nist_root)
    stdout_log = base_dir / "assess_stdout.log"
    normalized_tests = _normalize_selected_tests(selected_tests)
    interaction = _build_assess_interaction(
        input_path_for_assess=input_path_for_assess,
        sequence_count=sequence_count,
        input_mode=input_mode,
        selected_tests=normalized_tests,
    )
    with stdout_log.open("w", encoding="utf-8") as log_handle:
        try:
            completed = subprocess.run(
                [str(runtime_assess_path), str(sequence_bit_length)],
                cwd=str(runtime_nist_root),
                input=interaction,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"official assess timed out after {timeout_seconds} seconds"
            ) from exc
    if completed.returncode != 1:
        raise RuntimeError(f"official assess returned unexpected code {completed.returncode}")

    algorithm_testing_dst = runtime_nist_root / "experiments" / "AlgorithmTesting"

    final_report = algorithm_testing_dst / "finalAnalysisReport.txt"
    report_text = final_report.read_text(encoding="utf-8", errors="replace")
    summary_rows = _parse_final_analysis_report(report_text)
    summary_json = base_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "label": safe_label,
                "sequence_bit_length": sequence_bit_length,
                "sequence_count": sequence_count,
                "input_file": str(input_file),
                "staged_input_file": str(staged_input_file),
                "runtime_nist_root": str(runtime_nist_root),
                "input_path_for_assess": str(input_path_for_assess),
                "input_mode": input_mode,
                "selected_tests": normalized_tests,
                "return_code": completed.returncode,
                "tests": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return OfficialNISTRunArtifacts(
        run_dir=str(base_dir),
        input_file=str(input_file),
        staged_input_file=str(staged_input_file),
        runtime_nist_root=str(runtime_nist_root),
        final_report=str(final_report),
        algorithm_testing_dir=str(algorithm_testing_dst),
        summary_json=str(summary_json),
        stdout_log=str(stdout_log),
    )
