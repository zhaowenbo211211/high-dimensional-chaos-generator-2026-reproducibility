#!/usr/bin/env python3
# Maintainer: zhaowenbo
"""Candidate-only frozen-data validation and temporary publication replot adapter."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
CANDIDATE = Path(__file__).resolve().parents[1]
SOFTWARE = CANDIDATE / "software/experiments_rev1_alignment"
RESULTS = CANDIDATE / "data/experiments_rev1_alignment/results"
REVIEW_RELS = (
    "submission_csf/review/phase_b_scientific_plot_restyling/data_integrity.json",
    "submission_csf/review/phase_b2_lyapunov_restructuring/data_integrity.json",
)
LEFT_REL = "submission_csf/figures/scientific_publication_b2/figure2_le.pdf"
EXPECTED_RECORDS = {"B_golden_range": 595, "D_dimension_scalability": 45,
                    "E_high_dim_literature": 225}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_sources():
    """Read the original renderer's literal constants without importing it."""
    path = SOFTWARE / "scripts/restyle_publication_figures.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SOURCE_HASHES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Renderer SOURCE_HASHES literal is missing")


def validate():
    checks = []
    for rel, expected in frozen_sources().items():
        path = RESULTS / rel
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash differs: {rel}")
        with path.open(newline="", encoding="utf-8") as handle:
            count = len(list(csv.DictReader(handle)))
        wanted = 119 if rel.startswith("expB_") else 45
        if count != wanted:
            raise RuntimeError(f"Row count differs: {rel}: {count} != {wanted}")
        checks.append({"path": str(path.relative_to(CANDIDATE)), "sha256": actual,
                       "rows": count})
    for rel, wanted in {
        "expD_dimension_scalability/dimension_metrics_summary.csv": 9,
        "expD_dimension_scalability/dimension_metrics_detailed.csv": 288,
        "expE_high_dim_literature/high_dim_all_methods_detailed.csv": 1440,
    }.items():
        path = RESULTS / rel
        with path.open(newline="", encoding="utf-8") as handle:
            count = len(list(csv.DictReader(handle)))
        if count != wanted:
            raise RuntimeError(f"Row count differs: {rel}: {count} != {wanted}")
        checks.append({"path": str(path.relative_to(CANDIDATE)), "sha256": sha(path),
                       "rows": count, "hash_policy": "recorded; frozen plot hashes cover six sources"})
    counts = Counter()
    for path in sorted((RESULTS / "cache").glob("*.json")):
        record = json.loads(path.read_text())
        task = record["task"]
        experiment = task["experiment"]
        if experiment not in EXPECTED_RECORDS:
            raise RuntimeError(f"Unexpected retired/unknown task record: {path.name}")
        if record["status"] != "ok":
            raise RuntimeError(f"Archived task status is not ok: {path.name}")
        payload = {"semantic_version": record["semantic_version"], "task": task}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()[:24]
        if key != record["cache_key"] or path.stem != key:
            raise RuntimeError(f"Archived canonical task key differs: {path.name}")
        counts[experiment] += 1
    if dict(counts) != EXPECTED_RECORDS:
        raise RuntimeError(f"Archived B/D/E record counts differ: {dict(counts)}")
    for rel in REVIEW_RELS:
        path = CANDIDATE / "data/provenance" / rel
        obj = json.loads(path.read_text())
        if "data" not in obj:
            raise RuntimeError(f"Missing selected-array reference: {rel}")
    left = CANDIDATE / "data/figures/scientific_publication_b2/figure2_le.pdf"
    if not left.is_file():
        raise RuntimeError("Missing frozen upstream figure2_le.pdf")
    return {"status": "PASS_STORED_DATA_ONLY", "csv_checks": checks,
            "archived_task_counts": dict(counts), "archived_task_keys_checked": sum(counts.values()),
            "scientific_metrics_recomputed": False, "experiments_rerun": False,
            "historical_task_status_is_not_fresh_execution": True}


def replot(output):
    validation = validate()
    output = output.expanduser().resolve()
    if output == CANDIDATE or CANDIDATE in output.parents:
        raise ValueError("Choose an output directory outside the candidate archive")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if temporary_root not in output.parents:
        raise ValueError("Choose a fresh output directory under the system temporary directory")
    if output.exists():
        raise FileExistsError("Choose a new output directory to avoid overwriting existing files")
    output.mkdir(parents=True)
    log = {"adapter": "new release-only path adaptation; original algorithms unchanged",
           "validation": validation, "copies": [], "commands": [], "outputs": []}
    def copy(source, target, stage):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        log["copies"].append({"source": str(source.relative_to(CANDIDATE)),
                              "staged_relative_path": str(target.relative_to(stage)),
                              "source_sha256": sha(source), "staged_sha256": sha(target),
                              "transformation": "byte_exact_staging_copy"})
    try:
        with tempfile.TemporaryDirectory(prefix="paper2025-stored-data-") as temp:
            stage = Path(temp)
            for source in sorted((SOFTWARE / "scripts").rglob("*.py")):
                copy(source, stage / "experiments_rev1_alignment/scripts" /
                     source.relative_to(SOFTWARE / "scripts"), stage)
            for rel in frozen_sources():
                copy(RESULTS / rel, stage / "experiments_rev1_alignment/results" / rel, stage)
            for rel in REVIEW_RELS:
                copy(CANDIDATE / "data/provenance" / rel, stage / rel, stage)
            copy(CANDIDATE / "data/figures/scientific_publication_b2/figure2_le.pdf",
                 stage / LEFT_REL, stage)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                       MPLCONFIGDIR=str(stage / "matplotlib-config"))
            for script, manifest in [("restyle_phase_b2_figures.py", "b2_data_integrity.json"),
                                     ("restyle_lyapunov_comparison_bars.py", "bars_data_integrity.json")]:
                cmd = [sys.executable, "-B", str(stage / "experiments_rev1_alignment/scripts" / script),
                       "--output-dir", str(output / "figures"), "--manifest-path", str(output / manifest)]
                entry = {"script": script, "argv": cmd}
                log["commands"].append(entry)
                result = subprocess.run(cmd, cwd=stage, env=env, text=True, capture_output=True)
                entry.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                if result.returncode:
                    raise RuntimeError(f"Renderer failed: {script}; see path_adaptation_log.json")
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    log["outputs"].append({"path": str(path.relative_to(output)), "sha256": sha(path)})
            log["status"] = "PASS_STORED_DATA_REPLOT"
    except Exception as exc:
        log["status"] = "FAILED"
        log["error"] = str(exc)
        raise
    finally:
        (output / "path_adaptation_log.json").write_text(json.dumps(log, indent=2) + "\n")
    return {"status": log["status"], "output": str(output), "files": len(log["outputs"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="stdlib-only frozen CSV/task-record checks")
    renderer = sub.add_parser("replot", help="render stored values in disposable staging")
    renderer.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate() if args.command == "validate" else replot(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
