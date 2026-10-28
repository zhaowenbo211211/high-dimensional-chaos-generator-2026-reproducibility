# Maintainer: zhaowenbo
"""Run the rev1 experiment matrix (A, B, D, E) with caching and parallelism.

Usage:
    python3 scripts/run_all.py run --workers 10
    python3 scripts/run_all.py status
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SUBSYSTEM_ROOT / "src"
SCRIPTS_ROOT = SUBSYSTEM_ROOT / "scripts"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rev1_common import CACHE_ROOT, RESULTS_ROOT, run_task, task_cache_key
from tasks_rev1 import all_tasks


def cmd_run(workers: int, only: str | None) -> None:
    tasks = all_tasks()
    if only:
        tasks = [t for t in tasks if t["experiment"] == only]
    pending = []
    for task in tasks:
        cache_path = CACHE_ROOT / f"{task_cache_key(task)}.json"
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            if record.get("status") == "ok":
                continue
        pending.append(task)
    print(f"total tasks: {len(tasks)}, cached-ok: {len(tasks) - len(pending)}, to run: {len(pending)}")
    if not pending:
        return
    done = 0
    failed = 0
    if workers <= 1:
        for task in pending:
            record = run_task(task)
            done += 1
            failed += record["status"] != "ok"
            if done % 10 == 0 or record["status"] != "ok":
                print(f"[{done}/{len(pending)}] {record['status']} {task['experiment']} "
                      f"{task['metric']} {task['map']['slug']} {task['map']['kwargs'].get('n', 4)}",
                      flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for record in pool.map(_run_one, pending, chunksize=1):
                done += 1
                failed += record["status"] != "ok"
                if done % 20 == 0 or record["status"] != "ok":
                    print(f"[{done}/{len(pending)}] {record['status']} "
                          f"{record['task']['experiment']} {record['task']['metric']} "
                          f"{record['task']['map']['slug']} "
                          f"{record['task']['map']['kwargs'].get('n', 4)}",
                          flush=True)
    print(f"finished: {done} run, {failed} failed")


def _run_one(task):
    return run_task(task)


def cmd_status() -> None:
    tasks = all_tasks()
    counts: dict[str, dict[str, int]] = {}
    failures = []
    for task in tasks:
        exp = task["experiment"]
        counts.setdefault(exp, {"ok": 0, "failed": 0, "missing": 0})
        cache_path = CACHE_ROOT / f"{task_cache_key(task)}.json"
        if not cache_path.exists():
            counts[exp]["missing"] += 1
            continue
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        status = record.get("status", "missing")
        counts[exp][status if status in counts[exp] else "missing"] += 1
        if status != "ok":
            failures.append((exp, task["metric"], task["map"]["slug"], record.get("error", "?")))
    for exp, c in sorted(counts.items()):
        print(f"{exp}: ok={c['ok']} failed={c['failed']} missing={c['missing']}")
    for item in failures[:50]:
        print("FAILED:", item)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "run_status.json").write_text(
        json.dumps({"counts": counts, "failures": [
            {"experiment": e, "metric": m, "slug": s, "error": err} for e, m, s, err in failures
        ]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--only", choices=["A_4d_comparison", "B_golden_range",
                                          "D_dimension_scalability", "E_high_dim_literature"],
                       default=None)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args.workers, args.only)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
