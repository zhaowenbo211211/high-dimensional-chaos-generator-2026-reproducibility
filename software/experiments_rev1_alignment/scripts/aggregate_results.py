# Maintainer: zhaowenbo
"""Aggregate cached task records into per-experiment CSV files.

Usage: python3 scripts/aggregate_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SUBSYSTEM_ROOT / "src"
SCRIPTS_ROOT = SUBSYSTEM_ROOT / "scripts"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rev1_common import CACHE_ROOT, RESULTS_ROOT, task_cache_key
from tasks_rev1 import (
    BASELINE_4D,
    LITERATURE_METHODS,
    METRICS,
    tasks_experiment_a,
    tasks_experiment_b,
    tasks_experiment_d,
    tasks_experiment_e,
)

A_ROW_ORDER = ["The proposed method"] + [display for display, _ in BASELINE_4D]
E_ROW_ORDER = ["proposed_rev1"] + [slug for _, slug, _ in LITERATURE_METHODS]
E_DISPLAY = {"proposed_rev1": "The proposed method"}
E_DISPLAY.update({slug: display for display, slug, _ in LITERATURE_METHODS})


def load_record(task):
    cache_path = CACHE_ROOT / f"{task_cache_key(task)}.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def collect(experiment: str, tasks) -> dict:
    """Return {(slug, key): {metric: result}} plus failure list."""

    data: dict[tuple, dict] = {}
    failures = []
    for task in tasks:
        if task["experiment"] != experiment:
            continue
        slug = task["map"]["slug"]
        n = int(task["map"]["kwargs"].get("n", 4))
        group = task["map"].get("golden_group", {})
        key = (slug, n, int(group.get("A_odd", 0)))
        record = load_record(task)
        if record is None or record.get("status") != "ok":
            failures.append({"slug": slug, "n": n, "metric": task["metric"],
                             "A_odd": group.get("A_odd"),
                             "error": None if record is None else record.get("error")})
            continue
        data.setdefault(key, {})[task["metric"]] = record["result"]
    return {"data": data, "failures": failures}


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def aggregate_a() -> dict:
    tasks = tasks_experiment_a()
    collected = collect("A_4d_comparison", tasks)
    out_dir = RESULTS_ROOT / "expA_4d_comparison"
    rows_by_metric: dict[str, list[dict]] = {m: [] for m in METRICS}
    for display in A_ROW_ORDER:
        slug = "proposed_rev1" if display == "The proposed method" else next(
            s for d, s in
            [(d, cp.rsplit(".", 1)[1].lower()) for d, cp in BASELINE_4D] if d == display)
        entry = collected["data"].get((slug, 4, 0), {})
        for metric in METRICS:
            result = entry.get(metric)
            if result is None:
                continue
            row = {"map_name": display, "slug": slug}
            if metric == "le":
                row.update({
                    "lyap_max": result["lyap_max"], "lyap_mean": result["lyap_mean"],
                    "lyap_std": result["lyap_std"],
                    "renormalizations": result["renormalizations"],
                    "final_distance": result["final_distance"],
                    "is_chaotic": result["is_chaotic"],
                })
            else:
                comps = result[f"{metric}_components"]
                for i, value in enumerate(comps, start=1):
                    row[f"{metric}_x{i}"] = value
                row[f"{metric}_mean"] = result[f"{metric}_mean"]
                row["raw_length"] = result["raw_length"]
            rows_by_metric[metric].append(row)
    fields = {
        "le": ["map_name", "slug", "lyap_max", "lyap_mean", "lyap_std",
               "renormalizations", "final_distance", "is_chaotic"],
        **{m: ["map_name", "slug"] + [f"{m}_x{i}" for i in range(1, 5)]
           + [f"{m}_mean", "raw_length"] for m in ("se", "fe", "si", "pe")},
    }
    paths = {}
    for metric in METRICS:
        paths[metric] = str(_write_csv(out_dir / f"{metric}_comparison.csv",
                                       fields[metric], rows_by_metric[metric]))
    return {"paths": paths, "failures": collected["failures"]}


def aggregate_b() -> dict:
    tasks = tasks_experiment_b()
    collected = collect("B_golden_range", tasks)
    out_dir = RESULTS_ROOT / "expB_golden_range"
    groups = sorted({key[2] for key in collected["data"]})
    rows_by_metric: dict[str, list[dict]] = {m: [] for m in METRICS}
    for group_index, a_odd in enumerate(groups, start=1):
        entry = collected["data"].get(("proposed_rev1_A" + str(a_odd), 4, a_odd), {})
        for metric in METRICS:
            result = entry.get(metric)
            if result is None:
                continue
            row = {"group_index": group_index, "A_odd": a_odd,
                   "b": 3 * a_odd // 10}
            if metric == "le":
                row.update({"lyap_max": result["lyap_max"], "lyap_mean": result["lyap_mean"],
                            "lyap_std": result["lyap_std"],
                            "renormalizations": result["renormalizations"]})
            else:
                comps = result[f"{metric}_components"]
                for i, value in enumerate(comps, start=1):
                    row[f"{metric}_x{i}"] = value
                row[f"{metric}_mean"] = result[f"{metric}_mean"]
            rows_by_metric[metric].append(row)
    paths = {}
    for metric in METRICS:
        rows = rows_by_metric[metric]
        fields = list(rows[0].keys()) if rows else []
        paths[metric] = str(_write_csv(out_dir / f"golden_range_{metric}.csv", fields, rows))
    return {"paths": paths, "failures": collected["failures"], "groups": len(groups)}


def aggregate_d() -> dict:
    tasks = tasks_experiment_d()
    collected = collect("D_dimension_scalability", tasks)
    out_dir = RESULTS_ROOT / "expD_dimension_scalability"
    summary_rows = []
    detailed_rows = []
    for n in range(4, 13):
        entry = collected["data"].get(("proposed_rev1", n, 0), {})
        if not entry:
            continue
        le = entry.get("le", {})
        row = {"dimension": n,
               "lyap_max": le.get("lyap_max"), "lyap_mean": le.get("lyap_mean"),
               "lyap_components": json.dumps(le.get("lyap_components", []))}
        for metric in ("se", "fe", "si", "pe"):
            result = entry.get(metric, {})
            row[f"{metric}_mean"] = result.get(f"{metric}_mean")
            row[f"{metric}_components"] = json.dumps(result.get(f"{metric}_components", []))
            for i, value in enumerate(result.get(f"{metric}_components", []), start=1):
                detailed_rows.append({"dimension": n, "component": i,
                                      "metric": metric, "value": value})
        summary_rows.append(row)
    fields = ["dimension", "lyap_max", "lyap_mean", "lyap_components",
              "se_mean", "se_components", "fe_mean", "fe_components",
              "si_mean", "si_components", "pe_mean", "pe_components"]
    paths = {
        "summary": str(_write_csv(out_dir / "dimension_metrics_summary.csv", fields, summary_rows)),
        "detailed": str(_write_csv(out_dir / "dimension_metrics_detailed.csv",
                                   ["dimension", "component", "metric", "value"], detailed_rows)),
    }
    return {"paths": paths, "failures": collected["failures"]}


def aggregate_e() -> dict:
    tasks = tasks_experiment_e()
    collected = collect("E_high_dim_literature", tasks)
    out_dir = RESULTS_ROOT / "expE_high_dim_literature"
    summary_rows = []
    detailed_rows = []
    for n in range(4, 13):
        for slug in E_ROW_ORDER:
            entry = collected["data"].get((slug, n, 0), {})
            if not entry:
                continue
            le = entry.get("le", {})
            row = {"dimension": n, "method_slug": slug, "display_name": E_DISPLAY[slug],
                   "lambda_max": le.get("lyap_max"), "lyap_mean": le.get("lyap_mean"),
                   "lyap_components": json.dumps(le.get("lyap_components", []))}
            for metric in ("se", "fe", "si", "pe"):
                result = entry.get(metric, {})
                row[f"{metric}_mean"] = result.get(f"{metric}_mean")
                row[f"{metric}_components"] = json.dumps(result.get(f"{metric}_components", []))
                for i, value in enumerate(result.get(f"{metric}_components", []), start=1):
                    detailed_rows.append({"dimension": n, "method_slug": slug,
                                          "component": i, "metric": metric, "value": value})
            summary_rows.append(row)
    fields = ["dimension", "method_slug", "display_name", "lambda_max", "lyap_mean",
              "lyap_components", "se_mean", "se_components", "fe_mean", "fe_components",
              "si_mean", "si_components", "pe_mean", "pe_components"]
    paths = {
        "summary": str(_write_csv(out_dir / "high_dim_all_methods_summary.csv", fields, summary_rows)),
        "detailed": str(_write_csv(out_dir / "high_dim_all_methods_detailed.csv",
                                   ["dimension", "method_slug", "component", "metric", "value"],
                                   detailed_rows)),
    }
    return {"paths": paths, "failures": collected["failures"]}


def main() -> None:
    report = {
        "A_4d_comparison": aggregate_a(),
        "B_golden_range": aggregate_b(),
        "D_dimension_scalability": aggregate_d(),
        "E_high_dim_literature": aggregate_e(),
    }
    all_failures = [
        {"experiment": exp, **failure}
        for exp, payload in report.items() for failure in payload["failures"]
    ]
    print(json.dumps({exp: {"paths": p["paths"], "failures": len(p["failures"])}
                      for exp, p in report.items()}, indent=2))
    if all_failures:
        print("FAILURES:")
        print(json.dumps(all_failures, indent=2))
    (RESULTS_ROOT / "aggregate_report.json").write_text(
        json.dumps({"failures": all_failures}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
