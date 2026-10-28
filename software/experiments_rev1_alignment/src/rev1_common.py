# Maintainer: zhaowenbo
"""Shared infrastructure for the rev1 experiment subsystem.

Task-based metric evaluation with per-task JSON caching. A task fully specifies
map construction (class path + kwargs), initialization, and the metric with its
settings; the cache key is a sha256 over the canonical JSON of that spec plus a
semantic version string. Any semantic change invalidates old cache entries.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

SUBSYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SUBSYSTEM_ROOT / "src"
RESULTS_ROOT = SUBSYSTEM_ROOT / "results"
CACHE_ROOT = RESULTS_ROOT / "cache"

SEMANTIC_VERSION = "rev1-core-v1"

METRIC_DEFAULTS = {
    "le": {"max_steps": 5000, "transient": 1000, "perturb_lsb": 1, "renorm_threshold": 5e-2},
    "se": {"raw_length": 5000, "transient": 1000, "mode": "direct"},
    "fe": {"raw_length": 5000, "transient": 1000, "mode": "direct", "m": 2, "r": None,
           "tau": 1, "n": 2},
    "pe": {"raw_length": 5000, "transient": 1000, "mode": "direct", "m": 5, "tau": 1,
           "normalized": True},
    "si": {"raw_length": 40000, "transient": 3000, "mode": "direct", "wavelet": "morlet"},
}


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def task_cache_key(task: dict[str, Any]) -> str:
    payload = {"semantic_version": SEMANTIC_VERSION, "task": task}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:24]


def build_map(map_spec: dict[str, Any]):
    """Instantiate a map from a spec: {class_path, kwargs, init}."""

    class_path = map_spec["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    kwargs = {k: v for k, v in map_spec.get("kwargs", {}).items()}
    chaos_map = cls(**kwargs)
    init = map_spec.get("init", {})
    if init.get("kind") == "u16":
        chaos_map.reset(initial_state_u16=[int(v) for v in init["values"]])
    elif init.get("kind") == "float":
        chaos_map.reset(initial_state_float=[float(v) for v in init["values"]])
    elif init.get("kind") == "none":
        pass
    else:
        raise ValueError(f"unknown init kind: {init.get('kind')}")
    return chaos_map


def run_metric_uncached(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one metric task and return a JSON-serializable result dict."""

    import sys
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    metric = task["metric"]
    chaos_map = build_map(task["map"])

    if metric == "le":
        from metrics.lyapunov import wolf_lyapunov_full_state_nd
        cfg = METRIC_DEFAULTS["le"] | task.get("metric_settings", {})
        result = wolf_lyapunov_full_state_nd(
            chaos_map,
            max_steps=int(cfg["max_steps"]),
            transient=int(cfg["transient"]),
            perturb_lsb=int(cfg["perturb_lsb"]),
            renorm_threshold=float(cfg["renorm_threshold"]),
        )
        return {
            "lyap_max": float(result["lyap_max"]),
            "lyap_mean": float(result["lyap_mean"]),
            "lyap_std": float(result["lyap_std"]),
            "lyap_components": [float(v) for v in np.asarray(result["lyap_components"], dtype=np.float64)],
            "renormalizations": int(result["renormalizations"]),
            "final_distance": float(result["final_distance"]),
            "is_chaotic": bool(result["is_chaotic"]),
            "settings": {k: v for k, v in cfg.items()},
        }

    if metric == "se":
        from metrics.spectral_entropy import spectral_entropy_complexity
        cfg = METRIC_DEFAULTS["se"] | task.get("metric_settings", {})
        traj = chaos_map.iterate(num_steps=int(cfg["raw_length"]), discard_transient=int(cfg["transient"]))
        result = spectral_entropy_complexity(traj, mode=str(cfg["mode"]))
        return {
            "se_components": [float(v) for v in result["se_components"]],
            "se_mean": float(result["se_mean"]),
            "raw_length": int(result["raw_length"]),
            "test_length": int(result["test_length"]),
            "settings": {k: v for k, v in cfg.items()},
        }

    if metric == "fe":
        from metrics.fuzzy_entropy import fuzzy_entropy
        cfg = METRIC_DEFAULTS["fe"] | task.get("metric_settings", {})
        traj = chaos_map.iterate(num_steps=int(cfg["raw_length"]), discard_transient=int(cfg["transient"]))
        result = fuzzy_entropy(
            traj, mode=str(cfg["mode"]), m=int(cfg["m"]),
            r=None if cfg["r"] is None else float(cfg["r"]),
            tau=int(cfg["tau"]), n=int(cfg["n"]),
        )
        return {
            "fe_components": [float(v) for v in result["fe_components"]],
            "fe_mean": float(result["fe_mean"]),
            "raw_length": int(result["raw_length"]),
            "test_length": int(result["test_length"]),
            "settings": {k: v for k, v in cfg.items()},
        }

    if metric == "pe":
        from metrics.permutation_entropy import permutation_entropy
        cfg = METRIC_DEFAULTS["pe"] | task.get("metric_settings", {})
        traj = chaos_map.iterate(num_steps=int(cfg["raw_length"]), discard_transient=int(cfg["transient"]))
        result = permutation_entropy(
            traj, mode=str(cfg["mode"]), m=int(cfg["m"]), tau=int(cfg["tau"]),
            normalized=bool(cfg["normalized"]),
        )
        return {
            "pe_components": [float(v) for v in result["pe_components"]],
            "pe_mean": float(result["pe_mean"]),
            "raw_length": int(result["raw_length"]),
            "test_length": int(result["test_length"]),
            "settings": {k: v for k, v in cfg.items()},
        }

    if metric == "si":
        from metrics.nonperiodicity import scale_index_cwt
        cfg = METRIC_DEFAULTS["si"] | task.get("metric_settings", {})
        traj = chaos_map.iterate(num_steps=int(cfg["raw_length"]), discard_transient=int(cfg["transient"]))
        result = scale_index_cwt(traj, mode=str(cfg["mode"]), wavelet=str(cfg["wavelet"]))
        return {
            "si_components": [float(v) for v in result["si_components"]],
            "si_mean": float(result["si_mean"]),
            "raw_length": int(result["raw_length"]),
            "test_length": int(result["test_length"]),
            "settings": {k: v for k, v in cfg.items()},
        }

    raise ValueError(f"unknown metric: {metric}")


def run_task(task: dict[str, Any], cache_root: Path = CACHE_ROOT) -> dict[str, Any]:
    """Run one task with caching; returns the cached or fresh record."""

    key = task_cache_key(task)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{key}.json"
    if cache_path.exists():
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        if record.get("status") == "ok":
            record["cache_hit"] = True
            return record

    started = time.time()
    record: dict[str, Any] = {
        "cache_key": key,
        "semantic_version": SEMANTIC_VERSION,
        "task": task,
        "reused": False,
    }
    try:
        result = run_metric_uncached(task)
        record.update({"status": "ok", "result": result})
    except Exception as exc:  # noqa: BLE001 - failures must be recorded, not hidden
        record.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    record["elapsed_seconds"] = round(time.time() - started, 3)
    record["cache_hit"] = False
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return record


def run_task_star(args):
    task, cache_root = args
    return run_task(task, Path(cache_root))
