# Maintainer: zhaowenbo
from pathlib import Path
import sys

import numpy as np

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from chaos_maps import NDCS4DMap, Proposed4DBaseMap, Proposed4DEnhancedMap
from metrics.lyapunov import estimate_max_lyapunov_wolf_full_state


def test_wolf_lyapunov_runs_on_proposed_base() -> None:
    result = estimate_max_lyapunov_wolf_full_state(Proposed4DBaseMap(), max_steps=128, transient=16)
    assert np.isfinite(result["lyap_max"])
    assert len(result["distances"]) > 0
    assert isinstance(result["lyap_max"], float)
    assert np.all(np.isfinite(np.asarray(result["lyap_components"], dtype=np.float64)))


def test_wolf_lyapunov_runs_on_proposed_enhanced() -> None:
    result = estimate_max_lyapunov_wolf_full_state(Proposed4DEnhancedMap(), max_steps=128, transient=16)
    assert np.isfinite(result["lyap_max"])
    assert len(result["distances"]) > 0


def test_wolf_lyapunov_returns_required_fields() -> None:
    result = estimate_max_lyapunov_wolf_full_state(Proposed4DBaseMap(), max_steps=96, transient=8)
    required_fields = {
        "lyap_max",
        "lyap_mean",
        "lyap_std",
        "lyap_components",
        "renormalizations",
        "final_distance",
        "is_chaotic",
        "growth_rates",
        "distances",
    }
    assert required_fields.issubset(result.keys())
    assert isinstance(result["lyap_max"], float)
    assert len(result["lyap_components"]) == 4
    assert isinstance(result["growth_rates"], list)
    assert isinstance(result["distances"], list)
    assert result["growth_rates"] or result["distances"]


def test_wolf_lyapunov_runs_on_ndcs() -> None:
    result = estimate_max_lyapunov_wolf_full_state(NDCS4DMap(), max_steps=128, transient=16)
    assert np.isfinite(result["lyap_max"])
    assert len(result["distances"]) > 0


def test_wolf_lyapunov_default_length_runs_without_nan_inf() -> None:
    result = estimate_max_lyapunov_wolf_full_state(Proposed4DBaseMap())
    numeric_values = np.asarray(
        [result["lyap_max"], result["lyap_mean"], result["lyap_std"], result["final_distance"]],
        dtype=np.float64,
    )
    assert np.all(np.isfinite(numeric_values))
    assert np.all(np.isfinite(np.asarray(result["lyap_components"], dtype=np.float64)))


def main() -> None:
    tests = [
        ("wolf_lyapunov_runs_on_proposed_base", test_wolf_lyapunov_runs_on_proposed_base),
        ("wolf_lyapunov_runs_on_proposed_enhanced", test_wolf_lyapunov_runs_on_proposed_enhanced),
        ("wolf_lyapunov_runs_on_ndcs", test_wolf_lyapunov_runs_on_ndcs),
        ("wolf_lyapunov_returns_required_fields", test_wolf_lyapunov_returns_required_fields),
        ("wolf_lyapunov_default_length_runs_without_nan_inf", test_wolf_lyapunov_default_length_runs_without_nan_inf),
    ]
    print("运行 test_lyapunov.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_lyapunov.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
