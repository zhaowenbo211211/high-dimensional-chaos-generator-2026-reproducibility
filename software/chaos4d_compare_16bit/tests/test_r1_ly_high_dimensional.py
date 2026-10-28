# Maintainer: zhaowenbo
from pathlib import Path
import sys

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from utils.r1ly import synthesize_r1_ly_params


def test_r1_ly_synthesis_supports_dimensions_4_to_12() -> None:
    for dimension in range(4, 13):
        params = synthesize_r1_ly_params(dimension)
        assert params["n"] == dimension
        assert len(params["a_linear"]) == dimension - 1
        assert len(params["a_coupling"]) == dimension - 1
        assert params["condition_s_passed"] is True
        assert params["no_wrap_passed"] is True
        assert min(params["a_coupling"]) > 0
        assert params["A_odd_final"] % 2 == 1
        assert params["B_even_final"] % 2 == 0


def test_r1_ly_synthesis_auto_repairs_small_even_coupling_start() -> None:
    params = synthesize_r1_ly_params(12, B_even=20)
    assert params["no_wrap_passed"] is True
    assert min(params["a_coupling"]) > 0
    assert params["B_even_final"] >= 22
