# Maintainer: zhaowenbo
from pathlib import Path
import sys

try:
    from project_bootstrap import ensure_project_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_bootstrap import ensure_project_root

ensure_project_root(__file__)

from utils.r1ly import generate_r1ly_params_4d


def test_generate_r1ly_params_4d_defaults() -> None:
    params = generate_r1ly_params_4d()

    assert params["n"] == 4
    assert params["A_odd"] == 371
    assert params["B_even"] == 50
    assert params["Delta"] == 2
    assert params["eta"] == 61
    assert params["a_linear"] == [371, 373, 375]
    assert params["a_coupling"] == [50, 48, 46]
    assert params["b"] == 111


def test_generate_r1ly_params_4d_validates_parity() -> None:
    try:
        generate_r1ly_params_4d(A_odd=370)
    except ValueError:
        pass
    else:
        raise AssertionError("A_odd 为偶数时应抛出 ValueError。")

    try:
        generate_r1ly_params_4d(B_even=51)
    except ValueError:
        pass
    else:
        raise AssertionError("B_even 为奇数时应抛出 ValueError。")


def main() -> None:
    tests = [
        ("generate_r1ly_params_4d_defaults", test_generate_r1ly_params_4d_defaults),
        ("generate_r1ly_params_4d_validates_parity", test_generate_r1ly_params_4d_validates_parity),
    ]
    print("运行 test_r1ly.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_r1ly.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
