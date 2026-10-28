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

from utils.fixed16 import U16_MOD, float_to_u16, float_vec_to_u16, u16_to_float, u16_vec_to_float


def test_common_init_float_u16_roundtrip() -> None:
    values = (0.1, 0.2, 0.3, 0.4)
    u16 = float_vec_to_u16(values)
    expected = np.asarray([6553, 13107, 19660, 26214], dtype=np.int64)
    assert np.array_equal(u16, expected)

    restored = u16_vec_to_float(u16)
    assert restored.shape == (4,)
    assert np.allclose(restored, expected / U16_MOD)


def test_scalar_wrap_and_roundtrip() -> None:
    val = float_to_u16(1.1)
    assert val == 6553
    assert np.isclose(u16_to_float(val), 6553 / U16_MOD)


def main() -> None:
    tests = [
        ("common_init_float_u16_roundtrip", test_common_init_float_u16_roundtrip),
        ("scalar_wrap_and_roundtrip", test_scalar_wrap_and_roundtrip),
    ]
    print("运行 test_fixed16.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_fixed16.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
