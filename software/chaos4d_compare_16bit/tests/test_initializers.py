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

from utils.initializers import (
    COMMON_INIT_FLOAT,
    get_common_initial_state_float,
    get_common_initial_state_u16,
    validate_non_singular_init,
)


def test_common_initializers() -> None:
    assert COMMON_INIT_FLOAT == (0.1, 0.2, 0.3, 0.4)
    assert np.allclose(get_common_initial_state_float(), np.asarray(COMMON_INIT_FLOAT))
    assert np.array_equal(get_common_initial_state_u16(), np.asarray([6553, 13107, 19660, 26214]))


def test_validate_non_singular_init_placeholder() -> None:
    assert validate_non_singular_init(COMMON_INIT_FLOAT) is True


def main() -> None:
    tests = [
        ("common_initializers", test_common_initializers),
        ("validate_non_singular_init_placeholder", test_validate_non_singular_init_placeholder),
    ]
    print("运行 test_initializers.py")
    for name, test_func in tests:
        test_func()
        print(f"[PASS] {name}")
    print(f"test_initializers.py 完成，共通过 {len(tests)} 项测试。")


if __name__ == "__main__":
    main()
