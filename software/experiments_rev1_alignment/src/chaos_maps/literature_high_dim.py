# Maintainer: zhaowenbo
"""High-dimensional literature maps reproduced under shared 16-bit semantics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from chaos_maps.base import ChaosMapNDBase
from utils.fixed16 import U16_MOD, float_vec_to_u16, u16_vec_to_float


DEFAULT_ZANG_DIAGONAL = 2.0
DEFAULT_ZANG_ROW_COUPLING_BUDGET = 0.75


def get_linear_modular_initial_state_u16(n: int) -> np.ndarray:
    """Return the requested comparison initial state `[100, 200, ..., 100*n]`."""

    if int(n) < 1:
        raise ValueError("n must be positive.")
    return (100 * np.arange(1, int(n) + 1, dtype=np.int64)) % U16_MOD


def _as_state_u16(n: int, initial_state_u16: Iterable[int] | None) -> np.ndarray:
    """Return the comparison state vector using `x_i(0)=100*i` by default."""

    if initial_state_u16 is None:
        return get_linear_modular_initial_state_u16(int(n))
    state = np.asarray(list(initial_state_u16), dtype=np.int64) % U16_MOD
    if state.shape != (int(n),):
        raise ValueError(f"Initial state must be a {int(n)}-element vector.")
    return state


def build_yao_complex_pascal_matrix(n: int) -> np.ndarray:
    """Build the deterministic complex Pascal matrix used for Yao et al. 2025.

    The paper configures a parametric Pascal matrix from positive integer
    border values, multiplies the first row by the imaginary unit, and forms
    the complex parameter matrix from that Pascal matrix. The paper's 4D
    example has its own concrete random vector; for a reproducible 4D--12D
    comparison we use the same deterministic border rule for every dimension.
    """

    dimension = int(n)
    if dimension < 2:
        raise ValueError("n must be at least 2.")
    pascal = np.zeros((dimension, dimension), dtype=np.complex128)
    pascal[0, :] = np.arange(1, dimension + 1, dtype=np.float64)
    pascal[:, 0] = np.arange(1, dimension + 1, dtype=np.float64)
    for row in range(1, dimension):
        for col in range(1, dimension):
            pascal[row, col] = pascal[row - 1, col] + pascal[row, col - 1]
    pascal[0, :] *= 1j
    return pascal @ pascal


def build_yao_complex_vector(n: int) -> np.ndarray:
    """Build a fixed complex offset vector for Yao et al.'s `v` parameter."""

    dimension = int(n)
    indexes = np.arange(1, dimension + 1, dtype=np.float64)
    real = np.mod(0.137 * indexes + 0.019 * indexes**2, 1.0)
    imag = np.mod(0.071 * indexes + 0.023 * indexes**2, 1.0)
    return real + 1j * imag


def _validate_zero_imaginary_initial_state(n: int, initial_imag_u16: Iterable[int] | None) -> np.ndarray:
    """Return the mandated all-zero imaginary initial state for complex maps."""

    if initial_imag_u16 is None:
        return np.zeros(int(n), dtype=np.int64)
    imag = np.asarray(list(initial_imag_u16), dtype=np.int64) % U16_MOD
    if imag.shape != (int(n),):
        raise ValueError(f"initial_imag_u16 must have shape ({int(n)},).")
    if np.any(imag != 0):
        raise ValueError("Complex literature-map imaginary initial state must be fixed to zero.")
    return np.zeros(int(n), dtype=np.int64)


def build_zang_linear_modular_matrix(
    n: int,
    diagonal: float = DEFAULT_ZANG_DIAGONAL,
    row_coupling_budget: float = DEFAULT_ZANG_ROW_COUPLING_BUDGET,
) -> np.ndarray:
    """Build a deterministic strictly over-one diagonally dominant matrix.

    Zang et al. prove chaos for `x(k+1)=A x(k) mod 1` when `A` has positive
    diagonal entries and is strictly over-one diagonally dominant. The paper
    gives concrete 3D and 5D matrices rather than a full 4D--12D parameter
    table, so this helper generates reproducible small off-diagonal couplings
    and scales each row to keep `|a_ii| - sum_j!=i |a_ij| > 1`.
    """

    dimension = int(n)
    if dimension < 2:
        raise ValueError("n must be at least 2.")
    diagonal_value = float(diagonal)
    if diagonal_value <= 1.0:
        raise ValueError("diagonal must be greater than 1.")

    max_budget = diagonal_value - 1.0
    budget = min(float(row_coupling_budget), 0.75 * max_budget)
    if budget <= 0.0:
        raise ValueError("row_coupling_budget must be positive.")

    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    np.fill_diagonal(matrix, diagonal_value)

    for row in range(dimension):
        raw_values: list[float] = []
        columns: list[int] = []
        for col in range(dimension):
            if row == col:
                continue
            magnitude = 0.05 + 0.01 * float((7 * (row + 1) + 11 * (col + 1)) % 26)
            sign = -1.0 if ((row + 1) * (col + 2) + row + col) % 2 else 1.0
            raw_values.append(sign * magnitude)
            columns.append(col)

        row_abs_sum = float(np.sum(np.abs(raw_values)))
        scale = min(1.0, budget / max(row_abs_sum, 1e-15))
        for col, value in zip(columns, raw_values):
            matrix[row, col] = float(value) * scale

    return matrix


class ZangLinearModularMap(ChaosMapNDBase):
    """16-bit reproduction of Zang et al.'s linear modular n-D system.

    The continuous paper form is `x(k+1)=A x(k) mod 1`. In this project, each
    state is stored as a 16-bit integer. Each step converts the current state
    to `[0,1)` floats, applies the linear modular equation, and quantizes the
    wrapped result back to the shared u16 domain.
    """

    name = "ZangLinearModularMap"

    def __init__(
        self,
        n: int,
        diagonal: float = DEFAULT_ZANG_DIAGONAL,
        row_coupling_budget: float = DEFAULT_ZANG_ROW_COUPLING_BUDGET,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        self.diagonal = float(diagonal)
        self.row_coupling_budget = float(row_coupling_budget)
        self.matrix = (
            np.asarray(matrix, dtype=np.float64)
            if matrix is not None
            else build_zang_linear_modular_matrix(
                n=int(n),
                diagonal=self.diagonal,
                row_coupling_budget=self.row_coupling_budget,
            )
        )
        if self.matrix.shape != (int(n), int(n)):
            raise ValueError(f"matrix must have shape ({int(n)}, {int(n)}).")

        super().__init__(n=int(n), initial_state_u16=initial_state_u16)
        if initial_state_float is not None:
            self.reset(initial_state_float=initial_state_float)

    def step(self) -> np.ndarray:
        """Advance the linear modular map and return the float state."""

        state_float = u16_vec_to_float(self.state_u16)
        next_float = np.mod(self.matrix @ state_float, 1.0)
        self.state_u16 = float_vec_to_u16(next_float)
        return u16_vec_to_float(self.state_u16)

    def row_abs_sums(self) -> np.ndarray:
        """Return off-diagonal absolute row sums for diagnostics."""

        offdiag = np.array(self.matrix, copy=True)
        np.fill_diagonal(offdiag, 0.0)
        return np.sum(np.abs(offdiag), axis=1)

    def strict_over_one_margins(self) -> np.ndarray:
        """Return `|a_ii| - sum_j!=i |a_ij|` for every row."""

        return np.abs(np.diag(self.matrix)) - self.row_abs_sums()


class YaoComplexPascalMap(ChaosMapNDBase):
    """Yao et al. 2025 n-D complex chaotic system with CFOLD-1 semantics.

    The internal state is complex-valued with n real parts and n imaginary
    parts. The public trajectory returned to this project's metric pipeline is
    the n-component real part, while the imaginary part is retained as an
    auxiliary state initialized to zero as required by the comparison protocol.
    """

    name = "YaoComplexPascalMap"

    def __init__(
        self,
        n: int,
        initial_state_u16: Iterable[int] | None = None,
        initial_imag_u16: Iterable[int] | None = None,
        matrix: np.ndarray | None = None,
        vector: np.ndarray | None = None,
    ) -> None:
        self.matrix = np.asarray(
            build_yao_complex_pascal_matrix(int(n)) if matrix is None else matrix,
            dtype=np.complex128,
        )
        self.vector = np.asarray(
            build_yao_complex_vector(int(n)) if vector is None else vector,
            dtype=np.complex128,
        )
        if self.matrix.shape != (int(n), int(n)):
            raise ValueError(f"matrix must have shape ({int(n)}, {int(n)}).")
        if self.vector.shape != (int(n),):
            raise ValueError(f"vector must have shape ({int(n)},).")
        self.imag_u16 = np.zeros(int(n), dtype=np.int64)
        self._initial_imag_u16 = _validate_zero_imaginary_initial_state(int(n), initial_imag_u16)
        super().__init__(n=int(n), initial_state_u16=_as_state_u16(int(n), initial_state_u16))

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        initial_imag_u16: Iterable[int] | None = None,
    ) -> None:
        """Reset real and imaginary states."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)
        imag = (
            _validate_zero_imaginary_initial_state(self.n, initial_imag_u16)
            if initial_imag_u16 is not None
            else np.asarray(self._initial_imag_u16, dtype=np.int64) % U16_MOD
        )
        if imag.shape != (self.n,):
            raise ValueError(f"Initial imaginary state must be a {self.n}-element vector.")
        self.imag_u16 = imag.astype(np.int64)

    def get_imag_u16(self) -> np.ndarray:
        """Return a copy of the imaginary auxiliary state."""

        return self.imag_u16.copy()

    def step(self) -> np.ndarray:
        """Advance `z(k+1)=CFOLD_1(A z(k)+v)` and return the real part."""

        z = u16_vec_to_float(self.state_u16) + 1j * u16_vec_to_float(self.imag_u16)
        raw_next = self.matrix @ z + self.vector
        next_z = np.mod(np.real(raw_next), 1.0) + 1j * np.mod(np.imag(raw_next), 1.0)
        self.state_u16 = float_vec_to_u16(np.real(next_z))
        self.imag_u16 = float_vec_to_u16(np.imag(next_z))
        return u16_vec_to_float(self.state_u16)


class SineMapLatticeMemristiveMap(ChaosMapNDBase):
    """Sine-map lattice with discrete memristor coupling.

    For requested comparison dimension `n`, this reproduction uses `m=n`
    lattice sites as the public state. The memristor charge `q_i` and delay
    state `y_i=x_i(k-1)` are internal auxiliary states initialized to zero.
    """

    name = "SineMapLatticeMemristiveMap"

    def __init__(
        self,
        n: int,
        a: float = 1.0,
        psi: float = 0.99,
        k: float = 1.5,
        w: float = np.pi,
        initial_state_u16: Iterable[int] | None = None,
        initial_q_u16: Iterable[int] | None = None,
        initial_y_u16: Iterable[int] | None = None,
    ) -> None:
        self.a = float(a)
        self.psi = float(psi)
        self.k = float(k)
        self.w = float(w)
        self.q_u16 = np.zeros(int(n), dtype=np.int64)
        self.y_u16 = np.zeros(int(n), dtype=np.int64)
        self._initial_q_u16 = (
            np.zeros(int(n), dtype=np.int64)
            if initial_q_u16 is None
            else np.asarray(list(initial_q_u16), dtype=np.int64) % U16_MOD
        )
        self._initial_y_u16 = (
            np.zeros(int(n), dtype=np.int64)
            if initial_y_u16 is None
            else np.asarray(list(initial_y_u16), dtype=np.int64) % U16_MOD
        )
        if self._initial_q_u16.shape != (int(n),) or self._initial_y_u16.shape != (int(n),):
            raise ValueError(f"auxiliary states must have shape ({int(n)},).")
        super().__init__(n=int(n), initial_state_u16=_as_state_u16(int(n), initial_state_u16))

    def reset(
        self,
        initial_state_u16: Iterable[int] | None = None,
        initial_state_float: Iterable[float] | None = None,
        initial_q_u16: Iterable[int] | None = None,
        initial_y_u16: Iterable[int] | None = None,
    ) -> None:
        """Reset lattice state and memristor/delay auxiliary states."""

        super().reset(initial_state_u16=initial_state_u16, initial_state_float=initial_state_float)
        q = self._initial_q_u16 if initial_q_u16 is None else np.asarray(list(initial_q_u16), dtype=np.int64)
        y = self._initial_y_u16 if initial_y_u16 is None else np.asarray(list(initial_y_u16), dtype=np.int64)
        q = np.asarray(q, dtype=np.int64) % U16_MOD
        y = np.asarray(y, dtype=np.int64) % U16_MOD
        if q.shape != (self.n,) or y.shape != (self.n,):
            raise ValueError(f"auxiliary states must have shape ({self.n},).")
        self.q_u16 = q
        self.y_u16 = y

    def step(self) -> np.ndarray:
        """Advance the memristive sine-map lattice and return x components."""

        x = u16_vec_to_float(self.state_u16)
        q = u16_vec_to_float(self.q_u16)
        y = u16_vec_to_float(self.y_u16)
        neighbor_x = np.roll(x, -1)
        neighbor_y = np.roll(y, -1)
        next_x = self.a * self.psi * np.sin(self.w * x)
        next_x += (1.0 - self.psi) * self.k * (neighbor_x - neighbor_y) * np.sin(np.pi * q)
        next_q = q + neighbor_x - neighbor_y
        self.y_u16 = self.state_u16.copy()
        self.state_u16 = float_vec_to_u16(next_x)
        self.q_u16 = float_vec_to_u16(next_q)
        return u16_vec_to_float(self.state_u16)


class CyclicSymmetricLogisticMap(ChaosMapNDBase):
    """n-D cyclic symmetric map constructed from 1D logistic maps.

    This follows the paper's general construction
    `F(X)=(f(x_2), f(x_3), ..., f(x_1))` and uses logistic seed maps
    `f_i(x)=a_i*x*(1-x)` in the expanded n-D setting.
    """

    name = "CyclicSymmetricLogisticMap"

    def __init__(
        self,
        n: int,
        parameters: Iterable[float] | None = None,
        initial_state_u16: Iterable[int] | None = None,
    ) -> None:
        if parameters is None:
            base = np.asarray([3.99, 3.97, 3.95], dtype=np.float64)
            self.parameters = np.resize(base, int(n)).astype(np.float64)
        else:
            self.parameters = np.asarray(list(parameters), dtype=np.float64)
        if self.parameters.shape != (int(n),):
            raise ValueError(f"parameters must have shape ({int(n)},).")
        if np.any((self.parameters <= 0.0) | (self.parameters > 4.0)):
            raise ValueError("logistic parameters must lie in (0, 4].")
        super().__init__(n=int(n), initial_state_u16=_as_state_u16(int(n), initial_state_u16))

    def step(self) -> np.ndarray:
        """Advance the cyclic symmetric logistic map."""

        x = u16_vec_to_float(self.state_u16)
        source = np.roll(x, -1)
        params = np.roll(self.parameters, -1)
        next_x = params * source * (1.0 - source)
        self.state_u16 = float_vec_to_u16(next_x)
        return u16_vec_to_float(self.state_u16)
