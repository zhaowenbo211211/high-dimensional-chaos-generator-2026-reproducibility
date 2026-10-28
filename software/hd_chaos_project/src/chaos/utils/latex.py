#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
LaTeX generation utilities for chaotic system equations.
"""

import numpy as np
from typing import Callable, Optional, Any, Dict


def make_F(n: int, a11: np.ndarray, an: np.ndarray, b: int, *,
           mod1: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a discrete map F with internal x_n delay queue and mod-1 output.

    Equations:
      x_j(i+1) = a_{j,1} x_j(i) + a_{j,n} x_n(i - (n-1-j)),   j=1..n-1
      x_n(i+1) = b x_{n-2}(i) + 4(x_{n-1}(i)+x_n(i))(1 - (x_{n-1}(i)+x_n(i)))
    """
    assert n >= 3, "n must be >= 3"
    a11 = np.asarray(a11, dtype=float)
    assert a11.shape == (n - 1,)
    an = np.asarray(an, dtype=float)
    assert an.shape == (n - 1,)
    b = int(b)

    hist = np.zeros(n - 2, dtype=float) if n - 2 > 0 else np.zeros(0, dtype=float)

    def F(x: np.ndarray) -> np.ndarray:
        nonlocal hist
        x = np.asarray(x, dtype=float)
        if x.shape != (n,):
            raise ValueError(f"x must have shape ({n},)")
        x_mod = np.mod(x, 1.0)

        x_next_raw = np.empty_like(x_mod)
        for j in range(1, n - 1):
            delayed_xn = hist[j - 1] if (n - 2) >= 1 else 0.0
            x_next_raw[j - 1] = a11[j - 1] * x_mod[j - 1] + an[j - 1] * delayed_xn

        x_next_raw[n - 2] = a11[n - 2] * x_mod[n - 2] + an[n - 2] * x_mod[n - 1]

        z = x_mod[n - 2] + x_mod[n - 1]
        x_next_raw[n - 1] = b * x_mod[n - 3] + 4.0 * z * (1.0 - z)

        x_next = np.mod(x_next_raw, 1.0)

        if n - 2 > 0:
            hist = np.roll(hist, -1)
            hist[-1] = x_mod[n - 1]

        return x_next

    F._meta = {
        "n": int(n),
        "a11": np.array(a11, dtype=float),
        "an": np.array(an, dtype=float),
        "b": int(b),
        "mod1": bool(mod1),
        "structure": "delayed-n-map-xn-history"
    }
    return F


def _fmt_num(v: float) -> str:
    if np.isclose(v, round(v)):
        return str(int(round(v)))
    return f"{v:.6g}"


def latex_system_like_figure_full(F, with_mod: bool = True) -> str:
    """Generate LaTeX system of equations with numerical parameter values."""
    meta = getattr(F, "_meta", None)
    if meta is None:
        raise ValueError("F lacks _meta; create it via make_F(...)")
    n = int(meta["n"])
    if n < 3:
        raise ValueError("n must be >= 3")
    a11 = np.asarray(meta["a11"], dtype=float)
    an = np.asarray(meta["an"], dtype=float)
    b = float(meta["b"])
    mod_enabled = bool(meta.get("mod1", True)) and bool(with_mod)
    MOD = r"\bmod 1" if mod_enabled else ""

    lines = []
    for j in range(1, n):
        if j <= n - 2:
            k = (n - 1) - j
            rhs = rf"{_fmt_num(a11[j-1])}\,x_{j}(i) + {_fmt_num(an[j-1])}\,x_{n}(i-{k})"
        else:
            rhs = rf"{_fmt_num(a11[j-1])}\,x_{j}(i) + {_fmt_num(an[j-1])}\,x_{n}(i)"
        lines.append(rf"& x_{j}(i+1)      & ={{}} & {rhs} &\quad& {MOD} \\")

    rhs_nm1 = rf"{_fmt_num(a11[n-2])}\,x_{n-1}(i) + {_fmt_num(an[n-2])}\,x_{n}(i)"
    lines[-1] = rf"& x_{n-1}(i+1)  & ={{}} & {rhs_nm1} &\quad& {MOD} \\"

    rhs_n = rf"{_fmt_num(b)}\,x_{{{n-2}}}(i) + 4\!\big(x_{{{n-1}}}(i)+x_{{{n}}}(i)\big)\!\big(1-\big(x_{{{n-1}}}(i)+x_{{{n}}}(i)\big)\big)"
    lines.append(rf"& x_{{{n}}}(i+1)    & ={{}} & {rhs_n} &\quad& {MOD}")

    body = "\n".join(lines)
    tex = (
        "\\[\n"
        "\\left\\{%\n"
        "\\begin{alignedat}{3}\n"
        + body +
        "\n\\end{alignedat}\n"
        "\\right.\n"
        "\\]\n"
    )
    return tex


def latex_system_like_figure(F, with_mod=True) -> str:
    """Generate LaTeX system matching the paper figure layout."""
    meta = getattr(F, "_meta", None)
    if meta is None:
        raise ValueError("F lacks _meta; create it via make_F(...)")
    n = int(meta.get("n", 0))
    if n < 3:
        raise ValueError("n must be >= 3 in F._meta")
    mod_enabled = bool(meta.get("mod1", True)) and bool(with_mod)
    MOD = r"\bmod 1" if mod_enabled else ""

    lines = [
        r"& x_1(i+1)      & ={} & a_{1,1}x_1(i) + a_{1,n}\,x_n\!\big(i-(n-2)\big) &\quad& %s\\" % MOD,
        r"& x_2(i+1)      & ={} & a_{2,1}x_2(i) + a_{2,n}\,x_n\!\big(i-(n-3)\big) &\quad& %s\\" % MOD,
        r"&                &     & \vdots                                          &     &        \\",
        r"& x_{n-1}(i+1)  & ={} & a_{n-1,1}x_{n-1}(i) + a_{n-1,n}\,x_n(i)         &\quad& %s\\" % MOD,
        r"& x_n(i+1)      & ={} & b\,x_{n-2}(i) + 4\!\big(x_{n-1}(i)+x_n(i)\big)\!\big(1-\big(x_{n-1}(i)+x_n(i)\big)\big) &\quad& %s" % MOD,
    ]
    body = "\n".join(lines)
    tex = r"""\[
\left\{
\begin{alignedat}{3}
%s
\end{alignedat}
\right.
\]""" % body
    return tex


def latex_from_F(F, *, show_values: bool = True, align_env: str = "aligned") -> str:
    """Generate LaTeX equations from F metadata."""
    meta: Optional[Dict[str, Any]] = getattr(F, "_meta", None)
    if meta is None:
        raise ValueError("F lacks _meta; create it via make_F(...)")

    n = meta["n"]
    a11 = meta["a11"]
    an = meta["an"]
    b = meta["b"]
    mod1 = meta["mod1"]

    line1 = (r"x_{j}(i\!+\!1) \;=\; "
             r"%s\!\left(a_{j,1}\,x_{j}(i) \;+\; a_{j,n}\,x_{n}\!\big(i-(n\!-\!1\!-\!j)\big)\right)"
             ) % (r"\bmod 1\,\," if mod1 else "")
    line2 = (r"x_{n-1}(i\!+\!1) \;=\; "
             r"%s\!\left(a_{n-1,1}\,x_{n-1}(i) \;+\; a_{n-1,n}\,x_{n}(i)\right)"
             ) % (r"\bmod 1\,\," if mod1 else "")
    line3 = (r"x_{n}(i\!+\!1) \;=\; %s\!\Big(b\,x_{n-2}(i) \;+\; 4\,z(i)\big(1-z(i)\big)\Big),"
             r"\qquad z(i)=x_{n-1}(i)+x_{n}(i)"
             ) % (r"\bmod 1\,\," if mod1 else "")

    lines = [
        r"\begin{%s}" % align_env,
        r"\text{for } j=1,\dots,n-2:\quad " + line1 + r"\\",
        line2 + r"\\",
        line3,
        r"\end{%s}" % align_env,
    ]

    if show_values:
        def arr_to_tex(arr):
            return r"\big(" + r",\,".join(f"{v:g}" for v in arr) + r"\big)"

        lines.extend([
            r"\par\noindent\textbf{Parameters:}",
            r"\(",
            r"\;n=%d,\quad b=%g,\quad \left(a_{1,1},\dots,a_{n-1,1}\right)=%s,\quad \left(a_{1,n},\dots,a_{n-1,n}\right)=%s,"
            r"\)" % (n, b, arr_to_tex(a11), arr_to_tex(an)),
            r"\par\noindent\textbf{Convention:}\; x_n\text{ keeps an internal delay queue of length }n-2\text{ (init zeros).}"
        ])

    return "\n".join(lines)
