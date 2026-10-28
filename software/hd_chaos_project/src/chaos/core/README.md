# chaos.core — Chaotic Map Implementation

The core layer: parameter synthesis and state iteration of the n-dimensional chaotic map.

## Modules

### params.py — ComplementaryChaosParams

Implements **Algorithm R1-LY** with complementary odd-gain/even-coupling sequences and the strict algebraic Condition (S). Candidates that pass both checks return on the first attempt; otherwise the existing adaptive adjustments run for up to `max_iter=100` attempts.

The strict check requires `A_min > rho` and `discriminant > 0`, where
`rho = sqrt(145)`, `c = sqrt(b**2 + 144)`, and
`discriminant = (A_min - rho)**2 - 4*beta*c`.
The returned `c` and `discriminant` now have these theoretical meanings.
The width-budget formulas separately retain `c_width = sqrt(b**2 + 32)`;
this heuristic quantity is not returned and does not certify covering.
The parameter/return-key interface and adjustment policy are unchanged.

Passing this check verifies the algebraic condition for the paper's real-valued
model. It does not establish the invariant-measure hypotheses, two-band covering,
or chaos of the finite-precision implementation. The stored `mod` and `kappa`
remain unused by synthesis, and no coefficient ceiling of 511 is enforced here.

```python
from chaos.core import ComplementaryChaosParams

synth = ComplementaryChaosParams(n=4, A_odd=271, B_even=50, eta=0.3, tau=0.5)
params = synth.synthesize()
# Returns: {'a_j1': array, 'a_jn': array, 'b': int, 'c': float,
#           'A_min': int, 'beta': float, 'discriminant': float}
```

| Parameter | Constraint | Meaning |
|---|---|---|
| `n` | ≥ 3 | System dimension |
| `A_odd` | odd, ≥ 5 | Base odd gain for expansion sequence |
| `B_even` | even, > 2(n-1) | Base even gain for coupling sequence |
| `eta` | (0, 0.35] | Safety ratio for feedback parameter b |
| `tau` | (0, 1] | Band-width factor for no-wrap budget |

### base_map.py — HighDimChaosMap

Strict fixed-point implementation on $\mathbb{Z}_{2^w}$. Uses delay-chain state management and a hardware-friendly logistic core (no division).

**State layout**: $2n-2$ elements — $x_1,\dots,x_{n-1}$ (physical) + $s_0,\dots,s_{n-2}$ (delay chain, $s_0 = x_n$).

```python
from chaos.core import HighDimChaosMap

m = HighDimChaosMap(params, word_size=16, seed=42)
trajectory = m.iterate(100)     # shape: (100, n)
state = m.next()                # single step, shape: (n,)
m.set_state_from_physical(x0)   # set specific initial condition
```

| Method | Returns | Description |
|---|---|---|
| `iterate(steps)` | `(steps, n)` array | Multi-step iteration |
| `next()` | `(n,)` array | Single step |
| `set_state_from_physical(x)` | — | Set $[x_1,\dots,x_n]$ as internal state |
| `get_physical_state()` | `(n,)` array | Get current $[x_1,\dots,x_n]$ |
| `set_random_state(seed)` | — | Re-randomize internal state |
| `reset(seed)` | — | Alias for set_random_state |

### enhanced_map.py — HighDimChaosMapEnhanced

Inherits `HighDimChaosMap`. After each base iteration, applies **bit-rotation dynamics enhancement** (Algorithm 2):

$x^{(t+1)} = \Pi^{-1}\!\big(\mathrm{ROL}_{nw}(\Pi(x^{(t)}), k_t)\big)$, where $k_t = 1 + (t \bmod (w-1))$.

```python
from chaos.core import HighDimChaosMapEnhanced

m = HighDimChaosMapEnhanced(params, word_size=16, seed=42)
trajectory = m.iterate(100)  # enhanced trajectory
```

Additional methods:

| Method | Description |
|---|---|
| `debug_k_sequence(n)` | Print and return n rotation steps |
| `get_rotation_schedule_state()` | Save {iter_count} for restore |
| `set_rotation_schedule_state(s)` | Restore rotation state |

---

**Dependencies**: `numpy`

## Concrete 4D Instance (A_odd=271, B_even=50, 16-bit)

The parameter synthesis call above returns the following parameter diagnostics:

```
a_j1 = [271, 273, 275]     (odd expansion)
a_jn = [50, 48, 46]        (even coupling)
b    = 81                  (feedback)
c    = 81.884              (theoretical C-block norm bound)
Δ    = 39812.849 > 0       (cone-field condition satisfied)
```

The corresponding real-valued delayed model (not an exact statement of the bit-level update in `base_map.py`):

\[
\left\{
\begin{alignedat}{3}
x_1(i+1) &= 271\,x_1(i) + 50\,x_4(i-2) \\
x_2(i+1) &= 273\,x_2(i) + 48\,x_4(i-1) \\
x_3(i+1) &= 275\,x_3(i) + 46\,x_4(i) \\
x_4(i+1) &= 81\,x_2(i) + 4\bigl(x_3(i)+x_4(i)\bigr)\bigl(1-(x_3(i)+x_4(i))\bigr)
\end{alignedat}
\right.
\]

The odd base gain $A_{\mathrm{odd}}$ lies in the 4D **golden range** $[271,507]$; the other gains are $A_{\mathrm{odd}}+2$ and $A_{\mathrm{odd}}+4$. All 119 base gains in this range pass the strict condition with the fixed coupling vector and default feedback rule.

## Concrete 4D Instance (A_odd=371, B_even=50, 16-bit) — Top Performer

Calling `synthesize()` with `n=4, A_odd=371, B_even=50, eta=0.3, tau=0.5` returns:

```
a_j1 = [371, 373, 375]     (odd expansion)
a_jn = [50, 48, 46]        (even coupling)
b    = 111                 (feedback)
c    = 111.647             (theoretical C-block norm bound)
Δ    = 91701.106 > 0       (cone-field condition satisfied)
```

The corresponding real-valued delayed model (not an exact statement of the bit-level update in `base_map.py`):

\[
\left\{
\begin{alignedat}{3}
x_1(i+1) &= 371\,x_1(i) + 50\,x_4(i-2) \\
x_2(i+1) &= 373\,x_2(i) + 48\,x_4(i-1) \\
x_3(i+1) &= 375\,x_3(i) + 46\,x_4(i) \\
x_4(i+1) &= 111\,x_2(i) + 4\bigl(x_3(i)+x_4(i)\bigr)\bigl(1-(x_3(i)+x_4(i))\bigr)
\end{alignedat}
\right.
\]

**Hardware**: 3-stage pipeline, 100 MHz on xc7z020, 800 MB/s, P&R verified (WNS=+0.549ns).

**Documentation constraint**: If you add/remove/rename a method or change parameter semantics, update this README.

**Next layer**: [chaos.analysis](../analysis/README.md) — Lyapunov, sensitivity, distribution, orbit diagnostics.
