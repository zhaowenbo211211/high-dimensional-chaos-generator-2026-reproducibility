# Tests

Unit test suite for the HD Chaos project.

## Running

```bash
cd hd_chaos_project
python tests/test_core.py              # 23 tests
python tests/test_analysis.py          # 22 tests
python tests/test_prng.py              # 22 tests — PRNG engine
python tests/test_stream_cipher_371.py # 5 tests — stream cipher (HLS bit-exact)
python tests/test_r1_ly_generation.py  # R1-LY generation + LaTeX
```

Or with pytest:

```bash
PYTHONPATH="src:$PYTHONPATH" pytest tests/
```

## Test files

### test_core.py — Core Module Tests (23 tests)

| Group | Tests | Coverage |
|---|---|---|
| `test_params_*` | 4 | Parameter synthesis: constraints, determinism, multi-n, adaptive adjustment |
| `test_base_map_*` | 7 | Base map: init, shape, range, determinism, seed, physical state, next() |
| `test_enhanced_*` | 5 | Enhanced map: init, shape, range, base-vs-enhanced, iter_count |
| `test_bit_*` | 1 | Bit pack/unpack roundtrip |
| `test_rol_*` | 1 | Cyclic shift roundtrip |
| `test_k_sequence_*` | 2 | Rotation step periodicity |
| `test_rotation_state_*` | 1 | State save/restore |

### test_analysis.py — Analysis Module Tests (22 tests)

| Group | Tests | Coverage |
|---|---|---|
| `test_lyapunov_*` | 4 | LE spectrum positivity, shape, chaos verification, multi-n |
| `test_sensitivity_*` | 3 | Distance growth, result structure, enhanced map compat |
| `test_wolf_*` | 5 | Full-state, embedding, result keys, enhanced vs base |
| `test_orbit_*` | 4 | Step function, Floyd cycle, sampling, digital system |
| `test_distribution_*` | 4 | Histogram, CDF deviation, report, custom dims |

### test_r1_ly_generation.py — R1-LY Generation Test

Generates the canonical 4D 16-bit instance (A_odd=271, B_even=50) and verifies:
- Parity constraints on expansion/coupling sequences
- Cone-field condition (Δ > 0)
- Map iteration correctness
- Chaos verification (all LEs > 0.01)
- LaTeX system equation output

Generated parameters and system:

```
A_odd = 271, B_even = 50
a_j1  = [271, 273, 275]    (odd expansion)
a_jn  = [50, 48, 46]       (even coupling)
b     = 81                 (feedback)
Δ     = 45881.914 > 0      (cone-field condition)
LE    = [5.59, 5.60, 5.65, 2.60]  (all > 0)
```

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

### test_prng.py — Chaos PRNG Tests (22 tests)

| Group | Tests | Coverage |
|---|---|---|
| `test_concat_*` | 1 | 4×16 → 64-bit concatenation |
| `test_rotl64_*` | 2 | ROTL64 correctness and roundtrip |
| `test_post_processor_*` | 4 | Mix output range, determinism, reset, batch |
| `test_seed_from_key_*` | 3 | Key seeding: determinism, uniqueness, processor reset |
| `test_export_*` | 3 | Binary export: raw file, ascii file, mode validation |
| `test_prng_*` | 9 | Factory, next(), generate(), bytes, determinism, seed, state, reset, non-constant |

### test_stream_cipher_371.py — Stream Cipher Verification (5 tests)

Bit-exact stream cipher test using golden_prng. Verifies Python-HLS alignment.

| Test | Description |
|---|---|
| Keystream alignment | Python ciphertext == HLS CSim ciphertext (bit-exact) |
| Encrypt/decrypt | XOR roundtrip recovers original plaintext |
| Different init | Different inits → different keystreams |
| Ciphertext random | Ciphertext is non-zero and differs from plaintext |
| Large transfer | 1 MB keystream generated, non-repeating |

```bash
python tests/test_stream_cipher_371.py
```

### prng_export.py — PRNG Binary Sequence Export Utility

Interactive GUI with R1-LY synthesis, sequence generation, and design report.

| Step | Dialog | Content |
|---|---|---|
| 1 | Parameters | **A_odd** (default 371), B_even=50 (auto), 4 initial values, sequence length |
| — | Auto-generate | Click "Generate" → shows a_j1, a_jn, b, Δ |
| 2 | Confirm | Generated N million bits — save to file? |
| 3 | Mode | Raw binary (.bin, ~12.5 MB) or ASCII text (.txt, ~100 MB) |
| 4 | Path | System file-save dialog |
| 5 | **Report** | Terminal: chaotic LaTeX + enhancer + HW notes (shift-and-add, DSP, pipeline) |

```bash
python tests/prng_export.py
```

### prng_export_371.py — HLS Bit-Exact Export (A_odd=371)

Uses the C golden shared library for 100% hardware-verified output.

| Step | Dialog | Content |
|---|---|---|
| 1 | Parameters | 4 initial values + sequence length (A_odd=371 fixed) |
| 2 | Confirm | Save to file? |
| 3 | Mode | Raw (.bin) or ASCII (.txt) |
| 4 | Path | System file-save dialog |
| 5 | Report | Design report + golden model verification note |

```bash
python tests/prng_export_371.py
```

**Key difference from prng_export.py**: Uses `golden_371.so` (C shared lib) instead of Python `ChaosPRNG` — 100% bit-exact with HLS hardware (100/100 verified against golden_ref_371.txt).

**New features**:
- A_odd input with auto-synthesis (B_even=50, η=0.3 auto)
- Terminal prints chaotic system LaTeX equations with actual parameter values
- Post-export design report: shift-and-add decompositions, pipeline architecture, HW metrics

### Legacy test files

`test_basic.py` and `test_basic_enhanced.py` are pre-refactor legacy files kept for reference. They import from `chaos.R1_LY_Complementary` (still backward-compatible) but should not be used as templates for new tests.

## Writing New Tests

```python
# Use canonical imports
from chaos.core import ComplementaryChaosParams, HighDimChaosMap
from chaos.analysis import verify_chaos, wolf_lyapunov_full

def test_my_feature():
    p = ComplementaryChaosParams(n=4, A_odd=271, B_even=50).synthesize()
    m = HighDimChaosMap(p, word_size=16, seed=42)
    # ... assertions ...
```

**Rules**:
- Each test function starts with `test_`
- Use `assert` for validation (no unittest.TestCase needed)
- No tkinter, no `plt.show()` — tests must run headless
- Keep tests fast (< 10s each when possible)

**Documentation constraint**: When adding a new module under `chaos/`, add corresponding tests and update this README with the test count and coverage description.
