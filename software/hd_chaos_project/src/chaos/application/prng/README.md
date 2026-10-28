# chaos.application.prng — Chaos-Based PRNG

A 64-bit pseudo-random number generator driven by the enhanced 4D chaotic map with lightweight ROTL-XOR post-processing.

## Architecture

```
x^{(t)} = G_enhanced(x^{(t-1)})                           ← enhanced 4D chaos
u_t     = x1(t) ∥ x2(t) ∥ x3(t) ∥ x4(t)                 ← 4×16 → 64-bit concat
y_t     = u_t ⊕ ROTL64(u_{t-1}, 17) ⊕ ROTL64(u_{t-2}, 43) ← inter-cycle mixing
output  = y_t                                             ← 64-bit PRNG block
```

## Hardware Cost

- Direct state concatenation (wiring only)
- Two 64-bit delay registers
- Two fixed circular rotations (17 and 43 bits)
- A small number of 64-bit XOR gates

No nonlinear scrambler, hash extractor, or multi-stage correction needed.

## Modules

### post_processor.py — PostProcessor

Lightweight word-level mixing stage with delay-line ROTL-XOR.

```python
from chaos.application.prng import PostProcessor

pp = PostProcessor(word_bits=16)
y = pp.mix(np.array([x1, x2, x3, x4], dtype=np.int64))  # → 64-bit int
outputs = pp.mix_batch(states)  # states.shape == (N, 4) → (N,) uint64
pp.reset()  # clear delay registers
```

| Method | Description |
|---|---|
| `mix(state_4d)` | Mix one 4D state → 64-bit word |
| `mix_batch(states)` | Mix N states → array of N uint64 |
| `reset()` | Clear delay registers `u_{t-1}`, `u_{t-2}` |

### prng.py — ChaosPRNG

Full PRNG engine combining enhanced chaos with post-processing.

```python
from chaos.application.prng import ChaosPRNG

# Factory: create from R1-LY parameters
prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)

# Initialization from key
prng.seed_from_key(0xDEADBEEFCAFE)      # deterministic initial state from key

# Generate outputs
word  = prng.next()                     # single 64-bit word
words = prng.generate(1000)             # 1000 words, shape (1000,) uint64
data  = prng.generate_bytes(1024)       # 1024 bytes

# Binary sequence export to file
prng.export_binary_sequence("output.bin", length=10**8, mode="raw")
prng.export_binary_sequence("output.txt", length=10**8, mode="ascii")

# State management
prng.set_state(x1, x2, x3, x4)          # set internal 4D state
state = prng.get_state()                # [x1, x2, x3, x4]
prng.reset(seed=42)                     # full reset
```

| Method | Returns | Description |
|---|---|---|
| `next()` | `int` | Single 64-bit mixed output |
| `generate(n, progress_interval=0)` | `np.ndarray (n,) uint64` | n consecutive outputs; `progress_interval` > 0 prints progress + enables Ctrl+C |
| `generate_bytes(n)` | `bytes` | n bytes of PRNG data |
| `seed_from_key(key)` | — | Derive state from integer key; auto warmup 1000 |
| `export_binary_sequence(path, length, mode)` | — | Export bits to file (`"raw"` or `"ascii"`, default 10⁸ bits) |
| `from_r1_ly(A_odd, B_even, seed, warmup=1000)` | `ChaosPRNG` | Factory from R1-LY params; `warmup=0` to skip transient discard |
| `set_state(x1,x2,x3,x4)` | — | Set state + auto warmup 1000; all-zero → defaults [1000,2000,3000,4000] |
| `get_state()` | `np.ndarray (4,)` | Current 4D state (after warmup) |
| `reset(seed)` | — | Reset map + post-processor (call `set_state` after to re-warmup) |

### Binary sequence export

```python
prng.export_binary_sequence(filepath, length=100_000_000, mode="raw")
```

| mode | Output | Size for 10^8 bits |
|---|---|---|
| `"raw"` | Packed bytes (8 bits/byte) | 12.5 MB |
| `"ascii"` | '0'/'1' characters (1 byte/bit) | 100 MB |

### Interactive export utility

`prng_export.py` — GUI script with 4-step dialog workflow:

```
[Dialog 1] Parameters: 4 initial values + sequence length
[Dialog 2] Confirm save
[Dialog 3] Choose mode: raw / ascii
[Dialog 4] Choose file path
```

```bash
# From project root
python src/chaos/application/prng/prng_export.py

# Or via module
python -m chaos.application.prng.prng_export
```

## Canonical Instance

```python
from chaos.application.prng import ChaosPRNG

prng = ChaosPRNG.from_r1_ly(A_odd=271, B_even=50, seed=42)
words = prng.generate(1000)  # 1000 × 64-bit = 8000 bytes
```

Generated system (from `test_r1_ly_generation.py`):

\[
\begin{cases}
x_1(i+1) = 271 x_1(i) + 50 x_4(i-2) \\
x_2(i+1) = 273 x_2(i) + 48 x_4(i-1) \\
x_3(i+1) = 275 x_3(i) + 46 x_4(i) \\
x_4(i+1) = 81 x_2(i) + 4(x_3(i)+x_4(i))(1-x_3(i)-x_4(i))
\end{cases}
\]

With bit-rotation enhancement: $k_t = 1 + (t \bmod 15)$.

---

**Dependencies**: `chaos.core`, `numpy`

**Documentation constraint**: If the mixing formula, rotation constants, or output width change, update this README.

**Next layer**: tests → `tests/test_prng.py` (16 tests).
