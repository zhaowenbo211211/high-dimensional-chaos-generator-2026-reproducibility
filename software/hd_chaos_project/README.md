# HD Chaos Project

> Historical project documentation: claims, configurations and performance figures below describe prior work and have not been newly verified. The public candidate excludes the complete FPGA project and backend artifacts; see `../../docs/HARDWARE.md` for scope. The paper treats throughput as an estimated maximum.

High-dimensional discrete chaotic map generator for lightweight cryptography on resource-constrained devices.

**Historical hardware record**: the earlier report stated 100 MHz on xc7z020 and an estimated maximum of 800 MB/s with P&R results (WNS=+0.549ns, 925/925 nets routed); this package did not rerun or reassess that result.

## Mathematical Foundation

The general $n$-dimensional delayed discrete-time system on $[0,1)^n$ (mod 1):

$$
\begin{cases}
x_1(i+1) = a_{1,1}\,x_1(i) + a_{1,n}\,x_n\!\big(i-(n-2)\big),\\[0.5mm]
x_2(i+1) = a_{2,1}\,x_2(i) + a_{2,n}\,x_n\!\big(i-(n-3)\big),\\[0.5mm]
\hspace{16mm}\vdots\\[0.5mm]
x_{n-1}(i+1) = a_{n-1,1}\,x_{n-1}(i) + a_{n-1,n}\,x_n(i),\\[1mm]
x_n(i+1) = b\,x_{n-2}(i) + 4\bigl(x_{n-1}(i)+x_n(i)\bigr)\Bigl(1-\bigl(x_{n-1}(i)+x_n(i)\bigr)\Bigr),
\end{cases}
$$

where $a_{j,1},a_{j,n}\in\mathbb{Z}_{>0}$ and $b\in\mathbb{Z}_{>0}$ are integer parameters from R1-LY synthesis.

### Canonical 4D Instance (A_odd=371, B_even=50, 16-bit)

All operations modulo $2^{16}$. Verified Lyapunov exponents all > 0, $\Delta = 99917.2$.

$$
\begin{cases}
x_1(i+1) = 371\,x_1(i) + 50\,x_4(i-2),\\[0.5mm]
x_2(i+1) = 373\,x_2(i) + 48\,x_4(i-1),\\[0.5mm]
x_3(i+1) = 375\,x_3(i) + 46\,x_4(i),\\[1mm]
x_4(i+1) = 111\,x_2(i) + 4\bigl(x_3(i)+x_4(i)\bigr)\Bigl(1-\bigl(x_3(i)+x_4(i)\bigr)\Bigr),
\end{cases}
$$

## Architecture

### R1-LY Parameter Synthesis

| Parameter | Constraint | Meaning |
|---|---|---|
| `n` | ≥ 3 | System dimension |
| `A_odd` | odd, ≥ 5 | Base odd gain for expansion sequence $a_{j,1}$ |
| `B_even` | even, > 2(n−1) | Base even gain for coupling sequence $a_{j,n}$ |
| `eta` | (0, 0.35] | Safety ratio for feedback $b = \max(2, \lfloor\eta A_{min}\rfloor)$ |
| `tau` | (0, 1] | Band-width factor for no-wrap budget |

Cone-field condition: $\Delta = (A_{min}-1)^2 - 4\beta c > 0$

**Golden range** (4D, 16-bit): $271 \le a_{1,1} \le 507$. Lower bound ensures $\Delta > 0$, upper bound satisfies 16-bit overflow safety.

### Bit-Rotation Enhancer

After each chaos iteration: $x^{(t+1)} = \Pi^{-1}\!\big(\mathrm{ROL}_{nw}(\Pi(x^{(t)}), k_t)\big)$, where $k_t = 1 + (t \bmod (w-1))$. For 16-bit: $k_t = 1,2,3,\dots,15,1,2,\dots$

### ROTL-XOR Post-Processor

$y_t = u_t \oplus \mathrm{ROTL}_{64}(u_{t-1}, 17) \oplus \mathrm{ROTL}_{64}(u_{t-2}, 43)$, where $u_t = x_1 \parallel x_2 \parallel x_3 \parallel x_4$.

### Key Components

| Component | Algorithm | Status |
|---|---|---|
| Parameter synthesis | R1-LY (one-shot, parity-alternating) | Verified |
| Base chaotic map | n-D fixed-point map on $\mathbb{Z}_{2^{16}}$ | Verified |
| Dynamics enhancer | Bit-rotation ($k_t = 1 + t \bmod 15$) | Verified |
| Post-processor | ROTL-XOR mixing | Verified |
| Hardware IP | 3-stage pipeline, 100 MHz, 800 MB/s | **P&R Verified** |
| Stream cipher | XOR keystream, HLS ↔ Python 100% bit-exact | **Verified** |

## Project Structure

```
hd_chaos_project/
├── src/chaos/                  # Main Python package
│   ├── core/                   #   Map implementation
│   ├── analysis/               #   Lyapunov, sensitivity, distribution
│   ├── visualization/          #   4D plots, verification figures
│   ├── utils/                  #   LaTeX, NIST sequence export
│   └── application/prng/       #   PRNG engine + post-processor
├── hw/                         # Hardware design sources
│   ├── src_371/                #   HLS C++ (canonical, A_odd=371)
│   ├── fpga/                   #   FPGA engineer handoff
│   ├── application/            #   App developer guide
│   ├── embedded/               #   Embedded C driver
│   ├── docs/                   #   Design summary, P&R results
│   ├── reports/                #   OOC + P&R reports
│   └── testbench/              #   C testbench (golden model)
├── tests/                      # Test suite (23+22+22+5 tests)
├── fpga_experiment/            # ★ FPGA physical experiment package
│   ├── FPGA_EXPERIMENT_GUIDE.md
│   ├── sources/                #   HLS C++ + Verilog wrapper
│   ├── bitstream/              #   JTAG-ready .bit file
│   ├── driver/                 #   C driver (bare-metal)
│   └── verification/           #   Golden model (.so + .py + ref data)
├── tools/                      # Utility scripts
├── docs/                       # Design documents & paper context
├── figures/                    # Generated figures (EPS/PDF)
└── requirements.txt
```

## Quick Start

### Software PRNG

```python
from chaos.core import ComplementaryChaosParams, HighDimChaosMap
from chaos.analysis import verify_chaos

synth = ComplementaryChaosParams(n=4, A_odd=371, B_even=50)
params = synth.synthesize()
chaos_map = HighDimChaosMap(params, word_size=16, seed=42)
result = verify_chaos(chaos_map, steps=3000)
```

### Hardware PRNG (HLS Bit-Exact)

```python
from hw.src_371.golden_prng import ChaosPRNG371

prng = ChaosPRNG371()                       # A_odd=371, warmup=2000
w = prng.next()                             # 0x515c697507ec42e9
data = prng.generate_bytes(1024)

# Stream cipher
keystream = prng.generate_bytes(len(msg))
ciphertext = bytes(p ^ k for p, k in zip(msg, keystream))
```

### GUI Export

```bash
python tests/prng_export_371.py   # HLS-verified (A_odd=371)
python tests/prng_export.py       # Generic (A_odd dialog)
```

### Core API

| Method | Returns | Description |
|---|---|---|
| `ComplementaryChaosParams(n, A_odd, B_even).synthesize()` | `dict` | Generate R1-LY parameters |
| `HighDimChaosMap(params).iterate(steps)` | `(steps, n)` array | Multi-step iteration |
| `.next()` | `(n,)` array | Single step |
| `.set_state_from_physical(x)` | — | Set $[x_1,\dots,x_n]$ |
| `.get_physical_state()` | `(n,)` array | Get current state |

### Running Tests

```bash
python tests/test_core.py                # 23 tests — core map
python tests/test_analysis.py            # 22 tests — analysis
python tests/test_prng.py                # 22 tests — PRNG engine
python tests/test_stream_cipher_371.py   # 5 tests — stream cipher
python tests/test_r1_ly_generation.py    # R1-LY generation + LaTeX
```

## Hardware Implementation

| Metric | Value |
|---|---|
| Target | xc7z020clg400-1 (Zynq-7020) |
| Clock | **100 MHz** |
| Throughput | **800 MB/s** |
| Pipeline | 3-stage (shift-add → rotation → post-process) |
| LUT / FF / DSP | 500 / 370 / 1 (post-P&R) |
| Power | 0.12 W |
| Setup WNS | +0.549 ns (real P&R) |
| Routed nets | 925/925 (100%) |
| Bitstream | `fpga_experiment/bitstream/prng_wrapper.bit` |

### FPGA Physical Experiment

The historical `fpga_experiment/` directory, `FPGA_EXPERIMENT_GUIDE.md` and `hw/docs/DESIGN_SUMMARY.md` are not included in this public candidate. The selected fixed-4D HLS source and its limits are described in `../../docs/HARDWARE.md`.

## Dependencies

- Python 3.8+
- numpy >= 1.20
- matplotlib >= 3.3 (for visualization)
- Vitis HLS 2024.1 + Vivado 2024.1 (for hardware synthesis)

---

## Documentation Constraint

> **Every code change MUST synchronize the corresponding README.**

| When you change... | Update... |
|---|---|
| `core/` modules | `src/chaos/core/README.md` |
| `analysis/` modules | `src/chaos/analysis/README.md` |
| `visualization/` modules | `src/chaos/visualization/README.md` |
| `utils/` modules | `src/chaos/utils/README.md` |
| `hw/` files | `hw/README.md` + affected sub-folder README |
| Test files | `tests/README.md` |
| Public API (`__init__.py`) | Root `README.md` |
| Dependencies | `requirements.txt` + root `README.md` |
