# Semantic Alignment Audit — rev1 digital-core re-experiments

Date: 2026-09-09. Author: Kimi (executing agent of `KIMI_WORK_PROMPT.md`).

This document records the semantic gap between the old experiment project
(`chaos4d_compare_16bit`, read-only reference) and the current §3 definitions in
`pre_art_rev1.tex` (sha256 `96f2df3137a4ed262f8a7f8233b44ad1bff0f13056ac384062ec9a5069c90264`,
matching `configs/replacement_manifest.json:source_sha256`), and fixes the exact
protocol implemented by the new subsystem in `experiments_rev1_alignment/`.

## 1. Old-vs-new semantic differences (must be eliminated)

| # | Aspect | Old project (`chaos4d_compare_16bit`) | Current §3 (authoritative) |
|---|--------|----------------------------------------|-----------------------------|
| 1 | Nonlinear feedback term | `b*x_2` fixed for all `n`, float logistic `4z(1-z)` | `b*x_{n-2}`; integer `floor(4*S*(Q-S)/Q)` with `S = x_{n-1}+s0` in `0..2Q-2`, no pre-mod of `S` |
| 2 | Floor vs truncation | float round-trip quantization | exact integer floor (floor for negatives; values here are non-negative) |
| 3 | `b` rule | `b = B_even + eta + 4(n-4)*tau`, `eta = 61` | `b = max(2, floor(0.30*A_min))` = `3*A_min//10` |
| 4 | `beta` | `(sum(coupling) + tau*max(coupling))/A_min`; `condition_s_passed` flag | `beta = sqrt(sum(a_jn^2))`, `c = sqrt(b^2+144)`, `rho = sqrt(145)`; strong check = `A_min > rho` AND `Dcone = (A_min-rho)^2 - 4*beta*c > 0` |
| 5 | Width budget | not present in this form | Steps 5–7 with `chat = sqrt(b^2+32)` (never `c`), `tau` retry `tau <- 0.8*tau`, fail if `tau < 0.1`, attempt limit `M = 100` |
| 6 | Mode II packing | big-endian: `packed = (packed<<16)|value` (x1 in the most significant word) | little-endian word order: `Z = sum(current[j] << (16*j))`, x1 in the least significant word |
| 7 | Mode II write-back | rotated value written back to ALL state words, and rotated `x_n` also written into the delay line (`xn_delay_line[-1]`) | rotation result overwrites only the first `n` words of `U`; the remaining `n-2` delay words keep the base-update values `U_{s_k}=s_{k-1}`; phase `k_t = 1 + (t mod 15)` on width `16n` |
| 8 | Delay history | roll-based `xn_delay_line` storing past `x_n` | explicit `s_1..s_{n-2}` shift register, `s_k^+ = s_{k-1}` |
| 9 | Golden Range `b` | fixed `b = 111` for the whole range | per-group `b = 3*A//10` |
| 10 | State exposed to metrics | n current coordinates (kept) | unchanged: `state_u16` = `(x_1..x_{n-1}, s_0)`; delay words and rotation phase are hidden state, re-cloned/synchronized at every Wolf renormalization |

## 2. Unchanged protocol (copied verbatim from the old project)

Metric implementations are copied byte-identical (see `audit/copied_source_hashes.txt`):

- LE: `metrics/lyapunov.py::wolf_lyapunov_full_state_nd`, `max_steps=5000`,
  `transient=1000`, `perturb_lsb=1`, `renorm_threshold=0.05`. Distance measured on
  the `n` current coordinates (normalized float); the perturbed system is
  re-cloned from the main system at every renormalization, so delay words and the
  rotation phase stay synchronized. `lyap_max` = total log growth / total counted
  steps; `lyap_mean`/`lyap_std` = unweighted statistics of per-interval rates.
- SE: `metrics/spectral_entropy.py::spectral_entropy_complexity`, mode `direct`,
  5000 samples after 1000 transient.
- FE: `metrics/fuzzy_entropy.py::fuzzy_entropy`, mode `direct`, m=2, exponent 2,
  tau=1, r=0.2*SD (per component), 5000 samples after 1000 transient.
- PE: `metrics/permutation_entropy.py::permutation_entropy`, m=5, tau=1,
  normalized, 5000 samples after 1000 transient.
- SI: `metrics/nonperiodicity.py::scale_index_cwt`, Morlet wavelet, 48 scales,
  40000 samples after 3000 transient; diagnostics via `scale_index_cwt_diagnostics`.
- Quantization helpers: `utils/fixed16.py` (floor quantization `x -> floor(65536*x)`,
  `u16 -> x/65536`).

Baselines reused as code (not as data): the seven 4D baseline maps
(`hccdm_4d, idhnn_4d, mhm_4d, mprc_4d, mslc_4d, ndcs_4d, uruk_4d_dcm`) and the
four high-dimensional literature generators (`literature_high_dim.py`: Yao complex
Pascal, He sine-lattice memristive, Zhang cyclic symmetric logistic, Zang linear
modular) are copied byte-identical. All metric data are RE-COMPUTED in this
subsystem (no old result CSV is reused), so the "reuse" question of
`KIMI_WORK_PROMPT.md` §四 is settled by full recomputation; `reused` flags will
accordingly be `false` everywhere.

## 3. Experiment protocols (new subsystem)

- A. 4D direct comparison: proposed = R1-LY( n=4, A_odd=371, B_even=50 ) ->
  `a_linear=[371,373,375]`, `a_coupling=[50,48,46]`, `b=111`; new digital core +
  new Mode II. Initial state `(0.1,0.2,0.3,0.4)` floor-quantized, zero history,
  phase 0. Seven baselines unchanged (their own constructors + common init).
- B. Golden Range: `A = 271,273,...,507` (119 groups), `a_linear=[A,A+2,A+4]`,
  `a_coupling=[50,48,46]`, `b=3*A//10`; same init as A.
- C. SI diagnostics: representative 4D instance of A; scalogram overview +
  4 inner-scalogram curves.
- D. Single-family scalability: `n=4..12`, `A_odd=371`, `B_even` from the new
  table (4:50, 5:44, 6:40, 7:38, 8:36, 9:36, 10:34, 11:34, 12:34),
  `b=111`; dimension-extended initial state of `utils/initializers.py`
  (`(0.1,...,0.9,0.15,...)`), zero history, phase 0. The n=4 row reuses the
  group-A proposed results only if the D-protocol init coincides — it does NOT
  (D uses the dimension-extended rule, which for n=4 equals `(0.1,0.2,0.3,0.4)`;
  verified identical to A's init), so D n=4 reuses A's proposed record.
- E. High-dimensional literature comparison: 5 methods x `n=4..12` = 45 groups,
  common init `x_i(0)=100*i` (u16), literature hidden states per their own
  classes, Yao imaginary part zero. Proposed uses the same `x_i(0)=100i` init and
  the D-parameter table per dimension. E n=4 proposed is computed independently
  (different init than A/D) and is NOT forced to equal D's n=4 row.

Per-run caching: every (experiment, method, n, A, metric) result is stored as one
JSON under `results/`, keyed by a config hash covering the semantic version of the
core, parameters, init, and metric settings; any change invalidates the cache.
Failed runs are marked `failed` with the error message; no zero/NaN filling.

## 4. Frozen items (not touched)

`tab:complexity-arithmetic`, `tab:proposed-4d-map-resource`, all FPGA/PRNG/NIST/
Cooja/Renode figures and data, §3 math (real-valued QR Lyapunov paragraph of the
4D instance is kept separate from these digital Mode-II results), section titles,
labels, and figure layout of §4.

## 5. Environment

- Python: system `python3` (numpy 2.2.3, matplotlib 3.10.9, PyYAML 6.0.1).
  Full version list: `requirements_rev1.txt` (to be frozen at run time).
- Old project virtualenv not used; no writes to the old project.
