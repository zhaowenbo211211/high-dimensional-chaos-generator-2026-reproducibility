# chaos4d_compare_16bit

Unified 16-bit comparison platform for multiple 4D discrete chaotic maps.

This project evaluates several 4D chaotic systems under a shared 16-bit fixed-point realization. The current codebase already supports representative-instance comparison, golden-range family scans for the paper's proposed enhanced system, and multiple paper-facing metrics.

## Project Navigation

The project now includes folder-level documentation for the main functional
directories. Use these local guides when navigating the codebase:

- [chaos_maps/README.md](chaos_maps/README.md)
- [configs/README.md](configs/README.md)
- [experiments/README.md](experiments/README.md)
- memory-bank/README.md (historical path not included)
- [metrics/README.md](metrics/README.md)
- nist_sp_800_22/README.md (historical path not included)
- paper_sections/README.md (historical path not included)
- results/README.md (historical path not included)
- temp/README.md (historical path not included)
- [tests/README.md](tests/README.md)
- [utils/README.md](utils/README.md)

For high-level maintenance notes, see:

- PROJECT_LOG.md (historical path not included)

The documentation scope covers the main functional directories plus the
managed temporary-output area. Cache folders and virtual environment folders
remain excluded from this layer of documentation.

## Implemented Maps

- `Proposed4DBaseMap`
- `Proposed4DEnhancedMap`
- `NDCS4DMap`
- `URUK4DDCMMap`
- `MHM4DMap`
- `IDHNN4DMap`
- `MPRC4DMap`
- `MSLC4DMap`
- `HCCDM4DMap`
- The proposed method
- Yao et al.'s method
- He et al.'s method
- Zhang et al.'s method
- Zang et al.'s method

## Implemented Metrics

- `LE`: maximum Lyapunov exponent by the Wolf full-state method
- `SE`: spectral entropy complexity
- `FE`: fuzzy entropy
- `PE`: permutation entropy
- `SI`: scale index based non-periodicity analysis via CWT

## Shared Numeric Semantics

- All map states follow a unified 16-bit fixed-point realization in `[0, 1)`.
- The common initial state is `(0.1, 0.2, 0.3, 0.4)`.
- Metric-side analysis is aligned with the 16-bit realization:
  - `SE`, `FE`, and `SI` first quantize each component sequence into the shared 16-bit integer domain before final metric evaluation.
  - `PE` also first quantizes each component sequence into the shared 16-bit integer domain before ordinal-pattern counting.
  - `LE` uses the map's float-state view for distance evaluation, while perturbation and evolution remain consistent with the map's 16-bit dynamics.

The fixed-point conversion rule is implemented in [utils/fixed16.py](utils/fixed16.py).

## Current Default Metric Settings

### Parameter Comparison Table

| Metric | Mode | Raw Length | Transient | Main Parameters |
| --- | --- | ---: | ---: | --- |
| `LE` | Wolf full-state | `5000` | `1000` | `renorm_threshold`, `eps`, full-state Euclidean distance |
| `SE` | `direct` | `5000` | `1000` | FFT after mean removal |
| `FE` | `direct` | `5000` | `1000` | `m=2`, `tau=1`, `n=2`, `r=0.2 * std(component_u16_sequence)` |
| `PE` | `direct` | `5000` | `1000` | `m=5`, `tau=1`, `normalized=True` |
| `SI` | `direct` | `40000` | `3000` | `wavelet=morlet`, study interval `[2, 256]` |

### LE

- method: Wolf full-state
- `transient = 1000`
- `max_steps = 5000`

### SE

- mode: `direct`
- `raw_length = 5000`
- `transient = 1000`
- minimum allowed sequence length: `5000`

### FE

- mode: `direct`
- `raw_length = 5000`
- `transient = 1000`
- `m = 2`
- `tau = 1`
- `n = 2`
- `r = 0.2 * std(component_u16_sequence)` when `r` is not explicitly given

### SI

- mode: `direct`
- `raw_length = 40000`
- `transient = 3000`
- wavelet: `morlet`
- study interval: `[2, 256]`
- total CWT scale grid: geometric spacing over the extended interval up to `2 * s1`

The SI implementation is in [metrics/nonperiodicity.py](metrics/nonperiodicity.py).

### PE

- mode: `direct`
- `raw_length = 5000`
- `transient = 1000`
- `m = 5`
- `tau = 1`
- `normalized = True`

The current PE implementation follows the standard Bandt-Pompe ordinal-pattern
definition. After 16-bit quantization, each component sequence is converted
into permutation patterns, Shannon entropy is computed from the empirical
pattern distribution, and the default result is normalized as:

`PE_norm = PE / log(m!)`

Ties are handled deterministically by stable sorting. If equal values appear in
one embedding vector, their original time order is used to break the tie.

The PE implementation is in [metrics/permutation_entropy.py](metrics/permutation_entropy.py).

## Golden-Range Family Support

The current family scan targets the paper's enhanced system:

- map: `Proposed4DEnhancedMap`
- fixed coupling configuration through the map's paper-facing parameter logic
- golden interval:
  - `271 <= a11 <= 507`
  - `a21 = a11 + 2`
  - `a31 = a11 + 4`

Family scan entry points currently support:

- `SE`
- `FE`
- `PE`
- `SI`
- `LE` representative/golden-range support for proposed-family workflows

## High-Dimensional Enhanced Family Evaluation

The project now includes a dedicated experiment branch for the paper's enhanced
high-dimensional chaotic family. This branch studies only the proposed method
itself and evaluates how the average dynamical behavior changes when the system
dimension increases from `4D` to `12D`.

### Construction Scope

- dimensions: `4, 5, 6, 7, 8, 9, 10, 11, 12`
- system type: enhanced proposed family only
- enhancement rule: one base-map update followed by the paper's bit-rotation based dynamics enhancer
- state initialization: dimension-aware extension of the common `(0.1, 0.2, 0.3, 0.4)` rule

### R1--LY Parameter Synthesis

High-dimensional parameters are not hard-coded. They are generated by
`synthesize_r1_ly_params()` in [utils/r1ly.py](utils/r1ly.py).

Default synthesis inputs:

- `A_odd = 371`
- `B_even = 50`
- `Delta = 2`
- `eta = 61`
- `kappa = 4`
- `tau = 1`

The synthesis loop explicitly performs:

- Step 1: complementary parity-alternating sequences
- Step 1a: positivity check of the smallest coupling coefficient
- Step 2: computation of `A_min` and `beta`
- Step 3: computation of `b`
- Step 4: verification of cone-field Condition `(S)`
- Step 5: computation of band widths
- Step 6: computation of slab thickness / no-wrap budget
- Step 7: verification of the no-wrap budget
- Step 8: return of the validated parameter set

If validation fails, the algorithm automatically repairs the parameter set by:

- increasing `A_odd` by `2`
- reducing `eta`
- reducing `B_even` by `2`

### Included Metrics

- `LE`: Wolf full-state maximum Lyapunov exponent
- `SE`: spectral entropy complexity
- `FE`: fuzzy entropy
- `SI`: scale index / non-periodicity
- `PE`: permutation entropy

Average-value definition:

- for `SE`, `FE`, `SI`, and `PE`, each state direction is evaluated separately and the reported value is the arithmetic mean over all `n` state components
- for `LE`, the summary table keeps the system-level `lyap_max` and also stores directional component estimates in the raw CSV

### Outputs

The high-dimensional experiment writes to:

- `results/high_dimensional/raw/`
- `results/high_dimensional/tables/`
- `results/high_dimensional/figures/`
- `results/high_dimensional/diagnostics/`

Main files:

- `results/high_dimensional/raw/r1_ly_dimension_params.csv`
- `results/high_dimensional/raw/dimension_metrics_summary.csv`
- `results/high_dimensional/raw/dimension_metrics_detailed.csv`
- `results/high_dimensional/tables/dimension_metrics_summary.tex`
- `results/high_dimensional/figures/dimension_metrics_summary.pdf`
- `results/high_dimensional/figures/dimension_metrics_summary.eps`

The project also includes a paper-ready subsection draft at
paper_sections/high_dimensional_performance.tex (historical path not included).

### Run the Experiment

```bash
python experiments/run_dimension_scalability.py
```

Example with explicit lengths:

```bash
python experiments/run_dimension_scalability.py \
  --dimensions 4 5 6 7 8 9 10 11 12 \
  --se-length 5000 \
  --fe-length 5000 \
  --pe-length 5000 \
  --si-length 40000 \
  --le-steps 5000
```

For faster smoke-style local checks you may shorten `FE`, `PE`, `SI`, and `LE`,
but `SE` still follows the project's minimum raw-length requirement of `5000`.

## High-Dimensional Literature Comparison

The project contains a completed all-methods comparison branch for four
high-dimensional chaotic-system papers plus the proposed enhanced n-D map. The
standard entry point is [experiments/run_high_dim_literature_comparison.py](experiments/run_high_dim_literature_comparison.py).
For long full runs, [experiments/run_high_dim_literature_full_incremental.py](experiments/run_high_dim_literature_full_incremental.py) writes CSV outputs after each completed method/dimension pair.

### Reproducible Framework Generator

A separate framework/template entry point is available at
[experiments/run_high_dim_all_methods_full.py](experiments/run_high_dim_all_methods_full.py).
It is designed to generate the reproduction directory structure, CSV
templates, LaTeX table template, placeholder figures, and a one-step smoke
check without running the full metric experiment.

Default smoke/template run:

```bash
.venv/bin/python experiments/run_high_dim_all_methods_full.py
```

Explicit modes:

```bash
.venv/bin/python experiments/run_high_dim_all_methods_full.py --mode smoke
.venv/bin/python experiments/run_high_dim_all_methods_full.py --mode full
```

The `full` mode is intentionally a future hook in this framework generator: it
still does not execute the long full simulation. Use
`experiments/run_high_dim_literature_full_incremental.py` when the actual full
metric computation is required.

Framework outputs:

- `results/high_dim_literature_comparison/raw/high_dim_all_methods_summary.csv`
- `results/high_dim_literature_comparison/raw/high_dim_all_methods_detailed.csv`
- `results/high_dim_literature_comparison/raw/method_dimension_support.csv`
- `results/high_dim_literature_comparison/tables/high_dim_all_methods_comparison_table.tex`
- `results/high_dim_literature_comparison/figures/high_dim_comparison_lambda_max.pdf`
- `results/high_dim_literature_comparison/figures/high_dim_comparison_lambda_max.eps`
- corresponding `SE_mean`, `FE_mean`, `SI_mean`, and `PE_mean` placeholder figures
- `results/high_dim_literature_comparison/diagnostics/high_dim_all_methods_framework_metadata.json`

Compared methods:

- The proposed method
- Yao et al.'s method
- He et al.'s method
- Zhang et al.'s method
- Zang et al.'s method

The implementation is maintained in `chaos_maps/proposed_enhanced.py` and
`chaos_maps/literature_high_dim.py`.

The four literature references used by this comparison have been added to
`out_paper/gn_refs.bib` with keys `Yao2025NDComplexChaotic`,
`He2024SineMapLatticeMemristor`, `Zhang2023CyclicSymmetricMap`, and
`Zang2021HighDimDiscrete`.

Unified comparison rules:

- dimensions: `n=4,...,12`
- public initial state: `x_i(0)=100*i`
- complex imaginary initial states are fixed to zero and are never randomized
- lattice/memristor auxiliary states start from zero unless a paper-specific rule overrides them
- every iteration returns the public state to uint16 `[0,65535]`
- the proposed base-only map is excluded

Metric implementations used by the future full computation:

- `lambda_max`: `metrics/lyapunov.py`
- `SE_mean`: `metrics/spectral_entropy.py`
- `FE_mean`: `metrics/fuzzy_entropy.py`
- `SI_mean`: `metrics/nonperiodicity.py`
- `PE_mean`: `metrics/permutation_entropy.py`

Generated all-method outputs:

- `results/high_dim_literature_comparison/algorithm_analysis_all.md`
- `results/high_dim_literature_comparison/raw/high_dim_all_methods_summary.csv`
- `results/high_dim_literature_comparison/raw/high_dim_all_methods_detailed.csv`
- `results/high_dim_literature_comparison/raw/method_dimension_support.csv`
- `results/high_dim_literature_comparison/tables/high_dim_all_methods_comparison_table.tex`
- `results/high_dim_literature_comparison/figures/high_dim_comparison_*.pdf`
- `results/high_dim_literature_comparison/figures/high_dim_comparison_*.eps`

Run focused preparation checks before the full experiment:

```bash
.venv/bin/python -m pytest tests/test_high_dim_literature_maps.py
```

Run the full incremental experiment:

```bash
.venv/bin/python -u experiments/run_high_dim_literature_full_incremental.py
```

Current metadata status:

- `full_data_exists = true`
- `full_experiment_executed = true`

Mean results over `n=4,...,12` from
`raw/high_dim_all_methods_summary.csv`:

| Method | lambda_max | SE_mean | FE_mean | SI_mean | PE_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| The proposed method | 8.746552 | 0.945825 | 2.212639 | 0.862096 | 0.997504 |
| Yao et al.'s method | 10.267517 | 0.945962 | 2.213823 | 0.850580 | 0.997505 |
| He et al.'s method | 0.651100 | 0.937398 | 1.293055 | 0.493149 | 0.687903 |
| Zhang et al.'s method | 0.619146 | 0.585561 | 1.830282 | 0.045485 | 0.893021 |
| Zang et al.'s method | 0.927984 | 0.940617 | 1.527322 | 0.751083 | 0.922437 |

Under this specific 16-bit protocol, The proposed method has the highest
average `SI_mean` and remains close to Yao et al.'s method on the entropy
metrics, while Yao et al.'s method has the largest average `lambda_max`.

## Official NIST/PRNG Scope

The active PRNG/NIST workflow is now limited to the final system-level 64-bit
block rule:

```text
u_t = x1(t) || x2(t) || x3(t) || x4(t)
y_t = u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43)
```

The runnable entry points for this workflow live in `experiments/nist_prng/`.
Older Python-side NIST approximations and PRNG search branches were removed
from `experiments/`.

## Project Layout

- `chaos_maps/`: 4D chaotic map implementations
- `metrics/`: metric implementations for LE, SE, FE, PE, SI, and related helpers
- `experiments/`: runnable CLI entry points and plotting scripts; NIST/PRNG scripts are under `experiments/nist_prng/`
- `paper_sections/`: paper-facing subsection snippets generated for manuscript integration
- `utils/`: fixed-point helpers, IO utilities, initializers, and trajectory helpers
- `tests/`: smoke tests and metric-level tests
- `results/`: generated raw tables and figures
- `temp/`: temporary scratch outputs and short-run test artifacts

## Quick Start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run one map trajectory preview:

```bash
python experiments/run_single_map.py --map proposed_enhanced
```

Run one metric on one map:

```bash
python experiments/run_single_map.py --map proposed_enhanced --metric le
python experiments/run_single_map.py --map proposed_enhanced --metric se
python experiments/run_single_map.py --map proposed_enhanced --metric fe
python experiments/run_single_map.py --map proposed_enhanced --metric pe
python experiments/run_single_map.py --map proposed_enhanced --metric si
```

Run cross-system comparison:

```bash
python experiments/run_cross_compare.py
```

Run only one cross-comparison metric:

```bash
python experiments/run_cross_compare.py --metric le
python experiments/run_cross_compare.py --metric se
python experiments/run_cross_compare.py --metric fe
python experiments/run_cross_compare.py --metric pe
python experiments/run_cross_compare.py --metric si
```

Run golden-range family scans:

```bash
python experiments/run_family_scan.py --metric se
python experiments/run_family_scan.py --metric fe
python experiments/run_family_scan.py --metric pe
python experiments/run_family_scan.py --metric si
```

Run the high-dimensional enhanced-family evaluation:

```bash
python experiments/run_dimension_scalability.py
```

Generate the retained system-level PRNG source data:

```bash
python experiments/nist_prng/run_nist_prng_plain_concat_only.py
```

Run the retained system-level PRNG through bundled official NIST STS:

```bash
python experiments/nist_prng/run_nist_prng_system_level.py
```

Run the local official NIST STS browser UI:

```bash
python experiments/nist_prng/run_official_nist_ui.py
```

## Main Experiment Scripts

- [experiments/run_single_map.py](experiments/run_single_map.py)
  - single-system trajectory or metric run
- [experiments/run_cross_compare.py](experiments/run_cross_compare.py)
  - cross-system comparison table export
- [experiments/run_family_scan.py](experiments/run_family_scan.py)
  - golden-range scan entry
- `experiments/nist_prng/run_nist_prng_plain_concat_only.py`
  - generate retained PRNG source data only
- `experiments/nist_prng/run_nist_prng_system_level.py`
  - run the retained PRNG source through bundled official NIST SP 800-22 STS
- `experiments/nist_prng/run_official_nist_ui.py`
  - simple browser UI for the bundled official NIST SP 800-22 suite
Related plotting helpers:

- [experiments/plot_golden_range_se.py](experiments/plot_golden_range_se.py)
- [experiments/plot_golden_range_fe.py](experiments/plot_golden_range_fe.py)
- [experiments/plot_golden_range_si.py](experiments/plot_golden_range_si.py)
- [experiments/plot_golden_range_pe.py](experiments/plot_golden_range_pe.py)
- [experiments/plot_si_diagnostics.py](experiments/plot_si_diagnostics.py)

## Result Directories

The project now separates SI outputs into dedicated subfolders so figures and tables stay organized.

Permutation entropy uses its own dedicated output tree:

- `results/permutation_entropy/raw/`
- `results/permutation_entropy/tables/`
- `results/permutation_entropy/figures/`
- `results/permutation_entropy/diagnostics/`

Current PE outputs include:

- `results/permutation_entropy/raw/pe_comparison.csv`
- `results/permutation_entropy/raw/enhanced_golden_range_pe.csv`
- `results/permutation_entropy/figures/enhanced_golden_range_pe.pdf`
- `results/permutation_entropy/figures/enhanced_golden_range_pe.eps`

Typical outputs include:

- `results/raw/se_comparison.csv`
- `results/raw/fe_comparison.csv`
- `results/raw/scale_index_comparison.csv`
- `results/raw/scale_index/enhanced_golden_range_scale_index.csv`
- `results/raw/scale_index/diagnostics/enhanced_scale_index_diagnostics.json`
- `results/figures/enhanced_golden_range_se_components.pdf`
- `results/figures/enhanced_golden_range_fe.pdf`
- `results/figures/scale_index/enhanced_golden_range_scale_index.pdf`
- `results/figures/scale_index/diagnostics/enhanced_scale_index_scalogram.pdf`
- `results/figures/scale_index/diagnostics/enhanced_scale_index_si_x1.pdf`
- `results/figures/scale_index/diagnostics/enhanced_scale_index_si_x2.pdf`
- `results/figures/scale_index/diagnostics/enhanced_scale_index_si_x3.pdf`
- `results/figures/scale_index/diagnostics/enhanced_scale_index_si_x4.pdf`

The retained official NIST/PRNG workflow writes to:

- `results/nist_prng_system_level/`
- `results/nist_prng_system_level_plain_only/`
- `results/official_nist_ui/`

## System-Level PRNG Official STS Evaluation

This branch evaluates the representative enhanced `4D` instance as one
system-level PRNG output and uses the bundled official `NIST SP 800-22 STS`
as the final decision source. The goal is to compare the simplest overall
system output using only the retained final PRNG block rule.

### Final Design Choice

From the hardware-design point of view, the final selected system-level PRNG
scheme is the simplest overall structure:

- `plain_concat_linear_mixing`

The reason is that this scheme keeps the datapath maximally regular and
lightweight:

- no per-direction pre-delay-XOR stage
- no extra per-lane delay registers before system assembly
- direct `4 x 16-bit -> 64-bit` concatenation
- one word-level mixing stage only:
  - `y_t = u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43)`

### Retained Scheme

The baseline system-level PRNG scheme is intentionally as simple and regular as
possible:

- Stage 0:
  - `raw_word_t = x1(t) || x2(t) || x3(t) || x4(t)`
- Stage 1:
  - `L_t = raw_word_t XOR ROTL64(raw_word_(t-1), 17) XOR ROTL64(raw_word_(t-2), 43)`

No nonlinear scrambler, no truncation, no per-lane rotation, and no extra
delay-XOR are added.

### Official STS Configuration

- suite: bundled official `NIST SP 800-22 STS`
- sequence length: `1000000`
- sequence count: `100`
- total bits: `100 x 10^6`
- warm-up discard: `1000`
- input mode: ASCII `0/1`
- tail post-processing: none

### Outputs

- `results/nist_prng_system_level/raw/system_level_summary.csv`
- `results/nist_prng_system_level/raw/plain_concat_linear_mixing_official_category_summary.csv`
- `results/nist_prng_system_level/tables/system_level_nist_summary.tex`
- `results/nist_prng_system_level/official_reports/plain_concat_linear_mixing/`

If you only need the final selected hardware-oriented system-level source data,
the generated ASCII bitstream is:

- `results/nist_prng_system_level/diagnostics/plain_concat_linear_mixing/plain_concat_linear_mixing_100000000bit_ascii.txt`

For a dedicated final-design-only data-generation path, use:

```bash
python experiments/nist_prng/run_nist_prng_plain_concat_only.py \
  --sequence-bit-length 1000000 \
  --sequence-count 100 \
  --warmup-steps 1000 \
  --output-root results/nist_prng_system_level_plain_only
```

This program generates only the selected final hardware-oriented scheme:

- `plain_concat_linear_mixing`

and writes the source files to:

- `results/nist_prng_system_level_plain_only/diagnostics/plain_concat_linear_mixing/plain_concat_linear_mixing_100000000bit_ascii.txt`

### Run The Experiment

```bash
python experiments/nist_prng/run_nist_prng_system_level.py \
  --sequence-bit-length 1000000 \
  --sequence-count 100 \
  --warmup-steps 1000 \
  --nist-root nist_sp_800_22 \
  --output-root results/nist_prng_system_level
```

This program generates and evaluates only the final hardware-oriented source
data for:

- `plain_concat_linear_mixing`

## Bundled Official NIST SP 800-22 UI

The project now also includes a simple local browser UI for the bundled
`nist_sp_800_22` folder. The goal is convenience: manually choose only the
input file, the input data type, the test categories, and the output location,
then let the UI drive the legacy `assess` prompt sequence automatically.

### What The UI Does

- lets you choose a data source manually
  - upload from the operating-system file picker
  - or use an existing file path directly
- lets you choose the official STS input mode manually
  - ASCII `0/1`
  - Binary bytes
- lets you choose the NIST test categories manually
- lets you choose the result output root manually
- automatically creates an isolated runtime sandbox for each run
- automatically copies the selected input file into that sandbox
- automatically runs the bundled `nist_sp_800_22/assess` inside the sandbox
- automatically applies the project's default official-run settings
  - `sequence_bit_length = 1000000`
  - `sequence_count = 100`
- keeps the official runtime outputs under
  `<your-output-root>/<timestamp>_<auto-label>/runtime/nist_sp_800_22/`

### Manual Source Selection

The UI supports two source modes:

- `upload_file`
  - opens the browser's operating-system file picker
  - saves the selected file into `nist_sp_800_22/ui_inputs/<original-name>`
- `existing_file`
  - uses an already prepared input file directly

For the official STS executable, paths with spaces are still risky because the
legacy prompt parser is space-sensitive. For that reason, the recommended
workflow is to keep manual input files under:

- `nist_sp_800_22/ui_inputs/`

and use a no-space filename there. The UI no longer copies inputs to `/tmp` by
default.

### Runtime Isolation

To avoid cross-run contamination, the UI no longer executes the shared project
copy of `nist_sp_800_22` directly. For every run it now:

- creates a fresh runtime sandbox under the selected output root
- copies the official suite into that runtime directory
- excludes the shared `ui_inputs/` and stale `AlgorithmTesting/` state from the copied template
- stages the selected source file into the runtime sandbox as a local input file
- executes `assess` only inside that isolated runtime tree

This makes repeated UI runs much safer when multiple experiments or users may
be invoking the official suite around the same time.

### Automatic Defaults

The bundled official suite expects:

- command-line argument = single-sequence length
- for the project's default formal runs:
  - `sequence_bit_length = 1000000`
  - `sequence_count = 100`
- input mode:
  - ASCII `0/1`

So the official call is organized as:

- `assess 1000000`
- with `100` bitstreams
- not `assess 100000000`

### Start The UI

```bash
python experiments/nist_prng/run_official_nist_ui.py
```

Then open:

```text
http://127.0.0.1:8765
```

To list the official 15 test items without starting the server:

```bash
python experiments/nist_prng/run_official_nist_ui.py --list-datasets
```

## Tests

Run all tests:

```bash
pytest
```

Run the smoke suite only:

```bash
python tests/test_metrics_smoke.py
```

## Notes

- Python `3.10+` is recommended.
- The enhanced map retained here is the current paper-facing enhanced variant in this codebase.
- `SI` is currently used as a continuous non-periodicity indicator. No hard binary SI threshold is enforced in the code.
