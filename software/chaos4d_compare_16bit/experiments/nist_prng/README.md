# experiments/nist_prng

This folder contains the retained NIST/PRNG entry points.

The only project-generated PRNG output scheme kept here is the final
system-level 64-bit block rule:

```text
u_t = x1(t) || x2(t) || x3(t) || x4(t)
y_t = u_t XOR ROTL64(u_(t-1),17) XOR ROTL64(u_(t-2),43)
```

where each `x_i(t)` is a 16-bit digital state output from the representative
enhanced 4D map, `||` is direct concatenation, and `y_t` is emitted as the PRNG
output block.

## Files

- `run_nist_prng_plain_concat_only.py`
  - generates only the final PRNG source data
  - writes binary payload and ASCII `0/1` input files for official STS
  - does not run any NIST tests by itself
- `run_nist_prng_system_level.py`
  - generates the same final PRNG source data
  - runs the bundled official `nist_sp_800_22/assess`
  - exports category summaries, official report paths, metadata, and a LaTeX summary table
- `run_official_nist_ui.py`
  - starts a small local UI for manually running the bundled official STS on a selected input file
  - supports choosing input type, selected official test categories, and output location

## Removed Experiment Families

Older NIST/PRNG search programs were removed from `experiments/` because they
do not match the retained final block rule. This includes Python-side NIST
approximations, delay-XOR searches, two-stage truncation searches, direction-4
linear-mixing searches, truncation-position comparisons, and nonlinear PRNG
mixing searches.

## Commands

Generate only the final PRNG source data:

```bash
python experiments/nist_prng/run_nist_prng_plain_concat_only.py \
  --sequence-bit-length 1000000 \
  --sequence-count 100 \
  --warmup-steps 1000 \
  --output-root results/nist_prng_system_level_plain_only
```

Run the official STS on the retained PRNG source:

```bash
python experiments/nist_prng/run_nist_prng_system_level.py \
  --sequence-bit-length 1000000 \
  --sequence-count 100 \
  --warmup-steps 1000 \
  --nist-root nist_sp_800_22 \
  --output-root results/nist_prng_system_level
```

Start the manual official STS UI:

```bash
python experiments/nist_prng/run_official_nist_ui.py
```

List official test categories without starting the UI:

```bash
python experiments/nist_prng/run_official_nist_ui.py --list-datasets
```

The UI option name is retained for compatibility, but the command lists the
official 15 STS test categories.
