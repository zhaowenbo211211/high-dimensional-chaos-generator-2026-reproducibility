# Existing scripts

Current publication plotting scripts are preserved under `software/experiments_rev1_alignment/scripts/`, including `restyle_phase_b2_figures.py`, `restyle_publication_figures.py`, and the later `restyle_lyapunov_comparison_bars.py`. The safe stored-data adapter is `software/stored_data.py`; it validates archived values before rendering and requires output under a fresh temporary directory. This package preparation ran only its `validate` operation, not replotting or numerical experiments.

`scripts/prng371/` and `scripts/nist/` hold original historical scripts. They may write output, expect missing tools or historical layouts, and were not executed; use isolated copies.
