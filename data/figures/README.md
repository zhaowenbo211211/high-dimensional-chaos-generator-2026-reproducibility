# Figure → CSV → script → configuration

Manuscript baseline SHA-256 `5927a817097c448a00838763184a2519e1b40c6caa95f6f9408148286cba2b8e`. CSVs are byte-identical stored scientific results, not raw trajectories or recomputations. Paths below are repository-root relative. B groups are 119 odd A values 271–507, with `group_index` 1–119. E contains five method slugs across dimensions 4–12. Numerical values are dimensionless metrics/estimates; dimension and index are counts. Actual task-specific settings are in `data/experiments_rev1_alignment/results/cache/` (B 595, D 45, E 225 records); selected recommended values are in `data/lineage/experiment_configs/recommended_parameters.csv`. D and E initialization protocols differ.

| Figure / TeX label | Panel or quantity | CSV and exact columns | Producer / configuration |
|---|---|---|---|
| 1 `fig:research-framework` | Schematic; SI inset | `data/figures/expB_golden_range/golden_range_si.csv`: `group_index,si_x1,si_x2,si_x3,si_x4,si_mean`; inset k=51–69 | Existing EPS/PDF and CSV hash chain and original SI plotting code support this mapping; see `data/lineage/figure1_si_inset_provenance.md`. No new rendering. |
| 2 `fig:golden-range-metrics` | `fig:SE-complexity-gold-range` | `expB_golden_range/golden_range_se.csv`: `group_index,se_x1,se_x2,se_x3,se_x4,se_mean` | B; `restyle_phase_b2_figures.py` and `restyle_publication_figures.py` under `software/experiments_rev1_alignment/scripts/`. |
| 2 | `fig:FE-gold-range` | `expB_golden_range/golden_range_fe.csv`: `group_index,fe_x1,fe_x2,fe_x3,fe_x4,fe_mean` | Same B scripts and task records. |
| 2 | `fig:PE-gold-range` | `expB_golden_range/golden_range_pe.csv`: `group_index,pe_x1,pe_x2,pe_x3,pe_x4,pe_mean` | Same B scripts and task records. |
| 2 | `fig:SI-gold-range` | `expB_golden_range/golden_range_si.csv`: `group_index,si_x1,si_x2,si_x3,si_x4,si_mean` | Same B scripts and task records. |
| 3 `fig:lyapunov-combined` | `fig:LEs-for-proposed maps` | `expB_golden_range/golden_range_le.csv`: `group_index,lyap_max` | B; same B2 renderer. |
| 3 | `fig:high-dimensional-comparison-lambda-max` | `expE_high_dim_literature/high_dim_all_methods_summary.csv`: `dimension,method_slug,lambda_max` | E; later `restyle_lyapunov_comparison_bars.py` grouped-bar renderer supersedes the earlier line rendering. |
| 4 `fig:high-dimensional-comparison` | `fig:high-dimensional-comparison-se-mean` | E summary CSV: `dimension,method_slug,se_mean` | E; B2 renderer and archived E task settings. |
| 4 | `fig:high-dimensional-comparison-fe-mean` | E summary CSV: `dimension,method_slug,fe_mean` | Same E source. |
| 4 | `fig:high-dimensional-comparison-pe-mean` | E summary CSV: `dimension,method_slug,pe_mean` | Same E source. |
| 4 | `fig:high-dimensional-comparison-si-mean` | E summary CSV: `dimension,method_slug,si_mean` | Same E source. |
| 5 `fig:runtime-core-architecture` | Conceptual schematic | No CSV | Conceptual analytical architecture; no measured 12D or runtime-switch implementation. |
| 6 `fig:prng371-design-block-diagram` | Schematic | No CSV | HLS functional source in `hardware/prng371_hls/`. |
| 7 `fig:cooja-topology` | Schematic and image thumbnails | No CSV for diagram | Thumbnail pixels withheld for pending rights; complete image reconstruction unavailable. |
| 8 `fig:cooja-packet-loss-reconstruction` | Original/encrypted/mask/decrypted image panels | No image pixel CSV provided | Images and payload withheld for pending rights. `data/lineage/cooja/03_results/metrics.csv` and receipt bitmap are supporting metrics, not the image. |

For short paths starting `expB_` or `expE_`, prefix `data/figures/`. Source copies also exist under `data/experiments_rev1_alignment/results/` for the existing stored-data adapter; CSV bytes match. `software/stored_data.py validate` validates archive selections, not scientific claims. Original plot scripts and configs may have historical paths; inspect before executing. No plotting was rerun in preparation.
