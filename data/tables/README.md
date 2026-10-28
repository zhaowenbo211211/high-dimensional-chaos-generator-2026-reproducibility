# Table data and provenance

Eight `tab_*.tex` files are exact extracts of the current manuscript tables, not newly calculated tables.

| Table label | Source / limits |
|---|---|
| `tab:r1cs-recommended-parameters` | Analytic calculation at A=271; `data/lineage/experiment_configs/recommended_parameters.csv` instead uses A=371 and is not a direct table export. |
| `tab:Dyna-Per-high-dimensional-family` | `data/figures/expD_dimension_scalability/dimension_metrics_summary.csv` uses D protocol and rounded values. |
| `tab:complexity-arithmetic` | Analytical operation counts; no direct CSV identified. |
| `tab:runtime-canonical-state` | Analytical state requirements; no implemented runtime hardware result. |
| `tab:proposed-4d-map-resource` | Historical minimal-delta fixed 4D netlist reports; source report not bundled as backend material. |
| `tab:prng371-implementation-summary-power` | Independent full PRNG netlist; power estimated. HLS/RTL outputs and implementation runs not redistributed. |
| `tab:prng-recent-primary-compact` | Published literature comparisons; historical CSV alone is not a verified exact source. Third-party rights pending. |
| `tab:prng371-nist-summary` | `nist/nist_summary.csv`, `nist/nonoverlapping_template_family_summary.csv`, and `data/lineage/nist/official_run/finalAnalysisReport.txt`; retain flagged template row. |

The manuscript table extracts and scientific records are historical; no values have been rerun or adjusted.
