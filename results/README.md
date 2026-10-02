# Results

> **Before first publish:** copy your final run directories into this folder.
> Nothing is shipped pre-populated, so that the repository can never contain
> superseded numbers.
>
> ```bash
> cp out_v2_m9/T*.csv out_v2_m9/T*.json out_v2_m9/F*.png   results/main_weekly/
> cp out_monthly_m9/T*.csv out_monthly_m9/T*.json out_monthly_m9/F*.png \
>                                                          results/main_monthly/
> cp out_crossmarket_v2/X*.csv out_crossmarket_v2/Y*.csv   results/crossmarket/
> cp out_supply_v4/S*.csv                                  results/supply/
> cp out_v2_n6/T1_accuracy_metrics.csv \
>     results/robustness_placebo6/T1_weekly_placebo6.csv
> cp out_monthly_n6/T1_accuracy_metrics.csv \
>     results/robustness_placebo6/T1_monthly_placebo6.csv
> rm results/*/PLACEHOLDER.txt
> ```
>
> The single-market directories must come from runs with **`--placebo-n 9`**, so
> that the noise block matches the GPR block exactly; the run log confirms this
> with `[ok] placebo dimension-matched`. The two `*_n6` files are the unmatched
> comparison, kept deliberately as robustness evidence — see
> `robustness_placebo6/README.md`.
>
> Two globs are easy to get wrong. `T6_model_confidence_set.json` and
> `T7_fluctuation_test.json` are JSON, so a `T*.csv` glob alone silently skips
> them. And the pipeline's `F1`–`F3` are **PNG** figures, not CSV plot inputs, so
> `F*.csv` matches nothing at all.
>
> Use the **final** run directories: the cross-market run whose log reports
> both `pinball_gain_q95` and `pinball_gain_plc_q95`, and the supply run whose
> log reports `OK: no arm is dominated by isolated weeks`.

Output files from the analysis modules. Every number in the paper traces to a
file here, and where the circulated manuscript and these files disagree, these
files are correct — see the note on pinned library versions in
`../requirements.txt`.

Six directories:

| Directory | Produced by |
|---|---|
| `main_weekly/`, `main_monthly/` | `src/main_pipeline.py` at the two frequencies |
| `crossmarket/` | `src/crossmarket_falsification.py`, then `src/crossmarket_inference.py` |
| `supply/` | `src/supply_interaction.py` |
| `robustness_placebo6/` | the same pipeline with the unmatched 6-series placebo |
| `placebo_randomization/` | `src/placebo_randomization.py` over multi-seed draws |

## `crossmarket/`

Produced by `src/crossmarket_falsification.py`:

| File | Contents |
|---|---|
| `X0_market_oos.csv` | Per-market out-of-sample results: *R*²<sub>OS</sub> by information set, gains, tail statistics, Diebold–Mariano *p*-values |
| `X1_market_results.csv` | `X0` merged with market characteristics and microstructure diagnostics. **This is the file the inference layer and the figure script read.** |
| `X2_rank_correlations.csv` | Cross-sectional rank correlations, gain against market characteristics |
| `X3_skipped_markets.csv` | Markets excluded by the screens, with the reason |
| `X4_sign_tests.csv` | Cross-market sign-consistency tests. The `null_status` column marks each row `descriptive` (control-referenced, nested) or `well-specified` (placebo-referenced) |
| `X5_estimator_matched.csv` | Gains computed holding the estimator fixed across information sets |

Produced by `src/crossmarket_inference.py`:

| File | Contents |
|---|---|
| `Y1_aggregation.csv` | Distribution, independence-based tests, dependence-adjusted Stouffer combinations, effective-*K* sign tests, equivalence bounds |
| `Y2_gradient_exact.csv` | Exact permutation Spearman and Jonckheere–Terpstra tests |
| `Y3_robustness.csv` | Robustness screens |
| `Y4_market_multiplicity.csv` | Market-level test family with Bonferroni, Holm and Benjamini–Hochberg adjustments |
| `Y5_power_curve.csv` | Minimum detectable gradient by simulation |

## `supply/`

Produced by `src/supply_interaction.py`:

| File | Contents |
|---|---|
| `S0_variables.csv` | Variables used, by role, with coverage |
| `S1_oos_forecasts.csv` | The out-of-sample forecast panel |
| `S2_randomization_test.csv` | Randomisation placebo test, per estimator |
| `S2b_placebo_draws.csv` | The placebo null distribution, one row per draw |
| `S2c_insample_interaction.csv` | In-sample HAC interaction coefficients (descriptive) |
| `S2d_forecast_health.csv` | Forecast-health diagnostics per information set |
| `S3_policy_channel.csv` | Policy-response Wald tests with condition numbers |
| `S4_tnac_robustness.csv` | TNAC-week leave-out check |
| `S5_summary.csv` | One row per headline number; each maps to a LaTeX macro |

## `main_weekly/` and `main_monthly/`

Output of `src/main_pipeline.py`, at the two sampling frequencies whose contrast
organises the paper. `src/frequency_figures.py` reads these two directories
directly to build Figures 1–3.

| File | Contents |
|---|---|
| `T00_feature_coverage.csv` | Coverage of every predictor in the modelling window |
| `T0_oos_forecasts.csv` | The out-of-sample forecast panel, all specifications |
| `T0b_model_coverage.csv` | Which specifications produced a forecast in which period |
| `T1_accuracy_metrics.csv` | R2_OS, RMSE, MAE, directional accuracy by specification |
| `T2_clark_west.csv` | Clark–West tests for nested pairs |
| `T3_giacomini_white_cpa.csv` | Conditional predictive ability tests |
| `T4_state_dependent_R2OS.csv` | R2_OS by geopolitical state |
| `T5_romano_wolf.csv` | Romano–Wolf stepdown, FWER-controlled |
| `T6_model_confidence_set.json` | Hansen–Lunde–Nason MCS. **JSON, not CSV** — the MCS is a set plus elimination *p*-values, which does not flatten to a rectangle |
| `T7_fluctuation_test.json` | Giacomini–Rossi fluctuation test, when the out-of-sample window is long enough to compute it. **JSON, not CSV**; absent at the monthly frequency |
| `T8_quantile_tail_evaluation.csv` | Pinball loss by quantile |
| `T8b_tail_inference.csv` | Tail tests with multiplicity correction |
| `T9_economic_value.csv` | Certainty-equivalent return, Sharpe, turnover |
| `T10_block_bootstrap_importance.csv` | Feature importance with bootstrap intervals |
| `T0_oos_forecasts_all.csv` | The forecast panel before the common-sample intersection, written when the quantile arm runs |
| `F1_cssed.png` | Cumulative squared-error difference against the benchmark |
| `F2_r2os.png` | R2_OS by specification |
| `F3_tail_gains.png` | Pinball-loss gains across the quantile grid |

The three `F*.png` files are **figures, not plot inputs**, and their numbering is
internal to the pipeline: it is unrelated to the paper's Figure 1–3, which are
the `F1`–`F3` PDFs in `figures/` produced by `src/frequency_figures.py`.

The frequency contrast is the comparison of `T8b_tail_inference.csv` across
the two directories.

---

## Reading the results correctly

**Two references, two meanings.** Statistics referenced to the control set are
**descriptive**; statistics referenced to the dimension-matched placebo are
**tests**. The control comparison is nested, so under the null of no
incremental information its expectation is negative and a median-zero test has
no size control. Columns and rows are labelled accordingly:

- `pinball_gain_q95` / `gain_gpr_vs_ctrl` → descriptive
- `pinball_gain_plc_q95` / `gain_gpr_vs_placebo` → well-specified

**Levels versus gains.** Out-of-sample *R*² *levels* vary widely across markets
for microstructural reasons unrelated to geopolitical risk: in thinly traded
markets, stale quotes and bid–ask bounce generate return autocorrelation that
mechanically makes momentum controls appear predictive. The interpretable
quantity is the **gain** within each market, which differences that component
out. The microstructure diagnostics in `X1` support this decision.

**Dependence.** All markets are scored against the same global GPR series, so
independence-based *p*-values in `X4` are anti-conservative. Use the
dependence-adjusted versions in `Y1`.

**Health before interpretation.** In `S2_randomization_test.csv`, check the
`status` column and `sse_share_worst1pct` before reading any *R*². An arm
flagged `PATHOLOGICAL` has its *R*² decided by a handful of weeks.
