# Results

> **Before first publish:** copy the two output directories from your final
> runs into this folder. Nothing is shipped pre-populated, so that the
> repository can never contain superseded numbers.
>
> ```bash
> cp out_crossmarket_v2/X*.csv  results/crossmarket/
> cp out_crossmarket_v2/Y*.csv  results/crossmarket/
> cp out_supply_v4/S*.csv       results/supply/
> ```
>
> Use the **final** run directories: the cross-market run whose log reports
> both `pinball_gain_q95` and `pinball_gain_plc_q95`, and the supply run whose
> log reports `OK: no arm is dominated by isolated weeks`.

Output files from the two estimation modules. Every number in the paper's
cross-market and supply-side sections traces to a file here.

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

Produced by `src/supply_interaction.py`

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
