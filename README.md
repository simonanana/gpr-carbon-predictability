# Geopolitical Risk and Carbon Price Predictability

**Out-of-sample evaluation of geopolitical risk as a carbon-price predictor, under a dimension-matched placebo null.**

Replication code and results for *"Geopolitical Risk and Carbon Price Predictability: An Effect That Disappears"* (Yihan Guo, Nanyang Technological University). Accepted at the 2026 Massey Sustainable Finance Conference.


## What this is

A growing literature reports that geopolitical risk (GPR) improves out-of-sample forecasts of carbon allowance prices. This project re-examines that claim and identifies what produces it.

The headline result is a non-result, and the interest lies in *how* it arises:

| Exercise | Finding |
|---|---|
| **Frequency contrast** | At the monthly frequency conventional in this literature (*N* = 138, 75 out of sample), a GPR block lowers 95th-percentile pinball loss by **3.33%**. Holding the model space, estimators and tests fixed and changing only the sampling frequency to weekly (*N* = 604, 386 out of sample), the sign reverses to **−2.39%**. Neither is significant even nominally (Diebold–Mariano *p* = 0.329 and 0.059); the smallest *p*-value across all fourteen tests is 0.059; and **4 of 7 quantile levels change sign** between the two frequencies. |
| **Dimension-matched placebo** | Replacing the GPR block with an equal number of AR(1) noise series locates the source. Among the 35 modelled weekly specifications, the only one attaining a positive out-of-sample *R*² is the placebo (**+0.14%**), and all three estimators place the GPR block below the matched noise block. At the monthly frequency the same comparison is uninterpretable: a single noise draw moves *R*²<sub>OS</sub> by up to **3.7 percentage points**, against 1.4 points weekly. |
| **Cross-market falsification** | Across nine allowance markets on four continents, the GPR block beats the noise block in five markets and loses in four (exact *p* = 1.00). Inverting the test bounds the mean cross-market improvement above by **+0.67 percentage points** of *R*²<sub>OS</sub> at the one-sided 95% level. |
| **Dimensionality penalty** | Referencing the upper tail to the placebo rather than to the controls measures the cost of appending an uninformative block directly: **1.1 percentage points** of pinball loss. Against the correctly specified benchmark the tail effect disappears (6 of 9 markets, *p* = 0.51). |
| **Supply-side channel** | In a conditional test on 2,998 primary-market auctions, the pricing of supply news is not geopolitically state-dependent against a randomisation-based placebo null (*p*<sub>rand</sub> = 0.73–0.94), and GPR does not predict the policy supply response (min Holm *p* = 1.00). |

The conclusion is not that geopolitical risk is irrelevant to carbon markets, but that the existing evidence does not establish *economically exploitable* predictive value.

## The methodological contribution

Three ideas in this repository are reusable beyond the application, and are documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md):

**1. The dimension-matched placebo.** Adding a block of *k* regressors to a forecasting model has an out-of-sample cost even when the block is pure noise, because the larger model must estimate *k* additional coefficients. Comparing "controls" against "controls + GPR" therefore tests a misspecified null: under the null of no incremental information, the larger model is *expected* to lose. Replacing the candidate block with an equal-cardinality block of AR(1) noise restores a median-zero null and makes the Diebold–Mariano statistic pivotal, because the two specifications are then non-nested and equal-dimensional.

**2. The placebo as a reference distribution.** A single placebo draw is a comparison, not a test: it has no reference distribution, so it cannot produce a *p*-value, and it inherits whatever that one draw happened to do. Across three draws in an early version of the supply-side test, the apparent advantage of the GPR interaction over the noise interaction ranged from +5.0 to +15.8 percentage points while the GPR arm itself barely moved — the entire range came from how badly one noise draw performed. The sensitivity is not confined to interactions: in the single-market analysis, changing the noise block from six series to nine while holding the panel, the seed and every other setting fixed moved the placebo arm's *R*²<sub>OS</sub> by up to 3.7 percentage points monthly and 1.4 weekly, in inconsistent directions across estimators. Drawing *D* blocks turns the comparison into a randomisation test with a genuine *p*-value; `placebo_randomization.py` does this for the single-market pipeline and `supply_interaction.py` has it built in.

**3. Forecast-health diagnostics.** Out-of-sample *R*² is a mean-square statistic, so one catastrophic week moves it by hundreds of percentage points: over ~360 evaluation weeks, a single week with error thirty times the typical magnitude lowers it by roughly 250 points. Every *R*² reported here carries the share of squared error contributed by its worst one per cent of weeks, and an arm whose worst few weeks dominate is flagged rather than interpreted.

## Repository layout

```
.
├── src/
│   ├── main_pipeline.py               Single-market pipeline at either frequency:
│   │                                  feature construction, rolling-origin forecasts,
│   │                                  dimension-matched placebo (T* and F* files)
│   ├── oos_evaluation.py              Evaluation layer: metrics, tests, VaR
│   │                                  backtests, bootstrap importance, plot inputs
│   ├── placebo_randomization.py       Aggregates several placebo draws into a
│   │                                  randomisation p-value, with an
│   │                                  exchangeability check across draws
│   ├── frequency_figures.py           Figures 1-3 and the frequency-contrast table
│   ├── crossmarket_falsification.py   Cross-market out-of-sample evaluation (9 markets)
│   ├── crossmarket_inference.py       Dependence-adjusted inference, exact permutation
│   │                                  tests, equivalence bounds, power analysis
│   ├── crossmarket_figures.py         Appendix figures A4 and A5
│   ├── supply_interaction.py          Supply-side channel: randomisation placebo test,
│   │                                  policy-response test, numerical safeguards
│   ├── fill_latex_macros.py           Writes results into the manuscript's macro block
│   ├── forecast_eval.py               R2_OS, Diebold-Mariano, Clark-West,
│   │                                  Giacomini-White, MCS, Romano-Wolf, CER
│   └── fast_estimator.py              Hyperparameter-cached estimator group
├── results/
│   ├── main_weekly/                   Weekly single-market results (T*, F*)
│   ├── main_monthly/                  Monthly single-market results (T*, F*)
│   ├── crossmarket/                   X0–X5 (per-market), Y1–Y5 (inference layer)
│   ├── supply/                        S0–S5
│   ├── robustness_placebo6/           The accuracy table under the unmatched
│   │                                  6-series placebo, for comparison
│   └── placebo_randomization/         P1–P2, multi-draw randomisation test
│                                      (present once the draws have been run)
├── figures/
│   ├── F1_frequency_contrast.pdf      The frequency reversal (paper Figure 1)
│   ├── F2_placebo_weekly.pdf          Placebo comparison, weekly
│   ├── F2_placebo_monthly.pdf         Placebo comparison, monthly
│   ├── F3_r2_distribution.pdf         R2_OS by model family, both frequencies
│   ├── FA4_crossmarket_gains.pdf      Per-market gains vs controls and vs noise
│   └── FA5_dependence_power.pdf       Dependence sensitivity and gradient-test power
└── docs/
    ├── METHODOLOGY.md                 The three design ideas, stated formally
    ├── REPRODUCIBILITY.md             How to run; what is and is not included
    └── DATA.md                        Sources, licensing, what cannot be redistributed
```

## Quickstart

```bash
pip install -r requirements.txt
```

Three modules are **self-contained**: they need no raw data, only the result
files already in `results/` (see [`results/README.md`](results/README.md)):

```bash
# Dependence-adjusted inference, exact permutation tests, equivalence bounds
python src/crossmarket_inference.py --out results/crossmarket

# Appendix figures
python src/crossmarket_figures.py --xdir results/crossmarket --out figures

# Figures 1-3 and the frequency-contrast table, from the shipped results
python src/frequency_figures.py --weekly results/main_weekly \
    --monthly results/main_monthly --out figures
```

The estimation modules need the constructed panels (see [`docs/DATA.md`](docs/DATA.md)):

```bash
# Single-market pipeline, weekly (~3 min with --fast) and monthly (~20 s).
# --placebo-n must equal the number of GPR variables for a matched placebo;
# the module prints both counts and refuses a frequency/panel mismatch.
python src/main_pipeline.py --panel data/panel_weekly_v2.csv \
    --out out_weekly --placebo-n 9 --fast
python src/main_pipeline.py --panel data/panel_monthly_v2.csv \
    --out out_monthly --freq M --placebo-n 9 --fast

# Multi-draw placebo: one run per seed, then aggregate into a p-value
for s in $(seq 1 50); do
  python src/main_pipeline.py --panel data/panel_monthly_v2.csv \
    --out draws_monthly/seed$s --freq M --placebo-n 9 \
    --placebo-seed $s --fast
done
python src/placebo_randomization.py --dirs "draws_monthly/seed*" \
    --label monthly --out results/placebo_randomization

# Cross-market falsification (~5 min)
python src/crossmarket_falsification.py --data data --out out_crossmarket --fast

# Supply-side channel: inspect variable detection first, then run (~2 min)
python src/supply_interaction.py --panel data/panel_weekly_v2.csv \
    --out out_supply --inspect
python src/supply_interaction.py --panel data/panel_weekly_v2.csv \
    --out out_supply --placebo-draws 50 \
    --news auc_cover_vw,auc_discount,auc_bid_range \
    --volume auc_volume_dlog,auc_n,auc_n_cancelled
```

Naming the columns explicitly is preferred over keyword detection, so that the specification is reproducible from the command line alone.

## Data availability

Raw price data is **not** redistributed here. Allowance prices come from the ICAP Allowance Price Explorer and commercial vendor exports whose terms do not permit redistribution. What this repository contains is the code, the market-level and summary-level results, and the figures — sufficient to verify every inference step, though not to re-estimate forecasts from raw prices. [`docs/DATA.md`](docs/DATA.md) lists every source with its access route.

Two exceptions are freely redistributable and can be downloaded directly: the Caldara–Iacoviello geopolitical risk index, and the Eurostat gas-import series.

## Citation

The working paper is being revised onto the pinned-environment results in
`results/`, and the revised PDF will be attached to the tagged release. Until
then this repository is the authoritative source for every number: where the
circulated manuscript and `results/` disagree, `results/` is correct, and
`requirements.txt` explains why the two can differ.

```bibtex
@unpublished{Guo2026GPRCarbon,
  author = {Guo, Yihan},
  title  = {Geopolitical Risk and Carbon Price Predictability:
            An Effect That Disappears},
  year   = {2026},
  note   = {Working paper, Nanyang Technological University.
            Accepted at the 2026 Massey Sustainable Finance Conference}
}
```

## License

Code is released under the MIT License. Results files and figures are released under CC BY 4.0. See [`LICENSE`](LICENSE).
