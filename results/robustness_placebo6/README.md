# Placebo-size robustness

The main results use `--placebo-n 9`, which matches the cardinality of the GPR
block exactly. The two files here repeat the accuracy table with
`--placebo-n 6`, the unmatched default, at both frequencies:

| File | Produced by |
|---|---|
| `T1_weekly_placebo6.csv` | `main_pipeline.py --panel data/panel_weekly_v2.csv --placebo-n 6 --fast` |
| `T1_monthly_placebo6.csv` | `main_pipeline.py --panel data/panel_monthly_v2.csv --freq M --placebo-n 6 --fast` |

## Two things these files establish

**1. The hyperparameter cache is correctly isolated.** Every row outside the
four `*_CTRL_PLACEBO` specifications is bit-identical to the matched run. That
confirms the per-feature-set namespacing in `src/fast_estimator.py`: the size of
the noise block cannot leak into any other arm, so switching the placebo from
six series to nine changes nothing in `T2`–`T10` and nothing in the benchmark or
GPR rows of `T1`.

**2. A single placebo draw is fragile, and measurably more so at the monthly
frequency.** The `*_CTRL_PLACEBO` rows move as follows (*R*²<sub>OS</sub>,
percentage points, absolute difference between the 6-series and 9-series runs):

| Estimator | Weekly (386 oos) | Monthly (75 oos) |
|---|---|---|
| Ridge | 0.17 | 1.60 |
| ENet | 0.10 | 2.88 |
| PCR | 0.37 | 2.23 |
| XGB | 1.43 | **3.66** |

The monthly placebo is roughly 2.5 times more sensitive, and the direction of
the shift is not consistent across estimators: at the monthly frequency Ridge
and PCR do *better* with nine noise series than with six, while ENet and XGB do
worse. Adding noise regressors is not supposed to help, so the inconsistency is
draw variation rather than a dimensionality effect.

With 75 out-of-sample observations a single draw is not an adequate reference
distribution for a nine-variable block. This is why `src/supply_interaction.py`
uses a 50-draw randomisation test (`results/supply/S2b_placebo_draws.csv`), and
why `src/placebo_randomization.py` exists to do the same for the single-market
pipeline — see `results/placebo_randomization/`.

## What does not change

The qualitative verdict is identical under both placebo sizes at both
frequencies. Weekly, all three estimators place the GPR block below the noise
block under either size. Monthly, two of three place it above under either size.
The size of the noise block changes how large the gap looks; it does not change
its sign.
