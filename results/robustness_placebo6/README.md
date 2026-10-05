# Placebo-size robustness

The main results use `--placebo-n 9`, which matches the cardinality of the GPR
block exactly. The two files here repeat the accuracy table with
`--placebo-n 6`, the unmatched default, at both frequencies:

| File | Produced by |
|---|---|
| `T1_weekly_placebo6.csv` | `main_pipeline.py --panel data/panel_weekly_v2.csv --out out_v2_n6 --placebo-n 6 --fast` |
| `T1_monthly_placebo6.csv` | `main_pipeline.py --panel data/panel_monthly_v2.csv --out out_monthly_n6 --freq M --placebo-n 6 --fast` |

Each file is that run's `T1_accuracy_metrics.csv`, renamed.

## What these files are for, and what they are not for

They answer one question: **does the conclusion depend on whether the noise block
is matched to the GPR block?** It does not. At both frequencies and under both
sizes the qualitative verdict is the same, so a reader comparing against the
originally circulated numbers — which used the unmatched 6-series default — can
see that the switch to a matched 9 changed the design's correctness without
changing its answer.

They are **not** the evidence that a single draw is fragile. That argument now
belongs entirely to `../placebo_randomization/`, which holds the block size fixed
at 9 and varies the draw across 20 weekly and 50 monthly replications. Comparing
6 series against 9 confounds two things — the size of the block and the
particular noise realisation — and a two-point comparison cannot separate them.
The randomisation test does, and its null is far more informative: the monthly
null has a standard deviation of 2.5 to 6.5 percentage points, which dwarfs
anything visible here.

## What these files do establish on their own

**The hyperparameter cache is correctly isolated.** Every row outside the four
`*_CTRL_PLACEBO` specifications is bit-identical between the matched and
unmatched runs. That confirms the per-feature-set namespacing in
`src/fast_estimator.py`: the size of the noise block cannot leak into any other
arm, so `T2`–`T10` and the benchmark and GPR rows of `T1` are untouched by the
choice. The exchangeability check in `placebo_randomization.py` relies on exactly
this property and verified it again across all 70 draws.

For the record, the `*_CTRL_PLACEBO` rows move as follows between the two sizes
(absolute difference, percentage points):

| Estimator | Weekly (386 oos) | Monthly (75 oos) |
|---|---|---|
| Ridge | 0.17 | 1.60 |
| ENet | 0.10 | 2.88 |
| PCR | 0.37 | 2.23 |
| XGB | 1.43 | 3.66 |

The monthly column is larger, which is consistent with the much wider monthly
null measured properly in `../placebo_randomization/`. But read that directory
for the magnitude; these four numbers are two draws of different sizes, not a
distribution.
