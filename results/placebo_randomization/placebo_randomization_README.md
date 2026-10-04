# Multi-draw placebo randomisation test

Produced by `src/placebo_randomization.py` from several runs of
`src/main_pipeline.py` that differ only in `--placebo-seed`.

| File | Contents |
|---|---|
| `P1_placebo_randomization_weekly.csv` | One row per estimator: the null summary, the randomisation *p*-value and its multiplicity-adjusted versions, weekly |
| `P1_placebo_randomization_monthly.csv` | The same, monthly |
| `P2_placebo_draws_weekly.csv` | The null distribution, one row per draw per estimator, weekly |
| `P2_placebo_draws_monthly.csv` | The same, monthly |

## What the monthly test found

Fifty draws, all four estimators, matched 9-against-9 blocks. The four
`r2_CTRL` and `r2_GPRfull` values agree with `../main_monthly/T1_accuracy_metrics.csv`
to within 2e-5 percentage points, which confirms the draws are the same
specification as the headline table.

| Estimator | *R*²(GPR) | Null mean | Null sd | Null range | Beat GPR | Raw *p* | Holm |
|---|---|---|---|---|---|---|---|
| PCR | −0.43 | −6.74 | 2.50 | 10.75 | 0/50 | 0.0196 | 0.078 |
| Ridge | −5.63 | −13.14 | 6.52 | 22.47 | 7/50 | 0.157 | 0.471 |
| XGB | −13.71 | −13.85 | 5.92 | 32.24 | 21/50 | 0.431 | 0.863 |
| ENet | −0.08 | −2.10 | 5.84 | 32.78 | 28/50 | 0.569 | 0.863 |

Smallest Holm-adjusted *p* = 0.078: **no estimator rejects at 5%**. PCR's raw
0.0196 is the attainable floor at *D* = 50, so it is censored rather than
measured.

Two things this establishes that the single draw could not.

**The monthly null is enormous.** Its standard deviation runs from 2.5 to 6.5
percentage points and its range from 10.8 to 32.8. ENet's null spans −31.3 to
+1.5. Against that spread, a difference of a few points between the GPR arm and
one noise draw carries no information at all.

**The single draw was actively misleading.** Seed 42, used for the headline
table, happened to land 1.88 standard deviations *below* ENet's own null mean
(−13.11 against a mean of −2.10). That one unlucky noise block is the entire
reason the single-draw comparison read "ENet: the GPR block beats noise". Under
50 draws ENet gives *p* = 0.569. The same thing happened in reverse for XGB,
whose seed-42 draw was better than average and so made the GPR block look worse
than the null says it is. Of the three single-draw verdicts at the monthly
frequency, two do not survive.

## What the test does

A single placebo draw is a comparison, not a test. Drawing *D* noise blocks and
treating their out-of-sample *R*² values as the null distribution gives

    p_rand = (1 + #{d : R2(PLACEBO_d) >= R2(GPR)}) / (1 + D)

a valid one-sided randomisation *p*-value for the null that the GPR block
carries no more information than an equal-cardinality block of AR(1) noise. The
`+1` in both numerator and denominator keeps the test exact at finite *D*
(Phipson and Smyth 2010).

## Reading the table

`randomization_p` is **one-sided against the null of no incremental
information**. A *large* *p* means the GPR block is not distinguishable from
noise, which is the paper's conclusion; it is not a failed test.

**Read `holm`, not `randomization_p`, for any claim about a single estimator.**
The raw *p*-value is exact for one estimator, but the table runs the same null
against all four, so the smallest of the four is a minimum over a family rather
than a 5% test. One of four landing under 0.05 is close to what the family
produces under the null. `bonferroni`, `holm` (step-down, FWER) and `bh_q`
(Benjamini–Hochberg, FDR) are all reported because they answer different
questions; quoting whichever happens to be smallest is the error.

Read `placebo_range_pp` next. It is the spread of the null across draws, and it
says how much arbitrary variation any single-draw comparison would have
inherited. Where that range is wide relative to the GPR-versus-control gap, the
single-draw comparison was never informative in either direction.

`gpr_percentile_in_null` locates the GPR arm inside the null. Near 50 means the
GPR block performs like a typical noise block of the same size.

### A p-value pinned at the floor is censored, not precise

With *D* draws the smallest attainable *p* is 1/(1+*D*). An estimator whose
`n_draws_beating_gpr` is 0 sits exactly at that floor, and the module says so.
At *D* = 50 the floor is 0.0196, and the test cannot distinguish that from an
arbitrarily small value: resolving below 0.01 needs *D* ≥ 99. Such a figure must
not be quoted as though it were a measured *p*-value.

## Number of draws

With *D* draws the smallest attainable *p*-value is 1/(1+*D*). Below about ten
draws a non-rejection is uninformative rather than evidence for the null, and
the module prints a warning saying so. Monthly runs take roughly 20 seconds
each, weekly roughly three minutes with `--fast`, so the weekly test is normally
run with fewer draws than the monthly one; `n_draws` in `P1` records how many
were actually used.

## Exchangeability

The module verifies that every non-placebo row is identical across draw
directories before computing anything, and exits naming the offending models if
not. Draws that differ by more than the seed — a changed panel, a changed flag,
a different library version — are not exchangeable, and a *p*-value computed
from them would be meaningless.
