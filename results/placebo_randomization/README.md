# Multi-draw placebo randomisation test

Produced by `src/placebo_randomization.py` from runs of `src/main_pipeline.py`
that differ only in `--placebo-seed`: 20 draws weekly, 50 monthly, with the noise
block matched to the GPR block at 9 series in every run.

| File | Contents |
|---|---|
| `P1_placebo_randomization_weekly.csv` | One row per estimator: the null summary, the randomisation *p*-value and its multiplicity-adjusted versions, weekly |
| `P1_placebo_randomization_monthly.csv` | The same, monthly |
| `P2_placebo_draws_weekly.csv` | The null distribution, one row per draw per estimator, weekly |
| `P2_placebo_draws_monthly.csv` | The same, monthly |

## What the test does

A single placebo draw is a comparison, not a test. Drawing *D* noise blocks and
treating their out-of-sample *R*² values as the null distribution gives

    p_rand = (1 + #{d : R2(PLACEBO_d) >= R2(GPR)}) / (1 + D)

a valid one-sided randomisation *p*-value for the null that the GPR block
carries no more information than an equal-cardinality block of AR(1) noise. The
`+1` in both numerator and denominator keeps the test exact at finite *D*
(Phipson and Smyth 2010).

## Results

*R*²<sub>OS</sub> in percentage points. `pct` is where the GPR arm falls inside
its own null.

**Weekly, *D* = 20, 386 out-of-sample observations**

| Estimator | *R*²(CTRL) | *R*²(GPR) | Null mean | Null sd | Null range | Beat GPR | pct | Raw *p* | Holm |
|---|---|---|---|---|---|---|---|---|---|
| Ridge | −2.45 | −3.15 | −3.64 | 0.75 | 2.89 | 3/20 | 85 | 0.190 | 0.762 |
| PCR | −1.96 | −1.19 | −1.39 | 0.46 | 1.71 | 8/20 | 60 | 0.429 | 1.000 |
| ENet | −0.15 | −0.63 | −0.40 | 0.44 | 1.54 | 16/20 | 20 | 0.810 | 1.000 |
| XGB | −3.87 | −7.23 | −5.86 | 1.40 | 4.75 | 16/20 | 20 | 0.810 | 1.000 |

**Monthly, *D* = 50, 75 out-of-sample observations**

| Estimator | *R*²(CTRL) | *R*²(GPR) | Null mean | Null sd | Null range | Beat GPR | pct | Raw *p* | Holm |
|---|---|---|---|---|---|---|---|---|---|
| PCR | −6.78 | −0.43 | −6.74 | 2.50 | 10.75 | 0/50 | 100 | 0.0196 | 0.078 |
| Ridge | −16.61 | −5.63 | −13.14 | 6.52 | 22.47 | 7/50 | 86 | 0.157 | 0.471 |
| XGB | −10.25 | −13.71 | −13.85 | 5.92 | 32.24 | 21/50 | 58 | 0.431 | 0.863 |
| ENet | +0.86 | −0.08 | −2.10 | 5.84 | 32.78 | 28/50 | 44 | 0.569 | 0.863 |

**Headline: no estimator rejects at either frequency.** Smallest Holm-adjusted
*p* is 0.762 weekly and 0.078 monthly. The GPR arm scatters through the middle
of its null — percentiles 20 to 85 weekly, 44 to 100 monthly — which is what the
null predicts.

Integrity: every `r2_CTRL` and `r2_GPRfull` above matches
`../main_weekly/T1_accuracy_metrics.csv` and
`../main_monthly/T1_accuracy_metrics.csv` to within 2e-5 percentage points, and
the module's exchangeability check passed on all 70 draws. The draws are the same
specification as the headline tables.

## Three things the single draw could not have told you

**1. The monthly null is 4 to 13 times wider than the weekly one.** Standard
deviations run 2.5–6.5 percentage points monthly against 0.4–1.4 weekly (median
ratio 7.1×); ranges 10.8–32.8 against 1.5–4.8. Sample size alone predicts
√(386/75) = 2.3×, so roughly two thirds of the gap is something else: the
monthly model estimates 20 coefficients on 75 out-of-sample observations, and the
estimation-error component of *R*²<sub>OS</sub> grows faster than √*N*. Against a
null that wide, no single-draw comparison at the monthly frequency carries
information in either direction.

**2. Seed 42 was atypical, in opposite directions at the two frequencies.** The
draw behind the specification tables sits 2.15, 1.23 and 2.30 standard
deviations *above* the null mean for Ridge, ENet and XGB weekly — an unusually
strong noise block — and 1.88 standard deviations *below* it for ENet monthly, an
unusually weak one. That one draw is the whole reason the single-draw comparison
read "the GPR block loses to noise" weekly and "beats noise" monthly. The
inconsistency between frequencies was draw luck, not a frequency effect, and the
randomisation test removes it: nothing is distinguishable from noise anywhere.

**3. PCR's monthly result does not replicate.** PCR is the only cell anywhere
near significance (monthly raw *p* = 0.0196, 0 of 50 draws beating the GPR arm),
and weekly the same estimator gives *p* = 0.429 with the GPR arm at the 60th
percentile. Principal component regression is the estimator most damaged by
appending uninformative dimensions, because nine independent noise series spread
variance across nine new component directions while the nine GPR variables are
mutually correlated and collapse into far fewer. That asymmetry is an artefact of
the dimension reduction, not information about geopolitical risk, and the failure
to replicate at the frequency with five times the data is what identifies it as
such.

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
inherited.

`gpr_percentile_in_null` locates the GPR arm inside the null. Near 50 means the
GPR block performs like a typical noise block of the same size.

### A p-value pinned at the floor is censored, not precise

With *D* draws the smallest attainable *p* is 1/(1+*D*). An estimator whose
`n_draws_beating_gpr` is 0 sits exactly at that floor, and the module says so.
Monthly PCR is such a case: at *D* = 50 the floor is 0.0196, and the test cannot
distinguish that from an arbitrarily small value, so the figure must not be
quoted as though it were measured. Resolving below 0.01 would need *D* ≥ 99.

No weekly cell is at the floor (1/21 = 0.0476; the smallest raw *p* is 0.190), so
20 draws are adequate there and more would not change the conclusion.

## Number of draws

With *D* draws the smallest attainable *p* is 1/(1+*D*), so below about ten draws
a non-rejection is uninformative rather than evidence for the null, and the
module warns. Monthly runs take roughly 20 seconds each and weekly roughly three
minutes with `--fast`, which is why the monthly test uses more draws; `n_draws`
in `P1` records how many were actually used.

## Exchangeability

The module verifies that every non-placebo row is identical across draw
directories before computing anything, and exits naming the offending models if
not. Draws that differ by more than the seed — a changed panel, a changed flag, a
different library version — are not exchangeable, and a *p*-value computed from
them would be meaningless.
