# Methodology

This document states the three design ideas in the repository formally, so
that they can be reused independently of the carbon-market application.

---

## 1. The dimension-matched placebo

### The problem

The standard way to ask whether a candidate predictor block carries
information is to compare two nested models:

```
    CTRL              controls only
    CTRL + X          controls plus the candidate block
```

Let *H*₀ be the hypothesis that the block **X** carries no incremental
predictive content. Because `CTRL ⊂ CTRL + X`, the two models are nested, and
under *H*₀ the larger model still estimates *k* additional coefficients whose
population values are zero. The resulting estimation error raises expected
out-of-sample loss (Clark and McCracken 2001; Clark and West 2007). Hence

> **E[gain] < 0 under the null, not zero.**

A test of `median(gain) = 0` therefore has no size control: it rejects when the
null is true. This applies equally to out-of-sample *R*² and to quantile loss.

### The fix

Add a third specification in which the candidate block is replaced by an
equal-cardinality block of uninformative regressors:

```
    CTRL + P          controls plus k AR(1) noise series, x_t = 0.9 x_{t-1} + e_t
```

with **P** drawn independently of the data. Because `dim(P) = dim(X)`, the
estimation-error cost is common to the two specifications and differences out.
Let *H*₀ᵉᵍᵘⁱᵛ be the hypothesis that **X** is exchangeable with a
dimension-matched block of uninformative regressors. Then:

- under *H*₀ᵉᵍᵘⁱᵛ the gain of `CTRL + X` over `CTRL + P` is **median-zero**; and
- `CTRL + P` versus `CTRL + X` is a **non-nested** comparison, so the
  Diebold–Mariano statistic recovers its standard asymptotic distribution
  (Diebold and Mariano 1995; Harvey, Leybourne and Newbold 1997).

The placebo is therefore not merely an economic control for dimensionality. It
is the device that makes the null well-specified and the test statistic
pivotal.

### Consequences for reporting

Any statistic referenced to `CTRL` is **descriptive**; any statistic referenced
to `CTRL + P` is a **test**. The code carries this distinction explicitly: the
`null_status` column of `X4_sign_tests.csv` labels every row as
`descriptive` or `well-specified`, and the module's run log repeats the warning.

The gap between the two references is itself informative. It estimates the
out-of-sample cost of appending an uninformative block of that size — a
quantity that, to our knowledge, has not previously been measured in this
literature. In the cross-market exercise it is **1.1 percentage points** of
upper-tail pinball loss, which is the number a reader should net out before
interpreting any control-referenced tail result.

### Precedent

The logic has a well-known precedent in asset pricing. Novy-Marx (2014) shows
that predictors chosen for their manifest irrelevance — sunspots, planetary
conjunctions, the weather in Manhattan — forecast anomaly returns as
convincingly as economically motivated ones, and concludes that a predictor
must clear a plausibility hurdle as well as a statistical one. The
dimension-matched placebo makes that argument operational rather than
rhetorical: instead of asking whether an absurd predictor also works, it asks
whether the candidate beats a benchmark constructed to have its exact
dimensionality and persistence but none of its content.

---

## 2. The placebo as a reference distribution

A single placebo draw is adequate when the candidate block consists of level
regressors. It is **not** adequate when the block contains interactions.

With three supply surprises and three GPR variables, the interaction block
contributes nine products, so a majority of the added regressors are
second-order terms. Products of noisy regressors are far noisier than their
factors, and the sampling variability of a single noise draw becomes large
relative to the effect being measured.

This is not hypothetical. In an early version of the supply-side test, across
three placebo draws:

| Draw | GPR arm (*R*²<sub>OS</sub>) | Placebo arm | Apparent "gain" |
|---|---|---|---|
| 1 | −7.43% | −14.61% | **+7.19 pp** |
| 2 | −7.36% | −12.40% | **+5.05 pp** |
| 3 | −7.43% | −23.21% | **+15.79 pp** |

The GPR arm barely moves. The entire ten-point range comes from how badly one
particular noise draw happened to perform. A headline number computed from any
single row of that table is measuring the draw, not the predictor.

### The fix

Draw *D* independent placebo blocks, compute the out-of-sample *R*² of each,
and locate the candidate within the resulting empirical null:

```
    p_rand = (1 + #{d ≤ D : R²_d(PLACEBO) ≥ R²(CANDIDATE)}) / (1 + D)
```

Report `p_rand` together with the percentile of the candidate within the null,
the null's mean and standard deviation, and its 5th and 95th percentiles. Do
**not** report a single-draw difference.

The granularity of `p_rand` is 1/(1 + *D*), which matters when a value falls
near a conventional threshold. A standard deviation of the null that is large
relative to the effect being tested is itself a signal that the design is
underpowered or numerically unstable.

---

## 3. Forecast-health diagnostics

Out-of-sample *R*² against a benchmark,

```
    R²_OS = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²
```

is a mean-square statistic. A single catastrophic forecast can therefore move
it by hundreds of percentage points. Over ~360 evaluation weeks:

| Weeks with error *m*× the typical magnitude | Fall in *R*²<sub>OS</sub> |
|---|---|
| 1 week, *m* = 10 | ≈ 28 pp |
| 1 week, *m* = 20 | ≈ 111 pp |
| 1 week, *m* = 30 | ≈ 250 pp |

A reported *R*² of −250% is not a statement about information content. It is a
statement about one week.

### How this manifested

An early version of the supply-side module returned:

```
    Ridge  CTRL      =   −2.04%
    Ridge  CTRL_SUP  = −109.77%     (only three surprises added)
    Ridge  GPRX      = −251.49%
    placebo null: mean −263%, sd 376%, 5th percentile −815%
```

The cause was a chain, not a single error:

1. one auction variable was near-constant over the early training window, so
   its expanding-window standardisation scale approached zero;
2. later realisations of that variable therefore entered the design matrix
   with enormous magnitude;
3. the interaction terms squared the problem;
4. Ridge, whose L2 penalty cannot zero a coefficient, propagated it, while
   ElasticNet's L1 penalty removed the offending term.

A two-order-of-magnitude disagreement between two linear shrinkage estimators
on identical data is a diagnostic, not a finding. The placebo null's standard
deviation of 376 percentage points independently indicated that the entire
reference distribution was unusable.

### The three safeguards

1. **Robust surprises.** AR(1) residuals are standardised by an
   expanding-window standard deviation and winsorised at ±4. Both steps use
   data dated before *t* only, so neither introduces look-ahead. This removes
   the scale pathology at source.

2. **Forecast truncation.** Each forecast is truncated to the [0.5, 99.5]
   percentile range of the training-sample target. The range is computed from
   the training window only, and the rule is applied identically to every
   information set including every placebo draw. Untruncated *R*² is reported
   alongside so the two can be compared.

3. **Health reporting.** Every *R*² is accompanied by the share of
   out-of-sample squared error contributed by its worst one per cent of weeks,
   and by max|ŷ| / sd(*y*). An arm whose worst few weeks account for more than
   half of its squared error is flagged `PATHOLOGICAL` and must not carry a
   headline result.

After these safeguards the same specification returned *R*² of −7.30% with the
worst one per cent of weeks contributing 17.4% of squared error — above the
8–10% a Gaussian error would produce, but an order of magnitude below the level
at which a mean-square statistic ceases to be interpretable.

**Recommendation.** All three safeguards are worth applying whenever an
out-of-sample *R*² is reported for a high-dimensional or interacted
specification, regardless of application.

---

## 4. Cross-market inference under a common shock

When the same design is applied to *K* markets, the informative object is the
cross-market distribution of outcomes rather than any single market's test.
But the markets are not independent: all of them are scored against the same
global GPR series, and their returns share global energy and risk factors. Any
procedure that treats *K* as *K* independent draws overstates significance.

The inference layer (`src/crossmarket_inference.py`) handles this three ways.

**(a) Exact sign and signed-rank tests.** With *K* = 9 both are exact. They
assume independence and are reported for reference only.

**(b) Dependence-adjusted combination.** Let `p_k` be the market-level
two-sided *p*-value and `z_k = sgn(gain_k) · Φ⁻¹(1 − p_k/2)` the signed normal
score. Under equicorrelation ρ̄ among the scores, the Stouffer statistic
becomes

```
    Z(ρ̄) = Σ z_k / sqrt(K + K(K−1)ρ̄)  →  N(0,1)
```

Because ρ̄ is not identified from *K* summary statistics, it is reported over a
grid so the reader can locate the value at which the evidence expires. In the
cross-market exercise, an equicorrelation of only 0.1 raises the upper-tail
*p*-value from 0.023 to 0.089 — and every market is scored against the
identical global series, so a correlation of at least 0.1 is close to certain.

The same logic applied to the sign test replaces *K* by an effective number of
independent markets *K*<sub>eff</sub> ≤ *K*.

**(c) Equivalence bounds instead of a failure to reject.** A null result stated
as *p* > 0.05 is uninformative about magnitude. Inverting the test
(Schuirmann 1987) gives a one-sided upper confidence bound on the mean gain,

```
    ub = mean + t_{0.95,K−1} · (s/√K) · sqrt(1 + (K−1)ρ̄)
```

which converts "no evidence of an effect" into "any mean effect larger than
*ub* is inconsistent with these data."

**Power.** Before interpreting any gradient test, the module reports its
minimum detectable effect by simulation. With nine markets and coarse ordinal
characteristics, 80% power requires a gradient of roughly 2.5 percentage points
per exposure level — larger than the entire observed cross-market range of the
gain (3.69 pp). The gradient test is therefore powered only against effects
larger than anything the data contain, and its point estimates are reported but
not interpreted **in either direction**, including the direction that would
flatter the paper's argument.

With coarse ordinal regressors the code also uses exact permutation *p*-values
rather than asymptotic ones, and supplements Spearman's ρ with the
Jonckheere–Terpstra test, which is the appropriate statistic for a monotone
ordered alternative. It additionally reports the **minimum attainable**
*p*-value given the tie structure: for the high-versus-low exposure comparison,
with two and three markets, that minimum is 1/C(5,2) = 0.100, meaning the test
**cannot** reject at the 5% level no matter what the data look like.

---

## Multiplicity

The cross-market module runs six aggregate sign tests, eighteen rank
correlations, eighteen ordered-alternative tests, one group comparison and
thirty-six market-level quantile tests. Raw *p*-values are reported alongside
Holm (1979) and Benjamini–Hochberg (1995) adjustments within each family, and
no result is described as significant unless it survives correction within its
own family.

---

## References

Benjamini, Y. and Y. Hochberg (1995). "Controlling the False Discovery Rate."
*JRSS-B* 57(1), 289–300.

Clark, T. E. and M. W. McCracken (2001). "Tests of Equal Forecast Accuracy and
Encompassing for Nested Models." *Journal of Econometrics* 105(1), 85–110.

Clark, T. E. and K. D. West (2007). "Approximately Normal Tests for Equal
Predictive Accuracy in Nested Models." *Journal of Econometrics* 138(1),
291–311.

Diebold, F. X. and R. S. Mariano (1995). "Comparing Predictive Accuracy."
*JBES* 13(3), 253–263.

Harvey, D., S. Leybourne and P. Newbold (1997). "Testing the Equality of
Prediction Mean Squared Errors." *IJF* 13(2), 281–291.

Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure."
*Scandinavian Journal of Statistics* 6(2), 65–70.

Jonckheere, A. R. (1954). "A Distribution-Free k-Sample Test against Ordered
Alternatives." *Biometrika* 41(1/2), 133–145.

Novy-Marx, R. (2014). "Predicting Anomaly Performance with Politics, the
Weather, Global Warming, Sunspots, and the Stars." *JFE* 112(2), 137–146.

Schuirmann, D. J. (1987). "A Comparison of the Two One-Sided Tests Procedure
and the Power Approach for Assessing the Equivalence of Average
Bioavailability." *Journal of Pharmacokinetics and Biopharmaceutics* 15(6),
657–680.

Stouffer, S. A. et al. (1949). *The American Soldier.* Princeton University
Press.
