# Reproducibility

## Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11 with numpy 2.x, pandas 2.x, scipy 1.x, scikit-learn 1.x
and matplotlib 3.x.

---

## What runs out of the box

Three modules are **self-contained** — they depend only on the packages above
and on files already in `results/`, with no raw data required:

| Module | Input | Runtime |
|---|---|---|
| `crossmarket_inference.py` | `results/crossmarket/X1_market_results.csv` | < 2 min |
| `crossmarket_figures.py` | same | < 10 s |
| `frequency_figures.py` | `results/main_{weekly,monthly}/T8b_tail_inference.csv` and `T1_accuracy_metrics.csv` | < 10 s |

```bash
python src/crossmarket_inference.py --out results/crossmarket
python src/crossmarket_figures.py --xdir results/crossmarket --out figures
python src/frequency_figures.py --weekly results/main_weekly \
    --monthly results/main_monthly --out figures
```

`crossmarket_inference.py` regenerates every statistic in the paper's
cross-market section that is not already in the raw per-market output:
dependence-adjusted Stouffer combinations, effective-*K* sign tests,
equivalence bounds, exact permutation Spearman and Jonckheere–Terpstra tests,
the minimum-detectable-gradient simulation, robustness screens, leave-one-out
means, and the multiplicity-adjusted market-level test family. It writes
`Y1`–`Y5`.

`supply_interaction.py` is also self-contained in its *dependencies* (numpy,
pandas, scipy, scikit-learn only) but requires the constructed weekly panel as
input.

---

## What is included

All eight analysis modules plus their two dependencies are in `src/`:

| Module | Role | External deps |
|---|---|---|
| `main_pipeline.py` | Single-market pipeline, either frequency | `oos_evaluation`, `forecast_eval`, `fast_estimator` |
| `oos_evaluation.py` | Evaluation layer for the single-market pipeline | `forecast_eval` |
| `frequency_figures.py` | Figures 1–3, frequency-contrast table | + matplotlib |
| `crossmarket_falsification.py` | Cross-market out-of-sample evaluation | `forecast_eval`, `fast_estimator` |
| `crossmarket_inference.py` | Dependence-adjusted inference layer | numpy/pandas/scipy |
| `crossmarket_figures.py` | Appendix figures | + matplotlib |
| `supply_interaction.py` | Supply-side channel | + scikit-learn |
| `fill_latex_macros.py` | Writes results into the manuscript | pandas |
| `forecast_eval.py` | R2_OS, DM, Clark-West, Giacomini-White, MCS, Romano-Wolf, CER | numpy/scipy |
| `fast_estimator.py` | Hyperparameter-cached estimator group | scikit-learn, **xgboost** |

`supply_interaction.py` deliberately implements its own estimator and
statistics, so it depends on neither helper module.

### A note on how `dm_test` is called

`crossmarket_falsification.py` passes **pinball losses** to
`forecast_eval.dm_test` with `power=1`, not forecast errors. This is
deliberate and correct: pinball loss is non-negative, so
`|L_a|^1 - |L_b|^1` reduces exactly to the loss differential `L_a - L_b`.
Passing losses with the default `power=2` would square them and is **not**
equivalent.

## What is NOT in this repository

### The raw price panels

The two constructed panels (`data/panel_weekly_v2.csv` for the single-market
and supply-side modules, and the per-market files read by
`crossmarket_falsification.py`) are **not** redistributed, because allowance
prices come from the ICAP Allowance Price Explorer and commercial vendor
exports whose terms do not permit redistribution. `docs/DATA.md` lists every
source with its access route and the construction steps, and the two
redistributable inputs — the Caldara–Iacoviello GPR index and the Eurostat
gas-import series — can be downloaded directly.

What this means in practice: the inference layer and every figure can be
regenerated from the shipped `results/`, and the estimation code is complete
and auditable, but re-estimating the forecasts from raw prices requires
rebuilding the panels from the sources named in `docs/DATA.md`.

### A superseded feature-construction routine

An earlier version of the single-market pipeline contained its own
`build_features()` and command-line entry point, which `main_pipeline.py`
supersedes. Those routines are **not** included, and `oos_evaluation.py`
carries a scope note saying so. They had three defects that would make a
reader's output disagree with the paper: the weekly panel was resampled to
monthly before modelling; `combine_first` applied to the long multi-market
panel inflated the row count; and the control set was truncated to three
variables. Only the evaluation layer from that version — metrics, tests, VaR
backtests, bootstrap importance and plot inputs, all of which the paper does
use — is retained, in `oos_evaluation.py`.

---

## Full pipeline

```bash
# 0. Single-market pipeline at both frequencies.
#    --placebo-n sets the size of the AR(1) noise block. It MUST equal the
#    number of GPR variables in CTRL_GPRfull, or the placebo is not
#    dimension-matched; the module prints the two counts and warns if they
#    differ. The default of 6 reproduces the originally circulated run.
python src/main_pipeline.py --panel data/panel_weekly_v2.csv \
    --out out_v2 --placebo-n 9                     # ~25 min, ~7 min with --fast
python src/main_pipeline.py --panel data/panel_weekly_v2.csv \
    --out out_monthly --freq M --placebo-n 9

#    Figures 1-3 and the frequency-contrast table
python src/frequency_figures.py --weekly out_v2 --monthly out_monthly \
    --out figures

# 1. Cross-market falsification, ~5 min with --fast
python src/crossmarket_falsification.py --data data --out out_crossmarket --fast

#    Observation-threshold sensitivity. The paper reports both settings;
#    they yield the same nine markets, because the series admitted at the
#    lower threshold then fails the out-of-sample screen.
python src/crossmarket_falsification.py --data data \
    --out out_crossmarket_min100 --fast --min-obs 100

# 2. Inference layer
python src/crossmarket_inference.py --out out_crossmarket

# 3. Figures
python src/crossmarket_figures.py --xdir out_crossmarket --out figures

# 4. Supply-side channel. Check variable detection first.
python src/supply_interaction.py --panel data/panel_weekly_v2.csv \
    --out out_supply --inspect

python src/supply_interaction.py --panel data/panel_weekly_v2.csv \
    --out out_supply --placebo-draws 50 \
    --news auc_cover_vw,auc_discount,auc_bid_range \
    --volume auc_volume_dlog,auc_n,auc_n_cancelled

# 5. Write results into the manuscript's macro block
python src/fill_latex_macros.py --supply out_supply --tex paper/manuscript.tex
```

---

## Checks to run before trusting the output

The modules self-report several conditions. Each one below has bitten this
project at least once.

**Single-market pipeline:**

- The feature-set summary prints the size of `CTRL_PLACEBO` and of
  `CTRL_GPRfull`. The difference between the two is the size of the noise block
  versus the size of the GPR block, and they must be **equal**. If they are
  not, the module prints `PLACEBO IS NOT DIMENSION-MATCHED` with both counts and
  the `--placebo-n` value needed. An unmatched placebo does not invalidate the
  direction of the result but it does change its interpretation: the smaller
  block estimates fewer coefficients, pays a smaller estimation-error penalty,
  and is therefore favoured, so "the GPR block fails to beat noise" is then
  partly a statement about dimension rather than about information.

**Cross-market:**

- `feature sets: ... (placebo and GPR blocks dimension-matched: k == k)`.
  If the two differ, the entire placebo design is invalid and the module exits.
- Both tail families must appear in the output. If
  `pinball_gain_q95` or `pinball_gain_plc_q95` is missing, the module prints an
  explicit error: the paper's tail columns, dependence panels, robustness table
  and both appendix figures all depend on them.
- `envelope attained by different estimators across information sets in: [...]`
  names the markets where the headline gain is not a paired comparison.
  `X5_estimator_matched.csv` gives the paired version.

**Supply-side:**

- `controls (7)`. A smaller control set means the baseline is too weak and
  every downstream gain is measured against the wrong benchmark. Run
  `--list-columns` to inspect names.
- Every auction-news column must be prefixed `auc`. Keyword detection on
  substrings such as "cover" will otherwise pick up the country-coverage
  variables of the exposure-weighted GPR index, which are not auction data.
  The module refuses to continue if this happens.
- The surprise summary table should show standard deviations near 1 and
  min/max within ±4. Values outside that range mean winsorisation did not take
  effect.
- `OK: no arm is dominated by isolated weeks`. If any arm is flagged
  `PATHOLOGICAL`, its *R*² is decided by a handful of weeks and must not carry
  a headline result.
- A placebo null with standard deviation above 20 pp is itself unusable for
  inference, and the module says so.

---

## Determinism

The cross-market module runs the point-forecast and quantile arms with two
**independent** estimator instances and separate key spaces. This matters: the
acceleration layer refreshes hyperparameters on a global call counter, so a
single shared instance makes each arm's refresh schedule depend on how many
calls the other arm made. In an earlier version that shared one instance,
adding a third quantile information set moved one market's control-only *R*²
by 0.87 percentage points — larger than any effect the module reports —
without a single change to that specification. Per-market results would then
depend on which other models happened to run in the same session.

The same reasoning applies inside `supply_interaction.py`, where every
information set gets its own estimator instance.

Placebo blocks are generated from `numpy.random.default_rng` with an explicit
seed (`--placebo-seed`, `--seed`), so runs are reproducible. The randomisation
test uses `seed0 + d` for draw *d*.

---

## Known behaviour

- `--fast` disables the random-forest estimator in the cross-market module. All
  reported results use `--fast`.
- Markets are screened on usable observations (`--min-obs`, default 150) and
  out-of-sample periods (`--min-oos`, default 60). Screens are applied to
  observation counts only, never to outcomes.
- `EUA_pre19` is excluded by default: it reports primary-market clearing prices
  for the same underlying instrument as `EUA` and would double-count the
  European market in the cross-section.
- The policy-response test excludes level quantities (`auc_volume`,
  `auc_revenue`) by default. On the untransformed level series the design is
  ill-conditioned and the Wald statistic is meaningless — an earlier version
  returned χ² ≈ 1.4 × 10⁴ with a coefficient sum of zero to four decimal places,
  and the identical statistic for two different regressors.
  `--include-level-volume` restores them, and the condition number of every
  design is reported either way.
