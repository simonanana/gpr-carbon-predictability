"""
=============================================================================
supply_interaction.py -- the supply-side channel
=============================================================================
Tests whether the pricing of primary-market supply news is geopolitically
state-dependent, and whether geopolitical risk (GPR) predicts the policy
supply response.

WHY AUCTIONS AND NOT TNAC
-------------------------
The natural test of the supply-side story is an event study around
publication of the Total Number of Allowances in Circulation (TNAC),
interacted with GPR. That test is not viable here:

  * TNAC has been published once a year since 2017, so the sample window
    contains at most ten events; partitioning them by geopolitical state
    gives four against five. A paper arguing that findings at N=138 do not
    survive cannot rest an extension on nine events.
  * There is no surprise measure. TNAC is an annual step function and no
    consensus-forecast series exists from which to extract its unanticipated
    component, so the interaction coefficient cannot be signed.
  * The "two offsetting channels" account, as usually stated, is
    unfalsifiable, and offering it as an explanation of one's own null is the
    practice this project criticises elsewhere.

The hand-collected EEX primary-market auction record answers the same
economic question with an instrument that has power: roughly six hundred
weekly observations instead of nine annual events, auction *timing* published
a year in advance (hence exogenous to the week's news), and auction
*outcomes* that are quantitatively signed.

THE FOUR TESTS
--------------
  [A] Supply-news surprises. Expanding-window AR(1) residuals of each auction
      outcome, standardised and winsorised. No look-ahead: the AR coefficients
      and the scaling standard deviation both use data dated before t only.

  [B] Out-of-sample interaction test. Four information sets:
          CTRL
          CTRL + SUPPLY  (+ a no-auction indicator)
          CTRL + SUPPLY + (SUPPLY x GPR)
          CTRL + SUPPLY + (SUPPLY x NOISE)   <- dimension-matched placebo
      The fourth has the same cardinality *and* the same product structure as
      the third, so the comparison is non-nested and equal-dimensional: the
      null is median-zero and standard inference applies. A comparison against
      CTRL + SUPPLY would be nested and its null misspecified.

      A single placebo draw is not enough. With three supply surprises and
      three GPR variables the interaction block contributes nine products, and
      products of noisy regressors are far noisier than their factors. In an
      earlier version the single-draw advantage of the GPR interaction over
      the noise interaction ranged from +5.0 to +15.8 percentage points of
      out-of-sample R-squared across three draws while the GPR arm itself
      barely moved: the entire range came from how badly one noise draw
      happened to perform. This module therefore draws D placebo blocks and
      reports a randomisation p-value,

          p_rand = (1 + #{d : R2_d(PLACEBO) >= R2(GPR)}) / (1 + D)

      which is the natural completion of the dimension-matched placebo idea:
      the placebo becomes a reference distribution rather than a single
      comparison.

  [C] The policy-response channel, made falsifiable. If geopolitical shocks
      provoked a supply-side policy response, GPR would predict auction
      volume. Wald test with Newey-West covariance on

          v_t = a + sum_j b_j v_{t-j} + sum_j c_j g_{t-j} + u_t,
          H0: c_1 = ... = c_L = 0

      with Holm correction across the family. Failure to reject does not prove
      the channel absent, but it does mean the two-channel account cannot be
      offered as an explanation of a null without evidence.

  [D] TNAC weeks as a leave-out robustness check -- never as an event study.

NUMERICAL ROBUSTNESS
--------------------
Out-of-sample R-squared is a mean-square statistic, so a single catastrophic
forecast moves it by hundreds of percentage points: over ~360 evaluation
weeks, one week with a forecast error thirty times the typical magnitude
lowers it by roughly 250 percentage points on its own. Reported values in that
range are not statements about information content; they are statements about
one week.

An earlier version of this module hit exactly that. One auction variable had a
near-constant early training window, so its expanding-window standardisation
scale approached zero and later realisations entered the design matrix with
enormous magnitude. Ridge, whose L2 penalty cannot zero a coefficient,
returned R-squared below -100% and a placebo null distribution with a standard
deviation of several hundred percentage points, while ElasticNet, whose L1
penalty removed the offending term, stayed within a few percent. A
two-order-of-magnitude disagreement between two linear shrinkage estimators on
identical data is a diagnostic, not a finding.

Three safeguards follow, and all three are recommended whenever an
out-of-sample R-squared is reported for a high-dimensional or interacted
specification:

  1. Surprises are standardised by an expanding-window standard deviation and
     winsorised at +/- 4, removing the scale pathology at source.
  2. Forecasts are truncated to the [0.5, 99.5] percentile range of the
     training-sample target -- a no-look-ahead rule applied identically to
     every information set including every placebo draw. Untruncated
     R-squared is reported alongside so the two can be compared.
  3. Every R-squared is accompanied by the share of out-of-sample squared
     error contributed by its worst one per cent of weeks. An arm whose worst
     few weeks account for more than half of its squared error is flagged
     PATHOLOGICAL and must not carry a headline result.

Weeks with no auction (August volumes are halved and year-end auctions
suspended) carry an explicit indicator, and their supply surprises are set to
zero rather than inherited from a carried-forward price.

OUTPUTS (in --out)
------------------
    S0_variables.csv             variables used, by role, with coverage
    S1_oos_forecasts.csv         the out-of-sample forecast panel
    S2_randomization_test.csv    test [B]
    S2b_placebo_draws.csv        the placebo null distribution
    S2c_insample_interaction.csv in-sample HAC interaction coefficients
    S2d_forecast_health.csv      forecast-health diagnostics
    S3_policy_channel.csv        test [C]
    S4_tnac_robustness.csv       test [D]
    S5_summary.csv               one row per headline number

USAGE
-----
    # 1) check variable detection first (expect 7 controls, auction news all
    #    prefixed 'auc')
    python src/supply_interaction.py --panel data/panel_weekly_v2.csv \
        --out out_supply --inspect

    # 2) full run, ~2 minutes with 50 placebo draws
    python src/supply_interaction.py --panel data/panel_weekly_v2.csv \
        --out out_supply --placebo-draws 50 \
        --news auc_cover_vw,auc_discount,auc_bid_range \
        --volume auc_volume_dlog,auc_n,auc_n_cancelled

Naming the columns explicitly is preferred over keyword detection, so that the
specification is reproducible from the command line alone.

This module is self-contained: it depends only on numpy, pandas, scipy and
scikit-learn.
=============================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- variable detection rules ------------------------------------------------
AUCTION_PREFIX = "auc"
# Columns a naive keyword match would wrongly catch. gpr_exp_coverage and
# exp_covered are the country coverage of the exposure-weighted GPR index, not
# auction data, and the substring "cover" would otherwise pick them up.
NOT_AUCTION = {"gpr_exp_coverage", "exp_covered", "gpr_exp_cov",
               "coverage", "covered_share"}
# Level quantities of order 1e8 whose own lags are nearly collinear, because
# volume is largely determined by a published calendar. Excluded from the
# policy channel by default; --include-level-volume restores them.
LEVEL_VOLUME = {"auc_volume", "auc_revenue"}
VOLUME_HINTS = ("volume", "auc_n", "qty", "quantity")
NEWS_PREFERRED = ["auc_cover_vw", "auc_discount", "auc_bid_range",
                  "auc_cover_min", "auc_px_vs_median", "auc_bidders"]
GPR_CORE_PREFERRED = ["gprd_surprise", "gprd_dlog", "gprd_log"]

# Exogenous controls: (tag, candidate column names, take logs?)
EXOG_CTRL = [
    ("brent",  ["brent_ret", "brent_m_ret"], False),
    ("natgas", ["natgas_ret", "natgas_m_ret"], False),
    ("coal",   ["coal_ret", "coal_m_ret"], False),
    ("vstoxx", ["vstoxx_close", "vstoxx_m"], False),
    ("cpu",    ["cpu_log", "cpu__cpu_index_narrow", "cpu_index_narrow"], "auto"),
]


# =============================================================================
# 0. Statistics
# =============================================================================
def r2_os(y, yhat, bench) -> float:
    """Campbell-Thompson out-of-sample R-squared against a benchmark forecast."""
    y, yhat, bench = map(np.asarray, (y, yhat, bench))
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - bench) ** 2)


def hac_ols(y, X, lags: int | None = None):
    """OLS with Newey-West covariance. X must already include a constant.
    Returns (beta, se, t, p, V, condition_number_of_XtX)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    XtX = X.T @ X
    cond = float(np.linalg.cond(XtX))
    Xi = np.linalg.pinv(XtX)
    b = Xi @ X.T @ y
    e = y - X @ b
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        A = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += (1 - L / (lags + 1)) * (A + A.T)
    V = Xi @ S @ Xi
    se = np.sqrt(np.maximum(np.diag(V), 1e-18))
    t = b / se
    return b, se, t, 2 * (1 - st.t.cdf(np.abs(t), n - k)), V, cond


def holm(pvals) -> np.ndarray:
    """Holm step-down multiplicity adjustment."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for i, j in enumerate(np.argsort(p)):
        prev = min(max(prev, (m - i) * p[j]), 1.0)
        adj[j] = prev
    return adj


# =============================================================================
# 1. Variable detection and self-constructed own-state controls
# =============================================================================
def dedupe(df, cols, thresh=0.999):
    """Drop near-duplicate columns. Some panels carry the same series under two
    names, which would otherwise enter the design matrix twice."""
    keep, dropped = [], []
    for c in cols:
        dup = False
        for k in keep:
            s = df[[c, k]].dropna()
            if len(s) > 30 and abs(np.corrcoef(s[c], s[k])[0, 1]) > thresh:
                dup, _ = True, dropped.append((c, k))
                break
        if not dup:
            keep.append(c)
    for c, k in dropped:
        print(f"    - dropped duplicate column {c} "
              f"(correlation with {k} > {thresh})")
    return keep


def build_own_state(df: pd.DataFrame, target: str) -> list[str]:
    """
    Construct momentum and realised volatility from the return series itself
    rather than relying on column-name detection, so the control set matches
    the paper's main specification regardless of panel naming.

    The target is y_t = r_{t+1}, so the return realised as of t is r_t = y_{t-1}:

        own_mom4_t = sum_{j=1..4}  y_{t-j}
        own_rv12_t = sd_{j=1..12}  y_{t-j}

    Both use only information published at or before t, so there is no
    look-ahead. If the panel already carries a 'ret' column it is used instead,
    which is equivalent.
    """
    if "ret" in df.columns and df["ret"].notna().mean() > 0.9:
        base = df["ret"]
        note = "ret"
    else:
        base = df[target].shift(1)
        note = f"{target}.shift(1)"
    df["own_mom4"] = base.rolling(4).sum()
    df["own_rv12"] = base.rolling(12).std()
    print(f"    - built own_mom4 / own_rv12 (from {note}, no look-ahead)"
          f"  coverage {df['own_mom4'].notna().mean():.2f} / "
          f"{df['own_rv12'].notna().mean():.2f}")
    return ["own_mom4", "own_rv12"]


def detect(df, target, min_cov=0.60, news_override=None, vol_override=None,
           include_level_volume=False):
    cov = df.notna().mean()
    num = set(df.select_dtypes(include=[np.number]).columns)

    def ok(c):
        return c in num and cov.get(c, 0) >= min_cov

    ctrl, missing = [], []
    for tag, cands, logit in EXOG_CTRL:
        hit = next((c for c in cands if ok(c)), None)
        if hit is None:
            missing.append(tag)
            continue
        # Climate policy uncertainty enters the paper in logs; the panel may
        # carry it as a level of order 100, which would otherwise dominate the
        # penalty structure of the shrinkage estimators after standardisation.
        if logit == "auto" and not hit.endswith("_log") and df[hit].min() > 0 \
                and df[hit].median() > 5:
            df[f"{hit}_log"] = np.log(df[hit].clip(lower=1e-6))
            print(f"    - {hit} taken in logs -> {hit}_log "
                  f"(original median {df[hit].median():.0f})")
            hit = f"{hit}_log"
        ctrl.append(hit)
    ctrl += build_own_state(df, target)

    auction = [c for c in df.columns
               if c.lower().startswith(AUCTION_PREFIX)
               and c.lower() not in NOT_AUCTION and ok(c)]
    volume = [c for c in auction if any(h in c.lower() for h in VOLUME_HINTS)]
    news = [c for c in auction if c not in volume]

    if news_override:
        news = [c for c in news_override if c in df.columns]
    else:
        news = dedupe(df, news)
        news = ([c for c in NEWS_PREFERRED if c in news]
                + [c for c in news if c not in NEWS_PREFERRED])[:3]
    if vol_override:
        volume = [c for c in vol_override if c in df.columns]
    else:
        volume = dedupe(df, volume)
        if not include_level_volume:
            drop = [c for c in volume if c.lower() in LEVEL_VOLUME]
            if drop:
                print(f"    - policy channel excludes level quantities {drop} "
                      f"by default (magnitude makes the design matrix "
                      f"ill-conditioned; --include-level-volume keeps them)")
            volume = [c for c in volume if c.lower() not in LEVEL_VOLUME]

    gpr = [c for c in df.columns
           if "gpr" in c.lower() and ok(c) and c.lower() not in NOT_AUCTION
           and not c.startswith(AUCTION_PREFIX)]
    gpr_core = ([c for c in GPR_CORE_PREFERRED if c in gpr] or gpr)[:3]
    return {"ctrl": ctrl, "ctrl_missing": missing, "auction_news": news,
            "auction_volume": volume, "gpr_core": gpr_core, "coverage": cov}


# =============================================================================
# 2. Surprises, placebo blocks, estimator
# =============================================================================
def expanding_ar1_surprise(s: pd.Series, min_train: int = 52,
                           clip: float = 4.0, min_reg: int = 60) -> pd.Series:
    """
    Robust expanding-window AR(1) surprise.

        e_t   = residual of  x_t = a_t + rho_t * x_{t-1},
                with (a_t, rho_t) estimated on {x_j : j < t} only
        z_t   = e_t / sd({e_j : j < t})        expanding-window standardisation
        out_t = clip(z_t, -clip, +clip)        winsorisation

    Standardisation removes the scale problem: raw auction outcomes differ in
    magnitude by three orders of magnitude, and an unstandardised block lets
    the largest-scale variable dominate. Winsorisation removes the extreme
    residuals produced when the AR coefficients are still poorly estimated
    early in the sample. Both use data dated before t only, so neither
    introduces look-ahead. min_reg raises the minimum regression sample from 30
    to 60 observations, which further stabilises the early window.
    """
    v = s.astype(float).values
    n = len(v)
    raw = np.full(n, np.nan)
    for t in range(min_train, n):
        y, x = v[1:t], v[:t - 1]
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < min_reg or not np.isfinite(v[t]) or not np.isfinite(v[t - 1]):
            continue
        b = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), x[m]]), y[m],
                            rcond=None)[0]
        raw[t] = v[t] - (b[0] + b[1] * v[t - 1])
    out = np.full(n, np.nan)
    for t in range(n):
        if not np.isfinite(raw[t]):
            continue
        past = raw[:t][np.isfinite(raw[:t])]
        if len(past) < 30:
            continue
        sd = past.std(ddof=1)
        if sd <= 0:
            continue
        out[t] = float(np.clip(raw[t] / sd, -clip, clip))
    return pd.Series(out, index=s.index)


def no_auction_flag(df: pd.DataFrame) -> pd.Series | None:
    """
    Indicator for weeks with no auction (August volumes are halved and
    year-end auctions suspended). In those weeks price-type auction variables
    are carried forward and quantity-type variables are zero, so differencing
    them produces structural zeros and jumps. The indicator lets the model
    absorb that rather than read it as supply news.
    """
    for c in ("auc_n", "auc_n_auctions", "auc_count"):
        if c in df.columns and df[c].notna().mean() > 0.8:
            return (df[c].fillna(0) <= 0).astype(float)
    return None


def ar1_block(n, k, rng, rho=0.9):
    """k independent AR(1) noise series, the placebo block."""
    X = np.zeros((n, k))
    for j in range(k):
        e = rng.normal(0, 1, n)
        for t in range(1, n):
            X[t, j] = rho * X[t - 1, j] + e[t]
    return X


class Est:
    """Ridge and ElasticNet with hyperparameters re-selected on training data
    every `refresh` periods. Deliberately small: the contribution of this
    module is the design, not the model zoo."""

    GR = [0.1, 1.0, 10.0, 100.0]
    GE = [(0.01, 0.5), (0.1, 0.5), (1.0, 0.5), (0.1, 0.9)]

    def __init__(self, refresh=13):
        self.refresh, self.cache, self.count = refresh, {}, {}

    def _cv(self, Xs, y, key):
        cut = int(len(y) * 0.75)
        br, be, sr, se_ = self.GR[1], self.GE[1], np.inf, np.inf
        for a in self.GR:
            s = np.mean((y[cut:] - Ridge(alpha=a).fit(Xs[:cut], y[:cut])
                         .predict(Xs[cut:])) ** 2)
            if s < sr:
                sr, br = s, a
        for a, l1 in self.GE:
            s = np.mean((y[cut:] - ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000)
                         .fit(Xs[:cut], y[:cut]).predict(Xs[cut:])) ** 2)
            if s < se_:
                se_, be = s, (a, l1)
        self.cache[key] = (br, be)

    def fit_predict(self, key, Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        Xs, Xt = sc.transform(Xtr), sc.transform(Xte)
        c = self.count.get(key, 0)
        if key not in self.cache or c % self.refresh == 0:
            self._cv(Xs, ytr, key)
        self.count[key] = c + 1
        br, (ae, le) = self.cache[key]
        return {"Ridge": float(Ridge(alpha=br).fit(Xs, ytr).predict(Xt)[0]),
                "ENet": float(ElasticNet(alpha=ae, l1_ratio=le, max_iter=5000)
                              .fit(Xs, ytr).predict(Xt)[0])}


CLIP_FORECASTS = True   # set from --no-clip-forecast in main()


def run_oos(d, FS, target, refresh, train_frac, min_train=200, clip=None):
    """
    Expanding-window one-step-ahead forecasts.

    Each information set gets its own Est instance and its own key space, so
    results for one set cannot be perturbed by which other sets are run in the
    same session.

    When clip is True, each forecast is truncated to the [0.5, 99.5] percentile
    range of the training-sample target. That range uses data dated before t
    only, so there is no look-ahead, and the same rule is applied to every
    information set including every placebo draw. The untruncated forecasts are
    retained in raw_-prefixed columns so both versions of R-squared can be
    reported.
    """
    clip = CLIP_FORECASTS if clip is None else clip
    ests = {fn: Est(refresh) for fn in FS}
    d = d.dropna(subset=[target]).reset_index(drop=True)
    n = len(d)
    n_tr = max(int(n * train_frac), min_train)
    rows = []
    for t in range(n_tr, n):
        tr, te = d.iloc[:t], d.iloc[[t]]
        ytr_all = tr[target].values
        lo, hi = np.nanpercentile(ytr_all, [0.5, 99.5])
        rec = {"i": t, "y": float(te[target].values[0]),
               "B_histmean": float(np.nanmean(ytr_all))}
        if "date" in d.columns:
            rec["date"] = te["date"].values[0]
        for fn, feats in FS.items():
            s2 = tr.dropna(subset=feats + [target])
            if len(s2) < 80 or not np.isfinite(te[feats].values).all():
                continue
            for m, v in ests[fn].fit_predict(fn, s2[feats].values,
                                             s2[target].values,
                                             te[feats].values).items():
                rec[f"raw_{m}_{fn}"] = v
                rec[f"{m}_{fn}"] = float(np.clip(v, lo, hi)) if clip else v
        rows.append(rec)
    oos = pd.DataFrame(rows)
    mc = [c for c in oos.columns if c not in ("i", "date", "y")]
    return oos.dropna(subset=mc).reset_index(drop=True)


def forecast_health(y, yhat, bench, top_frac: float = 0.01) -> dict:
    """
    Forecast-health diagnostic. Out-of-sample R-squared is a mean-square
    statistic, so a single exploding week can move it by hundreds of
    percentage points. Every reported R-squared therefore needs a companion
    measure of whether it is decided by a handful of weeks.
    """
    y, yhat, bench = map(np.asarray, (y, yhat, bench))
    se = (y - yhat) ** 2
    k = max(int(np.ceil(len(se) * top_frac)), 1)
    worst = np.sort(se)[-k:]
    share = float(worst.sum() / se.sum()) if se.sum() > 0 else np.nan
    return {"sse_share_worst1pct": round(share * 100, 1),
            "n_worst": k,
            "max_abs_fcst_over_sd_y": round(
                float(np.max(np.abs(yhat)) / y.std(ddof=1)), 2),
            "pathological": bool(share > 0.50)}


# =============================================================================
# 3. [B] Randomisation placebo test
# =============================================================================
def randomization_test(df, det, target, news, D, refresh, train_frac, seed0,
                       sup_extra=()):
    ctrl, gpr = det["ctrl"], det["gpr_core"]
    sup = list(news) + list(sup_extra)          # supply block + no-auction flag
    inter_g = []
    for s in news:                              # interactions use the surprises only
        for g in gpr:
            c = f"ix_{s}_{g}"
            df[c] = df[s] * df[g]
            inter_g.append(c)
    FS = {"CTRL": ctrl, "CTRL_SUP": ctrl + sup,
          "CTRL_SUP_GPRX": ctrl + sup + gpr + inter_g}
    print(f"    CTRL k={len(FS['CTRL'])}  CTRL_SUP k={len(FS['CTRL_SUP'])}  "
          f"GPRX k={len(FS['CTRL_SUP_GPRX'])}")
    oos = run_oos(df, FS, target, refresh, train_frac)
    y, b = oos["y"].values, oos["B_histmean"].values
    real, real_raw, health = {}, {}, {}
    for m in ("Ridge", "ENet"):
        real[m] = {fn: r2_os(y, oos[f"{m}_{fn}"].values, b) * 100
                   for fn in FS if f"{m}_{fn}" in oos}
        real_raw[m] = {fn: r2_os(y, oos[f"raw_{m}_{fn}"].values, b) * 100
                       for fn in FS if f"raw_{m}_{fn}" in oos}
        health[m] = {fn: forecast_health(y, oos[f"{m}_{fn}"].values, b)
                     for fn in FS if f"{m}_{fn}" in oos}

    draws = {m: [] for m in ("Ridge", "ENet")}
    for d_ in range(D):
        rng = np.random.default_rng(seed0 + d_)
        P = ar1_block(len(df), len(gpr), rng)
        pc = []
        for j in range(len(gpr)):
            df[f"_p{j}"] = P[:, j]
            pc.append(f"_p{j}")
        ip = []
        for s in news:
            for j in range(len(gpr)):
                c = f"_ixp_{s}_{j}"
                df[c] = df[s] * df[f"_p{j}"]
                ip.append(c)
        feats = ctrl + sup + pc + ip
        assert len(feats) == len(FS["CTRL_SUP_GPRX"]), \
            "placebo block is not dimension-matched"
        o = run_oos(df, {"PLCX": feats}, target, refresh, train_frac)
        yy, bb = o["y"].values, o["B_histmean"].values
        for m in draws:
            if f"{m}_PLCX" in o:
                draws[m].append(r2_os(yy, o[f"{m}_PLCX"].values, bb) * 100)
        if (d_ + 1) % max(D // 10, 1) == 0:
            print(f"      placebo draw {d_+1}/{D}")

    rows = []
    for m in ("Ridge", "ENet"):
        v = np.array(draws[m], float)
        if not len(v) or "CTRL_SUP_GPRX" not in real[m]:
            continue
        g = real[m]["CTRL_SUP_GPRX"]
        h = health[m]["CTRL_SUP_GPRX"]
        rows.append({
            "estimator": m, "n_oos": len(oos), "n_placebo_draws": len(v),
            "r2_CTRL": round(real[m]["CTRL"], 3),
            "r2_CTRL_SUP": round(real[m]["CTRL_SUP"], 3),
            "r2_GPRX": round(g, 3),
            "placebo_mean": round(float(v.mean()), 3),
            "placebo_sd": round(float(v.std(ddof=1)), 3),
            "placebo_p05": round(float(np.percentile(v, 5)), 3),
            "placebo_p95": round(float(np.percentile(v, 95)), 3),
            "gpr_percentile_in_null": round(float((v < g).mean() * 100), 1),
            "randomization_p": round(float((1 + (v >= g).sum()) / (1 + len(v))), 4),
            # numerical-robustness diagnostics
            "r2_GPRX_unclipped": round(real_raw[m].get("CTRL_SUP_GPRX", np.nan), 3),
            "sse_share_worst1pct": h["sse_share_worst1pct"],
            "max_abs_fcst_over_sd_y": h["max_abs_fcst_over_sd_y"],
            "status": "PATHOLOGICAL" if h["pathological"] else "ok",
        })
    res = pd.DataFrame(rows)
    hrows = [{"estimator": m, "info_set": fn, **hh,
              "r2_clipped": round(real[m][fn], 3),
              "r2_unclipped": round(real_raw[m].get(fn, np.nan), 3)}
             for m in health for fn, hh in health[m].items()]
    return res, oos, pd.DataFrame(draws), pd.DataFrame(hrows)


def in_sample_interaction(df, target, det, news):
    """Descriptive HAC interaction regressions, reported alongside the
    out-of-sample result precisely to show the gap between them."""
    rows = []
    for s in news:
        for g in det["gpr_core"]:
            cols = det["ctrl"] + [s, g]
            d = df.dropna(subset=cols + [target]).copy()
            if len(d) < 120:
                continue
            d["ix"] = d[s] * d[g]
            X = np.column_stack([np.ones(len(d)), d[cols + ["ix"]].values])
            b, se, t, p, _, cond = hac_ols(d[target].values, X)
            rows.append({"supply_news": s, "gpr": g, "n": len(d),
                         "beta": round(float(b[-1]), 4),
                         "se": round(float(se[-1]), 4),
                         "t": round(float(t[-1]), 3),
                         "p": round(float(p[-1]), 4), "cond": f"{cond:.1e}"})
    out = pd.DataFrame(rows)
    if len(out):
        out["holm"] = np.round(holm(out["p"].values), 4)
        out = out.sort_values("p").reset_index(drop=True)
    return out


# =============================================================================
# 4. [C] Policy-response channel
# =============================================================================
def policy_channel(df, det, max_lag=4, cond_max=1e10):
    """
    Wald test of H0: geopolitical risk does not Granger-predict auction supply.

    Both series are standardised before estimation. This is not cosmetic: on
    the untransformed level series the design is ill-conditioned and the Wald
    statistic is meaningless -- an earlier version returned chi-squared of
    about 1.4e4 with a coefficient sum of zero to four decimal places, and the
    identical statistic for two different regressors. Specifications whose
    design or restriction covariance has a condition number above cond_max are
    flagged UNSTABLE and their p-values are suppressed rather than reported.
    """
    rows = []
    for v in det["auction_volume"]:
        for g in det["gpr_core"]:
            d = df[[v, g]].dropna()
            if len(d) < 150 or d[v].std() == 0:
                continue
            yv = ((d[v] - d[v].mean()) / d[v].std()).values
            gv = ((d[g] - d[g].mean()) / d[g].std()).values
            y = yv[max_lag:]
            cols = [np.ones(len(y))]
            cols += [yv[max_lag - j:-j] for j in range(1, max_lag + 1)]
            k_own = len(cols)
            cols += [gv[max_lag - j:-j] for j in range(1, max_lag + 1)]
            X = np.column_stack(cols)
            b, se, t, p, V, cond = hac_ols(y, X)
            idx = list(range(k_own, X.shape[1]))
            R = np.zeros((len(idx), X.shape[1]))
            for i, j in enumerate(idx):
                R[i, j] = 1
            RVR = R @ V @ R.T
            rcond = float(np.linalg.cond(RVR))
            bad = (cond > cond_max) or (rcond > cond_max)
            W = float((R @ b) @ np.linalg.pinv(RVR) @ (R @ b))
            rows.append({
                "volume_var": v, "gpr_var": g, "n": len(y), "lags": max_lag,
                "cond_XtX": f"{cond:.1e}", "cond_RVR": f"{rcond:.1e}",
                "wald_chi2": np.nan if bad else round(W, 3),
                "p_value": np.nan if bad else round(
                    float(1 - st.chi2.cdf(W, len(idx))), 4),
                "sum_gpr_coefs_std": round(float(b[idx].sum()), 4),
                "status": "UNSTABLE" if bad else "ok"})
    out = pd.DataFrame(rows)
    if len(out):
        g = out["status"] == "ok"
        out.loc[g, "holm"] = np.round(holm(out.loc[g, "p_value"].values), 4)
        out = out.sort_values(["status", "p_value"]).reset_index(drop=True)
    return out


# Publication dates of the Total Number of Allowances in Circulation.
TNAC_DATES = ["2017-05-15", "2018-05-15", "2019-05-14", "2020-05-08",
              "2021-06-08", "2022-05-11", "2023-05-15", "2024-05-15",
              "2025-05-15", "2026-05-15"]


def tnac_robustness(oos, window=1):
    """Leave-out check, NOT an event study: with nine to ten events there is no
    power for one. Reported only to show that the interaction result is not
    driven by TNAC weeks."""
    if "date" not in oos.columns:
        return pd.DataFrame()
    d = oos.copy()
    d["date"] = pd.to_datetime(d["date"])
    mask = np.zeros(len(d), bool)
    for e in pd.to_datetime(TNAC_DATES):
        mask |= (d["date"] - e).abs() <= pd.Timedelta(weeks=window)
    rows = []
    for lab, sub in (("all weeks", d), (f"excl. TNAC +/-{window}w", d[~mask])):
        if len(sub) < 30:
            continue
        y, b = sub["y"].values, sub["B_histmean"].values
        r = {"sample": lab, "n": len(sub)}
        for m in ("Ridge", "ENet"):
            for fn in ("CTRL_SUP", "CTRL_SUP_GPRX"):
                if f"{m}_{fn}" in sub:
                    r[f"{m}_{fn}"] = round(
                        r2_os(y, sub[f"{m}_{fn}"].values, b) * 100, 3)
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# 5. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel_weekly_v2.csv")
    ap.add_argument("--out", default="out_supply")
    ap.add_argument("--target", default="y")
    ap.add_argument("--refresh-every", type=int, default=13,
                    help="hyperparameter re-selection interval, in periods")
    ap.add_argument("--train-frac", type=float, default=0.33)
    ap.add_argument("--placebo-draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=202608)
    ap.add_argument("--news", default="",
                    help="comma-separated auction-news columns, overriding "
                         "keyword detection")
    ap.add_argument("--volume", default="",
                    help="comma-separated auction-volume columns, overriding "
                         "keyword detection")
    ap.add_argument("--include-level-volume", action="store_true")
    ap.add_argument("--clip-surprise", type=float, default=4.0,
                    help="winsorisation threshold for the standardised supply "
                         "surprise, in standard deviations")
    ap.add_argument("--no-clip-forecast", action="store_true",
                    help="disable forecast truncation (for demonstrating the "
                         "numerical pathology only; not for reported results)")
    ap.add_argument("--inspect", action="store_true",
                    help="print detected variables and stop")
    ap.add_argument("--list-columns", action="store_true",
                    help="print every column name and its coverage, then stop")
    a = ap.parse_args()

    global CLIP_FORECASTS
    CLIP_FORECASTS = not a.no_clip_forecast
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78 + "\nSupply-side interaction module\n" + "=" * 78)
    print(f"  forecast truncation={'on' if CLIP_FORECASTS else 'off'}  "
          f"surprise winsorisation=+/-{a.clip_surprise}sd")
    head = pd.read_csv(a.panel, nrows=1)
    df = pd.read_csv(a.panel,
                     parse_dates=["date"] if "date" in head.columns else None)
    print(f"  panel: {df.shape}")

    if a.list_columns:
        cv = df.notna().mean()
        print("\n  all columns and coverage:")
        for c in df.columns:
            print(f"    {c:34s} {cv[c]:.3f}")
        return

    print()
    det = detect(df, a.target,
                 news_override=[c.strip() for c in a.news.split(",") if c.strip()] or None,
                 vol_override=[c.strip() for c in a.volume.split(",") if c.strip()] or None,
                 include_level_volume=a.include_level_volume)
    print(f"\n  controls       ({len(det['ctrl'])}): {det['ctrl']}")
    if det["ctrl_missing"]:
        print(f"    WARNING: exogenous controls still missing: "
              f"{det['ctrl_missing']}  (use --list-columns to inspect names)")
    print(f"  auction news   ({len(det['auction_news'])}): {det['auction_news']}")
    print(f"  auction volume ({len(det['auction_volume'])}): {det['auction_volume']}")
    print(f"  GPR core       ({len(det['gpr_core'])}): {det['gpr_core']}")
    pd.DataFrame({
        "variable": det["ctrl"] + det["auction_news"] + det["auction_volume"] + det["gpr_core"],
        "role": (["control"] * len(det["ctrl"])
                 + ["auction_news"] * len(det["auction_news"])
                 + ["auction_volume"] * len(det["auction_volume"])
                 + ["gpr"] * len(det["gpr_core"]))
    }).assign(coverage=lambda t: t.variable.map(df.notna().mean()).round(3)
              ).to_csv(out / "S0_variables.csv", index=False)

    if a.inspect:
        print("\n  --inspect: stopping here. Confirm that there are 7 controls "
              "and that every auction-news column is prefixed 'auc'.")
        return
    bad = [c for c in det["auction_news"] if not c.startswith(AUCTION_PREFIX)]
    if bad:
        raise SystemExit(f"ERROR: non-auction columns {bad} detected as supply "
                         f"news; specify the columns with --news.")
    if len(det["ctrl"]) < 7:
        print(f"  WARNING: only {len(det['ctrl'])} controls "
              f"(the paper's main specification uses 7). Continuing, but this "
              f"must be disclosed.")

    naf = no_auction_flag(df)
    sup_extra = []
    if naf is not None:
        df["no_auction"] = naf
        sup_extra = ["no_auction"]
        print(f"\n[A0] no-auction indicator: {naf.mean()*100:.1f}% of weeks "
              f"contain no auction")
    else:
        print("\n[A0] WARNING: no auction-count column found, so the "
              "no-auction indicator could not be built. Weeks without an "
              "auction are not explicitly handled.")

    news = []
    for c in det["auction_news"]:
        col = f"sup_{c}"
        srp = expanding_ar1_surprise(df[c], clip=a.clip_surprise)
        if naf is not None:
            srp = srp.where(naf < 0.5, 0.0)      # no auction means no supply news
        df[col] = srp
        news.append(col)
    dsc = df[news].describe().T[["count", "mean", "std", "min", "max"]].round(3)
    print(f"[A] Supply-news surprises (expanding-window AR(1), standardised "
          f"and winsorised at +/-{a.clip_surprise})")
    print(dsc.to_string())
    if float(dsc["max"].abs().max()) > a.clip_surprise + 1e-6:
        print("    WARNING: winsorisation did not take effect; "
              "check expanding_ar1_surprise")

    print(f"\n[B] Randomisation placebo test ({a.placebo_draws} draws)")
    t0 = pd.Timestamp.now()
    res, oos, draws, hlth = randomization_test(df, det, a.target, news,
                                               a.placebo_draws, a.refresh_every,
                                               a.train_frac, a.seed,
                                               sup_extra=sup_extra)
    print(f"    elapsed {(pd.Timestamp.now()-t0).total_seconds()/60:.1f} min")
    print(res.to_string(index=False))
    res.to_csv(out / "S2_randomization_test.csv", index=False)
    draws.to_csv(out / "S2b_placebo_draws.csv", index=False)
    oos.to_csv(out / "S1_oos_forecasts.csv", index=False)
    hlth.to_csv(out / "S2d_forecast_health.csv", index=False)

    print("\n[B''] Forecast-health diagnostics "
          "(is R2_OS decided by a handful of weeks?)")
    print(hlth.to_string(index=False))
    bad = res[res["status"] == "PATHOLOGICAL"]["estimator"].tolist()
    if bad:
        print(f"    ERROR: R2_OS for {bad} is decided by the worst 1% of weeks "
              f"(>50% of squared error).")
        print("       These arms must not carry a headline result and should "
              "be reported as diagnostics only.")
    else:
        print("    OK: no arm is dominated by isolated weeks; R2_OS is "
              "interpretable.")
    if (res["placebo_sd"] > 20).any():
        print("    WARNING: the placebo null distribution has a standard "
              "deviation above 20pp and is itself unusable for inference.")

    ins = in_sample_interaction(df, a.target, det, news)
    if len(ins):
        print("\n[B'] In-sample HAC interaction coefficients (descriptive)")
        print(ins.to_string(index=False))
        ins.to_csv(out / "S2c_insample_interaction.csv", index=False)

    pol = policy_channel(df, det)
    print("\n[C] Policy channel: does GPR predict auction supply?")
    if len(pol):
        print(pol.to_string(index=False))
        pol.to_csv(out / "S3_policy_channel.csv", index=False)
        g = pol[pol["status"] == "ok"]
        if len(g) and (g["holm"] >= 0.05).all():
            print("    -> No evidence. The supply half of the two-channel "
                  "account is not in the data and therefore cannot be used to "
                  "explain the null.")

    tn = tnac_robustness(oos)
    if len(tn):
        print("\n[D] TNAC-week leave-out check (not an event study)")
        print(tn.to_string(index=False))
        tn.to_csv(out / "S4_tnac_robustness.csv", index=False)

    if len(res):
        r = res.set_index("estimator")
        k_sup = len(det["ctrl"]) + len(news) + len(sup_extra)
        k_gprx = k_sup + len(det["gpr_core"]) * (1 + len(news))
        s = [{"quantity": "N out-of-sample weeks", "value": int(res["n_oos"].iloc[0])},
             {"quantity": "N placebo draws", "value": int(res["n_placebo_draws"].iloc[0])},
             {"quantity": "N controls", "value": len(det["ctrl"])},
             {"quantity": "k of CTRL_SUP", "value": k_sup},
             {"quantity": "k of CTRL_SUP_GPRX", "value": k_gprx}]
        for m in r.index:
            for k in ("r2_CTRL", "r2_CTRL_SUP", "r2_GPRX", "placebo_mean",
                      "placebo_sd", "placebo_p05", "placebo_p95",
                      "gpr_percentile_in_null", "randomization_p"):
                s.append({"quantity": f"{m}: {k}", "value": r.loc[m, k]})
        if len(pol):
            gg = pol[pol["status"] == "ok"]
            s += [{"quantity": "policy: n stable specs", "value": len(gg)},
                  {"quantity": "policy: n UNSTABLE dropped",
                   "value": int((pol["status"] != "ok").sum())},
                  {"quantity": "policy: min raw p",
                   "value": float(gg["p_value"].min()) if len(gg) else np.nan},
                  {"quantity": "policy: min Holm p",
                   "value": float(gg["holm"].min()) if len(gg) else np.nan}]
        if len(ins):
            s += [{"quantity": "in-sample: min raw p", "value": float(ins["p"].min())},
                  {"quantity": "in-sample: min Holm p", "value": float(ins["holm"].min())}]
        if len(tn) > 1:
            s.append({"quantity": "TNAC leave-out: n weeks",
                      "value": int(tn["n"].iloc[1])})
        for m in r.index:
            s += [{"quantity": f"{m}: sse_share_worst1pct",
                   "value": r.loc[m, "sse_share_worst1pct"]},
                  {"quantity": f"{m}: r2_GPRX_unclipped",
                   "value": r.loc[m, "r2_GPRX_unclipped"]},
                  {"quantity": f"{m}: status", "value": r.loc[m, "status"]}]
        s.append({"quantity": "no-auction share of weeks (%)",
                  "value": round(float(naf.mean()) * 100, 1) if naf is not None else np.nan})
        pd.DataFrame(s).to_csv(out / "S5_summary.csv", index=False)
        print("\n[summary]  (each row maps to one LaTeX macro; "
              "see src/fill_latex_macros.py)")
        print(pd.DataFrame(s).to_string(index=False))
    print(f"\n[done] {out.resolve()}")


if __name__ == "__main__":
    main()
