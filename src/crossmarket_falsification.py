"""
=============================================================================
crossmarket_falsification.py -- cross-market falsification test (weekly)
=============================================================================
Extends the single-market out-of-sample design to a panel of emissions
allowance markets, in order to test the energy-security channel that would
rationalise a positive geopolitical-risk (GPR) effect on carbon prices.

The channel carries a sharp cross-sectional implication: if GPR forecasts
allowance returns *through* energy security, its predictive content must
increase in a market's energy-import dependence and in its exposure to
European geopolitics. Markets that are institutionally comparable but
geopolitically insulated (RGGI, California-Quebec) are the control group.

Three information sets are estimated per market, all of them harmonised so
that they are estimable everywhere in the panel:

    CTRL           controls only (energy, volatility, policy uncertainty,
                   plus each market's own momentum and realised volatility)
    CTRL_PLACEBO   CTRL + a block of AR(1) noise series
    CTRL_GPR       CTRL + the GPR variable block

CTRL_PLACEBO and CTRL_GPR have identical cardinality by construction. This is
the point of the design: because the two blocks are equal-dimensional and
non-nested, the estimation-error cost of enlarging the regressor set is common
to both and differences out, so the null "the GPR block is exchangeable with
uninformative regressors of the same size" is median-zero and the
Diebold-Mariano statistic recovers its standard distribution. The comparison
against CTRL, by contrast, is nested: under the null of no incremental
information the larger model is *expected* to lose, so a median-zero test
against CTRL has no size control and is reported as descriptive only.

DESIGN NOTE 1 -- estimator state isolation
------------------------------------------
The point-forecast arm and the quantile arm are driven by two *separate*
FastEstimator instances, with separate key spaces.

This is not cosmetic. FastEstimator refreshes hyperparameters on a global call
counter, so sharing one instance across the two arms makes each arm's
hyperparameter refresh schedule depend on how many calls the other arm made.
In an earlier version that shared a single instance, adding a third quantile
information set moved one market's CTRL out-of-sample R-squared by 0.87
percentage points -- larger than any effect this module reports -- without a
single change to the CTRL specification. Per-market results would then depend
on which other models were run in the same session, which is fatal for a paper
whose contribution is reproducibility. Two instances make that impossible by
construction, at the cost of a second pass over the data.

DESIGN NOTE 2 -- both tail references are reported
--------------------------------------------------
The quantile arm produces two families of tail statistics, and the paper needs
both:

    pinball_gain_q{5,95}       referenced to CTRL   -- descriptive; used to
                               calibrate the dimensionality penalty
    pinball_gain_plc_q{5,95}   referenced to the placebo -- the only tail
                               comparison whose null is well specified

The difference between the two is an estimate of the out-of-sample cost of
appending an uninformative block of this size to a quantile model.

DESIGN NOTE 3 -- envelope versus estimator-matched gains
--------------------------------------------------------
The headline R-squared per information set is the upper envelope over four
estimators. A gain computed from envelopes is a difference of two maxima that
may be attained by different estimators, so it is not a paired comparison.
X5_estimator_matched.csv therefore reports the gain separately for each
estimator, holding the estimator fixed across information sets, and the run
log names the markets where the envelope is not estimator-matched.

INPUTS (both in --data)
-----------------------
    panel_multimarket.csv   weekly, one row per (market, date); must carry
                            date, code, ret, y and the market characteristics
                            region / allocation / energy_import_dep /
                            euro_geo_exposure
    panel_weekly_v2.csv     weekly single-market panel supplying the globally
                            common GPR and control variables

OUTPUTS (in --out)
------------------
    X0_market_oos.csv          per-market out-of-sample results
    X1_market_results.csv      X0 merged with market characteristics
    X2_rank_correlations.csv   cross-sectional rank correlations
    X3_skipped_markets.csv     screened-out markets and the reason
    X4_sign_tests.csv          cross-market sign-consistency tests
    X5_estimator_matched.csv   estimator-matched gains (robustness)

USAGE
-----
    python src/crossmarket_falsification.py --data data --out out_crossmarket --fast

    # observation-threshold sensitivity (the paper reports both settings)
    python src/crossmarket_falsification.py --data data \
        --out out_crossmarket_min100 --fast --min-obs 100

REQUIRES
--------
    msfc_lib.py    r2_os, dm_test  (see docs/REPRODUCIBILITY.md)
    msfc_fast.py   FastEstimator   (see docs/REPRODUCIBILITY.md)
=============================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

import msfc_lib as ml
from msfc_fast import FastEstimator

warnings.filterwarnings("ignore")

QUANTILES = [0.05, 0.95]
TRAIN_FRAC = 0.45
ESTIMATORS = ("Ridge", "ENet", "PCR", "XGB")

ORD_IMPORT = {"low": 0, "medium": 1, "high": 2, "very high": 3}
ORD_EXPO = {"low": 0, "medium": 1, "high": 2}
ORD_ALLOC = {"free": 0, "mostly free": 1, "auction+free": 2, "auction": 3,
             "auction-dominant": 3}

# Information sets estimated in the quantile arm.
QFS = ("CTRL", "CTRL_PLACEBO", "CTRL_GPR")


# =============================================================================
# 1. Data assembly
# =============================================================================
def detect_freq(dates: pd.Series) -> str:
    d = pd.Series(pd.to_datetime(dates).sort_values().unique())
    med = float(d.diff().dt.days.median())
    return "daily" if med <= 2 else "weekly" if med <= 9 else "monthly"


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mm_p = data_dir / "panel_multimarket.csv"
    wk_p = data_dir / "panel_weekly_v2.csv"
    for p in (mm_p, wk_p):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p.name} (searched in {data_dir.resolve()})")
    mm = pd.read_csv(mm_p, parse_dates=["date"])
    wk = pd.read_csv(wk_p, parse_dates=["date"])
    f_mm, f_wk = detect_freq(mm["date"]), detect_freq(wk["date"])
    print(f"  panel_multimarket: {mm.shape}  freq={f_mm}  "
          f"{mm.date.min():%Y-%m-%d}~{mm.date.max():%Y-%m-%d}  "
          f"{mm.code.nunique()} markets")
    print(f"  panel_weekly_v2  : {wk.shape}  freq={f_wk}")
    if f_mm != "weekly":
        raise SystemExit(
            f"panel_multimarket.csv is {f_mm}; this module requires weekly data. "
            f"At monthly frequency the core markets (EUA, UKA, CEA) fail the "
            f"observation screen and the test loses its meaning.")
    return mm, wk


def build_market_features(mm, wk, placebo_seed: int = 42):
    """Attach the globally common GPR and control variables to each market's
    return series, and generate the dimension-matched placebo block."""
    keep = [c for c in ["date", "code", "price", "ret", "y", "region",
                        "allocation", "energy_import_dep", "euro_geo_exposure"]
            if c in mm.columns]
    if "post_stale" in mm.columns:
        keep.append("post_stale")
    mm = mm[keep].copy()

    w = wk.set_index("date")
    feat = pd.DataFrame(index=w.index)
    # Controls: prefer the high-frequency series, fall back to the month-end one.
    fb = {"brent_ret": "brent_m_ret", "natgas_ret": "natgas_m_ret",
          "coal_ret": "coal_m_ret", "vstoxx_close": "vstoxx_m"}
    for c in ("brent_ret", "natgas_ret", "coal_ret", "vstoxx_close"):
        hi = w[c] if c in w.columns else pd.Series(np.nan, index=w.index)
        lo = w[fb[c]] if fb.get(c) in w.columns else None
        feat[c] = hi.where(hi.notna(), lo) if lo is not None else hi
    if "cpu__cpu_index_narrow" in w.columns:
        feat["cpu_log"] = np.log(w["cpu__cpu_index_narrow"].clip(lower=1e-6))

    # GPR block: a global common factor, identical across markets.
    ren = {"gprd_log": "gpr_log", "gprd_dlog": "gpr_dlog",
           "gprd_surprise": "gpr_surprise", "gprd_vol6": "gpr_vol6",
           "gprd_shock": "gpr_shock", "gprd_threat_act_gap": "gpr_gap"}
    for s_, d_ in ren.items():
        if s_ in w.columns:
            feat[d_] = w[s_]
    if "gpr_surprise" in feat:
        feat["gpr_surprise_lag1"] = feat["gpr_surprise"].shift(1)
    if "gpr_log" in feat:
        feat["gpr_log_lag1"] = feat["gpr_log"].shift(1)

    # Placebo block: AR(1) noise with the same cardinality as the GPR block.
    rng = np.random.default_rng(placebo_seed)
    n_gpr = sum(1 for c in feat.columns if c.startswith("gpr"))
    for i in range(n_gpr):
        e = rng.normal(0, 1, len(feat))
        x = np.zeros(len(feat))
        for t in range(1, len(x)):
            x[t] = 0.9 * x[t - 1] + e[t]
        feat[f"placebo{i+1}"] = x

    df = mm.merge(feat.reset_index(), on="date", how="left")
    ctrl = [c for c in ("brent_ret", "natgas_ret", "coal_ret",
                        "vstoxx_close", "cpu_log") if c in df.columns]
    gpr = [c for c in df.columns if c.startswith("gpr")]
    plc = [c for c in df.columns if c.startswith("placebo")]
    # Momentum and realised volatility are market-specific state variables:
    # they are each market's own history and cannot be shared across markets.
    df = df.sort_values(["code", "date"])
    df["own_mom4"] = df.groupby("code")["ret"].transform(lambda s: s.rolling(4).sum())
    df["own_rv12"] = df.groupby("code")["ret"].transform(lambda s: s.rolling(12).std())
    ctrl += ["own_mom4", "own_rv12"]

    FS = {"CTRL": ctrl, "CTRL_PLACEBO": ctrl + plc, "CTRL_GPR": ctrl + gpr}
    print(f"  feature sets: CTRL={len(ctrl)}  +noise={len(plc)}  +GPR={len(gpr)}"
          f"  (placebo and GPR blocks dimension-matched: "
          f"{len(FS['CTRL_PLACEBO'])} == {len(FS['CTRL_GPR'])})")
    if len(FS["CTRL_PLACEBO"]) != len(FS["CTRL_GPR"]):
        raise SystemExit("ERROR: placebo and GPR blocks differ in dimension; "
                         "the entire placebo design is invalidated.")
    return df, FS


# =============================================================================
# 2. Per-market out-of-sample evaluation
#    The point-forecast and quantile arms use independent estimator instances.
# =============================================================================
def run_market(sub, FS, est_point: FastEstimator, est_quant: FastEstimator,
               do_quantiles: bool, code: str, min_obs: int, min_oos: int):
    sub = sub.dropna(subset=["y"]).reset_index(drop=True)
    n = len(sub)
    if n < min_obs:
        return {"code": code, "n": n,
                "status": f"too few observations ({n}<{min_obs})"}
    n_tr = max(int(n * TRAIN_FRAC), 100)
    if n - n_tr < min_oos:
        return {"code": code, "n": n,
                "status": f"too few out-of-sample periods ({n-n_tr}<{min_oos})"}

    # ---- pass 1: point forecasts -------------------------------------------
    rows = []
    for t in range(n_tr, n):
        tr, te = sub.iloc[:t], sub.iloc[[t]]
        rec = {"t": t, "y": te["y"].values[0],
               "B_histmean": float(np.nanmean(tr["y"].values))}
        for fn, feats in FS.items():
            s2 = tr.dropna(subset=feats + ["y"])
            if len(s2) < 80 or np.isnan(te[feats].values).any():
                continue
            preds, _ = est_point.fit_predict(f"{code}|P|{fn}", s2[feats].values,
                                             s2["y"].values, te[feats].values)
            for mn, v in preds.items():
                rec[f"{mn}_{fn}"] = v
        rows.append(rec)
    pt = pd.DataFrame(rows)

    # ---- pass 2: quantiles (independent instance, independent key space) ----
    qt = pd.DataFrame({"t": pt["t"]})
    if do_quantiles:
        qrows = []
        for t in range(n_tr, n):
            tr, te = sub.iloc[:t], sub.iloc[[t]]
            rec = {"t": t}
            for fn in QFS:
                feats = FS[fn]
                s2 = tr.dropna(subset=feats + ["y"])
                if len(s2) < 80 or np.isnan(te[feats].values).any():
                    continue
                qp = est_quant.fit_predict_quantiles(
                    s2[feats].values, s2["y"].values, te[feats].values, QUANTILES)
                for q, v in qp.items():
                    rec[f"Q{int(q*100)}_{fn}"] = v
            qrows.append(rec)
        qt = pd.DataFrame(qrows)

    oos = pt.merge(qt, on="t", how="left")
    mcols = [c for c in oos.columns if c not in ("t", "y")]
    oos = oos.dropna(subset=mcols)
    if len(oos) < min_oos:
        return {"code": code, "n": n,
                "status": f"too few common-sample periods ({len(oos)})"}

    y, bench = oos["y"].values, oos["B_histmean"].values
    out = {"code": code, "n": n, "n_oos": len(oos), "status": "OK"}

    # Upper envelope over estimators (the convention used in the paper's tables).
    per_est = {}
    for fn in FS:
        vals = {m: ml.r2_os(y, oos[f"{m}_{fn}"].values, bench) * 100
                for m in ESTIMATORS if f"{m}_{fn}" in oos.columns}
        per_est[fn] = vals
        out[f"r2_{fn}"] = max(vals.values()) if vals else np.nan
    out["gain_gpr_vs_ctrl"] = out["r2_CTRL_GPR"] - out["r2_CTRL"]
    out["gain_gpr_vs_placebo"] = out["r2_CTRL_GPR"] - out["r2_CTRL_PLACEBO"]
    # Record which estimator attains each envelope: if the three differ, the
    # "gain" is a difference of two maxima and not a paired comparison.
    for fn, vals in per_est.items():
        out[f"argmax_{fn}"] = max(vals, key=vals.get) if vals else None
    out["envelope_matched"] = (out.get("argmax_CTRL") == out.get("argmax_CTRL_GPR")
                               == out.get("argmax_CTRL_PLACEBO"))

    # ---- tails: control-referenced (descriptive) and placebo-referenced -----
    if do_quantiles:
        for q in QUANTILES:
            qi = int(q * 100)

            def loss(col):
                d = y - oos[col].values
                return np.maximum(q * d, (q - 1) * d)

            have = {fn: f"Q{qi}_{fn}" for fn in QFS
                    if f"Q{qi}_{fn}" in oos.columns}
            if "CTRL" in have and "CTRL_GPR" in have:
                la, lb = loss(have["CTRL"]), loss(have["CTRL_GPR"])
                out[f"pinball_gain_q{qi}"] = (la.mean() - lb.mean()) / la.mean() * 100
                out[f"DM_p_q{qi}"] = ml.dm_test(la, lb, h=1, power=1)[1]
            if "CTRL_PLACEBO" in have and "CTRL_GPR" in have:
                la, lb = loss(have["CTRL_PLACEBO"]), loss(have["CTRL_GPR"])
                out[f"pinball_gain_plc_q{qi}"] = (la.mean() - lb.mean()) / la.mean() * 100
                # Equal-dimensional and non-nested, so DM is pivotal here.
                out[f"DM_p_plc_q{qi}"] = ml.dm_test(la, lb, h=1, power=1)[1]

    out["_per_est"] = per_est
    return out


# =============================================================================
# 2b. Market microstructure diagnostics
#
# The *level* of out-of-sample R-squared varies enormously across markets, but
# that variation is microstructural rather than economic: in thinly traded
# markets stale quotes and bid-ask bounce generate strong return
# autocorrelation, which mechanically makes momentum controls appear
# predictive. The paper therefore interprets only the GPR *gain* within each
# market, never the level, and reports these diagnostics in support of that
# decision.
# =============================================================================
def market_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, sub in df.groupby("code"):
        s_ = sub.dropna(subset=["ret"]).sort_values("date")
        r = s_["ret"].values
        span_w = max((s_["date"].max() - s_["date"].min()).days / 7.0, 1)
        gaps = s_["date"].diff().dt.days.dropna()
        rows.append({
            "code": code,
            "ac1": float(pd.Series(r).autocorr(1)) if len(r) > 5 else np.nan,
            "pct_zero_ret": float((np.abs(r) < 1e-10).mean() * 100),
            "coverage": float(len(s_) / span_w),          # observations / calendar weeks
            "median_gap_days": float(gaps.median()) if len(gaps) else np.nan,
            "pct_gap_gt10d": float((gaps > 10).mean() * 100) if len(gaps) else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. Cross-market inference
# =============================================================================
OUTCOME_LABELS = [
    ("gain_gpr_vs_ctrl",     "R2_OS gain vs controls",          "descriptive"),
    ("gain_gpr_vs_placebo",  "R2_OS gain vs noise block",       "well-specified"),
    ("pinball_gain_q95",     "Pinball gain vs CTRL, tau=0.95",  "descriptive"),
    ("pinball_gain_q5",      "Pinball gain vs CTRL, tau=0.05",  "descriptive"),
    ("pinball_gain_plc_q95", "Pinball gain vs NOISE, tau=0.95", "well-specified"),
    ("pinball_gain_plc_q5",  "Pinball gain vs NOISE, tau=0.05", "well-specified"),
]

CHARACTERISTICS = [
    ("expo_ord",  "European geopolitical exposure"),
    ("imp_ord",   "Energy import dependence"),
    ("alloc_ord", "Degree of auctioning"),
]


def sign_tests(res: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-market sign-consistency tests.

    A single market's test has almost no power (out-of-sample lengths run from
    roughly 130 to 320 periods). The informative object is the cross-market
    distribution of outcomes.

    The null_status column is the most important column in this table:

      'descriptive'    control-referenced, hence nested. Under the null the
                       expectation is negative, so a median-zero test has no
                       size control and must not be read as inference.
      'well-specified' placebo-referenced, equal-dimensional and non-nested.
                       Under the null the median is genuinely zero.

    Independence caveat: every market is scored against the same global GPR
    series, so the K outcomes are not independent and these p-values are
    anti-conservative. crossmarket_inference.py reports dependence-adjusted
    versions.
    """
    rows = []
    for col, lab, status in OUTCOME_LABELS:
        if col not in res.columns:
            continue
        v = res[col].dropna()
        if len(v) < 4:
            continue
        neg = int((v < 0).sum())
        rows.append({
            "outcome": lab, "null_status": status, "n_markets": len(v),
            "n_negative": neg,
            "mean": round(float(v.mean()), 3),
            "median": round(float(v.median()), 3),
            "sd": round(float(v.std(ddof=1)), 3),
            "sign_test_p": round(float(st.binomtest(neg, len(v), 0.5).pvalue), 4),
            "wilcoxon_p": (round(float(st.wilcoxon(v).pvalue), 4)
                           if len(v) >= 6 else np.nan),
        })
    return pd.DataFrame(rows)


def cross_section(res, meta, out_dir: Path) -> pd.DataFrame:
    d = res.merge(meta, on="code", how="left")
    d["imp_ord"] = d["energy_import_dep"].map(ORD_IMPORT)
    d["expo_ord"] = d["euro_geo_exposure"].map(ORD_EXPO)
    d["alloc_ord"] = d["allocation"].map(ORD_ALLOC)

    print("\n[B] Cross-sectional heterogeneity")
    print(f"    WARNING: only {len(d)} markets. See the power analysis in "
          f"crossmarket_inference.py:")
    print("       80% power requires a gradient of roughly 2.5pp per exposure "
          "level, which exceeds the entire observed range of the gain.")
    print("       The point estimates below are not interpretable in either "
          "direction.")
    rows = []
    for yv, _, _ in OUTCOME_LABELS:
        if yv not in d.columns:
            continue
        for xv, lab in CHARACTERISTICS:
            s = d[[yv, xv]].dropna()
            if len(s) < 5 or s[xv].nunique() < 2:
                continue
            rho, p = st.spearmanr(s[xv], s[yv])
            rows.append({"outcome": yv, "characteristic": lab, "n": len(s),
                         "spearman_rho": round(float(rho), 3),
                         "p_value": round(float(p), 3)})
    xs = pd.DataFrame(rows)
    if len(xs):
        xs.to_csv(out_dir / "X2_rank_correlations.csv", index=False)
        print(xs.to_string(index=False))

    print("\n    Group means (GPR gain in R2_OS relative to controls, pp):")
    for xv, lab in (("euro_geo_exposure", "European geopolitical exposure"),
                    ("energy_import_dep", "Energy import dependence")):
        if xv not in d.columns:
            continue
        g = d.groupby(xv)["gain_gpr_vs_ctrl"].agg(["count", "mean"]).round(3)
        print(f"      by {lab}:")
        for k, r in g.iterrows():
            print(f"        {str(k):<12} n={int(r['count'])}  mean {r['mean']:+.3f}")

    d.to_csv(out_dir / "X1_market_results.csv", index=False)
    return d


# =============================================================================
# 4. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out_crossmarket")
    ap.add_argument("--fast", action="store_true",
                    help="disable the random-forest estimator")
    ap.add_argument("--no-quantiles", action="store_true")
    ap.add_argument("--refresh-every", type=int, default=13,
                    help="hyperparameter re-selection interval, in periods")
    ap.add_argument("--min-obs", type=int, default=150,
                    help="minimum usable weekly observations per market")
    ap.add_argument("--min-oos", type=int, default=60,
                    help="minimum out-of-sample periods per market")
    ap.add_argument("--placebo-seed", type=int, default=42)
    ap.add_argument("--drop", default="EUA_pre19",
                    help="comma-separated markets to exclude. EUA_pre19 is "
                         "dropped by default: it reports primary-market "
                         "clearing prices for the same underlying instrument "
                         "as EUA and would double-count the European market.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Cross-market falsification test (weekly)")
    print(f"  MIN_OBS={args.min_obs}  MIN_OOS={args.min_oos}  "
          f"placebo_seed={args.placebo_seed}  refresh_every={args.refresh_every}")
    print("=" * 78)

    mm, wk = load_data(Path(args.data))
    drop = [x.strip() for x in args.drop.split(",") if x.strip()]
    if drop:
        n0 = mm["code"].nunique()
        mm = mm[~mm["code"].isin(drop)]
        print(f"  excluded {drop} ({n0} -> {mm['code'].nunique()} markets)")
    df, FS = build_market_features(mm, wk, args.placebo_seed)
    diag = market_diagnostics(df)
    print("\n  [market microstructure diagnostics]")
    print(diag.round(3).to_string(index=False))

    # Two independent estimator instances: see DESIGN NOTE 1 in the module
    # docstring. The point-forecast arm cannot be perturbed by the quantile arm.
    est_point = FastEstimator(refresh_every=args.refresh_every, use_rf=not args.fast)
    est_quant = FastEstimator(refresh_every=args.refresh_every, use_rf=not args.fast)

    meta = (df.groupby("code")[["region", "allocation", "energy_import_dep",
                                "euro_geo_exposure"]].first().reset_index())

    print(f"\n[A] Per-market out-of-sample ({df.code.nunique()} markets, "
          f"point forecasts and quantiles in separate passes)")
    results, t0 = [], pd.Timestamp.now()
    for code, sub in df.groupby("code"):
        r = run_market(sub, FS, est_point, est_quant, not args.no_quantiles,
                       code, args.min_obs, args.min_oos)
        results.append(r)
        el = (pd.Timestamp.now() - t0).total_seconds() / 60
        if r["status"] == "OK":
            q = r.get("pinball_gain_plc_q95", np.nan)
            print(f"  [ok]   {code:<10} n={r['n']:>4} oos={r['n_oos']:>4}  "
                  f"CTRL {r['r2_CTRL']:+7.2f}%  +noise {r['r2_CTRL_PLACEBO']:+7.2f}%  "
                  f"+GPR {r['r2_CTRL_GPR']:+7.2f}%  q95(vs noise) {q:+6.2f}%"
                  f"   [{el:.1f}min]")
        else:
            print(f"  [skip] {code:<10} {r['status']}")

    ok = [r for r in results if r.get("status") == "OK"]
    per_est = {r["code"]: r.pop("_per_est") for r in ok}
    res = pd.DataFrame(ok)
    skipped = [r for r in results if r.get("status") != "OK"]
    if len(res) < 4:
        raise SystemExit(f"only {len(res)} markets available; "
                         f"cannot run the cross-section.")
    print(f"\n  retained {len(res)} markets, skipped {len(skipped)}")

    # Self-check: both tail families must be present.
    missing = [c for c in ("pinball_gain_q95", "pinball_gain_plc_q95")
               if c not in res.columns]
    if missing and not args.no_quantiles:
        print(f"\n  ERROR: missing quantile columns: {missing}")
        print("     The tail columns of the paper's market table, the "
              "dependence-adjusted panels, the tail rows of the gradient "
              "table, the robustness table and both appendix figures all "
              "depend on them. Resolve this before continuing.")

    if not res["envelope_matched"].all():
        bad = res.loc[~res["envelope_matched"], "code"].tolist()
        print(f"\n  NOTE: envelope attained by different estimators across "
              f"information sets in: {bad}")
        print("     For these markets the 'gain' is not a paired comparison. "
              "X5 reports the estimator-matched version.")

    n_fail = int((res["gain_gpr_vs_placebo"] <= 0).sum())
    print(f"\n  * Placebo (point forecasts): the GPR block fails to beat the "
          f"dimension-matched noise block in {n_fail}/{len(res)} markets")
    if "pinball_gain_plc_q95" in res.columns:
        nq = int((res["pinball_gain_plc_q95"] <= 0).sum())
        print(f"  * Placebo (tau=0.95): the GPR block fails to beat the "
              f"dimension-matched noise block in {nq}/{len(res)} markets")
        print("     ^ this is the module's only well-specified tail result; "
              "the paper should be based on it.")

    sg = sign_tests(res)
    if len(sg):
        print("\n  * Cross-market sign-consistency tests")
        print(sg.to_string(index=False))
        sg.to_csv(out_dir / "X4_sign_tests.csv", index=False)
        print("\n     WARNING: rows with null_status='descriptive' are not "
              "inference. The control-referenced comparison is nested, its "
              "expectation under the null is negative, and a median-zero test "
              "has no size control.")
        print("     WARNING: all markets share one global GPR series and are "
              "therefore not independent. For the dependence adjustment see "
              "crossmarket_inference.py.")

    res = res.merge(diag, on="code", how="left")
    cross_section(res, meta, out_dir)

    # Estimator-matched gains (robustness; see DESIGN NOTE 3).
    rows = []
    for code, vals in per_est.items():
        for m in ESTIMATORS:
            if all(m in vals[fn] for fn in FS):
                rows.append({"code": code, "estimator": m,
                             "r2_CTRL": vals["CTRL"][m],
                             "r2_CTRL_PLACEBO": vals["CTRL_PLACEBO"][m],
                             "r2_CTRL_GPR": vals["CTRL_GPR"][m],
                             "gain_vs_ctrl": vals["CTRL_GPR"][m] - vals["CTRL"][m],
                             "gain_vs_placebo": (vals["CTRL_GPR"][m]
                                                 - vals["CTRL_PLACEBO"][m])})
    pm = pd.DataFrame(rows)
    if len(pm):
        pm.to_csv(out_dir / "X5_estimator_matched.csv", index=False)
        agg = pm.groupby("estimator")[["gain_vs_ctrl", "gain_vs_placebo"]].agg(
            ["mean", "median", lambda s: int((s < 0).sum())])
        print("\n  [X5] Estimator-matched gains (robustness)")
        print(agg.round(3).to_string())

    res.drop(columns=[c for c in res.columns if c.startswith("argmax_")]
             ).to_csv(out_dir / "X0_market_oos.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "X3_skipped_markets.csv", index=False)
    print(f"\n[done] {out_dir.resolve()}")
    for f in ("X0_market_oos.csv", "X1_market_results.csv",
              "X2_rank_correlations.csv", "X3_skipped_markets.csv",
              "X4_sign_tests.csv", "X5_estimator_matched.csv"):
        print(f"  {f}")
    print("\nnext: python src/crossmarket_inference.py --out " + str(out_dir))


if __name__ == "__main__":
    main()
