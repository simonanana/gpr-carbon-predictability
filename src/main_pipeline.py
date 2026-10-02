"""
=============================================================================
main_pipeline.py -- single-market out-of-sample pipeline (weekly or monthly)
=============================================================================
Produces the paper's headline results: the frequency contrast, the
specification table, the dimension-matched placebo comparison, the model
confidence set, tail inference and economic value.

The same code runs at both frequencies. `--freq W` and `--freq M` change only
the window lengths that frequency mechanically implies (momentum, realised
volatility, variance window, initial training window); the model space, the
estimators and every test are identical. That is what makes the two
frequencies directly comparable, and the contrast between them is the
paper's central exercise.

`--freq` DOES NOT RESAMPLE
--------------------------
This is the single most important thing to know before running the module.
`--freq M` selects monthly window lengths; it does **not** convert a weekly
panel to monthly. Each frequency needs its own panel:

    --freq W   ->  data/panel_weekly_v2.csv    (about 604 rows, 7-day spacing)
    --freq M   ->  data/panel_monthly_v2.csv   (about 138 rows, month-end)

Passing the weekly panel with `--freq M` produces a hybrid that is neither
frequency: 604 weekly rows modelled with monthly-length momentum and variance
windows and a 60-period training window, yielding 544 "monthly" out-of-sample
periods. It runs to completion and labels its own output "monthly", so nothing
downstream reveals the error. A frequency guard in `main()` now refuses this
combination by comparing the panel's median row spacing against `--freq`.

DESIGN POINTS
-------------
  * Each panel is kept at its native frequency and is never resampled inside
    this module. The monthly arm is deliberately small (about 138 observations,
    78 out of sample): that is the sample size conventional in this literature,
    and demonstrating how fragile a result is at that size is the point of the
    exercise rather than an accident to be corrected. Resampling silently
    inside the weekly path, by contrast, is the defect that the superseded
    feature-construction routine contained (see oos_evaluation.py).

  * The single-market panel and the long multi-market panel are NOT merged.
    The multi-market panel has one row per market per date, so combining them
    on date inflates rows and silently mismatches. They answer different
    questions: this module does the single-market time series, and
    `crossmarket_falsification.py` does the cross-sectional test.

  * Feature sets are built from the columns the panel actually contains, and
    any set that cannot be estimated is dropped with an explicit message
    rather than failing silently.

  * Control variables fall back to a month-end series where the
    high-frequency vendor history is short, and the backfilled share of each
    variable is printed and reported in the paper's data appendix.

  * Auction variables need care. Roughly 5% of weeks contain no auction at
    all (August volumes are halved, year-end auctions suspended), so those
    rows are entirely missing. Dropping them would discard a large share of
    out-of-sample predictions and leave the models incomparable. Quantity
    variables are therefore filled with zero (no auction means zero volume,
    which is information), price variables are carried forward, and a
    no-auction indicator lets the model identify the state.

A NOTE ON THE PLACEBO BLOCK
---------------------------
The placebo is only interpretable as a dimension-matched control when it has
the same cardinality as the block it stands in for. With the default settings
the noise block has 6 series while CTRL_GPRfull adds 9 geopolitical
variables, so the two are NOT matched and the module prints a warning saying
so. `--placebo-n 9` matches them exactly and prints
`[ok] placebo dimension-matched`. The default is left at 6 so that previously
generated results reproduce; see docs/REPRODUCIBILITY.md for the direction of
the resulting bias.

USAGE
-----
    python src/main_pipeline.py --panel data/panel_weekly_v2.csv \
        --price-dir data --out out_weekly --placebo-n 9 --fast

    python src/main_pipeline.py --panel data/panel_monthly_v2.csv \
        --price-dir data --out out_monthly --freq M --placebo-n 9 --fast

Note the different panel on the second line. See "--freq DOES NOT RESAMPLE".
=============================================================================
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_eval as ml
from oos_evaluation import (evaluate, make_figures, make_tail_figure,
                            block_bootstrap_importance, CRISIS_WINDOWS,
                            _dmsfe_combine, QUANTILE_GRID)
from fast_estimator import FastEstimator

warnings.filterwarnings("ignore")


# =============================================================================
# 0. Parsing helpers (self-contained, so the module runs from any directory)
# =============================================================================
DATE_FORMATS = ["%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y",
                "%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d.%m.%y", "%Y%m%d"]


def parse_dates_robust(s: pd.Series, name: str = "", verbose: bool = True
                       ) -> tuple[pd.Series, str]:
    """
    Try each format in turn, then fill remaining gaps from compatible formats.

    Vendor exports sometimes mix two formats within a single column (for
    example 08/09/26 alongside 07/26/2026 in the same file). Parsing with one
    format silently drops a third of the observations.

    Ties are broken on the share of parsed days greater than 12: a real date
    series has roughly 60% of days above 12, so a format whose days are all
    at most 12 has almost certainly swapped month and day.
    """
    raw = s.astype(str).str.strip()
    raw = raw.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})
    scored = []
    for fmt in DATE_FORMATS:
        try:
            d = pd.to_datetime(raw, format=fmt, errors="coerce")
        except Exception:
            continue
        ok = float(d.notna().mean())
        if ok == 0:
            continue
        gt12 = float((d.dt.day > 12).mean()) if d.notna().any() else 0.0
        scored.append((ok, gt12, fmt, d))
    if not scored:
        return pd.to_datetime(raw, errors="coerce"), "fallback"
    scored.sort(key=lambda t: (round(t[0], 4), round(t[1], 4)), reverse=True)
    ok0, gt0, fmt0, out = scored[0]

    def _order(f):
        i_m, i_d = f.find("%m"), f.find("%d")
        return "M" if (i_m >= 0 and (i_d < 0 or i_m < i_d)) else "D"

    used = [fmt0]
    for ok, gt, fmt, d in scored[1:]:
        if out.isna().sum() == 0:
            break
        if _order(fmt) != _order(fmt0):      # never mix day-first with month-first
            continue
        fill = out.isna() & d.notna()
        if fill.any():
            out = out.where(~fill, d)
            used.append(fmt)
    if verbose:
        extra = f" + filled from {used[1:]}" if len(used) > 1 else ""
        print(f"      date format {fmt0}{extra} -> parsed {out.notna().mean():.1%}")
    return out, "+".join(used)


def _num(x):
    """'1,234.5' / ' 12.86K ' / '4.97%' / ':' -> float"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    t = str(x).strip().replace(",", "").replace(" ", "").replace("%", "")
    if t in ("", "-", ":", "n/a", "N/A", "na"):
        return np.nan
    import re as _re
    m = _re.match(r"^([-+]?[\d.]+)\s*([KMB])?$", t, _re.I)
    if m:
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get((m.group(2) or "").upper(), 1)
        return float(m.group(1)) * mult
    t = _re.sub(r"[^\d.\-+]", "", t)
    try:
        return float(t)
    except ValueError:
        return np.nan


def resolve_panel(user_path: str | None) -> Path:
    """Locate the panel. If the given path does not exist, search the module
    directory, ./data, ./out and the parent directory, and on failure list the
    CSV files that were actually found."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    here = Path(__file__).resolve().parent
    for root in (here, here / "data", here / "out", here.parent, Path.cwd()):
        if not root.exists():
            continue
        hits = sorted(root.rglob("panel_weekly*.csv"))
        if hits:
            print(f"  [note] panel not found at '{user_path}', using {hits[0]}")
            return hits[0]
    found = sorted({str(q.relative_to(here)) for q in here.rglob("*.csv")})[:25]
    raise FileNotFoundError(
        f"\nCannot find the panel.\n"
        f"  --panel was {user_path}\n"
        f"  module directory = {here}\n"
        f"  searched: module dir / ./data / ./out / parent\n"
        f"  csv files there: {found}\n"
        f"  -> point --panel at the right file, e.g. --panel data/panel_weekly_v2.csv")


# =============================================================================
# 1. Panel loading
# =============================================================================
def load_price_extras(price_dir: Path | None, index: pd.DatetimeIndex,
                      freq: str = "W-FRI") -> pd.DataFrame:
    """
    Optionally append energy and financial controls from vendor CSV exports.
    A file is recognised by a keyword in its name: brent / gas (or ttf) /
    coal / vstoxx (or vix) / stoxx.

    Without energy price controls the control baseline is too weak, and any
    conclusion that the geopolitical block "beats controls" is uninformative:
    it would only show that a larger model beats a smaller one.
    """
    out = pd.DataFrame(index=index)
    if price_dir is None or not Path(price_dir).exists():
        return out
    pats = {"brent": ["brent", "crude"], "natgas": ["ttf", "natural gas", "natgas"],
            "coal": ["coal", "api2"], "vstoxx": ["vstoxx", "vix"],
            "stoxx": ["stoxx 50", "euro stoxx"]}
    for name, keys in pats.items():
        hit = None
        for p in Path(price_dir).rglob("*.csv"):
            n = p.name.lower()
            if any(k in n for k in keys):
                hit = p
                break
        if hit is None:
            continue
        df = pd.read_csv(hit)
        df.columns = [c.strip().lower().replace(".", "") for c in df.columns]
        if "date" not in df.columns or "price" not in df.columns:
            print(f"  [controls] {hit.name} has no Date/Price column, skipped")
            continue
        print(f"  [controls] {name:<8} <- {hit.name}")
        d, _ = parse_dates_robust(df["date"], name=hit.name, verbose=True)
        s = pd.Series(df["price"].map(_num).values, index=d).dropna().sort_index()
        s = s[~s.index.duplicated(keep="last")].resample(freq).last()
        out[f"{name}_close"] = s.reindex(index)
        out[f"{name}_ret"] = np.log(out[f"{name}_close"].clip(lower=1e-9)).diff()
        cov = out[f"{name}_close"].notna().mean()
        print(f"      {s.notna().sum()} periods, covers {cov:.1%} of the window"
              + ("   [low coverage, check the date range]" if cov < 0.8 else ""))
    return out


# Frequency-dependent parameters. Only window lengths differ.
FREQ_PARAMS = {
    "W": dict(freq="W-FRI", mom=4, rv=12, vol_window=52, offset_days=7,
              min_train_sub=100, initial_train=200, label="weekly",
              periods_per_year=52),
    "M": dict(freq="ME", mom=3, rv=6, vol_window=24, offset_days=30,
              min_train_sub=30, initial_train=60, label="monthly",
              periods_per_year=12),
}


def build_features_weekly(panel_path: Path, price_dir: Path | None = None,
                          freq: str = "W-FRI", placebo_seed: int = 42,
                          fp: dict | None = None, n_placebo: int = 6) -> pd.DataFrame:
    fp = fp or FREQ_PARAMS["W"]
    df = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date")

    extras = load_price_extras(price_dir, df.index, freq)
    if len(extras.columns):
        df = df.join(extras)

    out = pd.DataFrame(index=df.index)
    out["y"] = df["y"]
    out["eua_return_t"] = df["eua_ret"]
    out["eua_mom4"] = df["eua_ret"].rolling(fp["mom"]).sum()
    out["eua_rv12"] = df["eua_ret"].rolling(fp["rv"]).std()

    # ---- controls: high frequency first, month-end fallback ----------------
    # Vendor histories are short for some series, and using them raw would
    # invalidate the most recent stretch of the out-of-sample window. Where a
    # high-frequency value is missing, the month-end series fills it, and the
    # backfilled share of each variable is recorded.
    fallback = {"brent_ret": "brent_m_ret", "natgas_ret": "natgas_m_ret",
                "coal_ret": "coal_m_ret", "vstoxx_close": "vstoxx_m"}
    patched = []
    for c in ("brent_ret", "natgas_ret", "coal_ret", "vstoxx_close", "stoxx_ret"):
        hi = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
        lo_name = fallback.get(c)
        lo = df[lo_name] if (lo_name and lo_name in df.columns) else None
        if lo is not None:
            n_fill = int((hi.isna() & lo.notna()).sum())
            merged = hi.where(hi.notna(), lo)
            if n_fill:
                patched.append((c, n_fill, float(n_fill / len(merged))))
        else:
            merged = hi
        if merged.notna().any():
            out[c] = merged
    if patched:
        print("\n  [control backfill] high-frequency gaps filled from month-end series:")
        for c, n, r in patched:
            print(f"       {c:<14} {n:>3} periods ({r:.1%})")
        print("     Report this in the data appendix, and report a "
              "high-frequency-only variant as a robustness check.")
    for c in ("epu_m", "clean_m_ret"):
        if c in df.columns and df[c].notna().mean() > 0.9:
            out[c] = df[c]
    if "cpu__cpu_index_narrow" in df:
        out["cpu_log"] = np.log(df["cpu__cpu_index_narrow"].clip(lower=1e-6))
        out["cpu_dlog"] = out["cpu_log"].diff()

    # ---- supply side (auctions) --------------------------------------------
    # Quantity variables are filled with zero; price variables are carried
    # forward; a no-auction indicator marks the state. See the module
    # docstring for why this matters.
    qty = {"auc_n": "auc_n", "auc_n_cancelled": "auc_cancel"}
    pxq = {"auc_cover_vw": "auc_cover", "auc_discount": "auc_discount",
           "auc_bid_range": "auc_bid_range"}
    if "auc_n" in df.columns:
        out["auc_no_auction"] = (df["auc_n"].fillna(0) == 0).astype(float)
    for src, dst in qty.items():
        if src in df.columns:
            out[dst] = df[src].fillna(0.0)
    for src, dst in pxq.items():
        if src in df.columns:
            out[dst] = df[src].ffill()
    if "auc_volume_dlog" in df.columns:
        out["auc_vol_dlog"] = df["auc_volume_dlog"].fillna(0.0)

    # ---- GPR variable family ------------------------------------------------
    ren = {"gprd_log": "gpr_log", "gprd_dlog": "gpr_dlog", "gprd_vol6": "gpr_vol6",
           "gprd_surprise": "gpr_surprise", "gprd_surp_pos": "gpr_surprise_pos",
           "gprd_surp_neg": "gpr_surprise_neg", "gprd_shock": "gpr_shock_dummy",
           "gprd_threat_act_gap": "gpr_threat_act_gap"}
    for s_, d_ in ren.items():
        if s_ in df.columns:
            out[d_] = df[s_]
    for L in (1, 2, 3):
        if "gpr_log" in out:
            out[f"gpr_log_lag{L}"] = out["gpr_log"].shift(L)
    if "gpr_surprise" in out:
        out["gpr_surprise_lag1"] = out["gpr_surprise"].shift(1)

    # ---- exposure measures --------------------------------------------------
    for c in ("exp_share_risky", "exp_hhi", "exp_n_eff", "gpr_x_exposure",
              "gpr_x_hhi", "gpr_exp_orth", "gprexp_surprise"):
        if c in df.columns:
            out[c] = df[c]

    # ---- placebo block ------------------------------------------------------
    # AR(1) noise: similar persistence to the real variables, no relation to y.
    # Separates the information value of the GPR block from the pure effect of
    # enlarging the regressor set.
    rng = np.random.default_rng(placebo_seed)
    for i in range(n_placebo):
        e = rng.normal(0, 1, len(out))
        x = np.zeros(len(out))
        for t in range(1, len(x)):
            x[t] = 0.9 * x[t - 1] + e[t]
        out[f"placebo{i+1}"] = x

    out = out.reset_index().rename(columns={"index": "date"})
    out["target_month"] = out["date"] + pd.Timedelta(days=fp["offset_days"])
    out = out.dropna(subset=["y", "eua_return_t"]).reset_index(drop=True)
    return out


# =============================================================================
# 2. Feature sets, trimmed to the columns the panel actually has
# =============================================================================
def make_feature_sets(cols: set[str]) -> dict[str, list[str]]:
    def keep(lst):
        return [c for c in lst if c in cols]

    ctrl = keep(["brent_ret", "natgas_ret", "coal_ret", "vstoxx_close", "stoxx_ret",
                 "cpu_log", "auc_cover", "auc_discount", "auc_vol_dlog",
                 "auc_n", "eua_mom4", "eua_rv12"])
    gpr_core = keep(["gpr_log", "gpr_dlog", "gpr_surprise", "gpr_surprise_lag1",
                     "gpr_vol6", "gpr_shock_dummy"])
    gpr_full = keep(["gpr_log", "gpr_dlog", "gpr_surprise", "gpr_surprise_lag1",
                     "gpr_vol6", "gpr_shock_dummy", "gpr_threat_act_gap",
                     "gpr_log_lag1", "gpr_log_lag2"])
    expo = keep(["exp_share_risky", "exp_hhi", "gpr_x_exposure", "gpr_exp_orth"])
    placebo = keep([f"placebo{i+1}" for i in range(24)])

    fs = {
        "CTRL": ctrl,
        # Dimension-matched placebo: intended to carry the same number of
        # added regressors as CTRL_GPRfull, but with noise instead of signal.
        # If the GPR set beats CTRL yet not this one, the improvement came
        # from dimensionality rather than information. The slice is capped by
        # how many placebo series were generated -- see the warning in main().
        "CTRL_PLACEBO": ctrl + placebo[:len(gpr_full)],
        "CTRL_GPRlevel": ctrl + keep(["gpr_log"]),
        "CTRL_GPRdlag": ctrl + keep(["gpr_log", "gpr_log_lag1", "gpr_log_lag2"]),
        "CTRL_GPRsurprise": ctrl + keep(["gpr_surprise", "gpr_surprise_lag1"]),
        "CTRL_GPRasym": ctrl + keep(["gpr_surprise_pos", "gpr_surprise_neg",
                                     "gpr_vol6", "gpr_shock_dummy"]),
        "CTRL_GPRfull": ctrl + gpr_full,
        "CTRL_GPRexpo": ctrl + gpr_core + expo,
    }
    return {k: v for k, v in fs.items() if len(v) >= 2}


# =============================================================================
# 2b. Feature-coverage audit -- catches silent collapse before the run starts
# =============================================================================
def audit_features(df: pd.DataFrame, feature_sets: dict, initial_train: int,
                   min_cov: float = 0.60, min_train_sub: int = 100,
                   out_dir: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """
    Why this step exists.

    The out-of-sample loop drops rows with `tr.dropna(subset=feats)`. A single
    poorly covered variable therefore collapses the usable sample for its
    entire feature set to near zero, and every model in that set is skipped
    silently. The symptom is a run that finishes suspiciously fast with only
    benchmark models populated, empty inference tables, and a bootstrap that
    fails with "high <= 0". The cause is usually a control variable
    downloaded over too short a date range.

    This function, before the run:
      1) computes coverage per feature;
      2) drops features below min_cov and says which and why;
      3) computes the usable training sample per feature set and disables any
         set that cannot be estimated;
      4) stops with an explicit error if the control set itself is empty,
         rather than letting the pipeline spin.
    """
    rows = []
    n = len(df)
    allfeat = sorted({f for v in feature_sets.values() for f in v})
    for f in allfeat:
        s_ = df[f]
        cov = float(s_.notna().mean())
        first = s_.first_valid_index()
        last = s_.last_valid_index()
        rows.append({"feature": f, "coverage": round(cov, 4),
                     "n_valid": int(s_.notna().sum()),
                     "first": df["date"].iloc[first].date() if first is not None else None,
                     "last": df["date"].iloc[last].date() if last is not None else None,
                     "dropped": cov < min_cov})
    audit = pd.DataFrame(rows).sort_values("coverage")
    if out_dir:
        audit.to_csv(out_dir / "T00_feature_coverage.csv", index=False)

    bad = audit[audit.dropped]["feature"].tolist()
    if bad:
        print(f"\n  The following features fall below {min_cov:.0%} coverage and "
              f"have been dropped (they would otherwise collapse their set):")
        for r in audit[audit.dropped].itertuples():
            print(f"       {r.feature:<20} coverage {r.coverage:>6.1%}  "
                  f"({r.n_valid} periods, {r.first} ~ {r.last})")
        print("     -> If one of these is a key control, re-download the full "
              "date range before running.")

    clean, report = {}, []
    for k, v in feature_sets.items():
        kept = [f for f in v if f not in bad]
        if len(kept) < 2:
            report.append((k, len(kept), 0, "fewer than 2 features"))
            continue
        usable = int(df[kept + ["y"]].dropna().shape[0])
        train_at_start = int(df.iloc[:initial_train][kept + ["y"]].dropna().shape[0])
        if train_at_start < min_train_sub:
            report.append((k, len(kept), train_at_start,
                           f"initial training sample {train_at_start} < {min_train_sub}"))
            continue
        clean[k] = kept
        report.append((k, len(kept), usable, "OK"))

    print(f"\n  [feature-set availability]  (total {n}, initial training {initial_train})")
    for k, nf, nu, st in report:
        flag = "[ok]  " if st == "OK" else "[skip]"
        print(f"     {flag} {k:<20} {nf:>2} features, usable {nu:>4}   {st}")

    if "CTRL" not in clean:
        raise SystemExit(
            "\nERROR: the CTRL baseline feature set is unavailable; stopping.\n"
            "   The usual cause is a control variable with too short a history.\n"
            "   Check the availability table above and T00_feature_coverage.csv,\n"
            "   re-export the series over the full date range, or lower\n"
            "   --min-feature-coverage.")
    return clean, audit


# =============================================================================
# 3. Out-of-sample loop
# =============================================================================
def run_oos_weekly(df: pd.DataFrame, feature_sets: dict, initial_train: int,
                   est: FastEstimator, do_quantiles: bool = True,
                   do_shap: bool = False, quant_sets=("CTRL", "CTRL_GPRfull"),
                   vol_window: int = 52, min_train_sub: int = 100
                   ) -> tuple[pd.DataFrame, dict]:
    import time
    n = len(df)
    rows, shap_store = [], {k: [] for k in feature_sets}
    skipped: dict[str, int] = {}
    t0 = time.time()
    for i, t in enumerate(range(initial_train, n)):
        tr, te = df.iloc[:t], df.iloc[[t]]
        y_tr = tr["y"].values
        rec = {"target_month": te["target_month"].values[0], "y": te["y"].values[0]}

        # benchmark family
        rec["B_zero"] = 0.0
        rec["B_histmean"] = float(np.nanmean(y_tr))
        rec["B_histmean_CT"] = max(rec["B_histmean"], 0.0)
        hist = tr["eua_return_t"].values
        m_ = ~np.isnan(hist)
        h = hist[m_]
        if len(h) > 5:
            A = np.column_stack([np.ones(len(h) - 1), h[:-1]])
            b, *_ = np.linalg.lstsq(A, h[1:], rcond=None)
            rec["B_AR1"] = float(b[0] + b[1] * h[-1])
        else:
            rec["B_AR1"] = rec["B_histmean"]
        rec["sigma2"] = float(np.nanvar(y_tr[-vol_window:]))

        for fname, feats in feature_sets.items():
            sub = tr.dropna(subset=feats + ["y"])
            if len(sub) < min_train_sub or np.isnan(te[feats].values).any():
                skipped[fname] = skipped.get(fname, 0) + 1
                continue
            preds, xgb_model = est.fit_predict(fname, sub[feats].values,
                                               sub["y"].values, te[feats].values)
            for mn, v in preds.items():
                rec[f"{mn}_{fname}"] = v
            if do_shap:
                try:
                    import shap
                    sv = shap.TreeExplainer(xgb_model).shap_values(te[feats].values)[0]
                    shap_store[fname].append(dict(zip(feats, sv.tolist()))
                                             | {"target_month": rec["target_month"]})
                except Exception:
                    pass

        if do_quantiles:
            for fname in quant_sets:
                feats = feature_sets.get(fname)
                if not feats:
                    continue
                sub = tr.dropna(subset=feats + ["y"])
                if len(sub) < min_train_sub or np.isnan(te[feats].values).any():
                    continue
                qp = est.fit_predict_quantiles(sub[feats].values, sub["y"].values,
                                               te[feats].values, QUANTILE_GRID)
                for q, v in qp.items():
                    rec[f"Q{int(q*100)}_{fname}"] = v
        rows.append(rec)

        if i % 25 == 0 or t == n - 1:
            el = time.time() - t0
            done = i + 1
            eta = el / done * (n - initial_train - done)
            print(f"    {done}/{n-initial_train} periods  elapsed {el/60:.1f}min  "
                  f"eta {eta/60:.1f}min", flush=True)

    total = n - initial_train
    if skipped:
        print("\n  Some feature sets had predictions skipped "
              "(training sample too small, or a missing test point):")
        for k, v in sorted(skipped.items(), key=lambda t: -t[1]):
            print(f"       {k:<20} skipped {v}/{total} periods ({v/total:.0%})")
        if max(skipped.values()) >= total:
            raise SystemExit(
                "\nERROR: a feature set was skipped in 100% of periods, which "
                "means one of its variables is unavailable across the entire "
                "out-of-sample window.\n   Check T00_feature_coverage.csv and "
                "re-download that variable.")

    oos = pd.DataFrame(rows)
    oos["target_month"] = pd.to_datetime(oos["target_month"])
    ml_cols = [c for c in oos.columns
               if any(c.startswith(p) for p in ("Ridge_", "ENet_", "PCR_", "RF_", "XGB_"))]
    gpr_cols = [c for c in ml_cols if "GPR" in c]
    if gpr_cols:
        oos["COMB_equal_GPR"] = oos[gpr_cols].mean(axis=1)
        oos["COMB_dmsfe_GPR"] = _dmsfe_combine(oos, gpr_cols, theta=0.9)
    ctrl_cols = [c for c in ml_cols if c.endswith("_CTRL")]
    if ctrl_cols:
        oos["COMB_equal_CTRL"] = oos[ctrl_cols].mean(axis=1)
    return oos, shap_store


# =============================================================================
# 4. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None,
                    help="path to the panel CSV; searched automatically if omitted")
    ap.add_argument("--price-dir", default=None,
                    help="directory of vendor CSVs for Brent/TTF/Coal/VSTOXX; "
                         "defaults to the panel's own directory")
    ap.add_argument("--out", default="out_weekly")
    ap.add_argument("--freq", default="W", choices=["W", "M"],
                    help="W = weekly (default) / M = monthly. Only window lengths "
                         "change; the model space and every test are identical, so "
                         "the two frequencies are directly comparable.")
    ap.add_argument("--initial-train", type=int, default=None,
                    help="defaults by frequency: 200 weekly, 60 monthly")
    ap.add_argument("--fast", action="store_true",
                    help="skip SHAP and the random forest")
    ap.add_argument("--min-feature-coverage", type=float, default=0.60,
                    help="features below this coverage are dropped (default 0.60)")
    ap.add_argument("--refresh-every", type=int, default=13,
                    help="hyperparameter re-selection interval; 1 = every period")
    ap.add_argument("--placebo-n", type=int, default=6,
                    help="number of AR(1) placebo series. The default of 6 "
                         "reproduces previously generated results but is SMALLER "
                         "than the GPR block, so the placebo is not exactly "
                         "dimension-matched. Set this equal to the size of the "
                         "GPR block (printed below) for an exact match.")
    ap.add_argument("--placebo-seed", type=int, default=42,
                    help="seed for the AR(1) placebo block (default 42). Only the "
                         "*_PLACEBO specifications depend on it; every other row of "
                         "every output table is unchanged, because the "
                         "hyperparameter cache is namespaced per feature set. Run "
                         "the pipeline once per seed into separate --out "
                         "directories, then aggregate them with "
                         "placebo_randomization.py to turn the single-draw "
                         "comparison into a randomisation test with a p-value.")
    ap.add_argument("--no-quantiles", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    FP = FREQ_PARAMS[args.freq]
    if args.initial_train is None:
        args.initial_train = FP["initial_train"]
    print("=" * 78)
    print(f"Single-market out-of-sample pipeline -- {FP['label']}")
    print("=" * 78)

    panel_path = resolve_panel(args.panel)
    price_dir = Path(args.price_dir) if args.price_dir else panel_path.parent
    if not price_dir.exists():
        print(f"  [note] --price-dir {price_dir} does not exist, "
              f"using the panel directory")
        price_dir = panel_path.parent
    print(f"  panel    : {panel_path}")
    print(f"  controls : {price_dir}")
    print(f"  placebo  : {args.placebo_n} AR(1) series, seed {args.placebo_seed}")
    df = build_features_weekly(panel_path, price_dir, freq=FP["freq"], fp=FP,
                               n_placebo=args.placebo_n,
                               placebo_seed=args.placebo_seed)
    FS = make_feature_sets(set(df.columns))

    # state variables
    df["gpr_state_level"] = df.get("gpr_log", pd.Series(np.nan, index=df.index))
    tm = df["target_month"]
    crisis = np.zeros(len(df), dtype=bool)
    for _, (s, e) in CRISIS_WINDOWS.items():
        crisis |= ((tm >= pd.Timestamp(s)) & (tm <= pd.Timestamp(e))).values
    df["gpr_state_crisis"] = crisis.astype(float)

    # ---- frequency guard ----------------------------------------------------
    # --freq selects window lengths only; it does NOT resample. The panel must
    # already be at the requested frequency. Passing the weekly panel with
    # --freq M produces a hybrid that is neither: ~604 weekly rows modelled
    # with monthly-length momentum and variance windows and a 60-period
    # training window. It runs to completion and prints "monthly periods",
    # so nothing downstream reveals the error. Refuse it here instead.
    spacing = float(pd.Series(df.index).diff().dt.days.median())
    expect = {"W": (4, 11), "M": (20, 45)}[args.freq]
    if not (expect[0] <= spacing <= expect[1]):
        other = "M" if args.freq == "W" else "W"
        raise SystemExit(
            f"\n[stop] --freq {args.freq} was requested, but the panel's median row "
            f"spacing is {spacing:.0f} days.\n"
            f"  Expected {expect[0]}-{expect[1]} days for --freq {args.freq}.\n"
            f"  --freq changes window lengths ONLY; it does not resample. The panel\n"
            f"  must already be at the requested frequency.\n"
            f"  Either pass the {'monthly' if args.freq == 'M' else 'weekly'} panel, "
            f"or run with --freq {other}.\n"
            f"  Panel given: {panel_path}")

    print(f"\n[data] {len(df)} {FP['label']} periods: "
          f"{df.date.min():%Y-%m-%d} ~ {df.date.max():%Y-%m-%d}")
    print(f"[out-of-sample] {len(df) - args.initial_train} periods "
          f"(initial training {args.initial_train})")
    n_ctrl = len(FS.get("CTRL", []))
    if n_ctrl < 6:
        print(f"\n  WARNING: CTRL has only {n_ctrl} control variables. Without "
              f"energy price controls the baseline is too weak and no "
              f"'GPR is better' conclusion is credible. Supply Brent / TTF / "
              f"Coal / VSTOXX via --price-dir.")
    FS, feat_audit = audit_features(df, FS, args.initial_train,
                                    min_cov=args.min_feature_coverage,
                                    min_train_sub=FP["min_train_sub"],
                                    out_dir=out_dir)
    print("\n[final feature sets]")
    for k, v in FS.items():
        print(f"  {k:<20} {len(v):>2}: {v}")

    # ---- placebo dimension check -------------------------------------------
    # The placebo is only interpretable as a dimension-matched control when
    # the two added blocks have equal cardinality.
    if "CTRL" in FS and "CTRL_PLACEBO" in FS and "CTRL_GPRfull" in FS:
        k_ctrl = len(FS["CTRL"])
        add_plc = len(FS["CTRL_PLACEBO"]) - k_ctrl
        add_gpr = len(FS["CTRL_GPRfull"]) - k_ctrl
        if add_plc != add_gpr:
            print(f"\n  {'!' * 70}")
            print(f"  PLACEBO IS NOT DIMENSION-MATCHED")
            print(f"    CTRL_PLACEBO adds {add_plc} noise series")
            print(f"    CTRL_GPRfull adds {add_gpr} geopolitical variables")
            print(f"  The smaller block estimates fewer parameters and so pays a "
                  f"smaller\n  estimation-error penalty. With the noise block "
                  f"smaller, it is advantaged,\n  and 'the GPR block fails to beat "
                  f"noise' is partly attributable to size\n  rather than to "
                  f"information content.")
            print(f"  -> rerun with --placebo-n {add_gpr} for an exact match.")
            print(f"  {'!' * 70}")
        else:
            print(f"\n  [ok] placebo dimension-matched: both blocks add "
                  f"{add_gpr} variables")
            print(f"       This is one draw (seed {args.placebo_seed}). A single "
                  f"draw is not a reference\n"
                  f"       distribution: repeat with several --placebo-seed values "
                  f"and aggregate\n"
                  f"       them with placebo_randomization.py for a p-value.")

    est = FastEstimator(refresh_every=args.refresh_every, use_rf=not args.fast)
    print(f"\n[estimator] hyperparameters re-selected every {args.refresh_every} "
          f"periods (training window only, no look-ahead); random forest "
          f"{'off' if args.fast else 'on'}")
    oos, shap_store = run_oos_weekly(df, FS, args.initial_train, est,
                                     do_quantiles=not args.no_quantiles,
                                     do_shap=not args.fast,
                                     vol_window=FP["vol_window"],
                                     min_train_sub=FP["min_train_sub"])
    oos = oos.merge(df[["target_month", "gpr_state_level", "gpr_state_crisis"]],
                    on="target_month", how="left")
    oos.to_csv(out_dir / "T0_oos_forecasts.csv", index=False)
    print(f"\n[done] {len(oos)} out-of-sample predictions")

    # Common sample: model comparison must be on identical observations.
    mcols = [c for c in oos.columns
             if any(c.startswith(pf) for pf in ("Ridge_", "ENet_", "PCR_", "RF_",
                                                "XGB_", "COMB_", "B_"))]
    cov = (1 - oos[mcols].isna().mean()).sort_values()
    bad = cov[cov < 0.999]
    if len(bad):
        print("\n  Models with incomplete predictions (below 100% coverage):")
        print("     " + ", ".join(f"{k}={v:.1%}" for k, v in bad.head(8).items()))
    core = [c for c in mcols if not c.startswith("COMB_dmsfe")]
    mask = oos[core].notna().all(axis=1)
    print(f"  common sample: {int(mask.sum())}/{len(oos)} periods ({mask.mean():.1%})")
    pd.DataFrame({"model": cov.index, "coverage": cov.values}).to_csv(
        out_dir / "T0b_model_coverage.csv", index=False)
    if mask.mean() < 1.0:
        oos_full, oos = oos, oos[mask].reset_index(drop=True)
        oos_full.to_csv(out_dir / "T0_oos_forecasts_all.csv", index=False)

    res = evaluate(oos, out_dir, periods_per_year=FP["periods_per_year"])
    print("\n--- T1 accuracy (top 12) ---")
    print(res["metrics"].head(12).to_string(index=False))

    # ---- reading the placebo comparison ------------------------------------
    m = res["metrics"].set_index("model")["R2_OS_vs_histmean_pct"]
    for mdl in ("Ridge", "XGB", "ENet"):
        a, b, c = f"{mdl}_CTRL", f"{mdl}_CTRL_PLACEBO", f"{mdl}_CTRL_GPRfull"
        if all(x in m.index for x in (a, b, c)) and not any(
                pd.isna(m[x]) for x in (a, b, c)):
            print(f"\n  [placebo test {mdl}] CTRL={m[a]:+.2f}%  "
                  f"CTRL+noise={m[b]:+.2f}%  CTRL+GPR={m[c]:+.2f}%")
            if m[c] <= m[b]:
                print("     The GPR block does not beat the noise block -> the "
                      "improvement comes from dimensionality, not information")
            else:
                print("     The GPR block beats the noise block -> there is a "
                      "genuine information increment")

    print("\n--- T3 Giacomini-White ---")
    print(res["gw"].to_string(index=False))
    print("\n--- T6 model confidence set ---")
    print(res["mcs"]["mcs_set"])
    print("\n--- T8b tail inference ---")
    if res.get("tail") is not None:
        print(res["tail"].round(4).to_string(index=False))
    print("\n--- T9 economic value (top 8) ---")
    print(res["econ"].head(8).to_string(index=False))

    imp = None
    if "CTRL_GPRfull" in FS:
        # Guard: with an empty post-dropna sample the bootstrap would raise
        # "ValueError: high <= 0", which is a downstream symptom of a feature
        # coverage problem rather than a bootstrap problem.
        n_imp = df[FS["CTRL_GPRfull"] + ["y"]].dropna().shape[0]
        if n_imp < 50:
            print(f"\n  Skipping feature importance: only {n_imp} usable rows "
                  f"(need >= 50)")
        else:
            imp = block_bootstrap_importance(df, FS["CTRL_GPRfull"], n_boot=200)
    if imp is not None:
        imp.to_csv(out_dir / "T10_block_bootstrap_importance.csv", index=False)
        print("\n--- T10 feature importance (top 8) ---")
        print(imp.head(8).to_string(index=False))
        separated = (imp["ci_lo_2.5"].max() < imp["ci_hi_97.5"].min())
        print("  Note: the 95% intervals of all features "
              + ("overlap -> the importance ordering is not statistically "
                 "distinguishable and should be reported as such"
                 if not separated else "are separated"))

    for k, v in shap_store.items():
        if v:
            pd.DataFrame(v).to_csv(out_dir / f"T11_oos_shap_{k}.csv", index=False)
    make_figures(oos, res["metrics"], out_dir)
    make_tail_figure(res.get("tail"), out_dir)
    print(f"\n[all done] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
