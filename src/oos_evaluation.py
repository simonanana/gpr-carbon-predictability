"""
=============================================================================
oos_evaluation.py -- the evaluation layer for the single-market pipeline
=============================================================================
Everything that turns a panel of out-of-sample forecasts into the paper's
tables and figures: accuracy metrics, nested and non-nested tests,
state-dependent decomposition, model confidence set, tail inference, economic
value, bootstrap feature importance, and the plotting routines.

It is consumed by `main_pipeline.py`, which supplies the forecast panel.

SCOPE NOTE
----------
This module contains the evaluation machinery only. The original project file
also carried a feature-construction routine and a standalone entry point for a
monthly, three-control specification. Those are deliberately NOT included
here, because they were superseded: they resampled the weekly panel back to
monthly, merged the long multi-market panel into the single-market panel with
`combine_first` (which inflates rows, since the long panel has one row per
market per date), and reduced the control set to three variables, which makes
the control baseline too weak for any comparison against it to be meaningful.
`main_pipeline.py` replaces all three. Publishing the superseded entry point
alongside it would let a reader generate numbers that contradict the paper.

OUTPUTS WRITTEN BY `evaluate()`
-------------------------------
    T1_accuracy_metrics.csv        R2_OS, RMSE, MAE, directional accuracy
    T2_clark_west.csv              nested-pair tests
    T3_giacomini_white_cpa.csv     conditional predictive ability
    T4_state_dependent_R2OS.csv    R2_OS by crisis / GPR state
    T5_romano_wolf.csv             FWER-controlled stepdown
    T6_model_confidence_set.json   Hansen-Lunde-Nason MCS
    T7_fluctuation_test.json       Giacomini-Rossi (when computable)
    T8_quantile_tail_evaluation.csv
    T8b_tail_inference.csv         the frequency-contrast input
=============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb

import forecast_eval as ml

RANDOM_STATE = 42
QUANTILE_GRID = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
GAMMA = 5.0                      # relative risk aversion for the CER calculation
CRISIS_WINDOWS = {               # used for the state decomposition
    "covid": ("2020-02-01", "2020-06-01"),
    "energy_crisis": ("2021-09-01", "2022-01-01"),
    "russia_ukraine": ("2022-02-01", "2022-12-01"),
    "middle_east": ("2023-10-01", "2024-06-01"),
}


# =============================================================================
# Forecast combination
# =============================================================================
def _dmsfe_combine(oos: pd.DataFrame, cols: list[str], theta: float = 0.9) -> np.ndarray:
    """Discounted mean squared forecast error weighting (Stock & Watson 2004).
    Weights use past errors only, so no look-ahead is introduced."""
    y = oos["y"].values
    P = oos[cols].values
    n, k = P.shape
    out = np.empty(n)
    dm = np.ones(k)
    for i in range(n):
        w = (1.0 / np.maximum(dm, 1e-12))
        w = w / w.sum()
        out[i] = float(P[i] @ w)
        e2 = (y[i] - P[i]) ** 2
        dm = theta * dm + e2 if i > 0 else e2 + 1e-8
    return out


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(oos: pd.DataFrame, out_dir: Path, periods_per_year: int = 12) -> dict:
    y = oos["y"].values
    bench = oos["B_histmean"].values
    _helpers = ("target_month", "y", "sigma2", "gpr_state_level", "gpr_state_crisis")
    model_cols = [c for c in oos.columns if c not in _helpers and not c.startswith("Q")]

    # ---- accuracy table -----------------------------------------------------
    rows = []
    for c in model_cols:
        p = oos[c].values
        rows.append({
            "model": c,
            "rmse": ml.rmse(y, p),
            "mae": ml.mae(y, p),
            "R2_OS_vs_histmean_pct": ml.r2_os(y, p, bench) * 100,
            "direction_accuracy": ml.direction_accuracy(y, p),
        })
    metrics = pd.DataFrame(rows).sort_values("R2_OS_vs_histmean_pct", ascending=False)
    metrics.to_csv(out_dir / "T1_accuracy_metrics.csv", index=False)

    # ---- nested pairs: Clark-West (HAC) ------------------------------------
    # DM is invalid for these pairs; see the warning in forecast_eval.dm_test.
    nested_pairs = [
        ("Ridge_CTRL", "Ridge_CTRL_GPRlevel"), ("Ridge_CTRL", "Ridge_CTRL_GPRsurprise"),
        ("Ridge_CTRL", "Ridge_CTRL_GPRfull"),
        ("XGB_CTRL", "XGB_CTRL_GPRlevel"), ("XGB_CTRL", "XGB_CTRL_GPRsurprise"),
        ("XGB_CTRL", "XGB_CTRL_GPRfull"),
        ("B_histmean", "Ridge_CTRL_GPRfull"), ("B_histmean", "XGB_CTRL_GPRfull"),
    ]
    cw_rows = []
    for r, u in nested_pairs:
        if r in oos.columns and u in oos.columns:
            s, p = ml.clark_west_test(y, oos[r].values, oos[u].values)
            cw_rows.append({"restricted": r, "unrestricted": u,
                            "CW_stat": s, "p_one_sided": p,
                            "delta_RMSE": ml.rmse(y, oos[u].values) - ml.rmse(y, oos[r].values)})
    cw = pd.DataFrame(cw_rows)
    cw.to_csv(out_dir / "T2_clark_west.csv", index=False)

    # ---- Giacomini-White conditional predictive ability ---------------------
    # z_t is a state variable observable at t: the lagged GPR level, or a
    # crisis indicator.
    gw_rows = []
    z_level = _lag_align(oos, "gpr_state_level")
    z_crisis = _lag_align(oos, "gpr_state_crisis")
    tests = [("XGB_CTRL", "XGB_CTRL_GPRfull"), ("Ridge_CTRL", "Ridge_CTRL_GPRfull"),
             ("B_histmean", "COMB_equal_GPR")]
    for a, b in tests:
        if a not in oos.columns or b not in oos.columns:
            continue
        la, lb = (y - oos[a].values) ** 2, (y - oos[b].values) ** 2
        for zname, z in (("unconditional", None), ("lagged_GPR_level", z_level),
                         ("crisis_dummy", z_crisis)):
            if zname != "unconditional" and z is None:
                continue
            r = ml.gw_cpa_test(la, lb, z)
            gw_rows.append({"model_A": a, "model_B": b, "conditioning_variable": zname,
                            "wald_stat": r["wald_stat"], "df": r["df"],
                            "p_value": r["p_value"],
                            "delta_coefficients": json.dumps(
                                [round(v, 6) for v in r["delta_coefficients"]])})
    gw = pd.DataFrame(gw_rows)
    gw.to_csv(out_dir / "T3_giacomini_white_cpa.csv", index=False)

    # ---- state-decomposed R2_OS --------------------------------------------
    state_rows = []
    for label, mask in _state_masks(oos).items():
        if mask.sum() < 6:
            continue
        ys, bs = y[mask], bench[mask]
        for c in ["Ridge_CTRL", "Ridge_CTRL_GPRfull", "XGB_CTRL", "XGB_CTRL_GPRfull",
                  "COMB_equal_GPR", "B_AR1"]:
            if c not in oos.columns:
                continue
            state_rows.append({"state": label, "n_obs": int(mask.sum()), "model": c,
                               "rmse": ml.rmse(ys, oos[c].values[mask]),
                               "R2_OS_pct": ml.r2_os(ys, oos[c].values[mask], bs) * 100})
    states = pd.DataFrame(state_rows)
    states.to_csv(out_dir / "T4_state_dependent_R2OS.csv", index=False)

    # ---- model confidence set + Romano-Wolf --------------------------------
    keep = [c for c in ["B_histmean", "B_AR1", "B_zero", "Ridge_CTRL", "Ridge_CTRL_GPRlevel",
                        "Ridge_CTRL_GPRsurprise", "Ridge_CTRL_GPRfull", "XGB_CTRL",
                        "XGB_CTRL_GPRfull", "PCR_CTRL_GPRfull", "COMB_equal_GPR"]
            if c in oos.columns]
    losses = {c: (y - oos[c].values) ** 2 for c in keep}
    mcs_res = ml.mcs(losses, alpha=0.10, n_boot=1000)
    rw_res = ml.romano_wolf(losses["B_histmean"],
                            {k: v for k, v in losses.items() if k != "B_histmean"},
                            n_boot=1000)
    pd.DataFrame({"model": rw_res["names"], "t_stat": rw_res["t_stats"],
                  "RW_adjusted_p": rw_res["rw_adjusted_p"]}
                 ).to_csv(out_dir / "T5_romano_wolf.csv", index=False)
    with open(out_dir / "T6_model_confidence_set.json", "w") as f:
        json.dump(mcs_res, f, indent=2, ensure_ascii=False)

    # ---- Giacomini-Rossi fluctuation test ----------------------------------
    fl = {}
    if "XGB_CTRL" in oos.columns and "XGB_CTRL_GPRfull" in oos.columns:
        fl = ml.fluctuation_test((y - oos["XGB_CTRL"].values) ** 2,
                                 (y - oos["XGB_CTRL_GPRfull"].values) ** 2,
                                 window=max(12, int(0.3 * len(y))))
        with open(out_dir / "T7_fluctuation_test.json", "w") as f:
            json.dump({k: v for k, v in fl.items()}, f, indent=2)

    # ---- quantile / tail evaluation ----------------------------------------
    q_rows = []
    for fname in ("CTRL", "CTRL_GPRfull"):
        cols = {q: f"Q{int(q*100)}_{fname}" for q in QUANTILE_GRID}
        if not all(c in oos.columns for c in cols.values()):
            continue
        qp = {q: oos[c].values for q, c in cols.items()}
        row = {"feature_set": fname, "CRPS": ml.crps_from_quantiles(y, qp)}
        for q in QUANTILE_GRID:
            row[f"pinball_q{int(q*100)}"] = ml.pinball_loss(y, qp[q], q)
            row[f"hitrate_q{int(q*100)}"] = float(np.mean(y <= qp[q]))
        q_rows.append(row)
    qtab = pd.DataFrame(q_rows)
    if len(qtab):
        qtab.to_csv(out_dir / "T8_quantile_tail_evaluation.csv", index=False)

    # ---- formal tail inference ---------------------------------------------
    # Reporting pinball losses alone is descriptive; a referee will ask for
    # inference. This block runs DM on the quantile-loss series (valid here:
    # the comparison is on losses, not nested forecasts), the GW conditional
    # test, and Kupiec / Christoffersen VaR backtests on the hit sequence.
    tail_rows = []
    zc = _lag_align(oos, "gpr_state_crisis")
    for q in QUANTILE_GRID:
        ca, cb = f"Q{int(q*100)}_CTRL", f"Q{int(q*100)}_CTRL_GPRfull"
        if ca not in oos.columns or cb not in oos.columns:
            continue
        da = np.asarray(y) - oos[ca].values
        db = np.asarray(y) - oos[cb].values
        la = np.maximum(q * da, (q - 1) * da)
        lb = np.maximum(q * db, (q - 1) * db)
        # power=1: the inputs are already pinball losses, which are
        # non-negative, so |L|^1 reduces to L and the differential is correct.
        dstat, dp = ml.dm_test(la, lb, h=1, power=1)
        gwr = ml.gw_cpa_test(la, lb, zc)
        hits = (np.asarray(y) > oos[cb].values).astype(int) if q >= 0.5 else \
               (np.asarray(y) < oos[cb].values).astype(int)
        exp_rate = (1 - q) if q >= 0.5 else q
        tail_rows.append({
            "quantile": q,
            "pinball_CTRL": float(la.mean()), "pinball_CTRL_GPRfull": float(lb.mean()),
            "pct_improvement": float((la.mean() - lb.mean()) / la.mean() * 100),
            "DM_stat": dstat, "DM_p_two_sided": dp,
            "GW_crisis_wald": gwr["wald_stat"], "GW_crisis_p": gwr["p_value"],
            "VaR_expected_hit_rate": exp_rate, "VaR_actual_hit_rate": float(hits.mean()),
            "kupiec_LR": _kupiec_lr(hits, exp_rate)[0],
            "kupiec_p": _kupiec_lr(hits, exp_rate)[1],
            "christoffersen_ind_p": _christoffersen_ind(hits)[1],
        })
    tail = pd.DataFrame(tail_rows)
    if len(tail):
        tail.to_csv(out_dir / "T8b_tail_inference.csv", index=False)

    # ---- economic value -----------------------------------------------------
    ev_rows = []
    s2 = oos["sigma2"].values
    for c in keep + [c for c in ("COMB_dmsfe_GPR",) if c in oos.columns]:
        r = ml.mv_timing_utility(y, oos[c].values, s2, gamma=GAMMA,
                                 periods_per_year=periods_per_year)
        r.pop("portfolio_returns")
        ev_rows.append({"model": c} | r)
    ev = pd.DataFrame(ev_rows)
    base = ev.loc[ev.model == "B_histmean", "CER_annualized_pct"]
    if len(base):
        ev["CER_gain_vs_histmean_pp"] = ev["CER_annualized_pct"] - float(base.iloc[0])
    ev = ev.sort_values("CER_annualized_pct", ascending=False)
    ev.to_csv(out_dir / "T9_economic_value.csv", index=False)

    return {"metrics": metrics, "cw": cw, "gw": gw, "states": states,
            "mcs": mcs_res, "econ": ev, "fluct": fl, "quantile": qtab,
            "tail": tail if len(tail) else None}


def _kupiec_lr(hits: np.ndarray, p_exp: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional coverage likelihood-ratio test."""
    from scipy import stats as st
    n, x = len(hits), int(hits.sum())
    if x == 0 or x == n:
        return (np.nan, np.nan)
    ph = x / n
    lr = -2 * ((n - x) * np.log(1 - p_exp) + x * np.log(p_exp)
               - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(lr), float(1 - st.chi2.cdf(lr, 1))


def _christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence test for clustering of exceedances."""
    from scipy import stats as st
    h = np.asarray(hits, dtype=int)
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return (np.nan, np.nan)
    p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0, 1) or p01 in (0,) or p11 in (0,):
        return (np.nan, np.nan)
    lr = -2 * ((n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
               - n00 * np.log(1 - p01) - n01 * np.log(p01)
               - n10 * np.log(1 - p11) - n11 * np.log(p11))
    return float(lr), float(1 - st.chi2.cdf(lr, 1))


def _lag_align(oos: pd.DataFrame, col: str):
    return oos[col].values if col in oos.columns else None


def _state_masks(oos: pd.DataFrame) -> dict:
    tm = oos["target_month"]
    masks = {"full_sample": np.ones(len(oos), dtype=bool)}
    crisis = np.zeros(len(oos), dtype=bool)
    for name, (s, e) in CRISIS_WINDOWS.items():
        m = ((tm >= pd.Timestamp(s)) & (tm <= pd.Timestamp(e))).values
        masks[f"crisis_{name}"] = m
        crisis |= m
    masks["any_crisis"] = crisis
    masks["non_crisis"] = ~crisis
    if "gpr_state_level" in oos.columns:
        med = np.nanmedian(oos["gpr_state_level"].values)
        masks["high_GPR"] = (oos["gpr_state_level"].values > med)
        masks["low_GPR"] = (oos["gpr_state_level"].values <= med)
    return masks


# =============================================================================
# Bootstrap feature importance
# =============================================================================
def block_bootstrap_importance(df: pd.DataFrame, feats: list[str], n_boot: int = 300,
                               block_len: float = 6.0) -> pd.DataFrame:
    """
    Feature importance with the stationary bootstrap rather than iid
    resampling. Plain `rng.integers(0, n, size=n)` destroys serial dependence,
    understates the sampling variability of importance, and produces intervals
    that are too narrow -- a technical error a referee can point to directly.
    """
    sub = df.dropna(subset=feats + ["y"]).reset_index(drop=True)
    X, y = sub[feats].values, sub["y"].values
    n = len(sub)
    rng = np.random.default_rng(RANDOM_STATE)
    store = {f: [] for f in feats}
    for b in range(n_boot):
        idx = ml.stationary_bootstrap_indices(n, block_len, rng)
        m = xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                             min_child_weight=5, random_state=b, verbosity=0)
        m.fit(X[idx], y[idx])
        for f, v in zip(feats, m.feature_importances_):
            store[f].append(float(v))
    rows = []
    for f in feats:
        a = np.array(store[f])
        rows.append({"feature": f, "mean_importance": a.mean(), "std": a.std(),
                     "ci_lo_2.5": np.percentile(a, 2.5), "ci_hi_97.5": np.percentile(a, 97.5)})
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False)


# =============================================================================
# Figures
# =============================================================================
def make_figures(oos: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path):
    y = oos["y"].values
    bench = oos["B_histmean"].values

    # Fig 1: cumulative squared error differential (Welch-Goyal style), which
    # shows whether any predictive advantage is concentrated in time.
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, lab in [("Ridge_CTRL_GPRfull", "Ridge (controls+GPR family)"),
                   ("XGB_CTRL_GPRfull", "XGB (controls+GPR family)"),
                   ("Ridge_CTRL", "Ridge (controls only)"),
                   ("COMB_equal_GPR", "Forecast combination (GPR models)")]:
        if c not in oos.columns:
            continue
        cssed = np.cumsum((y - bench) ** 2 - (y - oos[c].values) ** 2)
        ax.plot(oos["target_month"], cssed, label=lab, lw=1.6)
    for name, (s, e) in CRISIS_WINDOWS.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="grey", alpha=0.15)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylabel("Cumulative SSE difference vs. historical mean")
    ax.set_title("CSSED: when does the GPR information set pay off?\n"
                 "(upward slope = model beats benchmark; shaded = geopolitical stress windows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "F1_cssed.png", dpi=150)
    plt.close(fig)

    # Fig 2: R2_OS bar chart
    fig, ax = plt.subplots(figsize=(9, 6))
    d = metrics.head(14).sort_values("R2_OS_vs_histmean_pct")
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in d["R2_OS_vs_histmean_pct"]]
    ax.barh(d["model"], d["R2_OS_vs_histmean_pct"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Out-of-sample $R^2_{OS}$ vs. prevailing mean (%)")
    ax.set_title("Out-of-sample $R^2_{OS}$ (positive = beats the historical-mean benchmark)")
    fig.tight_layout()
    fig.savefig(out_dir / "F2_r2os.png", dpi=150)
    plt.close(fig)


def make_tail_figure(tail: pd.DataFrame, out_dir: Path):
    """Quantile-loss improvement curve.

    NOTE: the title here is deliberately neutral. An earlier version asserted
    that geopolitical risk improves the tails but not the conditional mean.
    That claim does not hold at weekly frequency, where the upper-tail gain
    reverses sign, so a figure carrying it would contradict the paper.
    """
    if tail is None or not len(tail):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = tail["quantile"].values
    a1.plot(x, tail["pinball_CTRL"], "o-", label="Controls only", lw=1.8)
    a1.plot(x, tail["pinball_CTRL_GPRfull"], "s-", label="Controls + GPR family", lw=1.8)
    a1.set_xlabel("Quantile level $\\tau$")
    a1.set_ylabel("Out-of-sample pinball loss")
    a1.set_title("(a) Quantile loss by $\\tau$")
    a1.legend(fontsize=8)

    cols = ["#27ae60" if v > 0 else "#c0392b" for v in tail["pct_improvement"]]
    a2.bar([str(v) for v in x], tail["pct_improvement"], color=cols)
    a2.axhline(0, color="black", lw=1)
    a2.set_xlabel("Quantile level $\\tau$")
    a2.set_ylabel("Pinball loss reduction from adding GPR (%)")
    a2.set_title("(b) Quantile-loss change from adding the GPR block")
    fig.suptitle("Quantile-loss evaluation of the geopolitical-risk block", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "F3_tail_gains.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
