"""
=============================================================================
forecast_eval.py -- out-of-sample evaluation and inference toolbox
=============================================================================
Implements the inference tools a referee will check in an out-of-sample
forecasting study:

  A. Loss and accuracy metrics
     - rmse / mae / R2_OS (Campbell & Thompson 2008)
     - pinball loss
     - CRPS, approximated from a set of quantile forecasts
     - directional accuracy (excluding degenerate zero forecasts)

  B. Predictive-accuracy tests
     - dm_test            Diebold-Mariano (1995) with the Harvey-Leybourne-
                          Newbold small-sample correction.
                          FOR NON-NESTED MODEL PAIRS ONLY.
     - clark_west_test    Clark & West (2007) MSPE-adjusted test for nested
                          models, with HAC standard errors
     - gw_cpa_test        Giacomini & White (2006) conditional predictive
                          ability
     - fluctuation_test   Giacomini & Rossi (2010), time variation in
                          relative predictive advantage
     - mcs                Hansen, Lunde & Nason (2011) model confidence set
     - romano_wolf        Romano & Wolf (2005) stepdown multiple testing

  C. Economic value
     - mv_timing_utility  certainty-equivalent return, Sharpe ratio and
                          turnover for a mean-variance timing investor

Pure numpy/scipy. References at the end of the file.

NOTE ON USAGE WITH crossmarket_falsification.py
-----------------------------------------------
That module calls `dm_test` on PINBALL LOSSES with power=1, not on forecast
errors. This is deliberate and valid: pinball loss is non-negative, so
|L_a|^1 - |L_b|^1 reduces exactly to the loss differential L_a - L_b. Passing
losses with power=2 would square them and is NOT equivalent.
=============================================================================
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "newey_west_lrv", "rmse", "mae", "r2_os", "pinball_loss", "crps_from_quantiles",
    "direction_accuracy", "dm_test", "clark_west_test", "gw_cpa_test",
    "fluctuation_test", "mcs", "romano_wolf", "mv_timing_utility",
    "stationary_bootstrap_indices",
]


# =============================================================================
# 0. Primitives
# =============================================================================
def newey_west_lrv(x: np.ndarray, lag: int | None = None) -> float:
    """Newey-West long-run variance with a Bartlett kernel.
    lag=None uses the Newey-West (1994) automatic bandwidth."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(int(lag), 0)
    xc = x - x.mean()
    lrv = float(np.dot(xc, xc) / n)
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = float(np.dot(xc[l:], xc[:-l]) / n)
        lrv += 2.0 * w * cov
    return max(lrv, 1e-16)


def stationary_bootstrap_indices(n: int, block_len: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap indices. Preserves serial
    dependence; used by the model confidence set and the Romano-Wolf stepdown."""
    p = 1.0 / max(block_len, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


# =============================================================================
# A. Loss and accuracy metrics
# =============================================================================
def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def r2_os(y: np.ndarray, yhat: np.ndarray, yhat_bench: np.ndarray) -> float:
    """
    Campbell & Thompson (2008) out-of-sample R-squared:

        R2_OS = 1 - SSE_model / SSE_benchmark

    The benchmark is normally the prevailing mean. R2_OS > 0 means the model
    beats the benchmark. This is the standard scale in the return
    predictability literature: more comparable and more interpretable than
    raw RMSE.
    """
    y = np.asarray(y, dtype=float)
    sse_m = float(np.sum((y - np.asarray(yhat, dtype=float)) ** 2))
    sse_b = float(np.sum((y - np.asarray(yhat_bench, dtype=float)) ** 2))
    return 1.0 - sse_m / sse_b if sse_b > 0 else np.nan


def pinball_loss(y: np.ndarray, yhat_q: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss. yhat_q must be a forecast of the q-th
    quantile, not of the conditional mean."""
    d = np.asarray(y, dtype=float) - np.asarray(yhat_q, dtype=float)
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def crps_from_quantiles(y: np.ndarray, q_preds: dict[float, np.ndarray]) -> float:
    """
    CRPS approximated from a set of quantile forecasts:
    CRPS ~ 2 * integral of pinball_q over q  (Gneiting & Ranjan 2011).
    q_preds maps quantile level -> forecast array. A denser grid is more
    accurate.
    """
    qs = np.array(sorted(q_preds.keys()), dtype=float)
    losses = np.array([pinball_loss(y, q_preds[q], q) for q in qs])
    # trapezoidal integration, times 2
    return float(2.0 * np.trapezoid(losses, qs) / (qs[-1] - qs[0]) * (qs[-1] - qs[0]))


def direction_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Directional accuracy, excluding observations whose forecast is exactly
    zero. Without that exclusion a zero-return benchmark is mechanically
    scored at 0% accuracy, which is a misuse of the metric rather than a
    property of the benchmark.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = yhat != 0
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y[m]) == np.sign(yhat[m])))


# =============================================================================
# B. Predictive-accuracy tests
# =============================================================================
def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2,
            hln: bool = True) -> tuple[float, float]:
    """
    Diebold-Mariano test on d_t = |e1|^power - |e2|^power. A positive mean
    indicates that model 2 is more accurate.

    IMPORTANT: the DM statistic does not have a standard limiting
    distribution when one model NESTS the other (Clark & McCracken 2001).
    Applying it to a "controls" versus "controls + extra block" pair is a
    misuse that recurs in the applied literature. For nested pairs use
    clark_west_test or gw_cpa_test instead.

    Implementation notes:
      1. The long-run variance uses the Newey-West automatic bandwidth, with
         lag = h-1 when h > 1. A naive loop `for lag in range(1, h)` silently
         does nothing at h=1 and reduces to the contemporaneous variance.
      2. The HLN small-sample correction scales the statistic and the
         reference distribution becomes t(n-1).
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    lag = None if h <= 1 else h - 1
    lrv = newey_west_lrv(d, lag=lag)
    stat = float(np.mean(d) / np.sqrt(lrv / n))
    if hln:
        k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= k
        p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, float(p)


def clark_west_test(y: np.ndarray, pred_r: np.ndarray, pred_u: np.ndarray,
                    h: int = 1) -> tuple[float, float]:
    """
    Clark & West (2007) MSPE-adjusted test for nested models, HAC version.
    The restricted model pred_r is nested in the unrestricted model pred_u.

        f_t = (y - yhat_r)^2 - [(y - yhat_u)^2 - (yhat_r - yhat_u)^2]

    One-sided right-tail test on the mean of f_t.

    INTERPRETATION BOUNDARY -- a referee will ask about this:
      The Clark-West null is that the ADDITIONAL POPULATION COEFFICIENTS are
      zero. Rejection implies population-level incremental information. It
      does NOT imply lower out-of-sample MSE, because parameter estimation
      error can more than offset that information.

      A significant CW statistic accompanied by a HIGHER RMSE and an
      insignificant DM statistic is therefore not a contradiction; it is the
      textbook case, and must be reported as such. Reading a CW rejection as
      evidence of superior point forecasts is over-interpretation.

    Standard errors are Newey-West HAC, not iid.
    """
    y = np.asarray(y, dtype=float)
    pr = np.asarray(pred_r, dtype=float)
    pu = np.asarray(pred_u, dtype=float)
    f = (y - pr) ** 2 - ((y - pu) ** 2 - (pr - pu) ** 2)
    n = len(f)
    lag = None if h <= 1 else h - 1
    se = np.sqrt(newey_west_lrv(f, lag=lag) / n)
    stat = float(np.mean(f) / se)
    return stat, float(1 - stats.norm.cdf(stat))


def gw_cpa_test(loss_a: np.ndarray, loss_b: np.ndarray, z: np.ndarray | None = None,
                lag: int | None = None) -> dict:
    """
    Giacomini & White (2006) conditional predictive ability (CPA) test.

    H0: E[ dL_{t+1} | F_t ] = 0, where dL = L_A - L_B.

        W = n * ( (1/n) sum h_t dL_{t+1} )' Omega^{-1}
                ( (1/n) sum h_t dL_{t+1} )   ~  chi2(q)

    with h_t = (1, z_t)' a test function observable at t.

    Why this matters beyond DM and CW:
      - DM and CW answer "which model is more accurate on average";
      - GW-CPA answers "in WHICH STATES is which model more accurate", and
        supports a conditional decision rule directly.
      - Setting z_t to the lagged predictor level, a shock dummy or a crisis
        indicator turns "the predictor only helps in stressed states" from a
        qualitative claim into a formal hypothesis with a p-value. That is
        the legitimate route from an unconditional null to a conditional
        result.

    SIGN CONVENTION: with dL = L_A - L_B, a significant statistic accompanied
    by a NEGATIVE mean loss difference is evidence that model A -- the
    benchmark -- is superior. The sign must be read alongside the p-value.

    Returns the Wald statistic, its p-value, and the auxiliary regression
    coefficients of dL on h_t, which indicate direction.
    """
    la = np.asarray(loss_a, dtype=float)
    lb = np.asarray(loss_b, dtype=float)
    dl = la - lb                      # > 0 means B is better
    n = len(dl)
    if z is None:
        H = np.ones((n, 1))
    else:
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z[:, None]
        H = np.column_stack([np.ones(n), z])
    q = H.shape[1]

    m = H * dl[:, None]               # n x q moments
    mbar = m.mean(axis=0)

    # Newey-West HAC covariance
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    mc = m - mbar
    S = mc.T @ mc / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = mc[l:].T @ mc[:-l] / n
        S += w * (G + G.T)
    S += np.eye(q) * 1e-12

    W = float(n * mbar @ np.linalg.solve(S, mbar))
    p = float(1 - stats.chi2.cdf(W, df=q))

    # Auxiliary regression dL_{t+1} = h_t'delta + u; the sign of delta says
    # when B is better.
    delta, *_ = np.linalg.lstsq(H, dl, rcond=None)

    return {"wald_stat": W, "df": q, "p_value": p,
            "delta_coefficients": delta.tolist(),
            "mean_loss_diff_A_minus_B": float(dl.mean())}


def fluctuation_test(loss_a: np.ndarray, loss_b: np.ndarray, window: int,
                     alpha: float = 0.05) -> dict:
    """
    Giacomini & Rossi (2010) fluctuation test: standardised loss differentials
    within a rolling window, testing whether relative predictive advantage is
    stable over the out-of-sample period.

        F_t = (1/sqrt(m)) sum_{j in window} dL_j / sigma

    with sigma the full-sample HAC long-run standard deviation. If max|F_t|
    exceeds the critical value, constant relative predictive ability is
    rejected.

    Relevance: carbon markets span COVID, the Russia-Ukraine conflict, the
    energy crisis and successive regulatory reforms. "Constant predictive
    value" is an implausible prior, and this test lets the paper describe
    WHEN a predictor helps rather than reporting a single average.
    """
    dl = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(dl)
    m = int(window)
    if m >= n:
        return {"error": "window >= sample size"}
    sigma = np.sqrt(newey_west_lrv(dl))
    stat_path, centers = [], []
    for s in range(0, n - m + 1):
        seg = dl[s:s + m]
        stat_path.append(float(seg.sum() / (np.sqrt(m) * sigma)))
        centers.append(s + m // 2)
    stat_path = np.array(stat_path)
    mu = m / n
    # Giacomini & Rossi (2010) Table 1, two-sided 5% critical values,
    # linearly interpolated in mu = m/P.
    grid_mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    cv_05 = np.array([3.393, 3.179, 3.012, 2.890, 2.779, 2.634, 2.560, 2.433, 2.248])
    cv = float(np.interp(mu, grid_mu, cv_05))
    return {"window": m, "mu_ratio": float(mu), "critical_value_5pct": cv,
            "max_abs_statistic": float(np.max(np.abs(stat_path))),
            "reject_constant_relative_performance": bool(np.max(np.abs(stat_path)) > cv),
            "statistic_path": stat_path.tolist(), "path_center_index": centers}


def mcs(losses: dict[str, np.ndarray], alpha: float = 0.10, n_boot: int = 2000,
        block_len: float = 4.0, seed: int = 42) -> dict:
    """
    Hansen, Lunde & Nason (2011) model confidence set, range statistic T_R
    with the stationary bootstrap.

    Why this is necessary: running several DM and CW comparisons without
    multiple-comparison control makes any single p < 0.05 fragile, especially
    with a short out-of-sample window. The MCS returns the set of models that
    cannot be excluded at confidence 1-alpha, which is the standard answer to
    "which of these models are distinguishable?".

    Returns the surviving set and each model's MCS p-value.
    """
    names = list(losses.keys())
    L = np.column_stack([np.asarray(losses[k], dtype=float) for k in names])
    n, M0 = L.shape
    rng = np.random.default_rng(seed)

    boot_idx = np.array([stationary_bootstrap_indices(n, block_len, rng)
                         for _ in range(n_boot)])

    alive = list(range(M0))
    p_values = {}
    while len(alive) > 1:
        sub = L[:, alive]
        k = len(alive)
        dbar = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                dbar[i, j] = np.mean(sub[:, i] - sub[:, j])

        # bootstrap distribution
        boot_d = np.empty((n_boot, k, k))
        for b in range(n_boot):
            sb = sub[boot_idx[b], :]
            mb = sb.mean(axis=0)
            boot_d[b] = mb[:, None] - mb[None, :]
        var_d = boot_d.var(axis=0) + 1e-16

        t_stat = dbar / np.sqrt(var_d)
        T_R = np.max(np.abs(t_stat))
        boot_t = np.abs((boot_d - dbar[None, :, :]) / np.sqrt(var_d)[None, :, :])
        boot_TR = boot_t.reshape(n_boot, -1).max(axis=1)
        p = float(np.mean(boot_TR >= T_R))

        if p >= alpha:
            for idx in alive:
                p_values.setdefault(names[idx], max(p, p_values.get(names[idx], 0.0)))
            break

        # eliminate the relatively worst model (largest row-mean t)
        t_i = (dbar / np.sqrt(var_d)).mean(axis=1)
        worst_local = int(np.argmax(t_i))
        worst_global = alive[worst_local]
        p_values[names[worst_global]] = p
        alive.pop(worst_local)

    if len(alive) == 1:
        p_values.setdefault(names[alive[0]], 1.0)

    return {"mcs_set": [names[i] for i in alive],
            "mcs_pvalues": p_values, "alpha": alpha}


def romano_wolf(loss_bench: np.ndarray, losses_alt: dict[str, np.ndarray],
                n_boot: int = 2000, block_len: float = 4.0, seed: int = 42,
                one_sided: bool = True) -> dict:
    """
    Romano & Wolf (2005) stepdown multiple testing, controlling the
    family-wise error rate. Used to test several candidate models against a
    common benchmark simultaneously; more powerful than Bonferroni.
    """
    names = list(losses_alt.keys())
    D = np.column_stack([np.asarray(loss_bench, float) - np.asarray(losses_alt[k], float)
                         for k in names])          # > 0 means candidate beats benchmark
    n, K = D.shape
    rng = np.random.default_rng(seed)
    boot_idx = np.array([stationary_bootstrap_indices(n, block_len, rng)
                         for _ in range(n_boot)])

    dbar = D.mean(axis=0)
    se = np.array([np.sqrt(newey_west_lrv(D[:, k]) / n) for k in range(K)])
    tobs = dbar / (se + 1e-16)
    boot_t = np.empty((n_boot, K))
    for b in range(n_boot):
        Db = D[boot_idx[b], :]
        boot_t[b] = (Db.mean(axis=0) - dbar) / (se + 1e-16)
    if not one_sided:
        tobs_use, boot_use = np.abs(tobs), np.abs(boot_t)
    else:
        tobs_use, boot_use = tobs, boot_t

    remaining = list(range(K))
    adj_p = np.ones(K)
    while remaining:
        maxdist = boot_use[:, remaining].max(axis=1)
        pv = np.array([np.mean(maxdist >= tobs_use[k]) for k in remaining])
        pv = np.maximum.accumulate(np.sort(pv))[np.argsort(np.argsort(pv))]
        for pos, k in enumerate(remaining):
            adj_p[k] = max(adj_p[k] if adj_p[k] < 1 else 0.0, pv[pos])
        rejected = [k for pos, k in enumerate(remaining) if pv[pos] < 0.05]
        if not rejected:
            break
        remaining = [k for k in remaining if k not in rejected]
    return {"names": names, "t_stats": tobs.tolist(), "rw_adjusted_p": adj_p.tolist()}


# =============================================================================
# C. Economic value
# =============================================================================
def mv_timing_utility(y: np.ndarray, mu_hat: np.ndarray, sigma2_hat: np.ndarray,
                      gamma: float = 5.0, rf: float = 0.0,
                      w_bounds: tuple[float, float] = (-0.5, 1.5),
                      periods_per_year: int = 12, tc_bps: float = 10.0) -> dict:
    """
    Economic value for a mean-variance timing investor
    (Campbell & Thompson 2008; Rapach, Strauss & Zhou 2010).

        w_t = (1/gamma) * mu_hat_{t+1} / sigma2_hat_{t+1},  clipped to w_bounds
        r_p,t+1 = w_t * y_{t+1} + (1 - w_t) * rf
        CER = mean(r_p) - (gamma/2) * var(r_p)

    Why this belongs in a finance paper rather than a forecasting paper: a
    0.001 difference in RMSE is of no interest, whereas an annualised
    certainty-equivalent gain is a result that can be stated in an abstract.
    Including transaction costs (tc_bps) addresses the standard objection
    that statistical significance is not tradeable.
    """
    y = np.asarray(y, float)
    w = (1.0 / gamma) * np.asarray(mu_hat, float) / np.maximum(
        np.asarray(sigma2_hat, float), 1e-8)
    w = np.clip(w, w_bounds[0], w_bounds[1])
    turnover = np.abs(np.diff(np.concatenate([[0.0], w])))
    rp = w * y + (1 - w) * rf - turnover * (tc_bps / 10000.0)
    cer = float(np.mean(rp) - 0.5 * gamma * np.var(rp))
    return {
        "CER_per_period": cer,
        "CER_annualized_pct": float(cer * periods_per_year * 100),
        "mean_return_annualized_pct": float(np.mean(rp) * periods_per_year * 100),
        "volatility_annualized_pct": float(np.std(rp) * np.sqrt(periods_per_year) * 100),
        "sharpe_annualized": float(np.mean(rp) / (np.std(rp) + 1e-12)
                                   * np.sqrt(periods_per_year)),
        "avg_absolute_weight": float(np.mean(np.abs(w))),
        "avg_turnover": float(np.mean(turnover)),
        "portfolio_returns": rp,
    }


# =============================================================================
# References
# =============================================================================
REFERENCES = """
Campbell, J.Y., Thompson, S.B. (2008). RFS 21(4), 1509-1531.
Clark, T.E., McCracken, M.W. (2001). J. Econometrics 105(1), 85-110.
Clark, T.E., West, K.D. (2007). J. Econometrics 138(1), 291-311.
Diebold, F.X., Mariano, R.S. (1995). JBES 13(3), 253-263.
Giacomini, R., White, H. (2006). Econometrica 74(6), 1545-1578.
Giacomini, R., Rossi, B. (2010). J. Applied Econometrics 25(4), 595-620.
Gneiting, T., Ranjan, R. (2011). JBES 29(3), 411-422.
Hansen, P.R., Lunde, A., Nason, J.M. (2011). Econometrica 79(2), 453-497.
Harvey, D., Leybourne, S., Newbold, P. (1997). IJF 13(2), 281-291.
Politis, D.N., Romano, J.P. (1994). JASA 89(428), 1303-1313.
Rapach, D.E., Strauss, J.K., Zhou, G. (2010). RFS 23(2), 821-862.
Romano, J.P., Wolf, M. (2005). Econometrica 73(4), 1237-1282.
"""
