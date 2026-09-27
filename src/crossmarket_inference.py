"""
=============================================================================
crossmarket_inference.py
Cross-market inference layer.
=============================================================================
Regenerates every statistic reported in Section "Cross-market falsification"
and the appendix diagnostics table that is NOT already in the raw output of
crossmarket_falsification.py. It reads only the market-level summary files, so
it runs in seconds and never re-estimates a forecast.

Input   out_crossmarket/X1_market_results.csv   (9 rows, one per market)
Output  out_crossmarket/Y1_aggregation.csv      -> paper Table 2
        out_crossmarket/Y2_gradient_exact.csv   -> paper Table 3
        out_crossmarket/Y3_robustness.csv       -> paper Table 4
        out_crossmarket/Y4_market_multiplicity.csv -> Appendix Table A4 panel C
        out_crossmarket/Y5_power_curve.csv      -> Section "power" table
        console: everything, in the order it appears in the paper

Why each piece exists
---------------------
1. The nine markets share ONE global GPR series.  Any test that treats them as
   independent overstates significance.  We therefore report every aggregate
   result as a function of an assumed equicorrelation rho-bar, and let the
   reader see where the evidence expires, rather than picking a single number.
2. CTRL vs CTRL+GPR is a NESTED comparison.  Under the null of no incremental
   information the larger model is *expected* to lose, so a median-zero null is
   misspecified.  The dimension-matched placebo (CTRL+PLACEBO vs CTRL+GPR) is
   non-nested and equal-dimensional, so its null IS median-zero.  Inference is
   based on the placebo comparison; control-referenced numbers are descriptive.
3. With K=9 and coarse ordinal characteristics, asymptotic p-values are
   unreliable and the Mann-Whitney test on 2-vs-3 markets cannot reject at 5%
   under any data.  We therefore use exact permutation p-values throughout and
   report the design's minimum detectable effect.

Usage
-----
    python src/crossmarket_inference.py --out out_crossmarket
=============================================================================
"""

from __future__ import annotations

import argparse
from itertools import combinations, permutations
from math import comb, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

# (column, short label, long label, null_status)
#   'descriptive'    = control-referenced, NESTED: under H0 the expected gain is
#                      NEGATIVE (paper Remark 1), so a median-zero test has no
#                      size control.  Report, do not infer from.
#   'well-specified' = placebo-referenced, equal-dimension and non-nested:
#                      under H0 the gain IS median-zero (paper Remark 2).
OUTCOMES = [
    ("gain_gpr_vs_ctrl",     r"$\Delta^{ctrl}$", "R2_OS gain vs controls (pp)",    "descriptive"),
    ("gain_gpr_vs_placebo",  r"$\Delta^{plc}$",  "R2_OS gain vs noise block (pp)", "well-specified"),
    ("pinball_gain_q95",     "G(0.95)",           "Pinball vs CTRL, tau=0.95 (%)",  "descriptive"),
    ("pinball_gain_q5",      "G(0.05)",           "Pinball vs CTRL, tau=0.05 (%)",  "descriptive"),
    ("pinball_gain_plc_q95", "Gp(0.95)",          "Pinball vs NOISE, tau=0.95 (%)", "well-specified"),
    ("pinball_gain_plc_q5",  "Gp(0.05)",          "Pinball vs NOISE, tau=0.05 (%)", "well-specified"),
]
CHARS = [
    ("expo_ord", "European geopolitical exposure"),
    ("imp_ord", "Energy import dependence"),
    ("alloc_ord", "Degree of auctioning"),
]
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]


# =============================================================================
# 1. Aggregation across markets
# =============================================================================
def sign_and_rank(v: np.ndarray) -> dict:
    """Exact sign test and exact Wilcoxon signed-rank test, K = 9."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    K = len(v)
    neg = int((v < 0).sum())
    out = {
        "K": K, "n_negative": neg,
        "mean": v.mean(), "median": float(np.median(v)), "sd": v.std(ddof=1),
        "sign_p": st.binomtest(neg, K, 0.5).pvalue,
    }
    out["wilcoxon_p"] = st.wilcoxon(v).pvalue if K >= 6 else np.nan
    return out


def stouffer(gains: np.ndarray, pvals: np.ndarray, rho: float) -> tuple[float, float]:
    """
    Dependence-adjusted Stouffer combination.

    z_k = sign(gain_k) * Phi^{-1}(1 - p_k/2)  turns a two-sided market-level
    p-value into a signed one-sided normal score (positive = GPR improves).
    Under equicorrelation rho among the z_k,
        Z = sum(z) / sqrt(K + K(K-1)rho)  ->  N(0,1).
    rho is NOT identified from K summary statistics; report over a grid.
    """
    z = np.sign(gains) * st.norm.ppf(1.0 - np.asarray(pvals, float) / 2.0)
    K = len(z)
    Z = z.sum() / np.sqrt(K + K * (K - 1) * rho)
    return float(Z), float(2 * (1 - st.norm.cdf(abs(Z))))


def equivalence_bound(v: np.ndarray, rho: float = 0.0, level: float = 0.95) -> dict:
    """
    One-sided upper confidence bound and two-sided CI on the mean gain,
    inflated by sqrt(1 + (K-1)rho) for equicorrelated markets.
    Converts 'we fail to reject' into 'effects above X are ruled out'.
    """
    v = np.asarray(v, float)
    K = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(K)
    infl = np.sqrt(1 + (K - 1) * rho)
    return {
        "mean": m,
        "ub_one_sided": m + st.t.ppf(level, K - 1) * se * infl,
        "ci_lo": m - st.t.ppf(0.5 + level / 2, K - 1) * se * infl,
        "ci_hi": m + st.t.ppf(0.5 + level / 2, K - 1) * se * infl,
    }


def effective_K_table(n_neg: int, K: int) -> pd.DataFrame:
    """Sign-test p if the K markets really carried only K_eff independent draws,
    holding the observed proportion of negatives fixed."""
    rows = []
    for Ke in range(K, 5, -1):   # below 6 the rounding of n_neg is not meaningful
        neg = int(round(n_neg / K * Ke))
        rows.append({"K_eff": Ke, "n_negative": neg,
                     "sign_p": st.binomtest(neg, Ke, 0.5).pvalue})
    return pd.DataFrame(rows)


# =============================================================================
# 2. Exact cross-sectional gradient tests
# =============================================================================
def exact_spearman(x, y) -> tuple[float, float]:
    """
    Exact permutation p-value for Spearman's rho.
    Enumerates DISTINCT arrangements of the tied characteristic x (1,260-5,040
    for our ordinals) rather than all 9! = 362,880 permutations.
    """
    rx, ry = st.rankdata(x), st.rankdata(y)
    rho0 = np.corrcoef(rx, ry)[0, 1]
    A = np.array(sorted(set(permutations(rx))), float)
    Ac = A - A.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean()
    rhos = (Ac @ ryc) / np.sqrt((Ac ** 2).sum(axis=1) * (ryc ** 2).sum())
    return float(rho0), float(np.mean(np.abs(rhos) >= abs(rho0) - 1e-12))


def _jt(groups) -> float:
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for a in groups[i]:
                for b in groups[j]:
                    J += (b > a) + 0.5 * (b == a)
    return J


def jonckheere_terpstra(values, ordinal, alternative="increasing"):
    """
    Exact Jonckheere-Terpstra test against an ORDERED alternative -- the correct
    statistic when the regressor is ordinal and the hypothesis is monotone
    ('the gain rises with exposure'), and more powerful than pairwise
    Mann-Whitney comparisons.  Exact by enumerating all group assignments.
    """
    values, ordinal = np.asarray(values, float), np.asarray(ordinal)
    levels = sorted(set(ordinal))
    sizes = [int((ordinal == l).sum()) for l in levels]
    J0 = _jt([values[ordinal == l] for l in levels])
    stats = []

    def rec(remaining, gi, acc):
        if gi == len(sizes) - 1:
            stats.append(_jt([values[list(a)] for a in acc + [tuple(remaining)]]))
            return
        for cmb in combinations(remaining, sizes[gi]):
            rec(tuple(x for x in remaining if x not in cmb), gi + 1, acc + [cmb])

    rec(tuple(range(len(values))), 0, [])
    s = np.array(stats)
    p = np.mean(s >= J0 - 1e-12) if alternative == "increasing" \
        else np.mean(s <= J0 + 1e-12)
    return float(J0), float(p), len(s)


def min_attainable_p(ordinal, K=9) -> float:
    """Smallest two-sided permutation p-value the tie structure permits."""
    prod = 1
    for c in pd.Series(ordinal).value_counts().values:
        prod *= factorial(int(c))
    return 2 * prod / factorial(K)


def power_curve(sd: float, expo: np.ndarray, betas, n_sim=2000, seed=7):
    """
    Minimum detectable gradient.  Simulate y = beta * exposure + N(0, sd),
    apply the exact permutation Spearman test, record rejection rate.
    beta is in percentage points of R2_OS per exposure level.
    """
    rng = np.random.default_rng(seed)
    rx = st.rankdata(expo)
    A = np.array(sorted(set(permutations(rx))), float)
    Ac = A - A.mean(axis=1, keepdims=True)
    rows = []
    for beta in betas:
        rej = 0
        for _ in range(n_sim):
            y = beta * expo + rng.normal(0, sd, len(expo))
            ry = st.rankdata(y)
            ryc = ry - ry.mean()
            rhos = (Ac @ ryc) / np.sqrt((Ac ** 2).sum(axis=1) * (ryc ** 2).sum())
            rho0 = np.corrcoef(rx, ry)[0, 1]
            if np.mean(np.abs(rhos) >= abs(rho0) - 1e-12) < 0.05:
                rej += 1
        rows.append({"beta_pp_per_level": beta, "power": rej / n_sim})
    return pd.DataFrame(rows)


# =============================================================================
# 3. Multiplicity within the market-level family
# =============================================================================
def multiplicity(d: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for _, r in d.iterrows():
        for q in ("q5", "q95"):
            for pref, tag in (("", "vs CTRL"), ("plc_", "vs NOISE")):
                gc, pc = f"pinball_gain_{pref}{q}", f"DM_p_{pref}{q}"
                if gc in r.index and pc in r.index and np.isfinite(r[pc]):
                    rec.append({"code": r["code"], "tau": q, "reference": tag,
                                "gain": r[gc], "p": r[pc]})
    t = pd.DataFrame(rec).sort_values("p").reset_index(drop=True)
    m = len(t)
    t["bonferroni"] = (t["p"] * m).clip(upper=1)
    holm, prev = [], 0.0
    for i, p in enumerate(t["p"]):
        prev = min(max(prev, (m - i) * p), 1.0)
        holm.append(prev)
    t["holm"] = holm
    bh, prev = [], 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, t["p"].iloc[i] * m / (i + 1))
        bh.append(prev)
    t["bh_q"] = bh[::-1]
    return t


# =============================================================================
# 4. Robustness screens
# =============================================================================
def robustness(d: pd.DataFrame) -> pd.DataFrame:
    screens = [
        ("Baseline (all retained markets)", d),
        ("Exclude zero-return share > 5%", d[d.pct_zero_ret < 5]),
        ("Exclude Chinese markets (CEA, GDEA, HBEA)",
         d[~d.code.isin(["CEA", "GDEA", "HBEA"])]),
        ("Exclude N_oos < 150", d[d.n_oos >= 150]),
        ("Exclude NZU_pre24", d[d.code != "NZU_pre24"]),
        ("Exclude quote coverage < 95%", d[d.coverage > 0.95]),
        ("High and medium European exposure only",
         d[d.euro_geo_exposure.isin(["high", "medium"])]),
        ("Low European exposure only (control group)",
         d[d.euro_geo_exposure == "low"]),
    ]
    rows = []
    for lab, sub in screens:
        for col, short, _, _st in OUTCOMES[:3]:
            s = sign_and_rank(sub[col].values)
            rows.append({"screen": lab, "outcome": short, **s})
    return pd.DataFrame(rows)


def jackknife(d: pd.DataFrame, col="pinball_gain_q95") -> pd.DataFrame:
    rows = []
    for c in d.code:
        v = d.loc[d.code != c, col].values
        rows.append({"dropped": c, **sign_and_rank(v)})
    return pd.DataFrame(rows)


# =============================================================================
# 5. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_crossmarket")
    ap.add_argument("--n-sim", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    d = pd.read_csv(out / "X1_market_results.csv")
    K = len(d)

    print("=" * 78)
    print(f"Cross-market inference layer   (K = {K} markets)")
    print("=" * 78)

    # ---- Table 2, Panels A/B/C/D/E -----------------------------------------
    print("\n[Table 2] Panel A/B: distribution and independence-based tests")
    panelA = []
    for col, short, long, status in OUTCOMES:
        if col not in d.columns:
            continue
        s = sign_and_rank(d[col].values)
        panelA.append({"outcome": short, "label": long,
                       "null_status": status, **s})
    pa = pd.DataFrame(panelA)
    print(pa.round(4).to_string(index=False))

    print("\n[Table 2] Panel C: dependence-adjusted Stouffer combination")
    rows = []
    specs = [("q95", "G(0.95)", "pinball_gain_q95", "DM_p_q95"),
             ("q5",  "G(0.05)", "pinball_gain_q5",  "DM_p_q5"),
             ("q95p", "Gp(0.95)", "pinball_gain_plc_q95", "DM_p_plc_q95"),
             ("q5p",  "Gp(0.05)", "pinball_gain_plc_q5",  "DM_p_plc_q5")]
    for _q, short, gcol, pcol in specs:
        if gcol not in d.columns or pcol not in d.columns:
            continue
        for rho in RHO_GRID:
            Z, p = stouffer(d[gcol].values, d[pcol].values, rho)
            rows.append({"outcome": short, "rho_bar": rho, "Z": round(Z, 3),
                         "p_two_sided": round(p, 4)})
    pc = pd.DataFrame(rows)
    print(pc.to_string(index=False))

    print("\n[Table 2] Panel D: sign test at K_eff effective independent markets")
    neg95 = int((d["pinball_gain_q95"] < 0).sum())
    pd_ = effective_K_table(neg95, K)
    print(pd_.round(4).to_string(index=False))

    print("\n[Table 2] Panel E: equivalence bounds on the mean gain")
    rows = []
    for col, short, _, _st in OUTCOMES[:2]:
        for rho in (0.0, 0.2):
            e = equivalence_bound(d[col].values, rho)
            rows.append({"outcome": short, "rho_bar": rho,
                         **{k: round(v, 3) for k, v in e.items()}})
    pe = pd.DataFrame(rows)
    print(pe.to_string(index=False))
    print("  -> read as: any mean cross-market effect above 'ub_one_sided' is")
    print("     inconsistent with these data at the 5% level.")

    pd.concat([pa.assign(panel="A/B"), pc.assign(panel="C"),
               pd_.assign(panel="D"), pe.assign(panel="E")],
              ignore_index=True).to_csv(out / "Y1_aggregation.csv", index=False)

    # ---- Table 3: exact gradient tests -------------------------------------
    print("\n[Table 3] Exact cross-sectional gradient tests")
    rows = []
    for col, short, _, _st in OUTCOMES:
        if col not in d.columns:
            continue
        for xv, lab in CHARS:
            rho, p_ex = exact_spearman(d[xv].values, d[col].values)
            J, p_inc, npm = jonckheere_terpstra(d[col].values, d[xv].values, "increasing")
            _, p_dec, _ = jonckheere_terpstra(d[col].values, d[xv].values, "decreasing")
            rows.append({"outcome": short, "characteristic": lab,
                         "spearman_rho": round(rho, 3),
                         "spearman_p_exact": round(p_ex, 4),
                         "spearman_p_asym": round(st.spearmanr(d[xv], d[col]).pvalue, 3),
                         "JT_J": J, "JT_p_increasing": round(p_inc, 4),
                         "JT_p_decreasing": round(p_dec, 4), "n_arrangements": npm})
    g = pd.DataFrame(rows)
    print(g.to_string(index=False))
    g.to_csv(out / "Y2_gradient_exact.csv", index=False)

    print("\n  Minimum attainable two-sided p given the tie structure:")
    for xv, lab in CHARS:
        print(f"    {lab:32s} {min_attainable_p(d[xv].values, K):.5f}")

    hi = d.loc[d.euro_geo_exposure == "high", "gain_gpr_vs_ctrl"].values
    lo = d.loc[d.euro_geo_exposure == "low", "gain_gpr_vs_ctrl"].values
    u = st.mannwhitneyu(hi, lo, alternative="greater")
    print(f"\n  High vs low European exposure: mean {hi.mean():+.3f} (n={len(hi)}) "
          f"vs {lo.mean():+.3f} (n={len(lo)}); one-sided p = {u.pvalue:.3f}")
    print(f"  !! minimum attainable one-sided p = 1/C({len(hi)+len(lo)},{len(hi)}) "
          f"= {1/comb(len(hi)+len(lo), len(hi)):.3f}: this test CANNOT reject at 5%.")

    y = d["gain_gpr_vs_ctrl"].values
    grp = d["euro_geo_exposure"].values
    gm = y.mean()
    ssb = sum(len(y[grp == l]) * (y[grp == l].mean() - gm) ** 2 for l in set(grp))
    ssw = sum(((y[grp == l] - y[grp == l].mean()) ** 2).sum() for l in set(grp))
    print(f"  eta^2 (exposure explains this share of gain variance) = {ssb/(ssb+ssw):.4f}")

    # ---- Power curve --------------------------------------------------------
    print("\n[Power] Minimum detectable monotone gradient, exact permutation test")
    pw = power_curve(d["gain_gpr_vs_ctrl"].std(ddof=1), d["expo_ord"].values.astype(float),
                     [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], n_sim=args.n_sim)
    print(pw.round(3).to_string(index=False))
    pw.to_csv(out / "Y5_power_curve.csv", index=False)
    print(f"  observed cross-market range of the gain = {y.max()-y.min():.2f} pp; "
          f"largest |gain| = {np.abs(y).max():.2f} pp")

    # ---- Table 4: robustness ------------------------------------------------
    print("\n[Table 4] Robustness screens")
    rb = robustness(d)
    print(rb[rb.outcome == "G(0.95)"].round(4).to_string(index=False))
    print("\n  placebo comparison across the same screens:")
    print(rb[rb.outcome == r"$\Delta^{plc}$"].round(4).to_string(index=False))
    rb.to_csv(out / "Y3_robustness.csv", index=False)

    print("\n  Leave-one-market-out, G(0.95):")
    print(jackknife(d).round(4).to_string(index=False))

    # ---- Appendix A4 Panel C ------------------------------------------------
    print("\n[Table A4 Panel C] Market-level tests, multiplicity-adjusted")
    mt = multiplicity(d)
    print(mt.round(4).to_string(index=False))
    mt.to_csv(out / "Y4_market_multiplicity.csv", index=False)
    sig = mt[mt.holm < 0.05]
    if len(sig):
        print("\n  ** survives Holm within the 18-test family:")
        for _, r in sig.iterrows():
            print(f"     {r.code} tau={r.tau} gain={r.gain:+.2f}% "
                  f"p={r.p:.4f} Holm={r.holm:.3f} BH q={r.bh_q:.3f}")
        print("     Report this explicitly in the paper rather than leaving it")
        print("     for a referee: note the tail, the market, and the placebo.")

    # ---- Microstructure correlations ---------------------------------------
    print("\n[A4 Panel B] Microstructure vs level and vs gain (Spearman)")
    for c in ("ac1", "pct_zero_ret", "coverage", "pct_gap_gt10d"):
        r1, p1 = st.spearmanr(d[c], d["r2_CTRL"])
        r2, p2 = st.spearmanr(d[c], d["gain_gpr_vs_ctrl"])
        print(f"  {c:14s} vs R2 level: rho={r1:+.3f} (p={p1:.3f})   "
              f"vs gain: rho={r2:+.3f} (p={p2:.3f})")
    print("  -> microstructure explains LEVELS; the gain differences it out.")
    print("     Note the sign on gap>10d: if it biases the gain at all it biases")
    print("     it UPWARD, so the reported null is conservative.")

    print(f"\n[done] wrote Y1-Y5 to {out.resolve()}")


if __name__ == "__main__":
    main()
