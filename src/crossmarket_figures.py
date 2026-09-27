"""
crossmarket_figures.py -- Figures A4 and A5 for the cross-market module.

    python src/crossmarket_figures.py --xdir out_crossmarket --out figures

FA4_crossmarket_gains.pdf   Panel (a) per-market gains vs controls and vs the
                            dimension-matched noise block, ordered by European
                            geopolitical exposure; Panel (b) upper-tail gain by
                            market with the single-market weekly EU value marked.
FA5_dependence_power.pdf    Panel (a) Stouffer p for G(0.95) as a function of the
                            assumed equicorrelation; Panel (b) power of the exact
                            permutation gradient test against the observed range.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

EXPO_ORDER = {"low": 0, "medium": 1, "high": 2}
WEEKLY_EU_Q95 = -2.29          # single-market weekly EU result, Section 5.4
RHO_GRID = np.linspace(0, 0.6, 61)
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
POWER = [0.046, 0.080, 0.209, 0.412, 0.626, 0.795, 0.906]   # 2,000 replications


def stouffer_p(gains, pvals, rho):
    z = np.sign(gains) * st.norm.ppf(1 - np.asarray(pvals) / 2)
    K = len(z)
    Z = z.sum() / np.sqrt(K + K * (K - 1) * rho)
    return 2 * (1 - st.norm.cdf(abs(Z)))


def fig_a4(d: pd.DataFrame, out: Path):
    d = d.copy()
    d["eo"] = d["euro_geo_exposure"].map(EXPO_ORDER)
    d = d.sort_values(["eo", "gain_gpr_vs_ctrl"]).reset_index(drop=True)
    lab = d["code"].str.replace("_pre24", r"$_{\rm pre24}$", regex=False)
    x = np.arange(len(d))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

    w = 0.38
    ax[0].bar(x - w / 2, d["gain_gpr_vs_ctrl"], w, label=r"vs controls",
              color="0.35", edgecolor="k", linewidth=0.4)
    ax[0].bar(x + w / 2, d["gain_gpr_vs_placebo"], w,
              label=r"vs dimension-matched noise", color="0.75",
              edgecolor="k", linewidth=0.4)
    ax[0].axhline(0, color="k", lw=0.8)
    for b in np.flatnonzero(np.diff(d["eo"].values)) + 0.5:
        ax[0].axvline(b, color="0.5", ls=":", lw=0.9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel(r"$R^2_{\rm OS}$ gain (pp)")
    ax[0].set_title("(a) Gains by market, ordered by European exposure", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].text(0.02, 0.03, "low          medium        high", transform=ax[0].transAxes,
               fontsize=7.5, color="0.4")

    d2 = d.sort_values("pinball_gain_q95").reset_index(drop=True)
    lab2 = d2["code"].str.replace("_pre24", r"$_{\rm pre24}$", regex=False)
    x2 = np.arange(len(d2))
    ax[1].barh(x2, d2["pinball_gain_q95"], color="0.45", edgecolor="k", linewidth=0.4)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].axvline(WEEKLY_EU_Q95, color="crimson", ls="--", lw=1.2)
    ax[1].text(WEEKLY_EU_Q95, len(d2) - 0.3, " weekly EU\n single-market",
               color="crimson", fontsize=7.5, va="top", ha="right")
    m = d2["pinball_gain_q95"].mean()
    ax[1].axvline(m, color="navy", ls=":", lw=1.2)
    ax[1].text(m, len(d2) - 0.3, f"mean {m:.2f}% ", color="navy",
               fontsize=7.5, ha="left", va="top")
    ax[1].set_yticks(x2); ax[1].set_yticklabels(lab2, fontsize=8)
    ax[1].set_xlabel(r"pinball-loss improvement at $\tau=0.95$ (\%)"
                     if plt.rcParams["text.usetex"] else
                     "pinball-loss improvement at tau = 0.95 (%)")
    ax[1].set_title("(b) Upper-tail gain: the dimensionality penalty", fontsize=10)

    fig.tight_layout()
    fig.savefig(out / "FA4_crossmarket_gains.pdf")
    plt.close(fig)


def fig_a5(d: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    ps = [stouffer_p(d["pinball_gain_q95"].values, d["DM_p_q95"].values, r)
          for r in RHO_GRID]
    ax[0].plot(RHO_GRID, ps, color="0.2", lw=1.6)
    ax[0].axhline(0.05, color="crimson", ls="--", lw=1.0)
    ax[0].text(0.30, 0.058, "5% level", color="crimson", fontsize=8)
    cross = RHO_GRID[np.argmax(np.array(ps) > 0.05)]
    ax[0].axvline(cross, color="0.5", ls=":", lw=1.0)
    ax[0].text(cross + 0.01, 0.30, rf"evidence expires at $\bar\rho={cross:.2f}$",
               fontsize=8, color="0.3")
    ax[0].set_xlabel(r"assumed equicorrelation $\bar\rho$ across markets")
    ax[0].set_ylabel(r"Stouffer two-sided $p$ for $G(0.95)$")
    ax[0].set_ylim(0, 0.45)
    ax[0].set_title("(a) The upper-tail pattern is not dependence-robust", fontsize=10)

    ax[1].plot(BETAS, POWER, "o-", color="0.2", lw=1.6, ms=4)
    ax[1].axhline(0.8, color="crimson", ls="--", lw=1.0)
    ax[1].text(0.05, 0.82, "80% power", color="crimson", fontsize=8)
    rng = d["gain_gpr_vs_ctrl"].max() - d["gain_gpr_vs_ctrl"].min()
    ax[1].axvspan(0, rng / 2, color="0.85", zorder=0)
    ax[1].text(rng / 4, 0.45, "gradients consistent\nwith the observed\ncross-market range",
               fontsize=7.5, ha="center", color="0.35")
    ax[1].set_xlabel(r"true gradient $\beta$ (pp of $R^2_{\rm OS}$ per exposure level)")
    ax[1].set_ylabel("rejection rate")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) The gradient test has no power where it matters", fontsize=10)

    fig.tight_layout()
    fig.savefig(out / "FA5_dependence_power.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdir", default="out_crossmarket")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.xdir) / "X1_market_results.csv")
    fig_a4(d, out)
    fig_a5(d, out)
    print(f"wrote FA4_crossmarket_gains.pdf and FA5_dependence_power.pdf to {out.resolve()}")


if __name__ == "__main__":
    main()
