"""
=============================================================================
frequency_figures.py - Headline figures: the monthly vs. weekly contrast
=============================================================================
This module produces the paper's central visual evidence. The same code, the
same data sources, the same estimators and the same tests are applied at two
sampling frequencies; only the frequency changes, and the sign of the
quantile-loss improvement reverses.

  Monthly (N=138, out-of-sample 78, common sample 75):
      tau=0.95 improvement +3.22%, DM p=0.020
  Weekly  (N=604, out-of-sample 404, common sample 386):
      tau=0.95 improvement -2.29%, DM p=0.026

Both are nominally significant with opposite signs, and neither survives a
Bonferroni correction across the seven quantile levels tested. The defensible
reading is that there is no robust evidence of a quantile-loss improvement at
either frequency, and that the sign of the point estimate itself flips with
sample size - which is what small-sample fragility means.

NOTE ON FIGURE TITLES
---------------------
`oos_evaluation.make_tail_figure` in this repository carries a neutral title.
An earlier version of that routine used the title "Geopolitical risk improves
the tails, not the conditional mean", which the weekly results contradict;
that claim is not reproduced anywhere in this repository.

INPUTS
------
Written by `main_pipeline.py` into the weekly and monthly output directories:
  T8b_tail_inference.csv   (required) columns: quantile, pct_improvement,
                                               DM_p_two_sided
  T1_accuracy_metrics.csv  (optional) columns: model, R2_OS_vs_histmean_pct

OUTPUTS
-------
  F1_frequency_contrast.{pdf,png}   headline figure
  F2_placebo_weekly.{pdf,png}       placebo comparison, weekly
  F2_placebo_monthly.{pdf,png}      placebo comparison, monthly
  F3_r2_distribution.{pdf,png}      R2_OS by model family
  T_frequency_contrast.csv          summary table quoted in the paper

USAGE
-----
    python frequency_figures.py --weekly out_v2 --monthly out_monthly \
                                --out figures
=============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.bbox": "tight",
})

C_W, C_M, C_NEG, C_POS = "#c0392b", "#2c7fb8", "#c0392b", "#27ae60"
N_TESTS = 7  # Bonferroni denominator: seven quantile levels tested jointly


def _load(d: Path, name: str) -> pd.DataFrame | None:
    p = Path(d) / name
    return pd.read_csv(p) if p.exists() else None


# =============================================================================
# Figure 1 (headline): quantile-loss improvement, monthly vs. weekly
# =============================================================================
def fig_frequency_contrast(mo: pd.DataFrame, wk: pd.DataFrame, out: Path,
                           n_mo: int, n_wk: int) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    # --- (a) improvement curves -------------------------------------------
    a1.axhline(0, color="black", lw=1)
    a1.plot(mo["quantile"], mo["pct_improvement"], "o-", color=C_M, lw=2,
            label=f"Monthly ($N_{{oos}}={n_mo}$)")
    a1.plot(wk["quantile"], wk["pct_improvement"], "s-", color=C_W, lw=2,
            label=f"Weekly ($N_{{oos}}={n_wk}$)")
    for d, col, dy in ((mo, C_M, 12), (wk, C_W, -16)):
        r = d[d["quantile"] == 0.95]
        if len(r):
            v = float(r["pct_improvement"].iloc[0])
            a1.annotate(f"{v:+.2f}%", (0.95, v), textcoords="offset points",
                        xytext=(-8, dy), color=col, fontweight="bold")
    a1.set_xlabel(r"Quantile level $\tau$")
    a1.set_ylabel("Pinball loss reduction from adding GPR (%)")
    a1.set_title("(a) The same design at two frequencies")
    a1.legend(loc="lower left")

    # --- (b) nominal vs. adjusted p-values at tau=0.95 --------------------
    labs, raws, adjs, vals = [], [], [], []
    for nm, d in (("Monthly", mo), ("Weekly", wk)):
        r = d[d["quantile"] == 0.95]
        if not len(r):
            continue
        labs.append(nm)
        p = float(r["DM_p_two_sided"].iloc[0])
        raws.append(p)
        adjs.append(min(p * N_TESTS, 1.0))
        vals.append(float(r["pct_improvement"].iloc[0]))

    x = np.arange(len(labs))
    w = 0.36
    a2.bar(x - w / 2, raws, w, label="Nominal $p$", color="#95a5a6")
    a2.bar(x + w / 2, adjs, w, label="Bonferroni-adjusted", color="#34495e")
    a2.axhline(0.05, color=C_NEG, ls="--", lw=1.2, label=r"$\alpha=0.05$")
    for i, (r_, a_, v_) in enumerate(zip(raws, adjs, vals)):
        a2.text(i - w / 2, r_ + .01, f"{r_:.3f}", ha="center", fontsize=8)
        a2.text(i + w / 2, a_ + .01, f"{a_:.2f}", ha="center", fontsize=8)
        a2.text(i, max(r_, a_) + .10, f"$\\Delta$={v_:+.2f}%", ha="center",
                fontsize=9, fontweight="bold",
                color=C_POS if v_ > 0 else C_NEG)
    a2.set_xticks(x)
    a2.set_xticklabels(labs)
    a2.set_ylabel("$p$-value")
    a2.set_ylim(0, 1.05)
    a2.set_title(r"(b) Inference at $\tau=0.95$")
    a2.legend(loc="upper left")

    fig.suptitle("An effect that disappears: quantile-loss gains from "
                 "geopolitical risk reverse sign with sample size", y=1.03,
                 fontsize=11.5)
    fig.savefig(out / "F1_frequency_contrast.png")
    fig.savefig(out / "F1_frequency_contrast.pdf")
    plt.close(fig)
    msg = ", ".join(f"{l} {v:+.2f}%" for l, v in zip(labs, vals))
    print(f"  [ok] F1_frequency_contrast  (tau=0.95: {msg})")


# =============================================================================
# Figure 2: dimension-matched placebo test
# =============================================================================
def fig_placebo(tab: pd.DataFrame, out: Path, freq_label: str) -> None:
    """tab: columns = estimator, ctrl, placebo, gpr (all in percent)."""
    x = np.arange(len(tab))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(x - w, tab["ctrl"], w, label="Controls only", color="#95a5a6")
    ax.bar(x, tab["placebo"], w, label="Controls + noise block",
           color="#34495e")
    ax.bar(x + w, tab["gpr"], w, label="Controls + GPR block", color=C_M)
    ax.axhline(0, color="black", lw=1)
    for i, r in enumerate(tab.itertuples()):
        ok = r.gpr > r.placebo
        ax.text(i, min(r.ctrl, r.placebo, r.gpr) - .55,
                "GPR > noise" if ok else "GPR $\\leq$ noise",
                ha="center", fontsize=8.5, fontweight="bold",
                color=C_POS if ok else C_NEG)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["estimator"])
    ax.set_ylabel("Out-of-sample $R^2_{OS}$ vs. prevailing mean (%)")
    ax.set_title(f"Dimension-matched placebo test ({freq_label})\n"
                 "The GPR block is compared against an equally sized block of "
                 "AR(1) noise")
    ax.legend(loc="lower right")
    fig.savefig(out / f"F2_placebo_{freq_label.lower()}.png")
    fig.savefig(out / f"F2_placebo_{freq_label.lower()}.pdf")
    plt.close(fig)
    print(f"  [ok] F2_placebo_{freq_label.lower()}")


# =============================================================================
# Figure 3: distribution of R2_OS by model family
# =============================================================================
def fig_r2_distribution(t1_w: pd.DataFrame, t1_m: pd.DataFrame, out: Path,
                        n_mo: int, n_wk: int, seed: int = 20260101) -> None:
    rng = np.random.default_rng(seed)

    def fam(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
        d = d.dropna(subset=["R2_OS_vs_histmean_pct"])
        return {
            "Benchmarks": d[d.model.str.startswith("B_")],
            "Controls only": d[d.model.str.endswith("_CTRL")],
            "+ noise block": d[d.model.str.contains("PLACEBO")],
            "+ GPR block": d[d.model.str.contains("GPR")],
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (d, lab, n) in zip(axes,
                               [(t1_m, "Monthly", n_mo), (t1_w, "Weekly", n_wk)]):
        groups = fam(d)
        for i, (k, g) in enumerate(groups.items()):
            v = g["R2_OS_vs_histmean_pct"].values
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), i) + rng.uniform(-.10, .10, len(v)),
                       v, s=26, alpha=.75,
                       color=C_M if "GPR" in k else "#7f8c8d")
            ax.hlines(np.median(v), i - .25, i + .25, color="black", lw=2)
        ax.axhline(0, color=C_NEG, ls="--", lw=1.2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(list(groups), rotation=15, ha="right")
        ax.set_title(f"{lab}  ($N_{{oos}}={n}$)")
    axes[0].set_ylabel("Out-of-sample $R^2_{OS}$ (%)")
    fig.suptitle("Out-of-sample $R^2_{OS}$ by model family. "
                 "Dashed line = prevailing-mean benchmark; bars = group median",
                 y=1.02, fontsize=11)
    fig.savefig(out / "F3_r2_distribution.png")
    fig.savefig(out / "F3_r2_distribution.pdf")
    plt.close(fig)
    print("  [ok] F3_r2_distribution")


# =============================================================================
# Summary table quoted directly in the paper
# =============================================================================
def summary_table(mo: pd.DataFrame, wk: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for q in sorted(set(mo["quantile"]) & set(wk["quantile"])):
        m = mo[mo["quantile"] == q].iloc[0]
        w = wk[wk["quantile"] == q].iloc[0]
        rows.append({
            "tau": q,
            "monthly_delta_pct": round(float(m["pct_improvement"]), 2),
            "monthly_DM_p": round(float(m["DM_p_two_sided"]), 4),
            "monthly_p_bonf": round(min(float(m["DM_p_two_sided"]) * N_TESTS, 1), 3),
            "weekly_delta_pct": round(float(w["pct_improvement"]), 2),
            "weekly_DM_p": round(float(w["DM_p_two_sided"]), 4),
            "weekly_p_bonf": round(min(float(w["DM_p_two_sided"]) * N_TESTS, 1), 3),
            "sign_flip": bool(np.sign(m["pct_improvement"])
                              != np.sign(w["pct_improvement"])),
        })
    t = pd.DataFrame(rows)
    t.to_csv(out / "T_frequency_contrast.csv", index=False)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the monthly-vs-weekly contrast figures.")
    ap.add_argument("--weekly", default="out_v2",
                    help="directory holding the weekly pipeline output")
    ap.add_argument("--monthly", default="out_monthly",
                    help="directory holding the monthly pipeline output")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--n-weekly", type=int, default=386,
                    help="weekly common out-of-sample size, for axis labels")
    ap.add_argument("--n-monthly", type=int, default=75,
                    help="monthly common out-of-sample size, for axis labels")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wk = _load(Path(args.weekly), "T8b_tail_inference.csv")
    mo = _load(Path(args.monthly), "T8b_tail_inference.csv")
    if wk is None or mo is None:
        raise SystemExit(
            "T8b_tail_inference.csv is missing.\n"
            f"  weekly directory  {args.weekly} -> "
            f"{'found' if wk is not None else 'NOT FOUND'}\n"
            f"  monthly directory {args.monthly} -> "
            f"{'found' if mo is not None else 'NOT FOUND'}\n"
            "Run main_pipeline.py at both frequencies first.")

    t1w = _load(Path(args.weekly), "T1_accuracy_metrics.csv")
    t1m = _load(Path(args.monthly), "T1_accuracy_metrics.csv")

    print("[building figures]")
    fig_frequency_contrast(mo, wk, out, n_mo=args.n_monthly, n_wk=args.n_weekly)

    if t1w is not None and t1m is not None:
        fig_r2_distribution(t1w, t1m, out,
                            n_mo=args.n_monthly, n_wk=args.n_weekly)
        # Read the placebo triplet straight out of T1 so no number is
        # transcribed by hand.
        for lab, t1 in (("Weekly", t1w), ("Monthly", t1m)):
            s = t1.set_index("model")["R2_OS_vs_histmean_pct"]
            rows = []
            for est in ("Ridge", "XGB", "ENet"):
                k = (f"{est}_CTRL", f"{est}_CTRL_PLACEBO", f"{est}_CTRL_GPRfull")
                if all(x in s.index for x in k):
                    rows.append({
                        "estimator": {"XGB": "Gradient boosting"}.get(est, est),
                        "ctrl": s[k[0]], "placebo": s[k[1]], "gpr": s[k[2]],
                    })
            if rows:
                fig_placebo(pd.DataFrame(rows), out, lab)
    else:
        print("  [skip] F2/F3 need T1_accuracy_metrics.csv in both directories")

    t = summary_table(mo, wk, out)
    print("\n[frequency contrast table]")
    print(t.to_string(index=False))
    n_flip = int(t["sign_flip"].sum())
    print(f"\n  {n_flip}/{len(t)} quantile levels change sign between the two "
          f"frequencies")
    print(f"\n[done] {out.resolve()}")


if __name__ == "__main__":
    main()
