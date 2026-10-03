"""
=============================================================================
placebo_randomization.py -- turn the single-draw placebo into a p-value
=============================================================================
WHY THIS MODULE EXISTS
----------------------
`main_pipeline.py` draws one block of AR(1) noise and compares the GPR block
against it. One draw is a comparison, not a test: it has no reference
distribution, so it cannot produce a p-value, and the comparison inherits
whatever that particular draw happened to do.

How much that matters is measurable, and it is not small. Holding the panel,
the seed and every other setting fixed and changing only the size of the noise
block from six series to nine moved the placebo arm's out-of-sample R2 by up to
3.7 percentage points at the monthly frequency and 1.4 points weekly, in
inconsistent directions across estimators. With 75 monthly out-of-sample
observations, a single draw is simply not an adequate reference for a
nine-variable block.

This module fixes that the same way `supply_interaction.py` already does: draw
D placebo blocks, treat their R2 values as the null distribution, and report

    p_rand = (1 + #{d : R2(PLACEBO_d) >= R2(GPR)}) / (1 + D)

which is a valid one-sided randomisation p-value for the null that the GPR
block carries no more information than an equal-cardinality noise block. The
+1 in both numerator and denominator is the standard correction that keeps the
test exact for finite D (Phipson and Smyth 2010).

HOW TO USE IT
-------------
Run the pipeline once per seed into its own output directory, then aggregate.
Only the *_PLACEBO rows differ between runs, so each run contributes exactly
one draw while reproducing every other number identically -- which this module
verifies rather than assumes.

    # monthly: about 20 s per draw
    for s in $(seq 1 50); do
      python src/main_pipeline.py --panel data/panel_monthly_v2.csv \
        --out draws_monthly/seed$s --freq M --placebo-n 9 \
        --placebo-seed $s --fast
    done
    python src/placebo_randomization.py --dirs "draws_monthly/seed*" \
        --label monthly --out results/placebo_randomization

    # weekly: about 3 min per draw, so run fewer or run it overnight
    for s in $(seq 1 20); do
      python src/main_pipeline.py --panel data/panel_weekly_v2.csv \
        --out draws_weekly/seed$s --placebo-n 9 --placebo-seed $s --fast
    done
    python src/placebo_randomization.py --dirs "draws_weekly/seed*" \
        --label weekly --out results/placebo_randomization

Quote the glob so the shell does not expand it, or pass the directories
individually. In zsh a bare `#` is not a comment, so do not paste trailing
comments into the terminal.

INPUTS
------
One `T1_accuracy_metrics.csv` per draw directory, written by
`main_pipeline.py`. Columns used: `model`, `R2_OS_vs_histmean_pct`.

OUTPUTS
-------
  P1_placebo_randomization_<label>.csv   one row per estimator: the null
                                         summary and the randomisation p-value
  P2_placebo_draws_<label>.csv           the null distribution, one row per
                                         draw per estimator

INTEGRITY CHECK
---------------
Before computing anything the module verifies that every non-placebo row is
identical across draw directories. If it is not, the runs differ by more than
the placebo seed -- a changed panel, a changed flag, a different library
version -- and the draws are not exchangeable, so the p-value would be
meaningless. The module says exactly which models disagree and exits.
=============================================================================
"""

from __future__ import annotations

import argparse
import glob as globmod
from pathlib import Path

import numpy as np
import pandas as pd

ESTIMATORS = ("Ridge", "ENet", "PCR", "XGB")
METRIC = "R2_OS_vs_histmean_pct"
#: rows allowed to differ between draws
PLACEBO_SUFFIX = "_CTRL_PLACEBO"
#: tolerance for the integrity check, in percentage points of R2_OS
TOL = 1e-9


# =============================================================================
# Loading
# =============================================================================
def resolve_dirs(patterns: list[str]) -> list[Path]:
    """Expand globs and plain paths into a sorted, de-duplicated list."""
    out: list[Path] = []
    for p in patterns:
        hits = [Path(h) for h in sorted(globmod.glob(p))]
        out.extend(hits if hits else [Path(p)])
    seen, uniq = set(), []
    for d in out:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(d)
    return uniq


def load_draws(dirs: list[Path]) -> dict[str, pd.Series]:
    """Return {dir_name: Series indexed by model, holding R2_OS}.

    Directories without a readable T1 are collected and reported together
    rather than aborting on the first one. A batch of draws launched from a
    shell loop fails as a batch -- if the pipeline raised, every directory was
    created and every one is empty -- and a report naming one of fifty would
    hide that.
    """
    draws: dict[str, pd.Series] = {}
    absent: list[Path] = []
    broken: list[str] = []

    for d in dirs:
        f = d / "T1_accuracy_metrics.csv"
        if not f.exists():
            absent.append(d)
            continue
        try:
            t = pd.read_csv(f)
        except Exception as exc:
            broken.append(f"{f}: unreadable ({type(exc).__name__})")
            continue
        miss = {"model", METRIC} - set(t.columns)
        if miss:
            broken.append(f"{f}: missing column(s) {sorted(miss)}")
            continue
        if t.empty:
            broken.append(f"{f}: no rows")
            continue
        draws[d.name] = t.set_index("model")[METRIC]

    if absent or broken:
        n_bad = len(absent) + len(broken)
        lines = [f"\n  [warn] {n_bad} of {len(dirs)} draw directories have no "
                 f"usable T1_accuracy_metrics.csv"]
        if absent:
            shown = ", ".join(d.name for d in absent[:6])
            more = f" (+{len(absent) - 6} more)" if len(absent) > 6 else ""
            lines.append(f"         missing the file: {shown}{more}")
        for b in broken[:6]:
            lines.append(f"         {b}")
        print("\n".join(lines))

        if not draws:
            raise SystemExit(
                "\n[stop] not one draw directory holds a usable "
                "T1_accuracy_metrics.csv.\n"
                "  The directories exist but the runs that were supposed to "
                "fill them did not\n"
                "  finish. Scroll back through the loop's output for the "
                "traceback, fix the\n"
                "  cause, delete the empty directories and run the loop "
                "again. A shell `for`\n"
                "  loop keeps going after a crash, so a batch can leave "
                "behind a full set of\n"
                "  empty directories and look as though it worked.")

    if not draws:
        raise SystemExit("\n[stop] no draws could be loaded.")
    return draws


# =============================================================================
# Integrity check: the draws must differ by the placebo seed and nothing else
# =============================================================================
def check_exchangeable(draws: dict[str, pd.Series]) -> None:
    names = list(draws)
    ref_name = names[0]
    ref = draws[ref_name]
    problems: list[str] = []

    for nm in names[1:]:
        s = draws[nm]
        only_ref = sorted(set(ref.index) - set(s.index))
        only_s = sorted(set(s.index) - set(ref.index))
        if only_ref or only_s:
            problems.append(
                f"    {nm}: model list differs from {ref_name}"
                + (f"; missing {only_ref}" if only_ref else "")
                + (f"; extra {only_s}" if only_s else ""))
            continue
        common = [m for m in ref.index if not m.endswith(PLACEBO_SUFFIX)]
        a, b = ref[common], s[common]
        diff = (a - b).abs()
        bad = diff[diff > TOL]
        if len(bad):
            worst = bad.sort_values(ascending=False).head(5)
            detail = ", ".join(f"{m} ({v:.4g}pp)" for m, v in worst.items())
            problems.append(f"    {nm}: {len(bad)} non-placebo row(s) differ "
                            f"from {ref_name}: {detail}")

    if problems:
        raise SystemExit(
            "\n[stop] the draw directories differ by more than the placebo "
            "seed:\n"
            + "\n".join(problems)
            + "\n\n  Only the *_CTRL_PLACEBO rows may change between draws. "
              "Anything else means\n"
              "  the runs used a different panel, a different flag, or a "
              "different library\n"
              "  version, so the draws are not exchangeable and a "
              "randomisation p-value\n"
              "  computed from them would be meaningless. Re-run the draws "
              "changing only\n"
              "  --placebo-seed and --out.")

    print(f"  [ok] {len(names)} draws agree on every non-placebo row "
          f"(tolerance {TOL:g}pp)")


# =============================================================================
# The test
# =============================================================================
def randomization_table(draws: dict[str, pd.Series]
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = draws[list(draws)[0]]
    D = len(draws)
    rows, long = [], []

    for est in ESTIMATORS:
        k_plc, k_gpr, k_ctrl = (f"{est}{PLACEBO_SUFFIX}",
                                f"{est}_CTRL_GPRfull", f"{est}_CTRL")
        if k_gpr not in ref.index or k_plc not in ref.index:
            print(f"  [skip] {est}: {k_gpr} or {k_plc} absent from T1")
            continue

        r_gpr = float(ref[k_gpr])
        r_ctrl = float(ref[k_ctrl]) if k_ctrl in ref.index else np.nan
        null = np.array([float(s[k_plc]) for s in draws.values()], dtype=float)
        null = null[np.isfinite(null)]
        if len(null) == 0:
            print(f"  [skip] {est}: every placebo draw is non-finite")
            continue

        n_ge = int((null >= r_gpr).sum())
        p_rand = (1 + n_ge) / (1 + len(null))
        rows.append({
            "estimator": est,
            "n_draws": len(null),
            "r2_CTRL": round(r_ctrl, 4),
            "r2_GPRfull": round(r_gpr, 4),
            "placebo_mean": round(float(null.mean()), 4),
            "placebo_sd": round(float(null.std(ddof=1)), 4) if len(null) > 1 else np.nan,
            "placebo_min": round(float(null.min()), 4),
            "placebo_p05": round(float(np.percentile(null, 5)), 4),
            "placebo_p95": round(float(np.percentile(null, 95)), 4),
            "placebo_max": round(float(null.max()), 4),
            "placebo_range_pp": round(float(null.max() - null.min()), 4),
            "n_draws_beating_gpr": n_ge,
            "gpr_percentile_in_null": round(
                float((null < r_gpr).mean() * 100), 1),
            "randomization_p": round(p_rand, 4),
        })
        for nm, s in draws.items():
            long.append({"estimator": est, "draw": nm,
                         "r2_placebo": float(s[k_plc]), "r2_GPRfull": r_gpr})

    if not rows:
        raise SystemExit("\n[stop] no estimator had both a placebo and a "
                         "GPRfull row; nothing to test.")

    tab = pd.DataFrame(rows)
    return add_multiplicity(tab), pd.DataFrame(long)


def add_multiplicity(tab: pd.DataFrame) -> pd.DataFrame:
    """Adjust the per-estimator p-values for testing several estimators.

    The randomisation p-value is exact for one estimator. The table runs the
    same null against every estimator that produced both arms, so the smallest
    of them is a minimum over a family and is not an exact 5% test. Reporting
    only the raw column invites exactly the misreading this module exists to
    prevent: one estimator out of four landing under 0.05 is close to what the
    family produces under the null.

    Bonferroni, Holm (step-down, FWER) and Benjamini-Hochberg (FDR) are all
    reported because they answer different questions, and because quoting the
    one that happens to be smallest is the error. Holm is the one to read for a
    claim about any single estimator.
    """
    p = tab["randomization_p"].to_numpy(dtype=float)
    m = len(p)
    if m < 2:
        tab["bonferroni"] = np.minimum(p, 1.0)
        tab["holm"] = np.minimum(p, 1.0)
        tab["bh_q"] = np.minimum(p, 1.0)
        return tab

    order = np.argsort(p, kind="stable")
    ps = p[order]

    bonf = np.minimum(p * m, 1.0)

    holm_sorted = np.minimum(
        np.maximum.accumulate(ps * (m - np.arange(m))), 1.0)
    holm = np.empty(m)
    holm[order] = holm_sorted

    bh_sorted = np.minimum(
        np.minimum.accumulate((ps * m / (np.arange(m) + 1))[::-1])[::-1], 1.0)
    bh = np.empty(m)
    bh[order] = bh_sorted

    tab["bonferroni"] = np.round(bonf, 4)
    tab["holm"] = np.round(holm, 4)
    tab["bh_q"] = np.round(bh, 4)
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate several placebo draws into a randomisation test.")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="draw directories, or a quoted glob such as "
                         "'draws_monthly/seed*'")
    ap.add_argument("--label", default="run",
                    help="suffix for the output filenames, e.g. weekly / monthly")
    ap.add_argument("--out", default="results/placebo_randomization",
                    help="output directory")
    ap.add_argument("--min-draws", type=int, default=10,
                    help="warn below this many draws (default 10). With D draws "
                         "the smallest attainable p-value is 1/(1+D).")
    args = ap.parse_args()

    dirs = resolve_dirs(args.dirs)
    if len(dirs) < 2:
        found = "\n".join(f"    {d}  ({'exists' if d.exists() else 'DOES NOT EXIST'})"
                          for d in dirs) or "    (nothing)"
        raise SystemExit(
            f"\n[stop] need at least 2 draw directories, found {len(dirs)}:\n"
            f"{found}\n\n"
            f"  Two things to check.\n"
            f"  1. Have the draws been run yet? Each one is a separate "
            f"main_pipeline.py run\n"
            f"     with its own --placebo-seed and its own --out directory. "
            f"Run the loop first,\n"
            f"     then aggregate.\n"
            f"  2. Quote the glob, as --dirs 'draws_monthly/seed*', so the "
            f"shell does not\n"
            f"     expand it before the module sees it. Or list the "
            f"directories individually.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"Placebo randomisation test -- {args.label}")
    print("=" * 78)
    print(f"  {len(dirs)} draw directories")

    draws = load_draws(dirs)
    check_exchangeable(draws)

    D = len(draws)
    p_min = 1 / (1 + D)
    print(f"  smallest attainable p-value with D={D}: {p_min:.4f}")
    if D < args.min_draws:
        print(f"  WARNING: only {D} draws. The test cannot reject below "
              f"{p_min:.3f}, so a\n"
              f"           non-rejection here is uninformative rather than "
              f"evidence for the null.")

    tab, long = randomization_table(draws)

    f1 = out / f"P1_placebo_randomization_{args.label}.csv"
    f2 = out / f"P2_placebo_draws_{args.label}.csv"
    tab.to_csv(f1, index=False)
    long.to_csv(f2, index=False)

    print(f"\n[{args.label}] randomisation test, R2_OS in percentage points")
    print(tab.to_string(index=False))

    print("\n  Reading this table:")
    print("    randomization_p is one-sided for H0: the GPR block carries no")
    print("    more information than an equal-cardinality noise block. A large")
    print("    p means the GPR block is NOT distinguishable from noise.")
    print("    Read `holm`, not `randomization_p`, for a claim about any one")
    print(f"    estimator: the table tests {len(tab)} of them against the same null.")
    wide = tab["placebo_range_pp"].max()
    print(f"    The widest placebo null spans {wide:.2f}pp across draws. Any")
    print("    single-draw comparison inherits that much arbitrary variation.")

    # --- saturation: a p-value pinned at the floor is censored, not precise ---
    # The column is rounded to four decimals, so compare on the integer count
    # of draws that matched or beat the GPR arm rather than on the float.
    at_floor = tab[tab["n_draws_beating_gpr"] == 0]
    if len(at_floor):
        names = ", ".join(at_floor["estimator"])
        need = max(1, int(np.ceil(1 / 0.01)) - 1)
        print(f"\n  NOTE: {names} sits exactly at the attainable floor "
              f"{p_min:.4f} ({1}/{1 + D}),")
        print(f"        because no draw matched or beat the GPR arm. The "
              f"p-value is CENSORED there:")
        print(f"        with D={D} the test cannot distinguish {p_min:.4f} from "
              f"an arbitrarily small")
        print(f"        value, so the figure must not be quoted as though it "
              f"were precise. To")
        print(f"        resolve below 0.01 the test needs D >= {need} draws.")

    sig = tab[tab["holm"] < 0.05]
    if len(sig):
        print(f"\n  After Holm adjustment across {len(tab)} estimators, "
              f"{len(sig)} still rejects at 5%:")
        print(f"        {', '.join(sig['estimator'])}")
    else:
        print(f"\n  After Holm adjustment across {len(tab)} estimators nothing "
              f"rejects at 5%")
        print(f"        (smallest adjusted p = {tab['holm'].min():.4f}). The "
              f"GPR block is not")
        print(f"        distinguishable from an equal-cardinality noise block "
              f"for any estimator.")

    print(f"\n[done] {f1}\n       {f2}")


if __name__ == "__main__":
    main()
