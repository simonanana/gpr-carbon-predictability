"""
fill_latex_macros.py -- write supply-side results straight into the LaTeX macro block.

Avoids transcribing 35 numbers by hand. After running supply_interaction.py:

    python src/fill_latex_macros.py --supply out_supply --tex paper/manuscript.tex

The script edits only the text between the "NUMBERS TO FILL IN" and
"END NUMBERS BLOCK" sentinels in the .tex file and leaves every other byte
untouched. The original is backed up to <tex>.bak before writing.

It prints the value assigned to each macro and flags any macro still left at
the "??" placeholder.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd


def fmt(v, nd: int = 2, signed: bool = False) -> str:
    """Format a summary value for LaTeX, returning '??' when it is unavailable."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "??"
    if isinstance(v, str):
        return v
    f = float(v)
    if abs(f - round(f)) < 1e-9 and abs(f) < 1e6 and nd == 0:
        return str(int(round(f)))
    return f"{f:+.{nd}f}" if signed else f"{f:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supply", default="out_supply",
                    help="output directory written by supply_interaction.py")
    ap.add_argument("--tex", default="paper/manuscript.tex",
                    help="LaTeX file containing the NUMBERS TO FILL IN block")
    a = ap.parse_args()
    d = Path(a.supply)

    s5 = pd.read_csv(d / "S5_summary.csv").set_index("quantity")["value"]

    def g(key, default=None):
        return s5[key] if key in s5.index else default

    s2 = pd.read_csv(d / "S2_randomization_test.csv").set_index("estimator")

    def e(estimator, col):
        if estimator in s2.index and col in s2.columns:
            return s2.loc[estimator, col]
        return None

    macros = {
        "supNoos":   fmt(g("N out-of-sample weeks"), 0),
        "supNdraws": fmt(g("N placebo draws"), 0),
        "supNctrl":  fmt(g("N controls"), 0),
        "supNnews":  "3",
        "supKsup":   fmt(g("k of CTRL_SUP"), 0),
        "supKgprx":  fmt(g("k of CTRL_SUP_GPRX"), 0),
        "supNoauc":  fmt(g("no-auction share of weeks (%)"), 1),
        "supClip":   "4",
        # --- Ridge arm ---
        "supRctrl":    fmt(e("Ridge", "r2_CTRL"), 2, True),
        "supRsup":     fmt(e("Ridge", "r2_CTRL_SUP"), 2, True),
        "supRgprx":    fmt(e("Ridge", "r2_GPRX"), 2, True),
        "supRplcmean": fmt(e("Ridge", "placebo_mean"), 2, True),
        "supRplcsd":   fmt(e("Ridge", "placebo_sd"), 2),
        "supRplclo":   fmt(e("Ridge", "placebo_p05"), 2, True),
        "supRplchi":   fmt(e("Ridge", "placebo_p95"), 2, True),
        "supRpct":     fmt(e("Ridge", "gpr_percentile_in_null"), 0),
        "supRp":       fmt(e("Ridge", "randomization_p"), 3),
        "supRsse":     fmt(e("Ridge", "sse_share_worst1pct"), 1),
        # --- ElasticNet arm ---
        "supEctrl":    fmt(e("ENet", "r2_CTRL"), 2, True),
        "supEsup":     fmt(e("ENet", "r2_CTRL_SUP"), 2, True),
        "supEgprx":    fmt(e("ENet", "r2_GPRX"), 2, True),
        "supEplcmean": fmt(e("ENet", "placebo_mean"), 2, True),
        "supEplcsd":   fmt(e("ENet", "placebo_sd"), 2),
        "supEplclo":   fmt(e("ENet", "placebo_p05"), 2, True),
        "supEplchi":   fmt(e("ENet", "placebo_p95"), 2, True),
        "supEpct":     fmt(e("ENet", "gpr_percentile_in_null"), 0),
        "supEp":       fmt(e("ENet", "randomization_p"), 3),
        "supEsse":     fmt(e("ENet", "sse_share_worst1pct"), 1),
        # --- policy channel / in-sample / TNAC leave-out ---
        "supPolNspec":   fmt(g("policy: n stable specs"), 0),
        "supPolNdrop":   fmt(g("policy: n UNSTABLE dropped"), 0),
        "supPolMinP":    fmt(g("policy: min raw p"), 3),
        "supPolMinHolm": fmt(g("policy: min Holm p"), 3),
        "supInsMinP":    fmt(g("in-sample: min raw p"), 3),
        "supInsMinHolm": fmt(g("in-sample: min Holm p"), 3),
        "supTnacN":      fmt(g("TNAC leave-out: n weeks"), 0),
    }

    # Both cardinalities are written by supply_interaction.py. Derive them only
    # as a fallback, for summary files produced by an older version.
    n_ctrl = g("N controls")
    if macros["supKsup"] == "??" and n_ctrl is not None:
        macros["supKsup"] = str(int(float(n_ctrl)) + 3 + 1)      # ctrl + news + no_auction
    if macros["supKgprx"] == "??" and n_ctrl is not None:
        macros["supKgprx"] = str(int(float(n_ctrl)) + 3 + 1 + 3 * (1 + 3))

    tex = Path(a.tex)
    src = tex.read_text(encoding="utf-8")
    start = src.index("% ############  NUMBERS TO FILL IN")
    end = src.index("% ############  END NUMBERS BLOCK")
    block = src[start:end]

    n_written = 0
    for key, value in macros.items():
        pattern = re.compile(r"(\\newcommand\{\\" + key + r"\}\{)[^}]*(\})")
        block, n = pattern.subn(lambda m: m.group(1) + value + m.group(2), block)
        n_written += n
        print(f"  \\{key:<14s} = {value}")

    shutil.copy(tex, str(tex) + ".bak")
    tex.write_text(src[:start] + block + src[end:], encoding="utf-8")

    unresolved = re.findall(r"\\newcommand\{\\(sup[A-Za-z]+)\}\{\?\?\}", block)
    print(f"\n  wrote {n_written} macros; backup at {tex}.bak")
    if unresolved:
        print(f"  WARNING: still unresolved: {unresolved} -- check the source by hand")
    else:
        print("  OK: no '??' placeholders remain")
    print("\n  next: pdflatex -> bibtex -> pdflatex x2")


if __name__ == "__main__":
    main()
