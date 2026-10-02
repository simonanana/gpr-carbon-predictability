# Data

## What is and is not in this repository

| | Included | Reason |
|---|---|---|
| Code | Yes | MIT |
| Market-level and summary results (`results/`) | Yes | Derived statistics, not raw data |
| Figures | Yes | CC BY 4.0 |
| Working paper | Yes | Author's own |
| **Raw allowance prices** | **No** | Vendor and ICAP terms do not permit redistribution |
| **Constructed panels** | **No** | Contain the raw prices above |

The included results are sufficient to verify every inference step in the
cross-market section, and to re-run the entire inference layer. They are not
sufficient to re-estimate forecasts from raw prices; that requires
reconstructing the panels from the sources below.

---

## Sources

### Freely redistributable

| Series | Source | Notes |
|---|---|---|
| Geopolitical risk index (daily, and threats/acts sub-indices) | [Caldara and Iacoviello](https://www.matteoiacoviello.com/gpr.html) | `data_gpr_daily_recent.xls` |
| Country-level GPR, 43 countries | same | `data_gpr_export.xls` |
| Climate Policy Uncertainty index | [Gavriilidis (2021)](https://www.policyuncertainty.com/climate_uncertainty.html) | `cpu_pu.xlsx` |
| EU27 natural gas imports by partner | Eurostat `nrg_ti_gasm` | Open data licence |
| EEX primary-market auction reports | [EEX](https://www.eex.com/) | Public PDFs; the derived variables are hand-collected |

### Not redistributable

| Series | Source | Notes |
|---|---|---|
| EUA futures settlement, weekly | Commercial vendor export | Terms prohibit redistribution |
| Brent, TTF gas, API2 coal, VSTOXX | same | same |
| Multi-market allowance prices (UKA, CEA, GDEA, HBEA, KAU, NZU, RGGI, CCA) | ICAP Allowance Price Explorer | Terms require attribution and prohibit bulk redistribution |

---

## Panel construction

Two panels feed the modules in this repository.

**`data/panel_weekly_v2.csv`** (606 × 75) — the weekly EU ETS panel.
Log returns `r_{t+1} = ln P_{t+1} − ln P_t`; the forecast target `y` is
`r_{t+1}`, a one-step-ahead weekly horizon. Carries the GPR variable family,
energy and financial controls, climate-policy uncertainty, and the EEX auction
variables.

**`data/panel_multimarket.csv`** (4200 × 70) — eleven weekly allowance series,
January 2015 to July 2026, with per-market characteristics: `region`,
`allocation`, `energy_import_dep`, `euro_geo_exposure`.

Prices are kept in **local currency**. Converting to a common numeraire would
inject exchange-rate variation into the measured return, and because the US
dollar is itself a first-order beneficiary of geopolitical risk-off flows,
dollar-denominated carbon returns would mechanically co-move with GPR through a
pure currency channel.

---

## Real-time alignment

Every predictor is aligned to its **publication** date rather than its
reference period. This matters most for annually published quantities: the
Total Number of Allowances in Circulation for reference year *Y* is published
in late May of *Y*+1, after the European close. Merging on the reference year
would introduce a twelve- to seventeen-month look-ahead.

All data-dependent constructions — the AR(1) surprise, expanding-window
quantile thresholds, orthogonalisation, the supply-surprise standardisation
scale — use training-window data only.

---

## Known data issues

These are recorded because each one caused a real error during construction.

**ICAP dates are US-format `M/D/YY`, not `D/M/YY`.** Parsing them as
day-first silently converts roughly 60% of observations to `NaT` and
transposes month and day on the rest.

**Vendor exports default to the last twelve months.** The date range must be
set manually before export. TTF gas history from this source begins only in
July 2022, which is why 64.2% of the weekly gas control is backfilled from the
month-end series. VSTOXX is backfilled for 12.2% of weeks. Both backfill shares
are reported in the paper's data appendix; the backfilled series are step
functions within the month.

**Eurostat TSV files must not be opened in Excel.** Excel inserts tab
characters and shifts columns. Read them programmatically.

**Eurostat dimension order is `freq,siec,partner,unit,geo`.** Filter to
`geo=EU27_2020`, and exclude intra-EU partners: with `geo=EU27`, partner-level
records for member states capture transit and re-export flows through the
Dutch, German, Belgian and Slovak hubs rather than extra-EU supply. Including
them both dilutes high-risk suppliers and inflates apparent country coverage.

**Do not select `G3200` (LNG) alongside `G3000`.** `G3000` already includes
LNG — Qatar and Trinidad appear under it — so selecting both double-counts.

**TNAC is retained as publication dates only.** The level series is 72%
missing and is an annual step function; it does not enter the main
specification. See the discussion of why a TNAC event study is not viable in
`src/supply_interaction.py`.

**Transit partners in the coverage calculation deserve scrutiny.** The
exposure-weighted index covers 76.2% of extra-EU gas imports on average. The
intra-EU filter described above removes member-state hubs, but a non-member
transit country is not caught by that rule while being economically the same
kind of record. Anyone reconstructing the weights should check the partner
breakdown directly rather than relying on the filter alone.

---

## Attribution

Users of the multi-market panel must attribute the International Carbon Action
Partnership (ICAP) Allowance Price Explorer. Users of the geopolitical risk
index should cite Caldara, D. and M. Iacoviello (2022), "Measuring Geopolitical
Risk," *American Economic Review* 112(4), 1194–1225.
