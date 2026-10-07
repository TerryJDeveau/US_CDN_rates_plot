# Batch 4: Germany, the plan and the provisional decisions (2026-10-06 late)

Terry, shutting the laptop down: "On decisions that you want from me, use your best judgement for
provisional decisions. Can always be tweaked later." So each call below is **provisional**:
build it, and report it in the pull request with its alternative, as batches 1-3 did.

Work on this branch (`batch4-de-sources`) or one cut from it, with a **draft** pull request whose
description is this plan; never `main` (a push to main changes the live page at its next Reboot).
Follow the UK's path (skill `ratesplot`, `references/uk.md` and `per-nation.md`, "Adding a
nation"): parsers written against the saved responses in `raw/`, a `tools/check_de_parsers.py`
like `check_uk_parsers.py`, synthetic answers in `tools/synthetic_sources.py`, proofs on
`--synthetic` that every old regression case and command line is unchanged, prune `raw/` to
what the checks read before the merge.

## Provisional decisions

1. **Drawn only when named**, like the UK. Nation key `de`, name "Germany", adjective "German".
   Codes `de` and `ger`. ⚠ Check the spelling rule first: `--de` may already be a leading part of
   `--debt` (and `--de:` of `--debt:LETTERS`). If so, the switch is `--ger[many]` and the
   per-nation prefix `--ger:`; `de` stays a code in `--nations:de` only. Never let table order
   decide (`cli-and-window.md`).
2. **Yields: the Bundesbank's Svensson term structure of listed Federal securities** (zero-coupon,
   fitted), terms 1, 2, 3, 5, 7, 10, 15, 20, 30 years. Daily from 1997-08 (30y from 2000-08),
   monthly before it (1-10y from 1972-09, 15 and 20 from 1986-06), joined monthly-then-daily as
   the UK's and Canada's are (`e5701a4`'s helper). Default terms **2, 5, 10, 30** (Schatz, Bobl,
   Bund, Buxl). Title "German Federal Yields (Svensson fitted)". Alternative: the ECB's 10-year
   convergence yield only.
3. **Before 1972-09, the 10-year carried back to 1955** by the Bundesbank's monthly yield on
   public debt securities outstanding (`bbk01_WU0004`), baked (it is frozen at 2020-04), as its own
   history like the UK's M10 col 31. `check_sources` measures the gap at the join.
4. **Policy rate: the ECB's main refinancing rate from 1999** (fixed rate; the minimum bid rate
   2000-06-28 to 2008-10-14, when the tenders were variable), on its change days. Alternative:
   the deposit facility rate, which has steered market rates since 2015 and is the ECB's stated
   policy rate since 2024-09; easy to switch. The Bundesbank's discount rate 1948-1998 is joined in
   front once its key is found on the laptop (below); until then the curve starts in 1999.
5. **Mortgages: MFI rates on new housing loans to households** (Germany, from 2003-01), by initial
   fixation: over 1 to 5 years, over 5 to 10, over 10, plus floating when its ECB key is fixed
   (`float` answered 500). Spellings must not clash with existing terms (30 15 5 3 1 5v prime 2f 3f
   5f svr); suggested `5d`, `10d`, `10+d`, `vd`, or better if the matcher prefers. Default: the
   **5-10 years** (the usual German fixation). Labelled "new loans, avg." (they are averages of
   loans made, not posted or quoted rates).
6. **Spreads default: 10y-2y and 30y-10y** (Germany has no 3-month term here).
7. **Debt: general government gross debt (Maastricht)**, quarterly, ECB GFS (`ecb_gfs_q_debt`)
   or Eurostat `gov_10q_ggdebt` (S13), whichever starts earlier; annual (`ecb_gfs_a_debt`)
   carried back where the quarters stop. Levels: `f` Federal (Bund, S1311), `p` Länder (S1312, the
   states), `m` Local (Gemeinden, S1313), `n` non-federal = Länder + local. Social security funds
   (S1314) are left out of the levels and noted. The subsectors are unconsolidated, so they need
   not add to the total: use the UK's rule for the bake's "levels add up" check.
8. **Interest: D.41 interest payable, general government**, quarterly (Eurostat `gov_10q_ggnfa`
   D41PAY, S13; levels from S1311-S1313), summed over four quarters (TTM), as every nation's is.
9. **GDP: current prices, quarterly, not seasonally adjusted** (ECB MNA or Eurostat), summed over
   four quarters. **Population**: Eurostat `namq_10_pe` (thousands, quarterly), annual `demo_pjan`
   before it.
10. **Before 1991: West Germany, joined as it is**, a step at unification (like the UK's 1922
    step for Ireland), Deutsche Mark figures in euros at 1.95583 (the Bundesbank's and Destatis'
    own convention). The sources for 1950-1990 still have to be found on the laptop and baked; until
    then the macro curves start where the euro-area series start (about 1991-1995).
11. **`--cur`: CNBC** `DE2Y-DE`, `DE5Y-DE`, `DE10Y-DE`, `DE30Y-DE` (and 1, 3, 7, 20 years if the
    saved response has them); used as they are, as the UK's. No market debt steers the projection.
12. **Currency** "EUR", prefix "€", axis "Nominal Units (EUR – Log Scale)". Euro values throughout,
    including before 1999 (converted DM, as published).

## Left for the laptop (the cloud cannot reach the sources)

- FRED's German series (timed out with a browser User-Agent; retry without one, as
  `http.fetch_fred_csv` does): IRLTLT01DEM156N, INTDSRDEM193N, IR3TIB01DEM156N, CPMNACSCAB1GQDE,
  GGGDTADEA188N, DEUPOPNDQ.
- The Bundesbank discount and Lombard rates 1948-1998 (SDMX flow `BBIN1` or `BBIM1`; the old
  `BBK01.SU0112` is gone).
- West German GDP, debt, interest and population 1950-1990 (Destatis long series or the
  Bundesbank), to bake.
- Mortgage rates before 2003 (the Bundesbank's old interest rate statistics, to 2003-06).
- ECB keys that failed: interest `GFS ... D41`, floating-rate housing loans, `ENA ... POP`.
- After the merge, on the laptop: `check_sources` with real data, the real regression run, the
  Reboot of the page (standing yes), the ideas file and the skill updated.
