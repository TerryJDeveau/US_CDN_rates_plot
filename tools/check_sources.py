#!/usr/bin/env python3
"""Check every live source the program reads: does it answer, in the form the program expects, and how fresh is it?

Each check runs the program's own fetch code, not a copy of it, so a source
that has changed its form fails here as it would in a chart, with the same
warning. For each source the newest date it has and its age in days are
printed, and the source is marked:

* OK      it answered and the program read it;
* STALE   it answered, but its newest value is older than that source
          normally is (it may have stopped updating); does not fail the run;
* WARN    the program printed a ``Warning:`` (it fell back to another source
          or left something out); fails the run;
* FAIL    the fetch raised; fails the run.

Two checks are of a different kind:

* CNBC's quote feed is unofficial and undocumented. Every symbol must still
  give a yield, a time and the day's change in the form
  ``latest.intraday_yields_after`` reads; if not, fix that function (the
  symbols and fields are in the us-cdn-rates-plot skill, section 3).
* ``uk_joins`` measures where the UK's baked history meets its live series
  (the 10- and 20-year yields, debt in 1975, GDP 1955-1994), for the
  judgement calls of batch 3; a note, never a failure.
* The Census publishes each fiscal year of state and local government
  finances about a year and a half after it ends. When a year newer than the
  one baked into ``us_archive_data`` is listed, the line says so: run
  ``python tools/bake_archives.py --only us``, check the charts, commit.

Usage (from the project root)::

    python tools/check_sources.py              # every source (a minute or two; the StatCan tables are large)
    python tools/check_sources.py --only cnbc,census
    python tools/check_sources.py --list

Exit code 1 if any source is WARN or FAIL.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import traceback
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot import bake, cdn_data, latest, rates, uk_archive_data, uk_data, us_data  # noqa: E402
from ratesplot.cli import parse_args  # noqa: E402
from ratesplot.config import (  # noqa: E402
    CDN_DEBT_COLUMN,
    CDN_INTEREST_COLUMN,
    GDP_COLUMN,
    MARKET_TIMEZONE,
    UK_DEBT_COLUMN,
    UK_INTEREST_COLUMN,
    UK_NET_DEBT_COLUMN,
    YIELD_COLUMNS,
    YIELD_TERMS,
    PlotConfig,
    component_column,
)
from ratesplot.frames import at_quarter_end  # noqa: E402
from ratesplot.http import fetch_fred_csv  # noqa: E402
from ratesplot.us_archive_data import EMBEDDED_US_DEBT_BY_LEVEL  # noqa: E402

TODAY = pd.Timestamp.now(tz=MARKET_TIMEZONE).tz_localize(None).normalize()
# How far back the daily sources are asked for: enough to hold a few business days.
RECENT = TODAY - pd.Timedelta(days=21)

# Usual age limits (days since the date the newest value describes).
DAILY = 7  # business days, a long weekend, a day's publication lag
QUOTES = 4  # Friday's last quotes on the Tuesday after a long weekend
# A quarter's figures come out one to three months after it ends and stay the
# newest until the next quarter's do: up to about half a year.
QUARTERLY = 200
# The spread cross-checks fail when more than this share of days disagree.
_SPREAD_MISMATCH_LIMIT = 0.001
# The Bank of England's monthly quoted mortgage rates, dated here by the
# month's first day: a month is published early in the month after next.
MONTHLY = 100
# BEA's annual state and local interest: a calendar year is published the
# following autumn and stays the newest for another year.
ANNUAL = 700


@dataclass(frozen=True)
class Check:
    name: str
    what: str
    max_age_days: int | None  # None: no age limit (the result is a note, not a date)
    run: Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]  # newest date, and a note


def _last(series: pd.Series | pd.DataFrame | None) -> pd.Timestamp | None:
    """The newest date with a value; for a frame, the oldest of its columns' newest dates (the stalest column)."""
    if series is None or series.empty:
        return None
    if isinstance(series, pd.DataFrame):
        lasts = [series[column].last_valid_index() for column in series.columns]
        if any(last is None for last in lasts):
            return None
        return min(pd.Timestamp(last) for last in lasts)
    last = series.last_valid_index()
    return None if last is None else pd.Timestamp(last)


def _fred(series_id: str, *, quarterly: bool) -> Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]:
    def run(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
        frame = fetch_fred_csv(series_id).set_index("DATE")[series_id]
        last = _last(frame)
        if last is None:
            return None, "no values"
        # FRED dates a period by its first day; the age runs from its last.
        return (at_quarter_end(last) if quarterly else last + pd.offsets.YearEnd(0)), f"FRED dates it {last:%Y-%m-%d}"

    return run


# The U.S. terms outside the default five (--yields:LIST).
EXTRA_US_COLUMNS = tuple(column for column in YIELD_TERMS.values() if column not in YIELD_COLUMNS)


def _all_yields(frame: pd.DataFrame, columns: tuple[str, ...] = YIELD_COLUMNS) -> pd.DataFrame:
    """``frame`` if it has every yield column; raises otherwise (a tenor lost is a curve missing from the chart)."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"no {', '.join(missing)} column")
    return frame[list(columns)]


def _cdn_yields(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    return _last(_all_yields(cdn_data.fetch_cdn_yields(config))), ""


def _cdn_column(fetch: Callable[[PlotConfig], pd.DataFrame | None], column: str) -> Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]:
    def run(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
        frame = fetch(config)
        return _last(None if frame is None else frame[column]), ""

    return run


def _cdn_levels(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    frame = cdn_data.fetch_cdn_components(config)
    return _last(frame), f"columns: {len(frame.columns) if frame is not None else 0}"


def _cdn_population(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    # A quarter's population is its first day's estimate, dated then.
    return _last(cdn_data.fetch_cdn_population()), "dated by the quarter's first day, as estimated"


def _market_debt(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    market = latest.cdn_market_debt(TODAY - pd.Timedelta(days=400))
    return _last(market), f"C${market.iloc[-1] / 1e12:.3f} trillion, from {market.index[0]:%Y-%m-%d}"


def _us_yields(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    return _last(_all_yields(us_data.fetch_us_yields(config).set_index("DATE"))), ""


def _us_extra_yields(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    every = replace(config, yield_terms=tuple(YIELD_TERMS))
    return _last(_all_yields(us_data.fetch_us_yields(every).set_index("DATE"), EXTRA_US_COLUMNS)), ""


def _treasury_yields(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    columns = tuple(YIELD_TERMS.values())
    return _last(_all_yields(latest.treasury_yields_after(RECENT, config, columns).set_index("DATE"), columns)), ""


def _debt_to_penny(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    debt = latest.treasury_debt_from(RECENT, config)
    return _last(debt), f"US${debt.iloc[-1] / 1e12:.3f} trillion" if len(debt) else ""


def _quotes(country: str, columns: tuple[str, ...] | None = None) -> Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]:
    def run(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
        quotes = latest.fetch_quotes(country, columns)
        low, high = latest._PLAUSIBLE_YIELD  # the program's own rules for a usable quote
        problems, newest = [], None
        for column, quote in quotes.items():
            try:
                value = latest._percent(quote["last"])
                when = pd.Timestamp(quote["last_time"]).tz_convert(MARKET_TIMEZONE)
                latest._day_change(quote)
                if not low <= value <= high:
                    raise ValueError(f"{value} is not a plausible yield")
            except (TypeError, KeyError, ValueError) as exc:
                problems.append(f"{column}: {exc!r}")
                continue
            newest = when if newest is None else max(newest, when)
        if problems:
            raise ValueError("unusable quotes: " + "; ".join(problems))
        return (None if newest is None else newest.tz_localize(None)), f"all {len(quotes)} symbols read; newest {newest:%Y-%m-%d %H:%M %Z}"

    return run


def _rate(fetch: Callable[[], rates.RateCurve | None]) -> Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]:
    """A curve of ``ratesplot.rates``, through its own fetcher; its label is the note (it names a join)."""

    def run(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
        curve = fetch()
        if curve is None:
            return None, "not fetched"
        return _last(curve.values), f"{curve.label}, from {curve.first:%Y-%m-%d}"

    return run


def _spread_crosscheck(series_id: str, pair: tuple[str, str]) -> Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]:
    """The program's spread (``rates.yield_spreads``, from the DGS yields) against FRED's own series of it.

    FRED's T10Y2Y and T10Y3M are the same two DGS series subtracted, and agree
    with the program's to rounding (0.005) on all but a handful of days.
    On those few FRED's spread is not the difference of the two yields it
    publishes for that day (Terry's run, 2026-10-06: 3 of 12,582 days, up to
    0.02 points, and 2 of 11,191, up to 0.10, even on days both yields have
    their own value). The program's spread is the difference of the yields
    it draws, so those days are named in the note, not failed. Many such
    days (more than ``_SPREAD_MISMATCH_LIMIT`` of them) mean the two are
    misaligned, which fails.
    """

    def run(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
        chosen = replace(config, start=pd.Timestamp("1900-01-01"), spreads=True, spread_pairs=(pair,))
        (curve,) = rates.yield_spreads("us", us_data.fetch_us_yields(chosen), chosen)
        fred = fetch_fred_csv(series_id).set_index("DATE")[series_id].dropna()
        common = fred.index.intersection(curve.values.index)
        if common.empty:
            raise ValueError("no day in common")
        gap = (curve.values.loc[common] - fred.loc[common]).abs()
        off = gap[gap > 0.005]
        if len(off) > max(_SPREAD_MISMATCH_LIMIT * len(common), 5):
            raise ValueError(f"differs from {series_id} by up to {gap.max():.3f} points ({len(off)} of {len(common)} days)")
        note = f"{len(common) - len(off)} of {len(common)} days agree with {series_id}; from {fred.index[0]:%Y-%m-%d}"
        if len(off):
            days = ", ".join(f"{day:%Y-%m-%d} ({curve.values[day]:+.2f} vs {fred[day]:+.2f})" for day in off.index)
            note += f"; FRED's spread is not its yields' difference on {days}"
        return _last(fred), note

    return run


def _cmhc_history(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    # Used only before the weekly posted rate begins (1975), so its age does not matter.
    history = rates.cdn_mortgage_history()
    return None, f"{len(history)} months, {history.index[0]:%Y-%m} to {history.index[-1]:%Y-%m}"


# ---------------------------------------------------------------------------
# The UK (batch 3)
# ---------------------------------------------------------------------------


def _uk_yields(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    chosen = replace(config, yield_terms=uk_data.UK_YIELD_TERMS)
    frame = uk_data.fetch_uk_yields(chosen)
    columns = tuple(uk_data.UK_YIELD_SERIES)
    return _last(_all_yields(frame, columns)), ", ".join(f"{column} {frame[column].dropna().iloc[-1]:.4f}" for column in columns)


def _uk_ons(fetch: Callable[[PlotConfig], pd.DataFrame | None], column: str, *, components: str = "") -> Callable[[PlotConfig], tuple[pd.Timestamp | None, str]]:
    """One of the UK's ONS-based curves, through its own fetcher; the note gives its newest value."""

    def run(config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
        frame = fetch(replace(config, components=components) if components else config)
        series = None if frame is None or column not in frame.columns else frame[column].dropna()
        if series is None or series.empty:
            return None, "no values"
        return _last(series), f"£{series.iloc[-1] / 1e9:,.1f} billion, from {series.index[0]:%Y-%m-%d}"

    return run


def _uk_population(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    people = uk_data.fetch_uk_population()
    return _last(people), f"{people.iloc[-1] / 1e6:.2f} million; a quarter dated by its middle"


def _uk_joins(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    """Where the baked UK history meets the live series: measured, for the PR's open questions (batch 3).

    The 10- and 20-year: the baked monthly averages against the monthly
    averages of the daily par yields over the months both have. Debt: the
    national debt and general government debt at 31 March 1975. GDP: the
    ratio of ONS's four-quarter sums to the annual history at each year end
    1955-1994 (what joins.chain scales the history by). A note, not a pass
    or fail: the numbers decide whether the joins are good enough.
    """
    notes = []
    for column, rows in (("10-Year", uk_archive_data.EMBEDDED_UK_10Y_HISTORY), ("20-Year", uk_archive_data.EMBEDDED_UK_20Y_HISTORY)):
        baked = pd.Series({pd.Timestamp(date): value for date, value in rows})
        daily = uk_data.iadb_series(uk_data.UK_YIELD_SERIES[column])
        monthly = daily.groupby(daily.index.to_period("M")).mean()
        monthly.index = monthly.index.start_time
        gap = (baked - monthly).dropna()
        notes.append(f"{column} baked less par: mean {gap.mean():+.3f}, largest {gap.abs().max():.3f} over {len(gap)} months")
    national = dict(uk_archive_data.EMBEDDED_UK_DEBT_HISTORY).get("1975-03-31")
    gross = uk_data.ons_series(uk_data.UK_DEBT_SERIES, "quarters").get(pd.Timestamp("1975-03-31"))
    if national and gross:
        notes.append(f"debt 1975-03-31: general government {gross / 1e9:.1f} bn, national {national / 1e9:.1f} bn (x{gross / national:.3f})")
    history = pd.Series({pd.Timestamp(date): value for date, value in uk_archive_data.EMBEDDED_UK_GDP_HISTORY})
    live = uk_data.ttm_sum(uk_data.ons_series(uk_data.UK_GDP_SERIES, "quarters"))
    ratio = (live / history).dropna()
    ratio = ratio.loc[ratio.index.year <= 1994]
    notes.append(f"GDP ONS / history 1955-1994: median {ratio.median():.4f}, {ratio.min():.4f} to {ratio.max():.4f}")
    return None, "; ".join(notes)


def _census(_config: PlotConfig) -> tuple[pd.Timestamp | None, str]:
    baked = max(pd.Timestamp(date).year for date, _value in EMBEDDED_US_DEBT_BY_LEVEL["p"])
    listed = [year for year in (baked + 1, baked + 2) if bake.census_estimates_url(year)]
    if listed:
        return None, (
            f"ACTION: fiscal {listed[-1]} is published, baked only to {baked}: "
            "run python tools/bake_archives.py --only us, check the charts, commit"
        )
    return None, f"baked to fiscal {baked}, the newest the Census lists"


CHECKS: tuple[Check, ...] = (
    Check("boc", "Bank of Canada benchmark yields and 3-month bill (Valet)", DAILY, _cdn_yields),
    Check("boc_debt", "Bank of Canada: GoC market debt outstanding (Valet)", DAILY, _market_debt),
    Check("statcan_debt", "StatCan 36-10-0467 general government gross debt", QUARTERLY, _cdn_column(cdn_data.fetch_cdn_debt, CDN_DEBT_COLUMN)),
    Check("statcan_gdp", "StatCan 36-10-0104 GDP", QUARTERLY, _cdn_column(cdn_data.fetch_cdn_gdp, GDP_COLUMN)),
    Check("statcan_interest", "StatCan 10-10-0015 interest", QUARTERLY, _cdn_column(cdn_data.fetch_cdn_interest, CDN_INTEREST_COLUMN)),
    Check("statcan_levels", "StatCan 10-10-0015 debt and interest by level", QUARTERLY, _cdn_levels),
    Check("statcan_population", "StatCan 17-10-0009 population", QUARTERLY, _cdn_population),
    Check("fred_yields", "FRED Treasury yields (DGS3MO, DGS2, DGS5, DGS10, DGS30)", DAILY, _us_yields),
    Check("fred_yields_more", "FRED Treasury yields (DGS1MO, DGS6MO, DGS1, DGS7, DGS20)", DAILY, _us_extra_yields),
    Check("fred_gfdebtn", "FRED GFDEBTN federal debt", QUARTERLY, _fred("GFDEBTN", quarterly=True)),
    Check("fred_slgsdodns", "FRED SLGSDODNS state and local debt", QUARTERLY, _fred("SLGSDODNS", quarterly=True)),
    Check("fred_gdp", "FRED GDP", QUARTERLY, _fred("GDP", quarterly=True)),
    Check("fred_interest", "FRED A180RC1Q027SBEA government interest", QUARTERLY, _fred("A180RC1Q027SBEA", quarterly=True)),
    Check("fred_interest_f", "FRED A091RC1Q027SBEA federal interest", QUARTERLY, _fred("A091RC1Q027SBEA", quarterly=True)),
    Check("fred_interest_n", "FRED B111RC1Q027SBEA state and local interest", QUARTERLY, _fred("B111RC1Q027SBEA", quarterly=True)),
    Check("fred_interest_p", "FRED W756RC1A027NBEA state interest (annual)", ANNUAL, _fred("W756RC1A027NBEA", quarterly=False)),
    Check("fred_interest_m", "FRED W856RC1A027NBEA local interest (annual)", ANNUAL, _fred("W856RC1A027NBEA", quarterly=False)),
    Check("fred_population", "FRED B230RC0Q173SBEA population", QUARTERLY, _fred("B230RC0Q173SBEA", quarterly=True)),
    Check("fred_dff", "FRED DFF effective federal funds rate (--policy)", DAILY, _rate(rates.us_policy_rate)),
    Check("boc_policy", "Bank of Canada Bank Rate V122530, CORRA AVG.INTWO (--policy)", DAILY, _rate(rates.cdn_policy_rate)),
    Check("fred_mortgage30", "FRED MORTGAGE30US (--mortgages:30)", DAILY, _rate(partial(rates.mortgage_rate, "30"))),
    Check("fred_mortgage15", "FRED MORTGAGE15US (--mortgages:15)", DAILY, _rate(partial(rates.mortgage_rate, "15"))),
    Check("boc_mortgage5", "Bank of Canada V80691335 posted 5-year, StatCan 34-10-0145 before", DAILY, _rate(partial(rates.mortgage_rate, "5"))),
    Check("boc_mortgage3", "Bank of Canada V80691334 posted 3-year", DAILY, _rate(partial(rates.mortgage_rate, "3"))),
    Check("boc_mortgage1", "Bank of Canada V80691333 posted 1-year", DAILY, _rate(partial(rates.mortgage_rate, "1"))),
    Check("boc_mortgage5v", "Bank of Canada BROKER_AVERAGE_5YR_VRM 5-year variable", DAILY, _rate(partial(rates.mortgage_rate, "5v"))),
    Check("boc_prime", "Bank of Canada V80691311 prime rate", DAILY, _rate(partial(rates.mortgage_rate, "prime"))),
    Check("statcan_mortgage", "StatCan 34-10-0145 CMHC 5-year rate (monthly; before 1975)", None, _cmhc_history),
    Check("fred_t10y2y", "FRED T10Y2Y against the program's 10y-2y spread (--spreads)", DAILY, _spread_crosscheck("T10Y2Y", ("10y", "2y"))),
    Check("fred_t10y3m", "FRED T10Y3M against the program's 10y-3m spread (--spreads)", DAILY, _spread_crosscheck("T10Y3M", ("10y", "3m"))),
    Check("treasury_yields", "U.S. Treasury daily par yield curve (all ten terms)", DAILY, _treasury_yields),
    Check("debt_to_penny", "U.S. Treasury Debt to the Penny", DAILY, _debt_to_penny),
    Check("cnbc_cdn", "CNBC quote feed, Canadian yields (unofficial)", QUOTES, _quotes("cdn")),
    Check("cnbc_us", "CNBC quote feed, U.S. yields (unofficial)", QUOTES, _quotes("us")),
    Check("cnbc_us_more", "CNBC quote feed, U.S. 1m 6m 1y 7y 20y (unofficial)", QUOTES, _quotes("us", EXTRA_US_COLUMNS)),
    Check("census", "U.S. Census state and local finances: a year newer than baked?", None, _census),
    # The UK (batch 3).
    Check("boe_yields", "Bank of England par yields IUDSNPY, IUDMNPY, IUDLNPY (5, 10, 20 years)", DAILY, _uk_yields),
    Check("boe_bank_rate", "Bank of England Bank Rate IUDBEDR (--policy)", DAILY, _rate(rates.uk_policy_rate)),
    Check("boe_mortgage2f", "Bank of England IUMBV34 quoted 2-year fixed", MONTHLY, _rate(partial(rates.mortgage_rate, "2f"))),
    Check("boe_mortgage3f", "Bank of England IUMBV37 quoted 3-year fixed", MONTHLY, _rate(partial(rates.mortgage_rate, "3f"))),
    Check("boe_mortgage5f", "Bank of England IUMBV42 quoted 5-year fixed", MONTHLY, _rate(partial(rates.mortgage_rate, "5f"))),
    Check("boe_mortgage_svr", "Bank of England IUMTLMV standard variable rate", MONTHLY, _rate(partial(rates.mortgage_rate, "svr"))),
    Check("ons_debt", "ONS BKPX general government gross debt (monthly)", QUARTERLY, _uk_ons(uk_data.fetch_uk_debt, UK_DEBT_COLUMN)),
    Check("ons_net_debt", "ONS HF6W public sector net debt ex banks (monthly)", QUARTERLY, _uk_ons(uk_data.fetch_uk_debt, UK_NET_DEBT_COLUMN)),
    Check("ons_gdp", "ONS YBHA GDP (four-quarter sums)", QUARTERLY, _uk_ons(uk_data.fetch_uk_gdp, GDP_COLUMN)),
    Check("ons_interest", "ONS NMYX general government interest (four-quarter sums)", QUARTERLY, _uk_ons(uk_data.fetch_uk_interest, UK_INTEREST_COLUMN)),
    Check("ons_central", "ONS BKPW central government debt (monthly)", QUARTERLY,
          _uk_ons(uk_data.fetch_uk_components, component_column("debt", "f"), components="fm")),
    Check("ons_local", "ONS MDYT local government debt", QUARTERLY,
          _uk_ons(uk_data.fetch_uk_components, component_column("debt", "m"), components="fm")),
    Check("ons_int_central", "ONS NMFX central government interest", QUARTERLY,
          _uk_ons(uk_data.fetch_uk_components, component_column("interest", "f"), components="fm")),
    Check("ons_int_local", "ONS NUGW local government interest", QUARTERLY,
          _uk_ons(uk_data.fetch_uk_components, component_column("interest", "m"), components="fm")),
    Check("ons_population", "ONS EBAQ UK population", QUARTERLY, _uk_population),
    Check("cnbc_uk", "CNBC quote feed, UK gilt yields, 5 10 20 years (unofficial)", QUOTES, _quotes("uk", tuple(uk_data.UK_YIELD_SERIES))),
    Check("uk_joins", "UK history against the live series where they meet (a measurement)", None, _uk_joins),
)


def run_check(check: Check, config: PlotConfig) -> tuple[str, list[str]]:
    """Run one check; return its report line and any lines to print under it."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            newest, note = check.run(config)
    except Exception as exc:
        detail = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return f"{check.name:18} {'':10} {'':>5}  FAIL   {check.what}", [detail]
    warnings = [line.strip() for line in output.getvalue().splitlines() if "Warning:" in line]
    if check.max_age_days is None:
        status, date_text, age_text = ("NOTE" if note.startswith("ACTION") else "OK"), "", ""
    elif newest is None:
        status, date_text, age_text = "FAIL", "", ""
        note = note or "no values"
    else:
        age = (TODAY - newest.normalize()).days
        status = "STALE" if age > check.max_age_days else "OK"
        date_text, age_text = f"{newest:%Y-%m-%d}", str(age)
    if warnings and status != "FAIL":
        status = "WARN"
    lines = ([note] if note else []) + warnings
    return f"{check.name:18} {date_text:10} {age_text:>5}  {status:5}  {check.what}", lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated check names")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    args = parser.parse_args(argv)
    if args.list:
        for check in CHECKS:
            print(f"{check.name:18} {check.what}")
        return 0
    checks = CHECKS if not args.only else tuple(check for check in CHECKS if check.name in args.only.split(","))
    unknown = [name for name in (args.only or "").split(",") if name and name not in {check.name for check in CHECKS}]
    if unknown:
        # Checking nothing must not read as a pass (a check name from another branch).
        parser.error(f"no such check: {', '.join(unknown)} (--list shows them)")

    # The program's own settings for a chart ending today, with --cur and every curve.
    config = parse_args(["--no-gui", "-c", "-u", "--d:fnpm", "--i", "--gdp", "--y", f"--s:{RECENT:%Y-%m-%d}"])
    print(f"Checking {len(checks)} source(s) on {TODAY:%Y-%m-%d} …")
    print(f"{'check':18} {'newest':10} {'age':>5}  status  source")
    failed = False
    for check in checks:
        line, notes = run_check(check, config)
        print(line, flush=True)
        for note in notes:
            print(f"{'':20}{note}")
        failed |= " WARN " in line or " FAIL " in line
    print("All sources answered as expected." if not failed else "Some sources need attention (WARN / FAIL above).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
