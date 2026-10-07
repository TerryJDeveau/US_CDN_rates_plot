"""Bank of England / Office for National Statistics data pipelines for the UK chart (batch 3).

Sources, measured from Terry's laptop on 2026-10-06 (the cloud copy cannot
reach them; ``tools/uk_fixtures/README.md`` holds the measurements):

* Yields: the Bank of England's fitted **par** gilt yields, daily, from its
  statistical database (IADB): 5-year from 1993-12, 10-year from 1993-11,
  20-year from 2000-01. Before them, the monthly average gilt yields of the
  Bank's "A millennium of macroeconomic data", baked (``uk_archive_data``):
  the 10-year from 1935 and the 20-year from 1963, drawn as steps, as
  Canada's monthly history is. Par rather than zero-coupon yields: the
  millennium's are redemption yields of real gilts, which the par curve
  follows (a par yield is the coupon a gilt priced at par would pay) and the
  zero-coupon curve does not, by up to 0.9 points in 1979-1981; and the
  market quotes the 10-year as a gilt's yield. The 2- and 30-year are not
  offered: only the Bank's workbooks have them (the 30-year from 2016 only),
  and the current ones are inside a 39 MB archive.
* Debt: general government consolidated gross debt (ONS BKPX, the
  Maastricht measure, the counterpart of Canada's general government gross
  debt), monthly from 1975-03. Before it the millennium's national debt
  (end of each financial year, from 1690/91), a different quantity (central
  government's debt), so a curve of its own, drawn dotted until BKPX begins,
  as Canada's federal debt before 1933 is; the two meet within 3 % (£52.1bn
  and £53.7bn at 31 March 1975). Beside it, public sector net debt
  excluding the public sector banks (ONS HF6W, the UK's headline measure),
  monthly from 1975-03 (Terry, 2026-10-06: "add net too").
* GDP: at market prices, current prices, seasonally adjusted (ONS YBHA),
  quarterly from 1955, summed over four quarters; before it the millennium's
  annual composite estimate, from 1700, joined as Canada's history is.
* Interest: general government interest and dividends paid to the private
  sector and the rest of the world (ONS NMYX), consolidated as Canada's and
  the U.S.'s aggregates are, quarterly from 1946, summed over four quarters.
* Levels (``--debt:LETTERS``): central government (f) and local government
  (m; n is the same, the only level that is not central). The devolved
  governments are inside central government in the UK accounts, so there is
  no "p" level. Central debt ONS BKPW (monthly from 1975), local MDYT
  (quarterly from 1966); central interest NMFX, local NUGW (from 1946).
  Debt one level owes another is netted out of the consolidated total
  (BKPX), so the levels add up to more than it.
* Population: UK resident population (ONS EBAQ, mid-year estimates
  interpolated quarterly, from 1955); before it the millennium's estimates
  (Great Britain 1707-1800, with all of Ireland 1801-1921, with Northern
  Ireland only from 1922: the step in 1922 is Ireland's independence).

Every figure is dated by the day it describes: a stock at the end of its
month or quarter, a four-quarter sum at the end of its last quarter. Values
are in pounds (and persons): ONS gives most series in £m, HF6W in £bn and
the population in thousands; each series' unit is checked against the one
measured, so a change of unit fails rather than shifting a curve a
thousandfold.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from .config import (
    BILLION,
    BOE_IADB_FROM,
    BOE_IADB_URL,
    COMPONENT_LETTERS,
    DATE_COLUMN,
    GDP_COLUMN,
    MILLION,
    ONS_TIMESERIES_URL,
    POPULATION_COLUMN,
    THOUSAND,
    UK_DEBT_COLUMN,
    UK_INTEREST_COLUMN,
    UK_NATIONAL_DEBT_COLUMN,
    UK_NET_DEBT_COLUMN,
    YIELD_TERMS,
    PlotConfig,
    component_column,
)
from .http import uk_get
from .joins import chain, embedded_frame, ttm_sum
from .uk_archive_data import (
    EMBEDDED_UK_10Y_HISTORY,
    EMBEDDED_UK_20Y_HISTORY,
    EMBEDDED_UK_DEBT_HISTORY,
    EMBEDDED_UK_GDP_HISTORY,
    EMBEDDED_UK_POPULATION_HISTORY,
)

# Chart column -> IADB code of the Bank of England's daily nominal par yield,
# and its first day (measured 2026-10-06).
UK_YIELD_SERIES = {"5-Year": "IUDSNPY", "10-Year": "IUDMNPY", "20-Year": "IUDLNPY"}
UK_YIELD_EARLIEST = {
    "5-Year": pd.Timestamp("1993-12-01"),
    "10-Year": pd.Timestamp("1993-11-01"),
    "20-Year": pd.Timestamp("2000-01-04"),
}
# The UK's terms (config.YIELD_TERMS keys), in their order.
UK_YIELD_TERMS = tuple(term for term, column in YIELD_TERMS.items() if column in UK_YIELD_SERIES)
# The baked monthly history of a term, drawn as steps before its daily series begins.
_YIELD_HISTORY = {"10-Year": EMBEDDED_UK_10Y_HISTORY, "20-Year": EMBEDDED_UK_20Y_HISTORY}
# Where ``fetch_uk_yields`` records, in the frame's ``attrs``, the last date
# of each column's monthly history ({column: date}), which is drawn as steps.
YIELD_HISTORY_ATTR = "monthly_history_through"
_NO_VALUES = pd.Series(dtype=float, index=pd.DatetimeIndex([], name=DATE_COLUMN))
# Short forward-fill of the daily yields: bridges a holiday or a missing
# print in one term, so a spread has a value on the days the other has (as
# for Canada's), never more than this many rows.
_DAILY_FILL_LIMIT = 3

# ONS series: CDID -> (the path of its time-series page, the unit its
# description gives, the multiplier to pounds or persons). MDYT's description
# (dataset EDP) has no unit: it is £m, as the EDP tables say, and its 2026 Q2
# value (£138,883m) is the right size beside BKPX's £3,183,350m.
ONS_SERIES = {
    "BKPX": ("/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/bkpx/pusf", "m", MILLION),
    "BKPW": ("/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/bkpw/pusf", "m", MILLION),
    "MDYT": ("/economy/governmentpublicsectorandtaxes/publicspending/timeseries/mdyt/edp", "", MILLION),
    "HF6W": ("/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/hf6w/pusf", "bn", BILLION),
    "NMYX": ("/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/nmyx/pusf", "m", MILLION),
    "NMFX": ("/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/nmfx/pusf", "m", MILLION),
    "NUGW": ("/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/nugw/pusf", "m", MILLION),
    "YBHA": ("/economy/grossdomesticproductgdp/timeseries/ybha/ukea", "m", MILLION),
    "EBAQ": ("/economy/grossdomesticproductgdp/timeseries/ebaq/ukea", ",000", THOUSAND),
}
UK_DEBT_SERIES = "BKPX"
UK_NET_DEBT_SERIES = "HF6W"
UK_INTEREST_SERIES = "NMYX"
UK_GDP_SERIES = "YBHA"
UK_POPULATION_SERIES = "EBAQ"
# Levels of government (--debt:LETTERS): central "f", local "m". "n" (every
# level but the central one) is local government too; there is no "p".
_LEVEL_DEBT_SERIES = {"f": "BKPW", "m": "MDYT"}
_LEVEL_INTEREST_SERIES = {"f": "NMFX", "m": "NUGW"}
_LEVEL_ALIASES = {"n": "m"}

# A quarter's population is dated this many days after the quarter begins:
# about its middle, where an interpolation of mid-year estimates centres it.
_QUARTER_MIDDLE_DAYS = 45

# First observation of each UK input other than the yields, to warn when
# ``--start`` is earlier (``uk_series_earliest`` adds the chosen yields).
UK_SERIES_EARLIEST = {
    # Interest: NMYX's first quarter (1946 Q1) starts here, though its first
    # four-quarter sum is dated 1946-12-31.
    "TTM UK Interest Paid (NMYX)": pd.Timestamp("1946-01-01"),
}


# ---------------------------------------------------------------------------
# Raw source access
# ---------------------------------------------------------------------------


def parse_iadb_csv(text: str) -> pd.DataFrame:
    """Parse a Bank of England database CSV in its ``CSVF=TN`` layout; date-indexed, one float column per series.

    The layout: ``DATE,CODE1,CODE2,...``, one row per date written
    ``01 Sep 2026``, an empty cell where a series has no value that day. A
    monthly series is dated by the month's last day. Anything else (its
    error page, the ``CSVF=CT`` layout) raises.
    """
    frame = pd.read_csv(io.StringIO(text), dtype=str)
    if frame.empty or frame.columns[0] != "DATE":
        raise ValueError(f"expected a CSV whose first column is DATE, got {list(frame.columns)[:3]}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], format="%d %b %Y"), name=DATE_COLUMN)
    values = frame.drop(columns="DATE").apply(pd.to_numeric, errors="coerce").set_axis(dates)
    if values.index.duplicated().any():
        raise ValueError("a date appears twice")
    return values.sort_index()


def iadb_series(code: str) -> pd.Series:
    """Return one Bank of England database series, every value from ``BOE_IADB_FROM`` on, without missing days.

    A request the database cannot serve (an unknown code, a start it
    refuses) is redirected to its error page, which a browser shows as a
    page: here it raises, naming the page, rather than being read as data.
    """
    params = {
        "csv.x": "yes",
        "Datefrom": BOE_IADB_FROM,
        "Dateto": "now",
        "SeriesCodes": code,
        "CSVF": "TN",
        "UsingCodes": "Y",
        "VPD": "Y",
        "VFD": "N",
    }
    response = uk_get(BOE_IADB_URL, params)
    if response.history or "ErrorPage" in (response.url or ""):
        raise ValueError(f"the Bank of England database redirected to {response.url} (an error page), not data")
    frame = parse_iadb_csv(response.text)
    if code not in frame.columns:
        raise ValueError(f"the Bank of England database returned no {code} column ({list(frame.columns)})")
    return frame[code].dropna().rename(code)


def _ons_date(row: dict, frequency: str) -> pd.Timestamp:
    """Return the last day of the period an ONS row describes: ``1975 Q1`` -> 1975-03-31, ``1975 MAR`` -> 1975-03-31."""
    text = str(row["date"]).strip()
    if frequency == "quarters":
        period = pd.Period(text.replace(" ", ""), freq="Q")
    elif frequency == "months":
        period = pd.Period(datetime.strptime(text.title(), "%Y %b"), freq="M")
    else:
        period = pd.Period(text, freq="Y")
    return period.end_time.normalize()


def parse_ons_series(data: dict, cdid: str, frequency: str) -> pd.Series:
    """Return one frequency ("months", "quarters" or "years") of an ONS time series as a date-indexed float series in its own unit.

    ONS writes values as strings, ``""`` where there is none; each row is
    dated by its period's last day (``_ons_date``). The unit the series'
    description gives must be the one measured (``ONS_SERIES``).
    """
    _path, unit, _multiplier = ONS_SERIES[cdid]
    found = str(data.get("description", {}).get("unit", "")).strip()
    if found != unit:
        raise ValueError(f"ONS {cdid}: expected unit {unit!r}, found {found!r}")
    rows = [row for row in data.get(frequency, []) if str(row.get("value", "")).strip()]
    values = pd.Series(
        [float(str(row["value"]).replace(",", "")) for row in rows],
        index=pd.DatetimeIndex([_ons_date(row, frequency) for row in rows], name=DATE_COLUMN),
        dtype=float,
        name=cdid,
    )
    if values.index.duplicated().any():
        raise ValueError(f"ONS {cdid}: a {frequency[:-1]} appears twice")
    return values.sort_index()


def ons_series(cdid: str, frequency: str) -> pd.Series:
    """Return one ONS series (``ONS_SERIES``) at ``frequency``, in pounds or persons; raises if it has no values there."""
    path, _unit, multiplier = ONS_SERIES[cdid]
    data = uk_get(ONS_TIMESERIES_URL.format(path=path)).json()
    values = parse_ons_series(data, cdid, frequency)
    if values.empty:
        raise ValueError(f"ONS {cdid} has no {frequency}")
    return values * multiplier


def _ons_stock(cdid: str) -> pd.Series:
    """Return a stock (a debt) at the end of each month when ONS has months, else of each quarter."""
    try:
        return ons_series(cdid, "months")
    except ValueError as exc:
        if "has no months" not in str(exc):
            raise
    return ons_series(cdid, "quarters")


# ---------------------------------------------------------------------------
# Per-curve fetchers
# ---------------------------------------------------------------------------


def uk_series_earliest(config: PlotConfig) -> dict[str, pd.Timestamp]:
    """Return the first observation of the UK inputs, by name, for the coverage warning.

    The yields are those of the chosen terms the UK has (and the spreads'
    terms), each from its baked history if it has one.
    """
    terms = set(config.yield_terms) | ({term for pair in config.spread_pairs for term in pair} if config.spreads else set())
    earliest: dict[str, pd.Timestamp] = {}
    for term, column in YIELD_TERMS.items():
        if term in terms and column in UK_YIELD_SERIES:
            history = _YIELD_HISTORY.get(column)
            source = "monthly average before the par yield" if history else UK_YIELD_SERIES[column]
            earliest[f"{column} Yield ({source})"] = pd.Timestamp(history[0][0]) if history else UK_YIELD_EARLIEST[column]
    named = {
        "UK National Debt (alone, before 1975)": EMBEDDED_UK_DEBT_HISTORY,
        "TTM Nominal GDP (UK, annual before 1955)": EMBEDDED_UK_GDP_HISTORY,
    }
    earliest.update({name: pd.Timestamp(rows[0][0]) for name, rows in named.items() if rows})
    return {**earliest, **UK_SERIES_EARLIEST}


def fetch_uk_yields(config: PlotConfig) -> pd.DataFrame:
    """Return the UK's gilt yields: each term's monthly history, then its daily par yield, date-indexed.

    The columns are the chosen ones the UK has (``config.fetched_yield_columns``),
    trimmed to the window. ``attrs[YIELD_HISTORY_ATTR]`` gives each column's
    last month of history (drawn as steps). A term whose daily series fails
    is drawn from its history alone, with a warning; one without a history
    is then left out. A chosen term the UK does not have is named.
    """
    chosen = [column for column in config.fetched_yield_columns if column in UK_YIELD_SERIES]
    absent = [column for column in config.yield_columns if column not in UK_YIELD_SERIES]
    if absent and len(absent) < len(config.yield_columns):  # none left: plotting._warn_no_yields
        print(f"  Note: no UK {', '.join(absent)} yield; the UK chart is drawn without it.")
    if not chosen:
        return pd.DataFrame()

    print("Fetching Bank of England gilt yields …")
    columns: dict[str, pd.Series] = {}
    history_through: dict[str, pd.Timestamp] = {}
    for column in chosen:
        code = UK_YIELD_SERIES[column]
        history = embedded_frame(_YIELD_HISTORY.get(column, []), column)
        try:
            daily = iadb_series(code)
        except Exception as exc:
            print(f"  Warning: Bank of England {column} par yield ({code}) unavailable ({exc}); "
                  + ("drawn from its monthly history alone." if history is not None else "drawn without it."))
            daily = _NO_VALUES
        monthly = history[column] if history is not None else _NO_VALUES
        if not daily.empty:
            monthly = monthly.loc[monthly.index < daily.index[0]]
        if monthly.empty and daily.empty:
            continue
        if not monthly.empty:
            history_through[column] = monthly.index[-1]
        columns[column] = pd.concat([monthly, daily]).rename(column)

    if not columns:
        return pd.DataFrame()
    frame = pd.concat(columns.values(), axis=1).sort_index()
    for column in frame.columns:
        daily_part = frame.index > history_through.get(column, pd.Timestamp.min)
        frame.loc[daily_part, column] = frame.loc[daily_part, column].ffill(limit=_DAILY_FILL_LIMIT)
    frame = frame.rename_axis(DATE_COLUMN).loc[config.start : config.end]
    frame.attrs[YIELD_HISTORY_ATTR] = history_through
    return frame


def fetch_uk_debt(config: PlotConfig) -> pd.DataFrame | None:
    """Return general government gross debt (BKPX), the national debt alone before it begins, and net debt (HF6W); monthly.

    The national debt (``UK_NATIONAL_DEBT_COLUMN``) is kept as it is, a
    different quantity: its years before BKPX's first month. If BKPX fails,
    the national debt alone, to 2017, with a warning; if HF6W fails, no
    net debt, with a warning.
    """
    if not config.include_debt:
        return None
    parts: list[pd.DataFrame] = []
    history = embedded_frame(EMBEDDED_UK_DEBT_HISTORY, UK_NATIONAL_DEBT_COLUMN)
    try:
        live = _ons_stock(UK_DEBT_SERIES).rename(UK_DEBT_COLUMN).to_frame()
    except Exception as exc:
        print(f"  Warning: ONS {UK_DEBT_SERIES} (UK debt) unavailable ({exc}); the national debt alone (to 2017).")
        live = None
    if live is not None:
        parts.append(live)
    if history is not None:
        parts.append(history if live is None else history.loc[history.index < live.index[0]])
    try:
        parts.append(_ons_stock(UK_NET_DEBT_SERIES).rename(UK_NET_DEBT_COLUMN).to_frame())
    except Exception as exc:
        print(f"  Warning: ONS {UK_NET_DEBT_SERIES} (UK net debt) unavailable ({exc}); not drawn.")
    return pd.concat(parts, axis=1).sort_index() if parts else None


def fetch_uk_gdp(config: PlotConfig) -> pd.DataFrame | None:
    """Return trailing-twelve-month nominal GDP: the annual history joined to the four-quarter sums of YBHA."""
    if not config.needs_gdp:
        return None
    history = embedded_frame(EMBEDDED_UK_GDP_HISTORY, GDP_COLUMN)
    try:
        live = ttm_sum(ons_series(UK_GDP_SERIES, "quarters")).rename(GDP_COLUMN).to_frame()
    except Exception as exc:
        print(f"  Warning: ONS {UK_GDP_SERIES} (UK GDP) unavailable ({exc}); using the baked history only (to 2016).")
        live = None
    return chain((history, live), GDP_COLUMN, annual_historical=False)


def fetch_uk_interest(config: PlotConfig) -> pd.DataFrame | None:
    """Return trailing-twelve-month general government interest (NMYX, four-quarter sums); None, with a warning, if ONS fails."""
    if not config.include_interest:
        return None
    try:
        return ttm_sum(ons_series(UK_INTEREST_SERIES, "quarters")).rename(UK_INTEREST_COLUMN).to_frame()
    except Exception as exc:
        print(f"  Warning: ONS {UK_INTEREST_SERIES} (UK interest) unavailable ({exc}); not drawn.")
        return None


def fetch_uk_components(config: PlotConfig) -> pd.DataFrame | None:
    """Return debt and TTM interest by level of government (--debt:LETTERS): central "f", local "m" (or "n").

    The UK accounts have no level between them (the devolved governments are
    inside central government): "p" is named and left out. "n", every level
    but the central one, is local government, drawn once if "m" is chosen too.
    A level whose source fails is left out, with a warning.
    """
    letters = config.components
    if "p" in letters:
        print("  Note: the UK accounts have no provincial or state level (the devolved governments are inside central government); not drawn.")
    if "n" in letters and "m" in letters:
        print("  Note: in the UK the non-central level is local government alone: drawn once, as local.")
    columns: list[pd.Series] = []
    for letter in COMPONENT_LETTERS:
        if letter not in letters or letter == "p" or (letter == "n" and "m" in letters):
            continue
        source = _LEVEL_ALIASES.get(letter, letter)
        for kind, include, series_of in (
            ("debt", config.include_debt, _LEVEL_DEBT_SERIES),
            ("interest", config.include_interest, _LEVEL_INTEREST_SERIES),
        ):
            if not include:
                continue
            cdid = series_of[source]
            try:
                values = _ons_stock(cdid) if kind == "debt" else ttm_sum(ons_series(cdid, "quarters"))
            except Exception as exc:
                print(f"  Warning: ONS {cdid} ({kind}, level {letter}) unavailable ({exc}); not drawn.")
                continue
            columns.append(values.rename(component_column(kind, letter)))
    return pd.concat(columns, axis=1).sort_index() if columns else None


def fetch_uk_population() -> pd.Series | None:
    """Return the UK population for -p: the baked annual estimates until EBAQ begins, then EBAQ (persons).

    EBAQ interpolates the mid-year estimates to quarters; each quarter is
    dated by its middle, the estimates by 30 June. The two agree where they
    meet (50,946 thousand in 1955 in both), so the baked years simply
    precede the live ones, as Canada's do.
    """
    history = embedded_frame(EMBEDDED_UK_POPULATION_HISTORY, POPULATION_COLUMN)
    try:
        quarterly = ons_series(UK_POPULATION_SERIES, "quarters")
        middles = quarterly.index.to_period("Q").start_time + pd.Timedelta(days=_QUARTER_MIDDLE_DAYS)
        live = quarterly.set_axis(pd.DatetimeIndex(middles, name=DATE_COLUMN))
    except Exception as exc:
        print(f"  Warning: ONS {UK_POPULATION_SERIES} (UK population) unavailable ({exc}); using the baked estimates only (to 2016).")
        live = None
    parts: list[pd.Series] = []
    if history is not None:
        early = history[POPULATION_COLUMN]
        parts.append(early if live is None or live.empty else early.loc[early.index < live.index.min()])
    if live is not None:
        parts.append(live)
    return pd.concat(parts).sort_index().rename(POPULATION_COLUMN) if parts else None
