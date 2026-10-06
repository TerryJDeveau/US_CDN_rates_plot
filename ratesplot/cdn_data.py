"""Bank of Canada / Statistics Canada data pipelines.

Each ``fetch_cdn_*`` macro function returns a date-indexed frame (or ``None``
when the curve is deselected or unavailable), each value dated by the day it
describes: a quarterly figure by the quarter's last day, although the sources
date it by the first (``frames.at_quarter_end``), and a year-end one by
31 December. ``align_cdn_macro`` then forward-fills those values onto the
daily yield dates for plotting.

Debt, GDP and interest are each a chain of sources, joined oldest to newest
with ``_splice_archived_series`` (the newer source is never modified):

* older history baked from terminated sources (``cdn_archive_data``): GDP and
  interest from 1926, debt of all governments from 1933;
* for debt and interest, the 1961-1994 archive (national balance sheets and
  government sector accounts); and
* the live modern table, authoritative from where it starts (1961 for GDP,
  1990 for debt and interest).

Before 1933 only federal debt is recorded. It is kept as its own series
(``CDN_FEDERAL_DEBT_COLUMN``), drawn as a separate curve, rather than joined
to the aggregate, of which it is only about half.

With levels of government chosen (``--debt:LETTERS``), ``fetch_cdn_components``
replaces the aggregate debt and interest: each level is chained the same way
from its own history (see there).
"""

from __future__ import annotations

import io
import warnings
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd

from .cdn_archive_data import (
    EMBEDDED_CDN_2Y_STAND_IN,
    EMBEDDED_CDN_5Y_STAND_IN,
    EMBEDDED_CDN_DEBT_HISTORY,
    EMBEDDED_CDN_DEBT_HISTORY_BY_LEVEL,
    EMBEDDED_CDN_EARLY_DEBT_HISTORY,
    EMBEDDED_CDN_EARLY_DEBT_HISTORY_BY_LEVEL,
    EMBEDDED_CDN_EARLY_INTEREST_HISTORY,
    EMBEDDED_CDN_EARLY_INTEREST_HISTORY_BY_LEVEL,
    EMBEDDED_CDN_FEDERAL_DEBT_HISTORY,
    EMBEDDED_CDN_GDP_HISTORY,
    EMBEDDED_CDN_INTEREST_HISTORY,
    EMBEDDED_CDN_INTEREST_HISTORY_BY_LEVEL,
    EMBEDDED_CDN_POPULATION_HISTORY,
)
from .cdn_hist_yields import build_cdn_hist_yields
from .config import (
    ARCHIVE_CALIBRATION_END_YEAR,
    ARCHIVE_SPLICE_YEARS,
    BOC_GROUP_URL,
    BOC_SERIES_URL,
    BOC_START_DATE,
    CANADIAN_SESSION,
    CDN_DEBT_COLUMN,
    CDN_FEDERAL_DEBT_COLUMN,
    CDN_INTEREST_COLUMN,
    DATE_COLUMN,
    GDP_COLUMN,
    HTTP_POST_TIMEOUT_SECONDS,
    MILLION,
    POPULATION_COLUMN,
    STATCAN_CDN_DEBT_FALLBACK_VECTOR,
    STATCAN_CDN_DEBT_TABLE,
    STATCAN_CDN_GDP_TABLE,
    STATCAN_CDN_INTEREST_TABLE,
    STATCAN_CDN_POPULATION_TABLE,
    STATCAN_TABLE_URL,
    STATCAN_TIMEOUT_SECONDS,
    STATCAN_WDS_URL,
    YIELD_COLUMNS,
    PlotConfig,
    component_column,
)
from .frames import at_quarter_end, normalize_date_column, rows_to_frame
from .http import canadian_get, fetch_fred_csv, parse_boc_csv

# Bank of Canada Valet series codes -> chart column names. The 3-month T-bill
# is not part of the benchmark group, so it is fetched as a separate series.
BOC_BENCHMARK_GROUP = "bond_yields_benchmark"
BOC_BENCHMARK_COLUMNS = {
    "BD.CDN.2YR.DQ.YLD": "2-Year",
    "BD.CDN.5YR.DQ.YLD": "5-Year",
    "BD.CDN.10YR.DQ.YLD": "10-Year",
    "BD.CDN.LONG.DQ.YLD": "30-Year",
}
BOC_3M_TBILL_SERIES = "V80691303"
# Baked term-band yields standing in for a benchmark before it begins.
_YIELD_STAND_INS = {"2-Year": EMBEDDED_CDN_2Y_STAND_IN, "5-Year": EMBEDDED_CDN_5Y_STAND_IN}
FRED_CDN_GDP_SERIES = "NGDPSAXDCCAQ"  # fallback if the StatCan GDP table fails

# First observation of each Canadian input, used to warn when ``--start`` is earlier.
CANADIAN_SERIES_EARLIEST = {
    "3-Month Yield (Bank of Canada historical table)": pd.Timestamp("1934-03-01"),
    "2-Year Yield (1-3 year average before 1982-06)": pd.Timestamp(EMBEDDED_CDN_2Y_STAND_IN[0][0]),
    "5-Year Yield (3-5 year average before 1980-11)": pd.Timestamp(EMBEDDED_CDN_5Y_STAND_IN[0][0]),
    "10-Year Yield (Bank of Canada historical table)": pd.Timestamp("1951-01-01"),
    "30-Year Yield / Over 10 Years (Bank of Canada historical table)": pd.Timestamp("1919-01-01"),
    # The macro series begin where their oldest baked source does, on its own
    # date (a quarterly one's first quarter starts there, though its value is
    # drawn from the quarter's end: see _dated_by_quarter_end).
    "Federal CDN Public Debt (alone, before 1933)": pd.Timestamp(EMBEDDED_CDN_FEDERAL_DEBT_HISTORY[0][0]),
    "Aggregate CDN Public Debt": pd.Timestamp(EMBEDDED_CDN_EARLY_DEBT_HISTORY[0][0]),
    "TTM Nominal GDP": pd.Timestamp(EMBEDDED_CDN_GDP_HISTORY[0][0]),
    "TTM Interest Payable": pd.Timestamp(EMBEDDED_CDN_EARLY_INTEREST_HISTORY[0][0]),
}


# ---------------------------------------------------------------------------
# Raw source access
# ---------------------------------------------------------------------------


def boc_valet_group(group: str, start: str = BOC_START_DATE) -> pd.DataFrame:
    """Download a Bank of Canada Valet observation group as a date-indexed frame."""
    response = canadian_get(BOC_GROUP_URL.format(group=group), params={"start_date": start})
    return parse_boc_csv(response.text).apply(pd.to_numeric, errors="coerce")


def boc_valet_series(series_code: str, start: str = BOC_START_DATE) -> pd.Series:
    """Download one Bank of Canada Valet series as a date-indexed numeric Series."""
    response = canadian_get(BOC_SERIES_URL.format(series_code=series_code), params={"start_date": start})
    frame = parse_boc_csv(response.text)
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce")


def statcan_zip_table_with_metadata(table_id: str) -> tuple[pd.DataFrame, str]:
    """Download a full Statistics Canada table ZIP and return ``(data, metadata_csv)``.

    The archive holds the data CSV plus a ``*_MetaData.csv`` describing the
    table's dimensions and members; the latter is returned as raw text.
    """
    print(f"Downloading StatCan table {table_id} (ZIP) …")
    response = canadian_get(STATCAN_TABLE_URL.format(table_id=table_id), timeout=STATCAN_TIMEOUT_SECONDS)

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        data_name = next(name for name in names if "MetaData" not in name)
        meta_name = next((name for name in names if "MetaData" in name), None)
        with archive.open(data_name) as csv_file:
            data = pd.read_csv(csv_file, low_memory=False)
        metadata = archive.read(meta_name).decode("utf-8-sig", errors="replace") if meta_name else ""
    return data, metadata


def statcan_zip_table(table_id: str) -> pd.DataFrame:
    """Download a full Statistics Canada table (CSV inside a ZIP) as a frame."""
    return statcan_zip_table_with_metadata(table_id)[0]


def statcan_wds_vectors(vector_ids: Iterable[str | int], *, latest_n: int = 250) -> pd.DataFrame:
    """Fetch the latest ``latest_n`` observations of StatCan WDS vectors.

    Returns one column per vector, named ``v<id>``. Vectors that the service
    reports as failed are skipped; if none succeed a ``RuntimeError`` is raised.
    """
    payload = [
        {"vectorId": int(str(vector).lstrip("vV")), "latestN": latest_n} for vector in vector_ids
    ]
    response = CANADIAN_SESSION.post(STATCAN_WDS_URL, json=payload, timeout=HTTP_POST_TIMEOUT_SECONDS)
    response.raise_for_status()

    frames: list[pd.DataFrame] = []
    for item in response.json():
        if item.get("status") != "SUCCESS":
            continue
        obj = item["object"]
        points = obj.get("vectorDataPoint", [])
        if not points:
            continue

        column = f"v{obj['vectorId']}"
        frame = pd.DataFrame(points)
        frame[DATE_COLUMN] = pd.to_datetime(frame["refPer"])
        frame[column] = pd.to_numeric(frame["value"], errors="coerce")
        frames.append(frame.set_index(DATE_COLUMN)[[column]])

    if not frames:
        raise RuntimeError("No data returned from WDS")
    return pd.concat(frames, axis=1).sort_index()


def _statcan_quarterly(table: pd.DataFrame, mask: pd.Series, column: str) -> pd.DataFrame:
    """Select ``mask`` rows of a StatCan table and return ``VALUE`` in dollars.

    ``REF_DATE`` is ``YYYY-MM`` for quarterly tables; the result is indexed by
    quarter start (``QS``), the source's own dating, like the baked histories,
    so all Canadian macro series share one calendar while they are joined.
    The fetchers then date the joined series by the quarter's end
    (``_dated_by_quarter_end``).
    """
    selected = table.loc[mask, ["REF_DATE", "VALUE"]]
    dates = pd.DatetimeIndex(pd.to_datetime(selected["REF_DATE"].astype(str) + "-01"), name=DATE_COLUMN)
    values = pd.to_numeric(selected["VALUE"], errors="coerce") * MILLION
    return values.rename(column).set_axis(dates).sort_index().to_frame().resample("QS").last()


# ---------------------------------------------------------------------------
# Archive splicing
# ---------------------------------------------------------------------------


def embedded_frame(rows: list[tuple[str, float]], column: str) -> pd.DataFrame | None:
    """Return a baked list as a date-indexed frame, or ``None`` when it is empty.

    A list is empty only if the bake has never filled it; callers
    then fall back to the remaining sources.
    """
    return rows_to_frame(rows, column) if rows else None


def _chain(sources: Iterable[pd.DataFrame | None], column: str, *, annual_historical: bool) -> pd.DataFrame | None:
    """Join ``sources`` (oldest first) pairwise with ``_splice_or_fallback``; missing ones are skipped.

    Each newer source is authoritative over the older ones where it exists.
    ``annual_historical`` is passed to every join (True when the older
    sources are year-end observations, as for debt).
    """
    result: pd.DataFrame | None = None
    for source in sources:
        result = _splice_or_fallback(result, source, column, annual_historical=annual_historical)
    return result


def _splice_archived_series(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    value_column: str,
    *,
    calibration_end_year: int = ARCHIVE_CALIBRATION_END_YEAR,
    transition_years: int = ARCHIVE_SPLICE_YEARS,
    annual_historical: bool = False,
) -> pd.DataFrame:
    """Join the baked archive to the live series with a smooth level bridge.

    The archived and live sources use related but not identical accounting
    definitions, so their levels differ by a roughly constant factor. That
    factor is estimated as the median ``current / historical`` ratio over the
    overlap window (from the live series' start through ``calibration_end_year``).
    The final ``transition_years`` of the archive are then scaled by a
    geometric ramp from 1 toward the ratio, so the archive meets the live
    series without a step. Live observations are never modified.

    Args:
        annual_historical: True when the archive holds year-end observations
            (debt); the live quarterly series is then compared year-end to
            year-end when estimating the ratio.
    """
    historical = historical[[value_column]].dropna().sort_index()
    current = current[[value_column]].dropna().sort_index()
    if historical.empty:
        return current
    if current.empty:
        return historical

    # --- 1. estimate the level ratio over the calibration overlap ------------
    overlap_start = max(historical.index.min(), current.index.min())
    overlap_end = min(historical.index.max(), current.index.max(), pd.Timestamp(year=calibration_end_year, month=12, day=31))

    hist_overlap = historical.loc[overlap_start:overlap_end, value_column]
    current_for_ratio = current.resample("YE").last() if annual_historical else current
    current_overlap = current_for_ratio.loc[overlap_start:overlap_end, value_column]
    if annual_historical:
        # Year-end dates differ slightly between sources; match on year instead.
        hist_overlap = hist_overlap.set_axis(hist_overlap.index.year)
        current_overlap = current_overlap.set_axis(current_overlap.index.year)

    ratios = (current_overlap / hist_overlap).replace([np.inf, -np.inf], np.nan).dropna()
    ratios = ratios[ratios > 0]
    scale_ratio = float(ratios.median()) if not ratios.empty else 1.0

    # --- 2. keep only archive rows that precede the live series --------------
    historical_part = historical.loc[historical.index < current.index.min()].copy()
    if historical_part.empty:
        return current

    # --- 3. geometric ramp over the last ``transition_years`` of the archive --
    # weight goes 0 -> 1 linearly by calendar year, so scale goes 1 -> ratio
    # geometrically. Scaling multiplicatively keeps the series positive.
    years = historical_part.index.year.to_numpy(dtype=float)
    transition_start_year = years.max() - transition_years + 1
    weights = np.clip((years - transition_start_year) / max(transition_years - 1, 1), 0.0, 1.0)
    historical_part[value_column] *= np.power(scale_ratio, weights)

    return pd.concat([historical_part, current]).sort_index()


def _splice_or_fallback(
    archive: pd.DataFrame | None, modern: pd.DataFrame | None, column: str, *, annual_historical: bool
) -> pd.DataFrame | None:
    """Combine whichever of the archive and live series are available."""
    if archive is None:
        return modern
    if modern is None:
        return archive
    return _splice_archived_series(archive, modern, column, annual_historical=annual_historical)


def _dated_by_quarter_end(series: pd.DataFrame | pd.Series | None) -> pd.DataFrame | pd.Series | None:
    """Return a joined macro series with each quarter-start date moved to its quarter's end (``frames.at_quarter_end``).

    Done after the joins, which compare the sources on their own dates; it
    moves every date within its year, so the joins are the same either way.
    Year-end debt stays on 31 December.
    """
    if series is None:
        return None
    return series.set_axis(at_quarter_end(series.index))


# ---------------------------------------------------------------------------
# Per-curve fetchers
# ---------------------------------------------------------------------------


def with_yield_stand_ins(historical: pd.DataFrame) -> pd.DataFrame:
    """Return ``historical`` with the 2- and 5-year columns extended back by their stand-ins.

    The Bank of Canada's 2- and 5-year benchmarks begin in 1982-06 and
    1980-11. Before that its average yields for the 1-3 and 3-5 year bands
    are used, the same kind of stand-in as the 5-10 and over-10 year bands
    that already supply the 10- and 30-year history. Over 1982-2000 the 1-3
    year band averages 0.08 points above the 2-year benchmark and the 3-5
    year band 0.00 points above the 5-year. Only months before a benchmark's
    first value are filled.
    """
    result = historical.copy()
    for column, rows in _YIELD_STAND_INS.items():
        stand_in = embedded_frame(rows, column)
        if stand_in is None or column not in result.columns:
            continue
        first_benchmark = result[column].first_valid_index()
        early = stand_in[column]
        if first_benchmark is not None:
            early = early.loc[early.index < first_benchmark]
        result[column] = result[column].combine_first(early)
    return result


def fetch_cdn_yields(config: PlotConfig) -> pd.DataFrame:
    """Return Canadian benchmark yields: transcribed history through 2000, live after.

    The 2- and 5-year history is extended back by stand-ins (see
    ``with_yield_stand_ins``). The result is date-indexed with the chosen
    yield columns (``config.yield_columns``) that Canada has, the
    ``YIELD_COLUMNS``, and those the spreads are taken from
    (``config.fetched_yield_columns``), trimmed to the configured window. A
    chosen term it does not have (--yields:7) is named, and the chart drawn
    without it.
    """
    chosen = [column for column in config.fetched_yield_columns if column in YIELD_COLUMNS]
    absent = [column for column in config.yield_columns if column not in YIELD_COLUMNS]
    if absent:
        print(f"  Note: no Canadian {', '.join(absent)} yield; the Canadian chart is drawn without it.")
    if not chosen:
        return pd.DataFrame()

    print("Fetching Bank of Canada benchmark yields …")
    yields = boc_valet_group(BOC_BENCHMARK_GROUP).rename(columns=BOC_BENCHMARK_COLUMNS)

    try:
        yields = yields.join(boc_valet_series(BOC_3M_TBILL_SERIES).rename("3-Month"), how="outer")
    except Exception as exc:
        print(f"  Warning: Bank of Canada 3-month T-bill ({BOC_3M_TBILL_SERIES}) unavailable ({exc}); drawn without it.")

    available = [column for column in YIELD_COLUMNS if column in yields.columns]
    # Short forward-fill bridges holidays and single missing prints only.
    yields = yields[available].dropna(how="all").ffill(limit=3)

    historical = with_yield_stand_ins(build_cdn_hist_yields())
    live_part = yields[yields.index > historical.index.max()]
    all_yields = pd.concat([historical, live_part]).sort_index().ffill(limit=3)
    return all_yields.loc[config.start : config.end, [column for column in all_yields.columns if column in chosen]]


def fetch_cdn_debt(config: PlotConfig) -> pd.DataFrame | None:
    """Return general-government gross debt, and federal debt alone before the aggregate begins.

    The aggregate (``CDN_DEBT_COLUMN``) chains the 1933-1975 history, the
    1961-1994 archive and the live table. ``CDN_FEDERAL_DEBT_COLUMN`` holds
    the federal-only years (1867-1932) as they are: a different quantity, so
    not scaled to the aggregate.
    """
    if not config.include_debt:
        return None

    try:
        table = statcan_zip_table(STATCAN_CDN_DEBT_TABLE)
        modern = _statcan_quarterly(table, table["Estimates"].eq("Debt"), CDN_DEBT_COLUMN)
    except Exception as exc:
        print(f"  Warning: StatCan debt table failed ({exc}); trying WDS vector fallback …")
        try:
            wds = statcan_wds_vectors([STATCAN_CDN_DEBT_FALLBACK_VECTOR], latest_n=250)
            modern = (wds[STATCAN_CDN_DEBT_FALLBACK_VECTOR] * MILLION).rename(CDN_DEBT_COLUMN).to_frame().resample("QS").last()
        except Exception as exc2:
            print(f"  Warning: WDS fallback failed ({exc2}); using the baked history only (to 1994).")
            modern = None

    aggregate = _chain(
        (
            embedded_frame(EMBEDDED_CDN_EARLY_DEBT_HISTORY, CDN_DEBT_COLUMN),
            embedded_frame(EMBEDDED_CDN_DEBT_HISTORY, CDN_DEBT_COLUMN),
            modern,
        ),
        CDN_DEBT_COLUMN,
        annual_historical=True,
    )
    federal_only = embedded_frame(EMBEDDED_CDN_FEDERAL_DEBT_HISTORY, CDN_FEDERAL_DEBT_COLUMN)
    parts = [_dated_by_quarter_end(part) for part in (aggregate, federal_only) if part is not None]
    return pd.concat(parts, axis=1).sort_index() if parts else None


def fetch_cdn_gdp(config: PlotConfig) -> pd.DataFrame | None:
    """Return trailing-twelve-month nominal GDP: the baked 1926-1994 history joined to the live table."""
    if not config.needs_gdp:
        return None

    history = embedded_frame(EMBEDDED_CDN_GDP_HISTORY, GDP_COLUMN)
    live = _fetch_live_cdn_gdp()
    if live is None and history is not None:
        print("  Warning: using the baked GDP history only (to 1994).")
    return _dated_by_quarter_end(_chain((history, live), GDP_COLUMN, annual_historical=False))


def _fetch_live_cdn_gdp() -> pd.DataFrame | None:
    """Return live TTM nominal GDP (mean of four SAAR quarters), or None if every source fails."""
    try:
        table = statcan_zip_table(STATCAN_CDN_GDP_TABLE)
        mask = (
            table["Estimates"].str.contains("Gross domestic product at market prices", case=False, na=False)
            & table["Prices"].str.contains("Current prices", case=False, na=False)
            & table["Seasonal adjustment"].str.contains("Seasonally adjusted at annual rates", case=False, na=False)
        )
        saar = _statcan_quarterly(table, mask, "GDP_SAAR")
    except Exception as exc:
        print(f"  Warning: StatCan GDP table failed ({exc}); trying FRED fallback …")
        try:
            fred = fetch_fred_csv(FRED_CDN_GDP_SERIES).set_index(DATE_COLUMN)
            saar = (fred.iloc[:, 0] * MILLION).rename("GDP_SAAR").to_frame().resample("QS").last()
        except Exception as exc2:
            print(f"  Warning: FRED GDP fallback failed ({exc2}).")
            return None

    # A SAAR value is already an annual rate, so the TTM level is the 4-quarter mean.
    return saar["GDP_SAAR"].rolling(4).mean().rename(GDP_COLUMN).to_frame()


def fetch_cdn_interest(config: PlotConfig) -> pd.DataFrame | None:
    """Return TTM public-debt interest: the 1926 history, the 1961-1994 archive and the live GFS table."""
    if not config.include_interest:
        return None

    try:
        table = statcan_zip_table(STATCAN_CDN_INTEREST_TABLE)
        mask = table["Government sectors"].eq("Consolidated government") & table[
            "Statement of government operations and balance sheet"
        ].eq("Interest")
        quarterly = _statcan_quarterly(table, mask, "Interest_q")
        # The GFS table reports actual quarterly flows (not annualised), so TTM is a 4-quarter sum.
        modern = quarterly["Interest_q"].rolling(4).sum().rename(CDN_INTEREST_COLUMN).to_frame()
    except Exception as exc:
        print(f"  Warning: StatCan interest table failed ({exc}); using the baked history only (to 1994).")
        modern = None

    return _dated_by_quarter_end(
        _chain(
            (
                embedded_frame(EMBEDDED_CDN_EARLY_INTEREST_HISTORY, CDN_INTEREST_COLUMN),
                embedded_frame(EMBEDDED_CDN_INTEREST_HISTORY, CDN_INTEREST_COLUMN),
                modern,
            ),
            CDN_INTEREST_COLUMN,
            annual_historical=False,
        )
    )


# ---------------------------------------------------------------------------
# Components: debt and interest by level of government (--debt:LETTERS)
# ---------------------------------------------------------------------------

# Levels in the live GFS table 10-10-0015 (its fifth sector, the CPP and QPP,
# is not a level of government and is left out, as from "n").
_GFS_SECTORS = {"f": "Federal government", "p": "Provincial and territorial government", "m": "Local government"}
_GFS_ITEM_COLUMN = "Statement of government operations and balance sheet"
# A level's gross debt: all its liabilities except equity, the GFS definition
# that the aggregate (36-10-0467 "Debt") also follows.
_GFS_DEBT_ITEMS = (
    "Special drawing rights (SDRs), liabilities",
    "Currency and deposits, liabilities",
    "Debt securities, liabilities",
    "Loans, liabilities",
    "Insurance and pension schemes, liabilities",
    "Other accounts payable, liabilities",
)


def _with_non_federal(levels: dict[str, pd.DataFrame | None], column: str) -> dict[str, pd.DataFrame | None]:
    """Add "n" (non-federal) = provincial + local, on the dates both exist."""
    provincial, local = levels.get("p"), levels.get("m")
    non_federal = None
    if provincial is not None and local is not None:
        non_federal = (provincial[column] + local[column]).dropna().rename(column).to_frame()
    return {**levels, "n": non_federal}


def _embedded_levels(by_level: dict[str, list[tuple[str, float]]], column: str) -> dict[str, pd.DataFrame | None]:
    """Return a baked by-level dictionary as frames keyed "f", "p", "m", plus "n"."""
    return _with_non_federal({key: embedded_frame(rows, column) for key, rows in by_level.items()}, column)


def _gfs_levels(table: pd.DataFrame, items: tuple[str, ...], column: str) -> dict[str, pd.DataFrame | None]:
    """Return each level's quarterly sum of ``items`` from the GFS table, in dollars, plus "n"."""
    levels: dict[str, pd.DataFrame | None] = {}
    for key, sector in _GFS_SECTORS.items():
        in_sector = table["Government sectors"].eq(sector)
        parts = [_statcan_quarterly(table, in_sector & table[_GFS_ITEM_COLUMN].eq(item), column)[column] for item in items]
        levels[key] = sum(parts[1:], parts[0]).rename(column).to_frame()
    return _with_non_federal(levels, column)


def fetch_cdn_components(config: PlotConfig) -> pd.DataFrame | None:
    """Return debt and/or interest for each level in ``config.components``, one ``component_column`` each.

    Each level is chained like its aggregate, oldest first, "n" being
    provincial + local within each source before the joins:

    * debt: *Historical Statistics of Canada* (federal from 1867, provincial
      and local from 1933), the 1961-1994 balance sheets, then the live GFS
      table 10-10-0015 (from 1990), where a level's debt is all its
      liabilities except equity;
    * interest: the 1968-SNA accounts (from 1926), the 1961-1994 sector
      accounts, then the live GFS interest (quarterly flows, TTM = four-quarter sum).

    The levels are each government's own figures. The consolidated aggregate
    nets out what one government owes another, so the levels need not add
    up to it. Returns None when neither curve is selected.
    """
    letters = config.components
    if not letters or not (config.include_debt or config.include_interest):
        return None
    try:
        table = statcan_zip_table(STATCAN_CDN_INTEREST_TABLE)
    except Exception as exc:
        print(f"  Warning: StatCan GFS table failed ({exc}); levels of government shown only to 1994.")
        table = None

    columns: list[pd.Series] = []
    if config.include_debt:
        early = _embedded_levels(EMBEDDED_CDN_EARLY_DEBT_HISTORY_BY_LEVEL, CDN_DEBT_COLUMN)
        archive = _embedded_levels(EMBEDDED_CDN_DEBT_HISTORY_BY_LEVEL, CDN_DEBT_COLUMN)
        live = _gfs_levels(table, _GFS_DEBT_ITEMS, CDN_DEBT_COLUMN) if table is not None else {}
        for letter in letters:
            chained = _chain(
                (early.get(letter), archive.get(letter), live.get(letter)), CDN_DEBT_COLUMN, annual_historical=True
            )
            if chained is not None:
                columns.append(chained[CDN_DEBT_COLUMN].rename(component_column("debt", letter)))
    if config.include_interest:
        early = _embedded_levels(EMBEDDED_CDN_EARLY_INTEREST_HISTORY_BY_LEVEL, CDN_INTEREST_COLUMN)
        archive = _embedded_levels(EMBEDDED_CDN_INTEREST_HISTORY_BY_LEVEL, CDN_INTEREST_COLUMN)
        live: dict[str, pd.DataFrame | None] = {}
        if table is not None:
            # The GFS table reports actual quarterly flows, so TTM is a 4-quarter sum.
            quarterly = _gfs_levels(table, ("Interest",), CDN_INTEREST_COLUMN)
            live = {key: frame[CDN_INTEREST_COLUMN].rolling(4).sum().to_frame() for key, frame in quarterly.items() if frame is not None}
        for letter in letters:
            chained = _chain(
                (early.get(letter), archive.get(letter), live.get(letter)), CDN_INTEREST_COLUMN, annual_historical=False
            )
            if chained is not None:
                columns.append(chained[CDN_INTEREST_COLUMN].rename(component_column("interest", letter)))
    # Each series is moved before they are put side by side: interest dated
    # 1 October and debt dated 31 December would otherwise share a row twice.
    return pd.concat([_dated_by_quarter_end(column) for column in columns], axis=1).sort_index() if columns else None


def fetch_cdn_population() -> pd.Series | None:
    """Return the population of Canada for -p: baked annual estimates until the live quarterly table begins.

    The estimates (as at 1 June, 1867-1977) and the live table (quarterly
    from 1946, first of the quarter) differ by about 0.2 % where both exist,
    mostly the month between their reference dates, too little to need a
    join: the baked years simply precede the live ones. Persons.
    """
    history = embedded_frame(EMBEDDED_CDN_POPULATION_HISTORY, POPULATION_COLUMN)
    try:
        table = statcan_zip_table(STATCAN_CDN_POPULATION_TABLE)
        selected = table.loc[table["GEO"].eq("Canada"), ["REF_DATE", "VALUE", "SCALAR_FACTOR"]]
        if set(selected["SCALAR_FACTOR"].astype(str).str.strip()) != {"units"} or selected["REF_DATE"].duplicated().any():
            raise ValueError("expected one row per quarter, in persons")
        dates = pd.DatetimeIndex(pd.to_datetime(selected["REF_DATE"]), name=DATE_COLUMN)
        live = pd.Series(pd.to_numeric(selected["VALUE"], errors="coerce").to_numpy(), index=dates).sort_index().dropna()
    except Exception as exc:
        print(f"  Warning: StatCan population table failed ({exc}); using the baked estimates only (to 1977).")
        live = None

    parts: list[pd.Series] = []
    if history is not None:
        early = history[POPULATION_COLUMN]
        parts.append(early if live is None or live.empty else early.loc[early.index < live.index.min()])
    if live is not None:
        parts.append(live)
    return pd.concat(parts).sort_index().rename(POPULATION_COLUMN) if parts else None


def align_cdn_macro(
    yields_all: pd.DataFrame, series: Iterable[pd.DataFrame | None], config: PlotConfig
) -> pd.DataFrame:
    """Forward-fill the macro series (frames of one or more columns; None = absent) onto the yield dates.

    Returns a frame with a ``DATE`` column plus one column per available series.
    When yields are deselected a daily calendar over the window is used instead.

    Where the window starts before the first yield, the window start and the
    macro series' own dates up to that first yield are added, so a curve
    that begins before any yield (federal debt from 1867, GDP from 1926; the
    oldest yield is 1919) is drawn from the window start, as it would be
    with yields deselected.

    Federal debt alone is kept only until the aggregate begins: forward
    filling would otherwise carry its 1932 value on for ever.
    """
    if yields_all.empty:
        plot_index = pd.date_range(config.start, config.end, freq="D")
    else:
        plot_index = pd.DatetimeIndex(yields_all.index)
    parts = [part for part in series if part is not None]
    if not parts:
        return pd.DataFrame({DATE_COLUMN: plot_index})

    with warnings.catch_warnings():
        # pandas warns about concatenating frames with differing index dtypes/names.
        warnings.simplefilter("ignore", FutureWarning)
        macro = pd.concat(parts, axis=1, sort=False)
        first_yield = plot_index.min()
        if not yields_all.empty and config.start < first_yield:
            before_yields = macro.index[(macro.index > config.start) & (macro.index < first_yield)]
            plot_index = plot_index.union(before_yields).union(pd.DatetimeIndex([config.start]))
        aligned = (
            macro.reindex(macro.index.union(plot_index)).sort_index().ffill().reindex(plot_index)
        )
    if CDN_FEDERAL_DEBT_COLUMN in aligned.columns and CDN_DEBT_COLUMN in aligned.columns:
        aligned.loc[aligned[CDN_DEBT_COLUMN].notna(), CDN_FEDERAL_DEBT_COLUMN] = np.nan
    return normalize_date_column(aligned.rename_axis(DATE_COLUMN).reset_index())
