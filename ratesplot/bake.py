"""The bake: download terminated StatCan tables and embed the rows.

Run by ``tools/bake_archives.py``, a maintenance step rather than an option of
the program (it was ``--bake-archives`` until 2026-09-27). The archived tables
never change, so their relevant rows are written once into
``ratesplot/cdn_archive_data.py`` as Python literals. Normal runs then read
that module instead of downloading a dozen multi-megabyte files. Bake again
only when an extraction changes or a source is added.

Two layers of history are baked, each joined to the next at run time (see
``cdn_data``), never here, so the module holds the sources' own values:

* 1961-1994: the national balance sheets (debt) and government sector
  accounts (interest), which the live tables take over from 1990.
* Before 1961: the 1968-SNA national accounts (GDP and interest, annual from
  1926, quarterly from 1947 or 1950), *Historical Statistics of Canada* (debt: federal
  from 1867, all governments from 1933), the Bank of Canada's term-band yields
  (stand-ins for the 2- and 5-year benchmarks before they begin) and the
  annual population estimates from 1867.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import py_compile
import re
import tempfile
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .cdn_data import statcan_zip_table, statcan_zip_table_with_metadata
from .config import (
    ARCHIVE_BEGIN_MARKER,
    ARCHIVE_CALIBRATION_END_YEAR,
    ARCHIVE_END_MARKER,
    ARCHIVED_CDN_FEDERAL_DEBT_TABLE,
    ARCHIVED_CDN_FEDERAL_DEBT_URL,
    ARCHIVED_CDN_INTEREST_TABLE,
    ARCHIVED_CDN_INTEREST_URL,
    ARCHIVED_CDN_LOCAL_DEBT_TABLE,
    ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE,
    ARCHIVED_CDN_PROV_LOCAL_DEBT_URL,
    ARCHIVED_CDN_PROVINCIAL_DEBT_TABLE,
    BOC_FINANCIAL_MARKET_TABLE,
    CDN_DEBT_COLUMN,
    CDN_FEDERAL_DEBT_COLUMN,
    CDN_INTEREST_COLUMN,
    CENSUS_HIST_FIN_URL,
    CENSUS_OLD_ESTIMATES_URL,
    CENSUS_TABLES_URL,
    DATE_COLUMN,
    GDP_COLUMN,
    HISTORICAL_CDN_GDP_ANNUAL_TABLE,
    HISTORICAL_CDN_GDP_QUARTERLY_TABLE,
    HISTORICAL_CDN_INTEREST_ANNUAL_TABLE,
    HISTORICAL_CDN_INTEREST_QUARTERLY_TABLE,
    HISTORICAL_CDN_POPULATION_TABLE,
    HSC_FEDERAL_DEBT_SERIES,
    HSC_LOCAL_DEBT_SERIES,
    HSC_PROVINCIAL_DEBT_SERIES,
    HSC_SECTION_H_URL,
    MILLION,
    POPULATION_COLUMN,
    STATCAN_TABLE_URL,
    THOUSAND,
    US_ARCHIVE_BEGIN_MARKER,
    US_ARCHIVE_END_MARKER,
)
from .http import canadian_get, get_if_published

# Balance-sheet members that constitute marketable debt securities.
DEBT_SECURITY_CATEGORIES = ("Short-term paper", "Bonds")

# 1968-SNA member names (tables 36-10-0137/0142/0150/0177).
_SAAR = "Seasonally adjusted at annual rates"
_GDP_MARKET_PRICES = "Gross domestic product (GDP) at market prices"
_INTEREST_ON_PUBLIC_DEBT = "Interest on the public debt"
_TOTAL_GOVERNMENT = "Total government"
# Levels of government for the component lines, keyed like the --debt:fpm letters.
# 36-10-0245 (1961-1994): federal + provincial + local = total.
_LEVELS_1961 = {"f": "Federal government", "p": "Provincial government", "m": "Local government"}
# 1968 SNA (1926-1994): hospitals are a level of their own from 1961 and part of
# provincial government in the later accounts, so they are counted as provincial.
_LEVELS_1968_SNA = {
    "f": ("Federal government",),
    "p": ("Provincial government", "Hospital"),
    "m": ("Local government",),
}

# Historical Statistics of Canada, section H: the series summed as debt
# securities outstanding, the nearest match to the balance sheets'
# liability-side "Short-term paper" + "Bonds" (at 1961-1975, where both
# exist, the balance sheets are 3.0-7.6 % higher; ``cdn_data`` joins them).
#   federal:    H37 unmatured bonded debt less sinking funds + H38 treasury bills
#   provincial: H384 funded debt less sinking funds + H385 short-term treasury bills
#   local:      H400 funded debt less sinking funds
# Figures are at the fiscal year end nearest 31 December of the year named
# (federal and provincial) or at 31 December (local), so all are stamped at
# year end like the balance sheets. Millions of dollars.
_HSC_FEDERAL_SECURITIES = ("H37", "H38")
_HSC_PROVINCIAL_SECURITIES = ("H384", "H385")
_HSC_LOCAL_SECURITIES = ("H400",)
# Treasury bills (H38, H385) did not exist in the earliest years; a blank there
# means none. The funded/bonded debt series must be present.
_HSC_MAY_BE_BLANK = ("H38", "H385")

# Bank of Canada term bands (table 10-10-0122) standing in for the 2- and
# 5-year benchmarks, used only for months before the benchmark begins.
_YIELD_BAND_FOR_BENCHMARK = {
    "2-Year": (
        "Government of Canada marketable bonds, average yield: 1-3 year",
        "Selected Government of Canada benchmark bond yields: 2 year",
    ),
    "5-Year": (
        "Government of Canada marketable bonds, average yield: 3-5 year",
        "Selected Government of Canada benchmark bond yields: 5 year",
    ),
}


# ---------------------------------------------------------------------------
# StatCan metadata parsing
# ---------------------------------------------------------------------------


def _metadata_section(metadata_csv: str, header_prefix: str) -> list[dict[str, str]]:
    """Return the rows of the metadata section whose header starts with ``header_prefix``.

    A StatCan ``*_MetaData.csv`` is several independent CSV tables separated by
    blank lines, each with its own header row.
    """
    lines = metadata_csv.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(header_prefix):
            section: list[str] = []
            for body_line in lines[index:]:
                if not body_line.strip():
                    break
                section.append(body_line)
            return list(csv.DictReader(io.StringIO("\n".join(section))))
    raise ValueError(f"Metadata section starting with {header_prefix!r} not found")


def liability_member_ids(metadata_csv: str, member_names: tuple[str, ...]) -> set[int]:
    """Return the ``Categories`` member IDs of ``member_names`` under *Liabilities*.

    The national balance sheet lists the same instrument names (e.g. "Bonds")
    twice: once under *Total financial assets* (securities the government
    holds) and once under *Liabilities* (securities it has issued). Only the
    liability side is debt, and the two are distinguishable only by the
    member's parent in the metadata, which is what this resolves.
    """
    dimensions = _metadata_section(metadata_csv, '"Dimension ID","Dimension name"')
    categories_dimension = next(
        row["Dimension ID"] for row in dimensions if row["Dimension name"] == "Categories"
    )
    members = [
        row
        for row in _metadata_section(metadata_csv, '"Dimension ID","Member Name"')
        if row["Dimension ID"] == categories_dimension
    ]
    liabilities_id = next(
        row["Member ID"] for row in members if row["Member Name"] == "Liabilities"
    )
    ids = {
        int(row["Member ID"])
        for row in members
        if row["Parent Member ID"] == liabilities_id and row["Member Name"] in member_names
    }
    if len(ids) != len(member_names):
        raise ValueError(
            f"Expected liability members {member_names} in metadata, found IDs {sorted(ids)}"
        )
    return ids


# ---------------------------------------------------------------------------
# Row extraction from the archived tables
# ---------------------------------------------------------------------------


def _require_columns(table: pd.DataFrame, required: set[str], what: str) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Archived {what} table is missing columns: {sorted(missing)}")


def _extract_archived_cdn_interest(table: pd.DataFrame, level: str = "Total government") -> pd.DataFrame:
    """Extract TTM interest on public debt for one level of government from table 36-10-0245.

    The source is a seasonally adjusted *annual-rate* flow, so each quarter's
    value is divided by four before the trailing four-quarter sum. Its levels
    are total, federal, provincial and local government and the two pension
    plans; federal + provincial + local equals the total in every quarter.
    """
    _require_columns(table, {"Seasonal adjustment", "Levels of government", "Sector accounts", "REF_DATE", "VALUE"}, "interest")

    mask = (
        table["Seasonal adjustment"].eq("Seasonally adjusted at annual rates")
        & table["Levels of government"].eq(level)
        # The member was renamed at some point; accept either spelling.
        & table["Sector accounts"].isin(["Interest on public debt", "Interest on the public debt"])
    )
    selected = table.loc[mask, ["REF_DATE", "VALUE"]]
    ttm = _ttm_from_saar_quarters(selected)
    return ttm.rename(CDN_INTEREST_COLUMN).rename_axis(DATE_COLUMN).to_frame()


def _ttm_from_saar_quarters(selected: pd.DataFrame) -> pd.Series:
    """Return the trailing-twelve-month level of a quarterly SAAR series in dollars.

    ``selected`` holds ``REF_DATE`` (``YYYY-MM``, the quarter's first month)
    and ``VALUE`` in millions at annual rates. A quarter's actual amount is a
    quarter of its annual rate, so TTM is the sum of four quarters divided by
    four. Stamped at the start of the fourth quarter, like the live series.
    """
    dates = pd.PeriodIndex(selected["REF_DATE"].astype(str), freq="Q").to_timestamp(how="start")
    quarterly = (pd.to_numeric(selected["VALUE"], errors="coerce") * MILLION / 4.0).set_axis(dates).sort_index()
    return quarterly.rolling(4, min_periods=4).sum().dropna()


def _extract_archived_cdn_debt(table: pd.DataFrame, metadata_csv: str) -> pd.DataFrame:
    """Extract book-value debt securities (liability side) from an archived balance sheet.

    Rows are selected by member ID (the last component of ``COORDINATE``) so
    that only the *Liabilities* "Short-term paper" and "Bonds" are counted, not
    the identically named asset holdings. Annual values are stamped at year end.
    """
    _require_columns(table, {"Valuation", "Categories", "COORDINATE", "REF_DATE", "VALUE"}, "debt")

    member_ids = liability_member_ids(metadata_csv, DEBT_SECURITY_CATEGORIES)
    category_member = table["COORDINATE"].astype(str).str.rsplit(".", n=1).str[-1].astype(int)
    mask = (
        table["Valuation"].eq("Book value")
        & table["Categories"].isin(DEBT_SECURITY_CATEGORIES)
        & category_member.isin(member_ids)
    )
    selected = table.loc[mask, ["REF_DATE", "VALUE"]]
    dates = pd.to_datetime(selected["REF_DATE"].astype(str) + "-12-31", errors="coerce")
    values = pd.to_numeric(selected["VALUE"], errors="coerce") * MILLION
    annual = values.groupby(dates.to_numpy()).sum()
    return annual.rename(CDN_DEBT_COLUMN).rename_axis(DATE_COLUMN).to_frame().sort_index()


# ---------------------------------------------------------------------------
# Row extraction: history before 1961
# ---------------------------------------------------------------------------


def _one_row_per_period(table: pd.DataFrame, mask: pd.Series, what: str, scalar_factor: str) -> pd.DataFrame:
    """Return the ``REF_DATE``/``VALUE`` rows ``mask`` selects, checked to be one per period.

    Raises if nothing is selected, if a period is selected twice (the filter
    left a dimension unpinned) or if the values are not in ``scalar_factor``
    units ("millions", "thousands", "units"; StatCan pads some with spaces).
    """
    selected = table.loc[mask, ["REF_DATE", "VALUE", "SCALAR_FACTOR"]]
    if selected.empty:
        raise RuntimeError(f"{what}: the filter selected no rows")
    repeated = selected["REF_DATE"].duplicated()
    if repeated.any():
        raise RuntimeError(
            f"{what}: {int(repeated.sum())} periods selected more than once, "
            f"e.g. {selected.loc[repeated, 'REF_DATE'].iloc[0]}"
        )
    scales = set(selected["SCALAR_FACTOR"].astype(str).str.strip())
    if scales != {scalar_factor}:
        raise RuntimeError(f"{what}: expected values in {scalar_factor}, found {sorted(scales)}")
    return selected[["REF_DATE", "VALUE"]]


def _annual_as_ttm(selected: pd.DataFrame) -> pd.Series:
    """Return calendar-year totals (``REF_DATE`` = ``YYYY``, millions) in dollars, stamped 1 October.

    A calendar year is the TTM ending with its fourth quarter, and TTM values
    are stamped at the start of that quarter, so the annual figure goes where
    the quarterly TTM series would put it.
    """
    dates = pd.to_datetime(selected["REF_DATE"].astype(str) + "-10-01")
    return (pd.to_numeric(selected["VALUE"], errors="coerce") * MILLION).set_axis(dates).sort_index()


def _annual_then_quarterly_ttm(annual: pd.DataFrame, quarterly: pd.DataFrame) -> pd.Series:
    """Use annual totals up to the first quarterly TTM value, and the quarterly TTM from there.

    Both come from the same 1968-SNA accounts: in the years both cover, the
    mean of the four SAAR quarters matches the annual figure within 1 %. The
    quarterly tables start in 1947, but a series may be blank at first
    (interest is ".." until 1950), so the switch happens where the quarterly
    TTM actually begins.
    """
    quarterly_ttm = _ttm_from_saar_quarters(quarterly)
    expected = pd.date_range(quarterly_ttm.index.min(), quarterly_ttm.index.max(), freq="QS")
    if len(quarterly_ttm) != len(expected):
        missing = expected.difference(quarterly_ttm.index)
        raise RuntimeError(f"Quarterly TTM has gaps after it begins, e.g. {missing[0]:%Y-%m-%d}")
    annual_ttm = _annual_as_ttm(annual).dropna()
    return pd.concat([annual_ttm.loc[annual_ttm.index < quarterly_ttm.index.min()], quarterly_ttm]).sort_index()


def _extract_historical_gdp(annual_table: pd.DataFrame, quarterly_table: pd.DataFrame) -> pd.DataFrame:
    """Extract TTM nominal GDP at market prices, 1926 onwards, from the 1968-SNA tables.

    Annual 36-10-0150 until 1946, then quarterly current-price SAAR 36-10-0137.
    """
    _require_columns(annual_table, {"Estimates", "REF_DATE", "VALUE", "SCALAR_FACTOR"}, "annual GDP")
    _require_columns(
        quarterly_table,
        {"Expenditure-based estimates", "Prices", "Seasonal adjustment", "REF_DATE", "VALUE", "SCALAR_FACTOR"},
        "quarterly GDP",
    )
    annual = _one_row_per_period(
        annual_table, annual_table["Estimates"].eq(_GDP_MARKET_PRICES), "annual GDP", "millions"
    )
    quarterly = _one_row_per_period(
        quarterly_table,
        quarterly_table["Expenditure-based estimates"].eq(_GDP_MARKET_PRICES)
        & quarterly_table["Prices"].eq("Current prices")
        & quarterly_table["Seasonal adjustment"].eq(_SAAR),
        "quarterly GDP",
        "millions",
    )
    ttm = _annual_then_quarterly_ttm(annual, quarterly)
    return ttm.rename(GDP_COLUMN).rename_axis(DATE_COLUMN).to_frame()


def _levels_summed(table: pd.DataFrame, base_mask: pd.Series, levels: tuple[str, ...], what: str) -> pd.DataFrame:
    """Return ``REF_DATE``/``VALUE`` rows for the sum of ``levels``, each checked to be one row per period.

    With one level this is just that level's rows. A level after the first
    that is blank in some periods counts as zero there (in the 1968-SNA
    tables hospitals are a level of their own only from 1961); the first
    level must be present, or the sum is blank.
    """
    parts = [
        _one_row_per_period(table, base_mask & table["Levels of government"].eq(level), f"{what} ({level})", "millions")
        for level in levels
    ]
    if len(parts) == 1:
        return parts[0]
    values = pd.concat(
        [pd.to_numeric(part.set_index("REF_DATE")["VALUE"], errors="coerce") for part in parts], axis=1
    )
    summed = values.iloc[:, 0] + values.iloc[:, 1:].fillna(0.0).sum(axis=1)
    return summed.rename("VALUE").rename_axis("REF_DATE").reset_index()


def _extract_historical_interest(
    annual_table: pd.DataFrame, quarterly_table: pd.DataFrame, levels: tuple[str, ...] = (_TOTAL_GOVERNMENT,)
) -> pd.DataFrame:
    """Extract TTM interest on the public debt, 1926 onwards (1968 SNA), summed over ``levels``.

    Annual 36-10-0177 until 1949, then quarterly SAAR 36-10-0142 (blank for
    1947-1949, so its first four-quarter total is 1950). "Total
    government" is federal, provincial, local, hospitals and the pension plans,
    as in the 1961-1994 archive (36-10-0245).
    """
    _require_columns(annual_table, {"Levels of government", "Estimates", "REF_DATE", "VALUE", "SCALAR_FACTOR"}, "annual interest")
    _require_columns(
        quarterly_table,
        {"Seasonal adjustment", "Levels of government", "Estimates", "REF_DATE", "VALUE", "SCALAR_FACTOR"},
        "quarterly interest",
    )
    annual = _levels_summed(
        annual_table, annual_table["Estimates"].eq(_INTEREST_ON_PUBLIC_DEBT), levels, "annual interest"
    )
    quarterly = _levels_summed(
        quarterly_table,
        quarterly_table["Seasonal adjustment"].eq(_SAAR) & quarterly_table["Estimates"].eq(_INTEREST_ON_PUBLIC_DEBT),
        levels,
        "quarterly interest",
    )
    ttm = _annual_then_quarterly_ttm(annual, quarterly)
    return ttm.rename(CDN_INTEREST_COLUMN).rename_axis(DATE_COLUMN).to_frame()


def parse_hsc_table(text: str, series: str) -> pd.DataFrame:
    """Parse a *Historical Statistics of Canada* CSV into a year-indexed frame of its series.

    The files are laid out for print: title and heading rows, one row of
    series numbers (e.g. ``,,35,36,37,…``) that fixes which column holds which
    series, then one row per year with the year in the second column, blank
    spacer rows and footnotes. Cells carry thousands separators; unnumbered
    columns hold footnote markers; a dash means "none". Anything that is not
    a number becomes NaN. Columns are named ``H<number>``.

    ``series`` is the file's name, e.g. ``"H35_51"``; the title must name it
    and the numbers row must run from its first to its last series, so a
    moved or reworked file is rejected rather than misread.
    """
    first, last = (number.lstrip("H") for number in series.split("_"))
    rows = list(csv.reader(io.StringIO(text)))
    title = f"Series H{first}-{last}."
    if not any(title in cell for row in rows[:3] for cell in row):
        raise RuntimeError(f"{series}: no title naming {title!r}; the file may have moved or changed")
    if not any("millions of dollars" in cell for row in rows[:3] for cell in row):
        raise RuntimeError(f"{series}: values are no longer stated in millions of dollars")

    numbers_row = next(
        (row for row in rows if first in (cell.strip() for cell in row) and last in (cell.strip() for cell in row)),
        None,
    )
    if numbers_row is None:
        raise RuntimeError(f"{series}: no row of series numbers {first} … {last}")
    columns = {index: f"H{cell.strip()}" for index, cell in enumerate(numbers_row) if cell.strip().isdigit()}

    data: dict[int, dict[str, float]] = {}
    for row in rows:
        year_text = row[1].strip() if len(row) > 1 else ""
        if not (year_text.isdigit() and len(year_text) == 4):
            continue
        if int(year_text) in data:
            raise RuntimeError(f"{series}: year {year_text} appears twice")
        values: dict[str, float] = {}
        for index, name in columns.items():
            cell = row[index].replace(",", "").strip() if index < len(row) else ""
            try:
                values[name] = float(cell)
            except ValueError:
                values[name] = float("nan")
        data[int(year_text)] = values
    return pd.DataFrame.from_dict(data, orient="index").sort_index()


def _hsc_securities(table: pd.DataFrame, series_names: tuple[str, ...]) -> pd.Series:
    """Sum the named series per year, in dollars; NaN in a year where a required series is blank."""
    parts = [table[name].fillna(0.0) if name in _HSC_MAY_BE_BLANK else table[name] for name in series_names]
    return sum(parts[1:], parts[0]) * MILLION


def _extract_hsc_debt(federal: pd.DataFrame, provincial: pd.DataFrame, local: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(aggregate, federal_only)`` debt securities at year end from the HSC tables.

    The aggregate (federal + provincial + local) exists only where all three
    do: 1933, 1937, 1939, 1941, 1943, then every year 1945-1975. Federal debt
    alone goes back to 1867; it is returned for the years before the
    aggregate begins, as a separate series (it is only about half of the
    aggregate: 53 % in 1933).
    """
    federal_debt = _hsc_securities(federal, _HSC_FEDERAL_SECURITIES)
    aggregate = (
        federal_debt
        + _hsc_securities(provincial, _HSC_PROVINCIAL_SECURITIES)
        + _hsc_securities(local, _HSC_LOCAL_SECURITIES)
    ).dropna()
    if aggregate.empty:
        raise RuntimeError("Historical Statistics of Canada: no year has federal, provincial and local debt")
    federal_only = federal_debt.loc[federal_debt.index < aggregate.index.min()]
    if federal_only.isna().any():
        raise RuntimeError(f"H35_51: federal debt blank in {list(federal_only.index[federal_only.isna()])}")

    return _at_year_end(aggregate, CDN_DEBT_COLUMN), _at_year_end(federal_only, CDN_FEDERAL_DEBT_COLUMN)


def _at_year_end(series: pd.Series, column: str) -> pd.DataFrame:
    """Turn a year-indexed series into a frame stamped at 31 December of each year."""
    dates = pd.to_datetime([f"{year}-12-31" for year in series.index])
    return series.set_axis(dates).rename(column).rename_axis(DATE_COLUMN).to_frame()


def _extract_hsc_debt_by_level(
    federal: pd.DataFrame, provincial: pd.DataFrame, local: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Return each level's debt securities at year end from the HSC tables, keyed "f", "p", "m".

    The same series as the aggregate (``_HSC_*_SECURITIES``), kept apart:
    federal for every year 1867-1975, provincial and local for the years
    they exist (1933, 1937, 1939, 1941, 1943, then 1945-1975).
    """
    levels = {
        "f": _hsc_securities(federal, _HSC_FEDERAL_SECURITIES),
        "p": _hsc_securities(provincial, _HSC_PROVINCIAL_SECURITIES),
        "m": _hsc_securities(local, _HSC_LOCAL_SECURITIES),
    }
    if levels["f"].isna().any():
        raise RuntimeError(f"H35_51: federal debt blank in {list(levels['f'].index[levels['f'].isna()])}")
    return {key: _at_year_end(series.dropna(), CDN_DEBT_COLUMN) for key, series in levels.items()}


def _extract_historical_population(table: pd.DataFrame) -> pd.DataFrame:
    """Extract the annual population estimates (as at 1 June) from table 17-10-0063, in persons."""
    _require_columns(table, {"Estimates", "REF_DATE", "VALUE", "SCALAR_FACTOR"}, "population")
    selected = _one_row_per_period(
        table, table["Estimates"].eq("Estimated population"), "population", "thousands"
    )
    dates = pd.to_datetime(selected["REF_DATE"].astype(str) + "-06-01")
    values = (pd.to_numeric(selected["VALUE"], errors="coerce") * 1_000).set_axis(dates).sort_index()
    return values.rename(POPULATION_COLUMN).rename_axis(DATE_COLUMN).to_frame()


def _extract_yield_stand_ins(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return each term band's monthly yields for the months before its benchmark begins.

    Keyed by chart column ("2-Year", "5-Year"). ``REF_DATE`` is ``YYYY-MM``;
    rows are stamped on the first of the month, like the transcribed history.
    """
    _require_columns(table, {"Rates", "REF_DATE", "VALUE", "SCALAR_FACTOR"}, "financial market")
    has_value = table["VALUE"].notna()
    result: dict[str, pd.DataFrame] = {}
    for column, (band, benchmark) in _YIELD_BAND_FOR_BENCHMARK.items():
        band_rows = _one_row_per_period(table, table["Rates"].eq(band) & has_value, band, "units")
        benchmark_rows = _one_row_per_period(table, table["Rates"].eq(benchmark) & has_value, benchmark, "units")
        first_benchmark = benchmark_rows["REF_DATE"].astype(str).min()
        before = band_rows.loc[band_rows["REF_DATE"].astype(str) < first_benchmark]
        dates = pd.to_datetime(before["REF_DATE"].astype(str) + "-01")
        values = pd.to_numeric(before["VALUE"], errors="coerce").set_axis(dates).sort_index()
        result[column] = values.rename(column).rename_axis(DATE_COLUMN).to_frame()
    return result


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------


def _format_embedded_rows(frame: pd.DataFrame, value_column: str) -> str:
    """Format a date-indexed frame as indented ``("YYYY-MM-DD", value),`` lines."""
    # ``.10g`` is more significant figures than any source value carries, so
    # nothing is rounded away.
    return "\n".join(
        f'    ("{index:%Y-%m-%d}", {float(value):.10g}),'
        for index, value in frame[value_column].dropna().items()
    )


def _write_validated(source_path: Path, source_text: str) -> None:
    """Replace ``source_path`` with ``source_text`` only if the new text compiles.

    The text goes to a sibling temporary file first; ``py_compile`` validates
    it; only then is it atomically moved over the real module. A failed bake
    therefore never leaves an import-breaking data module behind.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py",
            prefix=source_path.stem + "_bake_", dir=source_path.parent, delete=False,
        ) as temp_file:
            temp_file.write(source_text)
            temp_path = Path(temp_file.name)
        py_compile.compile(str(temp_path), doraise=True)
        os.replace(temp_path, source_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _replace_generated_block(source_path: Path, begin_marker: str, end_marker: str, block: str) -> None:
    """Put ``block`` (markers included) in place of the generated block of ``source_path``.

    The rest of the module (its docstring) is kept; a module without the
    markers gets the block appended. Written by ``_write_validated``.
    """
    source_text = source_path.read_text(encoding="utf-8")
    # Anchor both markers as complete lines so only the generated block is replaced.
    block_pattern = re.compile(
        rf"(?ms)^[ \t]*{re.escape(begin_marker)}[ \t]*\r?\n.*?^[ \t]*{re.escape(end_marker)}[ \t]*$"
    )
    if block_pattern.search(source_text):
        # A callable replacement avoids re.sub interpreting backslashes in the data.
        source_text = block_pattern.sub(lambda _match: block, source_text, count=1)
    else:
        source_text = source_text.rstrip() + "\n\n" + block + "\n"
    _write_validated(source_path, source_text)


def _hsc_url(series: str) -> str:
    return HSC_SECTION_H_URL.format(series=series)


def _download_hsc_table(series: str) -> pd.DataFrame:
    """Download and parse one *Historical Statistics of Canada* section H CSV."""
    print(f"Downloading Historical Statistics of Canada {series} …")
    response = canadian_get(_hsc_url(series))
    # The files are Windows-1252 (the "none" dash is byte 0x97).
    return parse_hsc_table(response.content.decode("cp1252", errors="replace"), series)


def _require_rows(frame: pd.DataFrame, what: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(f"{what} extraction returned no observations; data module not modified.")
    return frame


def _embedded_list(name: str, comment: str, frame: pd.DataFrame, value_column: str) -> str:
    """Return one ``NAME = [...]`` assignment, preceded by its ``#`` comment lines."""
    comment_lines = "\n".join(f"# {line}" for line in comment.splitlines())
    return f"{comment_lines}\n{name} = [\n{_format_embedded_rows(frame, value_column)}\n]\n"


def _embedded_level_dict(name: str, comment: str, frames: dict[str, pd.DataFrame], value_column: str) -> str:
    """Return one ``NAME = {"f": [...], ...}`` assignment (a list per level), preceded by its comment."""
    comment_lines = "\n".join(f"# {line}" for line in comment.splitlines())
    body = "".join(
        f'    "{key}": [\n{textwrap.indent(_format_embedded_rows(frame, value_column), "    ")}\n    ],\n'
        for key, frame in frames.items()
    )
    return f"{comment_lines}\n{name} = {{\n{body}}}\n"


def _through_calibration_end(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep rows up to ``ARCHIVE_CALIBRATION_END_YEAR``; raise if a level is left empty."""
    kept = {key: frame.loc[frame.index.year <= ARCHIVE_CALIBRATION_END_YEAR] for key, frame in frames.items()}
    for key, frame in kept.items():
        _require_rows(frame, f"Level {key!r} after the calibration cutoff")
    return kept


def _table_url(table_id: str) -> str:
    return STATCAN_TABLE_URL.format(table_id=table_id)


def bake_canadian_archives(source_path: Path | None = None) -> None:
    """Download the archived tables and embed the needed rows as Python literals.

    Writes into ``ratesplot/cdn_archive_data.py`` unless ``source_path`` is given.
    Rows after ``ARCHIVE_CALIBRATION_END_YEAR`` are dropped from the series
    that overlap the live tables: the live series is authoritative from 1990,
    and the history only needs to overlap it (or the 1961-1994 archive) long
    enough for the splice ratio to be estimated. Nothing is written unless
    every source is extracted successfully.
    """
    source_path = source_path or Path(__file__).resolve().with_name("cdn_archive_data.py")

    print("Downloading archived Canadian interest data …")
    archived_interest_table = statcan_zip_table(ARCHIVED_CDN_INTEREST_TABLE)
    interest = _require_rows(_extract_archived_cdn_interest(archived_interest_table), "Archived Canadian interest")
    interest_by_level = {
        key: _require_rows(_extract_archived_cdn_interest(archived_interest_table, level), f"Archived {level} interest")
        for key, level in _LEVELS_1961.items()
    }

    print("Downloading archived Canadian federal debt data …")
    federal = _require_rows(
        _extract_archived_cdn_debt(*statcan_zip_table_with_metadata(ARCHIVED_CDN_FEDERAL_DEBT_TABLE)),
        "Archived Canadian federal debt",
    )

    print("Downloading archived Canadian provincial/local debt data …")
    provincial_local = _require_rows(
        _extract_archived_cdn_debt(*statcan_zip_table_with_metadata(ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE)),
        "Archived Canadian provincial/local debt",
    )

    print("Downloading archived Canadian provincial and local debt data, each level alone …")
    debt_by_level = {
        "f": federal,
        "p": _require_rows(
            _extract_archived_cdn_debt(*statcan_zip_table_with_metadata(ARCHIVED_CDN_PROVINCIAL_DEBT_TABLE)),
            "Archived Canadian provincial debt",
        ),
        "m": _require_rows(
            _extract_archived_cdn_debt(*statcan_zip_table_with_metadata(ARCHIVED_CDN_LOCAL_DEBT_TABLE)),
            "Archived Canadian local debt",
        ),
    }

    # Aggregate public debt = federal + provincial/local securities outstanding.
    debt = federal.add(provincial_local, fill_value=0.0)

    debt = debt.loc[debt.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    interest = interest.loc[interest.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    if debt.empty or interest.empty:
        raise RuntimeError("Archived Canadian data became empty after the calibration cutoff; data module not modified.")
    debt_by_level = _through_calibration_end(debt_by_level)
    interest_by_level = _through_calibration_end(interest_by_level)

    print("Downloading 1968-SNA national accounts (GDP and government interest from 1926) …")
    gdp_history = _require_rows(
        _extract_historical_gdp(
            statcan_zip_table(HISTORICAL_CDN_GDP_ANNUAL_TABLE), statcan_zip_table(HISTORICAL_CDN_GDP_QUARTERLY_TABLE)
        ),
        "Historical GDP",
    )
    annual_interest_table = statcan_zip_table(HISTORICAL_CDN_INTEREST_ANNUAL_TABLE)
    quarterly_interest_table = statcan_zip_table(HISTORICAL_CDN_INTEREST_QUARTERLY_TABLE)
    early_interest = _require_rows(
        _extract_historical_interest(annual_interest_table, quarterly_interest_table), "Historical interest"
    )
    early_interest_by_level = _through_calibration_end(
        {
            key: _require_rows(
                _extract_historical_interest(annual_interest_table, quarterly_interest_table, levels),
                f"Historical interest ({', '.join(levels)})",
            )
            for key, levels in _LEVELS_1968_SNA.items()
        }
    )
    gdp_history = gdp_history.loc[gdp_history.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    early_interest = early_interest.loc[early_interest.index.year <= ARCHIVE_CALIBRATION_END_YEAR]

    hsc_tables = [
        _download_hsc_table(series)
        for series in (HSC_FEDERAL_DEBT_SERIES, HSC_PROVINCIAL_DEBT_SERIES, HSC_LOCAL_DEBT_SERIES)
    ]
    early_debt, federal_only_debt = _extract_hsc_debt(*hsc_tables)
    early_debt_by_level = _extract_hsc_debt_by_level(*hsc_tables)

    print("Downloading historical population estimates …")
    population = _require_rows(
        _extract_historical_population(statcan_zip_table(HISTORICAL_CDN_POPULATION_TABLE)), "Historical population"
    )

    print("Downloading Bank of Canada financial market statistics …")
    stand_ins = _extract_yield_stand_ins(statcan_zip_table(BOC_FINANCIAL_MARKET_TABLE))
    for column, frame in stand_ins.items():
        _require_rows(frame, f"{column} stand-in yields")

    hsc_urls = "\n".join(
        _hsc_url(series) for series in (HSC_FEDERAL_DEBT_SERIES, HSC_PROVINCIAL_DEBT_SERIES, HSC_LOCAL_DEBT_SERIES)
    )
    lists = [
        _embedded_list(
            "EMBEDDED_CDN_INTEREST_HISTORY",
            f"Interest on the public debt, all governments, TTM, 1961-{ARCHIVE_CALIBRATION_END_YEAR}:\n"
            f"{ARCHIVED_CDN_INTEREST_URL}",
            interest,
            CDN_INTEREST_COLUMN,
        ),
        _embedded_list(
            "EMBEDDED_CDN_DEBT_HISTORY",
            "Liability-side 'Short-term paper' + 'Bonds' at book value, federal plus\n"
            f"provincial/local, at year end, 1961-{ARCHIVE_CALIBRATION_END_YEAR}:\n"
            f"{ARCHIVED_CDN_FEDERAL_DEBT_URL}\n{ARCHIVED_CDN_PROV_LOCAL_DEBT_URL}",
            debt,
            CDN_DEBT_COLUMN,
        ),
        _embedded_list(
            "EMBEDDED_CDN_GDP_HISTORY",
            f"GDP at market prices, TTM, 1968 SNA: calendar years (stamped 1 October) until\n"
            f"the quarterly SAAR series has four quarters, then that:\n"
            f"{_table_url(HISTORICAL_CDN_GDP_ANNUAL_TABLE)}\n{_table_url(HISTORICAL_CDN_GDP_QUARTERLY_TABLE)}",
            gdp_history,
            GDP_COLUMN,
        ),
        _embedded_list(
            "EMBEDDED_CDN_EARLY_INTEREST_HISTORY",
            "Interest on the public debt, all governments, TTM, 1968 SNA: calendar years\n"
            "(stamped 1 October) until the quarterly SAAR series has four quarters, then that:\n"
            f"{_table_url(HISTORICAL_CDN_INTEREST_ANNUAL_TABLE)}\n{_table_url(HISTORICAL_CDN_INTEREST_QUARTERLY_TABLE)}",
            early_interest,
            CDN_INTEREST_COLUMN,
        ),
        _embedded_list(
            "EMBEDDED_CDN_EARLY_DEBT_HISTORY",
            "Debt securities of all governments at year end, Historical Statistics of Canada:\n"
            "federal H37+H38, provincial H384+H385, local H400 (1933, 1937, 1939, 1941, 1943,\n"
            f"then yearly):\n{hsc_urls}",
            early_debt,
            CDN_DEBT_COLUMN,
        ),
        _embedded_list(
            "EMBEDDED_CDN_FEDERAL_DEBT_HISTORY",
            "Federal debt securities alone (H37+H38) at year end, for the years before any\n"
            f"provincial and local figures:\n{_hsc_url(HSC_FEDERAL_DEBT_SERIES)}",
            federal_only_debt,
            CDN_FEDERAL_DEBT_COLUMN,
        ),
        _embedded_list(
            "EMBEDDED_CDN_POPULATION_HISTORY",
            f"Estimated population of Canada at 1 June:\n{_table_url(HISTORICAL_CDN_POPULATION_TABLE)}",
            population,
            POPULATION_COLUMN,
        ),
    ]
    for column, frame in stand_ins.items():
        band, benchmark = _YIELD_BAND_FOR_BENCHMARK[column]
        lists.append(
            _embedded_list(
                f"EMBEDDED_CDN_{column.split('-')[0]}Y_STAND_IN",
                f"{band}, monthly (last Wednesday),\n"
                f"for the months before '{benchmark}' begins:\n{_table_url(BOC_FINANCIAL_MARKET_TABLE)}",
                frame,
                column,
            )
        )
    # Each level of government alone, for the component lines (--debt:fnpm).
    # Keys: "f" federal, "p" provincial, "m" local (municipal); "n" is p + m,
    # formed at run time. Same definitions and sources as the aggregates above.
    lists += [
        _embedded_level_dict(
            "EMBEDDED_CDN_DEBT_HISTORY_BY_LEVEL",
            f"EMBEDDED_CDN_DEBT_HISTORY by level, 1961-{ARCHIVE_CALIBRATION_END_YEAR}: federal, provincial and\n"
            "local balance sheets (provincial + local = the combined table exactly):\n"
            f"{ARCHIVED_CDN_FEDERAL_DEBT_URL}\n{_table_url(ARCHIVED_CDN_PROVINCIAL_DEBT_TABLE)}\n"
            f"{_table_url(ARCHIVED_CDN_LOCAL_DEBT_TABLE)}",
            debt_by_level,
            CDN_DEBT_COLUMN,
        ),
        _embedded_level_dict(
            "EMBEDDED_CDN_INTEREST_HISTORY_BY_LEVEL",
            f"EMBEDDED_CDN_INTEREST_HISTORY by level (federal, provincial, local government), 1961-"
            f"{ARCHIVE_CALIBRATION_END_YEAR}:\n{ARCHIVED_CDN_INTEREST_URL}",
            interest_by_level,
            CDN_INTEREST_COLUMN,
        ),
        _embedded_level_dict(
            "EMBEDDED_CDN_EARLY_DEBT_HISTORY_BY_LEVEL",
            "Historical Statistics of Canada debt securities by level: federal H37+H38 (1867-1975),\n"
            f"provincial H384+H385 and local H400 (1933-1975, yearly from 1945):\n{hsc_urls}",
            early_debt_by_level,
            CDN_DEBT_COLUMN,
        ),
        _embedded_level_dict(
            "EMBEDDED_CDN_EARLY_INTEREST_HISTORY_BY_LEVEL",
            "EMBEDDED_CDN_EARLY_INTEREST_HISTORY by level (1968 SNA): federal; provincial plus\n"
            "hospitals (a level of their own there from 1961, part of provincial in later accounts);\n"
            f"local:\n{_table_url(HISTORICAL_CDN_INTEREST_ANNUAL_TABLE)}\n{_table_url(HISTORICAL_CDN_INTEREST_QUARTERLY_TABLE)}",
            early_interest_by_level,
            CDN_INTEREST_COLUMN,
        ),
    ]

    block = (
        f"{ARCHIVE_BEGIN_MARKER}\n"
        f"# Generated from archived StatCan tables on {dt.date.today():%Y-%m-%d}.\n"
        f"# These are historical observations only, in each source's own terms; the\n"
        f"# live modern series remain responsible for current data, and cdn_data joins\n"
        f"# the layers at run time.\n\n"
        + "\n".join(lists)
        + f"{ARCHIVE_END_MARKER}"
    )
    _replace_generated_block(source_path, ARCHIVE_BEGIN_MARKER, ARCHIVE_END_MARKER, block)

    print(f"Embedded archived Canadian data into {source_path}")
    summary = [
        ("Interest 1961-", interest),
        ("Debt 1961-", debt),
        ("GDP history", gdp_history),
        ("Interest history", early_interest),
        ("Debt history (all governments)", early_debt),
        ("Federal debt alone", federal_only_debt),
        ("Population", population),
        *((f"{column} stand-in", frame) for column, frame in stand_ins.items()),
    ]
    for name, by_level in (
        ("Debt 1961- by level", debt_by_level),
        ("Interest 1961- by level", interest_by_level),
        ("Debt history by level", early_debt_by_level),
        ("Interest history by level", early_interest_by_level),
    ):
        summary += [(f"{name} [{key}]", frame) for key, frame in by_level.items()]
    for label, frame in summary:
        print(f"  {label + ':':32} {len(frame):4,} rows, {frame.index.min():%Y-%m-%d} to {frame.index.max():%Y-%m-%d}")


# ---------------------------------------------------------------------------
# U.S. state and local debt apart (Census Bureau)
# ---------------------------------------------------------------------------

# The historical database: one table per level of government, one row per
# finance item, one column per fiscal year ("STA-2008", "Loc-2008",
# "S&L-2008"), $ thousands; negative values flag figures not published.
_HIST_FIN_TABLES = {"p": "4_State Governments", "m": "5_Local Governments", "sl": "3_State & Local Government Total"}
_HIST_FIN_TOTAL_DEBT = "C1203"  # "Total Debt Outstanding" at the end of the fiscal year
_HIST_FIN_LAST_YEAR = 2008
# The yearly estimates, fixed-width: state code (columns 1-2; "00" is the
# whole country), level (3: "1" state and local, "2" state, "3" local), item
# (5-7), amount in $ thousands (9-20). Total debt outstanding is long-term
# debt for private purposes (44T, folded into 49U from 2022), other long-term
# debt (49U) and short-term debt at the year end (64V): the historical
# database's C1203 is this sum (2005-2008 agree to 0.7 %, the state share to
# 0.13 points). Before 2005 the estimates files are not on this basis, so the
# historical database is used through 2008.
_CENSUS_LEVEL_CODES = {"sl": "1", "p": "2", "m": "3"}
_CENSUS_DEBT_ITEMS = ("44T", "49U", "64V")
# The Census year is the fiscal year ending by 30 June (as most states' do),
# so each year's figures are stamped then.
_CENSUS_FISCAL_YEAR_END = (6, 30)
# State + local may differ from the published total by rounding only.
_CENSUS_SUM_TOLERANCE = 1e-4


def _census_fiscal_year_end(year: int) -> pd.Timestamp:
    month, day = _CENSUS_FISCAL_YEAR_END
    return pd.Timestamp(year=year, month=month, day=day)


def _check_levels_add_up(levels: dict[str, pd.Series], what: str) -> None:
    """Raise unless state + local equals the state-and-local total in every year both are given."""
    total = levels["sl"]
    gap = ((levels["p"] + levels["m"] - total).abs() / total).dropna()
    if gap.empty or gap.max() > _CENSUS_SUM_TOLERANCE:
        raise RuntimeError(f"{what}: state + local differs from the total by up to {gap.max():.4%}; data module not modified.")


def _extract_census_hist_fin(mdb_path: Path) -> dict[str, pd.Series]:
    """Return total debt outstanding by fiscal year (dollars) for "p" state, "m" local and "sl" both.

    Reads the Census historical database, an Access file, through the
    Microsoft Access ODBC driver (installed with Microsoft Office);
    ``pyodbc`` is needed only for this bake.
    """
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError(
            "The U.S. bake needs pyodbc (pip install pyodbc) and the Microsoft Access ODBC driver."
        ) from exc
    connection = pyodbc.connect(f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};Dbq={mdb_path};")
    levels: dict[str, pd.Series] = {}
    try:
        for key, table in _HIST_FIN_TABLES.items():
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM [{table}] WHERE [SAS Var] = ?", _HIST_FIN_TOTAL_DEBT)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            if len(rows) != 1:
                raise RuntimeError(f"Census {table}: {len(rows)} rows for {_HIST_FIN_TOTAL_DEBT}, expected 1")
            record = dict(zip(columns, rows[0]))
            if str(record["Name"]).strip() != "Total Debt Outstanding" or str(record["State Code"]).strip() != "00":
                raise RuntimeError(f"Census {table}: {_HIST_FIN_TOTAL_DEBT} is {record['Name']!r}, not the national total debt")
            by_year = {
                int(column[-4:]): float(record[column]) * THOUSAND
                for column in columns
                if re.fullmatch(r"[A-Za-z&]+-\d{4}", column) and record[column] is not None and record[column] >= 0
            }
            levels[key] = pd.Series(by_year).sort_index()
    finally:
        connection.close()
    _check_levels_add_up(levels, "Census historical database")
    return levels


def _parse_census_estimates(text: str, year: int) -> dict[str, float]:
    """Return the country's total debt outstanding (dollars) at each level from one year's estimates."""
    totals = {key: 0.0 for key in _CENSUS_LEVEL_CODES}
    seen: set[tuple[str, str]] = set()
    codes = {code: key for key, code in _CENSUS_LEVEL_CODES.items()}
    for line in text.splitlines():
        if len(line) < 20 or line[0:2] != "00" or line[2] not in codes or line[4:7] not in _CENSUS_DEBT_ITEMS:
            continue
        key = codes[line[2]]
        totals[key] += int(line[8:20]) * THOUSAND
        seen.add((key, line[4:7]))
    missing = [key for key in totals if (key, "49U") not in seen]
    if missing:
        raise RuntimeError(f"Census {year} estimates: no long-term debt (49U) for {missing}")
    return totals


def _census_estimates_text(year: int) -> str | None:
    """Download one fiscal year's estimates by state and type of government; None if not published."""
    short = f"{year % 100:02d}"
    if year <= 2010:
        candidates = [f"{CENSUS_OLD_ESTIMATES_URL}{short}statetypepu.zip", f"{CENSUS_OLD_ESTIMATES_URL}{short}statetypepu.txt"]
    else:
        # The ZIP's name and folder vary by year, so it is found in the listing.
        candidates = []
        for folder in (f"{CENSUS_TABLES_URL}{year}/", f"{CENSUS_TABLES_URL}{year}/summary-tables/"):
            listing = get_if_published(folder)
            found = re.search(r'href="([^"]*Individual_Unit_[Ff]iles?\.zip)"', listing.text) if listing else None
            if found:
                candidates.append(folder + found.group(1))
                break
    for url in candidates:
        response = get_if_published(url)
        if response is None:
            continue
        content = response.content
        if content[:2] == b"PK":
            archive = zipfile.ZipFile(io.BytesIO(content))
            names = [name for name in archive.namelist() if re.search(r"statetype\w*pu\w*\.txt$", name, re.IGNORECASE)]
            if len(names) != 1:
                raise RuntimeError(f"Census {year}: {url} holds {len(names)} state-by-type estimates files, expected 1")
            content = archive.read(names[0])
        return content.decode("latin-1")
    return None


def _census_summary_table(year: int) -> pd.DataFrame | None:
    """Download one fiscal year's summary Table 1 (state and local finances by level); None if absent.

    Needed only for the years without an estimates file (2011 is ``.xls``,
    which pandas reads with ``xlrd``; 2016 is ``.xlsx``).
    """
    for folder in (f"{CENSUS_TABLES_URL}{year}/summary-tables/", f"{CENSUS_TABLES_URL}{year}/"):
        listing = get_if_published(folder)
        found = re.search(rf'href="({year % 100:02d}slsstab1a\.xlsx?)"', listing.text) if listing else None
        if found:
            response = get_if_published(folder + found.group(1))
            if response is not None:
                return pd.read_excel(io.BytesIO(response.content), header=None)
    return None


def _parse_census_summary_table(table: pd.DataFrame, year: int) -> dict[str, float]:
    """Return the country's total debt outstanding (dollars) at each level from a summary Table 1.

    The layout shifts between years, so the table is read by its labels: the
    "Description" header, the level names under it ("State & local",
    "State", "Local"), the "amount" columns (the others are coefficients of
    variation) and the "Debt outstanding" row. The United States block comes
    first, before the states'. Amounts are $ thousands.
    """
    cells = table.astype(str).apply(lambda column: column.str.strip())
    header = [(row, column) for row, column in zip(*np.nonzero(cells.eq("Description").to_numpy()))]
    if not header:
        raise RuntimeError(f"Census {year} summary table: no Description header")
    header_row, label_column = header[0]
    levels_row, units_row = cells.iloc[header_row + 1], cells.iloc[header_row + 3]
    columns: dict[str, int] = {}
    for key, name in (("sl", "State & local"), ("p", "State"), ("m", "Local")):
        for column in range(label_column + 1, label_column + 7):
            if levels_row.iloc[column] == name and units_row.iloc[column].lower().startswith("amount"):
                columns[key] = column
                break
        else:
            raise RuntimeError(f"Census {year} summary table: no United States {name} amount column")
    debt_rows = cells.index[cells.iloc[:, label_column].str.lower().eq("debt outstanding")]
    if len(debt_rows) != 1:
        raise RuntimeError(f"Census {year} summary table: {len(debt_rows)} 'Debt outstanding' rows, expected 1")
    row = table.loc[debt_rows[0]]
    return {key: float(row.iloc[column]) * THOUSAND for key, column in columns.items()}


def bake_us_archives(source_path: Path | None = None) -> None:
    """Download the Census counts of state and of local government debt and embed them.

    Writes into ``ratesplot/us_archive_data.py`` unless ``source_path`` is
    given. The Federal Reserve's quarterly state-and-local debt (FRED
    SLGSDODNS) is not split by level; ``us_data`` splits it by these yearly
    shares. Every fiscal year from 1902 (annual from 1952) to the latest the
    Census has published. The years without an estimates file (2011, 2016)
    are read from that year's summary Table 1 instead; a year with neither
    is left out, and ``us_data`` interpolates across it. Needs ``pyodbc``
    with the Microsoft Access ODBC driver, and ``xlrd`` for 2011's table.
    """
    source_path = source_path or Path(__file__).resolve().with_name("us_archive_data.py")

    print("Downloading the Census historical government finance database …")
    archive = zipfile.ZipFile(io.BytesIO(canadian_get(CENSUS_HIST_FIN_URL).content))
    with tempfile.TemporaryDirectory() as folder:
        mdb_path = Path(archive.extract("Hist_Fin.mdb", folder))
        history = _extract_census_hist_fin(mdb_path)
    levels = {key: series.loc[series.index <= _HIST_FIN_LAST_YEAR] for key, series in history.items()}

    later: dict[int, dict[str, float]] = {}
    from_summary_tables: list[int] = []
    unpublished: list[int] = []
    for year in range(_HIST_FIN_LAST_YEAR + 1, dt.date.today().year + 1):
        print(f"Downloading the Census estimates for fiscal {year} …")
        text = _census_estimates_text(year)
        if text is not None:
            later[year] = _parse_census_estimates(text, year)
            continue
        table = _census_summary_table(year)
        if table is not None:
            later[year] = _parse_census_summary_table(table, year)
            from_summary_tables.append(year)
        else:
            unpublished.append(year)
    if not later:
        raise RuntimeError("No Census estimates after 2008 were found; data module not modified.")
    for key in levels:
        levels[key] = pd.concat([levels[key], pd.Series({year: totals[key] for year, totals in later.items()})])
    _check_levels_add_up(levels, "Census estimates")

    frames = {
        key: pd.DataFrame(
            {"value": levels[key].to_numpy()},
            index=pd.DatetimeIndex([_census_fiscal_year_end(year) for year in levels[key].index], name=DATE_COLUMN),
        )
        for key in ("p", "m")
    }
    first, last = min(levels["p"].index), max(levels["p"].index)
    missing = [year for year in unpublished if year < last]
    block = (
        f"{US_ARCHIVE_BEGIN_MARKER}\n"
        f"# Generated from Census Bureau files on {dt.date.today():%Y-%m-%d}.\n"
        f"# Historical observations in the source's own terms; us_data applies them\n"
        f"# to the Federal Reserve's quarterly state-and-local debt at run time.\n\n"
        + _embedded_level_dict(
            "EMBEDDED_US_DEBT_BY_LEVEL",
            f"State (\"p\") and local (\"m\") governments' total debt outstanding at the end of\n"
            f"each fiscal year (stamped 30 June), dollars, {first}-{last}. Census Bureau: the\n"
            f"historical database of national totals, item {_HIST_FIN_TOTAL_DEBT}, through {_HIST_FIN_LAST_YEAR}\n"
            f"(selected years before 1952):\n{CENSUS_HIST_FIN_URL}\n"
            f"then each year's estimates by state and type of government, items\n"
            f"{' + '.join(_CENSUS_DEBT_ITEMS)} for the whole country:\n{CENSUS_OLD_ESTIMATES_URL}\n{CENSUS_TABLES_URL}"
            + (
                f"\n(fiscal {', '.join(map(str, from_summary_tables))}: the summary Table 1, \"Debt outstanding\")"
                if from_summary_tables
                else ""
            )
            + (f"\nNo figures for fiscal {', '.join(map(str, missing))}." if missing else ""),
            frames,
            "value",
        )
        + f"{US_ARCHIVE_END_MARKER}"
    )
    _replace_generated_block(source_path, US_ARCHIVE_BEGIN_MARKER, US_ARCHIVE_END_MARKER, block)

    print(f"Embedded Census state and local debt into {source_path}")
    share = levels["p"] / (levels["p"] + levels["m"])
    for key, name in (("p", "State"), ("m", "Local")):
        print(f"  {name + ' debt:':32} {len(levels[key]):4,} years, {first} to {last}")
    print(f"  State share of the two:          {share.iloc[0]:.1%} in {first}, {share.loc[1966]:.1%} in 1966, {share.iloc[-1]:.1%} in {last}")
    if from_summary_tables:
        print(f"  From the summary tables:         fiscal {', '.join(map(str, from_summary_tables))}")
    if unpublished:
        print(f"  Not published:                   fiscal {', '.join(map(str, unpublished))}")
