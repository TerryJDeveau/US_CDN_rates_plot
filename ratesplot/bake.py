"""``--bake-archives``: download terminated StatCan tables and embed the rows.

The archived tables never change, so their relevant rows are written once into
``ratesplot/cdn_archive_data.py`` as Python literals. Normal runs then read
that module instead of downloading three multi-megabyte ZIPs.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import py_compile
import re
import tempfile
from pathlib import Path

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
    ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE,
    ARCHIVED_CDN_PROV_LOCAL_DEBT_URL,
    CDN_DEBT_COLUMN,
    CDN_INTEREST_COLUMN,
    DATE_COLUMN,
    MILLION,
)

# Balance-sheet members that constitute marketable debt securities.
DEBT_SECURITY_CATEGORIES = ("Short-term paper", "Bonds")


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


def _extract_archived_cdn_interest(table: pd.DataFrame) -> pd.DataFrame:
    """Extract total-government TTM interest on public debt from table 36-10-0245.

    The source is a seasonally adjusted *annual-rate* flow, so each quarter's
    value is divided by four before the trailing four-quarter sum.
    """
    _require_columns(table, {"Seasonal adjustment", "Levels of government", "Sector accounts", "REF_DATE", "VALUE"}, "interest")

    mask = (
        table["Seasonal adjustment"].eq("Seasonally adjusted at annual rates")
        & table["Levels of government"].eq("Total government")
        # The member was renamed at some point; accept either spelling.
        & table["Sector accounts"].isin(["Interest on public debt", "Interest on the public debt"])
    )
    selected = table.loc[mask, ["REF_DATE", "VALUE"]]
    dates = pd.PeriodIndex(selected["REF_DATE"].astype(str), freq="Q").to_timestamp(how="start")
    quarterly = (pd.to_numeric(selected["VALUE"], errors="coerce") * MILLION / 4.0).set_axis(dates).sort_index()
    ttm = quarterly.rolling(4, min_periods=4).sum().dropna()
    return ttm.rename(CDN_INTEREST_COLUMN).rename_axis(DATE_COLUMN).to_frame()


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
# Source generation
# ---------------------------------------------------------------------------


def _format_embedded_rows(frame: pd.DataFrame, value_column: str) -> str:
    """Format a date-indexed frame as indented ``("YYYY-MM-DD", value),`` lines."""
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


def bake_canadian_archives(source_path: Path | None = None) -> None:
    """Download the archived StatCan tables and embed the needed rows as Python literals.

    Writes into ``ratesplot/cdn_archive_data.py`` unless ``source_path`` is given.
    Rows after ``ARCHIVE_CALIBRATION_END_YEAR`` are dropped: the live series is
    authoritative from 1990, and the archive only needs to overlap it long
    enough for the splice ratio to be estimated.
    """
    source_path = source_path or Path(__file__).resolve().with_name("cdn_archive_data.py")

    print("Downloading archived Canadian interest data …")
    interest = _extract_archived_cdn_interest(statcan_zip_table(ARCHIVED_CDN_INTEREST_TABLE))
    if interest.empty:
        raise RuntimeError("Archived Canadian interest extraction returned no observations; data module not modified.")

    print("Downloading archived Canadian federal debt data …")
    federal = _extract_archived_cdn_debt(*statcan_zip_table_with_metadata(ARCHIVED_CDN_FEDERAL_DEBT_TABLE))
    if federal.empty:
        raise RuntimeError("Archived Canadian federal debt extraction returned no observations; data module not modified.")

    print("Downloading archived Canadian provincial/local debt data …")
    provincial_local = _extract_archived_cdn_debt(*statcan_zip_table_with_metadata(ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE))
    if provincial_local.empty:
        raise RuntimeError("Archived Canadian provincial/local debt extraction returned no observations; data module not modified.")

    # Aggregate public debt = federal + provincial/local securities outstanding.
    debt = federal.add(provincial_local, fill_value=0.0)

    debt = debt.loc[debt.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    interest = interest.loc[interest.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    if debt.empty or interest.empty:
        raise RuntimeError("Archived Canadian data became empty after the calibration cutoff; data module not modified.")

    block = (
        f"{ARCHIVE_BEGIN_MARKER}\n"
        f"# Generated from archived StatCan tables on {dt.date.today():%Y-%m-%d}.\n"
        f"# Interest source: {ARCHIVED_CDN_INTEREST_URL}\n"
        f"# Federal debt source: {ARCHIVED_CDN_FEDERAL_DEBT_URL}\n"
        f"# Provincial/local debt source: {ARCHIVED_CDN_PROV_LOCAL_DEBT_URL}\n"
        f"# Debt is the liability-side 'Short-term paper' + 'Bonds' at book value,\n"
        f"# federal plus provincial/local, stamped at year end. Interest is TTM.\n"
        f"# These are historical observations only; the live modern series remains\n"
        f"# responsible for current data.\n"
        f"EMBEDDED_CDN_INTEREST_HISTORY = [\n{_format_embedded_rows(interest, CDN_INTEREST_COLUMN)}\n]\n\n"
        f"EMBEDDED_CDN_DEBT_HISTORY = [\n{_format_embedded_rows(debt, CDN_DEBT_COLUMN)}\n]\n"
        f"{ARCHIVE_END_MARKER}"
    )

    source_text = source_path.read_text(encoding="utf-8")
    # Anchor both markers as complete lines so only the generated block is replaced.
    block_pattern = re.compile(
        rf"(?ms)^[ \t]*{re.escape(ARCHIVE_BEGIN_MARKER)}[ \t]*\r?\n.*?^[ \t]*{re.escape(ARCHIVE_END_MARKER)}[ \t]*$"
    )
    if block_pattern.search(source_text):
        # A callable replacement avoids re.sub interpreting backslashes in the data.
        source_text = block_pattern.sub(lambda _match: block, source_text, count=1)
    else:
        source_text = source_text.rstrip() + "\n\n" + block + "\n"

    _write_validated(source_path, source_text)

    print(f"Embedded archived Canadian data into {source_path}")
    print(f"  Interest rows embedded: {len(interest):,}")
    print(f"  Debt rows embedded:     {len(debt):,}")
