"""``--bake-archives``: download archived StatCan tables and embed the rows."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .cdn_data import statcan_zip_table
from .config import (
    ARCHIVE_BEGIN_MARKER,
    ARCHIVE_CALIBRATION_END_YEAR,
    ARCHIVE_END_MARKER,
    ARCHIVED_CDN_FEDERAL_DEBT_URL,
    ARCHIVED_CDN_INTEREST_URL,
    ARCHIVED_CDN_PROV_LOCAL_DEBT_URL,
    DATE_COLUMN,
    STATCAN_MILLION_TO_DOLLAR,
)


def _extract_archived_cdn_interest(table: pd.DataFrame) -> pd.DataFrame:
    """Extract total-government interest-on-public-debt from archived table 36-10-0245."""
    required = {"Seasonal adjustment", "Levels of government", "Sector accounts", "REF_DATE", "VALUE"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Archived interest table is missing columns: {sorted(missing)}")

    mask = (
        table["Seasonal adjustment"].eq("Seasonally adjusted at annual rates")
        & table["Levels of government"].eq("Total government")
        & (table["Sector accounts"].eq("Interest on public debt")
           | table["Sector accounts"].eq("Interest on the public debt"))
    )
    result = table.loc[mask, ["REF_DATE", "VALUE"]].copy()
    result[DATE_COLUMN] = pd.PeriodIndex(result["REF_DATE"].astype(str), freq="Q").to_timestamp(how="start")
    # The archived source is an annual-rate flow; divide by four to obtain a true
    # quarterly amount before summing four quarters for the TTM measure.
    result["Quarterly Interest ($)"] = pd.to_numeric(result["VALUE"], errors="coerce") * STATCAN_MILLION_TO_DOLLAR / 4.0
    result["TTM Interest Payable ($)"] = result["Quarterly Interest ($)"].rolling(4, min_periods=4).sum()
    return result.dropna(subset=["TTM Interest Payable ($)"]).set_index(DATE_COLUMN)[["TTM Interest Payable ($)"]].sort_index()


def _extract_archived_cdn_debt(table: pd.DataFrame) -> pd.DataFrame:
    """Extract book-value short-term paper plus bonds from an archived balance sheet."""
    required = {"Valuation", "Categories", "REF_DATE", "VALUE"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Archived debt table is missing columns: {sorted(missing)}")

    mask = table["Valuation"].eq("Book value") & table["Categories"].isin(["Short-term paper", "Bonds"])
    result = table.loc[mask, ["REF_DATE", "Categories", "VALUE"]].copy()
    result["VALUE"] = pd.to_numeric(result["VALUE"], errors="coerce")
    result[DATE_COLUMN] = pd.to_datetime(result["REF_DATE"].astype(str) + "-12-31", errors="coerce")
    grouped = result.groupby([DATE_COLUMN, "Categories"], as_index=False)["VALUE"].sum()
    annual = grouped.groupby(DATE_COLUMN, as_index=False)["VALUE"].sum()
    annual["Total Canadian Debt ($)"] = annual["VALUE"] * STATCAN_MILLION_TO_DOLLAR
    return annual.set_index(DATE_COLUMN)[["Total Canadian Debt ($)"]].sort_index()


def _format_embedded_rows(frame: pd.DataFrame, value_column: str) -> str:
    """Format a date-indexed DataFrame as Python tuple literals for source embedding."""
    return "\n".join(
        f'    ("{index.strftime("%Y-%m-%d")}", {float(value):.10g}),'
        for index, value in frame[value_column].dropna().items()
    )


def bake_canadian_archives(source_path: Path | None = None) -> None:
    """Download inactive StatCan archives and safely embed required rows.

    The rows are written into ``ratesplot/cdn_archive_data.py`` (or ``source_path``).
    The generated source is first written to a temporary file and compiled with
    ``py_compile``. The real data module is replaced only after that validation
    succeeds, preventing a failed bake from leaving a syntactically broken module.
    """
    source_path = source_path or Path(__file__).resolve().with_name("cdn_archive_data.py")

    print("Downloading archived Canadian interest data …")
    interest_raw = statcan_zip_table("36100245")
    interest = _extract_archived_cdn_interest(interest_raw)
    if interest.empty:
        raise RuntimeError("Archived Canadian interest extraction returned no observations. The source file was not modified.")

    print("Downloading archived Canadian federal debt data …")
    federal_raw = statcan_zip_table("36100533")
    federal = _extract_archived_cdn_debt(federal_raw)
    if federal.empty:
        raise RuntimeError("Archived Canadian federal debt extraction returned no observations. The source file was not modified.")

    print("Downloading archived Canadian provincial/local debt data …")
    provincial_local_raw = statcan_zip_table("36100534")
    provincial_local = _extract_archived_cdn_debt(provincial_local_raw)
    if provincial_local.empty:
        raise RuntimeError("Archived Canadian provincial/local debt extraction returned no observations. The source file was not modified.")

    debt = (
        federal.rename(columns={"Total Canadian Debt ($)": "Federal Debt Securities ($)"})
        .join(provincial_local.rename(columns={"Total Canadian Debt ($)": "Provincial/Local Debt Securities ($)"}), how="outer")
        .fillna(0.0)
    )
    debt["Total Canadian Debt ($)"] = debt["Federal Debt Securities ($)"] + debt["Provincial/Local Debt Securities ($)"]
    debt = debt[["Total Canadian Debt ($)"]]

    debt = debt.loc[debt.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    interest = interest.loc[interest.index.year <= ARCHIVE_CALIBRATION_END_YEAR]
    if debt.empty or interest.empty:
        raise RuntimeError("Archived Canadian data became empty after the calibration cutoff. The source file was not modified.")

    block = (
        f"{ARCHIVE_BEGIN_MARKER}\n"
        f"# Generated from archived StatCan tables on the date this command is run.\n"
        f"# Interest source: {ARCHIVED_CDN_INTEREST_URL}\n"
        f"# Federal debt source: {ARCHIVED_CDN_FEDERAL_DEBT_URL}\n"
        f"# Provincial/local debt source: {ARCHIVED_CDN_PROV_LOCAL_DEBT_URL}\n"
        f"# These are historical observations only; the active modern series remains\n"
        f"# responsible for current data.\n"
        f"EMBEDDED_CDN_INTEREST_HISTORY = [\n{_format_embedded_rows(interest, 'TTM Interest Payable ($)')}\n]\n\n"
        f"EMBEDDED_CDN_DEBT_HISTORY = [\n{_format_embedded_rows(debt, 'Total Canadian Debt ($)')}\n]\n"
        f"{ARCHIVE_END_MARKER}"
    )

    source_text = source_path.read_text(encoding="utf-8")
    # Match the marker as a complete source line so the replacement is anchored
    # to the generated block and nothing else.
    begin_pattern = re.escape(ARCHIVE_BEGIN_MARKER)
    end_pattern = re.escape(ARCHIVE_END_MARKER)
    block_pattern = re.compile(
        rf"(?ms)^[ \t]*{begin_pattern}[ \t]*\r?\n.*?^[ \t]*{end_pattern}[ \t]*$"
    )
    if block_pattern.search(source_text):
        source_text = block_pattern.sub(block, source_text, count=1)
    else:
        # No existing block: append it to the end of the data module.
        source_text = source_text.rstrip() + "\n\n" + block + "\n"

    # Write to a sibling temporary file, compile it, and replace the working
    # source only after syntax validation succeeds.
    import os
    import py_compile
    import tempfile

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".py",
            prefix=source_path.stem + "_bake_", dir=source_path.parent, delete=False
        ) as temp_file:
            temp_file.write(source_text)
            temp_path = Path(temp_file.name)

        py_compile.compile(str(temp_path), doraise=True)
        os.replace(temp_path, source_path)
        temp_path = None
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    print(f"Embedded archived Canadian data into {source_path}")
    print(f"  Interest rows embedded: {len(interest):,}")
    print(f"  Debt rows embedded:     {len(debt):,}")
