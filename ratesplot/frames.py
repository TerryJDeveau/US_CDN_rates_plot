"""Small DataFrame helpers shared by the data and plotting layers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import DATE_COLUMN, PlotConfig

# Names that ``DataFrame.reset_index()`` can give the former index depending on
# whether the index was named ``DATE``, ``date`` (BoC/StatCan sources) or unnamed.
_DATE_COLUMN_ALIASES = (DATE_COLUMN, "date", "index")


def rows_to_frame(rows: Iterable[tuple[str, float]], value_column: str) -> pd.DataFrame:
    """Build a date-indexed single-column frame from ``(iso_date, value)`` tuples."""
    frame = pd.DataFrame(list(rows), columns=[DATE_COLUMN, value_column])
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    return frame.set_index(DATE_COLUMN).sort_index()


def at_quarter_end(dates: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex | pd.Series:
    """Move each date to the last day of its quarter; a date already there (31 December) stays.

    Statistics Canada and FRED date a quarterly figure by the quarter's first
    day, but the figure describes the quarter's end: a stock is the level on
    that day (FRED's GFDEBTN at 2026-01-01 is the Treasury's debt on
    2026-03-31, exactly) and a trailing-year flow runs to it. Dated there, each
    step is drawn when it happened rather than a quarter early. The baked
    histories keep the sources' dates and are moved in the same way when
    loaded; their year-end figures are already where they belong.
    """
    return dates + pd.offsets.QuarterEnd(0)


def normalize_date_column(df: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Return a copy whose date column is named ``date_column``.

    Accepts frames where the date column is already correct, or is one of the
    aliases pandas produces from ``reset_index()``. Keeping this in one place
    means callers never depend on the capitalisation used by an upstream source.
    """
    if df.empty or date_column in df.columns:
        return df.copy()

    for candidate in _DATE_COLUMN_ALIASES:
        if candidate in df.columns:
            return df.rename(columns={candidate: date_column})

    raise KeyError(
        f"Unable to identify date column; expected {date_column!r} or one of "
        f"{_DATE_COLUMN_ALIASES}."
    )


def filter_to_date_range(
    df: pd.DataFrame, date_column: str, config: PlotConfig
) -> pd.DataFrame:
    """Return a copy containing only rows inside ``[config.start, config.end]``."""
    normalized = normalize_date_column(df, date_column)
    if normalized.empty:
        return normalized

    dates = pd.to_datetime(normalized[date_column])
    normalized[date_column] = dates
    return normalized.loc[dates.between(config.start, config.end)].copy()
