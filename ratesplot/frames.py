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
