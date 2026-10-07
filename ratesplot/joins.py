"""Joining a nation's sources into one series, and laying the series on the chart's dates.

Shared by the nations' data modules (``cdn_data``, ``uk_data``, ``de_data``), moved here
word for word from ``cdn_data`` (batch 3, 2026-10-06) so a second nation
joins its baked history to its live tables by the same rules:

* ``embedded_frame``: a baked list as a frame;
* ``chain`` / ``splice_archived_series``: sources joined oldest to newest,
  the newer never modified, the older brought to its level by a ramp over
  its last years (see there);
* ``ttm_sum``: quarterly flows summed over the trailing four quarters;
* ``dated_by_quarter_end``: a joined series dated by the day each figure describes;
* ``align_macro``: the macro series forward-filled onto the yield dates.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ARCHIVE_CALIBRATION_END_YEAR, ARCHIVE_SPLICE_YEARS, DATE_COLUMN, PlotConfig
from .frames import at_quarter_end, normalize_date_column, rows_to_frame


# ---------------------------------------------------------------------------
# Archive splicing
# ---------------------------------------------------------------------------


def embedded_frame(rows: list[tuple[str, float]], column: str) -> pd.DataFrame | None:
    """Return a baked list as a date-indexed frame, or ``None`` when it is empty.

    A list is empty only if the bake has never filled it; callers
    then fall back to the remaining sources.
    """
    return rows_to_frame(rows, column) if rows else None


def chain(sources: Iterable[pd.DataFrame | None], column: str, *, annual_historical: bool) -> pd.DataFrame | None:
    """Join ``sources`` (oldest first) pairwise with ``_splice_or_fallback``; missing ones are skipped.

    Each newer source is authoritative over the older ones where it exists.
    ``annual_historical`` is passed to every join (True when the older
    sources are year-end observations, as for debt).
    """
    result: pd.DataFrame | None = None
    for source in sources:
        result = _splice_or_fallback(result, source, column, annual_historical=annual_historical)
    return result


def splice_archived_series(
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
    return splice_archived_series(archive, modern, column, annual_historical=annual_historical)


def ttm_sum(quarterly: pd.Series) -> pd.Series:
    """Return the sum of each four consecutive quarters (a flow over the trailing year), dated by the last one's end.

    The UK's (ONS) and Germany's (Eurostat) flows are given per quarter,
    not as annual rates: the trailing year is the sum, where the U.S.'s and
    Canada's annualised rates take the mean. A quarter missing inside
    leaves the sums that include it missing. Moved here from ``uk_data``
    (batch 4) so both nations sum alike.
    """
    regular = quarterly.resample("QE").sum(min_count=1)
    return regular.rolling(4, min_periods=4).sum().dropna()


def dated_by_quarter_end(series: pd.DataFrame | pd.Series | None) -> pd.DataFrame | pd.Series | None:
    """Return a joined macro series with each quarter-start date moved to its quarter's end (``frames.at_quarter_end``).

    Done after the joins, which compare the sources on their own dates; it
    moves every date within its year, so the joins are the same either way.
    Year-end debt stays on 31 December.
    """
    if series is None:
        return None
    return series.set_axis(at_quarter_end(series.index))


# ---------------------------------------------------------------------------
# Laying the series on the chart's dates
# ---------------------------------------------------------------------------


def align_macro(
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
    return normalize_date_column(aligned.rename_axis(DATE_COLUMN).reset_index())
