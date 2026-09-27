"""How the right-axis (macro) curves are measured: in dollars, or as a percentage of GDP (-r).

The data layers always produce dollar amounts, with each series forward-filled
onto the chart's dates. ``express`` converts that prepared frame (a ``DATE``
column plus one column per series) into the measure the configuration asks
for, just before drawing. The fetchers, the archive joins and the TTM
conventions therefore never depend on the measure, and the column names stay
the same, so drawing code finds the curves where it always does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CountryMetadata, PlotConfig


def express(macro: pd.DataFrame, config: PlotConfig, metadata: CountryMetadata) -> pd.DataFrame:
    """Return ``macro`` in the measure ``config`` asks for (unchanged for plain dollars)."""
    if config.relative:
        return as_percent_of_gdp(macro, metadata)
    return macro


def _numerator_columns(macro: pd.DataFrame, metadata: CountryMetadata) -> list[str]:
    """The debt and interest columns present in ``macro`` (GDP is the -r denominator)."""
    candidates = (metadata.debt_column, metadata.federal_debt_column, metadata.interest_column)
    return [column for column in candidates if column is not None and column in macro.columns]


def as_percent_of_gdp(macro: pd.DataFrame, metadata: CountryMetadata) -> pd.DataFrame:
    """Divide debt and interest by TTM GDP, in percent (see ``ratio_at_observations``).

    Where GDP is missing (in Canada, before 1926, so the federal-only debt
    from 1867 shows only from 1926) the ratio is missing too. The GDP column
    itself is left as it was; it is not drawn.
    """
    columns = _numerator_columns(macro, metadata)
    if macro.empty or not columns:
        return macro
    result = macro.copy()
    if metadata.gdp_column not in result.columns or not result[metadata.gdp_column].notna().any():
        print("  Warning: GDP is unavailable, so debt and interest cannot be shown as a percentage of it.")
        for column in columns:
            result[column] = np.nan
        return result

    gdp = result[metadata.gdp_column]
    for column in columns:
        result[column] = ratio_at_observations(result[column], gdp.where(gdp > 0)) * 100.0
    return result


def ratio_at_observations(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return ``numerator / denominator``, taken when the numerator is observed and held until its next observation.

    Both series are already carried forward onto the chart's dates, so an
    observation shows as a change of value. Dividing on every date instead
    would mix dates: Canadian debt before 1990 is a year-end figure (every
    few years in 1933-1943), while TTM GDP is stamped at the start of each
    year's fourth quarter, so for a quarter or more each year the old debt
    would sit over the new GDP, giving a sawtooth and brief dips that are not
    in the data. Taken at the observation, a year-end debt is divided by the
    TTM GDP current at the year end. The ratio ends where the numerator does
    (federal debt alone stops when the aggregate begins).
    """
    observed = numerator.notna() & numerator.ne(numerator.shift())
    return (numerator / denominator).where(observed).ffill().where(numerator.notna())
