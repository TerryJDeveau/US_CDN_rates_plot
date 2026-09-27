"""Federal Reserve (FRED) data pipelines for the U.S. chart.

All U.S. inputs come from FRED CSV downloads. Yields are daily; the macro
series are quarterly and are forward-filled onto the daily yield dates by
``plotting`` after fetching.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    BILLION,
    DATE_COLUMN,
    GDP_COLUMN,
    MILLION,
    POPULATION_COLUMN,
    THOUSAND,
    US_DEBT_COLUMN,
    US_INTEREST_COLUMN,
    PlotConfig,
    component_column,
)
from .http import fetch_fred_csv

# First observation of each U.S. input, used to warn when ``--start`` is earlier.
US_SERIES_EARLIEST = {
    "3-Month Yield (DGS3MO)": pd.Timestamp("1981-09-01"),
    "2-Year Yield (DGS2)": pd.Timestamp("1976-06-01"),
    "5-Year Yield (DGS5)": pd.Timestamp("1962-01-02"),
    "10-Year Yield (DGS10)": pd.Timestamp("1962-01-02"),
    "30-Year Yield (DGS30)": pd.Timestamp("1977-02-15"),
    "Federal debt (GFDEBTN)": pd.Timestamp("1966-01-01"),
    "GDP / Interest (GDP, A091RC1…)": pd.Timestamp("1947-01-01"),
}

# Chart column -> FRED constant-maturity Treasury yield series.
US_YIELD_SERIES = {
    "3-Month": "DGS3MO",
    "2-Year": "DGS2",
    "5-Year": "DGS5",
    "10-Year": "DGS10",
    "30-Year": "DGS30",
}

US_DEBT_SERIES_ID = "GFDEBTN"               # federal debt, total public, $ millions
US_STATE_LOCAL_DEBT_SERIES_ID = "SLGSDODNS"  # state & local debt securities, $ millions
US_INTEREST_SERIES_ID = "A091RC1Q027SBEA"   # federal interest payments, SAAR $ billions
# State and local interest payments (to persons and business), SAAR $ billions:
# the "n" (non-federal) interest component, --interest:n.
US_STATE_LOCAL_INTEREST_SERIES_ID = "Y705RC1Q027SBEA"
US_GDP_SERIES_ID = "GDP"                    # nominal GDP, SAAR $ billions
US_POPULATION_SERIES_ID = "B230RC0Q173SBEA"  # population (mid-period), thousands, quarterly from 1947


def fetch_us_yields(config: PlotConfig) -> pd.DataFrame:
    """Return the U.S. Treasury yield curves as a ``DATE``-column frame from ``config.start``.

    Series are outer-joined on date and forward-filled so holidays in one
    tenor do not create gaps in the others.
    """
    if not config.include_yield:
        return pd.DataFrame()

    print("Fetching FRED Treasury yields …")
    frames = [
        fetch_fred_csv(series_id).set_index(DATE_COLUMN).rename(columns={series_id: label})
        for label, series_id in US_YIELD_SERIES.items()
    ]
    data = pd.concat(frames, axis=1).sort_index().ffill().reset_index()
    return data.loc[data[DATE_COLUMN] >= config.start]


def fetch_us_population() -> pd.Series | None:
    """Return the U.S. population (persons, quarterly, from 1947) for -p, or None if FRED fails.

    BEA's mid-period population, the one it divides by for per-capita GDP.
    It starts with the GDP and interest series (1947), so nothing earlier is
    needed; debt starts later (1966).
    """
    try:
        frame = fetch_fred_csv(US_POPULATION_SERIES_ID)
    except Exception as exc:
        print(f"  Warning: U.S. population ({US_POPULATION_SERIES_ID}) unavailable ({exc}).")
        return None
    dates = pd.DatetimeIndex(frame[DATE_COLUMN], name=DATE_COLUMN)
    people = pd.Series(frame[US_POPULATION_SERIES_ID].to_numpy(dtype=float) * THOUSAND, index=dates, name=POPULATION_COLUMN)
    return people.dropna()


def _fetch_fred_dollars(series_id: str, column: str, multiplier: int) -> pd.DataFrame:
    """Fetch one FRED series and return ``[DATE, column]`` scaled to dollars."""
    frame = fetch_fred_csv(series_id)
    frame[column] = frame[series_id] * multiplier
    return frame[[DATE_COLUMN, column]]


def fetch_us_macro(config: PlotConfig, last_yield_date: pd.Timestamp | None) -> pd.DataFrame:
    """Fetch the selected U.S. macro series and align them on one ``DATE`` column.

    Args:
        last_yield_date: When later than the final macro observation, the last
            macro row is repeated at this date so the step curves extend to the
            right edge of the yield data instead of stopping a quarter early.
    """
    if not config.has_macro_series:
        return pd.DataFrame()

    print("Fetching FRED macro series …")
    frames: list[pd.DataFrame] = []

    split = bool(config.components)  # draw federal and non-federal lines instead of the aggregates
    state_local_debt_known = True
    if config.include_debt:
        federal = _fetch_fred_dollars(US_DEBT_SERIES_ID, "Fed_Debt", MILLION)
        frames.append(federal)
        try:
            frames.append(_fetch_fred_dollars(US_STATE_LOCAL_DEBT_SERIES_ID, "State_Local_Debt", MILLION))
        except Exception as exc:
            # State/local debt is a refinement; without it the chart still shows federal debt.
            print(f"  Warning: state/local debt unavailable ({exc}); using federal debt only.")
            frames.append(pd.DataFrame({DATE_COLUMN: federal[DATE_COLUMN], "State_Local_Debt": 0.0}))
            state_local_debt_known = False

    if config.include_interest:
        interest = _fetch_fred_dollars(US_INTEREST_SERIES_ID, "Interest_SAAR", BILLION)
        # SAAR is already annualised, so the TTM level is the 4-quarter mean.
        interest[US_INTEREST_COLUMN] = interest["Interest_SAAR"].rolling(4, min_periods=4).mean()
        frames.append(interest[[DATE_COLUMN, US_INTEREST_COLUMN]])
        if split:
            try:
                state_local = _fetch_fred_dollars(US_STATE_LOCAL_INTEREST_SERIES_ID, "SL_Interest_SAAR", BILLION)
                column = component_column("interest", "n")
                state_local[column] = state_local["SL_Interest_SAAR"].rolling(4, min_periods=4).mean()
                frames.append(state_local[[DATE_COLUMN, column]])
            except Exception as exc:
                print(f"  Warning: state and local interest ({US_STATE_LOCAL_INTEREST_SERIES_ID}) unavailable ({exc}).")

    if config.needs_gdp:
        gdp = _fetch_fred_dollars(US_GDP_SERIES_ID, "GDP_SAAR", BILLION)
        gdp[GDP_COLUMN] = gdp["GDP_SAAR"].rolling(4, min_periods=4).mean()
        frames.append(gdp[[DATE_COLUMN, GDP_COLUMN]])

    # Outer merge keeps every observation date from every source.
    data = frames[0]
    for frame in frames[1:]:
        data = data.merge(frame, on=DATE_COLUMN, how="outer")
    data = data.sort_values(DATE_COLUMN)

    if config.include_debt:
        # Debt is a stock: carry the last known level forward. Missing
        # state/local observations contribute zero rather than dropping the row.
        data["Fed_Debt"] = data["Fed_Debt"].ffill()
        state_local = data["State_Local_Debt"].ffill()
        data["State_Local_Debt"] = state_local.fillna(0)
        data[US_DEBT_COLUMN] = data["Fed_Debt"] + data["State_Local_Debt"]
        if split:
            data[component_column("debt", "f")] = data["Fed_Debt"]
            # As a line of its own, unknown state/local debt is missing, not zero.
            data[component_column("debt", "n")] = state_local if state_local_debt_known else float("nan")

    for column in (GDP_COLUMN, US_INTEREST_COLUMN, component_column("interest", "n")):
        if column in data:
            data[column] = data[column].ffill()
    if split and US_INTEREST_COLUMN in data:
        data[component_column("interest", "f")] = data[US_INTEREST_COLUMN]

    # Trim to the window and drop rows where no selected series has a value yet
    # (e.g. the first three quarters before a TTM rolling window is complete).
    data = data.loc[data[DATE_COLUMN] >= config.start].copy()
    candidates = [US_DEBT_COLUMN, GDP_COLUMN, US_INTEREST_COLUMN] + [
        component_column(kind, letter) for kind in ("debt", "interest") for letter in "fn"
    ]
    macro_columns = [c for c in candidates if c in data.columns]
    data = data.dropna(subset=macro_columns, how="all")

    if last_yield_date is not None and not data.empty and last_yield_date > data[DATE_COLUMN].max():
        tail = data.iloc[[-1]].copy()
        tail[DATE_COLUMN] = last_yield_date
        data = pd.concat([data, tail], ignore_index=True)

    return data
