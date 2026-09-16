"""Federal Reserve / FRED data pipelines."""

from __future__ import annotations

import pandas as pd

from .config import DATE_COLUMN, FRED_BILLION_TO_DOLLAR, STATCAN_MILLION_TO_DOLLAR, PlotConfig
from .http import fetch_fred_csv


US_SERIES_EARLIEST = {
    "3-Month Yield (DGS3MO)": pd.Timestamp("1981-09-01"),
    "2-Year Yield (DGS2)": pd.Timestamp("1976-06-01"),
    "5-Year Yield (DGS5)": pd.Timestamp("1962-01-02"),
    "10-Year Yield (DGS10)": pd.Timestamp("1962-01-02"),
    "30-Year Yield (DGS30)": pd.Timestamp("1977-02-15"),
    "Federal debt (GFDEBTN)": pd.Timestamp("1966-01-01"),
    "GDP / Interest (GDP, A091RC1…)": pd.Timestamp("1947-01-01"),
}

US_YIELD_SERIES = {
    "3-Month": "DGS3MO",
    "2-Year": "DGS2",
    "5-Year": "DGS5",
    "10-Year": "DGS10",
    "30-Year": "DGS30",
}

US_DEBT_SERIES_ID = "GFDEBTN"
US_STATE_LOCAL_DEBT_SERIES_ID = "SLGSDODNS"
US_INTEREST_SERIES_ID = "A091RC1Q027SBEA"
US_GDP_SERIES_ID = "GDP"
US_CANADIAN_GDP_FRED_ID = "NGDPSAXDCCAQ"


def fetch_us_yields(config: PlotConfig) -> pd.DataFrame:
    """Fetch the selected U.S. Treasury yield series and forward-fill gaps."""
    if not config.include_yield:
        return pd.DataFrame()

    print("Fetching FRED Treasury yields …")
    frames = [
        fetch_fred_csv(series_id).rename(columns={series_id: label})
        for label, series_id in US_YIELD_SERIES.items()
    ]
    data = frames[0].copy()
    for frame in frames[1:]:
        data = data.merge(frame, on=DATE_COLUMN, how="outer")
    data = data.sort_values(DATE_COLUMN).set_index(DATE_COLUMN).ffill().reset_index()
    return data.loc[data[DATE_COLUMN] >= config.start].sort_values(DATE_COLUMN)



def fetch_us_macro(config: PlotConfig, last_yield_date: pd.Timestamp | None) -> pd.DataFrame:
    """Fetch only the requested U.S. macro series and align their date ranges."""
    if not (config.include_debt or config.include_gdp or config.include_interest):
        return pd.DataFrame()

    print("Fetching FRED macro series …")

    macro_frames: list[pd.DataFrame] = []
    base_dates: pd.Series | None = None

    if config.include_debt:
        # Federal debt is reported in millions of dollars by this FRED series.
        federal = fetch_fred_csv(US_DEBT_SERIES_ID)
        federal["Fed_Debt"] = federal[US_DEBT_SERIES_ID] * STATCAN_MILLION_TO_DOLLAR
        federal = federal[[DATE_COLUMN, "Fed_Debt"]]
        base_dates = federal[DATE_COLUMN]
        macro_frames.append(federal)

        try:
            state_local = fetch_fred_csv(US_STATE_LOCAL_DEBT_SERIES_ID)
            state_local["State_Local_Debt"] = (
                state_local[US_STATE_LOCAL_DEBT_SERIES_ID] * STATCAN_MILLION_TO_DOLLAR
            )
            state_local = state_local[[DATE_COLUMN, "State_Local_Debt"]]
        except Exception:
            # Preserve the original fallback: unavailable state/local debt contributes zero.
            state_local = pd.DataFrame({DATE_COLUMN: base_dates, "State_Local_Debt": 0.0})
        macro_frames.append(state_local)
    else:
        # The date backbone is needed only when another selected series requires a
        # placeholder column during the merge, so obtain it from the first selected source.
        pass

    if config.include_interest:
        interest = fetch_fred_csv(US_INTEREST_SERIES_ID)
        interest["Interest_SAAR"] = interest[US_INTEREST_SERIES_ID] * FRED_BILLION_TO_DOLLAR
        interest["TTM Interest Payments ($)"] = interest["Interest_SAAR"].rolling(
            4, min_periods=4
        ).mean()
        macro_frames.append(interest[[DATE_COLUMN, "TTM Interest Payments ($)"]])

    if config.include_gdp:
        gdp = fetch_fred_csv(US_GDP_SERIES_ID)
        gdp["GDP_SAAR"] = gdp[US_GDP_SERIES_ID] * FRED_BILLION_TO_DOLLAR
        gdp["TTM Nominal GDP ($)"] = gdp["GDP_SAAR"].rolling(4, min_periods=4).mean()
        macro_frames.append(gdp[[DATE_COLUMN, "TTM Nominal GDP ($)"]])

    # Outer merges preserve all observations even when sources report on different dates.
    data = macro_frames[0].copy()
    for frame in macro_frames[1:]:
        data = data.merge(frame, on=DATE_COLUMN, how="outer")
    data = data.sort_values(DATE_COLUMN)

    if "Fed_Debt" in data:
        data["Fed_Debt"] = data["Fed_Debt"].ffill()
    if "State_Local_Debt" in data:
        data["State_Local_Debt"] = data["State_Local_Debt"].ffill().fillna(0)
        data["Fed_Debt"] = data.get("Fed_Debt", pd.Series(index=data.index, dtype=float))
        data["Total Aggregate Debt ($)"] = data["Fed_Debt"] + data["State_Local_Debt"]

    for column in ("TTM Nominal GDP ($)", "TTM Interest Payments ($)"):
        if column in data:
            data[column] = data[column].ffill()

    # Keep the original behavior of discarding rows before the requested start
    # and requiring at least one selected macro series to have data.
    data = data.loc[data[DATE_COLUMN] >= config.start].copy()
    macro_columns = [
        column
        for column in (
            "Total Aggregate Debt ($)",
            "TTM Nominal GDP ($)",
            "TTM Interest Payments ($)",
        )
        if column in data.columns
    ]
    if macro_columns:
        data = data.dropna(subset=macro_columns, how="all")

    if last_yield_date is not None and not data.empty and last_yield_date > data[DATE_COLUMN].max():
        tail = data.iloc[[-1]].copy()
        tail[DATE_COLUMN] = last_yield_date
        data = pd.concat([data, tail], ignore_index=True)

    return data
