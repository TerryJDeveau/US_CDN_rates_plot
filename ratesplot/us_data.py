"""Federal Reserve (FRED) data pipelines for the U.S. chart.

All U.S. inputs come from FRED CSV downloads. Yields are daily; the macro
series are quarterly and are forward-filled onto the daily yield dates by
``plotting`` after fetching. State and local apart are yearly (BEA interest,
and Census debt baked into ``us_archive_data``); they split the quarterly
state-and-local figures (``_split_state_and_local_interest``,
``_split_state_and_local_debt``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BILLION,
    COMPONENT_LETTERS,
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
from .latest import extend_us_federal_debt, observed_through, project_to_now
from .us_archive_data import EMBEDDED_US_DEBT_BY_LEVEL

# First observation of each U.S. input, used to warn when ``--start`` is earlier.
US_SERIES_EARLIEST = {
    "3-Month Yield (DGS3MO)": pd.Timestamp("1981-09-01"),
    "2-Year Yield (DGS2)": pd.Timestamp("1976-06-01"),
    "5-Year Yield (DGS5)": pd.Timestamp("1962-01-02"),
    "10-Year Yield (DGS10)": pd.Timestamp("1962-01-02"),
    "30-Year Yield (DGS30)": pd.Timestamp("1977-02-15"),
    "Federal debt (GFDEBTN)": pd.Timestamp("1966-01-01"),
    "GDP / Interest (GDP, A180RC1…)": pd.Timestamp("1947-01-01"),
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
# Interest payments, SAAR $ billions, quarterly from 1947. The aggregate is all
# levels of government, as the Canadian one is. BEA's general-government
# interest (NIPA table 3.1) is exactly federal (3.2) plus state and local
# (3.3), which are the "f" and "n" lines of --interest:LETTERS. The
# state-and-local series is the total; Y705RC1 leaves out the interest paid
# abroad (about 1 % of it since 2008).
US_INTEREST_SERIES_ID = "A180RC1Q027SBEA"
US_FEDERAL_INTEREST_SERIES_ID = "A091RC1Q027SBEA"
US_STATE_LOCAL_INTEREST_SERIES_ID = "B111RC1Q027SBEA"
# State and local interest apart (NIPA tables 3.20 and 3.21, the "p" and "m"
# lines) are annual only, from 1959, $ billions; see _split_state_and_local_interest.
US_STATE_INTEREST_SERIES_ID = "W756RC1A027NBEA"
US_LOCAL_INTEREST_SERIES_ID = "W856RC1A027NBEA"
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


def _fetch_billions(series_id: str) -> pd.Series:
    """Fetch one FRED series in $ billions and return it in dollars, indexed by date."""
    frame = fetch_fred_csv(series_id)
    dates = pd.DatetimeIndex(frame[DATE_COLUMN], name=DATE_COLUMN)
    return pd.Series(frame[series_id].to_numpy(dtype=float) * BILLION, index=dates, name=series_id)


def _ttm_interest(saar: pd.Series, column: str) -> pd.DataFrame:
    """Return quarterly SAAR interest (dollars, date-indexed) as a TTM ``[DATE, column]`` frame.

    SAAR is already annualised, so the TTM level is the 4-quarter mean.
    """
    return saar.rolling(4, min_periods=4).mean().rename(column).reset_index()


def _split_state_and_local_interest(state_and_local: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """Split quarterly state-and-local interest (SAAR) into ``(state, local, last_year)``.

    BEA publishes state and local interest apart only annually (from 1959),
    and the two add up to the annual mean of the quarterly total. Every
    quarter of a year is split by that year's state share, so the TTM at each
    fourth quarter is exactly the published annual figure, while the quarters
    keep the movements of the total. Quarters after ``last_year``, the last
    annual figures, use its share; quarters before 1959 are missing.
    """
    state = _fetch_billions(US_STATE_INTEREST_SERIES_ID)
    local = _fetch_billions(US_LOCAL_INTEREST_SERIES_ID)
    share = (state / (state + local)).dropna()
    share.index = share.index.year
    last_year = int(share.index.max())
    years = state_and_local.index.year
    state_share = pd.Series(years.where(years <= last_year, last_year), index=state_and_local.index).map(share)
    return state_and_local * state_share, state_and_local * (1.0 - state_share), last_year


def _census_state_debt_share() -> pd.Series | None:
    """Return the state share of state and local debt at each Census fiscal year end, or None if not baked.

    Dated as FRED dates quarterly levels, by the quarter's first day: a
    fiscal year ending 30 June is the second quarter's level, dated 1 April.
    """
    state, local = (dict(EMBEDDED_US_DEBT_BY_LEVEL.get(key, [])) for key in ("p", "m"))
    if not state or not local:
        return None
    share = (pd.Series(state) / (pd.Series(state) + pd.Series(local))).dropna()
    share.index = pd.DatetimeIndex(pd.to_datetime(share.index)).to_period("Q").start_time
    return share.sort_index()


def _split_state_and_local_debt(
    dates: pd.Series, state_and_local: pd.Series, config: PlotConfig
) -> tuple[pd.Series, pd.Series] | None:
    """Split quarterly state-and-local debt (indexed like ``dates``) into ``(state, local)``; None if not baked.

    The Census counts state and local debt apart once a year. Its state
    share is interpolated linearly in time between fiscal years (and across
    the years before 1952 that it skips) and applied to the Fed's quarterly
    total, so state plus local is the total exactly and each fiscal year's
    share is the Census's. After the last fiscal year its share is held,
    with a warning; before the first (1902) there is none.
    """
    share = _census_state_debt_share()
    if share is None:
        print("  Warning: U.S. state and local debt apart have not been baked (tools/bake_archives.py); not drawn.")
        return None
    last = share.index.max()
    if config.end > last + pd.DateOffset(months=3):
        print(
            f"  Warning: U.S. state and local debt are counted apart only yearly, to fiscal {last.year}; "
            f"later quarters split the combined figure by fiscal {last.year}'s shares."
        )
    when = pd.DatetimeIndex(pd.to_datetime(dates)).as_unit("ns").asi8
    known = share.index.as_unit("ns").asi8
    # np.interp holds the last share after it; before the first there is none.
    values = np.interp(when, known, share.to_numpy(dtype=float))
    values[when < known[0]] = np.nan
    state_share = pd.Series(values, index=dates.index)
    return state_and_local * state_share, state_and_local * (1.0 - state_share)


def _fetch_us_interest_levels(letters: str, config: PlotConfig) -> list[pd.DataFrame]:
    """Return one TTM ``[DATE, component_column("interest", letter)]`` frame per level in ``letters``.

    Federal interest is required, as the aggregate is; the other levels are
    a refinement, so their failure is a warning.
    """
    frames = []
    if "f" in letters:
        frames.append(_ttm_interest(_fetch_billions(US_FEDERAL_INTEREST_SERIES_ID), component_column("interest", "f")))
    if not set(letters) & set("npm"):
        return frames
    try:
        state_and_local = _fetch_billions(US_STATE_LOCAL_INTEREST_SERIES_ID)
    except Exception as exc:
        print(f"  Warning: state and local interest ({US_STATE_LOCAL_INTEREST_SERIES_ID}) unavailable ({exc}).")
        return frames
    quarterly = {"n": state_and_local}
    if set(letters) & set("pm"):
        try:
            quarterly["p"], quarterly["m"], last_year = _split_state_and_local_interest(state_and_local)
            if config.end.year > last_year:
                print(
                    f"  Warning: U.S. state and local interest are published apart only yearly, to {last_year}; "
                    f"later quarters split the combined figure by {last_year}'s shares."
                )
        except Exception as exc:
            print(
                f"  Warning: state and local interest apart ({US_STATE_INTEREST_SERIES_ID}, "
                f"{US_LOCAL_INTEREST_SERIES_ID}) unavailable ({exc})."
            )
    for letter in letters:
        if letter in quarterly:
            frames.append(_ttm_interest(quarterly[letter], component_column("interest", letter)))
    return frames


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
        # With --cur, continued daily past FRED's last quarter (ratesplot.latest).
        federal = extend_us_federal_debt(_fetch_fred_dollars(US_DEBT_SERIES_ID, "Fed_Debt", MILLION), "Fed_Debt", config)
        frames.append(federal)
        try:
            frames.append(_fetch_fred_dollars(US_STATE_LOCAL_DEBT_SERIES_ID, "State_Local_Debt", MILLION))
        except Exception as exc:
            # State/local debt is a refinement; without it the chart still shows federal debt.
            print(f"  Warning: state/local debt unavailable ({exc}); using federal debt only.")
            frames.append(pd.DataFrame({DATE_COLUMN: federal[DATE_COLUMN], "State_Local_Debt": 0.0}))
            state_local_debt_known = False

    if config.include_interest:
        if split:
            frames += _fetch_us_interest_levels(config.components, config)
        else:
            frames.append(_ttm_interest(_fetch_billions(US_INTEREST_SERIES_ID), US_INTEREST_COLUMN))

    if config.needs_gdp:
        gdp = _fetch_fred_dollars(US_GDP_SERIES_ID, "GDP_SAAR", BILLION)
        gdp[GDP_COLUMN] = gdp["GDP_SAAR"].rolling(4, min_periods=4).mean()
        frames.append(gdp[[DATE_COLUMN, GDP_COLUMN]])

    # The date each drawn curve's data describe up to, for --cur's projection to
    # today (ratesplot.latest). The aggregate debt is known as late as the later
    # of its parts; each level of debt as its source.
    ends = {
        column: observed_through(frame.loc[frame[column].notna(), DATE_COLUMN])
        for frame in frames
        for column in frame.columns.drop(DATE_COLUMN)
    }
    observed = {column: ends[column] for column in (GDP_COLUMN, US_INTEREST_COLUMN) if column in ends}
    observed.update({column: date for column, date in ends.items() if column.startswith("Interest [")})
    if config.include_debt:
        parts = [ends.get("Fed_Debt")] + ([ends.get("State_Local_Debt")] if state_local_debt_known else [])
        observed[US_DEBT_COLUMN] = max(date for date in parts if date is not None)
        observed[component_column("debt", "f")] = ends.get("Fed_Debt")
        for letter in "npm":
            observed[component_column("debt", letter)] = ends.get("State_Local_Debt") if state_local_debt_known else None

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
            if not state_local_debt_known:
                state_local = pd.Series(float("nan"), index=data.index)
            data[component_column("debt", "n")] = state_local
            if set(config.components) & set("pm"):
                parts = _split_state_and_local_debt(data[DATE_COLUMN], state_local, config)
                if parts is not None:
                    data[component_column("debt", "p")], data[component_column("debt", "m")] = parts

    interest_levels = [component_column("interest", letter) for letter in COMPONENT_LETTERS]
    for column in [GDP_COLUMN, US_INTEREST_COLUMN, *interest_levels]:
        if column in data:
            data[column] = data[column].ffill()

    # Trim to the window and drop rows where no selected series has a value yet
    # (e.g. the first three quarters before a TTM rolling window is complete).
    history = data.set_index(DATE_COLUMN)  # the whole span, for --cur's projection
    data = data.loc[data[DATE_COLUMN] >= config.start].copy()
    candidates = [US_DEBT_COLUMN, GDP_COLUMN, US_INTEREST_COLUMN] + [
        component_column(kind, letter) for kind in ("debt", "interest") for letter in COMPONENT_LETTERS
    ]
    macro_columns = [c for c in candidates if c in data.columns]
    data = data.dropna(subset=macro_columns, how="all")

    if last_yield_date is not None and not data.empty and last_yield_date > data[DATE_COLUMN].max():
        tail = data.iloc[[-1]].copy()
        tail[DATE_COLUMN] = last_yield_date
        data = pd.concat([data, tail], ignore_index=True)

    # --cur, on a chart reaching today: carried on from the last data, as a debt clock is.
    return project_to_now(data, observed, config, history)
