"""Bank of Canada / Statistics Canada data pipelines."""

from __future__ import annotations

import io
import warnings
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd

from .cdn_archive_data import EMBEDDED_CDN_DEBT_HISTORY, EMBEDDED_CDN_INTEREST_HISTORY
from .cdn_hist_yields import build_cdn_hist_yields
from .config import (
    ARCHIVE_CALIBRATION_END_YEAR,
    ARCHIVE_SPLICE_YEARS,
    BOC_GROUP_URL,
    BOC_SERIES_URL,
    BOC_START_DATE,
    CANADIAN_HISTORICAL_START,
    CANADIAN_SESSION,
    DATE_COLUMN,
    HTTP_POST_TIMEOUT_SECONDS,
    STATCAN_MILLION_TO_DOLLAR,
    STATCAN_TABLE_URL,
    STATCAN_TIMEOUT_SECONDS,
    STATCAN_WDS_URL,
    YIELD_COLUMNS,
    PlotConfig,
)
from .http import canadian_get, fetch_fred_csv, parse_boc_csv


def boc_valet_group(group: str = "bond_yields_benchmark", start: str = BOC_START_DATE) -> pd.DataFrame:
    """Download a Bank of Canada Valet observation group as a time-indexed frame."""
    url = BOC_GROUP_URL.format(group=group)
    response = canadian_get(url, params={"start_date": start})
    frame = parse_boc_csv(response.text)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def boc_valet_series(series_code: str, start: str = BOC_START_DATE) -> pd.DataFrame:
    """Download one Bank of Canada Valet series as a time-indexed frame."""
    url = BOC_SERIES_URL.format(series_code=series_code)
    response = canadian_get(url, params={"start_date": start})
    frame = parse_boc_csv(response.text)
    frame.iloc[:, 0] = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    return frame


def statcan_zip_table(table_id: str) -> pd.DataFrame:
    """Download the non-metadata CSV from a Statistics Canada ZIP table."""
    url = STATCAN_TABLE_URL.format(table_id=table_id)
    print(f"Downloading StatCan table {table_id} (ZIP) …")
    response = canadian_get(url, timeout=STATCAN_TIMEOUT_SECONDS)

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".csv") and "MetaData" not in name
        )
        with archive.open(csv_name) as csv_file:
            return pd.read_csv(csv_file, low_memory=False)


def _splice_archived_series(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    value_column: str,
    *,
    calibration_end_year: int = ARCHIVE_CALIBRATION_END_YEAR,
    transition_years: int = ARCHIVE_SPLICE_YEARS,
    quarterly_current: bool = False,
) -> pd.DataFrame:
    """Join embedded historical data to the live series with a smooth level bridge.

    The archived and current sources use related, but not necessarily identical,
    accounting definitions. A median overlap ratio is estimated from 1990 through
    the calibration end year. The final ``transition_years`` of the historical
    segment are then multiplied by a geometric ramp toward that ratio. The live
    series remains unchanged from its first observation onward.
    """
    if historical is None or historical.empty:
        return current.copy()
    if current is None or current.empty:
        return historical.copy()

    historical = historical[[value_column]].dropna().sort_index()
    current = current[[value_column]].dropna().sort_index()
    if historical.empty or current.empty:
        return historical if not historical.empty else current

    overlap_start = max(historical.index.min(), current.index.min())
    overlap_end = min(
        historical.index.max(),
        current.index.max(),
        pd.Timestamp(f"{calibration_end_year}-12-31"),
    )

    hist_for_ratio = historical
    current_for_ratio = current
    if quarterly_current:
        # The archived debt series is annual year-end data, while the live series is
        # quarterly. Compare like-for-like year-end observations.
        current_for_ratio = current.resample("YE").last()

    hist_overlap = hist_for_ratio.loc[
        (hist_for_ratio.index >= overlap_start) & (hist_for_ratio.index <= overlap_end),
        value_column,
    ]
    current_overlap = current_for_ratio.loc[
        (current_for_ratio.index >= overlap_start) & (current_for_ratio.index <= overlap_end),
        value_column,
    ]

    if quarterly_current:
        hist_overlap = hist_overlap.copy()
        hist_overlap.index = hist_overlap.index.year
        current_overlap = current_overlap.copy()
        current_overlap.index = current_overlap.index.year
        overlap = pd.concat([hist_overlap.rename("historical"), current_overlap.rename("current")], axis=1).dropna()
    else:
        overlap = pd.concat([hist_overlap.rename("historical"), current_overlap.rename("current")], axis=1).dropna()

    ratios = overlap["current"] / overlap["historical"]
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    ratios = ratios[ratios > 0]
    scale_ratio = float(ratios.median()) if not ratios.empty else 1.0

    # Use only historical observations through the point where the modern series begins.
    live_start = current.index.min()
    historical_part = historical.loc[historical.index < live_start].copy()
    if historical_part.empty:
        return current.copy()

    # Geometric/exponential scaling preserves positivity and avoids the abrupt vertical
    # step that a single multiplicative correction would introduce at the splice.
    end_year = historical_part.index.year.max()
    transition_start_year = end_year - transition_years + 1
    years = historical_part.index.year.to_numpy(dtype=float)
    weights = np.clip(
        (years - transition_start_year) / max(transition_years - 1, 1),
        0.0,
        1.0,
    )
    scale = np.power(scale_ratio, weights)
    historical_part[value_column] = historical_part[value_column].to_numpy() * scale

    return pd.concat([historical_part, current]).sort_index()



def load_embedded_canadian_archives() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load hard-coded archived Canadian debt and interest data, when present.

    Returns:
        A pair ``(debt_history, interest_history)``. An absent archive block is
        represented by ``None`` so callers can retain a one-time download fallback.
    """
    debt_rows = EMBEDDED_CDN_DEBT_HISTORY
    interest_rows = EMBEDDED_CDN_INTEREST_HISTORY

    debt = None
    interest = None
    if debt_rows:
        debt = pd.DataFrame(debt_rows, columns=[DATE_COLUMN, "Total Canadian Debt ($)"])
        debt[DATE_COLUMN] = pd.to_datetime(debt[DATE_COLUMN])
        debt = debt.set_index(DATE_COLUMN).sort_index()
    if interest_rows:
        interest = pd.DataFrame(interest_rows, columns=[DATE_COLUMN, "TTM Interest Payable ($)"])
        interest[DATE_COLUMN] = pd.to_datetime(interest[DATE_COLUMN])
        interest = interest.set_index(DATE_COLUMN).sort_index()
    return debt, interest


def statcan_wds_vectors(vector_ids: Iterable[str | int], *, latest_n: int = 250) -> pd.DataFrame:
    """Retrieve the latest observations for one or more StatCan WDS vectors."""
    payload = [
        {"vectorId": int(str(vector).lstrip("vV")), "latestN": latest_n}
        for vector in vector_ids
    ]
    response = CANADIAN_SESSION.post(STATCAN_WDS_URL, json=payload, timeout=HTTP_POST_TIMEOUT_SECONDS)
    response.raise_for_status()

    frames: list[pd.DataFrame] = []
    for item in response.json():
        if item.get("status") != "SUCCESS":
            continue

        obj = item["object"]
        vector_id = obj["vectorId"]
        points = obj.get("vectorDataPoint", [])
        if not points:
            continue

        frame = pd.DataFrame(points)
        frame[DATE_COLUMN] = pd.to_datetime(frame["refPer"])
        column_name = f"v{vector_id}"
        frame = frame.set_index(DATE_COLUMN)[["value"]].rename(columns={"value": column_name})
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No data returned from WDS")
    return pd.concat(frames, axis=1).sort_index()


CANADIAN_SERIES_EARLIEST = {
    "3-Month Yield (Bank of Canada historical table)": pd.Timestamp("1934-03-01"),
    "2-Year Yield (Bank of Canada historical table)": pd.Timestamp("1982-06-01"),
    "5-Year Yield (Bank of Canada historical table)": pd.Timestamp("1980-11-01"),
    "10-Year Yield (Bank of Canada historical table)": pd.Timestamp("1951-01-01"),
    "30-Year Yield / Over 10 Years (Bank of Canada historical table)": pd.Timestamp("1919-01-01"),
    "Aggregate CDN Public Debt": CANADIAN_HISTORICAL_START,
    "TTM Nominal GDP": pd.Timestamp("1961-10-01"),
    "TTM Interest Payable": pd.Timestamp("1961-10-01"),
}


def fetch_cdn_yields(config: PlotConfig) -> pd.DataFrame:
    """Fetch Canadian benchmark yields and merge them with the historical table."""
    if not config.include_yield:
        return pd.DataFrame()

    print("Fetching Bank of Canada benchmark yields …")
    yields_raw = boc_valet_group("bond_yields_benchmark", start=BOC_START_DATE)
    yield_column_map = {
        "BD.CDN.2YR.DQ.YLD": "2-Year",
        "BD.CDN.5YR.DQ.YLD": "5-Year",
        "BD.CDN.10YR.DQ.YLD": "10-Year",
        "BD.CDN.LONG.DQ.YLD": "30-Year",
    }
    yields = yields_raw.rename(columns=yield_column_map)

    try:
        # The BoC group does not supply the 3-month series in the desired form,
        # so retrieve it separately and add it to the group observations.
        three_month = boc_valet_series("V80691303", start=BOC_START_DATE)
        three_month = three_month.rename(columns={three_month.columns[0]: "3-Month"})
        yields = yields.join(three_month[["3-Month"]], how="outer")
    except Exception as exc:
        print("  3-month T-bill failed:", exc)

    available_columns = [column for column in YIELD_COLUMNS if column in yields.columns]
    yields = yields[available_columns].dropna(how="all").ffill(limit=3)

    historical = build_cdn_hist_yields()
    historical_end = historical.index.max()
    api_part = yields[yields.index > historical_end]
    all_yields = pd.concat([historical, api_part]).sort_index().ffill(limit=3)

    return all_yields.loc[
        (all_yields.index >= config.start) & (all_yields.index <= config.end)
    ]


def fetch_cdn_debt(config: PlotConfig) -> pd.DataFrame | None:
    """Combine embedded archival debt history with the live modern debt series."""
    if not config.include_debt:
        return None

    embedded_debt, _ = load_embedded_canadian_archives()

    try:
        debt_raw = statcan_zip_table("36100467")
        debt = (
            debt_raw.query('Estimates == "Debt"')
            .assign(DATE=lambda frame: pd.to_datetime(frame["REF_DATE"] + "-01"))
            .set_index("DATE")
            .sort_index()
        )
        debt["Total Canadian Debt ($)"] = debt["VALUE"] * STATCAN_MILLION_TO_DOLLAR
        modern = debt[["Total Canadian Debt ($)"]].resample("QS").last()
    except Exception:
        try:
            # The vector is the compact fallback used when the full ZIP table fails.
            wds = statcan_wds_vectors(["v111463452"], latest_n=250)
            debt_quarterly = wds["v111463452"] * STATCAN_MILLION_TO_DOLLAR
            modern = debt_quarterly.to_frame("Total Canadian Debt ($)").resample("QS").last()
        except Exception:
            modern = None

    if embedded_debt is not None:
        return _splice_archived_series(
            embedded_debt,
            modern,
            "Total Canadian Debt ($)",
            quarterly_current=True,
        ) if modern is not None else embedded_debt
    return modern


def fetch_cdn_gdp(config: PlotConfig) -> pd.DataFrame | None:
    """Fetch Canadian current-price GDP and derive trailing-twelve-month GDP."""
    if not config.include_gdp:
        return None

    try:
        gdp_raw = statcan_zip_table("36100104")
        mask = (
            gdp_raw["Estimates"].str.contains(
                "Gross domestic product at market prices", case=False, na=False
            )
            & gdp_raw["Prices"].str.contains("Current prices", case=False, na=False)
            & gdp_raw["Seasonal adjustment"].str.contains(
                "Seasonally adjusted at annual rates", case=False, na=False
            )
        )
        gdp = (
            gdp_raw.loc[mask]
            .assign(DATE=lambda frame: pd.to_datetime(frame["REF_DATE"]))
            .set_index("DATE")
            .sort_index()
        )
        gdp["GDP_SAAR"] = gdp["VALUE"] * STATCAN_MILLION_TO_DOLLAR
        gdp["TTM Nominal GDP ($)"] = gdp["GDP_SAAR"].rolling(4).mean()
        return gdp[["TTM Nominal GDP ($)"]].resample("QS").last()
    except Exception:
        try:
            # FRED provides a compatible quarterly nominal-GDP fallback.
            fred = fetch_fred_csv("NGDPSAXDCCAQ").set_index(DATE_COLUMN)
            fred["GDP_SAAR"] = fred.iloc[:, 0] * STATCAN_MILLION_TO_DOLLAR
            fred["TTM Nominal GDP ($)"] = fred["GDP_SAAR"].rolling(4).mean()
            return fred[["TTM Nominal GDP ($)"]].resample("QS").last()
        except Exception:
            return None


def fetch_cdn_interest(config: PlotConfig) -> pd.DataFrame | None:
    """Combine embedded archival interest history with the live modern series."""
    if not config.include_interest:
        return None

    _, embedded_interest = load_embedded_canadian_archives()

    try:
        gfs_raw = statcan_zip_table("10100015")
        interest = (
            gfs_raw.query('`Government sectors` == "Consolidated government"')
            .query('`Statement of government operations and balance sheet` == "Interest"')
            .assign(DATE=lambda frame: pd.to_datetime(frame["REF_DATE"] + "-01"))
            .set_index("DATE")
            .sort_index()
        )
        interest["Interest_q"] = interest["VALUE"] * STATCAN_MILLION_TO_DOLLAR
        interest["TTM Interest Payable ($)"] = interest["Interest_q"].rolling(4).sum()
        modern = interest[["TTM Interest Payable ($)"]].resample("QS").last()
    except Exception:
        modern = None

    if embedded_interest is not None:
        return _splice_archived_series(
            embedded_interest,
            modern,
            "TTM Interest Payable ($)",
            quarterly_current=False,
        ) if modern is not None else embedded_interest
    return modern


def align_cdn_macro(
    yields_all: pd.DataFrame,
    debt_q: pd.DataFrame | None,
    gdp_q: pd.DataFrame | None,
    interest_q: pd.DataFrame | None,
    config: PlotConfig,
) -> pd.DataFrame:
    """Forward-fill quarterly macro data onto the yield observation dates."""
    plot_index = (
        yields_all.index
        if not yields_all.empty
        else pd.date_range(config.start, config.end, freq="D")
    )
    parts = [part for part in (debt_q, gdp_q, interest_q) if part is not None]
    if not parts:
        return pd.DataFrame({DATE_COLUMN: plot_index})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        macro = pd.concat(parts, axis=1, sort=False)
        aligned = (
            macro.reindex(macro.index.union(plot_index))
            .sort_index()
            .ffill()
            .reindex(plot_index)
            .reset_index()
        )

    # reset_index() can produce DATE, date, or index depending on the source index.
    for candidate in (DATE_COLUMN, "date", "index"):
        if candidate in aligned.columns:
            return aligned.rename(columns={candidate: DATE_COLUMN})
    raise KeyError("Unable to identify the date column in aligned Canadian macro data")
