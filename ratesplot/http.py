"""HTTP and source-format helpers for FRED, Bank of Canada and StatCan."""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from .config import (
    CANADIAN_SESSION,
    DATE_COLUMN,
    DEFAULT_FRED_RETRIES,
    DEFAULT_GET_RETRIES,
    FRED_CSV_URL,
    HTTP_TIMEOUT_SECONDS,
    RETRY_BACKOFF_BASE_SECONDS,
)


def canadian_get(
    url: str,
    params: dict | None = None,
    *,
    max_retries: int = DEFAULT_GET_RETRIES,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> requests.Response:
    """Fetch a Bank of Canada or Statistics Canada URL.

    Args:
        url: Complete Canadian data endpoint.
        params: Optional query parameters, such as ``start_date``.
        max_retries: Number of total attempts after transient request failures.
        timeout: Per-request timeout in seconds.

    The Canadian path is deliberately independent from the FRED download
    implementation so provider-specific networking changes cannot interfere
    with one another.
    """
    last_exception: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = CANADIAN_SESSION.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exception = exc
            if attempt == max_retries:
                break
            wait_seconds = RETRY_BACKOFF_BASE_SECONDS ** attempt
            print(
                f"  Canadian request attempt {attempt}/{max_retries} failed ({exc}). "
                f"Retrying in {wait_seconds}s …"
            )
            time.sleep(wait_seconds)

    assert last_exception is not None
    raise last_exception


def fetch_fred_csv(
    series_id: str,
    *,
    max_retries: int = DEFAULT_FRED_RETRIES,
) -> pd.DataFrame:
    """Download one FRED series using pandas' direct URL reader.

    FRED is intentionally handled differently from the BoC/StatCan sources.
    The original working version passed the complete FRED CSV URL directly to
    ``pandas.read_csv()``. That access path is retained here because routing
    FRED through the shared ``requests.Session`` caused connection resets and
    long retry delays in practice.

    Args:
        series_id: FRED series identifier, such as ``"DGS10"`` or ``"GDP"``.
        max_retries: Number of complete pandas read attempts before failing.

    Returns:
        A DataFrame containing the canonical ``DATE`` column plus the series.
    """
    url = f"{FRED_CSV_URL}?id={series_id}"
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            # Important: do not replace this with SESSION.get() + StringIO.
            # The original direct pandas access is materially more reliable for
            # this application's FRED downloads.
            data = pd.read_csv(url)
            if len(data.columns) < 2:
                raise ValueError(f"FRED series {series_id} returned no value column")

            data.columns = [str(column).upper() for column in data.columns]
            data = data.rename(columns={data.columns[0]: DATE_COLUMN, data.columns[1]: series_id})
            data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
            data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
            return data
        except Exception as exc:
            last_exception = exc
            wait_seconds = RETRY_BACKOFF_BASE_SECONDS**attempt
            print(
                f"  {series_id} attempt {attempt}/{max_retries} failed ({exc}). "
                f"Retrying in {wait_seconds}s …"
            )
            if attempt < max_retries:
                time.sleep(wait_seconds)

    assert last_exception is not None
    raise last_exception


def parse_boc_csv(response_text: str) -> pd.DataFrame:
    """Parse a BoC Valet CSV response, skipping any metadata before the header."""
    lines = response_text.splitlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith('"date"') or line.startswith("date")
    )

    df = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    df.columns = [column.strip().strip('"') for column in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()
