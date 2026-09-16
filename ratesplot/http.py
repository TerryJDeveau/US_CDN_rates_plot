"""HTTP and source-format helpers for FRED, Bank of Canada and Statistics Canada."""

from __future__ import annotations

import io
import time
from typing import Callable, TypeVar

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

T = TypeVar("T")


def _with_retries(
    describe: str,
    attempt_once: Callable[[], T],
    *,
    max_retries: int,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``attempt_once`` up to ``max_retries`` times with exponential backoff.

    The wait after attempt *n* is ``RETRY_BACKOFF_BASE_SECONDS ** n`` seconds.
    The last failure is re-raised unchanged so callers see the real exception.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return attempt_once()
        except retry_on as exc:
            if attempt == max_retries:
                raise
            wait_seconds = RETRY_BACKOFF_BASE_SECONDS**attempt
            print(
                f"  {describe} attempt {attempt}/{max_retries} failed ({exc}). "
                f"Retrying in {wait_seconds}s …"
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable: max_retries must be >= 1")


def canadian_get(
    url: str,
    params: dict | None = None,
    *,
    max_retries: int = DEFAULT_GET_RETRIES,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> requests.Response:
    """GET a Bank of Canada or Statistics Canada URL with retries.

    Args:
        url: Complete endpoint URL.
        params: Optional query parameters, such as ``start_date``.
        max_retries: Total attempts before the last ``RequestException`` is raised.
        timeout: Per-request timeout in seconds.
    """

    def attempt() -> requests.Response:
        response = CANADIAN_SESSION.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response

    return _with_retries(
        "Canadian request", attempt, max_retries=max_retries, retry_on=(requests.RequestException,)
    )


def fetch_fred_csv(series_id: str, *, max_retries: int = DEFAULT_FRED_RETRIES) -> pd.DataFrame:
    """Download one FRED series as a two-column ``DATE`` / ``series_id`` frame.

    FRED is intentionally read by handing the URL straight to ``pandas.read_csv``
    rather than through a ``requests.Session``. Routing FRED through the shared
    session caused connection resets and long retry delays in practice, so do
    not "tidy" this into the Canadian download path.

    Args:
        series_id: FRED series identifier, such as ``"DGS10"`` or ``"GDP"``.
        max_retries: Complete read attempts before the last exception is raised.
    """
    url = f"{FRED_CSV_URL}?id={series_id}"

    def attempt() -> pd.DataFrame:
        data = pd.read_csv(url)
        if len(data.columns) < 2:
            raise ValueError(f"FRED series {series_id} returned no value column")
        # FRED's header is ``observation_date,<SERIES>``; normalise positionally
        # so a header rename upstream cannot break the pipeline.
        data = data.iloc[:, :2].set_axis([DATE_COLUMN, series_id], axis=1)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        data[series_id] = pd.to_numeric(data[series_id], errors="coerce")
        return data

    return _with_retries(series_id, attempt, max_retries=max_retries)


def parse_boc_csv(response_text: str) -> pd.DataFrame:
    """Parse a Bank of Canada Valet CSV, skipping the metadata block above the header.

    Valet responses begin with several ``"key","value"`` metadata lines; the
    observation table starts at the first line whose first field is ``date``.
    """
    lines = response_text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(('"date"', "date"))
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Bank of Canada response contains no 'date' header line")

    df = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    df.columns = [column.strip().strip('"') for column in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()
