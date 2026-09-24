"""HTTP and source-format helpers for FRED, Bank of Canada and Statistics Canada."""

from __future__ import annotations

import io
import threading
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

# ---------------------------------------------------------------------------
# Session download cache
# ---------------------------------------------------------------------------
# Off by default: a command-line run downloads everything fresh, as it always
# has. The GUI turns it on so that redrawing after a change re-processes what
# was already downloaded instead of fetching it again; "Reload data" clears it.
# Keyed by URL (and query parameters). Only the two download helpers below use
# it; the StatCan WDS fallback is rare and stays uncached.

_download_cache: dict[tuple, object] | None = None
_download_cache_lock = threading.Lock()


def enable_download_cache() -> None:
    """Keep every download for the rest of the session (see above)."""
    global _download_cache
    with _download_cache_lock:
        if _download_cache is None:
            _download_cache = {}


def clear_download_cache() -> None:
    """Forget cached downloads so the next fetch goes to the network again."""
    with _download_cache_lock:
        if _download_cache is not None:
            _download_cache.clear()


def _cached(key: tuple, download: Callable[[], T]) -> T:
    """Return the cached result for ``key``, downloading it first if needed (or if caching is off)."""
    with _download_cache_lock:
        if _download_cache is not None and key in _download_cache:
            return _download_cache[key]  # type: ignore[return-value]
    result = download()
    with _download_cache_lock:
        if _download_cache is not None:
            _download_cache[key] = result
    return result


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

    # A Response keeps its body, so a cached one can be read again by later callers.
    key = ("canadian", url, tuple(sorted((params or {}).items())))
    return _cached(
        key,
        lambda: _with_retries(
            "Canadian request", attempt, max_retries=max_retries, retry_on=(requests.RequestException,)
        ),
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

    # Callers modify the frame they get, so each receives its own copy of the cached one.
    return _cached(("fred", series_id), lambda: _with_retries(series_id, attempt, max_retries=max_retries)).copy()


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
