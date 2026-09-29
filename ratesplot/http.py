"""HTTP and source-format helpers for FRED, Bank of Canada, Statistics Canada and the latest-value sources."""

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
    LATEST_SESSION,
    RETRY_BACKOFF_BASE_SECONDS,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Session download cache
# ---------------------------------------------------------------------------
# Off by default: a command-line run downloads everything fresh, as it always
# has. The GUI turns it on so that redrawing after a change re-processes what
# was already downloaded instead of fetching it again; "Reload data" clears it.
# The command line turns it on for a run without --start, whose trial pass
# (plotting.resolve_start) and drawing then share each download; it is still
# fresh for every run.
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


def _session_get(
    session: requests.Session,
    tag: str,
    url: str,
    params: dict | None,
    *,
    max_retries: int,
    timeout: int,
    cached: bool = True,
    missing_ok: bool = False,
) -> requests.Response | None:
    """GET ``url`` through ``session`` with retries; cached under ``(tag, url, params)`` unless ``cached`` is False.

    With ``missing_ok`` a 404 returns None at once: the thing does not exist
    (yet), and asking again will not change that.
    """

    def attempt() -> requests.Response | None:
        response = session.get(url, params=params, timeout=timeout)
        if missing_ok and response.status_code == 404:
            return None
        response.raise_for_status()
        return response

    def download() -> requests.Response | None:
        return _with_retries(f"{tag.capitalize()} request", attempt, max_retries=max_retries, retry_on=(requests.RequestException,))

    if not cached:
        return download()
    # A Response keeps its body, so a cached one can be read again by later callers.
    return _cached((tag, url, tuple(sorted((params or {}).items()))), download)


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
    response = _session_get(CANADIAN_SESSION, "canadian", url, params, max_retries=max_retries, timeout=timeout)
    assert response is not None  # only missing_ok returns None
    return response


def latest_get(
    url: str,
    params: dict | None = None,
    *,
    cached: bool = True,
    missing_ok: bool = False,
    max_retries: int = DEFAULT_GET_RETRIES,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> requests.Response | None:
    """GET a source of the latest values (--cur, see ``ratesplot.latest``) with retries.

    ``cached=False`` is for intraday quotes: every drawing should show the
    quotes of the moment, not those of the window's first drawing.
    ``missing_ok`` returns None for a 404 (a year the source does not have).
    """
    return _session_get(
        LATEST_SESSION, "latest", url, params, max_retries=max_retries, timeout=timeout, cached=cached, missing_ok=missing_ok
    )


def get_if_published(
    url: str, *, max_retries: int = DEFAULT_GET_RETRIES, timeout: int = HTTP_TIMEOUT_SECONDS
) -> requests.Response | None:
    """GET a file that may not exist, such as a Census year not yet published; None on 404.

    Connection failures and timeouts are retried as in ``canadian_get``; a
    missing file is not (it will not appear on retry), and any other HTTP
    error is raised. Used by the bake, so not cached.
    """

    def attempt() -> requests.Response | None:
        response = CANADIAN_SESSION.get(url, timeout=timeout)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response

    return _with_retries(
        "Request", attempt, max_retries=max_retries, retry_on=(requests.ConnectionError, requests.Timeout)
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


def parse_boc_csv(response_text: str, key: str = "date") -> pd.DataFrame:
    """Parse a Bank of Canada Valet CSV, skipping the metadata block above the header.

    Valet responses begin with several ``"key","value"`` metadata lines; the
    observation table starts at the first line whose first field is ``key``:
    ``date`` for time series, another name for a table keyed by something
    else (``dom_dbt_id``, the "as of" date of the debt outstanding). The
    result is indexed by that field, read as dates.
    """
    lines = response_text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith((f'"{key}"', key))
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Bank of Canada response contains no {key!r} header line")

    df = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    df.columns = [column.strip().strip('"') for column in df.columns]
    df[key] = pd.to_datetime(df[key])
    return df.set_index(key).sort_index()
