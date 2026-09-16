"""Application constants, HTTP sessions, and the immutable PlotConfig."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Application constants
# ---------------------------------------------------------------------------

DEFAULT_START = pd.Timestamp("1966-01-01")
DEFAULT_CANVAS_PX = (2048, 1536)
CANVAS_DPI = 100
MIN_CANVAS_PX = 800

# Axis defaults are intentionally centralized so chart behavior is maintained
# from one place instead of scattering numeric literals through plot functions.
DEFAULT_YIELD_YLIM = (-0.3, 7.5)
DEFAULT_MACRO_YLIM = (10_000_000_000, 100_000_000_000_000)
YIELD_BOTTOM_PAD_FRACTION = 0.015
YIELD_TOP_PAD_FRACTION = 0.04
YIELD_MIN_BOTTOM_PAD = 0.02
YIELD_MIN_TOP_PAD = 0.08
MACRO_PAD_FRACTION = 0.05
MACRO_MIN_PAD_DECADES = 0.05
DATE_PAD_FRACTION = 0.015
MIN_DATE_PAD_DAYS = 1

TITLE_FS = 24
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 16

HTTP_TIMEOUT_SECONDS = 90
HTTP_POST_TIMEOUT_SECONDS = 60
STATCAN_TIMEOUT_SECONDS = 120
DEFAULT_GET_RETRIES = 4
DEFAULT_FRED_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 2

BOC_START_DATE = "1990-01-01"
CANADIAN_MODERN_MACRO_START = pd.Timestamp("1990-01-01")
CANADIAN_HISTORICAL_START = pd.Timestamp("1961-01-01")
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BOC_GROUP_URL = "https://www.bankofcanada.ca/valet/observations/group/{group}/csv"
BOC_SERIES_URL = "https://www.bankofcanada.ca/valet/observations/{series_code}/csv"
STATCAN_TABLE_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/{table_id}-eng.zip"
STATCAN_WDS_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"

# Archived Statistics Canada sources. These tables are no longer updated, so
# ``--bake-archives`` can download the required historical rows once and write
# them into this source file. Exact source URLs are retained here for auditability.
ARCHIVED_CDN_INTEREST_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100245-eng.zip"
ARCHIVED_CDN_FEDERAL_DEBT_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100533-eng.zip"
ARCHIVED_CDN_PROV_LOCAL_DEBT_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100534-eng.zip"

STATCAN_MILLION_TO_DOLLAR = 1_000_000
FRED_BILLION_TO_DOLLAR = 1_000_000_000

YIELD_COLUMNS = ("3-Month", "2-Year", "5-Year", "10-Year", "30-Year")
DATE_COLUMN = "DATE"

# Plot styles are named because the same semantic curves are rendered in
# both country-specific chart functions.
MACRO_PLOT_STYLES = {
    "debt": {"color": "black", "linewidth": 2.5, "drawstyle": "steps-post"},
    "gdp": {"color": "darkgreen", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "-."},
    "interest": {"color": "darkred", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "--"},
}

CANADIAN_YIELD_HIST_END = pd.Timestamp("2000-12-31")
ARCHIVED_CDN_DEBT_END_YEAR = 1989
ARCHIVE_BAKE_START_YEAR = 1961
ARCHIVE_CALIBRATION_END_YEAR = 1994
ARCHIVE_SPLICE_YEARS = 5
ARCHIVE_BEGIN_MARKER = "# BEGIN AUTO-GENERATED CANADIAN ARCHIVE DATA"
ARCHIVE_END_MARKER = "# END AUTO-GENERATED CANADIAN ARCHIVE DATA"

# Human-readable chart metadata lives together so plot functions stay focused
# on constructing the data lines instead of repeating labels and titles.
COUNTRY_METADATA = {
    "cdn": {
        "currency_prefix": "C$",
        "currency_label": "Dollar Amount (CAD – Log Scale)",
        "debt_column": "Total Canadian Debt ($)",
        "debt_label": "Aggregate CDN Public Debt",
        "interest_column": "TTM Interest Payable ($)",
        "interest_label": "TTM Interest Payable",
        "title": "CDN Benchmark Yields v. Aggregate CDN Public Debt, TTM GDP & Interest Outlays",
    },
    "us": {
        "currency_prefix": "$",
        "currency_label": "Dollar Amount (USD – Log Scale)",
        "debt_column": "Total Aggregate Debt ($)",
        "debt_label": "Aggregate US Public Debt",
        "interest_column": "TTM Interest Payments ($)",
        "interest_label": "TTM Interest Payable",
        "title": "U.S. Treasury Yields v. Aggregate US Public Debt, TTM GDP & Interest Outlays",
    },
}

# Shared HTTP session. Keeping it at module scope allows connection reuse while
# the configuration needed for requests remains explicit in each helper call.
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-CA,en;q=0.9",
    }
)


# Keep Canadian-source connections isolated from other providers. FRED uses
# a separate direct pandas download path and never uses this session.
CANADIAN_SESSION = requests.Session()
CANADIAN_SESSION.headers.update(SESSION.headers)


@dataclass(frozen=True)
class PlotConfig:
    """All runtime choices used by the data and plotting layers.

    Keeping these values together removes the need for plotting functions to
    read mutable module-level state. Callers can construct a configuration
    once in ``main`` and pass it to every subprogram that needs it.
    """

    start: pd.Timestamp = DEFAULT_START
    end: pd.Timestamp | None = None
    width_px: int = DEFAULT_CANVAS_PX[0]
    height_px: int = DEFAULT_CANVAS_PX[1]
    yield_ymin: float | None = None
    yield_ymax: float | None = None
    macro_bottom: float | None = None
    macro_top: float | None = None
    include_yield: bool = True
    include_debt: bool = True
    include_gdp: bool = True
    include_interest: bool = True
    show_cdn: bool = True
    show_us: bool = True
    bake_archives: bool = False

    @property
    def figsize_inches(self) -> tuple[float, float]:
        """Return the configured pixel canvas converted to inches for Matplotlib."""
        return self.width_px / CANVAS_DPI, self.height_px / CANVAS_DPI
