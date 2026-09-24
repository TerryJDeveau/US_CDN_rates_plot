"""Application constants, per-country chart metadata, HTTP session, and PlotConfig.

Everything tunable lives here so the data and plotting modules contain logic
only. Numeric literals that affect chart appearance (padding, font sizes,
default limits) are deliberately centralised rather than scattered through the
plot functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Canvas and date defaults
# ---------------------------------------------------------------------------

DEFAULT_START = pd.Timestamp("1966-01-01")
DEFAULT_CANVAS_PX = (2048, 1536)
CANVAS_DPI = 100
MIN_CANVAS_PX = 800


def _today() -> pd.Timestamp:
    """Return today's date as a midnight Timestamp (the default chart end)."""
    return pd.Timestamp.today().normalize()


# ---------------------------------------------------------------------------
# Axis behaviour
# ---------------------------------------------------------------------------

# Fallback limits used when an axis has no data or the data span is degenerate.
DEFAULT_YIELD_YLIM = (-0.3, 7.5)
DEFAULT_MACRO_YLIM = (10_000_000_000, 100_000_000_000_000)

# Yield axis (linear): padding is a fraction of the *displayed* span, clamped
# to an absolute minimum so a flat series still gets visible breathing room.
YIELD_BOTTOM_PAD_FRACTION = 0.015
YIELD_TOP_PAD_FRACTION = 0.04
YIELD_MIN_BOTTOM_PAD = 0.02
YIELD_MIN_TOP_PAD = 0.08

# Macro axis (log10): padding is expressed in decades.
MACRO_PAD_FRACTION = 0.05
MACRO_MIN_PAD_DECADES = 0.05

# Date axis: small symmetric pad so lines do not touch the plot border.
DATE_PAD_FRACTION = 0.015
MIN_DATE_PAD_DAYS = 1

# Font sizes in points, as designed for the default 2048 x 1536 canvas. Every
# use multiplies them by ``PlotConfig.font_scale`` so other canvas sizes keep
# the same proportions (a fixed 24 pt title overflowed a 1100 px canvas).
TITLE_FS = 24
LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 16
# Gap between a y-axis label and its tick labels, in points (scaled like the fonts).
AXIS_LABEL_PAD_PT = 8

# Lower bound on ``PlotConfig.font_scale``. Linear scaling would take the
# 800 px minimum canvas to 0.39 (7 pt tick labels, about 10 px tall); 0.5
# keeps them legible. At the floor the title may still be too wide, so it is
# also shrunk to fit on its own (see ``legend.finish_legend_and_title``).
MIN_FONT_SCALE = 0.5

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

HTTP_TIMEOUT_SECONDS = 90
HTTP_POST_TIMEOUT_SECONDS = 60
STATCAN_TIMEOUT_SECONDS = 120
DEFAULT_GET_RETRIES = 4
DEFAULT_FRED_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 2

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BOC_GROUP_URL = "https://www.bankofcanada.ca/valet/observations/group/{group}/csv"
BOC_SERIES_URL = "https://www.bankofcanada.ca/valet/observations/{series_code}/csv"
STATCAN_TABLE_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/{table_id}-eng.zip"
STATCAN_WDS_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"

# Single session for the Canadian providers (Bank of Canada, Statistics Canada)
# so TCP connections are reused across the several downloads per run. FRED is
# deliberately *not* routed through this session: see ``http.fetch_fred_csv``.
CANADIAN_SESSION = requests.Session()
CANADIAN_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-CA,en;q=0.9",
    }
)

# ---------------------------------------------------------------------------
# Data sources and units
# ---------------------------------------------------------------------------

# The Bank of Canada Valet API is only needed from the point where the
# transcribed historical yield table (which ends 2000-12) stops. A one-month
# overlap is kept so forward-filling across the join sees the last live values.
BOC_START_DATE = "2000-12-01"
CANADIAN_YIELD_HIST_END = pd.Timestamp("2000-12-31")

# First observation available from the archived StatCan balance sheets.
CANADIAN_HISTORICAL_START = pd.Timestamp("1961-01-01")

# Live StatCan tables (downloaded on every run).
STATCAN_CDN_DEBT_TABLE = "36100467"       # general government gross debt, quarterly
STATCAN_CDN_GDP_TABLE = "36100104"        # GDP at market prices, SAAR, quarterly
STATCAN_CDN_INTEREST_TABLE = "10100015"   # consolidated government interest, quarterly
STATCAN_CDN_DEBT_FALLBACK_VECTOR = "v111463452"  # WDS vector behind the debt table

# Archived (terminated) StatCan tables. They no longer change, so
# ``--bake-archives`` downloads them once and writes the required rows into
# ``ratesplot/cdn_archive_data.py``. URLs are kept for auditability.
ARCHIVED_CDN_INTEREST_TABLE = "36100245"
ARCHIVED_CDN_FEDERAL_DEBT_TABLE = "36100533"
ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE = "36100534"
ARCHIVED_CDN_INTEREST_URL = STATCAN_TABLE_URL.format(table_id=ARCHIVED_CDN_INTEREST_TABLE)
ARCHIVED_CDN_FEDERAL_DEBT_URL = STATCAN_TABLE_URL.format(table_id=ARCHIVED_CDN_FEDERAL_DEBT_TABLE)
ARCHIVED_CDN_PROV_LOCAL_DEBT_URL = STATCAN_TABLE_URL.format(table_id=ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE)

# Archive splice tuning: the median live/archived ratio is estimated over their
# overlap up to this year, and the last ``ARCHIVE_SPLICE_YEARS`` of the archive
# are ramped geometrically toward that ratio.
ARCHIVE_CALIBRATION_END_YEAR = 1994
ARCHIVE_SPLICE_YEARS = 5
ARCHIVE_BEGIN_MARKER = "# BEGIN AUTO-GENERATED CANADIAN ARCHIVE DATA"
ARCHIVE_END_MARKER = "# END AUTO-GENERATED CANADIAN ARCHIVE DATA"

# Unit multipliers. StatCan tables and FRED's GFDEBTN/SLGSDODNS report millions;
# FRED's GDP and interest series report billions.
MILLION = 1_000_000
BILLION = 1_000_000_000

# ---------------------------------------------------------------------------
# Column names shared across data and plotting layers
# ---------------------------------------------------------------------------

DATE_COLUMN = "DATE"
YIELD_COLUMNS = ("3-Month", "2-Year", "5-Year", "10-Year", "30-Year")

GDP_COLUMN = "TTM Nominal GDP ($)"
CDN_DEBT_COLUMN = "Total Canadian Debt ($)"
CDN_INTEREST_COLUMN = "TTM Interest Payable ($)"
US_DEBT_COLUMN = "Total Aggregate Debt ($)"
US_INTEREST_COLUMN = "TTM Interest Payments ($)"

# ---------------------------------------------------------------------------
# Chart styling and per-country metadata
# ---------------------------------------------------------------------------

YIELD_LINE_STYLE = {"linewidth": 1.2, "alpha": 0.9}

# The same three semantic macro curves are drawn on both country charts.
MACRO_PLOT_STYLES = {
    "debt": {"color": "black", "linewidth": 2.5, "drawstyle": "steps-post"},
    "gdp": {"color": "darkgreen", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "-."},
    "interest": {"color": "darkred", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "--"},
}


@dataclass(frozen=True)
class CountryMetadata:
    """Labels, column names and title wording for one country's chart.

    ``*_label`` strings appear in the legend; ``*_title`` strings are the
    phrases assembled into the chart title for whichever series were drawn.
    """

    key: str
    country_name: str
    currency_prefix: str
    currency_label: str
    yield_title: str
    debt_column: str
    debt_label: str
    debt_title: str
    interest_column: str
    interest_label: str
    interest_title: str = "Interest Outlays"
    gdp_column: str = GDP_COLUMN
    gdp_label: str = "TTM Nominal GDP"
    gdp_title: str = "TTM GDP"

    def macro_specs(self, config: PlotConfig) -> tuple[tuple[str, bool, str, str], ...]:
        """Return ``(style_key, enabled, column, label)`` for each macro curve."""
        return (
            ("debt", config.include_debt, self.debt_column, self.debt_label),
            ("gdp", config.include_gdp, self.gdp_column, self.gdp_label),
            ("interest", config.include_interest, self.interest_column, self.interest_label),
        )

    def title_for(self, *, yields_drawn: bool, macro_keys_drawn: Iterable[str]) -> str:
        """Compose the chart title from the series that were actually drawn.

        Parts are joined with commas and a final ampersand, e.g.
        ``"U.S. Treasury Yields, Aggregate US Public Debt & TTM GDP"``.
        """
        macro_titles = {"debt": self.debt_title, "gdp": self.gdp_title, "interest": self.interest_title}
        parts = ([self.yield_title] if yields_drawn else []) + [
            macro_titles[key] for key in macro_titles if key in set(macro_keys_drawn)
        ]
        if not parts:
            return f"{self.country_name}: no series selected"
        if len(parts) == 1:
            return parts[0]
        return f"{', '.join(parts[:-1])} & {parts[-1]}"


CDN = CountryMetadata(
    key="cdn",
    country_name="Canada",
    currency_prefix="C$",
    currency_label="Nominal Units (CAD – Log Scale)",
    yield_title="CDN Benchmark Yields",
    debt_column=CDN_DEBT_COLUMN,
    debt_label="Aggregate CDN Public Debt",
    debt_title="Aggregate CDN Public Debt",
    interest_column=CDN_INTEREST_COLUMN,
    interest_label="TTM Interest Payable",
)

US = CountryMetadata(
    key="us",
    country_name="United States",
    currency_prefix="$",
    currency_label="Nominal Units (USD – Log Scale)",
    yield_title="U.S. Treasury Yields",
    debt_column=US_DEBT_COLUMN,
    debt_label="Aggregate US Public Debt",
    debt_title="Aggregate US Public Debt",
    interest_column=US_INTEREST_COLUMN,
    interest_label="TTM Interest Payable",
)


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotConfig:
    """All runtime choices used by the data and plotting layers.

    Built once by ``cli.parse_args`` and passed down, so no module reads
    mutable global state to find out what the user asked for.
    """

    start: pd.Timestamp = DEFAULT_START
    end: pd.Timestamp = field(default_factory=_today)
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

    @property
    def font_scale(self) -> float:
        """Return the factor applied to every font size and font-related spacing.

        Exactly 1.0 on the default canvas, so the default chart is unchanged.
        Otherwise it is the smaller of the width and height ratios to the
        default. That way a wide, short canvas is scaled for the height it
        lacks, and a tall, narrow one for the width. It never goes below
        ``MIN_FONT_SCALE``. Canvases larger than the default scale up, so text
        does not shrink to a speck on a 4096 px render.
        """
        ratio = min(self.width_px / DEFAULT_CANVAS_PX[0], self.height_px / DEFAULT_CANVAS_PX[1])
        return max(MIN_FONT_SCALE, ratio)

    @property
    def has_dollar_series(self) -> bool:
        """True when at least one curve on the log-dollar axis is selected."""
        return self.include_debt or self.include_gdp or self.include_interest

    @property
    def has_explicit_macro_limits(self) -> bool:
        """True when the user pinned either end of the dollar axis."""
        return self.macro_bottom is not None or self.macro_top is not None
