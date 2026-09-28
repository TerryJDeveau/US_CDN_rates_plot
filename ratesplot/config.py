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

# Fallback limits used when an axis has no data or the data span is degenerate,
# and for the other end when only one end of the right axis is pinned. The
# right axis has one per measure: dollars, percent of GDP (-r), dollars per
# person (-p).
DEFAULT_YIELD_YLIM = (-0.3, 7.5)
DEFAULT_MACRO_YLIM = (10_000_000_000, 100_000_000_000_000)
DEFAULT_RELATIVE_YLIM = (0.1, 1_000.0)
DEFAULT_PER_CAPITA_YLIM = (1.0, 1_000_000.0)

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

# Live StatCan tables (downloaded on every run).
STATCAN_CDN_DEBT_TABLE = "36100467"       # general government gross debt, quarterly
STATCAN_CDN_GDP_TABLE = "36100104"        # GDP at market prices, SAAR, quarterly
STATCAN_CDN_INTEREST_TABLE = "10100015"   # consolidated government interest, quarterly
STATCAN_CDN_DEBT_FALLBACK_VECTOR = "v111463452"  # WDS vector behind the debt table

# Archived (terminated) StatCan tables. They no longer change, so the bake
# (``tools/bake_archives.py``) downloads them once and writes the required rows
# into ``ratesplot/cdn_archive_data.py``. URLs are kept for auditability.
ARCHIVED_CDN_INTEREST_TABLE = "36100245"
ARCHIVED_CDN_FEDERAL_DEBT_TABLE = "36100533"
ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE = "36100534"
ARCHIVED_CDN_INTEREST_URL = STATCAN_TABLE_URL.format(table_id=ARCHIVED_CDN_INTEREST_TABLE)
ARCHIVED_CDN_FEDERAL_DEBT_URL = STATCAN_TABLE_URL.format(table_id=ARCHIVED_CDN_FEDERAL_DEBT_TABLE)
ARCHIVED_CDN_PROV_LOCAL_DEBT_URL = STATCAN_TABLE_URL.format(table_id=ARCHIVED_CDN_PROV_LOCAL_DEBT_TABLE)
# The same balance sheet split by level (provincial + local = 36100534 exactly),
# for the debt components (--debt:p, --debt:m).
ARCHIVED_CDN_PROVINCIAL_DEBT_TABLE = "36100535"
ARCHIVED_CDN_LOCAL_DEBT_TABLE = "36100536"

# Older history, also baked (see ``ratesplot.bake``).
# National accounts on the 1968 SNA basis, the source of GDP and government
# interest before 1961: annual from 1926, quarterly (SAAR) from 1947 for GDP
# and from 1950 for interest (the quarterly table is blank before that).
HISTORICAL_CDN_GDP_ANNUAL_TABLE = "36100150"
HISTORICAL_CDN_GDP_QUARTERLY_TABLE = "36100137"
HISTORICAL_CDN_INTEREST_ANNUAL_TABLE = "36100177"
HISTORICAL_CDN_INTEREST_QUARTERLY_TABLE = "36100142"
# Historical Statistics of Canada (StatCan 11-516-X), section H: federal debt
# from 1867, provincial and local debt from 1933 (1933, 1937, 1939, 1941,
# 1943, then yearly from 1945). Plain CSV files, not StatCan tables.
HSC_SECTION_H_URL = "https://www150.statcan.gc.ca/n1/pub/11-516-x/sectionh/{series}-eng.csv"
HSC_FEDERAL_DEBT_SERIES = "H35_51"
HSC_PROVINCIAL_DEBT_SERIES = "H382_397"
HSC_LOCAL_DEBT_SERIES = "H398_403"
# Bank of Canada financial market statistics (monthly, last Wednesday). Its
# average yields for the 1-3 and 3-5 year bands stand in for the 2- and 5-year
# benchmarks before those begin (1982-06 and 1980-11), as the 5-10 year and
# over-10 year bands already do for the 10- and 30-year history.
BOC_FINANCIAL_MARKET_TABLE = "10100122"
# Population: annual estimates from 1867 (baked), quarterly from 1946 (live).
HISTORICAL_CDN_POPULATION_TABLE = "17100063"
STATCAN_CDN_POPULATION_TABLE = "17100009"

# Archive splice tuning: the median live/archived ratio is estimated over their
# overlap up to this year, and the last ``ARCHIVE_SPLICE_YEARS`` of the archive
# are ramped geometrically toward that ratio.
ARCHIVE_CALIBRATION_END_YEAR = 1994
ARCHIVE_SPLICE_YEARS = 5
ARCHIVE_BEGIN_MARKER = "# BEGIN AUTO-GENERATED CANADIAN ARCHIVE DATA"
ARCHIVE_END_MARKER = "# END AUTO-GENERATED CANADIAN ARCHIVE DATA"

# U.S. state and local government debt apart (Census Bureau), baked into
# ``ratesplot/us_archive_data.py``: the historical database of national totals
# (fiscal years 1902-2008, an Access file), then each later year's estimates
# by state and type of government (fixed-width text; 2009-2010 on the old
# site, later years inside each year's "Individual Unit File" ZIP).
CENSUS_HIST_FIN_URL = "https://www2.census.gov/programs-surveys/gov-finances/datasets/historical/hist_fin.zip"
CENSUS_OLD_ESTIMATES_URL = "https://www2.census.gov/govs/estimate/"
CENSUS_TABLES_URL = "https://www2.census.gov/programs-surveys/gov-finances/tables/"
US_ARCHIVE_BEGIN_MARKER = "# BEGIN AUTO-GENERATED U.S. ARCHIVE DATA"
US_ARCHIVE_END_MARKER = "# END AUTO-GENERATED U.S. ARCHIVE DATA"

# Unit multipliers. StatCan tables and FRED's GFDEBTN/SLGSDODNS report millions;
# FRED's GDP and interest series report billions; its population, thousands.
THOUSAND = 1_000
MILLION = 1_000_000
BILLION = 1_000_000_000

# ---------------------------------------------------------------------------
# Column names shared across data and plotting layers
# ---------------------------------------------------------------------------

DATE_COLUMN = "DATE"
YIELD_COLUMNS = ("3-Month", "2-Year", "5-Year", "10-Year", "30-Year")

GDP_COLUMN = "TTM Nominal GDP ($)"
CDN_DEBT_COLUMN = "Total Canadian Debt ($)"
# Before 1933 only federal debt is recorded; it is a separate curve, not part of
# the aggregate, because it is only about half of it (53 % in 1933).
CDN_FEDERAL_DEBT_COLUMN = "Federal Canadian Debt ($)"
POPULATION_COLUMN = "Population"
CDN_INTEREST_COLUMN = "TTM Interest Payable ($)"
US_DEBT_COLUMN = "Total Aggregate Debt ($)"
US_INTEREST_COLUMN = "TTM Interest Payments ($)"

# ---------------------------------------------------------------------------
# Chart styling and per-country metadata
# ---------------------------------------------------------------------------

YIELD_LINE_STYLE = {"linewidth": 1.2, "alpha": 0.9}

# The same three semantic macro curves are drawn on both country charts, plus
# (Canada only) federal debt alone before the aggregate begins: dotted, so the
# drop in level where it hands over reads as a change of coverage.
MACRO_PLOT_STYLES = {
    "debt": {"color": "black", "linewidth": 2.5, "drawstyle": "steps-post"},
    "federal_debt": {"color": "black", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": ":"},
    "gdp": {"color": "darkgreen", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "-."},
    "interest": {"color": "darkred", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "--"},
}

# Components of debt and interest (--debt:LETTERS / --interest:LETTERS): f
# federal, n non-federal (provincial + local), p provincial or state (s is
# accepted for it), m municipal (local). Drawing order, and one colour each;
# debt lines are solid and interest lines dashed, as for the aggregates. The
# colours are darker than the thin yield lines' and avoid GDP's green.
COMPONENT_LETTERS = ("f", "n", "p", "m")
COMPONENT_SYNONYMS = {"s": "p"}
COMPONENT_COLORS = {"f": "#1f3a93", "n": "#8b4513", "p": "#d35400", "m": "#008080"}
MACRO_PLOT_STYLES.update(
    {
        f"{kind}_{letter}": {"color": color, "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": linestyle}
        for kind, linestyle in (("debt", "-"), ("interest", "--"))
        for letter, color in COMPONENT_COLORS.items()
    }
)


def component_column(kind: str, letter: str) -> str:
    """Return the column holding one level's ``kind`` ("debt" or "interest"), e.g. ``"Debt [f] ($)"``."""
    return f"{kind.capitalize()} [{letter}] ($)"


# Wording for -r (debt and interest as a percentage of GDP) and -p (per
# person): the right axis label, and what is added to the legend labels and
# to the title's macro phrase. The -p axis label is per country (its currency).
RELATIVE_AXIS_LABEL = "Percent of TTM GDP (Log Scale)"
RELATIVE_LABEL_SUFFIX = " / TTM GDP"
RELATIVE_TITLE_SUFFIX = " as % of TTM GDP"
PER_CAPITA_SUFFIX = " per Capita"
# Ends the title when debt or interest is drawn for more than one level of
# government; the legend names the levels (Terry, 2026-09-27: the title need
# not repeat the legend where that makes it awkward).
LEVELS_TITLE_SUFFIX = " by Level of Government"


def _measure_suffixes(config: PlotConfig) -> tuple[str, str]:
    """Return ``(legend_suffix, title_suffix)`` for the measure ``config`` asks for ("" for dollars)."""
    if config.relative:
        return RELATIVE_LABEL_SUFFIX, RELATIVE_TITLE_SUFFIX
    if config.per_capita:
        return PER_CAPITA_SUFFIX, PER_CAPITA_SUFFIX
    return "", ""


def _with_suffix(label: str, suffix: str) -> str:
    """Add ``suffix`` to a legend label, before a closing parenthetical such as "(pre-1933)"."""
    if suffix and label.endswith(")") and " (" in label:
        head, _, tail = label.rpartition(" (")
        return f"{head}{suffix} ({tail}"
    return label + suffix


@dataclass(frozen=True)
class CountryMetadata:
    """Labels, column names and title wording for one country's chart.

    ``*_label`` strings appear in the legend; ``*_title`` strings are the
    phrases assembled into the chart title for whichever series were drawn.
    Under -r and -p both are qualified (see ``_measure_suffixes``).
    """

    key: str
    country_name: str
    currency_prefix: str
    currency_label: str
    per_capita_label: str
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
    # Federal debt alone, for the years before the aggregate exists (None: no such curve).
    federal_debt_column: str | None = None
    federal_debt_label: str = ""
    federal_debt_title: str = ""
    # Components (--debt:LETTERS): the name of each level, and the wording
    # around a name ("{}" is replaced by it, or in a title by nothing).
    level_names: tuple[tuple[str, str], ...] = (
        ("f", "Federal"), ("n", "Non-federal"), ("p", "Provincial"), ("m", "Municipal"),
    )
    component_debt_label: str = "{} Public Debt"
    component_interest_label: str = "{} TTM Interest Payable"
    component_debt_title: str = "{} Public Debt"
    component_interest_title: str = "{} Interest Outlays"

    def level_name(self, letter: str) -> str:
        """Return the name of level ``letter`` ("f" -> "Federal")."""
        return dict(self.level_names)[letter]

    def macro_axis_label(self, config: PlotConfig) -> str:
        """Return the right (macro) axis label for the measure ``config`` asks for."""
        if config.relative:
            return RELATIVE_AXIS_LABEL
        return self.per_capita_label if config.per_capita else self.currency_label

    def macro_specs(self, config: PlotConfig) -> tuple[tuple[str, bool, str, str], ...]:
        """Return ``(style_key, enabled, column, label)`` for each macro curve, in legend order.

        Under -r GDP is the denominator and is not drawn itself. With
        components chosen, each of debt and interest is one line per level
        instead of the aggregate (and the federal-only debt before 1933 is
        simply the start of the federal line).
        """
        suffix = _measure_suffixes(config)[0]
        if config.components:
            specs = [
                (f"debt_{letter}", config.include_debt, component_column("debt", letter),
                 self.component_debt_label.format(self.level_name(letter)))
                for letter in config.components
            ]
            specs.append(("gdp", config.draws_gdp, self.gdp_column, self.gdp_label))
            specs += [
                (f"interest_{letter}", config.include_interest, component_column("interest", letter),
                 self.component_interest_label.format(self.level_name(letter)))
                for letter in config.components
            ]
            return tuple((key, enabled, column, _with_suffix(label, suffix)) for key, enabled, column, label in specs)
        specs = [
            ("debt", config.include_debt, self.debt_column, self.debt_label),
            ("gdp", config.draws_gdp, self.gdp_column, self.gdp_label),
            ("interest", config.include_interest, self.interest_column, self.interest_label),
        ]
        if self.federal_debt_column is not None:
            # Chosen with the debt curve; listed straight after it.
            specs.insert(1, ("federal_debt", config.include_debt, self.federal_debt_column, self.federal_debt_label))
        return tuple((key, enabled, column, _with_suffix(label, suffix)) for key, enabled, column, label in specs)

    def title_for(self, *, yields_drawn: bool, macro_keys_drawn: Iterable[str], config: PlotConfig) -> str:
        """Compose the chart title from the series that were actually drawn.

        Parts are joined with commas and a final ampersand, e.g.
        ``"U.S. Treasury Yields, Aggregate US Public Debt & TTM GDP"``.
        Federal debt alone is named only when the aggregate is not drawn.
        Under -r and -p the macro phrase is qualified and set off from the
        yields by a semicolon, since the qualifier does not apply to them:
        ``"CDN Benchmark Yields; Aggregate CDN Public Debt & Interest Outlays
        as % of TTM GDP"``.
        """
        drawn = set(macro_keys_drawn)
        if "debt" in drawn:
            drawn.discard("federal_debt")
        macro_titles = {
            "debt": self.debt_title,
            "federal_debt": self.federal_debt_title,
            "gdp": self.gdp_title,
            "interest": self.interest_title,
        }
        # Component lines ("debt_f", "interest_p" …). A single level is named,
        # "Federal CDN Public Debt"; with more, the legend names them and the
        # title ends "… by Level of Government" (after the -r / -p qualifier,
        # set off by a comma). Debt and interest split alike are named once,
        # after GDP: "TTM GDP & CDN Public Debt and Interest Outlays by Level
        # of Government", "Federal CDN Public Debt and Interest Outlays".
        levels = {
            kind: [letter for letter in COMPONENT_LETTERS if f"{kind}_{letter}" in drawn] for kind in ("debt", "interest")
        }
        split = [kind for kind in ("debt", "interest") if levels[kind]]
        by_level = any(len(levels[kind]) > 1 for kind in split)
        names = {kind: "" if by_level else self.level_name(levels[kind][0]) for kind in split}
        templates = {"debt": self.component_debt_title, "interest": self.component_interest_title}
        if len(split) == 2 and names["debt"] == names["interest"]:
            both = templates["debt"].format(names["debt"]).strip() + " and " + templates["interest"].format("").strip()
            macro_parts = ([self.gdp_title] if "gdp" in drawn else []) + [both]
        else:
            for kind in split:
                drawn.add(kind)
                macro_titles[kind] = templates[kind].format(names[kind]).strip()
            macro_parts = [macro_titles[key] for key in macro_titles if key in drawn]
        levels_suffix = LEVELS_TITLE_SUFFIX if by_level else ""
        yield_parts = [self.yield_title] if yields_drawn else []
        title_suffix = _measure_suffixes(config)[1]
        if title_suffix and macro_parts:
            macro_phrase = _join_title_parts(macro_parts) + title_suffix + ("," if levels_suffix else "") + levels_suffix
            return "; ".join(yield_parts + [macro_phrase])
        parts = yield_parts + macro_parts
        if not parts:
            return f"{self.country_name}: no series selected"
        return _join_title_parts(parts) + levels_suffix


def _join_title_parts(parts: list[str]) -> str:
    """Join title phrases with commas and a final ampersand: "A, B & C"."""
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} & {parts[-1]}"


CDN = CountryMetadata(
    key="cdn",
    country_name="Canada",
    currency_prefix="C$",
    currency_label="Nominal Units (CAD – Log Scale)",
    per_capita_label="Nominal Units per Capita (CAD – Log Scale)",
    yield_title="CDN Benchmark Yields",
    debt_column=CDN_DEBT_COLUMN,
    debt_label="Aggregate CDN Public Debt",
    debt_title="Aggregate CDN Public Debt",
    interest_column=CDN_INTEREST_COLUMN,
    interest_label="TTM Interest Payable",
    federal_debt_column=CDN_FEDERAL_DEBT_COLUMN,
    federal_debt_label="Federal CDN Public Debt (pre-1933)",
    federal_debt_title="Federal CDN Public Debt",
    component_debt_label="{} CDN Public Debt",
    component_debt_title="{} CDN Public Debt",
)

US = CountryMetadata(
    key="us",
    country_name="United States",
    currency_prefix="$",
    currency_label="Nominal Units (USD – Log Scale)",
    per_capita_label="Nominal Units per Capita (USD – Log Scale)",
    yield_title="U.S. Treasury Yields",
    debt_column=US_DEBT_COLUMN,
    debt_label="Aggregate US Public Debt",
    debt_title="Aggregate US Public Debt",
    interest_column=US_INTEREST_COLUMN,
    interest_label="TTM Interest Payable",
    # FRED has federal and state-and-local debt and interest quarterly; state
    # and local apart are yearly (Census debt, BEA interest) and split the
    # quarterly combined figures (see us_data).
    level_names=(("f", "Federal"), ("n", "State & Local"), ("p", "State"), ("m", "Local")),
    component_debt_label="{} US Public Debt",
    component_debt_title="{} US Public Debt",
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
    # --debt:LETTERS / --interest:LETTERS: the levels of government to split
    # debt and interest into, as letters from COMPONENT_LETTERS in that order
    # ("" = the aggregate lines).
    components: str = ""
    # -r: debt and interest as a percentage of TTM GDP, on a log percent axis;
    # GDP itself is not drawn.
    relative: bool = False
    # -p: GDP, debt and interest per person, still on a log dollar axis. At
    # most one of relative / per_capita is set (options.config_from_choices).
    per_capita: bool = False
    # -l: print each drawn line's last value at its right-hand end, in the
    # line's colour; the date axis is widened to make room (ratesplot.endlabels).
    end_labels: bool = False
    # Open the interactive window (ratesplot.gui) rather than plain matplotlib windows.
    gui: bool = True

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
    def line_scale(self) -> float:
        """Return the factor applied to line widths, grid lines, tick marks and the axes frame.

        On canvases larger than the default it equals ``font_scale``, so a
        4096 px render is a true enlargement rather than big text over
        hairlines. It never goes below 1.0: the default yield lines are
        1.2 pt, only about 1.7 px, and thinning them on a small canvas
        would make them fade.
        """
        return max(1.0, self.font_scale)

    @property
    def draws_gdp(self) -> bool:
        """True when the GDP curve is drawn: selected, and not the -r denominator."""
        return self.include_gdp and not self.relative

    @property
    def needs_gdp(self) -> bool:
        """True when GDP must be fetched: to draw it, or to divide debt or interest by it (-r)."""
        return self.include_gdp or (self.relative and (self.include_debt or self.include_interest))

    @property
    def has_macro_series(self) -> bool:
        """True when at least one curve on the right (log) axis is drawn."""
        return self.include_debt or self.draws_gdp or self.include_interest

    @property
    def has_explicit_macro_limits(self) -> bool:
        """True when the user pinned either end of the right axis."""
        return self.macro_bottom is not None or self.macro_top is not None

    @property
    def default_macro_ylim(self) -> tuple[float, float]:
        """Return the right axis's fallback limits, in the units of its measure."""
        if self.relative:
            return DEFAULT_RELATIVE_YLIM
        return DEFAULT_PER_CAPITA_YLIM if self.per_capita else DEFAULT_MACRO_YLIM
