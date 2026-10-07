"""Application constants, per-country chart metadata, HTTP session, and PlotConfig.

Everything tunable lives here so the data and plotting modules contain logic
only. Numeric literals that affect chart appearance (padding, font sizes,
default limits) are deliberately centralised rather than scattered through the
plot functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Canvas and date defaults
# ---------------------------------------------------------------------------

# Without --start the chart begins where the last of the chosen curves begins
# (plotting.resolve_start). It is found by preparing the data from this date;
# it is also the first date the page's calendar and its "Earlier" button
# reach. Terry, 2026-10-06 (batch 3): as early as pandas allows, with a
# small margin. pandas' timestamps (nanoseconds) begin 1677-09-21; 1680-01-01
# leaves two years, and precedes every series: the UK's national debt (1691)
# and GDP (1700), Canada's from Confederation (1867). Spans longer than
# pandas' 292-year Timedelta are added as date offsets
# (frontend.move_dates_updates). Before batch 3 it was 1867-01-01; no Canadian
# or U.S. curve begins before that, so their automatic start is as it was.
EARLIEST_DATA_START = pd.Timestamp("1680-01-01")
# Without --start the chart begins no earlier than this date (Terry,
# 2026-10-06: he kept setting the start by hand, "so maybe just making the
# default 2000 is just as good as anything else"). The start is the later of
# this date and the first date on which every chosen curve has data, so a
# curve that begins after it is still not cut at the left. A chart ending
# before this date takes the automatic start alone. Any date before
# EARLIEST_DATA_START (e.g. "1500-01-01") restores the fully automatic start
# of 2026-09-29, as Terry asked: a setting, not a hard-coded year.
DEFAULT_START_FLOOR = pd.Timestamp("2000-01-01")
# The shortest window: --end at least this many days after --start.
MIN_WINDOW_DAYS = 7
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

# Regression segments (--reg, ratesplot.regression): how far a piece's line
# may lie from any observation it covers, as a percentage of the right axis's
# height (in log terms, so the same distance on the picture everywhere). 1 %
# is about 12 px on the default canvas, a gap the eye still reads as "on the
# curve" next to a line this thick. --reg:TOL sets it
# (PlotConfig.regression_tolerance).
DEFAULT_REGRESSION_TOLERANCE_PCT = 1.0
# ...but never less than this percentage of the value (ln 1.01 for 1 %), with
# --reg:TOL too: below it, on a window of a year or two, the pieces would
# follow each quarter's noise.
MIN_REGRESSION_TOLERANCE_PCT = 1.0

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

HTTP_TIMEOUT_SECONDS = 90
HTTP_POST_TIMEOUT_SECONDS = 60
# Statistics Canada's table ZIPs. A requests timeout is the longest silence
# allowed (before the first byte, or between two), not the whole download.
# Measured 2026-09-30 over 42 downloads of the 4 live and 11 archived tables:
# the first byte came within 1.4 s and no silence lasted 0.2 s. So 30 s is
# some twenty times the worst seen, and a request StatCan accepts and never
# answers (one did, 2026-09-30) is retried after 30 s rather than 120.
STATCAN_TIMEOUT_SECONDS = 30
DEFAULT_GET_RETRIES = 4
DEFAULT_FRED_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 2

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BOC_GROUP_URL = "https://www.bankofcanada.ca/valet/observations/group/{group}/csv"
BOC_SERIES_URL = "https://www.bankofcanada.ca/valet/observations/{series_code}/csv"
STATCAN_TABLE_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/{table_id}-eng.zip"
STATCAN_WDS_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en;q=0.9",
}
# Single session for the Canadian providers (Bank of Canada, Statistics Canada)
# so TCP connections are reused across the several downloads per run. FRED is
# deliberately *not* routed through this session: see ``http.fetch_fred_csv``.
CANADIAN_SESSION = requests.Session()
CANADIAN_SESSION.headers.update(_BROWSER_HEADERS)
# The sources of the latest values (--cur, ``ratesplot.latest``): U.S. Treasury
# and the quote feed. A session of their own, so a problem there cannot
# disturb the Canadian downloads.
LATEST_SESSION = requests.Session()
LATEST_SESSION.headers.update(_BROWSER_HEADERS)
# The UK's sources (Bank of England, ONS): a session of their own, as the
# latest values have, with a browser's headers, which both require.
UK_SESSION = requests.Session()
UK_SESSION.headers.update(_BROWSER_HEADERS | {"Accept-Language": "en-GB,en;q=0.9"})
# Germany's sources (Deutsche Bundesbank, the ECB, Eurostat; batch 4): a
# session of their own, with a browser's headers, as they were measured with
# from Terry's laptop (2026-10-06).
DE_SESSION = requests.Session()
DE_SESSION.headers.update(_BROWSER_HEADERS | {"Accept-Language": "en-GB,en;q=0.9,de;q=0.8"})
# The quote feed is an extra: one quick try and one retry, then the chart is
# drawn without it (with a warning).
LATEST_QUOTE_TIMEOUT_SECONDS = 20
LATEST_QUOTE_RETRIES = 2

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

# The UK (batch 3, 2026-10-06): the Bank of England's statistical database
# (IADB: gilt yields, Bank Rate, quoted mortgage rates) and the Office for
# National Statistics (debt, interest, GDP, population), both read live; the
# Bank's "A millennium of macroeconomic data" (no longer updated) baked into
# ``ratesplot/uk_archive_data.py``. Spans measured from Terry's laptop
# 2026-10-06 (tools/uk_fixtures/README.md).
BOE_IADB_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
# The database answers a request from this date; from 1950 or 1960 it
# redirects to its error page 905 instead (measured 2026-10-06). No series
# read from it begins before 1975.
BOE_IADB_FROM = "01/Jan/1975"
ONS_TIMESERIES_URL = "https://www.ons.gov.uk{path}/data"
UK_MILLENNIUM_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/research-datasets/"
    "a-millennium-of-macroeconomic-data-for-the-uk.xlsx"
)
UK_ARCHIVE_BEGIN_MARKER = "# BEGIN AUTO-GENERATED UK ARCHIVE DATA"
UK_ARCHIVE_END_MARKER = "# END AUTO-GENERATED UK ARCHIVE DATA"

# Germany (batch 4, 2026-10-07): the Deutsche Bundesbank's time-series
# database (yields), the ECB Data Portal (key interest rates, bank lending
# rates, annual debt) and Eurostat (quarterly debt, interest, GDP,
# population), all read live; the Bundesbank's yield on public debt
# securities before 1972 (a frozen series) baked into
# ``ratesplot/de_archive_data.py``. Spans measured from Terry's laptop
# 2026-10-06 (tools/de_fixtures/README.md).
BUNDESBANK_SERIES_URL = "https://api.statistiken.bundesbank.de/rest/download/{flow}/{key}"
# The Bundesbank's old download, where the series of its former database
# (BBK01) that were never moved to the new one are kept, frozen at 2020-04.
BUNDESBANK_OLD_SERIES_URL = "https://www.bundesbank.de/statistic-rmi/StatisticDownload"
ECB_SERIES_URL = "https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
EUROSTAT_DATASET_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}"
DE_ARCHIVE_BEGIN_MARKER = "# BEGIN AUTO-GENERATED GERMAN ARCHIVE DATA"
DE_ARCHIVE_END_MARKER = "# END AUTO-GENERATED GERMAN ARCHIVE DATA"

# The latest values (--cur), newer than the regular series (see ratesplot.latest).
# U.S. Treasury's daily par yield curve: the source of FRED's DGS series,
# posted the same afternoon, where FRED follows a business day or more later.
TREASURY_YIELD_CURVE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all"
)
# U.S. Treasury's "Debt to the Penny": total public debt outstanding each
# business day, one day behind; FRED's quarterly GFDEBTN is its quarter-end value.
TREASURY_DEBT_TO_PENNY_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny"
# CNBC's quote feed (the one cnbc.com's own pages read): intraday government
# bond yields for both countries, a few minutes old. Unofficial, so it may
# change without notice; the chart is then drawn without it, with a warning.
CNBC_QUOTE_URL = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
# Quotes are timed in New York / Toronto time (both markets' hours).
MARKET_TIMEZONE = "America/Toronto"
# Bank of Canada: Government of Canada treasury bills and domestic marketable
# bonds outstanding, the total (real return bonds with their inflation
# adjustment), dollars, each business day, a few days behind; from 2025. It is
# the Fiscal Monitor's "market debt payable in Canadian currency" to within
# 0.5 %, months sooner. The series holds the current year; each earlier year
# is the same name with "_YYYY" added.
BOC_CDN_MARKET_DEBT_SERIES = "DOM_DBT_OUTSTANDING_AMOUNT_INF_ADJ_TOTAL"

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

# Yield terms (--yields:LIST): the short form a term is written in -> its
# column. Shortest first, which is the drawing and legend order. The five of
# YIELD_COLUMNS are drawn by default and are the only Canadian ones; the U.S.
# has all ten (FRED's DGS series; see us_data.US_YIELD_SERIES).
YIELD_TERMS = {
    "1m": "1-Month",
    "3m": "3-Month",
    "6m": "6-Month",
    "1y": "1-Year",
    "2y": "2-Year",
    "5y": "5-Year",
    "7y": "7-Year",
    "10y": "10-Year",
    "20y": "20-Year",
    "30y": "30-Year",
}
DEFAULT_YIELD_TERMS = ("3m", "2y", "5y", "10y", "30y")

GDP_COLUMN = "TTM Nominal GDP ($)"
CDN_DEBT_COLUMN = "Total Canadian Debt ($)"
# Before 1933 only federal debt is recorded; it is a separate curve, not part of
# the aggregate, because it is only about half of it (53 % in 1933).
CDN_FEDERAL_DEBT_COLUMN = "Federal Canadian Debt ($)"
POPULATION_COLUMN = "Population"
CDN_INTEREST_COLUMN = "TTM Interest Payable ($)"
US_DEBT_COLUMN = "Total Aggregate Debt ($)"
US_INTEREST_COLUMN = "TTM Interest Payments ($)"
UK_DEBT_COLUMN = "Total UK Debt (£)"
# Before general government debt begins (1975), the national debt alone: a
# curve of its own, dotted, as Canada's federal debt before 1933 is.
UK_NATIONAL_DEBT_COLUMN = "UK National Debt (£)"
# The UK's headline debt, beside the gross (Terry, 2026-10-06: "add net too").
UK_NET_DEBT_COLUMN = "UK Net Debt (£)"
UK_INTEREST_COLUMN = "TTM UK Interest Paid (£)"
DE_DEBT_COLUMN = "Total German Debt (€)"
DE_INTEREST_COLUMN = "TTM German Interest Paid (€)"

# ---------------------------------------------------------------------------
# Chart styling and per-country metadata
# ---------------------------------------------------------------------------

YIELD_LINE_STYLE = {"linewidth": 1.2, "alpha": 0.9}
# The left axis's label (PlotConfig.left_axis_label): with the yields drawn,
# or else with only other rates on it (--policy, --mortgages).
YIELD_AXIS_LABEL = "Bond Yield (%)"
RATE_AXIS_LABEL = "Rate (%)"
# --policy (ratesplot.rates): a step line, as a policy rate holds until it is
# changed; dark brown, a colour no yield term has, a little thicker than the
# yields so it reads as the anchor of the curve. (Dark gold at first: on
# real data it was hard to tell from the 2-year's orange where they cross;
# Terry chose dark brown, 2026-10-06.)
POLICY_RATE_STYLE = {"color": "#5c4033", "linewidth": 1.8, "drawstyle": "steps-post"}
# --mortgages (ratesplot.rates): the terms, as written on the command line ->
# the country that has them and the yield term whose colour they take (None:
# no yield of that term, so a colour of their own, MORTGAGE_OWN_COLORS). In
# this order on the chart. "5v" is Canada's 5-year variable rate; "prime" the
# chartered banks' prime rate, which Canadian variable-rate mortgages are
# priced from (Terry, 2026-10-06: the 3-year and 6-month variable mortgages
# are popular in Canada, and have no series of their own; prime in their place).
MORTGAGE_TERMS = {
    "30": ("us", "30y"),
    "15": ("us", None),
    "5": ("cdn", "5y"),
    "3": ("cdn", None),
    "1": ("cdn", "1y"),
    "5v": ("cdn", "5y"),
    "prime": ("cdn", None),
    # The UK (batch 3): the Bank of England's quoted rates for new loans at
    # 75 % loan to value, fixed for 2, 3 or 5 years, and the standard
    # variable rate (SVR) a fixed rate reverts to. Written with an "f" so
    # they are not Canada's 5 and 3 (posted rates of other lenders).
    "2f": ("uk", "2y"),
    "3f": ("uk", None),
    "5f": ("uk", "5y"),
    "svr": ("uk", None),
    # Germany (batch 4): the MFI interest rates on new housing loans to
    # households (ECB), by initial rate fixation: over 1 and up to 5 years,
    # over 5 and up to 10, over 10. Named by the band, as the statistics
    # are; coloured as the yield of the band's top (the 20-year for over 10).
    "1-5": ("de", "5y"),
    "5-10": ("de", "10y"),
    "over10": ("de", "20y"),
}
DEFAULT_MORTGAGE_TERMS = ("30", "5")
# The 3-year fixed has Canada's 3-year's colour (the same kind of loan, on
# another chart); the SVR the prime rate's, a variable base rate as it is.
MORTGAGE_OWN_COLORS = {"15": "#556b2f", "3": "#2f4f4f", "prime": "#696969", "3f": "#2f4f4f", "svr": "#696969"}
# Dashed steps (a posted or surveyed rate holds until the next), thinner than
# the policy rate; the variable rate dash-dotted, as it shares the 5-year's colour.
MORTGAGE_STYLE = {"linewidth": 1.5, "linestyle": "--", "drawstyle": "steps-post"}
VARIABLE_MORTGAGE_STYLE = MORTGAGE_STYLE | {"linestyle": "-."}
# The prime rate: dotted, a base rate rather than a mortgage of a term.
PRIME_RATE_STYLE = MORTGAGE_STYLE | {"linestyle": ":", "linewidth": 1.8}
# --spreads[:LIST] (ratesplot.rates): pairs of yield terms, the first less the
# second, in percentage points. Drawn thick in colours no yield term, rate or
# right-axis curve has, one per pair in the order given (then round again),
# with their inverted stretches shaded (SPREAD_INVERSION_ALPHA). When one is
# drawn the left axis is "Yield and Spread (%)".
DEFAULT_SPREADS = (("10y", "2y"), ("10y", "3m"), ("30y", "10y"))
SPREAD_COLORS = ("#4b0082", "#c71585", "#000080", "#800000", "#008080")
SPREAD_STYLE = {"linewidth": 2.5}
SPREAD_AXIS_LABEL = "Yield and Spread (%)"


# ---------------------------------------------------------------------------
# Nations: what each one has (batch 2, 2026-10-06)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Nation:
    """What one nation's chart can draw, and how the command line names it.

    The capabilities decide the controls in its panel (only its own yield
    terms and mortgage terms are offered) and what a choice made for every
    nation means for it (a term it lacks is left off its chart, with a note).
    Adding a nation means one of these, its CountryMetadata, its fetchers
    and its ``plotting.Country``.
    """

    key: str  # as plotting.Country.key and CountryMetadata.key
    name: str  # in messages and on its panel
    codes: tuple[str, ...]  # the prefixes that name it on the command line ("--us:top:20t"), lower case
    yield_terms: tuple[str, ...]  # the YIELD_TERMS keys it has a series for
    adjective: str  # in messages: "no Canadian 7-Year yield"
    # Drawn when the command line names no nation. Canada and the U.S. are,
    # as they always were; a nation added later is drawn only when asked for
    # (batch 3: every existing command line draws what it drew).
    shown_by_default: bool = True
    # Its own value of a list in PER_NATION_FIELDS, used while that list is
    # at the default for every chart (_LIST_DEFAULTS), where the default
    # would draw little or nothing on its chart: (field, value) pairs. With
    # none, the default is simply viewed as far as its chart has the terms.
    own_defaults: tuple[tuple[str, object], ...] = ()
    # Codes that name it in --nations:LIST only, never in front of a value:
    # there they would spell another option (batch 4: "--de:fp" is --debt:fp).
    list_only_codes: tuple[str, ...] = ()

    @property
    def prefix(self) -> str:
        """The code written in front of its values in messages ("cdn", "ger"): its key, if that is a prefix."""
        prefixes = [code for code in self.codes if code not in self.list_only_codes]
        return self.key if self.key in prefixes else prefixes[0]

    @property
    def mortgage_terms(self) -> tuple[str, ...]:
        """The MORTGAGE_TERMS keys it has, in their order."""
        return tuple(term for term, (country, _yield) in MORTGAGE_TERMS.items() if country == self.key)

    @property
    def show_field(self) -> str:
        """The PlotConfig field that chooses its chart ("show_cdn")."""
        return f"show_{self.key}"

    def view(self, field: str, value: object) -> object:
        """Return what a list in ``field`` draws on this nation's chart: the yield and mortgage terms it has, the spreads of them.

        A term or pair it lacks is left off its chart (with a note), so two
        lists with the same view draw the same chart (a spread's colour goes
        by its place among those drawn, ``rates.yield_spreads``). The
        default list for every chart draws the nation's own default where it
        has one (``own_defaults``). Any other field is its value.
        """
        own = dict(self.own_defaults)
        if field in own and value == _LIST_DEFAULTS[field]:
            return own[field]
        if field == "yield_terms":
            return tuple(term for term in value if term in self.yield_terms)
        if field == "mortgage_terms":
            return tuple(term for term in value if term in self.mortgage_terms)
        if field == "spread_pairs":
            return tuple(pair for pair in value if all(term in self.yield_terms for term in pair))
        return value


# In drawing order, as plotting.COUNTRIES. The codes are ISO 3166 two-letter
# codes, plus the program's own "cdn".
NATIONS: tuple[Nation, ...] = (
    Nation("cdn", "Canada", ("ca", "cdn"), DEFAULT_YIELD_TERMS, "Canadian"),
    Nation("us", "United States", ("us",), tuple(YIELD_TERMS), "U.S."),
    # Batch 3: drawn only when asked for, so every existing command line
    # draws what it drew. Its own defaults, since the defaults for every
    # chart would draw two of its three yield terms, none of its mortgage
    # terms and none of its spreads: all three terms, the standard variable
    # rate (Terry, 2026-10-06: its history runs from 1939), the 10y-5y and
    # 20y-10y spreads.
    Nation(
        "uk", "United Kingdom", ("gb", "uk"), ("5y", "10y", "20y"), "UK", shown_by_default=False,
        own_defaults=(
            ("yield_terms", ("5y", "10y", "20y")),
            ("mortgage_terms", ("svr",)),
            ("spread_pairs", (("10y", "5y"), ("20y", "10y"))),
        ),
    ),
    # Batch 4: drawn only when asked for, as the UK is. Its code "de" names it
    # in --nations only: "--de" is --debt and "--de:fp" --debt:fp, so a value
    # for Germany alone is written "--ger:max:8" (or "--deu:max:8", its ISO
    # three-letter code; Terry, 2026-10-07: "--deu:max:6 is ok"). Its own defaults, which are
    # what the defaults for every chart draw on its chart (it has no 3-month
    # yield), given so that no default chart notes the 3-month's absence;
    # and its usual fixation, 5 to 10 years, as its mortgage rate.
    Nation(
        "de", "Germany", ("de", "ger", "deu"), ("1y", "2y", "5y", "7y", "10y", "20y", "30y"), "German",
        shown_by_default=False, list_only_codes=("de",),
        own_defaults=(
            ("yield_terms", ("2y", "5y", "10y", "30y")),
            ("mortgage_terms", ("5-10",)),
            ("spread_pairs", (("10y", "2y"), ("30y", "10y"))),
        ),
    ),
)


def nation_by_code(code: str, *, prefix: bool = False) -> Nation | None:
    """Return the nation a code names ("ca", "CDN", "us"), or None; with ``prefix``, only a code allowed in front of a value."""
    code = code.lower()
    return next(
        (nation for nation in NATIONS if code in nation.codes and not (prefix and code in nation.list_only_codes)), None
    )


def nation_by_key(key: str) -> Nation | None:
    """Return the nation with this key ("cdn"), or None."""
    return next((nation for nation in NATIONS if nation.key == key), None)


# The choices that can differ by nation (PlotConfig.for_nation): the yield
# axis's lists, and the axis limits (Top and Bottom are in each nation's own
# currency, so a limit right for one is wrong for the other).
PER_NATION_FIELDS = (
    "yield_terms", "mortgage_terms", "spread_pairs", "yield_ymin", "yield_ymax", "macro_bottom", "macro_top",
)
# The defaults of the lists a nation may have its own default of (Nation.own_defaults).
_LIST_DEFAULTS = {
    "yield_terms": DEFAULT_YIELD_TERMS,
    "mortgage_terms": DEFAULT_MORTGAGE_TERMS,
    "spread_pairs": DEFAULT_SPREADS,
}
# The colour of each yield term when the terms are chosen (--yields:LIST,
# --no-yields:LIST). The five default terms have the colours matplotlib's
# colour cycle gives them when all five are drawn (blue, orange, green, red,
# purple), and the other five the rest of that palette. Without a choice of
# terms the lines still take the cycle in drawing order, as they always have,
# so those charts are unchanged (plotting._yield_style).
TERM_COLORS = {
    "3m": "#1f77b4",
    "2y": "#ff7f0e",
    "5y": "#2ca02c",
    "10y": "#d62728",
    "30y": "#9467bd",
    "1m": "#8c564b",
    "6m": "#e377c2",
    "1y": "#7f7f7f",
    "7y": "#bcbd22",
    "20y": "#17becf",
}

# The same three semantic macro curves are drawn on both country charts, plus
# (Canada only) federal debt alone before the aggregate begins: dotted, so the
# drop in level where it hands over reads as a change of coverage.
MACRO_PLOT_STYLES = {
    "debt": {"color": "black", "linewidth": 2.5, "drawstyle": "steps-post"},
    "federal_debt": {"color": "black", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": ":"},
    "gdp": {"color": "darkgreen", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "-."},
    "interest": {"color": "darkred", "linewidth": 2.5, "drawstyle": "steps-post", "linestyle": "--"},
    # Net debt (the UK's, batch 3), drawn with gross debt: solid as debt is,
    # in slate grey, a colour no other right-axis curve has.
    "net_debt": {"color": "#708090", "linewidth": 2.5, "drawstyle": "steps-post"},
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


# --cur on a chart reaching today: each right-axis curve's stretch from its last
# data to today, projected like a debt clock (ratesplot.latest.project_to_now),
# is drawn straight and fainter in the curve's own colour and dashes; one grey
# legend entry says what the faint stretches are, and their end labels read "≈".
# Canadian debt, when steered by the Government of Canada's market debt, is
# not straight, and the entry says so.
PROJECTION_ALPHA = 0.4
PROJECTION_LABEL = "Projected at the past year's pace"
PROJECTION_LABEL_STEERED = "Projected; debt follows market debt"
PROJECTION_KEY_COLOR = "0.35"
PROJECTION_LABEL_PREFIX = "≈"

# --spreads: a spread's inverted stretches (below zero) are shaded between it
# and zero in its own colour, this opaque (plotting.add_rate_lines).
SPREAD_INVERSION_ALPHA = 0.2


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
    # Net debt, drawn with the (gross) debt curve (None: no such curve).
    net_debt_column: str | None = None
    net_debt_label: str = ""
    net_debt_title: str = ""
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
        if self.net_debt_column is not None:
            # Chosen with the debt curve; listed after it.
            specs.insert(1, ("net_debt", config.include_debt, self.net_debt_column, self.net_debt_label))
        if self.federal_debt_column is not None:
            # Chosen with the debt curve; listed straight after it.
            specs.insert(1, ("federal_debt", config.include_debt, self.federal_debt_column, self.federal_debt_label))
        return tuple((key, enabled, column, _with_suffix(label, suffix)) for key, enabled, column, label in specs)

    def title_for(
        self, *, yields_drawn: bool, macro_keys_drawn: Iterable[str], config: PlotConfig, rate_titles: Iterable[str] = ()
    ) -> str:
        """Compose the chart title from the series that were actually drawn.

        Parts are joined with commas and a final ampersand, e.g.
        ``"U.S. Treasury Yields, Aggregate US Public Debt & TTM GDP"``.
        Federal debt alone is named only when the aggregate is not drawn.
        Under -r and -p the macro phrase is qualified and set off from the
        yields by a semicolon, since the qualifier does not apply to them:
        ``"CDN Benchmark Yields; Aggregate CDN Public Debt & Interest Outlays
        as % of TTM GDP"``. ``rate_titles`` name the yield axis's other
        curves drawn (``ratesplot.rates``), after the yields: "U.S. Treasury
        Yields, Policy Rate, Aggregate US Public Debt …".
        """
        drawn = set(macro_keys_drawn)
        if "debt" in drawn:
            drawn.discard("federal_debt")
        macro_titles = {
            "debt": self.debt_title,
            "federal_debt": self.federal_debt_title,
            "net_debt": self.net_debt_title,
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
        yield_parts = ([self.yield_title] if yields_drawn else []) + list(dict.fromkeys(rate_titles))
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

UK = CountryMetadata(
    key="uk",
    country_name="United Kingdom",
    currency_prefix="£",
    currency_label="Nominal Units (GBP – Log Scale)",
    per_capita_label="Nominal Units per Capita (GBP – Log Scale)",
    # The Bank of England's fitted par yields (monthly averages of gilt
    # redemption yields before them): the title says which kind they are.
    yield_title="UK Gilt Yields (fitted par)",
    debt_column=UK_DEBT_COLUMN,
    debt_label="UK General Government Gross Debt",
    debt_title="UK General Government Gross Debt",
    interest_column=UK_INTEREST_COLUMN,
    # ONS NMYX: interest (and the odd dividend) the government sector pays
    # outside itself, so consolidated as the other aggregates are.
    interest_label="TTM Interest Paid (consolidated)",
    federal_debt_column=UK_NATIONAL_DEBT_COLUMN,
    federal_debt_label="UK National Debt (pre-1975)",
    federal_debt_title="UK National Debt",
    # ONS HF6W: public sector net debt excluding the public sector banks,
    # the UK's own headline measure, beside gross general government debt.
    net_debt_column=UK_NET_DEBT_COLUMN,
    net_debt_label="UK Public Sector Net Debt (ex banks)",
    net_debt_title="Net Debt",
    # Central and local government only: the devolved governments are inside
    # central government in the UK accounts, so there is no "p" level, and
    # the non-central level "n" is local government (ratesplot.uk_data).
    level_names=(("f", "Central"), ("n", "Local"), ("p", "Devolved"), ("m", "Local")),
    component_debt_label="{} UK Public Debt",
    component_debt_title="{} UK Public Debt",
)

DE = CountryMetadata(
    key="de",
    country_name="Germany",
    currency_prefix="€",
    currency_label="Nominal Units (EUR – Log Scale)",
    per_capita_label="Nominal Units per Capita (EUR – Log Scale)",
    # The Bundesbank's Svensson term structure: zero-coupon yields fitted to
    # the Federal securities' prices; the title says which kind they are.
    yield_title="German Federal Yields (Svensson fitted)",
    debt_column=DE_DEBT_COLUMN,
    debt_label="German General Government Gross Debt",
    debt_title="German General Government Gross Debt",
    interest_column=DE_INTEREST_COLUMN,
    # D.41 payable by general government, consolidated (less than the sum of
    # its subsectors' by 0.2 % on average, 2002-2026).
    interest_label="TTM Interest Payable (consolidated)",
    # Bund, Länder, Gemeinden (ratesplot.de_data); the social security funds
    # are no level of the chart.
    level_names=(("f", "Federal"), ("n", "Länder & Local"), ("p", "Länder"), ("m", "Local")),
    component_debt_label="{} German Public Debt",
    component_debt_title="{} German Public Debt",
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

    # None: DEFAULT_START_FLOOR, or the first date on which every chosen
    # curve has a value if that is later (Terry, 2026-09-29 and 2026-10-06),
    # found from the data by plotting.resolve_start before the charts are
    # prepared; nothing downstream of it sees None.
    start: pd.Timestamp | None = None
    end: pd.Timestamp = field(default_factory=_today)
    width_px: int = DEFAULT_CANVAS_PX[0]
    height_px: int = DEFAULT_CANVAS_PX[1]
    yield_ymin: float | None = None
    yield_ymax: float | None = None
    macro_bottom: float | None = None
    macro_top: float | None = None
    include_yield: bool = True
    # --policy: each country's policy rate on the yield axis (ratesplot.rates);
    # additive, not under the curve rule.
    policy_rates: bool = False
    # --mortgages[:LIST]: mortgage rates on the yield axis, the terms of
    # MORTGAGE_TERMS chosen (each drawn on the chart of its country); additive.
    mortgages: bool = False
    mortgage_terms: tuple[str, ...] = DEFAULT_MORTGAGE_TERMS
    # --spreads[:LIST]: yield spreads on the yield axis, each a pair of
    # YIELD_TERMS keys (the first less the second); additive. Their terms are
    # fetched even when the yields are not drawn.
    spreads: bool = False
    spread_pairs: tuple[tuple[str, str], ...] = DEFAULT_SPREADS
    # --yields:LIST / --no-yields:LIST: the yield terms drawn, as keys of
    # YIELD_TERMS in that order. A term a country has no series for is
    # simply not drawn on its chart (Canada has the five default terms).
    yield_terms: tuple[str, ...] = DEFAULT_YIELD_TERMS
    include_debt: bool = True
    include_gdp: bool = True
    include_interest: bool = True
    show_cdn: bool = True
    show_us: bool = True
    # The UK's chart (batch 3): only when asked for (--uk, --nations:gb).
    show_uk: bool = False
    # Germany's chart (batch 4): only when asked for (--ger, --nations:de).
    show_de: bool = False
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
    # --reg: fit each drawn right-axis curve with the fewest straight pieces on
    # its log axis, each labelled with its growth in %/yr (ratesplot.regression).
    regression: bool = False
    # --reg:TOL: the pieces' tolerance, in percent of the right axis's height;
    # None = DEFAULT_REGRESSION_TOLERANCE_PCT. Never below
    # MIN_REGRESSION_TOLERANCE_PCT of the value (ratesplot.regression).
    regression_tolerance: float | None = None
    # Choices for one nation only ("--us:top:20t"): (nation key, ((field,
    # value), ...)) for each nation with any, the fields among
    # PER_NATION_FIELDS. Every other nation, and every field not named,
    # takes the value above. See ``for_nation``.
    nation_settings: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = ()
    # --cur (on unless --no-cur): extend the regular series with the latest
    # values from faster sources, down to intraday quotes (ratesplot.latest).
    current: bool = True
    # Open the interactive window (ratesplot.gui) rather than plain matplotlib windows.
    gui: bool = True

    def for_nation(self, key: str) -> PlotConfig:
        """Return the config one nation's chart is drawn with: these choices, with that nation's own over them.

        A list still at the default for every chart takes the nation's own
        default, if it has one (``Nation.own_defaults``).
        """
        own: dict[str, object] = {}
        for nation, overrides in self.nation_settings:
            if nation == key and overrides:
                own = dict(overrides)
        nation = nation_by_key(key)
        for field_name, value in nation.own_defaults if nation is not None else ():
            if field_name not in own and getattr(self, field_name) == _LIST_DEFAULTS[field_name]:
                own[field_name] = value
        return replace(self, **own) if own else self

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
    def yield_columns(self) -> tuple[str, ...]:
        """Return the yield columns drawn, in drawing order; none when the yield curves are off."""
        return tuple(YIELD_TERMS[term] for term in self.yield_terms) if self.include_yield else ()

    @property
    def fetched_yield_columns(self) -> tuple[str, ...]:
        """Return the yield columns to fetch: those drawn, and those the spreads are taken from, in drawing order."""
        spread_terms = {term for pair in self.spread_pairs for term in pair} if self.spreads else set()
        drawn = set(self.yield_terms) if self.include_yield else set()
        return tuple(column for term, column in YIELD_TERMS.items() if term in drawn | spread_terms)

    @property
    def has_left_axis_series(self) -> bool:
        """True when at least one curve on the left (yield) axis is chosen: the yields, or a curve of ``ratesplot.rates``."""
        return self.include_yield or self.policy_rates or self.mortgages or self.spreads

    @property
    def left_axis_label(self) -> str:
        """Return the left axis's label for the curves chosen on it: "Rate (%)" when the yields are not among them."""
        return YIELD_AXIS_LABEL if self.include_yield else RATE_AXIS_LABEL

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
