"""The latest values (--cur, on unless --no-cur): newer than the regular series.

The regular sources publish with a lag: FRED's Treasury yields follow the
Treasury by a business day or more, the Bank of Canada posts each day's
benchmark yields after the day, and FRED's federal debt is quarterly and a
quarter or more behind. With --cur each is carried forward from a faster
source, and to the minute by intraday quotes:

* U.S. yields: the Treasury's own daily par yield curve
  (``TREASURY_YIELD_CURVE_URL``), the figures FRED's DGS series carry a day
  or more later, for the days FRED does not have yet.
* U.S. federal debt: the Treasury's Debt to the Penny, daily and one day
  behind, from the day after the quarter FRED's last GFDEBTN value closes.
* Both countries' yields: CNBC's intraday quotes, for a day after the last
  official one. They are market quotes of the moment, not closing figures,
  so they are provisional; the subtitle gives their time.

  The feed quotes the traded (on-the-run) issues, not the official fitted
  curves, and on 2026-09-25 its closes differed from the official ones by
  up to 6 basis points (U.S. 3-month -6, 2-year +5, 10- and 30-year +1;
  Canada about -1.5). Appended as they are, the quotes would put a false
  step of that size on each line. So when the official series ends on the
  business day before the quote, the day's move in the feed (its last
  price less its own previous close) is added to the official close; only
  otherwise is the quote used as it is.

Only what the regular series lack, up to ``config.end``, is added, and it
is fetched only then: a window ending in the past is drawn exactly as before.
A source that fails prints a warning and adds nothing. GDP, interest,
population and Canadian debt have no faster free source than the
quarterly statistics already used.
"""

from __future__ import annotations

import io

import pandas as pd

from .config import (
    CNBC_QUOTE_URL,
    DATE_COLUMN,
    LATEST_QUOTE_RETRIES,
    LATEST_QUOTE_TIMEOUT_SECONDS,
    MARKET_TIMEZONE,
    TREASURY_DEBT_TO_PENNY_URL,
    TREASURY_YIELD_CURVE_URL,
    PlotConfig,
)
from .http import latest_get

# Where ``plotting.prepare_us`` / ``prepare_cdn`` record, in the yields frame's
# ``attrs``, the time of the newest quote added (for the subtitle).
QUOTE_TIME_ATTR = "latest_quote_time"

# Chart column -> CNBC symbol, per country (``plotting.Country.key``).
_QUOTE_SYMBOLS = {
    "us": {"3-Month": "US3M", "2-Year": "US2Y", "5-Year": "US5Y", "10-Year": "US10Y", "30-Year": "US30Y"},
    "cdn": {"3-Month": "CA3M", "2-Year": "CA2Y", "5-Year": "CA5Y", "10-Year": "CA10Y", "30-Year": "CA30Y"},
}
# Chart column -> column of the Treasury's yield-curve file.
_TREASURY_COLUMNS = {"3-Month": "3 Mo", "2-Year": "2 Yr", "5-Year": "5 Yr", "10-Year": "10 Yr", "30-Year": "30 Yr"}
# A quoted yield outside this range (percent) is a bad print and is left out.
_PLAUSIBLE_YIELD = (-5.0, 50.0)
# Debt to the Penny must agree this closely (fraction) with FRED's value for
# the same quarter end, or the two are not the same measure and are not joined.
_DEBT_JOIN_TOLERANCE = 0.001
# The feed's quotes are at most this old: a Friday's last quotes on the Tuesday
# after a long weekend. A window ending earlier than that cannot use them, so
# they are not fetched for it.
_QUOTE_MAX_AGE_DAYS = 4


def _quotes_can_reach(config: PlotConfig) -> bool:
    """True when the window ends late enough for the day's quotes to fall in it."""
    today = pd.Timestamp.now(tz=MARKET_TIMEZONE).tz_localize(None).normalize()
    return config.end >= today - pd.Timedelta(days=_QUOTE_MAX_AGE_DAYS)


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------


def treasury_yields_after(last: pd.Timestamp, config: PlotConfig) -> pd.DataFrame:
    """Return the Treasury's daily par yields after ``last``, up to ``config.end``, as a ``DATE``-column frame.

    The Treasury publishes one CSV per calendar year, newest day first.
    """
    frames = []
    for year in range(last.year, config.end.year + 1):
        params = {"type": "daily_treasury_yield_curve", "field_tdr_date_value": year, "_format": "csv"}
        table = pd.read_csv(io.StringIO(latest_get(TREASURY_YIELD_CURVE_URL.format(year=year), params).text))
        missing = {"Date", *_TREASURY_COLUMNS.values()} - set(table.columns)
        if missing:
            raise ValueError(f"the {year} yield-curve file has no column {sorted(missing)}")
        frame = pd.DataFrame({DATE_COLUMN: pd.to_datetime(table["Date"], format="%m/%d/%Y")})
        for column, source in _TREASURY_COLUMNS.items():
            frame[column] = pd.to_numeric(table[source], errors="coerce")
        frames.append(frame)
    data = pd.concat(frames).sort_values(DATE_COLUMN)
    return data.loc[(data[DATE_COLUMN] > last) & (data[DATE_COLUMN] <= config.end)].reset_index(drop=True)


def treasury_debt_from(first: pd.Timestamp, config: PlotConfig) -> pd.Series:
    """Return total public debt outstanding (dollars) each business day from ``first`` to ``config.end``."""
    params = {
        "filter": f"record_date:gte:{first:%Y-%m-%d}",
        "fields": "record_date,tot_pub_debt_out_amt",
        "sort": "record_date",
        "page[size]": 10000,
    }
    rows = latest_get(TREASURY_DEBT_TO_PENNY_URL, params).json()["data"]
    debt = pd.Series(
        [float(row["tot_pub_debt_out_amt"]) for row in rows],
        index=pd.DatetimeIndex([row["record_date"] for row in rows], name=DATE_COLUMN),
        dtype=float,
    )
    return debt.loc[debt.index <= config.end]


def _percent(text: object) -> float:
    """Parse a quoted yield such as ``"4.912%"``; raises ValueError (or TypeError) if it is not one."""
    return float(str(text).strip().rstrip("%"))


def intraday_yields_after(
    country: str, official: pd.Series, last: pd.Timestamp, config: PlotConfig
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Return one country's intraday yields dated after ``last`` (and not after ``config.end``).

    ``official`` holds the last official value of each yield column, on
    ``last``. Where ``last`` is the business day before a quote's day, the
    value is that official close plus the day's move in the feed; otherwise
    the quote itself (see the module docstring).

    Returns a date-indexed frame, one row per quote date (normally just
    today), and the time of the newest quote used, in ``MARKET_TIMEZONE``.
    A quote is dated by its own time there: at a weekend the feed still
    gives Friday's last quotes, which are then no newer than the official
    Friday figures and are left out. Fetched afresh every time (not cached),
    so each drawing shows the quotes of the moment.
    """
    symbols = _QUOTE_SYMBOLS[country]
    params = {
        "symbols": "|".join(symbols.values()),
        "requestMethod": "itv",
        "noform": "1",
        "partnerId": "2",
        "fund": "1",
        "exthrs": "1",
        "output": "json",
        "events": "1",
    }
    response = latest_get(
        CNBC_QUOTE_URL, params, cached=False, max_retries=LATEST_QUOTE_RETRIES, timeout=LATEST_QUOTE_TIMEOUT_SECONDS
    )
    quotes = response.json()["FormattedQuoteResult"]["FormattedQuote"]
    if isinstance(quotes, dict):  # a single symbol comes back unlisted
        quotes = [quotes]
    by_symbol = {quote.get("symbol"): quote for quote in quotes}

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    newest: pd.Timestamp | None = None
    for column, symbol in symbols.items():
        quote = by_symbol.get(symbol)
        try:
            value = _percent(quote["last"])
            when = pd.Timestamp(quote["last_time"]).tz_convert(MARKET_TIMEZONE)
        except (TypeError, KeyError, ValueError):
            print(f"  Warning: no usable quote for {symbol}; its line ends at the last official value.")
            continue
        if not _PLAUSIBLE_YIELD[0] <= value <= _PLAUSIBLE_YIELD[1]:
            print(f"  Warning: quote {symbol} = {value} is not a plausible yield; left out.")
            continue
        day = when.tz_localize(None).normalize()
        if not last < day <= config.end:
            continue
        close = official.get(column)
        if day - pd.offsets.BDay(1) == last and close is not None and pd.notna(close):
            try:
                value = float(close) + value - _percent(quote["previous_day_closing"])
            except (KeyError, TypeError, ValueError):
                pass  # no previous close in the feed: the quote as it is
        rows.setdefault(day, {})[column] = value
        newest = when if newest is None else max(newest, when)
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return frame.reindex(columns=[column for column in symbols if column in frame.columns]), newest


# ---------------------------------------------------------------------------
# Extending the regular series
# ---------------------------------------------------------------------------


def extend_us_yields(yields: pd.DataFrame, config: PlotConfig) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Add to FRED's yields (a ``DATE``-column frame) the Treasury's newer days, then the day's quotes.

    Returns the yields and the time of the newest quote added (None if none).
    New rows are not forward-filled: a tenor without a newer value ends at
    its last one rather than being carried to today.
    """
    if not config.current or yields.empty:
        return yields, None
    last = yields[DATE_COLUMN].max()
    if config.end <= last:
        return yields, None

    print("Fetching the latest U.S. yields (--cur) …")
    parts = [yields]
    official = yields.set_index(DATE_COLUMN).iloc[-1]
    try:
        treasury = treasury_yields_after(last, config)
    except Exception as exc:
        print(f"  Warning: U.S. Treasury daily yields unavailable ({exc}); FRED's, to {last.date()}, are the last official ones.")
    else:
        if not treasury.empty:
            parts.append(treasury)
            last = treasury[DATE_COLUMN].max()
            official = treasury.set_index(DATE_COLUMN).iloc[-1]

    quote_time = None
    if config.end > last and _quotes_can_reach(config):
        try:
            quotes, quote_time = intraday_yields_after("us", official, last, config)
        except Exception as exc:
            print(f"  Warning: intraday U.S. yield quotes unavailable ({exc}); drawn to {last.date()}.")
        else:
            if not quotes.empty:
                parts.append(quotes.rename_axis(DATE_COLUMN).reset_index())
    return pd.concat(parts, ignore_index=True), quote_time


def extend_cdn_yields(yields: pd.DataFrame, config: PlotConfig) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Add the day's quotes to the Bank of Canada's yields (date-indexed); see ``extend_us_yields``."""
    if not config.current or yields.empty:
        return yields, None
    last = yields.index.max()
    if config.end <= last or not _quotes_can_reach(config):
        return yields, None

    print("Fetching the latest Canadian yields (--cur) …")
    try:
        quotes, quote_time = intraday_yields_after("cdn", yields.iloc[-1], last, config)
    except Exception as exc:
        print(f"  Warning: intraday Canadian yield quotes unavailable ({exc}); drawn to {last.date()}.")
        return yields, None
    if quotes.empty:
        return yields, None
    return pd.concat([yields, quotes.rename_axis(yields.index.name)]), quote_time


def extend_us_federal_debt(federal: pd.DataFrame, column: str, config: PlotConfig) -> pd.DataFrame:
    """Continue FRED's quarterly federal debt (``[DATE, column]``, dollars) daily from Debt to the Penny.

    FRED dates each quarter's closing debt by the quarter's first day
    (GFDEBTN at 2026-01-01 is the Treasury's figure for 2026-03-31, exactly).
    So the daily figures join the day after that quarter ends, where they
    carry on from it. The join is checked: the Treasury's figure for that
    quarter end must match FRED's, or nothing is added.
    """
    observed = federal.dropna(subset=[column])
    if not config.current or observed.empty:
        return federal
    closes = observed[DATE_COLUMN].max() + pd.offsets.QuarterEnd(0)
    if config.end <= closes:
        return federal

    print("Fetching the latest U.S. federal debt (--cur) …")
    try:
        daily = treasury_debt_from(closes, config)
    except Exception as exc:
        print(f"  Warning: U.S. Treasury daily debt unavailable ({exc}); federal debt ends with the quarter to {closes.date()}.")
        return federal
    if closes in daily.index:
        fred_value = float(observed[column].iloc[-1])
        if abs(daily[closes] / fred_value - 1.0) > _DEBT_JOIN_TOLERANCE:
            print(
                f"  Warning: Treasury daily debt on {closes.date()} ({daily[closes]:,.0f}) does not match FRED's "
                f"({fred_value:,.0f}); not joined."
            )
            return federal
    newer = daily.loc[daily.index > closes]
    if newer.empty:
        return federal
    return pd.concat([federal, pd.DataFrame({DATE_COLUMN: newer.index, column: newer.to_numpy()})], ignore_index=True)
