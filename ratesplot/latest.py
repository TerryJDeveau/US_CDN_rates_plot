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
  business day before the quote, the day's move in the feed (its "change"
  field, which survives the close) is added to the official close; only
  otherwise is the quote used as it is. Quotes from 17:00 on are the next
  day's session and are not used.

Only what the regular series lack, up to ``config.end``, is added, and it
is fetched only then: a window ending in the past is drawn exactly as before.
A source that fails prints a warning and adds nothing. GDP, interest,
population and Canadian debt have no faster free source than the
quarterly statistics already used.

For those, a chart reaching today gets the effect of a debt clock
(usdebtclock.org, whose figures are each "base + rate x time elapsed";
Terry, 2026-09-28: emulate it, without scraping it). Each right-axis curve
is carried from its last observation to the chart's last date along the
straight line of its own change over the year before
(``project_to_now``). The projected stretch is drawn fainter, and its end
label reads "≈" (plotting). Quarterly figures are dated by the quarter's
last day, the day they describe (``frames.at_quarter_end``), so a
projection starts where the data really end (``observed_through``).

Canadian debt (the aggregate and the federal level) is not projected blind:
the Bank of Canada counts the Government of Canada's market debt (bills and
bonds outstanding) each business day, a few days behind
(``cdn_market_debt``). Its change is observed, the rest of the curve is
projected at its own pace (``_market_steered``). Tested on the quarters
since 2010, standing at each quarter's start, this predicts the next
quarter's debt with a root-mean-square error of 33 billion dollars for the
federal level and 37 for the aggregate, against 45 and 48 for the pace
alone. It is a better projection, not data: it is drawn as one.
"""

from __future__ import annotations

import io
from typing import Iterable

import pandas as pd

from .config import (
    BOC_CDN_MARKET_DEBT_SERIES,
    BOC_SERIES_URL,
    CDN_DEBT_COLUMN,
    CNBC_QUOTE_URL,
    DATE_COLUMN,
    LATEST_QUOTE_RETRIES,
    LATEST_QUOTE_TIMEOUT_SECONDS,
    MARKET_TIMEZONE,
    TREASURY_DEBT_TO_PENNY_URL,
    TREASURY_YIELD_CURVE_URL,
    YIELD_COLUMNS,
    PlotConfig,
    component_column,
)
from .http import latest_get, parse_boc_csv

# Where ``plotting.prepare_us`` / ``prepare_cdn`` record, in the yields frame's
# ``attrs``, the time of the newest quote added (for the subtitle).
QUOTE_TIME_ATTR = "latest_quote_time"

# Chart column -> CNBC symbol, per country (``plotting.Country.key``). Only the
# columns a chart draws are asked for (--yields:LIST); the U.S. terms outside
# the default five are CNBC's symbols of the same pattern, not yet seen
# answering from here (the cloud copy has no route to the feed).
_QUOTE_SYMBOLS = {
    "us": {
        "1-Month": "US1M", "3-Month": "US3M", "6-Month": "US6M", "1-Year": "US1Y", "2-Year": "US2Y",
        "5-Year": "US5Y", "7-Year": "US7Y", "10-Year": "US10Y", "20-Year": "US20Y", "30-Year": "US30Y",
    },
    "cdn": {"3-Month": "CA3M", "2-Year": "CA2Y", "5-Year": "CA5Y", "10-Year": "CA10Y", "30-Year": "CA30Y"},
}
# Chart column -> column of the Treasury's yield-curve file (the file has all ten).
_TREASURY_COLUMNS = {
    "1-Month": "1 Mo", "3-Month": "3 Mo", "6-Month": "6 Mo", "1-Year": "1 Yr", "2-Year": "2 Yr",
    "5-Year": "5 Yr", "7-Year": "7 Yr", "10-Year": "10 Yr", "20-Year": "20 Yr", "30-Year": "30 Yr",
}
# A quoted yield outside this range (percent) is a bad print and is left out.
_PLAUSIBLE_YIELD = (-5.0, 50.0)
# From this hour (market time) the feed's U.S. quotes are the next day's session:
# at 17:05 on 2026-09-28 its previous close was that day's close. The day's
# official figures are due by then, so such quotes are not used.
_NEXT_SESSION_HOUR = 17
# Debt to the Penny must agree this closely (fraction) with FRED's value for
# the same quarter end, or the two are not the same measure and are not joined.
_DEBT_JOIN_TOLERANCE = 0.001
# The feed's quotes are at most this old: a Friday's last quotes on the Tuesday
# after a long weekend. A window ending earlier than that cannot use them, so
# they are not fetched for it.
_QUOTE_MAX_AGE_DAYS = 4


# Where ``project_to_now`` records, in the macro frame's ``attrs``, the date each
# projected column's observations end: {column: date}; and the columns among
# them that follow market debt (``_market_steered``): [column, ...].
PROJECTION_ATTR = "projected_from"
STEERED_ATTR = "steered_by_market_debt"
# A curve whose last observation is older than this is not projected: that is
# a series that ended (federal debt alone, pre-1933) or a live source that
# failed, leaving only the baked history.
_PROJECTION_MAX_AGE_DAYS = 400
# The rate of a projection is the curve's change over this many days before its last observation.
_PROJECTION_RATE_DAYS = 365
# The Canadian debt curves steered by market debt (``_market_steered``).
_MARKET_STEERED_COLUMNS = (CDN_DEBT_COLUMN, component_column("debt", "f"))
# Market debt is read as the median of each day and the two before. Once a
# month new treasury bills settle a day before the old ones mature, lifting
# the total by about 2 % for that day (2025: every month); 2026-07-03 read
# 9.6 % high for one day. The median drops such a day and follows a real
# change (a new bond, a maturity) one business day late.
_MARKET_DEBT_MEDIAN_DAYS = 3
# Market debt must be known this close (days) before the dates it is read at:
# the curve's last observation, and the year before it.
_MARKET_DEBT_MAX_GAP_DAYS = 7


def _quotes_can_reach(config: PlotConfig) -> bool:
    """True when the window ends late enough for the day's quotes to fall in it."""
    today = pd.Timestamp.now(tz=MARKET_TIMEZONE).tz_localize(None).normalize()
    return config.end >= today - pd.Timedelta(days=_QUOTE_MAX_AGE_DAYS)


def observed_through(dates: pd.Index | pd.Series) -> pd.Timestamp | None:
    """Return the date a series' last observation describes, given its observation dates (None if none).

    Every series is dated by the day its values describe (a quarterly
    figure by the quarter's last day, see ``frames.at_quarter_end``), so
    that is simply its last date.
    """
    stamps = pd.to_datetime(pd.Series(dates).dropna())
    return pd.Timestamp(stamps.max()) if len(stamps) else None


def project_to_now(
    data: pd.DataFrame,
    observed: dict[str, pd.Timestamp],
    config: PlotConfig,
    history: pd.DataFrame,
    market: pd.Series | None = None,
) -> pd.DataFrame:
    """Carry each column of ``observed`` from the date its data end to the frame's last date (see the module docstring).

    ``data`` is a macro frame (a ``DATE`` column plus one column per curve,
    each held forward to the chart's last date); ``observed`` gives each
    curve's last observation date (``observed_through``); ``history`` holds
    the same curves over their whole span, date-indexed, since ``data`` may
    start after the year the pace is measured over. Along the way the value
    grows at the curve's own rate over that year, in a straight line as a
    debt clock does. With ``market`` (Canadian market debt, from
    ``market_debt_for_projection``) the Canadian debt curves follow it
    instead (``_market_steered``). Rows are added at each starting date so a
    projection begins exactly where its data end. Only with --cur, on a
    chart reaching today; the columns projected, with their starting dates,
    are recorded in ``attrs[PROJECTION_ATTR]``.
    """
    if not config.current or data.empty or not observed or not _quotes_can_reach(config):
        return data
    end = data[DATE_COLUMN].max()
    starts = {
        column: start
        for column, start in observed.items()
        if column in data.columns and start is not None and start < end
        and (end - start).days <= _PROJECTION_MAX_AGE_DAYS
    }
    if not starts:
        return data

    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    new_dates = sorted(set(starts.values()) - set(data[DATE_COLUMN]))
    if new_dates:
        data = pd.concat([data, pd.DataFrame({DATE_COLUMN: new_dates})], ignore_index=True)
        data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
        added = data[DATE_COLUMN].isin(new_dates)
        for column in data.columns.drop(DATE_COLUMN):
            # The held value, but not past a curve's own last value.
            held = data[column].ffill().where(data[column].bfill().notna())
            data.loc[added, column] = held[added]

    print(f"Projecting to {end:%Y-%m-%d} at each curve's pace over the year before its last data (--cur):")
    projected: dict[str, pd.Timestamp] = {}
    steered_columns: list[str] = []
    for column, start in starts.items():
        if column not in history.columns:
            continue
        series = history[column].sort_index()
        base = series.loc[:start].dropna()
        earlier = series.loc[: start - pd.Timedelta(days=_PROJECTION_RATE_DAYS)].dropna()
        if base.empty or earlier.empty:
            continue
        value = float(base.iloc[-1])
        per_day = (value - float(earlier.iloc[-1])) / _PROJECTION_RATE_DAYS
        later = (data[DATE_COLUMN] > start) & data[column].notna()
        elapsed = (data.loc[later, DATE_COLUMN] - start).dt.days
        steered = None
        if market is not None and column in _MARKET_STEERED_COLUMNS:
            steered = _market_steered(
                data.loc[later, DATE_COLUMN], start, value, (earlier.index[-1], float(earlier.iloc[-1])), market
            )
            if steered is None:
                print(f"  Warning: Government of Canada market debt does not cover the year to {start:%Y-%m-%d}; {column} goes on at its own pace alone.")
        projected[column] = start
        if steered is not None:
            values, note = steered
            data.loc[later, column] = values
            steered_columns.append(column)
            print(f"  {column}: from {start:%Y-%m-%d}, {note}")
            continue
        data.loc[later, column] = value + per_day * elapsed
        pace = per_day * 365 / value if value else float("nan")
        print(f"  {column}: from {start:%Y-%m-%d}, {pace:+.1%} a year")
    data.attrs[PROJECTION_ATTR] = projected
    data.attrs[STEERED_ATTR] = steered_columns
    return data


def _market_steered(
    dates: pd.Series, start: pd.Timestamp, value: float, earlier: tuple[pd.Timestamp, float], market: pd.Series
) -> tuple[pd.Series, str] | None:
    """Return a Canadian debt curve's values on ``dates`` (all after ``start``) steered by market debt, and a note; None if it cannot be.

    The curve is market debt plus the rest. Market debt (``market``,
    dollars, date-indexed) is read as observed up to its last day and goes
    on after it at its own pace over the year before that day. The rest goes
    on from ``start``, where the curve's data end at ``value``, at its own
    pace over the year to it, from ``earlier`` (the curve's date and value a
    year before). Paces are over ``_PROJECTION_RATE_DAYS``, as in
    ``project_to_now``. None when market debt is not known close to
    ``start``, to that earlier date, or to a year before its own last day.
    """
    known = market.dropna().sort_index()
    if known.empty:
        return None

    def on(day: pd.Timestamp) -> float | None:
        upto = known.loc[:day]
        if upto.empty or (day - upto.index[-1]).days > _MARKET_DEBT_MAX_GAP_DAYS:
            return None
        return float(upto.iloc[-1])

    earlier_date, earlier_value = earlier
    last_day, last_value = known.index[-1], float(known.iloc[-1])
    at_start, at_earlier = on(start), on(earlier_date)
    year_before_last = on(last_day - pd.Timedelta(days=_PROJECTION_RATE_DAYS))
    if at_start is None or at_earlier is None or year_before_last is None:
        return None
    rest_per_day = ((value - at_start) - (earlier_value - at_earlier)) / _PROJECTION_RATE_DAYS
    market_per_day = (last_value - year_before_last) / _PROJECTION_RATE_DAYS

    when = pd.DatetimeIndex(dates)
    market_then = known.reindex(known.index.union(when)).ffill().reindex(when).to_numpy(dtype=float, copy=True)
    beyond = when > last_day
    market_then[beyond] = last_value + market_per_day * (when[beyond] - last_day).days.to_numpy()
    values = value + (market_then - at_start) + rest_per_day * (when - start).days.to_numpy()
    note = (
        f"market debt as observed to {last_day:%Y-%m-%d} ({(last_value - at_start) / 1e9:+,.1f} billion since), "
        f"the rest {rest_per_day * 365 / value:+.1%} of the curve a year"
    )
    return pd.Series(values, index=dates.index), note


# ---------------------------------------------------------------------------
# The sources
# ---------------------------------------------------------------------------


def treasury_yields_after(
    last: pd.Timestamp, config: PlotConfig, columns: Iterable[str] = YIELD_COLUMNS
) -> pd.DataFrame:
    """Return the Treasury's daily par yields after ``last``, up to ``config.end``, as a ``DATE``-column frame.

    The Treasury publishes one CSV per calendar year, newest day first.
    ``columns`` are the chart columns wanted (the drawn yields).
    """
    wanted = {column: _TREASURY_COLUMNS[column] for column in columns}
    frames = []
    for year in range(last.year, config.end.year + 1):
        params = {"type": "daily_treasury_yield_curve", "field_tdr_date_value": year, "_format": "csv"}
        table = pd.read_csv(io.StringIO(latest_get(TREASURY_YIELD_CURVE_URL.format(year=year), params).text))
        missing = {"Date", *wanted.values()} - set(table.columns)
        if missing:
            raise ValueError(f"the {year} yield-curve file has no column {sorted(missing)}")
        frame = pd.DataFrame({DATE_COLUMN: pd.to_datetime(table["Date"], format="%m/%d/%Y")})
        for column, source in wanted.items():
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


# Government of Canada market debt is between these (dollars): C$1.4-1.6
# trillion in 2025-2026. Outside them the Bank has changed the units or the
# series, and it is not used.
_PLAUSIBLE_CDN_MARKET_DEBT = (2e11, 2e13)


def cdn_market_debt(first: pd.Timestamp) -> pd.Series:
    """Return Government of Canada market debt outstanding (dollars, date-indexed) from ``first``'s year to the latest day.

    One Bank of Canada series per year (``BOC_CDN_MARKET_DEBT_SERIES``); a
    year it does not have, before 2025 or not yet archived under its own
    name, is skipped. Each day is the median of that day and the two before
    (``_MARKET_DEBT_MEDIAN_DAYS``).
    """
    this_year = pd.Timestamp.now(tz=MARKET_TIMEZONE).year
    names = [f"{BOC_CDN_MARKET_DEBT_SERIES}_{year}" for year in range(first.year, this_year + 1)]
    parts = []
    for name in [*names, BOC_CDN_MARKET_DEBT_SERIES]:  # the current year's last, so it wins any overlap
        response = latest_get(BOC_SERIES_URL.format(series_code=name), missing_ok=True)
        if response is not None:
            table = parse_boc_csv(response.text, key="dom_dbt_id")
            parts.append(pd.to_numeric(table.iloc[:, 0], errors="coerce"))
    if not parts:
        raise ValueError(f"the Bank of Canada has no {BOC_CDN_MARKET_DEBT_SERIES} series")
    daily = pd.concat(parts).dropna().sort_index()
    daily = daily[~daily.index.duplicated(keep="last")]
    low, high = _PLAUSIBLE_CDN_MARKET_DEBT
    if daily.empty or not low <= daily.iloc[-1] <= high:
        raise ValueError(f"its latest value ({daily.iloc[-1] if len(daily) else None}) is not a plausible total in dollars")
    return daily.rolling(_MARKET_DEBT_MEDIAN_DAYS, min_periods=1).median().rename(BOC_CDN_MARKET_DEBT_SERIES)


def market_debt_for_projection(observed: dict[str, pd.Timestamp | None], config: PlotConfig) -> pd.Series | None:
    """Return Canadian market debt for ``project_to_now`` when a Canadian debt curve will be projected; else None.

    Fetched only for a chart reaching today, as the projection is. A failure
    prints a warning; the debt curves then go on at their own pace.
    """
    starts = [observed[column] for column in _MARKET_STEERED_COLUMNS if observed.get(column) is not None]
    if not config.current or not starts or not _quotes_can_reach(config):
        return None
    print("Fetching Government of Canada market debt (--cur) …")
    try:
        return cdn_market_debt(min(starts) - pd.Timedelta(days=_PROJECTION_RATE_DAYS))
    except Exception as exc:
        print(f"  Warning: Government of Canada market debt unavailable ({exc}); Canadian debt goes on at its own pace alone.")
        return None


def _percent(text: object) -> float:
    """Parse a quoted yield such as ``"4.912%"``; raises ValueError (or TypeError) if it is not one."""
    return float(str(text).strip().rstrip("%"))


def _day_change(quote: dict) -> float:
    """Return the feed's change on the day, in points; raises KeyError / ValueError / TypeError if it has none.

    Its ``change`` field ("+0.042", "-0.006", "UNCH") keeps the day's move
    after the close, when ``previous_day_closing`` has already been rolled
    over to the day's own close (seen for Canada at 16:35 on 2026-09-28:
    last = previous close, change +0.042). Only without it is the move
    taken as last less previous close.
    """
    text = str(quote.get("change", "")).strip()
    if text.upper() == "UNCH":
        return 0.0
    try:
        return _percent(text)
    except ValueError:
        return _percent(quote["last"]) - _percent(quote["previous_day_closing"])


def _quote_symbols(country: str, columns: Iterable[str] | None) -> dict[str, str]:
    """Return {chart column: symbol} for ``columns`` (None: the default five) that the feed has for ``country``."""
    every = _QUOTE_SYMBOLS[country]
    return {column: every[column] for column in (YIELD_COLUMNS if columns is None else columns) if column in every}


def fetch_quotes(country: str, columns: Iterable[str] | None = None) -> dict[str, dict | None]:
    """Return the feed's quote of the moment for each of one country's yield ``columns`` (None where it has none).

    ``columns`` defaults to the five default terms. Not cached: each drawing
    gets fresh quotes. ``tools/check_sources.py`` uses it to check the feed
    still answers in the expected form.
    """
    symbols = _quote_symbols(country, columns)
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
    return {column: by_symbol.get(symbol) for column, symbol in symbols.items()}


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
    symbols = _quote_symbols(country, official.index)
    quotes = fetch_quotes(country, symbols)

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    newest: pd.Timestamp | None = None
    for column, symbol in symbols.items():
        quote = quotes[column]
        try:
            value = _percent(quote["last"])
            when = pd.Timestamp(quote["last_time"]).tz_convert(MARKET_TIMEZONE)
        except (TypeError, KeyError, ValueError):
            print(f"  Warning: no usable quote for {symbol}; its line ends at the last official value.")
            continue
        if not _PLAUSIBLE_YIELD[0] <= value <= _PLAUSIBLE_YIELD[1]:
            print(f"  Warning: quote {symbol} = {value} is not a plausible yield; left out.")
            continue
        if when.hour >= _NEXT_SESSION_HOUR:
            continue  # the next day's session: its "change" no longer runs from the official close
        day = when.tz_localize(None).normalize()
        if not last < day <= config.end:
            continue
        close = official.get(column)
        if day - pd.offsets.BDay(1) == last and close is not None and pd.notna(close):
            try:
                value = float(close) + _day_change(quote)
            except (KeyError, TypeError, ValueError):
                pass  # no change in the feed: the quote as it is
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
        treasury = treasury_yields_after(last, config, yields.columns.drop(DATE_COLUMN))
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

    Each quarter's closing debt is dated by the quarter's last day (FRED's
    GFDEBTN for 2026 Q1, dated 2026-01-01 there, is the Treasury's figure
    for 2026-03-31, exactly). So the daily figures join the day after, where
    they carry on from it. The join is checked: the Treasury's figure for
    that quarter end must match FRED's, or nothing is added.
    """
    observed = federal.dropna(subset=[column])
    if not config.current or observed.empty:
        return federal
    closes = observed[DATE_COLUMN].max()
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
