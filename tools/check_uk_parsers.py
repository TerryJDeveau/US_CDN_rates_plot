#!/usr/bin/env python3
"""Check the UK's parsers against responses saved from the real sources (batch 3); offline, in seconds.

The cloud copy cannot reach the Bank of England, the ONS or CNBC, so the
parsers were written against responses saved on Terry's laptop on
2026-10-06 (``tools/uk_fixtures``). This keeps them honest when the code
changes: each check reads a saved response through the program's own
parser and compares a value read off the response by eye.

* the Bank of England database's CSV (daily and monthly), and the two
  answers it must refuse: its described layout and a redirect to its error
  page;
* each kind of ONS series: months, quarters, the units £m, £bn (HF6W),
  none (MDYT) and thousands, a changed unit refused, and a four-quarter sum against the
  ONS's own annual figure;
* CNBC's UK quotes: every symbol the chart asks for is there and readable,
  and the 3-month symbol (a repo rate, not a yield) is not asked for;
* the bake's reading of the millennium workbook's period cells.

The live check of the same sources is ``tools/check_sources.py``.

Usage (from the project root)::

    python tools/check_uk_parsers.py

Exit code 1 if any check fails.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot import bake, latest, uk_data  # noqa: E402
from ratesplot.config import CNBC_QUOTE_URL  # noqa: E402

FIXTURES = ROOT / "tools" / "uk_fixtures"
RESULTS: list[bool] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    RESULTS.append(bool(condition))
    shown = f"  [{detail}]" if not condition or detail else ""
    print(f"{'ok  ' if condition else 'FAIL'} {label}{shown}")


def raises(call, *args) -> str | None:
    """The exception's type and message if ``call(*args)`` raises, else None."""
    try:
        call(*args)
    except Exception as exc:  # what is raised is the point
        return f"{type(exc).__name__}: {exc}"
    return None


def text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def check_iadb() -> None:
    daily = uk_data.parse_iadb_csv(text("boe_iadb_daily.csv"))
    check("IADB daily: the 10-year par yield on 2026-09-30", daily.loc["2026-09-30", "IUDMNPY"] == 5.3714)
    check("IADB daily: Bank Rate on 2026-10-05, the yields blank that day",
          daily.loc["2026-10-05", "IUDBEDR"] == 3.75 and daily.loc["2026-10-05", "IUDMNPY"] != daily.loc["2026-10-05", "IUDMNPY"])
    check("IADB daily: every series the chart reads is a column",
          set(uk_data.UK_YIELD_SERIES.values()) | {"IUDBEDR"} <= set(daily.columns), list(daily.columns))
    monthly = uk_data.parse_iadb_csv(text("boe_iadb_monthly.csv"))
    check("IADB monthly: dated by the month's last day; the 2-year fixed for August 2026",
          monthly.loc["2026-08-31", "IUMBV34"] == 4.92 and monthly.index[0] == pd.Timestamp("2025-01-31"))
    check("IADB: the described layout (CSVF=CT) is refused", raises(uk_data.parse_iadb_csv, text("boe_iadb_described.csv")) is not None)
    check("IADB: its error page is refused", raises(uk_data.parse_iadb_csv, text("boe_iadb_error_905.txt")) is not None)

    def redirected(url, params=None, **_kwargs):
        response = requests.Response()
        response.status_code, response.url = 200, "https://www.bankofengland.co.uk/boeapps/database/ErrorPage.asp?ei=905"
        response.history = [requests.Response()]
        response._content = b"<html>error</html>"
        return response

    real_get, uk_data.uk_get = uk_data.uk_get, redirected
    try:
        message = raises(uk_data.iadb_series, "IUDMNPY")
    finally:
        uk_data.uk_get = real_get
    check("IADB: a redirect raises, naming the page", message is not None and "ErrorPage" in message, message)


def ons(cdid: str) -> dict:
    return json.loads(text(f"ons_{cdid.lower()}.json"))


def check_ons() -> None:
    bkpx = uk_data.parse_ons_series(ons("BKPX"), "BKPX", "months")
    check("ONS BKPX months: August 2026, £m, dated its last day", bkpx.loc["2026-08-31"] == 3_195_774)
    quarters = uk_data.parse_ons_series(ons("BKPX"), "BKPX", "quarters")
    check("ONS BKPX quarters: 1975 Q1 dated 1975-03-31", quarters.index[0] == pd.Timestamp("1975-03-31") and quarters.iloc[0] == 53_670)
    mdyt = uk_data.parse_ons_series(ons("MDYT"), "MDYT", "quarters")
    check("ONS MDYT (no unit in its description): 2026 Q2", mdyt.loc["2026-06-30"] == 138_883)
    check("ONS MDYT has no months", uk_data.parse_ons_series(ons("MDYT"), "MDYT", "months").empty)
    hf6w = uk_data.parse_ons_series(ons("HF6W"), "HF6W", "months")
    check("ONS HF6W (£bn): August 2026", hf6w.loc["2026-08-31"] == 2_985.5)
    ebaq = uk_data.parse_ons_series(ons("EBAQ"), "EBAQ", "quarters")
    check("ONS EBAQ (thousands): 2026 Q2", ebaq.loc["2026-06-30"] == 69_628)
    changed = ons("YBHA")
    changed["description"]["unit"] = "bn"
    check("ONS: a changed unit is refused", raises(uk_data.parse_ons_series, changed, "YBHA", "quarters") is not None)
    nmyx = uk_data.parse_ons_series(ons("NMYX"), "NMYX", "quarters")
    annual = {row["date"]: float(row["value"]) for row in ons("NMYX")["years"] if row["value"]}
    check("ONS NMYX: four quarters of 2025 sum to the ONS's 2025 figure",
          uk_data.ttm_sum(nmyx).loc["2025-12-31"] == annual["2025"], (uk_data.ttm_sum(nmyx).loc["2025-12-31"], annual["2025"]))
    # The saved response keeps 1946's four quarters, then 2024 on: one sum for 1946, none across the gap.
    sums = uk_data.ttm_sum(nmyx)
    check("ONS: four consecutive quarters make a sum; a gap of quarters leaves none across it",
          sums.index[0] == pd.Timestamp("1946-12-31") and sums.loc["1947":"2024-09-30"].empty, list(sums.index[:3]))


def check_cnbc() -> None:
    quotes = {quote["symbol"]: quote for quote in json.loads(text("cnbc_gb_quotes.json"))["FormattedQuoteResult"]["FormattedQuote"]}
    asked = latest._quote_symbols("uk", list(uk_data.UK_YIELD_SERIES))
    check("CNBC: every UK symbol asked for is in the feed", set(asked.values()) <= set(quotes), sorted(quotes))
    check("CNBC: the 3-month repo rate is not asked for", "GB3M-GB" not in asked.values())
    readable = []
    for symbol in asked.values():
        quote = quotes[symbol]
        value = latest._percent(quote["last"])
        when = pd.Timestamp(quote["last_time"]).tz_convert("UTC")
        latest._day_change(quote)
        readable.append(latest._PLAUSIBLE_YIELD[0] <= value <= latest._PLAUSIBLE_YIELD[1] and when.year == 2026)
    check("CNBC: each UK quote's yield, time (offsets differ) and change are read", all(readable))
    check("CNBC: the feed is the one latest reads", "quote.cnbc.com" in CNBC_QUOTE_URL)


def check_bake_periods() -> None:
    date = bake._millennium_date
    cases = [
        ("fy", ("1690/91", "End September"), None, "1691-09-30"),
        ("fy", ("1974/75", "31st March"), None, "1975-03-31"),
        ("fy", ("1750/51", "5th January"), None, "1751-01-05"),
        ("fy", ("1799/00", "5th January"), None, "1800-01-05"),
        ("year", (1700.0, None), None, "1700-12-31"),
        ("midyear", ("1922", None), None, "1922-06-30"),
        ("month", (1753, "Aug"), None, "1753-08-01"),
        ("month", (None, "Jan"), pd.Timestamp("1935-12-01"), "1936-01-01"),
        ("month", (None, "Feb"), pd.Timestamp("1936-01-01"), "1936-02-01"),
        ("day", (dt.datetime(1833, 1, 1), None), None, "1833-01-01"),
        ("day", ("02/11/2017", None), None, "2017-11-02"),
    ]
    found = [(cells, date(kind, cells, previous)) for kind, cells, previous, _expected in cases]
    check("bake: the millennium workbook's period cells", all(when == pd.Timestamp(expected) for (_c, when), (*_x, expected) in zip(found, cases)),
          [(cells, str(when.date())) for cells, when in found])
    check("bake: a row with no period has none", date("month", (None, None), pd.Timestamp("2017-03-01")) is None)
    check("bake: a financial year whose years do not follow is refused", raises(date, "fy", ("1690/92", "End September"), None) is not None)
    check("bake: a month with no year before it is refused", raises(date, "month", (None, "Aug"), None) is not None)


def main() -> int:
    for group in (check_iadb, check_ons, check_cnbc, check_bake_periods):
        group()
    passed = sum(RESULTS)
    print(f"[uk parsers] {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
