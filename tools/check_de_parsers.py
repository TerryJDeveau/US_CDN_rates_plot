#!/usr/bin/env python3
"""Check Germany's parsers against responses saved from the real sources (batch 4); offline, in seconds.

The cloud copy cannot reach the Bundesbank, the ECB, Eurostat or CNBC, so
the parsers were written against responses saved on Terry's laptop on
2026-10-06 (``tools/de_fixtures/raw``, listed with their URLs in
``_index.tsv``). Each check reads a saved response through the program's
own code and compares a value read off the response by eye. The fetchers
themselves are run too: ``de_get`` is replaced by a reader of the saved
files, chosen by the URL and query the program asks for, so a request that
differs from the one measured fails here.

* the Bundesbank's CSV: daily and end-of-month term structure, each term's
  first month as ``DE_YIELD_EARLIEST`` says, the old database's frozen
  series, and the answers it must refuse (an unknown series, a 404, a
  changed unit); the baked history equals the saved download;
* the ECB's CSV: key rates on their change days, annual debt, bank lending
  rates; a wrong key, a wrong unit and an error page refused;
* Eurostat's JSON-stat: debt and interest by sector, GDP, population; the
  joins the chart makes (annual debt and interest against the quarters,
  the levels against the total, population at unification);
* CNBC's German quotes: every symbol the chart asks for is there and readable;
* the joined curves: yields, debt, interest, GDP, population, levels, and
  (``rates``) the policy rate and the mortgage rates.

The live check of the same sources is ``tools/check_sources.py``.

Usage (from the project root)::

    python tools/check_de_parsers.py

Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot import de_archive_data, de_data, latest  # noqa: E402
from ratesplot.config import CNBC_QUOTE_URL, PlotConfig, component_column  # noqa: E402

RAW = ROOT / "tools" / "de_fixtures" / "raw"
RESULTS: list[bool] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    RESULTS.append(bool(condition))
    shown = f"  [{detail}]" if not condition or detail else ""
    print(f"{'ok  ' if condition else 'FAIL'} {label}{shown}")


def raises(call, *args, **kwargs) -> str | None:
    """The exception's type and message if ``call(*args)`` raises, else None."""
    try:
        call(*args, **kwargs)
    except Exception as exc:  # what is raised is the point
        return f"{type(exc).__name__}: {exc}"
    return None


def text(name: str) -> str:
    return (RAW / name).read_text(encoding="utf-8")


def _request_key(url: str, params: dict | None = None) -> tuple[str, frozenset]:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query)) | {key: str(value) for key, value in (params or {}).items()}
    return f"{parts.scheme}://{parts.netloc}{parts.path}", frozenset(query.items())


def _saved_answers() -> dict[tuple[str, frozenset], tuple[str, int]]:
    """The saved responses by request: ``_index.tsv``'s URL (and query) -> (file, HTTP status)."""
    answers = {}
    for line in (RAW / "_index.tsv").read_text(encoding="utf-8").splitlines():
        cells = line.split("\t")
        if len(cells) >= 7 and (RAW / cells[0]).exists() and cells[1].isdigit():
            answers[_request_key(cells[6])] = (cells[0], int(cells[1]))
    return answers


SAVED = _saved_answers()


def saved_get(url: str, params: dict | None = None, **_kwargs) -> requests.Response:
    """Answer a request from the saved responses, as ``http.de_get`` would; a request never measured raises."""
    key = _request_key(url, params)
    if key not in SAVED:
        raise requests.ConnectionError(f"no saved response for {url} {params}")
    name, status = SAVED[key]
    response = requests.Response()
    response.status_code, response.url, response.encoding = status, url, "utf-8"
    response._content = (RAW / name).read_bytes()
    response.raise_for_status()
    return response


def check_bundesbank() -> None:
    parse = de_data.parse_bundesbank_csv
    monthly = parse(text("bbk_ts_m_R10XX.csv"))
    check("Bundesbank monthly 10-year: 1972-09 dated its last day, 8.08",
          monthly.index[0] == pd.Timestamp("1972-09-30") and monthly.iloc[0] == 8.08, monthly.head(1).to_dict())
    check("Bundesbank monthly 10-year: 2026-09, 3.64", monthly.loc["2026-09-30"] == 3.64)
    daily = parse(text("bbk_ts_d_R10XX.csv"))
    check("Bundesbank daily 10-year: 2026-10-06, 3.52; the first value 1997-08-07",
          daily.loc["2026-10-06"] == 3.52 and daily.index[0] == pd.Timestamp("1997-08-07"), daily.index[0])
    check("Bundesbank daily: days without a value ('.') are left out", pd.Timestamp("2026-10-04") not in daily.index)
    first_30 = parse(text("bbk_ts_d_R30XX.csv"))
    check("Bundesbank daily 30-year from 2000-08-01, 5.49", first_30.index[0] == pd.Timestamp("2000-08-01") and first_30.iloc[0] == 5.49)
    firsts = {column: parse(text(f"bbk_ts_m_{code}.csv")).index[0] for column, code in de_data.DE_YIELD_SERIES.items()}
    check("Bundesbank: each term's first month is DE_YIELD_EARLIEST", firsts == de_data.DE_YIELD_EARLIEST, firsts)
    old_end_of_month = parse(text("bbk01_WZ9826.csv"))
    both = pd.concat([old_end_of_month, monthly], axis=1).dropna()
    apart = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    check("Bundesbank: the monthly term structure is the old database's end-of-month series (WZ9826), 1972-2020, within 0.01",
          len(both) > 500 and apart.max() <= 0.0100001, (len(both), int((apart > 0).sum()), apart.max()))
    check("Bundesbank: an unknown series of the old database is refused", raises(parse, text("bbk01_SU0112.csv")) is not None)
    check("Bundesbank: a 404 answer is refused", raises(parse, text("bbk_ts_m_R00X5.csv")) is not None)
    changed = text("bbk_ts_m_R10XX.csv").replace("BBK_UNIT_ENG,percent", "BBK_UNIT_ENG,index")
    check("Bundesbank: a changed unit is refused", raises(parse, changed) is not None)
    wu0004 = parse(text("bbk01_WU0004.csv"))
    baked = pd.Series({pd.Timestamp(date): value for date, value in de_archive_data.EMBEDDED_DE_PUBLIC_DEBT_YIELD_HISTORY})
    check("bake: the baked public debt yield is the saved WU0004, each month dated by its first day",
          list(baked.to_numpy()) == list(wu0004.to_numpy()) and (baked.index == wu0004.index.to_period("M").start_time).all()
          and baked.index[0] == pd.Timestamp("1956-05-01"), (len(baked), len(wu0004)))


def check_ecb() -> None:
    parse = de_data.parse_ecb_csv
    fixed = parse(text("ecb_mro_fixed.csv"), "FM.B.U2.EUR.4F.KR.MRR_FR.LEV", "PCPA")
    check("ECB MRO fixed rate: 3 on 1999-01-01, 2.65 from 2026-09-16",
          fixed.loc["1999-01-01"] == 3.0 and fixed.loc["2026-09-16"] == 2.65 and len(fixed) == 48)
    minbid = parse(text("ecb_mro_minbid.csv"), "FM.B.U2.EUR.4F.KR.MRR_MBR.LEV", "PCPA")
    check("ECB minimum bid rate: 4.25 from 2000-06-28; its row without a value (2008-10-15) left out",
          minbid.loc["2000-06-28"] == 4.25 and minbid.index[-1] == pd.Timestamp("2008-10-08"), minbid.index[-1])
    annual = parse(text("ecb_gfs_a_debt.csv"), "GFS." + de_data.ECB_ANNUAL_DEBT[1], "XDC")
    check("ECB annual debt: 1991, €618,218m, dated 31 December", annual.loc["1991-12-31"] == 618_218e6 and annual.index[0].year == 1991)
    mortgage = parse(text("ecb_mir_house_5_10.csv"), "MIR.M.DE.B.A2C.O.R.A.2250.EUR.N", "PCPA")
    check("ECB housing loans 5-10 years: 3.81 in 2026-08 (dated its last day), from 2000-01",
          mortgage.loc["2026-08-31"] == 3.81 and mortgage.index[0] == pd.Timestamp("2000-01-31"))
    check("ECB: another series' answer is refused", raises(parse, text("ecb_deposit.csv"), "FM.B.U2.EUR.4F.KR.MRR_FR.LEV", "PCPA") is not None)
    check("ECB: a changed unit is refused", raises(parse, text("ecb_mro_fixed.csv"), "FM.B.U2.EUR.4F.KR.MRR_FR.LEV", "PC") is not None)
    check("ECB: its error page (the floating-rate loans' 500) is refused",
          raises(parse, text("ecb_mir_house_float.csv"), "MIR.M.DE.B.A2C.F.R.A.2250.EUR.N", "PCPA") is not None)


def jsonstat(name: str) -> dict:
    return json.loads(text(f"{name}.json"))


def check_eurostat() -> None:
    parse = de_data.parse_jsonstat
    debt = parse(jsonstat("est_ggdebt_q"), "MIO_EUR", sector="S13")
    check("Eurostat debt S13: 2026 Q1 €2,902,035.3m, from 2000 Q1",
          debt.loc["2026-03-31"] == 2_902_035.3 and debt.index[0] == pd.Timestamp("2000-03-31"), debt.index[0])
    check("Eurostat: a dimension with several categories must be chosen", raises(parse, jsonstat("est_ggdebt_q"), "MIO_EUR") is not None)
    check("Eurostat: a changed unit is refused", raises(parse, jsonstat("est_ggdebt_q"), "THS_EUR", sector="S13") is not None)
    check("Eurostat: an unknown sector is refused", raises(parse, jsonstat("est_ggdebt_q"), "MIO_EUR", sector="S99") is not None)
    annual = de_data.parse_ecb_csv(text("ecb_gfs_a_debt.csv"), "GFS." + de_data.ECB_ANNUAL_DEBT[1], "XDC")
    years = [year for year in range(2000, 2026)]
    gaps = [abs(annual.loc[f"{year}-12-31"] / 1e6 / debt.loc[f"{year}-12-31"] - 1) for year in years]
    check("join: the ECB's year-end debt is Eurostat's fourth quarter, 2000-2025 (within 1e-6)", max(gaps) < 1e-6, max(gaps))
    levels = sum(parse(jsonstat("est_ggdebt_q"), "MIO_EUR", sector=sector) for sector in ("S1311", "S1312", "S1313", "S1314"))
    ratio = (levels / debt).dropna()
    check("levels: the subsectors add up to 0.6-2 % more than the consolidated total", ratio.between(1.006, 1.02).all(), (ratio.min(), ratio.max()))
    interest = parse(jsonstat("est_ggnfa_q"), "MIO_EUR", sector="S13")
    yearly = parse(jsonstat("est_edpt1_a"), "MIO_EUR", sector="S13", na_item="D41PAY")
    sums = de_data.ttm_sum(interest)
    # 2025 differs: the annual table (April) predates the quarters' revision (July).
    check("join: annual interest (edpt1) is the four-quarter sum, 2002-2024",
          all(sums.loc[f"{year}-12-31"] == yearly.loc[f"{year}-12-31"] for year in range(2002, 2025)))
    check("Eurostat interest: annual from 1995, quarterly from 2002", yearly.index[0].year == 1995 and interest.index[0] == pd.Timestamp("2002-03-31"))
    gdp = parse(jsonstat("est_gdp_q"), "CP_MEUR")
    check("Eurostat GDP: from 1991 Q1, €358,976.1m; 2026 Q2", gdp.index[0] == pd.Timestamp("1991-03-31") and gdp.loc["2026-06-30"] == 1_151_030.0)
    people = parse(jsonstat("est_pop_q"), "THS_PER")
    check("Eurostat population: 83,337 thousand in 2026 Q2", people.loc["2026-06-30"] == 83_337)
    january = parse(jsonstat("est_pjan_a"), "NR")
    check("Eurostat population on 1 January: from 1960; West Germany to 1990, then unified (+17.1 million)",
          january.index[0].year == 1960 and january.loc["1990-12-31"] == 62_679_035 and january.loc["1991-12-31"] == 79_753_227)


def check_fetchers() -> None:
    """The program's own fetchers, served the saved responses."""
    real_get, de_data.de_get = de_data.de_get, saved_get
    try:
        config = PlotConfig(start=pd.Timestamp("1950-01-01"), end=pd.Timestamp("2026-10-07"), yield_terms=de_data.DE_YIELD_TERMS)
        yields = de_data.fetch_de_yields(config)
        through = yields.attrs[de_data.YIELD_HISTORY_ATTR]
        check("yields: the seven terms", list(yields.columns) == list(de_data.DE_YIELD_SERIES), list(yields.columns))
        check("yields: the 10-year from the baked 1956-05 (6.4), monthly to 1997-07, daily from 1997-08-07",
              yields["10-Year"].first_valid_index() == pd.Timestamp("1956-05-01") and yields.loc["1956-05-01", "10-Year"] == 6.4
              and through["10-Year"] == pd.Timestamp("1997-07-31") and yields.loc["1997-08-07", "10-Year"] == daily_value("R10XX", "1997-08-07"),
              through)
        check("yields: the baked months end where the fitted 10-year begins (1972-08, then 8.08 at 1972-09-30)",
              yields["10-Year"].loc["1972-08-01":"1972-09-30"].dropna().to_dict() == {pd.Timestamp("1972-08-01"): 7.9, pd.Timestamp("1972-09-30"): 8.08})
        check("yields: the 30-year monthly 2000-01 to 2000-07, then daily", through["30-Year"] == pd.Timestamp("2000-07-31"), through["30-Year"])
        debt = de_data.fetch_de_debt(config)[de_data.DE_DEBT_COLUMN]
        check("debt: year-end 1991-1999 (ECB), then quarterly (Eurostat) from 2000 Q1",
              debt.index[0] == pd.Timestamp("1991-12-31") and debt.loc["1999-12-31"] == 1_253_598e6
              and pd.Timestamp("2000-03-31") in debt.index and debt.loc["2026-03-31"] == 2_902_035.3e6)
        interest = de_data.fetch_de_interest(config)[de_data.DE_INTEREST_COLUMN]
        check("interest: annual 1995-2001, then four-quarter sums from 2002 Q4",
              interest.index[0] == pd.Timestamp("1995-12-31") and interest.loc["2001-12-31"] > 0
              and interest.index[interest.index > pd.Timestamp("2001-12-31")][0] == pd.Timestamp("2002-12-31"))
        gdp = de_data.fetch_de_gdp(config)["TTM Nominal GDP ($)"]
        check("GDP: four-quarter sums from 1991 Q4", gdp.index[0] == pd.Timestamp("1991-12-31"))
        people = de_data.fetch_de_population()
        check("population: 1 January from 1960 (1991's the first unified), then mid-quarters (79,777 thousand on 1991-02-15)",
              people.index[0] == pd.Timestamp("1960-01-01") and people.loc["1991-01-01"] == 79_753_227
              and people.loc["1991-02-15"] == 79_777_000 and pd.Timestamp("1991-04-01") not in people.index)
        levels = de_data.fetch_de_components(PlotConfig(start=config.start, end=config.end, components="fnpm"))
        n = levels[component_column("debt", "n")]
        check("levels: f n p m debt and interest; n is the Länder and local together",
              len(levels.columns) == 8 and (n == levels[component_column("debt", "p")] + levels[component_column("debt", "m")]).all())
    finally:
        de_data.de_get = real_get


def daily_value(code: str, day: str) -> float:
    return de_data.parse_bundesbank_csv(text(f"bbk_ts_d_{code}.csv")).loc[day]


def check_cnbc() -> None:
    quotes = {quote["symbol"]: quote for quote in json.loads(text("cnbc_de_quotes.json"))["FormattedQuoteResult"]["FormattedQuote"]}
    asked = latest._quote_symbols("de", list(de_data.DE_YIELD_SERIES))
    check("CNBC: a German symbol for each of the seven terms", set(asked) == set(de_data.DE_YIELD_SERIES), asked)
    check("CNBC: every German symbol asked for is in the feed", set(asked.values()) <= set(quotes), sorted(quotes))
    check("CNBC: DE3M-DE (unknown to the feed) is not asked for", "DE3M-DE" not in asked.values() and quotes["DE3M-DE"]["code"] == 1)
    readable = []
    for symbol in asked.values():
        quote = quotes[symbol]
        value = latest._percent(quote["last"])
        when = pd.Timestamp(quote["last_time"]).tz_convert("UTC")
        latest._day_change(quote)
        readable.append(latest._PLAUSIBLE_YIELD[0] <= value <= latest._PLAUSIBLE_YIELD[1] and when.year == 2026)
    check("CNBC: each German quote's yield, time and change are read", all(readable))
    check("CNBC: the feed is the one latest reads", "quote.cnbc.com" in CNBC_QUOTE_URL)


def main() -> int:
    for group in (check_bundesbank, check_ecb, check_eurostat, check_fetchers, check_cnbc):
        group()
    passed = sum(RESULTS)
    print(f"[de parsers] {passed} of {len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
