"""Synthetic stand-ins for every download the program makes, for a machine with no route to the sources.

The cloud copy of this project (Claude Code on the web) has so far had FRED,
the Bank of Canada, Statistics Canada, the Treasury and CNBC all refused by
its network policy (2026-10-06). The values here are made up, smooth
plausible paths plus seeded noise, but they are deterministic and in each
source's own format (FRED frames, Valet CSV, StatCan table ZIPs), so:

* two program trees fed the same synthetic data can be compared byte for
  byte (``tools/regress_charts.py --synthetic``, ``tools/verify_web.py
  --synthetic``): a change proved neutral there is neutral in the code,
  though not shown against real data;
* new curves can be seen drawn, at plausible sizes, before the real data are
  reachable. They are never evidence about the data themselves.

``synth(key)`` answers one ``ratesplot.http._cached`` key: ``("fred", ID)``
-> DataFrame; ``("canadian", url, params)``, ``("uk", url, params)`` and
``("de", url, params)`` -> Response (the UK's: the Bank of England
database's CSV and the ONS's JSON, in the layouts saved from the laptop in
tools/uk_fixtures; batch 3. Germany's: the Bundesbank's CSV, the ECB's
csvdata and Eurostat's JSON-stat, as saved in tools/de_fixtures; batch 4). A source with no
synthetic answer (the Treasury, CNBC) raises, as a failed download would,
and the program falls back with its usual warning. The series end in
September 2026, past the regression set's pinned end, so a pinned chart
never asks for them. ``install_synthetic_cache`` puts ``synth`` behind the
regression tool's disk cache and refuses every real request.

Add a source here when the program reads a new one (its key is printed by
the "no synthetic answer" line).
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# The U.S. short rate's path (percent) and the 10y-less-short slope's, as
# anchors joined by straight lines; every synthetic yield, policy and
# mortgage rate is built from the two, so they move together as real ones do.
_SHORT = [
    ("1934-01-01", 0.8), ("1946-01-01", 0.4), ("1954-07-01", 1.0), ("1960-01-01", 3.5), ("1970-01-01", 7.5),
    ("1974-07-01", 11.0), ("1977-01-01", 4.7), ("1981-01-01", 17.0), ("1986-01-01", 7.0), ("1989-03-01", 9.8),
    ("1993-01-01", 3.0), ("2000-06-01", 6.5), ("2003-06-01", 1.0), ("2006-07-01", 5.25), ("2008-12-15", 0.1),
    ("2015-12-01", 0.15), ("2019-01-01", 2.4), ("2020-04-01", 0.05), ("2022-03-01", 0.1), ("2023-08-01", 5.33),
    ("2024-09-01", 5.1), ("2025-12-31", 3.9), ("2026-09-30", 3.6),
]
_SLOPE = [
    ("1934-01-01", 2.0), ("1966-01-01", 0.5), ("1969-06-01", -0.6), ("1972-01-01", 1.5), ("1973-08-01", -1.0),
    ("1975-06-01", 2.0), ("1979-01-01", -1.5), ("1981-06-01", -2.0), ("1982-12-01", 2.5), ("1989-03-01", -0.3),
    ("1992-06-01", 3.5), ("2000-09-01", -0.6), ("2003-06-01", 3.0), ("2006-12-01", -0.5), ("2008-12-01", 2.5),
    ("2013-01-01", 2.0), ("2019-08-01", -0.3), ("2021-01-01", 1.2), ("2023-06-01", -1.6), ("2024-12-01", 0.1),
    ("2026-09-30", 0.6),
]


def _anchors(points: list[tuple[str, float]], dates: pd.DatetimeIndex) -> np.ndarray:
    """Return the anchored path at ``dates`` (straight between anchors; flat beyond them)."""
    xs = np.array([pd.Timestamp(d).as_unit("ns").value for d, _ in points], dtype=float)
    ys = np.array([v for _, v in points], dtype=float)
    return np.interp(dates.as_unit("ns").asi8.astype(float), xs, ys)


def _rng(name: str) -> np.random.Generator:
    """A generator seeded by ``name``, so each series has its own noise, the same on every run."""
    return np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))


def _noise(name: str, n: int, scale: float) -> np.ndarray:
    """A smooth seeded wander: a random walk pulled back toward zero."""
    steps = _rng(name).normal(0.0, scale, n)
    out = np.empty(n)
    level = 0.0
    for i, step in enumerate(steps):
        level = 0.97 * level + step
        out[i] = level
    return out


def _yield_path(name: str, dates: pd.DatetimeIndex, years: float, offset: float = 0.0) -> np.ndarray:
    """A yield of term ``years``: the short rate plus its share of the slope, plus noise; two decimals."""
    short = _anchors(_SHORT, dates) + offset
    slope = _anchors(_SLOPE, dates)
    shape = (1 - np.exp(-years / 3.0)) / (1 - np.exp(-10 / 3.0))
    value = short + slope * shape + _noise(name, len(dates), 0.03)
    return np.round(np.maximum(value, 0.01), 2)


def _growth(dates: pd.DatetimeIndex, first: float, last: float, name: str) -> np.ndarray:
    """A level growing geometrically from ``first`` to ``last`` (debt, GDP, interest), with 1 % noise."""
    t = (dates - dates[0]).days.to_numpy(dtype=float)
    span = max(t[-1], 1.0)
    base = first * (last / first) ** (t / span)
    return base * (1 + 0.01 * _noise(name, len(dates), 0.2))


# FRED's daily series: each yield's term in years, and each series' first day.
_TENOR_YEARS = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1, "DGS2": 2, "DGS5": 5, "DGS7": 7,
    "DGS10": 10, "DGS20": 20, "DGS30": 30,
}
_DAILY_START = {
    "DGS1MO": "2001-07-31", "DGS3MO": "1981-09-01", "DGS6MO": "1981-09-01", "DGS1": "1962-01-02",
    "DGS2": "1976-06-01", "DGS5": "1962-01-02", "DGS7": "1969-07-01", "DGS10": "1962-01-02",
    "DGS20": "1962-01-02", "DGS30": "1977-02-15", "DFF": "1954-07-01", "T10Y2Y": "1976-06-01",
    "T10Y3M": "1982-01-04",
}
_GROWTH = {  # id -> (start, freq, first, last), FRED's units
    "GDP": ("1947-01-01", "QS", 243.0, 31_500.0),
    "GFDEBTN": ("1966-01-01", "QS", 320_000.0, 37_200_000.0),
    "SLGSDODNS": ("1945-10-01", "QS", 16_000.0, 3_350_000.0),
    "A180RC1Q027SBEA": ("1947-01-01", "QS", 4.6, 1_650.0),
    "A091RC1Q027SBEA": ("1947-01-01", "QS", 4.2, 1_250.0),
    "B111RC1Q027SBEA": ("1947-01-01", "QS", 0.4, 400.0),
    "W756RC1A027NBEA": ("1959-01-01", "YS", 0.6, 170.0),
    "W856RC1A027NBEA": ("1959-01-01", "YS", 1.2, 230.0),
    "B230RC0Q173SBEA": ("1947-01-01", "QS", 144_000.0, 343_000.0),
    "NGDPSAXDCCAQ": ("1961-01-01", "QS", 40_000.0, 3_200_000.0),
}


def fred(series_id: str) -> pd.DataFrame:
    """Return one FRED series as ``http.fetch_fred_csv`` returns it: ``DATE`` and the series' values."""
    if series_id in _GROWTH:
        start, freq, first, last = _GROWTH[series_id]
        end = "2026-07-01" if freq == "QS" else "2025-01-01"
        dates = pd.date_range(start, end, freq=freq)
        values = np.round(_growth(dates, first, last, series_id), 1)
    elif series_id in _DAILY_START or series_id.startswith("MORTGAGE"):
        if series_id.startswith("MORTGAGE"):
            start = "1971-04-02" if series_id == "MORTGAGE30US" else "1991-08-30"
            dates = pd.date_range(start, "2026-10-01", freq="W-THU")
            margin = 1.7 if series_id == "MORTGAGE30US" else 1.0
            values = _yield_path(series_id, dates, 10) + margin
        else:
            dates = pd.bdate_range(_DAILY_START[series_id], "2026-09-30")
            if series_id == "DFF":
                values = np.round(np.maximum(_anchors(_SHORT, dates) + _noise("DFF", len(dates), 0.01), 0.04), 2)
            elif series_id in ("T10Y2Y", "T10Y3M"):
                other = "DGS2" if series_id == "T10Y2Y" else "DGS3MO"
                ten, short = fred("DGS10").set_index("DATE")["DGS10"], fred(other).set_index("DATE")[other]
                both = (ten - short).dropna()
                both = both.loc[both.index >= dates[0]]
                dates, values = both.index, np.round(both.to_numpy(), 2)
            else:
                values = _yield_path(series_id, dates, _TENOR_YEARS[series_id])
    else:
        raise KeyError(f"no synthetic FRED series {series_id}")
    return pd.DataFrame({"DATE": dates, series_id: np.asarray(values, dtype=float)})


def _response(content: bytes) -> requests.Response:
    """A successful Response with ``content`` as its body, as ``http.canadian_get`` returns one."""
    response = requests.Response()
    response._content = content
    response.status_code = 200
    response.encoding = "utf-8"
    return response


def _valet_csv(columns: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> bytes:
    """A Valet CSV: a metadata block, then "date" and one column per series (``http.parse_boc_csv``)."""
    lines = ['"TERMS AND CONDITIONS"', '"https://www.bankofcanada.ca/terms/"', "", '"SERIES"', "", '"OBSERVATIONS"']
    lines.append(",".join(['"date"'] + [f'"{name}"' for name in columns]))
    for i, day in enumerate(dates):
        lines.append(",".join([f'"{day:%Y-%m-%d}"'] + [("" if np.isnan(v[i]) else f'"{v[i]:.2f}"') for v in columns.values()]))
    return "\n".join(lines).encode()


# Canadian yields run this far above the U.S. ones. Valet series: code ->
# (first day, frequency, term in years for the yield shape (0: the short rate
# itself, a policy rate), margin above that yield).
_CDN_OFFSET = 0.3
_VALET_SERIES = {
    "V80691303": ("2000-12-01", "B", 0.25, 0.0),
    "V122530": ("1935-03-01", "MS", 0.0, 0.25),
    "AVG.INTWO": ("1997-01-02", "B", 0.0, 0.0),
    "V80691335": ("1975-01-01", "W-WED", 10, 2.4),
    "V80691334": ("1980-01-02", "W-WED", 3, 2.6),
    "V80691333": ("1980-01-02", "W-WED", 1, 2.7),
    "BROKER_AVERAGE_5YR_VRM": ("2011-01-05", "W-WED", 0.25, 0.9),
    "V80691311": ("1975-01-01", "W-WED", 0.0, 2.2),
}


def valet(url: str, params: tuple) -> bytes:
    """Return a Valet response body: the benchmark-yield group, or one series, from its ``start_date``."""
    start = dict(params).get("start_date")
    if "group/bond_yields_benchmark" in url:
        dates = pd.bdate_range(start or "2000-12-01", "2026-08-31")
        columns = {
            code: _yield_path("cdn" + code, dates, years, _CDN_OFFSET)
            for code, years in (("BD.CDN.2YR.DQ.YLD", 2), ("BD.CDN.5YR.DQ.YLD", 5), ("BD.CDN.10YR.DQ.YLD", 10), ("BD.CDN.LONG.DQ.YLD", 30))
        }
        return _valet_csv(columns, dates)
    code = url.rstrip("/").split("/")[-2]
    first, freq, years, margin = _VALET_SERIES[code]
    begin = max(pd.Timestamp(first), pd.Timestamp(start)) if start else pd.Timestamp(first)
    dates = pd.bdate_range(begin, "2026-08-31") if freq == "B" else pd.date_range(begin, "2026-08-31", freq=freq)
    if years == 0:
        values = np.round(np.maximum(_anchors(_SHORT, dates) + _CDN_OFFSET - 0.4, 0.25) + margin, 2)
    else:
        values = _yield_path("cdn" + code, dates, years, _CDN_OFFSET) + margin
    return _valet_csv({code: values}, dates)


def _zip(table_id: str, frame: pd.DataFrame) -> bytes:
    """A StatCan table ZIP: the data CSV and a metadata CSV (``cdn_data.statcan_zip_table_with_metadata``)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{table_id}.csv", frame.to_csv(index=False))
        archive.writestr(f"{table_id}_MetaData.csv", "Cube Title\nsynthetic\n")
    return buffer.getvalue()


def statcan(table_id: str) -> bytes:
    """Return a StatCan table ZIP with the columns and members the program selects on (millions; percent)."""
    quarters = pd.date_range("1990-01-01", "2026-04-01", freq="QS")
    ref = [f"{d:%Y-%m}" for d in quarters]
    if table_id == "36100467":
        values = _growth(quarters, 780_000, 3_600_000, "cdn_debt")
        frame = pd.DataFrame({"REF_DATE": ref, "GEO": "Canada", "Estimates": "Debt", "VALUE": np.round(values, 0)})
        other = frame.assign(Estimates="Gross debt to GDP", VALUE=60.0)
        return _zip(table_id, pd.concat([frame, other]))
    if table_id == "36100104":
        q = pd.date_range("1961-01-01", "2026-04-01", freq="QS")
        values = _growth(q, 42_000, 3_150_000, "cdn_gdp")
        frame = pd.DataFrame({
            "REF_DATE": [f"{d:%Y-%m}" for d in q], "GEO": "Canada",
            "Prices": "Current prices", "Seasonal adjustment": "Seasonally adjusted at annual rates",
            "Estimates": "Gross domestic product at market prices", "VALUE": np.round(values, 0),
        })
        return _zip(table_id, frame)
    if table_id == "10100015":
        rows = []
        sectors = {
            "Consolidated government": 1.0, "Federal government": 0.5,
            "Provincial and territorial government": 0.42, "Local government": 0.08,
        }
        items = {
            "Interest": (11_000, 21_000),
            "Debt securities, liabilities": (500_000, 2_400_000),
            "Loans, liabilities": (60_000, 250_000),
            "Currency and deposits, liabilities": (20_000, 60_000),
            "Special drawing rights (SDRs), liabilities": (1_000, 20_000),
            "Insurance and pension schemes, liabilities": (100_000, 500_000),
            "Other accounts payable, liabilities": (40_000, 200_000),
        }
        for sector, share in sectors.items():
            for item, (first, last) in items.items():
                values = _growth(quarters, first * share, last * share, sector + item)
                rows.append(pd.DataFrame({
                    "REF_DATE": ref, "GEO": "Canada", "Government sectors": sector,
                    "Statement of government operations and balance sheet": item, "VALUE": np.round(values, 0),
                }))
        return _zip(table_id, pd.concat(rows))
    if table_id == "17100009":
        q = pd.date_range("1946-01-01", "2026-07-01", freq="QS")
        values = _growth(q, 12_000_000, 41_700_000, "cdn_pop")
        frame = pd.DataFrame({"REF_DATE": [f"{d:%Y-%m}" for d in q], "GEO": "Canada", "SCALAR_FACTOR": "units", "VALUE": np.round(values, 0)})
        return _zip(table_id, frame)
    if table_id == "34100145":
        months = pd.date_range("1951-01-01", "2026-08-01", freq="MS")
        values = _yield_path("cdn_mortgage5", months, 10, _CDN_OFFSET) + 2.0
        frame = pd.DataFrame({
            "REF_DATE": [f"{d:%Y-%m}" for d in months], "GEO": "Canada", "DGUID": "2016A000011124",
            "UOM": "Percent", "SCALAR_FACTOR": "units", "VALUE": values,
        })
        return _zip(table_id, frame)
    raise KeyError(f"no synthetic StatCan table {table_id}")


# The UK (batch 3): gilt yields run this far above the U.S. ones; the Bank
# of England's database series: code -> (first day, "B" daily or "ME" month
# end, term in years (0: the short rate, Bank Rate), margin above it).
_UK_OFFSET = 0.6
_IADB_SERIES = {
    "IUDSNPY": ("1993-12-01", "B", 5, 0.0),
    "IUDMNPY": ("1993-11-01", "B", 10, 0.0),
    "IUDLNPY": ("2000-01-04", "B", 20, 0.0),
    "IUDBEDR": ("1975-01-02", "B", 0, 0.25),
    "IUMBV34": ("1995-01-31", "ME", 2, 1.2),
    "IUMBV37": ("1995-01-31", "ME", 3, 1.25),
    "IUMBV42": ("1995-01-31", "ME", 5, 1.3),
    "IUMTLMV": ("1995-01-31", "ME", 0, 3.0),
}
# ONS series: CDID -> (unit as its description gives it, first period,
# "stock" or "flow", first and last value in that unit, and whether it has
# months). Flows are per quarter; stocks at the period's end.
_ONS_SERIES = {
    "BKPX": ("m", "1975-01-01", "stock", 53_670, 3_195_774, True),
    "BKPW": ("m", "1975-01-01", "stock", 41_319, 3_181_031, True),
    "MDYT": ("", "1966-01-01", "stock", 7_832, 138_883, False),
    "HF6W": ("bn", "1975-01-01", "stock", 52.1, 2_985.5, True),
    "NMYX": ("m", "1946-01-01", "flow", 135, 33_518, False),
    "NMFX": ("m", "1946-01-01", "flow", 124, 33_146, False),
    "NUGW": ("m", "1946-01-01", "flow", 11, 372, False),
    "YBHA": ("m", "1955-01-01", "flow", 4_645, 787_197, False),
    "EBAQ": (",000", "1955-01-01", "stock", 50_901, 69_628, False),
}


def iadb(params: tuple) -> bytes:
    """A Bank of England database CSV (``CSVF=TN``): DATE, then the series asked for, dates "01 Sep 2026"."""
    code = dict(params)["SeriesCodes"]
    first, freq, years, margin = _IADB_SERIES[code]
    dates = pd.bdate_range(first, "2026-10-02") if freq == "B" else pd.date_range(first, "2026-08-31", freq="ME")
    if years == 0:
        values = np.round(np.maximum(_anchors(_SHORT, dates) + _UK_OFFSET - 0.5, 0.1) + margin, 2)
    else:
        values = np.round(_yield_path("uk" + code, dates, years, _UK_OFFSET) + margin, 4)
    lines = [f"DATE,{code}"] + [f"{day:%d %b %Y},{value:g}" for day, value in zip(dates, values)]
    return "\n".join(lines).encode()


def ons(cdid: str) -> bytes:
    """An ONS time series's JSON: its description (with the unit) and lists of months, quarters and years."""
    unit, first, kind, low, high, has_months = _ONS_SERIES[cdid]
    quarters = pd.date_range(first, "2026-04-01", freq="QS")
    values = _growth(quarters, low, high, "uk" + cdid)

    def row(date: str, value: float) -> dict:
        return {"date": date, "value": f"{value:.1f}" if unit == "bn" else f"{value:.0f}", "label": date}

    data = {
        "description": {"title": f"synthetic {cdid}", "unit": unit, "cdid": cdid},
        "quarters": [row(f"{d.year} Q{d.quarter}", v) for d, v in zip(quarters, values)],
        "months": [],
        "years": [],
    }
    if has_months:
        months = pd.date_range(first, "2026-08-01", freq="MS")
        monthly = _growth(months, low, high * 1.004, "ukm" + cdid)
        data["months"] = [row(f"{d.year} {d:%b}".upper(), v) for d, v in zip(months, monthly)]
    return json.dumps(data).encode()


# Germany (batch 4): Bund yields run this far below the U.S. ones. The
# Bundesbank's term structure: term code -> (years, first month of the
# end-of-month series, first day of the daily one).
_DE_OFFSET = -1.2
_BUNDESBANK_TERMS = {
    "R01XX": (1, "1972-09", "1997-08-01"), "R02XX": (2, "1972-09", "1997-08-01"), "R03XX": (3, "1972-09", "1997-08-01"),
    "R05XX": (5, "1972-09", "1997-08-01"), "R07XX": (7, "1972-09", "1997-08-01"), "R10XX": (10, "1972-09", "1997-08-01"),
    "R15XX": (15, "1986-06", "1997-08-01"), "R20XX": (20, "1986-06", "1997-08-01"), "R30XX": (30, "2000-01", "2000-08-01"),
}


def bundesbank(url: str) -> bytes:
    """A Bundesbank series CSV: its key and metadata lines, then ``date,value,flag`` ("." on days without a value)."""
    key = url.split("/rest/download/")[1].replace("/", ".", 1)
    frequency, term = key.split(".")[1], key.split(".")[9]
    years, first_month, first_day = _BUNDESBANK_TERMS[term]
    if frequency == "M":
        dates = pd.date_range(first_month, "2026-09-30", freq="ME")
        labels = [f"{day:%Y-%m}" for day in dates]
    else:
        dates = pd.date_range(first_day, "2026-10-02", freq="D")
        labels = [f"{day:%Y-%m-%d}" for day in dates]
    values = np.round(_yield_path("de" + key, dates, years, _DE_OFFSET), 2)
    lines = [f'"",{key},{key}_FLAGS', f'"",synthetic {term} / {frequency},', "BBK_UNIT_ENG,percent,", "unit multiplier,One,"]
    for day, label, value in zip(dates, labels, values):
        weekend = frequency == "D" and day.dayofweek >= 5
        lines.append(f"{label},.,No value available" if weekend else f"{label},{value:.2f},")
    return ("\ufeff" + "\n".join(lines)).encode()


def _ecb_change_days(name: str, margin: float) -> pd.Series:
    """A key rate on the days it changed: the short rate less ``margin``, in quarter points, from 1999."""
    months = pd.date_range("1999-01-01", "2026-09-01", freq="MS")
    path = np.maximum(np.round((_anchors(_SHORT, months) - margin) * 4) / 4, 0.0)
    rates = pd.Series(path, index=months)
    return rates.loc[rates.ne(rates.shift())]


def ecb(url: str) -> bytes:
    """An ECB csvdata answer: one row per observation, KEY, TIME_PERIOD, OBS_VALUE, its unit and multiplier."""
    flow, key = url.split("/service/data/")[1].split("/", 1)
    unit_column, unit, multiplier = "UNIT", "PCPA", 0
    if flow == "FM":
        rates = _ecb_change_days("ecb", 1.0)
        variable = (rates.index >= "2000-06-28") & (rates.index < "2008-10-15")
        if key.endswith("MRR_MBR.LEV"):
            chosen = pd.concat([pd.Series([rates.asof(pd.Timestamp("2000-06-28"))], index=[pd.Timestamp("2000-06-28")]),
                                rates.loc[variable & (rates.index > "2000-06-28")],
                                pd.Series([np.nan], index=[pd.Timestamp("2008-10-15")])])
        else:
            chosen = pd.concat([rates.loc[~variable & (rates.index < "2000-06-28")],
                                pd.Series([rates.asof(pd.Timestamp("2008-10-15"))], index=[pd.Timestamp("2008-10-15")]),
                                rates.loc[rates.index > "2008-10-15"]])
        periods, values = [f"{day:%Y-%m-%d}" for day in chosen.index], list(chosen.to_numpy())
    elif flow == "MIR":
        months = pd.date_range("2000-01-31", "2026-08-31", freq="ME")
        band = {"I": 3, "O": 10, "P": 20}[key.split(".")[4]]
        values = list(np.round(_yield_path("demir" + key, months, band, _DE_OFFSET) + 1.4, 2))
        periods = [f"{day:%Y-%m}" for day in months]
    elif flow == "GFS":
        # Unification doubled the debt in the 1990s; from 2000 the year-ends
        # follow the Eurostat quarters' path (equal on real data).
        early = pd.date_range("1991-12-31", "1999-12-31", freq="YE")
        quarters = pd.date_range("2000-03-31", "2026-03-31", freq="QE")
        later = pd.Series(_growth(quarters, 1_265_509, 2_902_035, "de_debt"), index=quarters).loc[lambda q: q.index.month == 12]
        years = early.append(later.index)
        values = list(np.round(np.concatenate([_growth(early, 618_218, 1_253_598, "de_gfs_a"), later.to_numpy()]), 0))
        periods, unit_column, unit, multiplier = [str(day.year) for day in years], "UNIT_MEASURE", "XDC", 6
    else:
        raise KeyError(f"no synthetic ECB series {flow}.{key}")
    frame = pd.DataFrame({"KEY": f"{flow}.{key}", "TIME_PERIOD": periods, "OBS_VALUE": values,
                          unit_column: unit, "UNIT_MULT": str(multiplier)})
    return frame.to_csv(index=False).encode()


def _jsonstat(label: str, dims: list[tuple[str, list[str]]], value) -> bytes:
    """A JSON-stat 2.0 dataset over ``dims`` (the last is time); ``value(categories)`` gives each cell (None: no value)."""
    sizes = [len(codes) for _dim, codes in dims]
    values = {}
    for flat, combo in enumerate(itertools.product(*[codes for _dim, codes in dims])):
        cell = value(dict(zip([dim for dim, _codes in dims], combo)))
        if cell is not None:
            values[str(flat)] = round(float(cell), 1)
    return json.dumps({
        "version": "2.0", "class": "dataset", "label": label, "id": [dim for dim, _codes in dims], "size": sizes,
        "dimension": {dim: {"category": {"index": {code: i for i, code in enumerate(codes)}}} for dim, codes in dims},
        "value": values,
    }).encode()


# Eurostat: the German series, first and last value (millions of euros,
# thousands of persons, persons) and each sector's share of the total.
_DE_SECTOR_SHARES = {"S13": 1.0, "S1311": 0.70, "S1312": 0.235, "S1313": 0.07, "S1314": 0.005}


def eurostat(url: str, params: tuple) -> bytes:
    """A Eurostat JSON-stat answer for the query the program sends (``de_data.EUROSTAT_QUERIES``)."""
    dataset = url.rstrip("/").rsplit("/", 1)[1]
    query = dict(params)
    unit = query.get("unit", "NR")
    quarters = lambda first, last: [f"{p.year}-Q{p.quarter}" for p in pd.period_range(first, last, freq="Q")]  # noqa: E731
    fixed = [(dim, [query[dim]]) for dim in ("na_item", "s_adj", "age", "sex") if dim in query]
    common = [("freq", ["Q"]), ("unit", [unit]), *fixed, ("geo", ["DE"])]

    def path(name: str, periods: list[str], first: float, last: float) -> dict[str, float]:
        dates = pd.DatetimeIndex([pd.Period(period.replace("-", ""), freq="Q" if "Q" in period else "Y").end_time.normalize() for period in periods])
        return dict(zip(periods, _growth(dates, first, last, name)))

    if dataset == "gov_10q_ggdebt":
        times = quarters("2000Q1", "2026Q1")
        total = path("de_debt", times, 1_265_509, 2_902_035)
        dims = [*common[:2], ("sector", list(_DE_SECTOR_SHARES)), *common[2:], ("time", times)]
        return _jsonstat("Quarterly government debt", dims, lambda c: total[c["time"]] * _DE_SECTOR_SHARES[c["sector"]] * (1.01 if c["sector"] != "S13" else 1))
    if dataset == "gov_10q_ggnfa":
        times = quarters("2002Q1", "2026Q1")
        total = path("de_interest", times, 16_304, 12_135)
        dims = [*common[:2], ("sector", list(_DE_SECTOR_SHARES)), *common[2:], ("time", times)]
        return _jsonstat("Quarterly non-financial accounts", dims, lambda c: total[c["time"]] * _DE_SECTOR_SHARES[c["sector"]])
    if dataset == "gov_10dd_edpt1":
        times = [str(year) for year in range(1995, 2026)]
        interest = path("de_interest_a", times, 69_498, 49_544)
        dims = [("freq", ["A"]), ("unit", [unit]), ("sector", ["S13"]), ("na_item", ["D41PAY", "GD"]), ("geo", ["DE"]), ("time", times)]
        return _jsonstat("Government deficit/surplus, debt", dims, lambda c: interest[c["time"]] if c["na_item"] == "D41PAY" else None)
    if dataset == "namq_10_gdp":
        times = quarters("1991Q1", "2026Q2")
        gdp = path("de_gdp", times, 358_976, 1_151_030)
        return _jsonstat("GDP", [*common, ("time", times)], lambda c: gdp[c["time"]])
    if dataset == "namq_10_pe":
        times = quarters("1991Q1", "2026Q2")
        people = path("de_pop_q", times, 79_777, 83_337)
        return _jsonstat("Population", [*common, ("time", times)], lambda c: people[c["time"]])
    if dataset == "demo_pjan":
        times = [str(year) for year in range(1960, 2026)]
        # West Germany to 1990, then unified Germany: two paths, a step between.
        west = path("de_pop_w", times, 55_257_088, 62_679_035 * 62_679_035 / 55_257_088)
        unified = path("de_pop_u", times, 75_000_000, 83_577_140)
        dims = [("freq", ["A"]), ("unit", ["NR"]), *fixed, ("geo", ["DE"]), ("time", times)]
        return _jsonstat("Population on 1 January", dims, lambda c: west[c["time"]] if int(c["time"]) <= 1990 else unified[c["time"]])
    raise KeyError(f"no synthetic Eurostat dataset {dataset}")


def synth(key: tuple):
    """Return the synthetic answer to one download-cache key; KeyError for a source with none."""
    if key[0] == "fred":
        return fred(key[1])
    tag, url, params = key
    if "bankofcanada.ca/valet" in url:
        return _response(valet(url, params))
    if "statcan.gc.ca/n1/tbl/csv/" in url:
        return _response(statcan(url.rsplit("/", 1)[-1].split("-")[0]))
    if "bankofengland.co.uk/boeapps/database" in url:
        return _response(iadb(params))
    if "ons.gov.uk" in url:
        return _response(ons(url.split("/timeseries/")[1].split("/")[0].upper()))
    if "api.statistiken.bundesbank.de/rest/download/" in url:
        return _response(bundesbank(url))
    if "data-api.ecb.europa.eu/service/data/" in url:
        return _response(ecb(url))
    if "ec.europa.eu/eurostat/api/dissemination" in url:
        return _response(eurostat(url, params))
    raise KeyError(f"no synthetic source for {key!r}")


def _refused(*_args, **_kwargs):
    raise requests.ConnectionError("synthetic data only: real requests are refused")


def install_synthetic_cache(http_module, cache_dir: Path) -> bool:
    """Answer every download from ``synth`` (kept as pickles in ``cache_dir``), and refuse every real request.

    The same interface as ``regress_charts.install_disk_cache``: returns
    False, changing nothing, for a tree older than the program's download
    cache. ``cache_dir`` must not be the real download cache.
    """
    if not hasattr(http_module, "_cached"):
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(key, _download):
        path = cache_dir / (hashlib.sha1(repr(key).encode()).hexdigest() + ".pkl")
        if path.exists():
            return pickle.loads(path.read_bytes())
        try:
            result = synth(key)
        except KeyError as exc:
            print(f"[synthetic] no synthetic answer for {key!r}", file=sys.stderr)
            raise requests.ConnectionError(str(exc)) from None
        path.write_bytes(pickle.dumps(result))
        return result

    http_module._cached = cached
    # The quotes (not cached), the bake and anything else that would reach out.
    requests.Session.get = _refused  # type: ignore[assignment]
    requests.Session.post = _refused  # type: ignore[assignment]
    return True
