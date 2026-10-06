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
-> DataFrame; ``("canadian", url, params)`` -> Response. A source with no
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


def synth(key: tuple):
    """Return the synthetic answer to one download-cache key; KeyError for a source with none."""
    if key[0] == "fred":
        return fred(key[1])
    tag, url, params = key
    if "bankofcanada.ca/valet" in url:
        return _response(valet(url, params))
    if "statcan.gc.ca/n1/tbl/csv/" in url:
        return _response(statcan(url.rsplit("/", 1)[-1].split("-")[0]))
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
