"""The other curves on the yield (left) axis: policy rates, mortgage rates and yield spreads.

They are additive and off by default: each is drawn only when its option is
given (--policy, --mortgages, --spreads), and naming one does not hide the
default curves (they are not under the curve rule of ``options``). Each
country's are fetched by ``rate_curves`` into ``RateCurve`` records, which
``plotting.draw_country`` draws on the yield axis after the yields, in the
yields' legend group; ``plotting.curve_first_dates`` counts them for the
automatic start only when chosen.

Every value is a percentage (a spread, percentage points), dated by the day
it applies from. A source that fails prints a ``Warning:`` and its curve is
left out; nothing is substituted silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .cdn_data import boc_valet_series, statcan_zip_table
from .config import (
    MORTGAGE_OWN_COLORS,
    MORTGAGE_STYLE,
    MORTGAGE_TERMS,
    POLICY_RATE_STYLE,
    TERM_COLORS,
    VARIABLE_MORTGAGE_STYLE,
    YIELD_TERMS,
    PlotConfig,
)
from .http import fetch_fred_csv

# Policy rates (--policy). U.S.: the effective federal funds rate, daily from
# 1954-07-01. Canada: the Bank Rate, monthly from 1935, until the Canadian
# Overnight Repo Rate Average (CORRA) begins, daily from 1997; series and
# spans measured by Terry 2026-10-06. Since 1996 the Bank Rate has been the top
# of the Bank of Canada's operating band, a quarter point above its target
# for the overnight rate, which CORRA tracks: the joined curve steps down
# about 0.25 there (a change of measure, as the legend says).
US_POLICY_SERIES = "DFF"
CDN_BANK_RATE_SERIES = "V122530"
CDN_CORRA_SERIES = "AVG.INTWO"
# Valet returns a series from this date; it precedes every series asked for.
_VALET_FROM = "1900-01-01"

# Mortgage rates (--mortgages), by term (config.MORTGAGE_TERMS). U.S.: Freddie
# Mac's survey averages, weekly, on FRED (30-year from 1971, 15-year from 1991).
# Canada: the chartered banks' posted rates, weekly on Valet (5-year from 1975,
# 3- and 1-year from 1980), the 5-year carried back to 1951 by Statistics
# Canada's monthly CMHC conventional 5-year lending rate (table 34-10-0145);
# and the 5-year variable rate, the brokers' average, from 2011. Spans
# measured by Terry 2026-10-06.
US_MORTGAGE_SERIES = {"30": "MORTGAGE30US", "15": "MORTGAGE15US"}
CDN_MORTGAGE_SERIES = {"5": "V80691335", "3": "V80691334", "1": "V80691333", "5v": "BROKER_AVERAGE_5YR_VRM"}
CDN_MORTGAGE_HISTORY_TABLE = "34100145"
_MORTGAGE_LABELS = {
    "30": "30-Year Mortgage",
    "15": "15-Year Mortgage",
    "5": "5-Year Mortgage (posted)",
    "3": "3-Year Mortgage (posted)",
    "1": "1-Year Mortgage (posted)",
    "5v": "5-Year Variable Mortgage (broker avg.)",
}


@dataclass(frozen=True)
class RateCurve:
    """One curve on the yield axis other than the yields themselves."""

    key: str  # what it is, for its style and tests: "policy", "mortgage_30", "spread_10y-2y"
    label: str  # its legend label
    values: pd.Series  # date-indexed, sorted, in percent (a spread: percentage points); no missing values
    style: dict = field(default_factory=dict)  # matplotlib line properties
    # Drawn as steps up to and including this date and as a line after it (the
    # Canadian yields' monthly history before 2001); None: as ``style`` says.
    steps_until: pd.Timestamp | None = None
    # A spread: drawn thick, its inverted stretches (below zero) shaded, and
    # its end label written in points ("+0.47 pts").
    spread: bool = False
    title: str = ""  # its phrase in the chart title ("Policy Rate"); curves of a kind share one
    # The yield column whose drawn colour it takes, if that yield is on the
    # chart (a mortgage: its term's yield); its style's colour otherwise.
    color_of: str | None = None

    @property
    def first(self) -> pd.Timestamp:
        """The first date with a value (the automatic start counts it)."""
        return pd.Timestamp(self.values.index[0])


def in_window(values: pd.Series, config: PlotConfig) -> pd.Series:
    """Return ``values`` from ``config.start`` to ``config.end``, the value in effect at the start carried to it.

    A rate holds until it changes (a monthly Bank Rate, a weekly mortgage
    rate), so a window starting between two observations begins with the
    one before it, as the right-axis curves do (``us_data.fetch_us_macro``).
    """
    inside = values.loc[(values.index >= config.start) & (values.index <= config.end)]
    before = values.loc[values.index < config.start]
    if before.empty or (not inside.empty and inside.index[0] == config.start):
        return inside
    carried = pd.Series([before.iloc[-1]], index=pd.DatetimeIndex([config.start]))
    return pd.concat([carried, inside])


def _clean(values: pd.Series) -> pd.Series:
    """Return a fetched series as a RateCurve holds it: numeric, date-sorted, without missing values."""
    values = pd.to_numeric(values, errors="coerce").dropna().sort_index()
    return values.set_axis(pd.DatetimeIndex(values.index).normalize())


def _valet(series_code: str) -> pd.Series:
    """Return one whole Bank of Canada Valet series, cleaned (``_clean``)."""
    return _clean(boc_valet_series(series_code, start=_VALET_FROM))


def _fred(series_id: str) -> pd.Series:
    """Return one whole FRED series, cleaned (``_clean``)."""
    frame = fetch_fred_csv(series_id)
    return _clean(frame.set_index("DATE")[series_id])


def us_policy_rate() -> RateCurve | None:
    """Return the effective federal funds rate (FRED DFF), or None with a warning if FRED fails."""
    try:
        values = _fred(US_POLICY_SERIES)
    except Exception as exc:
        print(f"  Warning: effective federal funds rate ({US_POLICY_SERIES}) unavailable ({exc}); not drawn.")
        return None
    return RateCurve("policy", "Fed Funds Rate (effective)", values, POLICY_RATE_STYLE, title="Policy Rate")


def cdn_policy_rate() -> RateCurve | None:
    """Return the Bank Rate joined to CORRA where CORRA begins (see ``CDN_CORRA_SERIES``); None if both fail.

    Either alone is drawn, with a warning, if the other fails.
    """
    parts: dict[str, pd.Series] = {}
    for name, code in (("Bank Rate", CDN_BANK_RATE_SERIES), ("CORRA", CDN_CORRA_SERIES)):
        try:
            parts[name] = _valet(code)
        except Exception as exc:
            print(f"  Warning: Bank of Canada {name} ({code}) unavailable ({exc}).")
    if not parts:
        print("  Warning: no Canadian policy rate; not drawn.")
        return None
    if len(parts) == 1:
        (name, values), = parts.items()
        return RateCurve("policy", name, values, POLICY_RATE_STYLE, title="Policy Rate")
    bank, corra = parts["Bank Rate"], parts["CORRA"]
    joined = pd.concat([bank.loc[bank.index < corra.index[0]], corra])
    label = f"Bank Rate, CORRA from {corra.index[0]:%Y-%m}"
    return RateCurve("policy", label, joined, POLICY_RATE_STYLE, title="Policy Rate")


def cdn_mortgage_history() -> pd.Series:
    """Return the CMHC conventional 5-year mortgage lending rate, monthly from 1951 (StatCan 34-10-0145).

    The table is checked, not trusted: Canada only, in percent, one row per
    month; anything else raises (and the caller warns). Each month is dated
    by its first day.
    """
    table = statcan_zip_table(CDN_MORTGAGE_HISTORY_TABLE)
    selected = table.loc[table["GEO"].eq("Canada")] if "GEO" in table.columns else table
    if "UOM" in selected.columns and set(selected["UOM"].astype(str).str.strip()) != {"Percent"}:
        raise ValueError(f"expected percent, found units {sorted(set(selected['UOM'].astype(str)))}")
    if selected.empty or selected["REF_DATE"].duplicated().any():
        raise ValueError("expected one row per month for Canada")
    dates = pd.DatetimeIndex(pd.to_datetime(selected["REF_DATE"].astype(str) + "-01"))
    return _clean(pd.Series(selected["VALUE"].to_numpy(), index=dates))


def _mortgage_style(term: str) -> dict:
    """Return a mortgage term's line style, coloured as its yield term (or its own colour)."""
    yield_term = MORTGAGE_TERMS[term][1]
    colour = TERM_COLORS[yield_term] if yield_term is not None else MORTGAGE_OWN_COLORS[term]
    return (VARIABLE_MORTGAGE_STYLE if term.endswith("v") else MORTGAGE_STYLE) | {"color": colour}


def mortgage_rate(term: str) -> RateCurve | None:
    """Return one mortgage term's rate (``config.MORTGAGE_TERMS``), or None with a warning if its source fails.

    Canada's 5-year: the posted weekly rate, carried back before it begins by
    the monthly CMHC rate (with a warning, and from 1975 only, if that fails).
    """
    country, yield_term = MORTGAGE_TERMS[term]
    label = _MORTGAGE_LABELS[term]
    code = (US_MORTGAGE_SERIES if country == "us" else CDN_MORTGAGE_SERIES)[term]
    try:
        values = _fred(code) if country == "us" else _valet(code)
    except Exception as exc:
        print(f"  Warning: {label} ({code}) unavailable ({exc}); not drawn.")
        return None
    if country == "cdn" and term == "5":
        try:
            history = cdn_mortgage_history()
        except Exception as exc:
            print(
                f"  Warning: StatCan {CDN_MORTGAGE_HISTORY_TABLE} (5-year mortgage rate before "
                f"{values.index[0]:%Y-%m-%d}) unavailable ({exc}); drawn from then only."
            )
        else:
            values = pd.concat([history.loc[history.index < values.index[0]], values])
    color_of = YIELD_TERMS[yield_term] if yield_term is not None else None
    return RateCurve(f"mortgage_{term}", label, values, _mortgage_style(term), title="Mortgage Rates", color_of=color_of)


def rate_curves(country: str, yields: pd.DataFrame, config: PlotConfig) -> list[RateCurve]:
    """Return one country's chosen curves for the yield axis ("cdn" or "us"), in drawing order.

    ``yields`` is the country's prepared yield frame (date-indexed for
    Canada, a ``DATE`` column for the U.S.), which the spreads are taken from.
    Each curve is fetched whole (from its first observation) and cut to the
    window when drawn, so the automatic start sees where it begins.
    """
    curves: list[RateCurve | None] = []
    if config.policy_rates:
        print("Fetching the policy rate …")
        curves.append(us_policy_rate() if country == "us" else cdn_policy_rate())
    if config.mortgages:
        terms = [term for term in config.mortgage_terms if MORTGAGE_TERMS[term][0] == country]
        if not terms:
            name = "U.S." if country == "us" else "Canadian"
            print(f"  Note: none of the mortgage terms {','.join(config.mortgage_terms)} is {name}; no mortgage rate on this chart.")
        else:
            print("Fetching mortgage rates …")
        curves += [mortgage_rate(term) for term in terms]
    chosen = []
    for curve in curves:
        if curve is None:
            continue
        if curve.values.empty:
            print(f"  Warning: {curve.label}: the source returned no values; not drawn.")
        elif curve.first > config.end:
            print(f"  Note: {curve.label} begins {curve.first:%Y-%m-%d}, after the chart ends; not drawn.")
        else:
            chosen.append(curve)
    return chosen
