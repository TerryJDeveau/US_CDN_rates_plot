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
    CANADIAN_YIELD_HIST_END,
    DATE_COLUMN,
    MORTGAGE_OWN_COLORS,
    MORTGAGE_STYLE,
    MORTGAGE_TERMS,
    POLICY_RATE_STYLE,
    PRIME_RATE_STYLE,
    SPREAD_COLORS,
    SPREAD_STYLE,
    TERM_COLORS,
    VARIABLE_MORTGAGE_STYLE,
    YIELD_TERMS,
    PlotConfig,
    nation_by_key,
)
from .de_data import ecb_series
from .de_data import YIELD_HISTORY_ATTR as DE_YIELD_HISTORY_ATTR
from .http import fetch_fred_csv
from .uk_archive_data import EMBEDDED_UK_BANK_RATE_HISTORY, EMBEDDED_UK_MORTGAGE_HISTORY
from .uk_data import YIELD_HISTORY_ATTR, iadb_series

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
# measured by Terry 2026-10-06. The chartered banks' prime rate, weekly on
# Valet from 1975: V80691311 (found by a web search; fetched by
# tools/check_sources.py boc_prime on Terry's machine, 2026-10-06).
US_MORTGAGE_SERIES = {"30": "MORTGAGE30US", "15": "MORTGAGE15US"}
CDN_MORTGAGE_SERIES = {"5": "V80691335", "3": "V80691334", "1": "V80691333", "5v": "BROKER_AVERAGE_5YR_VRM", "prime": "V80691311"}
CDN_MORTGAGE_HISTORY_TABLE = "34100145"
# The UK (batch 3), from the Bank of England's database (IADB), measured from
# Terry's laptop 2026-10-06: Bank Rate daily from 1975, and before it its
# changes since 1833 (the millennium dataset, baked). Mortgages: the Bank's
# quoted rates for new loans at 75 % loan to value, monthly from 1995, the
# standard variable rate carried back to 1939 by the millennium dataset's
# spliced variable mortgage rate (baked). IADB dates a month by its last
# day; here by its first, as Canada's monthly rates are, so its step covers it.
UK_POLICY_SERIES = "IUDBEDR"
UK_MORTGAGE_SERIES = {"2f": "IUMBV34", "3f": "IUMBV37", "5f": "IUMBV42", "svr": "IUMTLMV"}
# Germany (batch 4), from the ECB Data Portal, measured from Terry's laptop
# 2026-10-06. Policy: the main refinancing rate, on the days it changed since
# 1999: the fixed rate of the fixed-rate tenders, and from 2000-06-28 to
# 2008-10-14, when the tenders were variable, their minimum bid rate. (The
# Bundesbank's discount rate before 1999 is still to be found.) Mortgages:
# the MFI interest rates on new housing loans to households in Germany, by
# initial rate fixation, monthly from 2000 (2000-2002 the ECB's estimates;
# reported from 2003), dated by each month's first day as the UK's are.
DE_POLICY_FIXED = ("FM", "B.U2.EUR.4F.KR.MRR_FR.LEV", "PCPA")
DE_POLICY_MINIMUM_BID = ("FM", "B.U2.EUR.4F.KR.MRR_MBR.LEV", "PCPA")
DE_MORTGAGE_SERIES = {"1-5": "M.DE.B.A2C.I.R.A.2250.EUR.N", "5-10": "M.DE.B.A2C.O.R.A.2250.EUR.N", "over10": "M.DE.B.A2C.P.R.A.2250.EUR.N"}
_MORTGAGE_LABELS = {
    "30": "30-Year Mortgage",
    "15": "15-Year Mortgage",
    "5": "5-Year Mortgage (posted)",
    "3": "3-Year Mortgage (posted)",
    "1": "1-Year Mortgage (posted)",
    "5v": "5-Year Variable Mortgage (broker avg.)",
    "prime": "Prime Rate (chartered banks)",
    "2f": "2-Year Fixed Mortgage (quoted, 75% LTV)",
    "3f": "3-Year Fixed Mortgage (quoted, 75% LTV)",
    "5f": "5-Year Fixed Mortgage (quoted, 75% LTV)",
    "svr": "Standard Variable Rate (quoted)",
    "1-5": "Mortgage, fixed 1–5 Years (new loans, avg.)",
    "5-10": "Mortgage, fixed 5–10 Years (new loans, avg.)",
    "over10": "Mortgage, fixed over 10 Years (new loans, avg.)",
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
    # its end label written in basis points ("+47 pts" for 0.47 points).
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


def _baked(rows: list[tuple[str, float]]) -> pd.Series:
    """Return a baked list of rates as a cleaned date-indexed series (``_clean``); empty if never baked."""
    return _clean(pd.Series([value for _date, value in rows], index=pd.DatetimeIndex([date for date, _value in rows])))


def _joined(history: pd.Series, live: pd.Series) -> pd.Series:
    """Return ``history`` until ``live`` begins, then ``live``."""
    return pd.concat([history.loc[history.index < live.index[0]], live]) if not live.empty else history


def uk_policy_rate() -> RateCurve | None:
    """Return the Bank of England's official Bank Rate: its changes since 1833 (baked), then daily from 1975.

    If the database fails, the baked changes alone (to 2017), with a warning;
    None if there is neither.
    """
    history = _baked(EMBEDDED_UK_BANK_RATE_HISTORY)
    try:
        live = _clean(iadb_series(UK_POLICY_SERIES))
    except Exception as exc:
        print(f"  Warning: Bank of England Bank Rate ({UK_POLICY_SERIES}) unavailable ({exc}); "
              + ("its baked changes alone, to 2017." if not history.empty else "not drawn."))
        live = history.iloc[:0]
    values = _joined(history, live)
    if values.empty:
        return None
    return RateCurve("policy", "Official Bank Rate", values, POLICY_RATE_STYLE, title="Policy Rate")


def _uk_mortgage(term: str) -> tuple[pd.Series, str]:
    """Return one UK mortgage term's monthly rate, dated by each month's first day, and its label.

    The standard variable rate is carried back by the baked variable
    mortgage rate (from 1939), and labelled so.
    """
    monthly = iadb_series(UK_MORTGAGE_SERIES[term])
    values = _clean(monthly.set_axis(monthly.index.to_period("M").start_time))
    label = _MORTGAGE_LABELS[term]
    history = _baked(EMBEDDED_UK_MORTGAGE_HISTORY) if term == "svr" else values.iloc[:0]
    if not history.empty and not values.empty and history.index[0] < values.index[0]:
        values = _joined(history, values)
        label = f"Variable Mortgage Rate, quoted SVR from {monthly.index[0]:%Y-%m}"
    return values, label


def de_policy_rate() -> RateCurve | None:
    """Return the ECB's main refinancing rate since 1999: fixed, the minimum bid while the tenders were variable, fixed again.

    Both series hold only the days the rate changed: each value is carried
    daily to the next change, and the last to today (it holds until changed). If the minimum bid rate
    fails, the fixed rate alone, with a warning (wrong 2000-2008); None if
    the fixed rate fails.
    """
    try:
        fixed = _clean(ecb_series(*DE_POLICY_FIXED))
    except Exception as exc:
        print(f"  Warning: ECB main refinancing rate ({DE_POLICY_FIXED[1]}) unavailable ({exc}); not drawn.")
        return None
    try:
        bid = _clean(ecb_series(*DE_POLICY_MINIMUM_BID))
    except Exception as exc:
        print(f"  Warning: ECB minimum bid rate ({DE_POLICY_MINIMUM_BID[1]}) unavailable ({exc}); "
              "the fixed rate alone, wrong while the tenders were variable (2000-2008).")
        bid = fixed.iloc[:0]
    values = fixed
    if not bid.empty:
        # The fixed rate's first change after the variable tenders began ends them.
        resumed = fixed.loc[fixed.index > bid.index[0]]
        end = resumed.index[0] if not resumed.empty else pd.Timestamp.max
        values = pd.concat([fixed.loc[fixed.index < bid.index[0]], bid.loc[bid.index < end], resumed])
    # Each day's rate, carried from its change to the next and on to today,
    # so a window ending between changes is drawn to its end (as the UK's
    # daily Bank Rate is).
    today = pd.Timestamp.today().normalize()
    values = values.reindex(pd.date_range(values.index[0], max(today, values.index[-1]), freq="D")).ffill()
    return RateCurve("policy", "ECB Main Refinancing Rate", values, POLICY_RATE_STYLE, title="Policy Rate")


def _de_mortgage(term: str) -> pd.Series:
    """Return one German mortgage band's monthly rate, dated by each month's first day."""
    monthly = ecb_series("MIR", DE_MORTGAGE_SERIES[term], "PCPA")
    return _clean(monthly.set_axis(monthly.index.to_period("M").start_time))


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
    if term == "prime":
        return PRIME_RATE_STYLE | {"color": colour}
    return (VARIABLE_MORTGAGE_STYLE if term.endswith("v") else MORTGAGE_STYLE) | {"color": colour}


def mortgage_rate(term: str) -> RateCurve | None:
    """Return one mortgage term's rate (``config.MORTGAGE_TERMS``), or None with a warning if its source fails.

    Canada's 5-year: the posted weekly rate, carried back before it begins by
    the monthly CMHC rate (with a warning, and from 1975 only, if that fails).
    """
    country, yield_term = MORTGAGE_TERMS[term]
    label = _MORTGAGE_LABELS[term]
    code = {"us": US_MORTGAGE_SERIES, "cdn": CDN_MORTGAGE_SERIES, "uk": UK_MORTGAGE_SERIES, "de": DE_MORTGAGE_SERIES}[country][term]
    try:
        if country == "uk":
            values, label = _uk_mortgage(term)
        elif country == "de":
            values = _de_mortgage(term)
        else:
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


def warn_none_drawn(country: str, curve: str, why: str) -> None:
    """Warn that a nation's chart draws no ``curve`` of a chosen list, ``why`` naming what it has.

    Terry, 2026-10-07, of a list for every chart that leaves a shown nation
    none of its terms (``-c --yields:7,20``): "plot no yields in that case,
    but issue a warning message". The same words for a nation's own list
    of none (``--cdn:yields:none``), which is how the window and the page
    hold that case, so the command line's log and theirs read alike.
    """
    print(f"  Warning: no {curve} on the {nation_by_key(country).adjective} chart: {why}.")


def spread_label(pair: tuple[str, str]) -> str:
    """Return a spread's legend label: ``("10y", "2y")`` -> "10Y–2Y Spread"."""
    return f"{pair[0].upper()}–{pair[1].upper()} Spread"


def yield_spreads(country: str, yields: pd.DataFrame, config: PlotConfig) -> list[RateCurve]:
    """Return the chosen spreads (``config.spread_pairs``), each the first term's yield less the second's.

    Taken from the country's prepared yields, so with --cur they run on to
    the day's quotes. A value is drawn where both terms have one. Canada's
    yields are monthly before 2001, so its spreads are steps there, as its
    yields are. A pair with a term the country has no yield for is named and
    left out, and the colours go to the pairs drawn, in order; with none of
    the pairs its terms make, one warning says so.
    """
    nation = nation_by_key(country)
    if not nation.view("spread_pairs", config.spread_pairs):
        warn_none_drawn(country, "spread", f"no spread chosen is of two of its yield terms ({','.join(nation.yield_terms)})")
        return []
    if yields.empty:
        return []
    frame = yields.set_index(DATE_COLUMN) if DATE_COLUMN in yields.columns else yields
    curves = []
    for pair in config.spread_pairs:
        first, second = (YIELD_TERMS[term] for term in pair)
        if first not in frame.columns or second not in frame.columns:
            name = nation_by_key(country).adjective
            print(f"  Note: no {name} {first if first not in frame.columns else second} yield; no {spread_label(pair)} on this chart.")
            continue
        values = _clean(frame[first] - frame[second])
        # Coloured by its place among the spreads drawn, so a pair left off
        # takes no colour: Canada's chart of 7y-1m,10y-2y draws its 10y-2y
        # as its chart of 10y-2y alone does (options.nation_view).
        style = SPREAD_STYLE | {"color": SPREAD_COLORS[len(curves) % len(SPREAD_COLORS)]}
        steps_until = CANADIAN_YIELD_HIST_END if country == "cdn" else _steps_until(yields, (first, second))
        key = f"spread_{pair[0]}-{pair[1]}"
        curves.append(RateCurve(key, spread_label(pair), values, style, steps_until, spread=True, title="Spreads"))
    return curves


def _steps_until(yields: pd.DataFrame, columns: tuple[str, str]) -> pd.Timestamp | None:
    """Return the last date a spread is monthly: where the later of its two terms' monthly history ends (None: neither has one).

    The UK's 10- and 20-year are monthly averages before their daily par
    yields begin (``uk_data.fetch_uk_yields``), and Germany's terms are
    end-of-month values before their daily ones (``de_data.fetch_de_yields``),
    so a spread of them is too. Empty for the U.S., whose yields carry no
    such record.
    """
    assert YIELD_HISTORY_ATTR == DE_YIELD_HISTORY_ATTR  # one record, read alike for both nations
    through = [date for column, date in yields.attrs.get(YIELD_HISTORY_ATTR, {}).items() if column in columns]
    return max(through) if through else None


def rate_curves(country: str, yields: pd.DataFrame, config: PlotConfig) -> list[RateCurve]:
    """Return one country's chosen curves for the yield axis ("cdn", "us", "uk" or "de"), in drawing order.

    ``yields`` is the country's prepared yield frame (date-indexed for
    Canada, the UK and Germany, a ``DATE`` column for the U.S.), which the spreads are taken from.
    Each curve is fetched whole (from its first observation) and cut to the
    window when drawn, so the automatic start sees where it begins.
    """
    curves: list[RateCurve | None] = []
    if config.policy_rates:
        print("Fetching the policy rate …")
        curves.append({"us": us_policy_rate, "cdn": cdn_policy_rate, "uk": uk_policy_rate, "de": de_policy_rate}[country]())
    if config.mortgages:
        terms = [term for term in config.mortgage_terms if MORTGAGE_TERMS[term][0] == country]
        if not terms:
            own = ",".join(nation_by_key(country).mortgage_terms)
            warn_none_drawn(country, "mortgage rate", f"none of its mortgage terms ({own}) is chosen")
        else:
            print("Fetching mortgage rates …")
        curves += [mortgage_rate(term) for term in terms]
    if config.spreads:
        curves += yield_spreads(country, yields, config)
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
