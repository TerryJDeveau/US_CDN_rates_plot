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

from .cdn_data import boc_valet_series
from .config import POLICY_RATE_STYLE, PlotConfig
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
