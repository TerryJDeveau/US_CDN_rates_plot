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

from .config import PlotConfig


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


def rate_curves(country: str, yields: pd.DataFrame, config: PlotConfig) -> list[RateCurve]:
    """Return one country's chosen curves for the yield axis ("cdn" or "us"), in drawing order.

    ``yields`` is the country's prepared yield frame (date-indexed for
    Canada, a ``DATE`` column for the U.S.), which the spreads are taken from.
    """
    return []
