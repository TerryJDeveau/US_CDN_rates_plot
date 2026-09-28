"""Matplotlib axis configuration: scales, tick locators, formatters and limits."""

from __future__ import annotations

import bisect
import math
import warnings
from typing import Iterable

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .config import (
    AXIS_LABEL_PAD_PT,
    DATE_PAD_FRACTION,
    DEFAULT_YIELD_YLIM,
    LABEL_FS,
    MACRO_MIN_PAD_DECADES,
    MACRO_PAD_FRACTION,
    MIN_DATE_PAD_DAYS,
    TICK_FS,
    YIELD_BOTTOM_PAD_FRACTION,
    YIELD_MIN_BOTTOM_PAD,
    YIELD_MIN_TOP_PAD,
    YIELD_TOP_PAD_FRACTION,
    CountryMetadata,
    PlotConfig,
)

# Largest-first so the first matching scale wins.
_CURRENCY_SCALES = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))


def format_currency(value: float, prefix: str = "$") -> str:
    """Format a dollar amount compactly with a T/B/M/K suffix.

    Up to three significant figures are shown and trailing zeros dropped, so
    ``2e9 -> "$2B"``, ``1.5e12 -> "$1.5T"``, ``1.6e10 -> "$16B"``.
    """
    for scale, suffix in _CURRENCY_SCALES:
        if value >= scale:
            return f"{prefix}{value / scale:.3g}{suffix}"
    return f"{prefix}{value:.3g}"


def format_percent(value: float) -> str:
    """Format a percentage with up to three significant figures: ``0.25%``, ``1.5%``, ``120%``.

    From 1000 up, whole numbers with thousands separators (``.3g`` would
    switch to scientific notation there).
    """
    return f"{value:,.0f}%" if value >= 1000 else f"{value:.3g}%"


def format_yield(value: float) -> str:
    """Format a bond yield in percent with two decimals, as the sources publish them: ``4.12%``."""
    return f"{value:.2f}%"


class LogNiceLocator(ticker.Locator):
    """Log-axis tick positions chosen for even *visual* spacing at "nice" values.

    Uniform coefficient sets do not work on a log axis: the 1→2 gap is six
    times the 8→9 gap, so any set dense enough to fill the former crowds the
    latter. Instead ticks are chosen greedily per gap:

    1. every power of ten in view is a tick;
    2. integer coefficients are added in the order 2, 5, 3, 7, 4, 6, 8, 9,
       each only if it stays at least ``_MIN_SPACING`` label heights from the
       ticks already accepted;
    3. finer coefficients (steps of 0.5, then 0.2, 0.1, 0.05, 0.02, 0.01) are
       added only where they fill a void, again respecting the minimum
       spacing. A void is a stretch of axis more than ``_MAX_GAP / 2`` label
       heights from any label: between two labels that means a gap of
       ``_MAX_GAP``, but beyond the outermost label it means an end gap of
       only half that, since there the end of the axis is the far point.

    So a five-decade span gets 1/2/5 per decade; a two-decade span gains
    3, 4, 7 and 1.5; a span of half a decade gains 1.2, 1.5, 2.5 … and a
    very narrow span gets two-significant-figure values such as 1.4, 1.6.
    The end rule is what labels a percent axis above 100 % (-r): the next
    integer, 200 %, is usually out of view, and the 6-8 label heights left
    above 100 % used to stay bare until 150 % was allowed there.
    """

    _INTEGER_ORDER = (2, 5, 3, 7, 4, 6, 8, 9)
    _REFINEMENT_STEPS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    # Spacing thresholds in multiples of the tick-label height.
    _MIN_SPACING = 2.5
    _MAX_GAP = 8.0
    # A fractional (void-filling) tick keeps at least this far from either end of the axis.
    _EDGE_CLEARANCE = 1.0

    def __init__(self, label_fontsize: float) -> None:
        super().__init__()
        self.label_fontsize = label_fontsize

    def __call__(self) -> list[float]:
        return self.tick_values(*self.axis.get_view_interval())

    def _axis_length_px(self) -> float:
        bbox = self.axis.axes.get_window_extent()
        return bbox.height if self.axis.axis_name == "y" else bbox.width

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        if not (vmin > 0 and vmax > vmin):
            return []

        label_px = self.label_fontsize * self.axis.get_figure().dpi / 72.0
        px_per_decade = self._axis_length_px() / math.log10(vmax / vmin)
        min_spacing = self._MIN_SPACING * label_px
        max_gap = self._MAX_GAP * label_px
        edge_clearance = self._EDGE_CLEARANCE * label_px
        exponents = range(math.floor(math.log10(vmin)), math.ceil(math.log10(vmax)) + 1)

        def in_view(values: Iterable[float]) -> list[float]:
            return sorted(v for v in values if vmin <= v <= vmax)

        def px_between(a: float, b: float) -> float:
            return abs(math.log10(b) - math.log10(a)) * px_per_decade

        accepted = in_view(10.0**e for e in exponents)

        def try_add(value: float, *, require_wide_gap: bool) -> None:
            index = bisect.bisect_left(accepted, value)
            below = accepted[index - 1] if index > 0 else None
            above = accepted[index] if index < len(accepted) else None
            if below is not None and px_between(below, value) < min_spacing:
                return
            if above is not None and px_between(value, above) < min_spacing:
                return
            if require_wide_gap:
                # Fractional labels only fill voids. The gap they split runs to
                # the view edge when there is no accepted tick on that side;
                # such an end gap is a void at half the width (see docstring).
                gap = px_between(below if below is not None else vmin, above if above is not None else vmax)
                is_end_gap = below is None or above is None
                if gap < (max_gap / 2 if is_end_gap else max_gap):
                    return
                # Nor on the frame itself: a label there half overhangs the end
                # of the axis and marks nothing the frame does not.
                if min(px_between(vmin, value), px_between(value, vmax)) < edge_clearance:
                    return
            accepted.insert(index, value)

        for coefficient in self._INTEGER_ORDER:
            for value in in_view(coefficient * 10.0**e for e in exponents):
                try_add(value, require_wide_gap=False)

        for step in self._REFINEMENT_STEPS:
            # Coefficients are built from integers (hundredths) to avoid float drift.
            hundredths = range(100, 1000, round(step * 100))
            candidates = in_view(c * 10.0 ** (e - 2) for e in exponents for c in hundredths)
            for value in candidates:
                if not any(math.isclose(value, a, rel_tol=1e-9) for a in accepted):
                    try_add(value, require_wide_gap=True)

        return accepted


class _ComplementLogLocator(ticker.LogLocator):
    """Integer-coefficient log ticks that are not already major ticks (for the minor grid)."""

    def __init__(self, major: ticker.Locator) -> None:
        super().__init__(base=10, subs=tuple(range(2, 10)), numticks=20)
        self._major = major

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        majors = self._major.tick_values(vmin, vmax)
        return [
            v for v in super().tick_values(vmin, vmax)
            if not any(math.isclose(v, m, rel_tol=1e-9) for m in majors)
        ]


def configure_yield_axis(ax: Axes, *, font_scale: float, label: str = "Bond Yield (%)") -> None:
    """Configure a linear percentage axis used for bond yields.

    ``font_scale`` is ``PlotConfig.font_scale``, applied to the label, its pad
    and the tick labels.

    Tick steps are restricted to 1, 2, 5 × 10ⁿ. Every such step is either a
    whole number or an exact divisor of 1, so whenever the visible range
    contains a whole-number yield that value is a labelled tick. (Matplotlib's
    default step set also allows 1.5, 2.5, 3, 4, 6 and 8, which produce ticks
    like 0.6, 1.2, 1.8 … that skip the integers.)
    """
    ax.set_ylabel(label, fontsize=LABEL_FS * font_scale, labelpad=AXIS_LABEL_PAD_PT * font_scale)
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12, steps=[1, 2, 5, 10], prune=None))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="y", which="both", labelsize=TICK_FS * font_scale)


def configure_macro_axis(ax: Axes, metadata: CountryMetadata, config: PlotConfig) -> None:
    """Configure the log-scaled axis used for the macroeconomic series.

    It carries dollars, or with ``-r`` percentages of GDP; either way the
    values span decades, so the same log scale and tick locator serve both,
    and only the label and tick format differ. Fonts are scaled by
    ``config.font_scale``; the tick locator is given the *scaled* label
    size, because its spacing rules are counted in label heights.
    """
    font_scale = config.font_scale
    tick_fs = TICK_FS * font_scale
    ax.set_ylabel(
        metadata.macro_axis_label(config), fontsize=LABEL_FS * font_scale, labelpad=AXIS_LABEL_PAD_PT * font_scale
    )
    ax.set_yscale("log")
    # Labelled major ticks are chosen for even spacing (see LogNiceLocator);
    # the remaining integer multiples carry the unlabelled minor grid. The
    # minor formatter must be silenced explicitly: matplotlib's default labels
    # minor log ticks in scientific notation on short spans.
    major = LogNiceLocator(tick_fs)
    ax.yaxis.set_major_locator(major)
    ax.yaxis.set_minor_locator(_ComplementLogLocator(major))
    if config.relative:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _pos: format_percent(value)))
    else:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _pos: format_currency(value, metadata.currency_prefix))
        )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="y", which="both", labelsize=tick_fs)


def set_yield_ylim(ax: Axes, config: PlotConfig) -> None:
    """Apply explicit or data-driven yield limits with readable padding.

    Either bound may be pinned by the user; the other is taken from the data.
    """
    ax.relim()
    if not ax.has_data():
        ax.set_ylim(*DEFAULT_YIELD_YLIM)
        return

    data_ymin, data_ymax = ax.dataLim.intervaly
    low = config.yield_ymin if config.yield_ymin is not None else float(data_ymin)
    high = config.yield_ymax if config.yield_ymax is not None else float(data_ymax)
    if low >= high:
        low, high = DEFAULT_YIELD_YLIM

    # Padding is a fraction of the *final* displayed span (so the data occupies
    # exactly 1 - bottom - top of the axis), clamped to absolute minimums so a
    # flat series still gets breathing room.
    final_span = (high - low) / (1.0 - YIELD_BOTTOM_PAD_FRACTION - YIELD_TOP_PAD_FRACTION)
    pad_low = max(YIELD_BOTTOM_PAD_FRACTION * final_span, YIELD_MIN_BOTTOM_PAD)
    pad_high = max(YIELD_TOP_PAD_FRACTION * final_span, YIELD_MIN_TOP_PAD)
    ax.set_ylim(low - pad_low, high + pad_high)


def set_macro_ylim(ax: Axes, config: PlotConfig) -> None:
    """Apply explicit limits, or pad the data range logarithmically.

    Fallbacks are in the units of the axis's measure (``config.default_macro_ylim``).
    """
    default_low, default_high = config.default_macro_ylim
    if config.has_explicit_macro_limits:
        # A single pinned bound keeps the default for the other end rather than
        # mixing user and data-driven values on one axis.
        lower = config.macro_bottom if config.macro_bottom is not None else default_low
        upper = config.macro_top if config.macro_top is not None else default_high
        ax.set_ylim(lower, upper)
        return

    ax.relim()
    if not ax.has_data():
        ax.set_ylim(default_low, default_high)
        return

    data_ymin, data_ymax = ax.dataLim.intervaly
    if data_ymin <= 0 or data_ymax <= 0 or data_ymin >= data_ymax:
        # Non-positive values cannot be shown on a log axis.
        ax.set_ylim(default_low, default_high)
        return

    log_low, log_high = np.log10(data_ymin), np.log10(data_ymax)
    pad = max((log_high - log_low) * MACRO_PAD_FRACTION, MACRO_MIN_PAD_DECADES)
    ax.set_ylim(10 ** (log_low - pad), 10 ** (log_high + pad))


# Tick spacing (days) from which ticks are monthly or longer, not daily.
_SHORTEST_MONTH_DAYS = 28


class _DateLocator(mdates.AutoDateLocator):
    """AutoDateLocator that takes its own fallback without warning about it.

    AutoDateLocator uses the coarsest frequency (years, months, days, hours …)
    that gives at least ``minticks`` ticks, then the first interval in that
    frequency's list that gives at most ``maxticks``. A span just short of
    ``minticks`` years drops to months, where even the longest interval (6
    months) gives one or two ticks more than ``maxticks``. Matplotlib then
    warns "unable to pick an appropriate interval" and uses 6 months anyway.
    With the chart's limits that happens for spans of about 5.6-6 years (major
    ticks) and 11.6-12 years (minor ticks).

    That fallback is the right answer: it continues the 6-monthly ticks of
    slightly shorter spans. The alternative that silences the warning by
    configuration (adding a 12-month interval) was tried and rejected: for the
    minor ticks it lands every 1 January, on top of the major ticks, and the
    chart loses its minor ticks and grid altogether. So the fallback is kept,
    and only that one warning is suppressed. Tick positions are exactly
    matplotlib's for every span (checked from 7 days to 120 years).

    With ``thin_month_ends`` (the labelled major ticks) one more thing is
    changed. Daily ticks every 2 or 4 days are laid on days of the month
    (1, 5, 9 … 29) and start again on the 1st, so the last tick of a month
    can fall only 1-3 days before the next month's 1st, and on windows of a
    few weeks their labels ran together ("2026-08-292026-09-01", Terry,
    2026-09-28). A tick closer than the regular spacing to a following
    month start is dropped; the month start is kept. Only daily ticks: the
    gaps between monthly or yearly ticks vary with the months' lengths, and
    a first version that ignored this dropped every tick after a short
    month. Checked over 2,580 windows from 7 days to 120 years: only such
    ticks go (windows of 10 to 42 days), nothing is added, no short gap is
    left, and the minor ticks are untouched.
    """

    def __init__(self, *, minticks: int, maxticks: int, thin_month_ends: bool = False) -> None:
        super().__init__(minticks=minticks, maxticks=maxticks)
        self.thin_month_ends = thin_month_ends

    def get_locator(self, dmin, dmax):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="AutoDateLocator was unable to pick an appropriate interval")
            return super().get_locator(dmin, dmax)

    def __call__(self):
        return self._thinned(super().__call__())

    def tick_values(self, vmin, vmax):
        return self._thinned(super().tick_values(vmin, vmax))

    def _thinned(self, ticks):
        """Drop a tick closer than the regular spacing to a following month start (see the class docstring)."""
        values = np.asarray(ticks, dtype=float)
        if not self.thin_month_ends or len(values) < 3:
            return ticks
        gaps = np.diff(values)
        regular = float(np.median(gaps))
        if regular >= _SHORTEST_MONTH_DAYS:
            # Monthly and yearly ticks: their gaps vary with the months' lengths, which is not crowding.
            return ticks
        keep = np.ones(len(values), dtype=bool)
        for index, gap in enumerate(gaps):
            if gap < regular - 1e-9 and mdates.num2date(values[index + 1]).day == 1:
                keep[index] = False
        return values[keep]


def _date_locator(*, minticks: int, maxticks: int, thin_month_ends: bool = False) -> mdates.AutoDateLocator:
    """Return the chart's date locator (see ``_DateLocator``)."""
    return _DateLocator(minticks=minticks, maxticks=maxticks, thin_month_ends=thin_month_ends)


def scale_line_widths(ax_yield: Axes, ax_macro: Axes, line_scale: float) -> None:
    """Widen the plotted lines, axes frame and tick marks by ``PlotConfig.line_scale``.

    The widths in ``config`` (and matplotlib's defaults for frame and ticks)
    are designed for the default canvas. Must run after every data line is
    plotted and before the legend is made, so legend samples copy the scaled
    widths. Dash patterns follow the width automatically (``lines.scale_dashes``).
    Grid lines are scaled where the grid is switched on, in ``apply_axes_formatting``.
    """
    for ax in (ax_yield, ax_macro):
        for line in ax.get_lines():
            line.set_linewidth(line.get_linewidth() * line_scale)
        for spine in ax.spines.values():
            spine.set_linewidth(spine.get_linewidth() * line_scale)
        for axis_name in ("x", "y"):
            for which in ("major", "minor"):
                ax.tick_params(
                    axis=axis_name,
                    which=which,
                    width=mpl.rcParams[f"{axis_name}tick.{which}.width"] * line_scale,
                    length=mpl.rcParams[f"{axis_name}tick.{which}.size"] * line_scale,
                )


def apply_axes_formatting(
    ax_yield: Axes, ax_macro: Axes, config: PlotConfig, metadata: CountryMetadata
) -> None:
    """Configure the shared x-axis and both y-axes after all lines are plotted.

    ``ax_macro`` is a ``twinx()`` of ``ax_yield``. When only one family of
    curves is selected the unused axis mirrors the other so the chart never
    shows a second, meaningless scale.
    """
    scale = config.font_scale
    scale_line_widths(ax_yield, ax_macro, config.line_scale)
    ax_yield.set_xlabel("Date", fontsize=LABEL_FS * scale)
    ax_yield.tick_params(axis="both", which="major", labelsize=TICK_FS * scale)
    ax_yield.tick_params(axis="both", which="minor", labelsize=(TICK_FS - 4) * scale)
    ax_yield.xaxis.set_major_locator(_date_locator(minticks=6, maxticks=12, thin_month_ends=True))
    ax_yield.xaxis.set_minor_locator(_date_locator(minticks=12, maxticks=24))
    grid_width = mpl.rcParams["grid.linewidth"] * config.line_scale
    ax_yield.grid(True, which="major", linestyle="--", alpha=0.40, linewidth=grid_width)
    ax_yield.grid(True, which="minor", linestyle=":", alpha=0.22, linewidth=grid_width)

    if config.include_yield and config.has_macro_series:
        configure_yield_axis(ax_yield, font_scale=scale)
        set_yield_ylim(ax_yield, config)
        configure_macro_axis(ax_macro, metadata, config)
        set_macro_ylim(ax_macro, config)
    elif config.include_yield:
        configure_yield_axis(ax_yield, font_scale=scale)
        set_yield_ylim(ax_yield, config)
        configure_yield_axis(ax_macro, font_scale=scale)
        ax_macro.set_ylim(ax_yield.get_ylim())
    elif config.has_macro_series:
        configure_macro_axis(ax_macro, metadata, config)
        set_macro_ylim(ax_macro, config)
        configure_macro_axis(ax_yield, metadata, config)
        ax_yield.set_ylim(ax_macro.get_ylim())
    else:
        # Nothing selected on either y-axis (e.g. ``--no-yield --no-debt
        # --no-gdp --no-interest``): keep the figure usable but hide the twin.
        ax_macro.set_visible(False)


def apply_date_xlim(ax: Axes, config: PlotConfig) -> None:
    """Pad the requested date range slightly so lines do not touch the plot border."""
    span_days = max((config.end - config.start).days, MIN_DATE_PAD_DAYS)
    pad = pd.Timedelta(days=max(span_days * DATE_PAD_FRACTION, MIN_DATE_PAD_DAYS))
    ax.set_xlim(config.start - pad, config.end + pad)
