"""Matplotlib axis configuration: scales, tick locators, formatters and limits."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .config import (
    DATE_PAD_FRACTION,
    DEFAULT_MACRO_YLIM,
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


def format_currency(value: float, prefix: str = "$", *, decimals: int = 0) -> str:
    """Format a dollar amount compactly with a T/B/M/K suffix, e.g. ``2.5e9 -> "$2B"``.

    The default of zero decimals suits log-axis tick labels, which sit at
    1/2/5 × 10^n and so are always whole numbers of the chosen unit.
    """
    for scale, suffix in _CURRENCY_SCALES:
        if value >= scale:
            return f"{prefix}{value / scale:.{decimals}f}{suffix}"
    return f"{prefix}{value:.{decimals}f}"


def configure_yield_axis(ax: Axes, *, label: str = "Bond Yield (%)") -> None:
    """Configure a linear percentage axis used for bond yields."""
    ax.set_ylabel(label, fontsize=LABEL_FS, labelpad=8)
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12, prune=None))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="y", which="both", labelsize=TICK_FS)


def configure_macro_axis(ax: Axes, metadata: CountryMetadata) -> None:
    """Configure the log-scaled dollar axis used for the macroeconomic series."""
    ax.set_ylabel(metadata.currency_label, fontsize=LABEL_FS, labelpad=8)
    ax.set_yscale("log")
    # Major ticks at 1, 2, 5 per decade; minor ticks fill in the rest.
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=20)
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _pos: format_currency(value, metadata.currency_prefix))
    )
    ax.tick_params(axis="y", which="both", labelsize=TICK_FS)


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
    """Apply explicit limits, or pad the data range logarithmically."""
    if config.has_explicit_macro_limits:
        # A single pinned bound keeps the default for the other end rather than
        # mixing user and data-driven values on one axis.
        lower = config.macro_bottom if config.macro_bottom is not None else DEFAULT_MACRO_YLIM[0]
        upper = config.macro_top if config.macro_top is not None else DEFAULT_MACRO_YLIM[1]
        ax.set_ylim(lower, upper)
        return

    ax.relim()
    if not ax.has_data():
        ax.set_ylim(*DEFAULT_MACRO_YLIM)
        return

    data_ymin, data_ymax = ax.dataLim.intervaly
    if data_ymin <= 0 or data_ymax <= 0 or data_ymin >= data_ymax:
        # Non-positive values cannot be shown on a log axis.
        ax.set_ylim(*DEFAULT_MACRO_YLIM)
        return

    log_low, log_high = np.log10(data_ymin), np.log10(data_ymax)
    pad = max((log_high - log_low) * MACRO_PAD_FRACTION, MACRO_MIN_PAD_DECADES)
    ax.set_ylim(10 ** (log_low - pad), 10 ** (log_high + pad))


def apply_axes_formatting(
    ax_yield: Axes, ax_macro: Axes, config: PlotConfig, metadata: CountryMetadata
) -> None:
    """Configure the shared x-axis and both y-axes after all lines are plotted.

    ``ax_macro`` is a ``twinx()`` of ``ax_yield``. When only one family of
    curves is selected the unused axis mirrors the other so the chart never
    shows a second, meaningless scale.
    """
    ax_yield.set_xlabel("Date", fontsize=LABEL_FS)
    ax_yield.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax_yield.tick_params(axis="both", which="minor", labelsize=TICK_FS - 4)
    ax_yield.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax_yield.xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=12, maxticks=24))
    ax_yield.grid(True, which="major", linestyle="--", alpha=0.40)
    ax_yield.grid(True, which="minor", linestyle=":", alpha=0.22)

    if config.include_yield and config.has_dollar_series:
        configure_yield_axis(ax_yield)
        set_yield_ylim(ax_yield, config)
        configure_macro_axis(ax_macro, metadata)
        set_macro_ylim(ax_macro, config)
    elif config.include_yield:
        configure_yield_axis(ax_yield)
        set_yield_ylim(ax_yield, config)
        configure_yield_axis(ax_macro)
        ax_macro.set_ylim(ax_yield.get_ylim())
    elif config.has_dollar_series:
        configure_macro_axis(ax_macro, metadata)
        set_macro_ylim(ax_macro, config)
        configure_macro_axis(ax_yield, metadata)
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
