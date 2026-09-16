"""Matplotlib y-axis configuration and limit helpers."""

from __future__ import annotations

from typing import Callable

import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

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
    PlotConfig,
)


def log_currency_formatter(prefix: str = "$") -> Callable[[float, int], str]:
    """Create a log-axis formatter using T/B/M/K suffixes."""
    scale_labels = (
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    )

    def format_value(value: float, _position: int) -> str:
        for scale, suffix in scale_labels:
            if value >= scale:
                return f"{prefix}{value / scale:.0f}{suffix}"
        return f"{prefix}{value:.0f}"

    return format_value


def configure_yield_axis(ax, *, label: str = "Bond Yield (%)") -> None:
    """Configure a linear percentage axis used for bond yields."""
    ax.set_ylabel(label, fontsize=LABEL_FS, labelpad=8)
    ax.set_yscale("linear")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=12, prune=None))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="y", which="both", labelsize=TICK_FS)


def configure_macro_axis(ax, *, label: str, currency_prefix: str) -> None:
    """Configure the log-scaled dollar axis used for macroeconomic series."""
    ax.set_ylabel(label, fontsize=LABEL_FS, labelpad=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(
        ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=20)
    )
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=20)
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_currency_formatter(currency_prefix)))
    ax.tick_params(axis="y", which="both", labelsize=TICK_FS)


def set_yield_ylim(ax, config: PlotConfig) -> None:
    """Apply explicit or data-driven yield limits with readable padding."""
    if not config.include_yield:
        return

    ax.relim()
    if not ax.has_data():
        ax.set_ylim(*DEFAULT_YIELD_YLIM)
        return

    data_ymin, data_ymax = ax.dataLim.intervaly
    data_low = config.yield_ymin if config.yield_ymin is not None else float(data_ymin)
    data_high = config.yield_ymax if config.yield_ymax is not None else float(data_ymax)

    if data_low >= data_high:
        data_low, data_high = DEFAULT_YIELD_YLIM

    data_span = data_high - data_low
    if data_span <= 0:
        ax.set_ylim(*DEFAULT_YIELD_YLIM)
        return

    # Padding is specified as a fraction of the final displayed span, then
    # clamped to minimum absolute margins so flat series remain legible.
    final_span = data_span / (1.0 - YIELD_BOTTOM_PAD_FRACTION - YIELD_TOP_PAD_FRACTION)
    pad_low = max(YIELD_BOTTOM_PAD_FRACTION * final_span, YIELD_MIN_BOTTOM_PAD)
    pad_high = max(YIELD_TOP_PAD_FRACTION * final_span, YIELD_MIN_TOP_PAD)
    ax.set_ylim(data_low - pad_low, data_high + pad_high)


def set_macro_ylim(ax, config: PlotConfig) -> None:
    """Apply explicit or logarithmically padded limits to the macro axis."""
    if config.macro_bottom is not None or config.macro_top is not None:
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
        ax.set_ylim(*DEFAULT_MACRO_YLIM)
        return

    log_low = np.log10(data_ymin)
    log_high = np.log10(data_ymax)
    log_span = log_high - log_low
    pad = max(log_span * MACRO_PAD_FRACTION, MACRO_MIN_PAD_DECADES)
    ax.set_ylim(10 ** (log_low - pad), 10 ** (log_high + pad))


def apply_axes_formatting(ax1, ax2, config: PlotConfig, *, currency_prefix: str, currency_label: str) -> None:
    """Configure both y-axes after all selected series have been plotted."""
    ax1.set_xlabel("Date", fontsize=LABEL_FS)
    ax1.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax1.tick_params(axis="both", which="minor", labelsize=TICK_FS - 4)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax1.xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=12, maxticks=24))
    ax1.grid(True, which="major", linestyle="--", alpha=0.40)
    ax1.grid(True, which="minor", linestyle=":", alpha=0.22)

    has_dollar_series = config.include_debt or config.include_gdp or config.include_interest

    if not config.include_yield and not has_dollar_series:
        # Nothing is plotted on either y-axis, so hide the secondary axis and leave
        # the figure usable for an intentionally empty selection.
        ax2.set_visible(False)
        return

    if config.include_yield and has_dollar_series:
        configure_yield_axis(ax1)
        set_yield_ylim(ax1, config)
        configure_macro_axis(ax2, label=currency_label, currency_prefix=currency_prefix)
        set_macro_ylim(ax2, config)
    elif config.include_yield:
        # Matplotlib creates ax2 via twinx(); mirror the yield scale when the
        # macro curves are disabled so there is no misleading second scale.
        configure_yield_axis(ax1)
        set_yield_ylim(ax1, config)
        configure_yield_axis(ax2)
        ax2.set_ylim(ax1.get_ylim())
    else:
        # With yields disabled, use the macro scale as the primary visual axis.
        configure_macro_axis(ax2, label=currency_label, currency_prefix=currency_prefix)
        set_macro_ylim(ax2, config)
        configure_macro_axis(ax1, label=currency_label, currency_prefix=currency_prefix)
        ax1.set_ylim(ax2.get_ylim())


def apply_date_xlim(ax, config: PlotConfig) -> None:
    """Pad the requested date range slightly so lines do not touch plot borders."""
    span_days = max((config.end - config.start).days, MIN_DATE_PAD_DAYS)
    pad_days = max(span_days * DATE_PAD_FRACTION, MIN_DATE_PAD_DAYS)
    ax.set_xlim(
        config.start - pd.Timedelta(days=pad_days),
        config.end + pd.Timedelta(days=pad_days),
    )
