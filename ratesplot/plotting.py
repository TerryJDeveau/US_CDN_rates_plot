"""Chart construction: line drawing, per-country plot assembly, and run drivers.

Both country charts share one layout: yield curves on the left (linear %)
axis, the three macro curves on a right (log $) twin axis, an auto-placed
legend and a title/date-range subtitle. Only the yield-line style differs:
the Canadian pre-2001 history is monthly and drawn as steps.
"""

from __future__ import annotations

from typing import Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .axes import apply_axes_formatting, format_currency
from .cdn_data import (
    CANADIAN_SERIES_EARLIEST,
    align_cdn_macro,
    fetch_cdn_debt,
    fetch_cdn_gdp,
    fetch_cdn_interest,
    fetch_cdn_yields,
)
from .config import (
    CANADIAN_YIELD_HIST_END,
    CDN,
    DATE_COLUMN,
    MACRO_PLOT_STYLES,
    US,
    YIELD_COLUMNS,
    YIELD_LINE_STYLE,
    CountryMetadata,
    PlotConfig,
)
from .frames import filter_to_date_range
from .legend import finish_legend_and_title
from .us_data import US_SERIES_EARLIEST, fetch_us_macro, fetch_us_yields

YieldLineDrawer = Callable[[Axes, pd.DataFrame, PlotConfig], list[Line2D]]


# ---------------------------------------------------------------------------
# Console warnings about coverage
# ---------------------------------------------------------------------------


def warn_series_coverage(start_date: pd.Timestamp, series_earliest: dict[str, pd.Timestamp]) -> None:
    """Warn when the requested start precedes a source's first observation."""
    late_series = [(name, earliest) for name, earliest in series_earliest.items() if start_date < earliest]
    if not late_series:
        return

    print(f"  Warning: requested start {start_date.date()} is earlier than the first available observation for:")
    for name, earliest in late_series:
        print(f"    • {name} begins {earliest.date()}")


def warn_dollar_limits_coverage(
    df_macro: pd.DataFrame, macro_columns: Iterable[str], config: PlotConfig
) -> None:
    """Warn when explicit ``--top``/``--bottom`` limits hide all or part of a curve."""
    if df_macro.empty or not config.has_explicit_macro_limits:
        return

    def limit_text(value: float | None) -> str:
        return "None" if value is None else format_currency(value, "", decimals=1)

    limits_text = f"--bottom:{limit_text(config.macro_bottom)}, --top:{limit_text(config.macro_top)}"
    bottom = config.macro_bottom if config.macro_bottom is not None else float("-inf")
    top = config.macro_top if config.macro_top is not None else float("inf")

    for column in macro_columns:
        series = df_macro[column].dropna() if column in df_macro.columns else pd.Series(dtype=float)
        if series.empty:
            continue
        low, high = series.min(), series.max()
        if low > top or high < bottom:
            print(f"  Warning: Dollar curve '{column}' cannot be shown at all within limits {limits_text}.")
        elif low < bottom or high > top:
            print(f"  Warning: Dollar curve '{column}' is only shown for part of its range due to limits {limits_text}.")


# ---------------------------------------------------------------------------
# Line drawing
# ---------------------------------------------------------------------------


def add_macro_line(
    ax: Axes, data: pd.DataFrame, *, enabled: bool, column: str, label: str, style: dict
) -> Line2D | None:
    """Plot one macro series if it is enabled and has data; return the line or ``None``."""
    if not enabled or data.empty or column not in data.columns or not data[column].notna().any():
        return None
    (line,) = ax.plot(data[DATE_COLUMN], data[column], label=label, **style)
    return line


def add_canadian_yield_lines(ax: Axes, yields: pd.DataFrame, config: PlotConfig) -> list[Line2D]:
    """Plot Canadian yield curves: monthly history as steps, daily live data as lines.

    ``yields`` is date-indexed. Each tenor is split at ``CANADIAN_YIELD_HIST_END``;
    the live segment reuses the history segment's colour and is hidden from the
    legend so each tenor appears once.
    """
    if not config.include_yield or yields.empty:
        return []

    data = filter_to_date_range(yields.reset_index(), DATE_COLUMN, config).set_index(DATE_COLUMN)
    if data.empty:
        return []

    legend_lines: list[Line2D] = []
    for column in YIELD_COLUMNS:
        if column not in data.columns:
            continue
        series = data[column].dropna()
        historical = series.loc[:CANADIAN_YIELD_HIST_END]
        current = series.loc[series.index > CANADIAN_YIELD_HIST_END]
        label = f"{column} Yield"
        color = None

        if not historical.empty:
            (line,) = ax.plot(historical.index, historical.values, label=label, drawstyle="steps-post", **YIELD_LINE_STYLE)
            color = line.get_color()
            legend_lines.append(line)

        if not current.empty:
            (line,) = ax.plot(
                current.index,
                current.values,
                label="_nolegend_" if color is not None else label,
                color=color,
                **YIELD_LINE_STYLE,
            )
            if color is None:
                legend_lines.append(line)

    return legend_lines


def add_us_yield_lines(ax: Axes, yields: pd.DataFrame, config: PlotConfig) -> list[Line2D]:
    """Plot U.S. yield curves as ordinary daily time-series lines.

    ``yields`` has a ``DATE`` column (FRED frames are not date-indexed).
    """
    if not config.include_yield or yields.empty:
        return []

    data = filter_to_date_range(yields, DATE_COLUMN, config)
    lines: list[Line2D] = []
    for column in YIELD_COLUMNS:
        if column in data.columns:
            (line,) = ax.plot(data[DATE_COLUMN], data[column], label=f"{column} Yield", **YIELD_LINE_STYLE)
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Chart assembly
# ---------------------------------------------------------------------------


def plot_country(
    yields: pd.DataFrame,
    macro: pd.DataFrame,
    config: PlotConfig,
    metadata: CountryMetadata,
    draw_yield_lines: YieldLineDrawer,
) -> None:
    """Build and display one country's chart from already-prepared data."""
    _figure, ax_yield = plt.subplots(figsize=config.figsize_inches)
    yield_lines = draw_yield_lines(ax_yield, yields, config)

    ax_macro = ax_yield.twinx()
    macro_in_range = filter_to_date_range(macro, DATE_COLUMN, config)
    macro_lines: list[Line2D] = []
    macro_columns_plotted: list[str] = []
    for style_key, enabled, column, label in metadata.macro_specs(config):
        line = add_macro_line(
            ax_macro, macro_in_range, enabled=enabled, column=column, label=label, style=MACRO_PLOT_STYLES[style_key]
        )
        if line is not None:
            macro_lines.append(line)
            macro_columns_plotted.append(column)

    apply_axes_formatting(ax_yield, ax_macro, config, metadata)
    warn_dollar_limits_coverage(macro_in_range, macro_columns_plotted, config)
    finish_legend_and_title(ax_yield, yield_lines + macro_lines, metadata.title, config)
    plt.show()


def run_cdn(config: PlotConfig) -> None:
    """Fetch all selected Canadian inputs and render the Canadian chart."""
    warn_series_coverage(config.start, CANADIAN_SERIES_EARLIEST)
    yields = fetch_cdn_yields(config)
    macro = align_cdn_macro(
        yields, fetch_cdn_debt(config), fetch_cdn_gdp(config), fetch_cdn_interest(config), config
    )
    plot_country(yields, macro, config, CDN, add_canadian_yield_lines)
    print("CDN chart finished.\n")


def run_us(config: PlotConfig) -> None:
    """Fetch all selected U.S. inputs and render the U.S. chart."""
    warn_series_coverage(config.start, US_SERIES_EARLIEST)
    yields = fetch_us_yields(config)
    if not yields.empty:
        yields = yields.loc[yields[DATE_COLUMN] <= config.end]
    last_yield_date = yields[DATE_COLUMN].max() if not yields.empty else config.end

    macro = fetch_us_macro(config, last_yield_date)
    if not macro.empty:
        macro = macro.loc[macro[DATE_COLUMN] <= config.end]

    plot_country(yields, macro, config, US, add_us_yield_lines)
    print("US chart finished.\n")
