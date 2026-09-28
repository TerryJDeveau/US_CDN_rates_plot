"""Chart construction: line drawing, per-country plot assembly, and run drivers.

Both country charts share one layout: yield curves on the left (linear %)
axis, the three macro curves on a right log twin axis (dollars; with -r debt
and interest as % of GDP; with -p dollars per person; see ``measures``; with
--debt:LETTERS one debt and one interest line per level of government), an auto-placed
legend, a title and a subtitle naming the dates the drawn data cover (which
can be narrower than the axis), and with -l each line's last value at its
end. Only the yield-line style differs: the Canadian pre-2001 history is
monthly and drawn as steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .axes import apply_axes_formatting, format_currency, format_percent, format_yield
from .cdn_data import (
    CANADIAN_SERIES_EARLIEST,
    align_cdn_macro,
    fetch_cdn_components,
    fetch_cdn_debt,
    fetch_cdn_gdp,
    fetch_cdn_interest,
    fetch_cdn_population,
    fetch_cdn_yields,
)
from .config import (
    CANADIAN_YIELD_HIST_END,
    CANVAS_DPI,
    CDN,
    DATE_COLUMN,
    MACRO_PLOT_STYLES,
    US,
    YIELD_COLUMNS,
    YIELD_LINE_STYLE,
    CountryMetadata,
    PlotConfig,
)
from .endlabels import ValueFormatter
from .frames import filter_to_date_range
from .latest import QUOTE_TIME_ATTR, extend_cdn_yields, extend_us_yields
from .legend import finish_legend_and_title
from .measures import express
from .us_data import US_SERIES_EARLIEST, fetch_us_macro, fetch_us_population, fetch_us_yields

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


def warn_macro_limits_coverage(
    df_macro: pd.DataFrame, macro_columns: Iterable[str], config: PlotConfig
) -> None:
    """Warn when explicit ``--top``/``--bottom`` limits hide all or part of a right-axis curve."""
    if df_macro.empty or not config.has_explicit_macro_limits:
        return

    def limit_text(value: float | None) -> str:
        if value is None:
            return "None"
        return format_percent(value) if config.relative else format_currency(value, "")

    limits_text = f"--bottom:{limit_text(config.macro_bottom)}, --top:{limit_text(config.macro_top)}"
    bottom = config.macro_bottom if config.macro_bottom is not None else float("-inf")
    top = config.macro_top if config.macro_top is not None else float("inf")

    for column in macro_columns:
        series = df_macro[column].dropna() if column in df_macro.columns else pd.Series(dtype=float)
        if series.empty:
            continue
        low, high = series.min(), series.max()
        if low > top or high < bottom:
            print(f"  Warning: Right-axis curve '{column}' cannot be shown at all within limits {limits_text}.")
        elif low < bottom or high > top:
            print(f"  Warning: Right-axis curve '{column}' is only shown for part of its range due to limits {limits_text}.")


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


def drawn_date_span(axes: Iterable[Axes]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the first and last dates at which any line on ``axes`` has a value, or None if none does.

    This is the span the subtitle names. It can be narrower than the axis
    (``--start``/``--end``), which is kept as requested: e.g. ``-s:1900`` on
    a series that begins in 1919 shows 1900 onwards but reports 1919.
    """
    first, last = math.inf, -math.inf
    for ax in axes:
        for line in ax.get_lines():
            xy = np.asarray(line.get_xydata(), dtype=float)  # dates already converted to numbers
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy):
                first, last = min(first, xy[:, 0].min()), max(last, xy[:, 0].max())
    if first > last:
        return None
    return tuple(pd.Timestamp(mdates.num2date(value).replace(tzinfo=None)).normalize() for value in (first, last))


# ---------------------------------------------------------------------------
# Chart assembly
# ---------------------------------------------------------------------------


def draw_country(
    ax_yield: Axes,
    yields: pd.DataFrame,
    macro: pd.DataFrame,
    config: PlotConfig,
    metadata: CountryMetadata,
    draw_yield_lines: YieldLineDrawer,
) -> None:
    """Draw one country's complete chart (lines, axes, title, legend) onto ``ax_yield``'s figure.

    Works on any figure: a pyplot one (``plot_country``) or a bare Agg one
    (``build_figure``), so the window and the GUI draw identical charts.
    """
    quote_time = yields.attrs.get(QUOTE_TIME_ATTR)  # --cur: when the day's quotes were taken, if any
    yield_lines = draw_yield_lines(ax_yield, yields, config)

    ax_macro = ax_yield.twinx()
    macro_in_range = filter_to_date_range(macro, DATE_COLUMN, config)
    macro_lines: list[Line2D] = []
    macro_keys_drawn: list[str] = []
    macro_columns_drawn: list[str] = []
    for style_key, enabled, column, label in metadata.macro_specs(config):
        line = add_macro_line(
            ax_macro, macro_in_range, enabled=enabled, column=column, label=label, style=MACRO_PLOT_STYLES[style_key]
        )
        if line is not None:
            macro_lines.append(line)
            macro_keys_drawn.append(style_key)
            macro_columns_drawn.append(column)

    apply_axes_formatting(ax_yield, ax_macro, config, metadata)
    warn_macro_limits_coverage(macro_in_range, macro_columns_drawn, config)
    # -l: every drawn curve gets its last value at its end, written as its
    # axis writes values: yields in percent, right-axis curves as the tick
    # labels there (dollars, or percent of GDP under -r).
    end_labels: list[tuple[Line2D, ValueFormatter]] = []
    if config.end_labels:
        macro_format = format_percent if config.relative else partial(format_currency, prefix=metadata.currency_prefix)
        end_labels = [(line, format_yield) for line in yield_lines] + [(line, macro_format) for line in macro_lines]
    # The title names only what is on the chart, and the subtitle only the
    # dates it covers; the legend keeps yields and macro curves as separate
    # column groups.
    title = metadata.title_for(yields_drawn=bool(yield_lines), macro_keys_drawn=macro_keys_drawn, config=config)
    finish_legend_and_title(
        ax_yield,
        [yield_lines, macro_lines],
        title,
        config,
        drawn_date_span((ax_yield, ax_macro)),
        end_labels,
        quote_time if yield_lines else None,
    )


def plot_country(
    yields: pd.DataFrame,
    macro: pd.DataFrame,
    config: PlotConfig,
    metadata: CountryMetadata,
    draw_yield_lines: YieldLineDrawer,
) -> None:
    """Build one country's chart on a pyplot figure and show it in a matplotlib window."""
    _figure, ax_yield = plt.subplots(figsize=config.figsize_inches)
    draw_country(ax_yield, yields, macro, config, metadata, draw_yield_lines)
    plt.show()


# ---------------------------------------------------------------------------
# Data preparation and run drivers
# ---------------------------------------------------------------------------


def prepare_cdn(config: PlotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and align all selected Canadian inputs, in the configured measure; return ``(yields, macro)``.

    With --cur the yields run on to the day's quotes, and the macro curves
    with them; the quotes' time is in ``yields.attrs[QUOTE_TIME_ATTR]``.
    """
    warn_series_coverage(config.start, CANADIAN_SERIES_EARLIEST)
    yields, quote_time = extend_cdn_yields(fetch_cdn_yields(config), config)
    if config.components:
        # Debt and interest by level of government replace the aggregate lines.
        series = (fetch_cdn_components(config), fetch_cdn_gdp(config))
    else:
        series = (fetch_cdn_debt(config), fetch_cdn_gdp(config), fetch_cdn_interest(config))
    macro = align_cdn_macro(yields, series, config)
    population = fetch_cdn_population() if config.per_capita else None
    yields.attrs[QUOTE_TIME_ATTR] = quote_time
    return yields, express(macro, config, CDN, population)


def prepare_us(config: PlotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all selected U.S. inputs, trimmed to the end date, in the configured measure; return ``(yields, macro)``.

    With --cur the yields run on to the day's quotes and federal debt daily
    (see ``ratesplot.latest``); the quotes' time is in ``yields.attrs[QUOTE_TIME_ATTR]``.
    """
    warn_series_coverage(config.start, US_SERIES_EARLIEST)
    yields, quote_time = extend_us_yields(fetch_us_yields(config), config)
    if not yields.empty:
        yields = yields.loc[yields[DATE_COLUMN] <= config.end]
    last_yield_date = yields[DATE_COLUMN].max() if not yields.empty else config.end

    macro = fetch_us_macro(config, last_yield_date)
    if not macro.empty:
        macro = macro.loc[macro[DATE_COLUMN] <= config.end]
    population = fetch_us_population() if config.per_capita else None
    yields.attrs[QUOTE_TIME_ATTR] = quote_time
    return yields, express(macro, config, US, population)


@dataclass(frozen=True)
class Country:
    """Everything needed to produce one country's chart."""

    key: str  # short name, as used for its tab and output file
    show_field: str  # PlotConfig field that selects it
    metadata: CountryMetadata
    prepare: Callable[[PlotConfig], tuple[pd.DataFrame, pd.DataFrame]]
    draw_yield_lines: YieldLineDrawer


# In drawing order (Canadian, then U.S.), as the command line always has.
COUNTRIES: tuple[Country, ...] = (
    Country("cdn", "show_cdn", CDN, prepare_cdn, add_canadian_yield_lines),
    Country("us", "show_us", US, prepare_us, add_us_yield_lines),
)


def build_figure(country: Country, config: PlotConfig) -> Figure:
    """Fetch, prepare and draw one country's chart on a new Agg figure, without pyplot.

    pyplot keeps global state and drives GUI windows, so it must not be used
    from a worker thread; a bare ``Figure`` with an Agg canvas is safe there
    as long as one thread uses it at a time. The DPI is pinned to
    ``CANVAS_DPI`` so the pixel size is exactly ``width_px`` x ``height_px``.
    """
    figure = Figure(figsize=config.figsize_inches, dpi=CANVAS_DPI)
    FigureCanvasAgg(figure)
    ax_yield = figure.subplots()
    yields, macro = country.prepare(config)
    draw_country(ax_yield, yields, macro, config, country.metadata, country.draw_yield_lines)
    return figure


def run_cdn(config: PlotConfig) -> None:
    """Fetch all selected Canadian inputs and show the Canadian chart in a matplotlib window."""
    yields, macro = prepare_cdn(config)
    plot_country(yields, macro, config, CDN, add_canadian_yield_lines)
    print("CDN chart finished.\n")


def run_us(config: PlotConfig) -> None:
    """Fetch all selected U.S. inputs and show the U.S. chart in a matplotlib window."""
    yields, macro = prepare_us(config)
    plot_country(yields, macro, config, US, add_us_yield_lines)
    print("US chart finished.\n")
