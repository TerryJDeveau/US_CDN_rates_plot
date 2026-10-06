"""Chart construction: line drawing, per-country plot assembly, and run drivers.

Both country charts share one layout: yield curves on the left (linear %)
axis, the three macro curves on a right log twin axis (dollars; with -r debt
and interest as % of GDP; with -p dollars per person; see ``measures``; with
--debt:LETTERS one debt and one interest line per level of government), an auto-placed
legend, a title and a subtitle naming the dates the drawn data cover (which
can be narrower than the axis), with -l each line's last value at its
end, and with --reg straight pieces fitted to the right-axis curves, each
labelled with its growth in %/yr. Only the yield-line style differs: the Canadian pre-2001 history is
monthly and drawn as steps.

Without --start the charts begin on ``config.DEFAULT_START_FLOOR``, or
on the first date on which every chosen curve has data if that is later
(``resolve_start``), the same date for both countries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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

from .axes import apply_axes_formatting, format_currency, format_percent, format_spread, format_yield
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
    DEFAULT_START_FLOOR,
    EARLIEST_DATA_START,
    MACRO_PLOT_STYLES,
    MIN_WINDOW_DAYS,
    PROJECTION_ALPHA,
    PROJECTION_KEY_COLOR,
    PROJECTION_LABEL,
    PROJECTION_LABEL_PREFIX,
    PROJECTION_LABEL_STEERED,
    SPREAD_INVERSION_ALPHA,
    US,
    DEFAULT_YIELD_TERMS,
    TERM_COLORS,
    YIELD_LINE_STYLE,
    YIELD_TERMS,
    CountryMetadata,
    PlotConfig,
)
from .console import this_thread_output_to
from .endlabels import ValueFormatter
from .frames import filter_to_date_range
from .latest import (
    PROJECTION_ATTR,
    QUOTE_TIME_ATTR,
    STEERED_ATTR,
    extend_cdn_yields,
    extend_us_yields,
    market_debt_for_projection,
    observed_through,
    project_to_now,
)
from .legend import finish_legend_and_title
from .measures import express
from .rates import RateCurve, in_window, rate_curves
from .regression import SlopeLabel, add_regression_segments
from .regression import legend_entry as regression_legend_entry
from .us_data import fetch_us_macro, fetch_us_population, fetch_us_yields, us_series_earliest

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
    ax: Axes,
    data: pd.DataFrame,
    *,
    enabled: bool,
    column: str,
    label: str,
    style: dict,
    projected_from: pd.Timestamp | None = None,
) -> Line2D | None:
    """Plot one macro series if it is enabled and has data; return the (legend) line or ``None``.

    With ``projected_from`` (--cur, ``latest.project_to_now``) the rows after
    that date are a projection, not data: they are drawn as a straight,
    fainter continuation, hidden from the legend, after the curve's steps.
    """
    if not enabled or data.empty or column not in data.columns or not data[column].notna().any():
        return None
    if projected_from is None:
        (line,) = ax.plot(data[DATE_COLUMN], data[column], label=label, **style)
        return line

    observed = data.loc[data[DATE_COLUMN] <= projected_from]
    ahead = data.loc[data[DATE_COLUMN] >= projected_from]
    faint = {key: value for key, value in style.items() if key != "drawstyle"} | {"alpha": PROJECTION_ALPHA}
    if not observed[column].notna().any():
        # The window lies wholly in the projection.
        (line,) = ax.plot(ahead[DATE_COLUMN], ahead[column], label=label, **faint)
        return line
    (line,) = ax.plot(observed[DATE_COLUMN], observed[column], label=label, **style)
    ax.plot(ahead[DATE_COLUMN], ahead[column], label="_projection", **faint)
    return line


def _yield_style(column: str, config: PlotConfig) -> dict:
    """Return the line style of one yield column: its term's own colour, unless the terms are the default five.

    With the default terms (no --yields:LIST) the lines take matplotlib's
    colour cycle in drawing order, as they always have, so those charts are
    unchanged; the cycle's colours are the default terms' ``TERM_COLORS``
    when all five are drawn. With terms chosen, each term has its own
    colour, so the 10-year is red however few are drawn.
    """
    if config.yield_terms == DEFAULT_YIELD_TERMS:
        return YIELD_LINE_STYLE
    term = next(key for key, name in YIELD_TERMS.items() if name == column)
    return YIELD_LINE_STYLE | {"color": TERM_COLORS[term]}


def add_canadian_yield_lines(ax: Axes, yields: pd.DataFrame, config: PlotConfig) -> list[Line2D]:
    """Plot Canadian yield curves: monthly history as steps, daily live data as lines.

    ``yields`` is date-indexed. Each tenor is split at ``CANADIAN_YIELD_HIST_END``;
    the live segment reuses the history segment's colour and is hidden from the
    legend so each tenor appears once.
    """
    if not config.yield_columns or yields.empty:
        return []

    data = filter_to_date_range(yields.reset_index(), DATE_COLUMN, config).set_index(DATE_COLUMN)
    if data.empty:
        return []

    legend_lines: list[Line2D] = []
    for column in config.yield_columns:
        if column not in data.columns:
            continue
        series = data[column].dropna()
        historical = series.loc[:CANADIAN_YIELD_HIST_END]
        current = series.loc[series.index > CANADIAN_YIELD_HIST_END]
        label = f"{column} Yield"
        color = None

        if not historical.empty:
            (line,) = ax.plot(
                historical.index, historical.values, label=label, drawstyle="steps-post", **_yield_style(column, config)
            )
            color = line.get_color()
            legend_lines.append(line)

        if not current.empty:
            (line,) = ax.plot(
                current.index,
                current.values,
                label="_nolegend_" if color is not None else label,
                **(_yield_style(column, config) | ({"color": color} if color is not None else {})),
            )
            if color is None:
                legend_lines.append(line)

    return legend_lines


def add_us_yield_lines(ax: Axes, yields: pd.DataFrame, config: PlotConfig) -> list[Line2D]:
    """Plot U.S. yield curves as ordinary daily time-series lines.

    ``yields`` has a ``DATE`` column (FRED frames are not date-indexed).
    """
    if not config.yield_columns or yields.empty:
        return []

    data = filter_to_date_range(yields, DATE_COLUMN, config)
    lines: list[Line2D] = []
    for column in config.yield_columns:
        if column in data.columns:
            (line,) = ax.plot(data[DATE_COLUMN], data[column], label=f"{column} Yield", **_yield_style(column, config))
            lines.append(line)
    return lines


def add_rate_lines(
    ax: Axes, rates: Iterable[RateCurve], config: PlotConfig, yield_lines: Iterable[Line2D] = ()
) -> list[tuple[Line2D, RateCurve]]:
    """Plot the yield axis's other curves (``ratesplot.rates``); return each drawn curve's legend line with it.

    Each is drawn from the value in effect at the window's start
    (``rates.in_window``). Where ``steps_until`` is set, the part up to it is
    drawn as steps and the rest as a line in the same colour, as the Canadian
    yields are. A spread's stretches below zero (an inverted curve) are
    shaded between it and zero, in its colour. A curve with ``color_of`` (a
    mortgage) takes the colour of that yield's line among ``yield_lines``
    when it is drawn, so the two match whatever colour the yield was given.
    """
    yield_colours = {line.get_label(): line.get_color() for line in yield_lines}
    drawn: list[tuple[Line2D, RateCurve]] = []
    for curve in rates:
        values = in_window(curve.values, config)
        if values.empty:
            continue
        style = curve.style
        if curve.color_of is not None and f"{curve.color_of} Yield" in yield_colours:
            style = style | {"color": yield_colours[f"{curve.color_of} Yield"]}
        if curve.steps_until is None:
            parts = [(values, style)]
        else:
            steps = values.loc[values.index <= curve.steps_until]
            line = values.loc[values.index > curve.steps_until]
            parts = [(steps, style | {"drawstyle": "steps-post"}), (line, style)]
        first: Line2D | None = None
        for part, style in parts:
            if part.empty:
                continue
            colour = {} if first is None else {"color": first.get_color()}
            (line,) = ax.plot(part.index, part.to_numpy(), label=curve.label if first is None else "_nolegend_", **(style | colour))
            first = first or line
            if curve.spread:
                steps = style.get("drawstyle") == "steps-post"
                ax.fill_between(
                    part.index, part.to_numpy(), 0.0, where=part.to_numpy() < 0, interpolate=not steps,
                    step="post" if steps else None, color=line.get_color(), alpha=SPREAD_INVERSION_ALPHA,
                    linewidth=0, label="_nolegend_",
                )
        if first is not None:
            drawn.append((first, curve))
    return drawn


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
    rates: list[RateCurve],
    config: PlotConfig,
    metadata: CountryMetadata,
    draw_yield_lines: YieldLineDrawer,
) -> None:
    """Draw one country's complete chart (lines, axes, title, legend) onto ``ax_yield``'s figure.

    Works on any figure: a pyplot one (``plot_country``) or a bare Agg one
    (``build_figure``), so the window and the GUI draw identical charts.
    """
    quote_time = yields.attrs.get(QUOTE_TIME_ATTR)  # --cur: when the day's quotes were taken, if any
    projected = macro.attrs.get(PROJECTION_ATTR, {})  # --cur: {column: date its data end}
    yield_lines = draw_yield_lines(ax_yield, yields, config)
    # Policy and mortgage rates and spreads (ratesplot.rates): on the yield
    # axis, listed with the yields.
    rate_lines = add_rate_lines(ax_yield, rates, config, yield_lines)

    ax_macro = ax_yield.twinx()
    macro_in_range = filter_to_date_range(macro, DATE_COLUMN, config)
    macro_lines: list[Line2D] = []
    macro_keys_drawn: list[str] = []
    macro_columns_drawn: list[str] = []
    for style_key, enabled, column, label in metadata.macro_specs(config):
        line = add_macro_line(
            ax_macro,
            macro_in_range,
            enabled=enabled,
            column=column,
            label=label,
            style=MACRO_PLOT_STYLES[style_key],
            projected_from=projected.get(column),
        )
        if line is not None:
            macro_lines.append(line)
            macro_keys_drawn.append(style_key)
            macro_columns_drawn.append(column)

    apply_axes_formatting(ax_yield, ax_macro, config, metadata)
    warn_macro_limits_coverage(macro_in_range, macro_columns_drawn, config)
    # --reg: each right-axis curve fitted by straight pieces on its log axis,
    # drawn now that the axis limits (which set the tolerance) are known; their
    # slope labels wait for the final layout (ratesplot.regression).
    slope_labels: list[SlopeLabel] = []
    if config.regression:
        slope_labels = add_regression_segments(ax_macro, macro, zip(macro_lines, macro_columns_drawn), projected, config)
    # -l: every drawn curve gets its last value at its end, written as its
    # axis writes values: yields in percent, right-axis curves as the tick
    # labels there (dollars, or percent of GDP under -r); a projected value
    # (--cur) reads "≈".
    end_labels: list[tuple[Line2D, ValueFormatter]] = []
    if config.end_labels:
        macro_format = format_percent if config.relative else partial(format_currency, prefix=metadata.currency_prefix)
        end_labels = [(line, format_yield) for line in yield_lines] + [
            (line, format_spread if curve.spread else format_yield) for line, curve in rate_lines
        ] + [
            (line, (lambda value, fmt=macro_format: PROJECTION_LABEL_PREFIX + fmt(value)) if column in projected else macro_format)
            for line, column in zip(macro_lines, macro_columns_drawn)
        ]
    # One legend entry says what the faint stretches are. Made after the line
    # widths are scaled (apply_axes_formatting), so it matches them.
    legend_macro = list(macro_lines)
    if any(column in projected for column in macro_columns_drawn):
        width = max(style["linewidth"] for style in MACRO_PLOT_STYLES.values()) * config.line_scale
        steered = set(macro.attrs.get(STEERED_ATTR, [])) & set(macro_columns_drawn)
        label = PROJECTION_LABEL_STEERED if steered else PROJECTION_LABEL
        legend_macro.append(Line2D([], [], color=PROJECTION_KEY_COLOR, linewidth=width, alpha=PROJECTION_ALPHA, label=label))
    if slope_labels:
        legend_macro.append(regression_legend_entry(config))
    # The title names only what is on the chart, and the subtitle only the
    # dates it covers; the legend keeps yields and macro curves as separate
    # column groups.
    title = metadata.title_for(
        yields_drawn=bool(yield_lines),
        macro_keys_drawn=macro_keys_drawn,
        config=config,
        rate_titles=[curve.title for _line, curve in rate_lines],
    )
    finish_legend_and_title(
        ax_yield,
        [yield_lines + [line for line, _curve in rate_lines], legend_macro],
        title,
        config,
        drawn_date_span((ax_yield, ax_macro)),
        end_labels,
        quote_time if yield_lines else None,
        slope_labels,
    )


def plot_country(
    yields: pd.DataFrame,
    macro: pd.DataFrame,
    rates: list[RateCurve],
    config: PlotConfig,
    metadata: CountryMetadata,
    draw_yield_lines: YieldLineDrawer,
) -> None:
    """Build one country's chart on a pyplot figure and show it in a matplotlib window."""
    _figure, ax_yield = plt.subplots(figsize=config.figsize_inches)
    draw_country(ax_yield, yields, macro, rates, config, metadata, draw_yield_lines)
    plt.show()


# ---------------------------------------------------------------------------
# Data preparation and run drivers
# ---------------------------------------------------------------------------


def prepare_cdn(config: PlotConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[RateCurve]]:
    """Fetch and align all selected Canadian inputs, in the configured measure; return ``(yields, macro, rates)``.

    With --cur the yields run on to the day's quotes, and the macro curves
    with them; the quotes' time is in ``yields.attrs[QUOTE_TIME_ATTR]``.
    ``rates`` are the yield axis's other chosen curves (``ratesplot.rates``).
    """
    _require_start(config)
    warn_series_coverage(config.start, CANADIAN_SERIES_EARLIEST)
    yields, quote_time = extend_cdn_yields(fetch_cdn_yields(config), config)
    rates = rate_curves("cdn", yields, config)
    warn_series_coverage(config.start, {curve.label: curve.first for curve in rates})
    if config.components:
        # Debt and interest by level of government replace the aggregate lines.
        series = (fetch_cdn_components(config), fetch_cdn_gdp(config))
    else:
        series = (fetch_cdn_debt(config), fetch_cdn_gdp(config), fetch_cdn_interest(config))
    # --cur on a chart reaching today: each curve carried on from its last data,
    # debt steered by the Government of Canada's market debt (ratesplot.latest).
    parts = [part for part in series if part is not None]
    observed = {column: observed_through(part[column].dropna().index) for part in parts for column in part.columns}
    history = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    market = market_debt_for_projection(observed, config)
    macro = project_to_now(align_cdn_macro(yields, series, config), observed, config, history, market)
    projected = macro.attrs.get(PROJECTION_ATTR, {})
    steered = macro.attrs.get(STEERED_ATTR, [])
    population = fetch_cdn_population() if config.per_capita else None
    yields.attrs[QUOTE_TIME_ATTR] = quote_time
    expressed = express(macro, config, CDN, population)
    expressed.attrs[PROJECTION_ATTR] = projected
    expressed.attrs[STEERED_ATTR] = steered
    return yields, expressed, rates


def prepare_us(config: PlotConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[RateCurve]]:
    """Fetch all selected U.S. inputs, trimmed to the end date, in the configured measure; return ``(yields, macro, rates)``.

    With --cur the yields run on to the day's quotes and federal debt daily
    (see ``ratesplot.latest``); the quotes' time is in ``yields.attrs[QUOTE_TIME_ATTR]``.
    ``rates`` are the yield axis's other chosen curves (``ratesplot.rates``).
    """
    _require_start(config)
    warn_series_coverage(config.start, us_series_earliest(config))
    yields, quote_time = extend_us_yields(fetch_us_yields(config), config)
    if not yields.empty:
        yields = yields.loc[yields[DATE_COLUMN] <= config.end]
    last_yield_date = yields[DATE_COLUMN].max() if not yields.empty else config.end
    rates = rate_curves("us", yields, config)
    warn_series_coverage(config.start, {curve.label: curve.first for curve in rates})

    macro = fetch_us_macro(config, last_yield_date)
    projected = macro.attrs.get(PROJECTION_ATTR, {})
    if not macro.empty:
        macro = macro.loc[macro[DATE_COLUMN] <= config.end]
    population = fetch_us_population() if config.per_capita else None
    yields.attrs[QUOTE_TIME_ATTR] = quote_time
    expressed = express(macro, config, US, population)
    expressed.attrs[PROJECTION_ATTR] = projected
    return yields, expressed, rates


@dataclass(frozen=True)
class Country:
    """Everything needed to produce one country's chart."""

    key: str  # short name, as used for its tab and output file
    show_field: str  # PlotConfig field that selects it
    metadata: CountryMetadata
    prepare: Callable[[PlotConfig], tuple[pd.DataFrame, pd.DataFrame, list[RateCurve]]]
    draw_yield_lines: YieldLineDrawer


# In drawing order (Canadian, then U.S.), as the command line always has.
COUNTRIES: tuple[Country, ...] = (
    Country("cdn", "show_cdn", CDN, prepare_cdn, add_canadian_yield_lines),
    Country("us", "show_us", US, prepare_us, add_us_yield_lines),
)


# ---------------------------------------------------------------------------
# The automatic start (no --start)
# ---------------------------------------------------------------------------


def _require_start(config: PlotConfig) -> None:
    """Refuse a config whose automatic start has not been found yet (``resolve_start``)."""
    if config.start is None:
        raise ValueError("the start date is automatic: call plotting.resolve_start(config) before preparing a chart")


def curve_first_dates(
    yields: pd.DataFrame, macro: pd.DataFrame, rates: list[RateCurve], config: PlotConfig, metadata: CountryMetadata
) -> dict[str, pd.Timestamp]:
    """Return the first date on which each curve ``draw_country`` would draw has a value, by its legend label.

    ``yields`` and ``macro`` are as ``prepare_*`` returns them (the Canadian
    yields are date-indexed, the other frames have a ``DATE`` column). The
    curves are every yield tenor, if yields are chosen, and each chosen
    right-axis curve (``metadata.macro_specs``) in the chosen measure: under
    -r debt begins where both debt and GDP do; and the yield axis's other
    curves (``rates``), which are there only when chosen.
    """
    firsts: dict[str, pd.Timestamp] = {}

    def note(frame: pd.DataFrame, dates: pd.DatetimeIndex, column: str, label: str) -> None:
        if column in frame.columns:
            has_value = frame[column].notna().to_numpy()
            if has_value.any():
                firsts[label] = dates[has_value][0]

    if config.yield_columns and not yields.empty:
        dates = pd.DatetimeIndex(yields[DATE_COLUMN] if DATE_COLUMN in yields.columns else yields.index)
        for column in config.yield_columns:
            note(yields, dates, column, f"{column} Yield")
    if not macro.empty:
        dates = pd.DatetimeIndex(macro[DATE_COLUMN])
        for _key, enabled, column, label in metadata.macro_specs(config):
            if enabled:
                note(macro, dates, column, label)
    for curve in rates:
        firsts[curve.label] = curve.first
    return firsts


def resolve_start(config: PlotConfig) -> PlotConfig:
    """Return ``config`` with its automatic start found; as it is when --start was given.

    Terry, 2026-09-29: without -s the start is "the oldest date that fully
    populates the curves that were specified", with no curve missing at the
    left; from that country's curves under -u or -c alone, otherwise from
    both countries' together. So each chosen country is prepared from
    ``EARLIEST_DATA_START``, the first date of each curve it would draw is
    read off (``curve_first_dates``), and the start is the latest of them.
    Terry, 2026-10-06: the start is then no earlier than
    ``DEFAULT_START_FLOOR``, unless the chart ends less than
    ``MIN_WINDOW_DAYS`` after that date; set to a date before
    ``EARLIEST_DATA_START``, the floor never applies and this is the rule of
    2026-09-29. The charts are then
    prepared from the start exactly as with -s set to it (the command line
    and the window keep the downloads for that second pass).

    The trial pass still begins at ``EARLIEST_DATA_START``, not at the floor:
    prepared from the floor, every curve would seem to begin there (the frames
    open with a row carrying the value then in effect), and the U.S. yields on
    the first trading day after it.

    The trial pass runs without --cur, which only adds values after the data
    end. It prints nothing: the real pass fetches the same data and prints
    the same progress and warnings. Only the calling thread's output is
    discarded (``console``): the window's main thread, or another visitor's
    drawing on the web page, prints as before. A curve that begins within
    ``MIN_WINDOW_DAYS`` of the end, or after it, cannot be whole in any
    window: it is left out of the choice (and said so), and drawn from where
    it begins, if at all.
    """
    if config.start is not None:
        return config
    trial = replace(config, start=EARLIEST_DATA_START, current=False)
    firsts: dict[tuple[str, str], pd.Timestamp] = {}  # (country, curve label) -> first date
    with this_thread_output_to(None):
        for country in COUNTRIES:
            if getattr(config, country.show_field):
                yields, macro, rates = country.prepare(trial)
                for label, first in curve_first_dates(yields, macro, rates, trial, country.metadata).items():
                    firsts[(country.metadata.country_name, label)] = first

    def named(curves: list[tuple[str, str]]) -> str:
        """``5-Year Yield and 10-Year Yield (Canada)``, one bracket per country."""
        countries = dict.fromkeys(country for country, _label in curves)
        return "; ".join(
            f"{' and '.join(label for c, label in curves if c == country)} ({country})" for country in countries
        )

    latest_start = config.end - pd.Timedelta(days=MIN_WINDOW_DAYS)
    whole = {curve: first for curve, first in firsts.items() if first <= latest_start}
    late = [curve for curve, first in firsts.items() if first > latest_start]
    left_out = f"; beginning after {latest_start:%Y-%m-%d}, too late to count for this end: {named(late)}" if late else ""
    if not whole:
        print(f"  Warning: no chosen curve begins by {latest_start:%Y-%m-%d}; the chart shows the last {MIN_WINDOW_DAYS} days{left_out}.")
        return replace(config, start=latest_start)
    start = max(whole.values())
    if start < DEFAULT_START_FLOOR <= latest_start:
        print(f"Start {DEFAULT_START_FLOOR:%Y-%m-%d}, the default start; every chosen curve has data by then{left_out}.")
        return replace(config, start=DEFAULT_START_FLOOR)
    # The curves begin after the floor, or the chart ends too soon after it.
    before_floor = ""
    if DEFAULT_START_FLOOR > latest_start:
        when = "before" if DEFAULT_START_FLOOR > config.end else f"less than {MIN_WINDOW_DAYS} days after"
        before_floor = f" (the chart ends {when} the default start, {DEFAULT_START_FLOOR:%Y-%m-%d})"
    last = [curve for curve, first in whole.items() if first == start]
    print(f"Start {start:%Y-%m-%d}, the first date on which every chosen curve has data{before_floor}; the last to begin: {named(last)}{left_out}.")
    return replace(config, start=start)


def build_figure(country: Country, config: PlotConfig) -> Figure:
    """Fetch, prepare and draw one country's chart on a new Agg figure, without pyplot.

    pyplot keeps global state and drives GUI windows, so it must not be used
    from a worker thread; a bare ``Figure`` with an Agg canvas is safe there
    as long as one thread uses it at a time. The DPI is pinned to
    ``CANVAS_DPI`` so the pixel size is exactly ``width_px`` x ``height_px``.
    An automatic start must be found first (``resolve_start``), once for all
    the charts drawn together.
    """
    figure = Figure(figsize=config.figsize_inches, dpi=CANVAS_DPI)
    FigureCanvasAgg(figure)
    ax_yield = figure.subplots()
    yields, macro, rates = country.prepare(config)
    draw_country(ax_yield, yields, macro, rates, config, country.metadata, country.draw_yield_lines)
    return figure


def run_cdn(config: PlotConfig) -> None:
    """Fetch all selected Canadian inputs and show the Canadian chart in a matplotlib window."""
    yields, macro, rates = prepare_cdn(config)
    plot_country(yields, macro, rates, config, CDN, add_canadian_yield_lines)
    print("CDN chart finished.\n")


def run_us(config: PlotConfig) -> None:
    """Fetch all selected U.S. inputs and show the U.S. chart in a matplotlib window."""
    yields, macro, rates = prepare_us(config)
    plot_country(yields, macro, rates, config, US, add_us_yield_lines)
    print("US chart finished.\n")
