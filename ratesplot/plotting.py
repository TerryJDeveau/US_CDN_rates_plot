"""Chart construction: line drawing, per-country plot functions, run_* drivers."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .axes import apply_axes_formatting
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
    COUNTRY_METADATA,
    DATE_COLUMN,
    MACRO_PLOT_STYLES,
    YIELD_COLUMNS,
    PlotConfig,
)
from .legend import finish_legend_and_title
from .us_data import US_SERIES_EARLIEST, fetch_us_macro, fetch_us_yields


def warn_series_coverage(start_date: pd.Timestamp, series_earliest: dict[str, pd.Timestamp]) -> None:
    """Warn when a requested chart begins before a source's earliest observation."""
    late_series = [
        (name, earliest)
        for name, earliest in series_earliest.items()
        if start_date < earliest
    ]
    if not late_series:
        return

    print(
        f"  Warning: requested start {start_date.date()} is earlier than the first "
        "available observation for:"
    )
    for name, earliest in late_series:
        print(f"    • {name} begins {earliest.date()}")


def format_dollar_value(value: float | None) -> str:
    """Format an internal dollar limit for warning text."""
    if value is None:
        return "None"
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= divisor:
            return f"{value / divisor:.1f}{suffix}"
    return f"{value:.1f}"


def warn_dollar_limits_coverage(
    df_macro: pd.DataFrame,
    macro_columns: Iterable[str],
    config: PlotConfig,
) -> None:
    """Warn when explicit macro-axis limits hide all or part of a selected curve."""
    if df_macro.empty or not list(macro_columns):
        return
    if config.macro_bottom is None and config.macro_top is None:
        return

    bottom_text = format_dollar_value(config.macro_bottom)
    top_text = format_dollar_value(config.macro_top)

    for column in macro_columns:
        if column not in df_macro.columns:
            continue
        series = df_macro[column].dropna()
        if series.empty:
            continue

        series_min, series_max = series.min(), series.max()
        if (config.macro_top is not None and series_min > config.macro_top) or (
            config.macro_bottom is not None and series_max < config.macro_bottom
        ):
            print(
                f"  Warning: Dollar curve '{column}' cannot be shown at all within limits "
                f"--bottom:{bottom_text}, --top:{top_text}."
            )
        elif (config.macro_bottom is not None and series_min < config.macro_bottom) or (
            config.macro_top is not None and series_max > config.macro_top
        ):
            print(
                f"  Warning: Dollar curve '{column}' is only shown for part of its range "
                f"due to limits --bottom:{bottom_text}, --top:{top_text}."
            )



def normalize_date_column(df: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """Return a copy with the requested date column normalized to ``DATE``.

    Pandas uses the index name when ``reset_index()`` is called.  Some source
    frames in this program therefore produce ``date`` rather than ``DATE``.
    Keeping that conversion in one helper prevents callers from depending on
    the exact capitalization/name used by an upstream data source.
    """
    if df.empty or date_column in df.columns:
        return df.copy()

    for candidate in ("DATE", "date", "index"):
        if candidate in df.columns:
            return df.rename(columns={candidate: date_column}).copy()

    raise KeyError(
        f"Unable to identify date column; expected {date_column!r}, 'date', or 'index'."
    )


def filter_to_date_range(
    df: pd.DataFrame, date_column: str, config: PlotConfig
) -> pd.DataFrame:
    """Return a copy containing only rows inside the requested chart window.

    ``date_column`` is the canonical name expected by the caller.  The helper
    first normalizes common Pandas/source variants so ``reset_index()`` does not
    silently create a ``KeyError`` when an index is named ``date``.
    """
    normalized = normalize_date_column(df, date_column)
    if normalized.empty:
        return normalized

    normalized[date_column] = pd.to_datetime(normalized[date_column])
    return normalized.loc[
        (normalized[date_column] >= config.start)
        & (normalized[date_column] <= config.end)
    ].copy()


def add_macro_line(
    ax,
    data: pd.DataFrame,
    *,
    enabled: bool,
    column: str,
    label: str,
    style: dict,
) -> object | None:
    """Plot one macro series if enabled and populated; return the created line."""
    if not enabled or data.empty or column not in data.columns:
        return None
    if not data[column].notna().any():
        return None

    line, = ax.plot(data[DATE_COLUMN], data[column], label=label, **style)
    return line


def add_canadian_yield_lines(ax, yields: pd.DataFrame, config: PlotConfig) -> list:
    """Plot Canadian yield curves, preserving the historical step-style segment."""
    if not config.include_yield or yields.empty:
        return []

    data = filter_to_date_range(yields.reset_index(), DATE_COLUMN, config)
    if data.empty:
        return []

    lines = []
    for column in YIELD_COLUMNS:
        if column not in data.columns:
            continue

        series = data.set_index(DATE_COLUMN)[column]
        historical = series.loc[series.index <= CANADIAN_YIELD_HIST_END].dropna()
        current = series.loc[series.index > CANADIAN_YIELD_HIST_END].dropna()
        line_color = None

        if not historical.empty:
            historical_line, = ax.plot(
                historical.index,
                historical.values,
                label=f"{column} Yield",
                linewidth=1.2,
                alpha=0.9,
                drawstyle="steps-post",
            )
            line_color = historical_line.get_color()
            lines.append(historical_line)

        if not current.empty:
            current_line, = ax.plot(
                current.index,
                current.values,
                label="_nolegend_" if not historical.empty else f"{column} Yield",
                linewidth=1.2,
                alpha=0.9,
                color=line_color,
            )
            if historical.empty:
                lines.append(current_line)

    return lines


def add_us_yield_lines(ax, yields: pd.DataFrame, config: PlotConfig) -> list:
    """Plot U.S. yield curves as ordinary time-series lines."""
    if not config.include_yield or yields.empty:
        return []

    data = filter_to_date_range(yields, DATE_COLUMN, config)
    lines = []
    for column in YIELD_COLUMNS:
        if column not in data.columns:
            continue
        line, = ax.plot(
            data[DATE_COLUMN],
            data[column],
            label=f"{column} Yield",
            linewidth=1.2,
            alpha=0.9,
        )
        lines.append(line)
    return lines


def create_chart_figure(config: PlotConfig):
    """Create a two-y-axis Matplotlib figure using the configured canvas size."""
    return plt.subplots(figsize=config.figsize_inches)


def plot_cdn(yields_all: pd.DataFrame, df_macro: pd.DataFrame, config: PlotConfig) -> None:
    """Build and display the Canadian chart from already-prepared data."""
    figure, ax1 = create_chart_figure(config)
    lines = add_canadian_yield_lines(ax1, yields_all, config)

    macro = filter_to_date_range(df_macro, DATE_COLUMN, config)
    ax2 = ax1.twinx()
    macro_handles = []
    macro_columns_plotted = []

    metadata = COUNTRY_METADATA["cdn"]
    macro_specs = (
        (
            "debt",
            config.include_debt,
            metadata["debt_column"],
            metadata["debt_label"],
        ),
        (
            "gdp",
            config.include_gdp,
            "TTM Nominal GDP ($)",
            "TTM Nominal GDP",
        ),
        (
            "interest",
            config.include_interest,
            metadata["interest_column"],
            metadata["interest_label"],
        ),
    )

    for style_key, enabled, column, label in macro_specs:
        line = add_macro_line(
            ax2,
            macro,
            enabled=enabled,
            column=column,
            label=label,
            style=MACRO_PLOT_STYLES[style_key],
        )
        if line is not None:
            macro_handles.append(line)
            macro_columns_plotted.append(column)

    all_handles = lines + macro_handles
    apply_axes_formatting(
        ax1,
        ax2,
        config,
        currency_prefix=metadata["currency_prefix"],
        currency_label=metadata["currency_label"],
    )
    warn_dollar_limits_coverage(macro, macro_columns_plotted, config)
    finish_legend_and_title(ax1, all_handles, metadata["title"], config)
    plt.show()


def run_cdn(config: PlotConfig) -> None:
    """Fetch all selected Canadian inputs and render the Canadian chart."""
    warn_series_coverage(config.start, CANADIAN_SERIES_EARLIEST)
    yields_all = fetch_cdn_yields(config)
    debt_q = fetch_cdn_debt(config)
    gdp_q = fetch_cdn_gdp(config)
    interest_q = fetch_cdn_interest(config)
    macro = align_cdn_macro(yields_all, debt_q, gdp_q, interest_q, config)
    plot_cdn(yields_all, macro, config)
    print("CDN chart finished.\n")


def plot_us(data_yields: pd.DataFrame, data_macro: pd.DataFrame, config: PlotConfig) -> None:
    """Build and display the U.S. chart from already-prepared data."""
    yields = filter_to_date_range(data_yields, DATE_COLUMN, config)
    macro = filter_to_date_range(data_macro, DATE_COLUMN, config)

    figure, ax1 = create_chart_figure(config)
    lines = add_us_yield_lines(ax1, yields, config)

    ax2 = ax1.twinx()
    macro_handles = []
    macro_columns_plotted = []
    metadata = COUNTRY_METADATA["us"]
    macro_specs = (
        (
            "debt",
            config.include_debt,
            metadata["debt_column"],
            metadata["debt_label"],
        ),
        (
            "gdp",
            config.include_gdp,
            "TTM Nominal GDP ($)",
            "TTM Nominal GDP",
        ),
        (
            "interest",
            config.include_interest,
            metadata["interest_column"],
            metadata["interest_label"],
        ),
    )

    for style_key, enabled, column, label in macro_specs:
        line = add_macro_line(
            ax2,
            macro,
            enabled=enabled,
            column=column,
            label=label,
            style=MACRO_PLOT_STYLES[style_key],
        )
        if line is not None:
            macro_handles.append(line)
            macro_columns_plotted.append(column)

    all_handles = lines + macro_handles
    apply_axes_formatting(
        ax1,
        ax2,
        config,
        currency_prefix=metadata["currency_prefix"],
        currency_label=metadata["currency_label"],
    )
    warn_dollar_limits_coverage(macro, macro_columns_plotted, config)
    finish_legend_and_title(ax1, all_handles, metadata["title"], config)
    plt.show()


def run_us(config: PlotConfig) -> None:
    """Fetch all selected U.S. inputs and render the U.S. chart."""
    warn_series_coverage(config.start, US_SERIES_EARLIEST)
    data_yields = fetch_us_yields(config)
    if not data_yields.empty:
        data_yields = data_yields.loc[data_yields[DATE_COLUMN] <= config.end]
    last_yield_date = data_yields[DATE_COLUMN].max() if not data_yields.empty else config.end

    data_macro = fetch_us_macro(config, last_yield_date)
    if not data_macro.empty:
        data_macro = data_macro.loc[data_macro[DATE_COLUMN] <= config.end]

    plot_us(data_yields, data_macro, config)
    print("US chart finished.\n")
