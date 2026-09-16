"""Legend auto-placement and figure title/layout finishing."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .axes import apply_date_xlim
from .config import CANVAS_DPI, LEGEND_FS, TITLE_FS, PlotConfig


def line_xy_numeric(line) -> tuple[np.ndarray, np.ndarray] | None:
    """Return finite numeric x/y arrays from a Matplotlib line."""
    x_raw = np.asarray(line.get_xdata())
    y_raw = np.asarray(line.get_ydata())
    if x_raw.size == 0 or y_raw.size == 0:
        return None

    count = min(x_raw.size, y_raw.size)
    x_raw, y_raw = x_raw[:count], y_raw[:count]

    if np.issubdtype(x_raw.dtype, np.datetime64) or np.issubdtype(x_raw.dtype, np.object_):
        try:
            x = mdates.date2num(pd.to_datetime(x_raw))
        except Exception:
            return None
    else:
        x = np.asarray(x_raw, dtype=float)

    y = np.asarray(y_raw, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return x[mask], y[mask]


def collect_display_samples(ax1, *, max_points_per_line: int = 600) -> list[tuple[float, float]]:
    """Sample rendered line segments in display coordinates for legend placement."""
    samples: list[tuple[float, float]] = []
    for axis in ax1.figure.axes:
        transform = axis.transData
        for line in axis.get_lines():
            parsed = line_xy_numeric(line)
            if parsed is None:
                continue

            x, y = parsed
            if len(x) > max_points_per_line:
                indices = np.linspace(0, len(x) - 1, max_points_per_line).astype(int)
                x, y = x[indices], y[indices]

            try:
                display_points = transform.transform(np.column_stack([x, y]))
            except Exception:
                continue
            if len(display_points) < 2:
                continue

            deltas = np.diff(display_points, axis=0)
            segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
            for index, segment_length in enumerate(segment_lengths):
                sample_count = max(1, int(segment_length / 3.0))
                t_values = np.linspace(0.0, 1.0, sample_count, endpoint=False)
                xs = display_points[index, 0] + t_values * deltas[index, 0]
                ys = display_points[index, 1] + t_values * deltas[index, 1]
                samples.extend(zip(xs.tolist(), ys.tolist()))

    return samples


def legend_clearance(box: tuple[float, float, float, float], points: np.ndarray) -> float:
    """Return the minimum Euclidean distance from a legend box to plotted points."""
    if points.size == 0:
        return float("inf")

    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - points[:, 0], 0.0), points[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - points[:, 1], 0.0), points[:, 1] - y1)
    return float(np.min(np.hypot(dx, dy)))


def auto_place_legend(ax1, handles, labels, *, columns: int = 2):
    """Place a legend where it overlaps the plotted lines as little as possible."""
    figure = ax1.figure
    try:
        figure.canvas.draw()
    except Exception:
        figure.canvas.draw_idle()

    samples = collect_display_samples(ax1)
    points = np.asarray(samples, dtype=float).reshape(-1, 2) if samples else np.empty((0, 2))

    candidate_legend = ax1.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncols=columns,
        fontsize=LEGEND_FS,
        framealpha=0.95,
    )

    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
    except Exception:
        figure.canvas.draw_idle()
        renderer = figure.canvas.get_renderer()

    legend_box = candidate_legend.get_window_extent(renderer)
    axes_box = ax1.get_window_extent(renderer)
    legend_width, legend_height = legend_box.width, legend_box.height
    if min(axes_box.width, axes_box.height, legend_width, legend_height) <= 1:
        return candidate_legend

    padding_px = 8.0
    x_min = axes_box.x0 + padding_px
    x_max = axes_box.x1 - legend_width - padding_px
    y_min = axes_box.y0 + legend_height + padding_px
    y_max = axes_box.y1 - padding_px

    if x_max < x_min:
        x_min = x_max = axes_box.x0 + padding_px
    if y_max < y_min:
        y_min = y_max = axes_box.y1 - padding_px

    best_clearance = -1.0
    best_display_position = (x_min, y_max)
    for candidate_y in np.linspace(y_min, y_max, 8):
        for candidate_x in np.linspace(x_min, x_max, 10):
            box = (
                candidate_x,
                candidate_y - legend_height,
                candidate_x + legend_width,
                candidate_y,
            )
            clearance = legend_clearance(box, points)
            if clearance > best_clearance:
                best_clearance = clearance
                best_display_position = (candidate_x, candidate_y)

    axes_inverse = ax1.transAxes.inverted()
    x_axes, y_axes = axes_inverse.transform(best_display_position)
    candidate_legend.remove()
    return ax1.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(float(x_axes), float(y_axes)),
        ncols=columns,
        fontsize=LEGEND_FS,
        framealpha=0.95,
    )


def finish_legend_and_title(ax1, handles, title: str, config: PlotConfig) -> None:
    """Add the legend, title/subtitle, layout adjustments, and date limits."""
    if handles:
        # Two blank legend entries preserve the visual spacing used by the original
        # chart while allowing the custom placement algorithm to choose the position.
        blank_handle = plt.Line2D([], [], color="none", label="")
        all_handles = list(handles) + [blank_handle, blank_handle]
        labels = [handle.get_label() for handle in all_handles]
        auto_place_legend(ax1, all_handles, labels)

    figure = ax1.figure
    figure.suptitle(title, fontsize=TITLE_FS, fontweight="bold", y=0.995, va="top")
    subtitle = f"{config.start.strftime('%Y-%m-%d')} – {config.end.strftime('%Y-%m-%d')}"
    subtitle_y = 0.968
    figure.text(
        0.5,
        subtitle_y,
        subtitle,
        ha="center",
        va="top",
        fontsize=TITLE_FS - 6,
        style="italic",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.99], pad=0.4)

    figure_height = figure.get_figheight()
    subtitle_bottom = subtitle_y - (TITLE_FS - 6) / 72.0 / max(figure_height, 0.1)
    gap_fraction = 10.0 / (figure_height * CANVAS_DPI)
    figure.subplots_adjust(top=subtitle_bottom - gap_fraction)
    apply_date_xlim(ax1, config)
