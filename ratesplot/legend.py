"""Legend auto-placement and figure title/layout finishing.

Matplotlib's ``loc="best"`` only considers a handful of anchor positions and
ignores lines drawn on a twin axis. The placement here samples every line on
every axis of the figure in display (pixel) space, then scans a grid of
candidate legend positions and picks the one with the greatest clearance from
any sampled point.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend

from .axes import apply_date_xlim
from .config import CANVAS_DPI, LEGEND_FS, TITLE_FS, PlotConfig

# Display-space sampling density along each line segment, in pixels.
_SAMPLE_SPACING_PX = 3.0
# Cap per line so very long daily series do not dominate the cost.
_MAX_POINTS_PER_LINE = 600
# Candidate legend anchors: rows × columns across the axes area.
_CANDIDATE_ROWS = 8
_CANDIDATE_COLS = 10
_EDGE_PADDING_PX = 8.0


def _line_display_samples(line, transform) -> np.ndarray:
    """Return points sampled every few pixels along one line, in display space."""
    xy = np.asarray(line.get_xydata(), dtype=float)  # numeric (date-converted) data
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return np.empty((0, 2))

    if len(xy) > _MAX_POINTS_PER_LINE:
        keep = np.linspace(0, len(xy) - 1, _MAX_POINTS_PER_LINE).astype(int)
        xy = xy[keep]

    try:
        display = transform.transform(xy)
    except Exception:
        return np.empty((0, 2))

    # Sample each segment at roughly _SAMPLE_SPACING_PX intervals, excluding the
    # segment's end point (which is the next segment's start).
    deltas = np.diff(display, axis=0)
    counts = np.maximum(1, (np.hypot(deltas[:, 0], deltas[:, 1]) / _SAMPLE_SPACING_PX).astype(int))
    segment = np.repeat(np.arange(len(counts)), counts)
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    t = k * (1.0 / counts[segment])
    return display[:-1][segment] + t[:, None] * deltas[segment]


def collect_display_samples(ax: Axes) -> np.ndarray:
    """Sample every line on every axis of ``ax.figure`` in display coordinates."""
    parts = [
        _line_display_samples(line, axis.transData)
        for axis in ax.figure.axes
        for line in axis.get_lines()
    ]
    return np.concatenate(parts) if parts else np.empty((0, 2))


def legend_clearance(box: tuple[float, float, float, float], points: np.ndarray) -> float:
    """Return the minimum Euclidean distance from a display-space box to any point.

    Points inside the box have zero clearance.
    """
    if points.size == 0:
        return float("inf")

    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - points[:, 0], 0.0), points[:, 0] - x1)
    dy = np.maximum(np.maximum(y0 - points[:, 1], 0.0), points[:, 1] - y1)
    return float(np.min(np.hypot(dx, dy)))


def _place_legend(ax: Axes, handles, labels, anchor: tuple[float, float], columns: int) -> Legend:
    """Draw the legend with its upper-left corner at ``anchor`` (axes fraction)."""
    return ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=anchor,
        ncols=columns,
        fontsize=LEGEND_FS,
        framealpha=0.95,
    )


def auto_place_legend(ax: Axes, handles, labels, *, columns: int = 2) -> Legend:
    """Place the legend where it overlaps the plotted lines as little as possible."""
    figure = ax.figure
    figure.canvas.draw()  # needed so data→display transforms are final
    points = collect_display_samples(ax)

    # Draw a provisional legend purely to measure its rendered size.
    provisional = _place_legend(ax, handles, labels, (0.02, 0.98), columns)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = provisional.get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    width, height = legend_box.width, legend_box.height
    if min(axes_box.width, axes_box.height, width, height) <= 1:
        return provisional  # degenerate layout; nothing sensible to optimise

    # Feasible range for the legend's upper-left corner, keeping it inside the axes.
    x_min = axes_box.x0 + _EDGE_PADDING_PX
    x_max = max(axes_box.x1 - width - _EDGE_PADDING_PX, x_min)
    y_max = axes_box.y1 - _EDGE_PADDING_PX
    y_min = min(axes_box.y0 + height + _EDGE_PADDING_PX, y_max)

    best_clearance = -1.0
    best_corner = (x_min, y_max)
    for y in np.linspace(y_min, y_max, _CANDIDATE_ROWS):
        for x in np.linspace(x_min, x_max, _CANDIDATE_COLS):
            clearance = legend_clearance((x, y - height, x + width, y), points)
            if clearance > best_clearance:
                best_clearance, best_corner = clearance, (x, y)

    provisional.remove()
    x_axes, y_axes = ax.transAxes.inverted().transform(best_corner)
    return _place_legend(ax, handles, labels, (float(x_axes), float(y_axes)), columns)


def finish_legend_and_title(
    ax: Axes, handles: Sequence[Artist], title: str, config: PlotConfig
) -> None:
    """Add the legend, title and date-range subtitle, then fix the layout and x-limits."""
    if handles:
        # Two blank entries pad the legend to the same footprint the original
        # chart used, which keeps the placement search comparable across runs.
        blank = plt.Line2D([], [], color="none", label="")
        all_handles = [*handles, blank, blank]
        auto_place_legend(ax, all_handles, [handle.get_label() for handle in all_handles])

    figure = ax.figure
    subtitle_fs = TITLE_FS - 6
    subtitle_y = 0.968
    figure.suptitle(title, fontsize=TITLE_FS, fontweight="bold", y=0.995, va="top")
    figure.text(
        0.5,
        subtitle_y,
        f"{config.start:%Y-%m-%d} – {config.end:%Y-%m-%d}",
        ha="center",
        va="top",
        fontsize=subtitle_fs,
        style="italic",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.99], pad=0.4)

    # tight_layout does not know about suptitle/text, so push the axes top down
    # to sit a fixed 10 px below the subtitle's baseline (points → inches → fraction).
    figure_height = figure.get_figheight()
    subtitle_bottom = subtitle_y - subtitle_fs / 72.0 / max(figure_height, 0.1)
    gap_fraction = 10.0 / (figure_height * CANVAS_DPI)
    figure.subplots_adjust(top=subtitle_bottom - gap_fraction)
    apply_date_xlim(ax, config)
