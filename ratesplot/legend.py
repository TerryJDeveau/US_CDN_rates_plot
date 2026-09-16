"""Legend auto-placement and figure title/layout finishing.

Matplotlib's ``loc="best"`` considers only nine anchor positions and ignores
lines drawn on a twin axis, so the legend is placed here instead:

1. The figure layout (title, subtitle, margins, x-limits) is finalised first,
   because moving the axes afterwards would invalidate everything below.
2. Every line on every axis of the figure is sampled a few pixels apart and
   rasterised into a coarse occupancy grid covering the axes area.
3. Candidate legend *shapes* are tried in preference order: the default two
   columns at full size, then other column counts, then the same shapes at
   reduced font sizes. Each shape is measured, and every position at which
   it fits inside the axes is scored in O(1) with a summed-area table.
4. The first shape that can be placed without covering any sampled point
   wins. If some positions keep it at least ``_TARGET_MARGIN_PX`` from every
   line, the one nearest an axes corner is used; otherwise the position with
   the greatest clearance is used wherever it falls. If no shape fits
   anywhere, the shape/position covering the fewest points is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
# Occupancy-grid cell size. Lines are 1-3 px wide, so a 4 px cell marks
# exactly the cells a line passes through without exaggerating its footprint.
_CELL_PX = 4
# Minimum gap between the legend and the axes frame.
_EDGE_PADDING_PX = 8.0
# Clearance from lines the search aims for. Once a shape can be placed this
# far from every line, corner proximity decides between positions; a smaller
# clearance is accepted only when no position reaches the target, in which
# case the largest achievable clearance wins wherever it is on the chart.
_TARGET_MARGIN_PX = 200
# Shapes in preference order: column counts first, then reduced font sizes.
_COLUMN_PREFERENCE = (2, 3, 1)
_FONT_SIZE_PREFERENCE = (LEGEND_FS, LEGEND_FS - 2, LEGEND_FS - 4)


# ---------------------------------------------------------------------------
# Line sampling and occupancy grid
# ---------------------------------------------------------------------------


def _line_display_samples(line, transform) -> np.ndarray:
    """Return points sampled every few pixels along one line, in display space."""
    xy = np.asarray(line.get_xydata(), dtype=float)  # numeric (date-converted) data
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return np.empty((0, 2))

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


@dataclass(frozen=True)
class _OccupancyGrid:
    """Boolean raster of the axes area: True where a plotted line passes.

    ``summed`` is the summed-area table of the grid padded by ``pad`` empty
    cells on every side, so window sums can be evaluated for boxes that
    extend past the axes edge (used when measuring clearance margins).
    """

    x0: float  # display x of the grid's left edge
    y0: float  # display y of the grid's bottom edge
    rows: int
    cols: int
    pad: int
    summed: np.ndarray

    @classmethod
    def from_axes(cls, ax: Axes, points: np.ndarray, *, pad: int) -> "_OccupancyGrid":
        box = ax.get_window_extent(ax.figure.canvas.get_renderer())
        cols = max(1, math.ceil(box.width / _CELL_PX))
        rows = max(1, math.ceil(box.height / _CELL_PX))
        grid = np.zeros((rows + 2 * pad, cols + 2 * pad), dtype=np.int32)

        if points.size:
            col = ((points[:, 0] - box.x0) / _CELL_PX).astype(int)
            row = ((points[:, 1] - box.y0) / _CELL_PX).astype(int)
            inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
            grid[row[inside] + pad, col[inside] + pad] = 1

        # Summed-area table with a leading zero row/column so that the sum of
        # any window is four lookups.
        summed = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), dtype=np.int64)
        summed[1:, 1:] = grid.cumsum(axis=0).cumsum(axis=1)
        return cls(box.x0, box.y0, rows, cols, pad, summed)

    def window_sums(self, height: int, width: int, *, margin: int) -> np.ndarray:
        """Occupied-cell counts for a ``height × width`` box at every interior position.

        Result ``[r, c]`` is the count inside the box whose bottom-left cell is
        ``(r, c)`` in unpadded grid coordinates, after growing the box by
        ``margin`` cells on every side. Shape is ``(rows - height + 1, cols - width + 1)``.
        """
        h, w = height + 2 * margin, width + 2 * margin
        s = self.summed
        total = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
        offset = self.pad - margin
        return total[offset : offset + self.rows - height + 1, offset : offset + self.cols - width + 1]


# ---------------------------------------------------------------------------
# Legend shapes
# ---------------------------------------------------------------------------


def _blank_handle() -> Artist:
    return plt.Line2D([], [], color="none", label="")


def arrange_legend_groups(groups: Sequence[Sequence[Artist]], ncols: int) -> tuple[list[Artist], int]:
    """Pad ``groups`` with blank entries so each group starts at the top of a column.

    Matplotlib fills legend columns top to bottom, so with the yield curves
    and the macro curves as two groups the result is e.g. yields in the left
    column and macro curves in the right one, rather than the macro entries
    continuing wherever the yields happen to end.

    Every group (including the last) is padded to a whole number of columns
    so the entry count is exactly ``rows × columns``; matplotlib then gives
    every column the same height, and trailing blanks cost no extra space.
    Returns the padded handle list and the number of columns actually used,
    which can be fewer than ``ncols`` for small legends.
    """
    groups = [list(group) for group in groups if group]
    total = sum(len(group) for group in groups)
    if total == 0:
        return [], 1
    ncols = max(1, min(ncols, total))
    if len(groups) == 1 or ncols == 1:
        return [handle for group in groups for handle in group], ncols

    # Fewest rows such that each group fits in whole columns within ncols.
    rows = math.ceil(total / ncols)
    while sum(math.ceil(len(group) / rows) for group in groups) > ncols:
        rows += 1

    handles: list[Artist] = []
    for group in groups:
        handles.extend(group)
        handles.extend(_blank_handle() for _ in range(-len(group) % rows))
    return handles, len(handles) // rows


def _draw_legend(ax: Axes, handles: Sequence[Artist], *, ncols: int, fontsize: float, anchor: tuple[float, float]) -> Legend:
    """Draw the legend with its upper-left corner at ``anchor`` (axes fraction)."""
    return ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper left",
        bbox_to_anchor=anchor,
        ncols=ncols,
        fontsize=fontsize,
        framealpha=0.95,
    )


@dataclass(frozen=True)
class _Placement:
    """One candidate shape at its best position."""

    handles: list[Artist]
    ncols: int
    fontsize: float
    covered: int  # occupied cells under the legend at its best position
    margin: int  # clearance achieved, in cells (only meaningful when covered == 0)
    row: int  # bottom-left cell of the legend
    col: int
    height: int  # legend size in cells
    width: int


def _best_position(grid: _OccupancyGrid, height: int, width: int, edge_cells: int) -> tuple[int, int, int, int] | None:
    """Return ``(covered, margin, row, col)`` for the best position of a box, or None if it cannot fit."""
    if height + 2 * edge_cells > grid.rows or width + 2 * edge_cells > grid.cols:
        return None

    def interior(sums: np.ndarray) -> np.ndarray:
        # Keep the legend at least ``edge_cells`` from the axes frame.
        return sums[edge_cells : sums.shape[0] - edge_cells, edge_cells : sums.shape[1] - edge_cells]

    base = interior(grid.window_sums(height, width, margin=0))
    if base.min() > 0:
        r, c = np.unravel_index(base.argmin(), base.shape)
        return int(base.min()), 0, int(r) + edge_cells, int(c) + edge_cells

    # Overlap-free positions exist. Aim for the target clearance; if that is
    # not achievable, binary-search the largest margin that is.
    target = _TARGET_MARGIN_PX // _CELL_PX
    if interior(grid.window_sums(height, width, margin=target)).min() == 0:
        margin = target
    else:
        low, high = 0, target - 1
        while low < high:
            mid = (low + high + 1) // 2
            if interior(grid.window_sums(height, width, margin=mid)).min() == 0:
                low = mid
            else:
                high = mid - 1
        margin = low

    free = interior(grid.window_sums(height, width, margin=margin)) == 0
    rows, cols = np.nonzero(free)
    # Among acceptable positions prefer the one nearest an axes corner, where
    # a legend conventionally sits.
    corner_distance = np.minimum(rows, free.shape[0] - 1 - rows) ** 2 + np.minimum(cols, free.shape[1] - 1 - cols) ** 2
    pick = corner_distance.argmin()
    return 0, margin, int(rows[pick]) + edge_cells, int(cols[pick]) + edge_cells


def auto_place_legend(ax: Axes, groups: Sequence[Sequence[Artist]]) -> Legend | None:
    """Place the legend where it obscures the plotted lines least. See module docstring."""
    figure = ax.figure
    figure.canvas.draw()  # data→display transforms must be final before sampling
    renderer = figure.canvas.get_renderer()
    grid = _OccupancyGrid.from_axes(ax, collect_display_samples(ax), pad=_TARGET_MARGIN_PX // _CELL_PX)
    edge_cells = math.ceil(_EDGE_PADDING_PX / _CELL_PX)

    best: _Placement | None = None
    for fontsize in _FONT_SIZE_PREFERENCE:
        for requested_cols in _COLUMN_PREFERENCE:
            handles, ncols = arrange_legend_groups(groups, requested_cols)
            if not handles:
                return None
            if best is not None and (ncols, fontsize) == (best.ncols, best.fontsize):
                continue  # small legends collapse several requests to the same shape

            # Measure this shape by drawing it provisionally.
            probe = _draw_legend(ax, handles, ncols=ncols, fontsize=fontsize, anchor=(0.0, 1.0))
            extent = probe.get_window_extent(renderer)
            probe.remove()
            height = math.ceil(extent.height / _CELL_PX)
            width = math.ceil(extent.width / _CELL_PX)

            result = _best_position(grid, height, width, edge_cells)
            if result is None:
                continue
            covered, margin, row, col = result
            candidate = _Placement(handles, ncols, fontsize, covered, margin, row, col, height, width)
            if covered == 0:
                best = candidate
                break
            if best is None or covered < best.covered:
                best = candidate
        if best is not None and best.covered == 0:
            break

    if best is None:
        # Nothing fits inside the axes at all; fall back to matplotlib's default corner.
        handles, ncols = arrange_legend_groups(groups, _COLUMN_PREFERENCE[0])
        return _draw_legend(ax, handles, ncols=ncols, fontsize=LEGEND_FS, anchor=(0.02, 0.98))

    # Convert the chosen cell to the legend's upper-left corner in axes fraction.
    upper_left_display = (
        grid.x0 + best.col * _CELL_PX,
        grid.y0 + (best.row + best.height) * _CELL_PX,
    )
    x_axes, y_axes = ax.transAxes.inverted().transform(upper_left_display)
    return _draw_legend(ax, best.handles, ncols=best.ncols, fontsize=best.fontsize, anchor=(float(x_axes), float(y_axes)))


# ---------------------------------------------------------------------------
# Title, subtitle, layout
# ---------------------------------------------------------------------------


def finish_legend_and_title(
    ax: Axes, legend_groups: Sequence[Sequence[Artist]], title: str, config: PlotConfig
) -> None:
    """Add title and date-range subtitle, fix the layout and x-limits, then place the legend.

    The legend goes last: its position is chosen against the final axes
    geometry, so nothing may move after it is placed.
    """
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

    auto_place_legend(ax, legend_groups)
