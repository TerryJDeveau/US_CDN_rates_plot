"""Legend auto-placement and figure title/layout finishing.

Matplotlib's ``loc="best"`` considers only nine anchor positions and ignores
lines drawn on a twin axis, so the legend is placed here instead:

1. The figure layout (title, subtitle, margins, x-limits) is finalised first,
   because moving the axes afterwards would invalidate everything below.
2. Every line on every axis of the figure is sampled a few pixels apart and
   rasterised into a coarse occupancy grid covering the axes area. With -l
   the value labels at the line ends (``endlabels``) fill their cells too.
3. Candidate legend *shapes* are tried in preference order: the default two
   columns at full size, then other column counts, then the same shapes at
   reduced font sizes. Each shape is measured, and every position at which
   it fits inside the axes is scored in O(1) with a summed-area table.
4. The first shape that can be placed without covering any sampled point
   wins. If some positions keep it at least ``_TARGET_MARGIN_PX`` from every
   line, the one nearest an axes corner is used. Otherwise the target is
   treated as aspirational: the legend goes where it is furthest from both
   the lines and the axes frame, i.e. centred in the largest pocket. If no
   shape fits anywhere, the shape/position covering the fewest points is used.

Font sizes and the pixel distances tied to them (clearance target, frame
padding, title spacing) are multiplied by ``PlotConfig.font_scale``, so a small
canvas is laid out as a scaled-down copy of the default one. The occupancy
grid itself (cell size, sampling density) is not scaled: it measures lines,
whose widths are fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from .axes import apply_date_xlim
from .config import CANVAS_DPI, LEGEND_FS, TITLE_FS, PlotConfig
from .endlabels import ValueFormatter, add_end_labels

# Display-space sampling density along each line segment, in pixels.
_SAMPLE_SPACING_PX = 3.0
# Occupancy-grid cell size. Lines are 1-3 px wide, so a 4 px cell marks
# exactly the cells a line passes through without exaggerating its footprint.
_CELL_PX = 4
# Minimum gap between the legend and the axes frame (at font scale 1).
_EDGE_PADDING_PX = 8.0
# Clearance from lines the search aims for. Once a shape can be placed this
# far from every line, corner proximity decides between positions; a smaller
# clearance is accepted only when no position reaches the target, in which
# case the largest achievable clearance wins wherever it is on the chart.
# Given at font scale 1; a scaled-down legend needs proportionally less room.
_TARGET_MARGIN_PX = 200
# Shapes in preference order: column counts first, then reduced font sizes
# (points below LEGEND_FS, before scaling).
_COLUMN_PREFERENCE = (2, 3, 1)
_FONT_SIZE_REDUCTIONS_PT = (0, 2, 4)

# Title block, at font scale 1. The subtitle's top sits this many title font
# sizes below the title's top (the spacing tuned on the default canvas: 0.027
# of its 15.36 in height), and the axes start a fixed gap below the subtitle.
_TITLE_TOP_Y = 0.995
_SUBTITLE_DROP_TITLE_SIZES = 0.027 * 15.36 * 72.0 / TITLE_FS
_SUBTITLE_FS_REDUCTION_PT = 6
_AXES_GAP_BELOW_SUBTITLE_PX = 10.0
# The title may use at most this fraction of the figure width before it is
# shrunk further than font_scale alone would make it.
_TITLE_MAX_WIDTH_FRACTION = 0.98


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
    def from_axes(cls, ax: Axes, points: np.ndarray, *, pad: int, boxes: Sequence[Bbox] = ()) -> "_OccupancyGrid":
        """Rasterise ``points`` (display coordinates), and every cell any of ``boxes`` touches, over ``ax``."""
        box = ax.get_window_extent(ax.figure.canvas.get_renderer())
        cols = max(1, math.ceil(box.width / _CELL_PX))
        rows = max(1, math.ceil(box.height / _CELL_PX))
        grid = np.zeros((rows + 2 * pad, cols + 2 * pad), dtype=np.int32)

        if points.size:
            col = ((points[:, 0] - box.x0) / _CELL_PX).astype(int)
            row = ((points[:, 1] - box.y0) / _CELL_PX).astype(int)
            inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
            grid[row[inside] + pad, col[inside] + pad] = 1

        # Solid obstacles such as the -l value labels: every cell they touch is occupied.
        for obstacle in boxes:
            col0 = max(0, math.floor((obstacle.x0 - box.x0) / _CELL_PX))
            col1 = min(cols, math.ceil((obstacle.x1 - box.x0) / _CELL_PX))
            row0 = max(0, math.floor((obstacle.y0 - box.y0) / _CELL_PX))
            row1 = min(rows, math.ceil((obstacle.y1 - box.y0) / _CELL_PX))
            if col0 < col1 and row0 < row1:
                grid[row0 + pad : row1 + pad, col0 + pad : col1 + pad] = 1

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
    clearance: int  # distance to the nearest line, in cells (only meaningful when covered == 0)
    row: int  # bottom-left cell of the legend
    col: int
    height: int  # legend size in cells
    width: int


def _best_position(
    grid: _OccupancyGrid, height: int, width: int, edge_cells: int, target: int
) -> tuple[int, int, int, int] | None:
    """Return ``(covered, clearance, row, col)`` for the best position of a box, or None if it cannot fit.

    ``covered`` is the number of occupied cells under the box (0 when it
    obscures nothing) and ``clearance`` its distance from the nearest line,
    in cells, capped at the grid padding. ``target`` is the clearance goal in
    cells (``_TARGET_MARGIN_PX`` scaled and converted).
    """
    if height + 2 * edge_cells > grid.rows or width + 2 * edge_cells > grid.cols:
        return None

    def interior(sums: np.ndarray) -> np.ndarray:
        # Keep the legend at least ``edge_cells`` from the axes frame.
        return sums[edge_cells : sums.shape[0] - edge_cells, edge_cells : sums.shape[1] - edge_cells]

    covered = interior(grid.window_sums(height, width, margin=0))
    if covered.min() > 0:
        r, c = np.unravel_index(covered.argmin(), covered.shape)
        return int(covered.min()), 0, int(r) + edge_cells, int(c) + edge_cells

    # Clearance map: a position is at least m cells clear when the box grown
    # by m on every side still covers nothing. Growing is monotone, so the
    # number of margins that stay free is the clearance itself.
    clearance = np.zeros(covered.shape, dtype=np.int64)
    for margin in range(1, grid.pad + 1):
        free = interior(grid.window_sums(height, width, margin=margin)) == 0
        if not free.any():
            break
        clearance += free
    clearance[covered > 0] = -1

    rows, cols = np.indices(covered.shape)
    meets_target = clearance >= target
    if meets_target.any():
        # Plenty of room: among positions with the target clearance take the
        # one nearest an axes corner, where a legend conventionally sits.
        corner_distance = (
            np.minimum(rows, covered.shape[0] - 1 - rows) ** 2
            + np.minimum(cols, covered.shape[1] - 1 - cols) ** 2
        ).astype(float)
        corner_distance[~meets_target] = np.inf
        r, c = np.unravel_index(corner_distance.argmin(), covered.shape)
    else:
        # Cramped: the target is aspirational. Treat the axes frame as a soft
        # obstacle too, so the legend centres itself in whatever pocket exists
        # instead of hugging the frame at one end of it.
        frame_distance = np.minimum(
            np.minimum(rows, covered.shape[0] - 1 - rows),
            np.minimum(cols, covered.shape[1] - 1 - cols),
        ) + edge_cells
        score = np.minimum(clearance, frame_distance).astype(float)
        score[covered > 0] = -np.inf
        # Tie-break on line clearance so, within a pocket, the legend still
        # sits as far from the data as the frame allows.
        score += clearance / (10.0 * grid.pad)
        r, c = np.unravel_index(score.argmax(), covered.shape)

    return 0, int(clearance[r, c]), int(r) + edge_cells, int(c) + edge_cells


def auto_place_legend(
    ax: Axes, groups: Sequence[Sequence[Artist]], font_scale: float, obstacles: Sequence[Bbox] = ()
) -> Legend | None:
    """Place the legend where it obscures the plotted lines (and ``obstacles``, display boxes) least. See module docstring."""
    figure = ax.figure
    figure.canvas.draw()  # data→display transforms must be final before sampling
    renderer = figure.canvas.get_renderer()
    target = round(_TARGET_MARGIN_PX * font_scale / _CELL_PX)
    grid = _OccupancyGrid.from_axes(ax, collect_display_samples(ax), pad=target, boxes=obstacles)
    edge_cells = math.ceil(_EDGE_PADDING_PX * font_scale / _CELL_PX)

    best: _Placement | None = None
    for reduction in _FONT_SIZE_REDUCTIONS_PT:
        fontsize = (LEGEND_FS - reduction) * font_scale
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

            result = _best_position(grid, height, width, edge_cells, target)
            if result is None:
                continue
            covered, clearance, row, col = result
            candidate = _Placement(handles, ncols, fontsize, covered, clearance, row, col, height, width)
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
        return _draw_legend(ax, handles, ncols=ncols, fontsize=LEGEND_FS * font_scale, anchor=(0.02, 0.98))

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
    ax: Axes,
    legend_groups: Sequence[Sequence[Artist]],
    title: str,
    config: PlotConfig,
    data_span: tuple[pd.Timestamp, pd.Timestamp] | None,
    end_labels: Sequence[tuple[Line2D, ValueFormatter]] = (),
    quote_time: pd.Timestamp | None = None,
) -> None:
    """Add title and date-range subtitle, fix the layout and x-limits, label line ends, then place the legend.

    The subtitle names ``data_span``, the first and last dates of the data
    actually drawn, which may be narrower than the axis (``config.start`` to
    ``config.end``). With nothing drawn it falls back to the axis range.
    When the drawing ends with intraday quotes (--cur), ``quote_time`` is when
    they were taken, and the subtitle says so: "… – 2026-09-28 13:50 EDT
    (intraday, provisional)".

    ``end_labels`` (-l) are the curves to label at their ends, each with the
    formatter for its value. The labels widen the date axis to the right, so
    they follow the requested limits; the legend then avoids them.

    The legend goes last: its position is chosen against the final axes
    geometry, so nothing may move after it is placed.
    """
    first, last = data_span if data_span is not None else (config.start, config.end)
    figure = ax.figure
    scale = config.font_scale
    figure_height = figure.get_figheight()
    title_artist = figure.suptitle(title, fontsize=TITLE_FS * scale, fontweight="bold", y=_TITLE_TOP_Y, va="top")

    # Scaling keeps the title's share of the width, but at MIN_FONT_SCALE (or
    # with a long composed title on a narrow canvas) it can still overflow;
    # shrink it to fit rather than let both ends fall off the figure.
    title_width = title_artist.get_window_extent(figure.canvas.get_renderer()).width
    max_width = _TITLE_MAX_WIDTH_FRACTION * figure.bbox.width
    if title_width > max_width:
        title_artist.set_fontsize(title_artist.get_fontsize() * max_width / title_width)

    # The subtitle follows the title's actual size, so a shrunken title does
    # not leave a gap above the date range.
    subtitle_fs = (TITLE_FS - _SUBTITLE_FS_REDUCTION_PT) * scale
    subtitle_y = _TITLE_TOP_Y - _SUBTITLE_DROP_TITLE_SIZES * title_artist.get_fontsize() / 72.0 / max(figure_height, 0.1)
    subtitle = f"{first:%Y-%m-%d} – {last:%Y-%m-%d}"
    if quote_time is not None and quote_time.tz_localize(None).normalize() == last:
        subtitle += f" {quote_time:%H:%M %Z} (intraday, provisional)"
    figure.text(
        0.5,
        subtitle_y,
        subtitle,
        ha="center",
        va="top",
        fontsize=subtitle_fs,
        style="italic",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.99], pad=0.4)

    # tight_layout does not know about suptitle/text, so push the axes top down
    # to sit a fixed gap below the subtitle's baseline (points → inches → fraction).
    subtitle_bottom = subtitle_y - subtitle_fs / 72.0 / max(figure_height, 0.1)
    gap_fraction = _AXES_GAP_BELOW_SUBTITLE_PX * scale / (figure_height * CANVAS_DPI)
    figure.subplots_adjust(top=subtitle_bottom - gap_fraction)
    apply_date_xlim(ax, config)

    obstacles = add_end_labels(ax, end_labels, config) if end_labels else []
    auto_place_legend(ax, legend_groups, scale, obstacles)
