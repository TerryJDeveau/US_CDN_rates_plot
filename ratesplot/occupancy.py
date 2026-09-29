"""Where the chart already has ink: every drawn line sampled into a coarse grid of the axes area.

Shared by the legend placement (``legend``) and the slope labels of the
regression segments (``regression``), which both look for room where they
cover as few lines as possible:

1. Every line on every axis of the figure is sampled a few pixels apart, in
   display coordinates (``collect_display_samples``).
2. The samples, and any solid obstacles such as the -l value labels, mark the
   cells of a grid over the axes area (``Raster``).
3. For the legend, a summed-area table of that grid scores every position of
   an axis-aligned box in O(1) (``OccupancyGrid``).

The grid is not scaled with ``PlotConfig.font_scale``: it measures lines,
whose widths are fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

# Display-space sampling density along each line segment, in pixels.
_SAMPLE_SPACING_PX = 3.0
# Occupancy-grid cell size. Lines are 1-3 px wide, so a 4 px cell marks
# exactly the cells a line passes through without exaggerating its footprint.
CELL_PX = 4


# ---------------------------------------------------------------------------
# Line sampling
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


def collect_display_samples(ax: Axes, exclude: Collection[Artist] = ()) -> np.ndarray:
    """Sample every line on every axis of ``ax.figure`` in display coordinates, except those in ``exclude``."""
    parts = [
        _line_display_samples(line, axis.transData)
        for axis in ax.figure.axes
        for line in axis.get_lines()
        if line not in exclude
    ]
    return np.concatenate(parts) if parts else np.empty((0, 2))


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Raster:
    """Integer raster of the axes area, padded by ``pad`` empty cells on every side: 1 where something is drawn.

    Cell ``(r, c)`` of the unpadded grid is ``cells[r + pad, c + pad]`` and
    covers display x ``x0 + c * CELL_PX`` onwards, y ``y0 + r * CELL_PX`` upwards.
    """

    x0: float  # display x of the grid's left edge
    y0: float  # display y of the grid's bottom edge
    rows: int
    cols: int
    pad: int
    cells: np.ndarray

    @classmethod
    def from_axes(cls, ax: Axes, points: np.ndarray, *, pad: int = 0, boxes: Sequence[Bbox] = ()) -> "Raster":
        """Rasterise ``points`` (display coordinates), and every cell any of ``boxes`` touches, over ``ax``."""
        box = ax.get_window_extent(ax.figure.canvas.get_renderer())
        cols = max(1, math.ceil(box.width / CELL_PX))
        rows = max(1, math.ceil(box.height / CELL_PX))
        raster = cls(box.x0, box.y0, rows, cols, pad, np.zeros((rows + 2 * pad, cols + 2 * pad), dtype=np.int32))
        raster.mark_points(points)
        for obstacle in boxes:
            raster.mark_box(obstacle)
        return raster

    def mark_points(self, points: np.ndarray) -> None:
        """Mark the cells ``points`` (display coordinates) fall in; points outside the axes are ignored."""
        if not points.size:
            return
        col = ((points[:, 0] - self.x0) / CELL_PX).astype(int)
        row = ((points[:, 1] - self.y0) / CELL_PX).astype(int)
        inside = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        self.cells[row[inside] + self.pad, col[inside] + self.pad] = 1

    def mark_box(self, obstacle: Bbox) -> None:
        """Mark every cell a solid obstacle (a display box, such as an -l value label) touches."""
        col0 = max(0, math.floor((obstacle.x0 - self.x0) / CELL_PX))
        col1 = min(self.cols, math.ceil((obstacle.x1 - self.x0) / CELL_PX))
        row0 = max(0, math.floor((obstacle.y0 - self.y0) / CELL_PX))
        row1 = min(self.rows, math.ceil((obstacle.y1 - self.y0) / CELL_PX))
        if col0 < col1 and row0 < row1:
            self.cells[row0 + self.pad : row1 + self.pad, col0 + self.pad : col1 + self.pad] = 1

    def cells_at(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the unpadded ``(rows, cols)`` of the cells ``points`` fall in, and a mask of those inside the axes."""
        col = np.floor((points[:, 0] - self.x0) / CELL_PX).astype(int)
        row = np.floor((points[:, 1] - self.y0) / CELL_PX).astype(int)
        inside = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        return np.stack([row, col], axis=1), inside


@dataclass(frozen=True)
class OccupancyGrid:
    """Summed-area table of a ``Raster``: occupied-cell counts of any axis-aligned box in four lookups.

    The raster's padding lets window sums be evaluated for boxes that extend
    past the axes edge (used when measuring clearance margins).
    """

    x0: float  # display x of the grid's left edge
    y0: float  # display y of the grid's bottom edge
    rows: int
    cols: int
    pad: int
    summed: np.ndarray

    @classmethod
    def from_axes(cls, ax: Axes, points: np.ndarray, *, pad: int, boxes: Sequence[Bbox] = ()) -> "OccupancyGrid":
        """Rasterise ``points`` (display coordinates), and every cell any of ``boxes`` touches, over ``ax``."""
        raster = Raster.from_axes(ax, points, pad=pad, boxes=boxes)
        # Summed-area table with a leading zero row/column so that the sum of
        # any window is four lookups.
        summed = np.zeros((raster.cells.shape[0] + 1, raster.cells.shape[1] + 1), dtype=np.int64)
        summed[1:, 1:] = raster.cells.cumsum(axis=0).cumsum(axis=1)
        return cls(raster.x0, raster.y0, raster.rows, raster.cols, raster.pad, summed)

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
