"""Regression segments on the right-axis curves, each labelled with its slope in %/yr (--reg).

The right axis is logarithmic, so a straight line on it is a constant rate
of growth. Each drawn right-axis curve (debt, GDP, interest, each level of
government, federal debt before 1933; in dollars, as % of GDP or per person)
is cut into as few pieces as possible, and each piece is fitted by least
squares of ln(value) on time in years. The fitted slope times 100 is the
piece's growth rate in %/yr, the unit of such a slope (Terry, 2026-09-29).

What is fitted
--------------
- The curve as plotted, over the window: its *observations*, the dates on
  which the forward-filled curve takes a new value (a quarterly figure at its
  quarter end, an annual one at its year end, the Treasury's debt each
  business day). The value carried to the window start from before it is
  not an observation there, and neither is --cur's projection to today (a
  straight line by construction): pieces run from the first observation in
  the window to the last.
- Each observation is weighted by the time it stands for, half the gap to
  each neighbour, so the daily Treasury figures after FRED's last quarter
  count per year as the quarterly ones before them do, not ninety times more.

How the pieces are chosen
-------------------------
- A piece is acceptable when its fitted line lies within the tolerance of
  every observation in it, so at its ends too. The tolerance is a vertical
  distance on the log axis: ``_AXIS_FRACTION`` of the axis height (what the
  eye sees as "close", whatever the span), but not less than
  ``_MIN_TOLERANCE``, below which a zoomed-in window would be cut up by
  quarter-to-quarter noise.
- The curve is covered by the fewest acceptable pieces; among equally few,
  the least total squared error decides where they break (``fit_pieces``).
- The pieces are separate regressions and do not join: a jump in the data
  (debt in 2020) falls between two of them.

How they are drawn
------------------
- Each piece is a thick, paler, translucent line in its curve's colour,
  under the curves, from its first observation to its last.
- Its slope is written above it, turned to lie along it, in the curve's own
  colour, after the layout, the date limits and the -l labels are final
  (``add_slope_labels``, called by ``legend.finish_legend_and_title``). It
  goes where it covers the fewest other lines and labels, nearest the
  piece's middle; below the piece only if above would leave the axes. The
  legend then avoids the labels, and the pieces, as it avoids everything drawn.

Sizes are multiplied by ``PlotConfig.font_scale`` and line widths by
``PlotConfig.line_scale`` like everything else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from .config import DATE_COLUMN, PlotConfig
from .occupancy import CELL_PX, Raster, collect_display_samples

# The tolerance, as a share of the right axis's height (in log terms): 1 % is
# about 12 px on the default canvas, a gap the eye still reads as "on the
# curve" next to a line this thick.
_AXIS_FRACTION = 0.01
# ...and never less than 1 % of the value (ln 1.01): below that, on a window
# of a year or two, the pieces would follow each quarter's noise.
_MIN_TOLERANCE = math.log1p(0.01)
# A curve needs this many observations in the window to be fitted at all.
_MIN_OBSERVATIONS = 3
_DAYS_PER_YEAR = 365.25

# The pieces: the curve's colour this far towards white, at this opacity and
# width (points, before PlotConfig.line_scale; the curves are 2.5), under the
# curves (lines are at zorder 2).
_LIGHTEN = 0.45
_ALPHA = 0.5
_LINE_WIDTH_PT = 8.0
_ZORDER = 1.5
LEGEND_LABEL = "Regression segments, slope in %/yr"

# The labels, in points before PlotConfig.font_scale: size, gap between the
# piece's edge and the text, and the white halo that keeps grid lines out of
# the digits.
_LABEL_FS = 12
_LABEL_GAP_PT = 2.0
_HALO_PT = 2.5
# Candidate positions along a piece, and the sampling of a label's area when
# counting what it would cover, in pixels.
_STEP_PX = CELL_PX
_AREA_SAMPLE_PX = CELL_PX / 2
_SAMPLES_PER_CELL = (CELL_PX / _AREA_SAMPLE_PX) ** 2
# What covering a cell costs: a line's 1 (the halo keeps the digits readable
# across it), another label's this much (text over text is not readable).
_LABEL_CELL_COST = 10
# A label may stand further off its piece, by up to this many steps of half
# its height, where that uncovers something; each step costs as much as
# covering this many cells of a line.
_LIFT_STEPS = 3
_LIFT_COST_CELLS = 4
# A label goes below its piece only where above it would cover this many
# more cells (lines) than below: "above" is asked for, "if possible".
_BELOW_COST_CELLS = 40
# A label keeps this far inside the axes frame, in points.
_FRAME_GAP_PT = 2.0


@dataclass(frozen=True)
class Piece:
    """One fitted piece: observations ``first`` to ``last`` (inclusive) and their line ``ln y = intercept + slope * x``."""

    first: int
    last: int
    slope: float  # ln units per year; x 100 = %/yr
    intercept: float  # at x = 0 (the x of ``fit_pieces``)


@dataclass(frozen=True)
class SlopeLabel:
    """A drawn piece waiting for its label (placed once the layout is final)."""

    piece: Line2D  # the pale line; its two points are the piece's ends
    text: str
    color: object  # the curve's own colour


# ---------------------------------------------------------------------------
# The observations
# ---------------------------------------------------------------------------


def observations(macro: pd.DataFrame, column: str, config: PlotConfig, projected_from: pd.Timestamp | None) -> pd.Series:
    """Return a curve's observations in the window (value by date): where the plotted curve takes a new value.

    ``macro`` is the whole prepared frame. Both data layers begin it with a
    row at the window start carrying the values then in effect
    (``cdn_data.align_cdn_macro``, ``us_data.fetch_us_macro``), so its first
    row is never taken as an observation: the value there was observed
    before the window. Rows after ``projected_from`` (--cur) are projection,
    not data. Values a log axis cannot show (zero or less) are left out.
    """
    if column not in macro.columns or macro.empty:
        return pd.Series(dtype=float)
    values = macro[column]
    dates = pd.to_datetime(macro[DATE_COLUMN])
    changed = values.notna() & values.ne(values.shift())
    changed.iloc[0] = False
    last = config.end if projected_from is None else min(config.end, projected_from)
    keep = changed & (dates >= config.start) & (dates <= last) & (values > 0)
    return pd.Series(values[keep].to_numpy(dtype=float), index=pd.DatetimeIndex(dates[keep]))


def time_weights(x: np.ndarray) -> np.ndarray:
    """Weight each observation by the time it stands for: half the gap to each neighbour (in ``x``'s units)."""
    weights = np.zeros(len(x))
    gaps = np.diff(x) / 2
    weights[:-1] += gaps
    weights[1:] += gaps
    return weights


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class _Sums:
    """Running weighted sums, so the least-squares line of any run of observations costs O(1).

    ``x`` and ``z`` are measured from their first values, which keeps the
    centred sums below well conditioned.
    """

    def __init__(self, x: np.ndarray, z: np.ndarray, w: np.ndarray) -> None:
        def running(values: np.ndarray) -> np.ndarray:
            return np.concatenate(([0.0], np.cumsum(values)))

        self.w = running(w)
        self.x = running(w * x)
        self.z = running(w * z)
        self.xx = running(w * x * x)
        self.xz = running(w * x * z)
        self.zz = running(w * z * z)

    def fit(self, first, last) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(slope, intercept, squared error)`` of observations ``first..last`` (either may be an array)."""
        a, b = first, np.asarray(last) + 1
        sw = self.w[b] - self.w[a]
        sx = self.x[b] - self.x[a]
        sz = self.z[b] - self.z[a]
        mean_x, mean_z = sx / sw, sz / sw
        cxx = (self.xx[b] - self.xx[a]) - sx * mean_x
        cxz = (self.xz[b] - self.xz[a]) - sx * mean_z
        czz = (self.zz[b] - self.zz[a]) - sz * mean_z
        slope = cxz / cxx
        return slope, mean_z - slope * mean_x, np.maximum(czz - slope * cxz, 0.0)


def fit_pieces(x: np.ndarray, z: np.ndarray, w: np.ndarray, tolerance: float) -> list[Piece]:
    """Cover the observations with the fewest least-squares lines each within ``tolerance`` of all its points.

    ``x`` ascending (years), ``z`` = ln(value), ``w`` the weights. Among the
    coverings with fewest pieces, the one with the least total weighted
    squared error is taken. Pieces have at least two observations.

    1. For each first observation ``i``, ``reach[i]`` is the last one a piece
       starting there can extend to. A piece that fits still fits with its
       first observation dropped, so ``reach`` never decreases, and one pass
       with two pointers finds it all.
    2. Dynamic programming over the end of the covered run: the best covering
       of observations ``0..j`` is the best covering of ``0..i-1`` plus the
       piece ``i..j``, over every ``i`` that can reach ``j``, compared by
       (pieces, squared error).

    A run of three observations is always allowed, even beyond the tolerance,
    so that a covering exists however noisy the data (two are always exact).
    """
    count = len(x)
    if count < 2:
        return []
    sums = _Sums(x - x[0], z - z[0], w)

    def fits(first: int, last: int) -> bool:
        slope, intercept, _ = sums.fit(first, last)
        residuals = (z[first : last + 1] - z[0]) - (intercept + slope * (x[first : last + 1] - x[0]))
        return float(np.abs(residuals).max()) <= tolerance

    reach = np.empty(count, dtype=int)
    last = 1
    for first in range(count - 1):
        last = max(last, first + 1)
        while last + 1 < count and fits(first, last + 1):
            last += 1
        reach[first] = last
    reach[count - 1] = count - 1
    reach = np.maximum.accumulate(np.maximum(reach, np.minimum(np.arange(count) + 2, count - 1)))

    # best_*[k]: the best covering of observations 0..k-1; start[k] is where its last piece begins.
    best_pieces = np.full(count + 1, np.inf)
    best_error = np.full(count + 1, np.inf)
    best_pieces[0] = best_error[0] = 0.0
    start = np.zeros(count + 1, dtype=int)
    for end in range(1, count):
        firsts = np.arange(int(np.searchsorted(reach, end)), end)
        if not len(firsts):
            continue
        pieces = best_pieces[firsts] + 1
        error = best_error[firsts] + sums.fit(firsts, end)[2]
        choice = np.lexsort((error, pieces))[0]
        best_pieces[end + 1], best_error[end + 1], start[end + 1] = pieces[choice], error[choice], firsts[choice]
    if not np.isfinite(best_pieces[count]):
        return []

    runs: list[tuple[int, int]] = []
    end = count
    while end > 0:
        runs.append((int(start[end]), end - 1))
        end = start[end]
    return [_fitted(x, z, w, first, last) for first, last in reversed(runs)]


def _fitted(x: np.ndarray, z: np.ndarray, w: np.ndarray, first: int, last: int) -> Piece:
    """Fit observations ``first..last`` directly (centred), for the drawn line itself."""
    xs, zs, ws = x[first : last + 1], z[first : last + 1], w[first : last + 1]
    mean_x, mean_z = np.average(xs, weights=ws), np.average(zs, weights=ws)
    slope = float(np.sum(ws * (xs - mean_x) * (zs - mean_z)) / np.sum(ws * (xs - mean_x) ** 2))
    return Piece(first, last, slope, float(mean_z - slope * mean_x))


def tolerance_for(ax: Axes) -> float:
    """Return the fitting tolerance in ln units for curves on ``ax`` (a log axis with its limits set)."""
    low, high = ax.get_ylim()
    span = abs(math.log(high / low)) if low > 0 and high > 0 else 0.0
    return max(_AXIS_FRACTION * span, _MIN_TOLERANCE)


def slope_text(slope: float) -> str:
    """Write a slope in ln units per year as %/yr with one decimal and its sign: ``+6.8%/yr``, ``−1.2%/yr``."""
    percent = round(100.0 * slope, 1)
    if percent == 0:
        return "0.0%/yr"
    return f"{percent:+.1f}%/yr".replace("-", "\N{MINUS SIGN}")


# ---------------------------------------------------------------------------
# Drawing the pieces
# ---------------------------------------------------------------------------


def _lighter(color) -> tuple[float, float, float]:
    """Return ``color`` moved ``_LIGHTEN`` of the way towards white."""
    return tuple(channel + (1.0 - channel) * _LIGHTEN for channel in to_rgb(color))


def _piece_style(color, config: PlotConfig) -> dict:
    return {
        "color": _lighter(color),
        "alpha": _ALPHA,
        "linewidth": _LINE_WIDTH_PT * config.line_scale,
        "solid_capstyle": "butt",
    }


def legend_entry(config: PlotConfig) -> Line2D:
    """The one legend entry for all the pieces: a grey piece and what the labels are."""
    return Line2D([], [], label=LEGEND_LABEL, **_piece_style("black", config))


def add_regression_segments(
    ax: Axes,
    macro: pd.DataFrame,
    curves: Iterable[tuple[Line2D, str]],
    projected: dict[str, pd.Timestamp],
    config: PlotConfig,
) -> list[SlopeLabel]:
    """Fit and draw the pieces of each drawn right-axis curve; return their labels, to be placed later.

    ``ax`` is the right axis, with its limits already set (they give the
    tolerance, and the pieces must not change them). ``curves`` pairs each
    drawn curve's line with its column in ``macro``, the whole prepared frame;
    ``projected`` maps a column to the date its data end (--cur).
    """
    tolerance = tolerance_for(ax)
    labels: list[SlopeLabel] = []
    for line, column in curves:
        observed = observations(macro, column, config, projected.get(column))
        if len(observed) < _MIN_OBSERVATIONS:
            continue
        dates = observed.index
        x = ((dates - dates[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float) / _DAYS_PER_YEAR
        z = np.log(observed.to_numpy())
        for piece in fit_pieces(x, z, time_weights(x), tolerance):
            ends = [piece.first, piece.last]
            values = np.exp(piece.intercept + piece.slope * x[ends])
            (drawn,) = ax.plot(
                dates[ends], values, label="_regression", zorder=_ZORDER, scalex=False, scaley=False,
                **_piece_style(line.get_color(), config),
            )
            labels.append(SlopeLabel(drawn, slope_text(piece.slope), line.get_color()))
    return labels


# ---------------------------------------------------------------------------
# Placing the labels
# ---------------------------------------------------------------------------


def _label_area(anchors: np.ndarray, along: np.ndarray, up: np.ndarray, width: float, height: float) -> np.ndarray:
    """Return points filling the label rectangle whose baseline-side centre is each of ``anchors``.

    ``anchors`` has shape (N, 2); the result (N, P, 2). ``along`` is the
    unit vector along the text, ``up`` the unit vector from the side facing
    the piece to the far side.
    """
    s = np.linspace(-width / 2, width / 2, max(2, math.ceil(width / _AREA_SAMPLE_PX) + 1))
    t = np.linspace(0.0, height, max(2, math.ceil(height / _AREA_SAMPLE_PX) + 1))
    offsets = (s[:, None, None] * along + t[None, :, None] * up).reshape(-1, 2)
    return anchors[:, None, :] + offsets[None, :, :]


def add_slope_labels(ax: Axes, labels: Sequence[SlopeLabel], config: PlotConfig, obstacles: Sequence[Bbox] = ()) -> list[Bbox]:
    """Write each piece's slope along it, where it covers the least; return the labels' display extents.

    ``ax`` is the chart's main axes (its twin has the same frame). The
    layout and both axes' limits must be final. ``obstacles`` (the -l
    labels) and each label already placed are avoided as other text; the
    shortest pieces, which have the least room, are labelled first.

    Candidates: the label's centre steps along the piece, keeping the label
    within the piece's length where it fits in it, and anywhere over the
    piece where it does not; and the label may stand up to ``_LIFT_STEPS``
    half heights further off, above the piece or below it. Each is scored
    by what it covers (a line's cell 1, a label's ``_LABEL_CELL_COST``), plus
    ``_LIFT_COST_CELLS`` per step off and ``_BELOW_COST_CELLS`` below; the
    lowest wins, then the one nearest the piece's middle. A label that would
    leave the axes is not a candidate.
    """
    figure = ax.figure
    renderer = figure.canvas.get_renderer()
    px_per_pt = figure.dpi / 72.0
    raster = Raster.from_axes(ax, collect_display_samples(ax, exclude=[label.piece for label in labels]))
    for obstacle in obstacles:
        raster.mark_box(obstacle, _LABEL_CELL_COST)
    frame = ax.get_window_extent(renderer).padded(-_FRAME_GAP_PT * config.font_scale * px_per_pt)
    halo_px = _HALO_PT * config.font_scale * px_per_pt / 2
    # From the piece's centre line to the text: half the piece's width, then the gap.
    offset_px = (_LINE_WIDTH_PT * config.line_scale / 2 + _LABEL_GAP_PT * config.font_scale) * px_per_pt

    def ends_px(label: SlopeLabel) -> np.ndarray:
        return label.piece.axes.transData.transform(np.asarray(label.piece.get_xydata(), dtype=float))

    extents: list[Bbox] = []
    for label in sorted(labels, key=lambda item: float(np.hypot(*np.diff(ends_px(item), axis=0)[0]))):
        start, end = ends_px(label)
        length = float(np.hypot(*(end - start)))
        along = (end - start) / length if length > 0 else np.array([1.0, 0.0])
        normal = np.array([-along[1], along[0]])  # upwards: the dates increase along the piece
        text = label.piece.axes.annotate(
            label.text, xy=(0, 0), xycoords="data", xytext=(0, 0), textcoords="offset points",
            fontsize=_LABEL_FS * config.font_scale, color=label.color, ha="center", va="bottom",
            rotation_mode="anchor", annotation_clip=False,
            path_effects=[patheffects.withStroke(linewidth=_HALO_PT * config.font_scale, foreground="white")],
        )
        # Measured before it is turned to lie along the piece: the text's own width and height.
        size = text.get_window_extent(renderer)
        text.set_rotation(math.degrees(math.atan2(along[1], along[0])))
        width, height = size.width + 2 * halo_px, size.height + 2 * halo_px

        if length > width:
            centres = np.arange(width / 2, length - width / 2 + 1e-9, _STEP_PX)
        else:
            centres = np.linspace(0.0, length, max(2, math.ceil(length / _STEP_PX) + 1))
        lifts = np.arange(_LIFT_STEPS + 1)
        centre_grid, lift_grid = (grid.ravel() for grid in np.meshgrid(centres, lifts, indexing="ij"))
        stand_off = offset_px + lift_grid * height / 2

        # Every candidate above the piece, then every one below it.
        sides = np.repeat([1.0, -1.0], len(centre_grid))
        centre_all, lift_all = np.tile(centre_grid, 2), np.tile(lift_grid, 2)
        stand_all = np.tile(stand_off, 2)
        anchors = start + centre_all[:, None] * along + (sides * stand_all)[:, None] * normal
        area = _label_area(anchors, along, normal, width, height)
        # Below the piece the label's far side is downwards.
        below = sides < 0
        area[below] = _label_area(anchors[below], along, -normal, width, height)
        inside = np.all(
            (area[..., 0] >= frame.x0) & (area[..., 0] <= frame.x1) & (area[..., 1] >= frame.y0) & (area[..., 1] <= frame.y1),
            axis=1,
        )
        if not inside.any():
            text.remove()  # the piece lies outside the axes (explicit --top/--bottom)
            continue
        cells, within = raster.cells_at(area.reshape(-1, 2))
        cost = np.zeros(len(cells))
        cost[within] = raster.cells[cells[within, 0] + raster.pad, cells[within, 1] + raster.pad]
        score = cost.reshape(area.shape[:2]).sum(axis=1) / _SAMPLES_PER_CELL + _LIFT_COST_CELLS * lift_all + _BELOW_COST_CELLS * below
        score[~inside] = np.inf
        best = int(np.lexsort((np.abs(centre_all - length / 2), score))[0])
        side = sides[best]
        point = start + centre_all[best] * along
        text.xy = tuple(label.piece.axes.transData.inverted().transform(point))
        shift_pt = side * stand_all[best] / px_per_pt * normal
        text.xyann = (float(shift_pt[0]), float(shift_pt[1]))
        text.set_va("bottom" if side > 0 else "top")
        raster.mark_points(area[best], _LABEL_CELL_COST)
        extents.append(text.get_window_extent(renderer).padded(halo_px))
    return extents
