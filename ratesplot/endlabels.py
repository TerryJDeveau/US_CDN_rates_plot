"""Value labels at the right-hand end of every drawn line (-l / --label).

Each curve's last plotted value is printed just right of where the curve
ends, in the curve's colour: a yield as ``4.12%``, a right-axis value as that
axis's tick labels write it (``$36.2T``, or ``123%`` under -r). A curve that
stops short of the right end of the date axis (federal debt before 1933, say)
is labelled where it stops.

The labels go in after the layout and the requested date limits are final,
and before the legend, which then avoids them as it avoids the lines (see
``legend.finish_legend_and_title``):

1. Every label is measured, at the legend's font size to begin with.
2. The right end of the date axis is moved out just far enough for every
   label to fit inside the axes. Curves that stop short of the right end need
   less room, and none at all once their labels fit where they stop.
3. Labels that would overlap are spread vertically: within each group of
   labels whose horizontal extents overlap, the centres move as little as
   possible (least squares) while staying in order, apart and inside the axes.
   Labels that end up touching form a *stack*.
4. A stack in which some label sits more than half its own height from its
   curve's end is crowded: its font is reduced 1 pt at a time, down to the
   legend's smallest size, until none does. Other labels keep the legend's
   size, so only the crowded values are smaller. A stack still crowded at the
   smallest size moves a little further from the curve ends, and each of its
   labels is joined to its curve's end by a thin leader line in its colour
   (all of them, so the column reads alike).

Sizes are multiplied by ``PlotConfig.font_scale`` like every other text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import same_color
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from .config import LEGEND_FS, PlotConfig

# Turns a curve's last value into its label text (see plotting.draw_country).
ValueFormatter = Callable[[float], str]

# Font sizes for a crowded stack, in points below LEGEND_FS (before scaling):
# the legend's own size first, down to the smallest the legend itself goes to.
_FONT_SIZE_REDUCTIONS_PT = (0, 1, 2, 3, 4)
# A label whose centre is within this many of its own heights of its curve's
# end still reads as attached to it. Beyond that its stack is crowded.
_MAX_SHIFT_HEIGHTS = 0.5
# Gap between a curve's end and its label's background, and between any label
# and the axes frame, in points. A stack with leader lines stands further off,
# so the leaders slope rather than look like the curves carrying on.
_GAP_PT = 4.0
_LEADER_GAP_PT = 16.0
_FRAME_GAP_PT = 4.0
# Leader lines, in points: width (before PlotConfig.line_scale), and the space
# left between a leader and its curve's end.
_LEADER_WIDTH_PT = 0.8
_LEADER_SHRINK_PT = 2.0
# The white background behind each label, so grid lines (and, for a curve that
# stops early, other curves) do not run through the digits. Its padding is a
# fraction of the font size, as matplotlib's box styles take it.
_BOX_PAD = 0.15
_BOX_ALPHA = 0.85
# Labels this close (pixels) after spreading are touching: the same stack.
_TOUCH_PX = 0.5


@dataclass(frozen=True)
class _Label:
    """One curve's label before it is placed."""

    text: str
    color: object  # the curve's colour, in any form matplotlib accepts
    axes: Axes  # the curve's axes (the yield axis or its twin)
    end: tuple[float, float]  # the curve's last point, in its axes' data coordinates


@dataclass(frozen=True)
class _Layout:
    """All labels placed, each at its own font size and gap."""

    right: float  # right date limit of the axes (matplotlib date number)
    ends: list[tuple[float, float]]  # display position of each curve's end
    centres: list[float]  # display y of each label's centre
    heights: list[float]  # each label's height with its background, in pixels
    stacks: list[list[int]]  # indices of labels that touch one another, bottom to top

    def shift_ok(self, index: int) -> bool:
        """True when label ``index`` sits within ``_MAX_SHIFT_HEIGHTS`` of its own height of its curve's end."""
        return abs(self.centres[index] - self.ends[index][1]) <= _MAX_SHIFT_HEIGHTS * self.heights[index]

    def crowded(self, stack: Sequence[int]) -> bool:
        """True when some label of ``stack`` has had to move too far from its curve's end."""
        return not all(self.shift_ok(index) for index in stack)


# ---------------------------------------------------------------------------
# Where each curve ends
# ---------------------------------------------------------------------------


def _curve_end(line: Line2D) -> tuple[float, float] | None:
    """Return the rightmost finite point of the curve ``line`` is the legend entry for, or None.

    A curve can be drawn in pieces: the Canadian yields are monthly steps to
    2000 and a daily line after it, and with --cur a right-axis curve's
    projection to today follows its data. The later pieces have the same
    colour and dashes and are hidden from the legend (their labels start with
    "_", matplotlib's mark for that). Such pieces on the same axes are part of
    the curve, so its end is where the last of them ends. The dashes matter:
    federal debt before 1933 is dotted black and the aggregate solid black.
    """
    pieces = [line] + [
        other
        for other in line.axes.get_lines()
        if other is not line
        and other.get_label().startswith("_")
        and same_color(other.get_color(), line.get_color())
        and other.get_linestyle() == line.get_linestyle()
    ]
    end: tuple[float, float] | None = None
    for piece in pieces:
        xy = np.asarray(piece.get_xydata(), dtype=float)  # dates already converted to numbers
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            continue
        last = len(xy) - 1 - int(xy[::-1, 0].argmax())  # the last of any points sharing the largest date
        if end is None or xy[last, 0] > end[0]:
            end = (float(xy[last, 0]), float(xy[last, 1]))
    return end


def _in_view(axes: Axes, value: float) -> bool:
    """True when ``value`` lies within the y limits of ``axes`` (explicit --top/--bottom can cut a curve off)."""
    low, high = sorted(axes.get_ylim())
    return low <= value <= high and (axes.get_yscale() != "log" or value > 0)


# ---------------------------------------------------------------------------
# Measuring, widening the date axis, spreading
# ---------------------------------------------------------------------------


def _text_size_px(figure: Figure, renderer: RendererBase, text: str, fontsize: float) -> tuple[float, float]:
    """Return the width and height in pixels of ``text`` drawn at ``fontsize`` points."""
    probe = figure.text(0, 0, text, fontsize=fontsize)
    try:
        extent = probe.get_window_extent(renderer)
    finally:
        probe.remove()
    return extent.width, extent.height


def _right_limit(ends: Sequence[float], room_px: Sequence[float], left: float, right: float, width_px: float) -> float:
    """Return the smallest right date limit, not below ``right``, at which every label fits inside the axes.

    With the date limits ``left``..``R`` a curve ending at date ``e`` is drawn
    ``(e - left) / (R - left)`` of the way across the axes' ``width_px``, and its
    label needs ``room`` pixels to the right of that. So
    ``R >= left + (e - left) / (1 - room / width_px)``. A label wider than the
    axes (not possible at the minimum canvas size) is left out of the bound.
    """
    for end, room in zip(ends, room_px):
        fraction = 1.0 - room / width_px
        if fraction > 0:
            right = max(right, left + (end - left) / fraction)
    return right


def _overlapping_groups(lefts: Sequence[float], rights: Sequence[float]) -> list[list[int]]:
    """Group label indices whose horizontal extents overlap, directly or through others in the group."""
    groups: list[list[int]] = []
    reach = -np.inf  # right edge of the group being built
    for index in sorted(range(len(lefts)), key=lambda i: lefts[i]):
        if groups and lefts[index] < reach:
            groups[-1].append(index)
            reach = max(reach, rights[index])
        else:
            groups.append([index])
            reach = rights[index]
    return groups


def _spread(desired: Sequence[float], heights: Sequence[float], low: float, high: float) -> list[float]:
    """Return label centres as near ``desired`` as possible with no two overlapping, all within ``low``..``high``.

    ``desired`` is in ascending order (display y, upwards) and the order is
    kept. Writing each centre as ``p[i] = q[i] + offset[i]``, where
    ``offset[i]`` is the height of the stack below label i when the labels
    touch, "no overlap" becomes simply "``q`` never decreases". The
    least-squares fit is then the isotonic regression of ``desired - offset``
    (pool adjacent violators), and keeping the stack inside the axes clips
    that fit to one interval, which for an isotonic fit gives the constrained
    optimum because the bound is the same for every ``q``.
    """
    offsets = [0.0]
    for below, above in zip(heights, heights[1:]):
        offsets.append(offsets[-1] + (below + above) / 2)

    blocks: list[tuple[float, int]] = []  # (mean, count) of each pooled run
    for target in (value - offset for value, offset in zip(desired, offsets)):
        mean, count = target, 1
        while blocks and blocks[-1][0] > mean:
            previous_mean, previous_count = blocks.pop()
            mean = (previous_mean * previous_count + mean * count) / (previous_count + count)
            count += previous_count
        blocks.append((mean, count))
    fitted = [mean for mean, count in blocks for _ in range(count)]

    lowest = low + heights[0] / 2  # q[0] is the bottom label's centre
    highest = high - heights[-1] / 2 - offsets[-1]
    if lowest > highest:
        # Taller than the axes: centre the stack, so it overhangs both ends alike.
        fitted = [(lowest + highest) / 2] * len(fitted)
    else:
        fitted = [min(max(q, lowest), highest) for q in fitted]
    return [q + offset for q, offset in zip(fitted, offsets)]


def _lay_out(
    ax: Axes,
    labels: Sequence[_Label],
    fontsizes: Sequence[float],
    gaps_pt: Sequence[float],
    config: PlotConfig,
    renderer: RendererBase,
    left: float,
    right: float,
) -> _Layout:
    """Measure the labels at their font sizes, widen the date axis for them, and spread them apart.

    ``fontsizes`` are scaled points; ``gaps_pt`` (before scaling) separate
    each label from its curve's end. ``left``..``right`` are the requested
    date limits. Leaves the axes' date limits at the returned layout's ``right``.
    """
    figure = ax.figure
    px_per_pt = figure.dpi / 72.0
    frame_px = _FRAME_GAP_PT * config.font_scale * px_per_pt
    gaps_px = [gap * config.font_scale * px_per_pt for gap in gaps_pt]
    widths: list[float] = []
    heights: list[float] = []
    for label, fontsize in zip(labels, fontsizes):
        width, height = _text_size_px(figure, renderer, label.text, fontsize)
        pad_px = _BOX_PAD * fontsize * px_per_pt
        widths.append(width + 2 * pad_px)
        heights.append(height + 2 * pad_px)

    box = ax.get_window_extent(renderer)
    rooms = [gap + width + frame_px for gap, width in zip(gaps_px, widths)]
    right = _right_limit([label.end[0] for label in labels], rooms, left, right, box.width)
    ax.set_xlim(left, right)

    # Display positions under the new limits (both axes share them).
    ends = [tuple(label.axes.transData.transform(label.end)) for label in labels]
    lefts = [x + gap for (x, _), gap in zip(ends, gaps_px)]
    centres = [y for _, y in ends]
    stacks: list[list[int]] = []
    for group in _overlapping_groups(lefts, [start + width for start, width in zip(lefts, widths)]):
        order = sorted(group, key=lambda i: ends[i][1])
        placed = _spread([ends[i][1] for i in order], [heights[i] for i in order], box.y0 + frame_px, box.y1 - frame_px)
        for index, centre in zip(order, placed):
            centres[index] = centre
        # Split the group where neighbours do not touch.
        stacks.append([order[0]])
        for below, above in zip(order, order[1:]):
            clearance = centres[above] - centres[below] - (heights[above] + heights[below]) / 2
            if clearance <= _TOUCH_PX:
                stacks[-1].append(above)
            else:
                stacks.append([above])
    return _Layout(right, ends, centres, heights, stacks)


# ---------------------------------------------------------------------------
# The labels
# ---------------------------------------------------------------------------


def add_end_labels(ax: Axes, curves: Sequence[tuple[Line2D, ValueFormatter]], config: PlotConfig) -> list[Bbox]:
    """Label each curve's end with its last value, widening the date axis to fit; see the module docstring.

    ``ax`` is the axes whose date limits are the chart's (the yield axis; its
    twin shares them), already set to the requested window. ``curves`` pairs
    each drawn curve (its legend line) with the formatter for its axis. A curve
    whose end lies outside its axis's limits is not labelled. Returns the
    labels' display extents, for the legend placement to avoid.
    """
    labels: list[_Label] = []
    for line, format_value in curves:
        end = _curve_end(line)
        if end is not None and _in_view(line.axes, end[1]):
            labels.append(_Label(format_value(end[1]), line.get_color(), line.axes, end))
    if not labels:
        return []

    figure = ax.figure
    renderer = figure.canvas.get_renderer()
    left, right = ax.get_xlim()
    fontsizes = [LEGEND_FS * config.font_scale] * len(labels)
    gaps_pt = [_GAP_PT] * len(labels)

    def lay_out() -> _Layout:
        return _lay_out(ax, labels, fontsizes, gaps_pt, config, renderer, left, right)

    # Stacks are found at the legend's size and keep one size each, so the
    # labels of a stack match even if a smaller font lets them part.
    layout = lay_out()
    crowded = [stack for stack in layout.stacks if layout.crowded(stack)]
    for reduction in _FONT_SIZE_REDUCTIONS_PT[1:]:
        if not crowded:
            break
        for stack in crowded:
            for index in stack:
                fontsizes[index] = (LEGEND_FS - reduction) * config.font_scale
        layout = lay_out()
        crowded = [stack for stack in crowded if layout.crowded(stack)]
    # Still crowded at the smallest size: stand off, with leader lines.
    with_leaders = {index for stack in crowded for index in stack}
    if with_leaders:
        for index in with_leaders:
            gaps_pt[index] = _LEADER_GAP_PT
        layout = lay_out()
    # The date limits are those of the last layout, which is the one drawn.

    px_per_pt = figure.dpi / 72.0
    extents: list[Bbox] = []
    for index, label in enumerate(labels):
        pad_pt = _BOX_PAD * fontsizes[index]
        shift_pt = (layout.centres[index] - layout.ends[index][1]) / px_per_pt
        leader = None
        if index in with_leaders:
            # From the label's left edge (clipped to its background) to just short of the curve's end.
            leader = {
                "arrowstyle": "-",
                "color": label.color,
                "linewidth": _LEADER_WIDTH_PT * config.line_scale,
                "relpos": (0.0, 0.5),
                "shrinkA": 0.0,
                "shrinkB": _LEADER_SHRINK_PT * config.font_scale,
            }
        annotation = label.axes.annotate(
            label.text,
            xy=label.end,
            xycoords="data",
            # The text starts one background pad inside the background's left edge.
            xytext=(gaps_pt[index] * config.font_scale + pad_pt, shift_pt),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=fontsizes[index],
            color=label.color,
            bbox={"boxstyle": f"square,pad={_BOX_PAD}", "facecolor": "white", "edgecolor": "none", "alpha": _BOX_ALPHA},
            arrowprops=leader,
            annotation_clip=False,
        )
        # The extent covers the text and any leader; the background adds its pad.
        extents.append(annotation.get_window_extent(renderer).padded(pad_pt * px_per_pt))
    return extents
