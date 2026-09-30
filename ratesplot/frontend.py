"""What the window and the web page share, without a GUI toolkit.

The desktop window (``gui``, tkinter) and the web page (``web``, Streamlit)
are two front ends over the same option table and the same drawing
(``plotting.build_figure``). This module holds the parts of a front end that
need neither toolkit, so that the web page never imports tkinter (a server
may not have it) and the two cannot drift apart:

* ``Choices``: what a user chose, as ``(payloads, flags)`` keyed by option
  name (see ``options``);
* the starting choices: defaults, remembered settings and the command line
  combined (``starting_choices``), and the window's remembered settings file;
* each option's help as the controls show it (``option_help_lines``);
* the drawn chart as PNG bytes (``png_bytes``) and the file name it is saved
  under (``saved_png_name``); the tab titles;
* zoom and pan: where the axes sit in a chart image (``ChartGeometry``) and
  the field texts a zoom to a box, a pan or a change of the dates' scale
  gives (``zoom_updates``, ``pan_updates``, ``scale_dates_updates``). The
  window's mouse and the web page's drag use the same rules; the page's
  buttons move the dates by ``move_dates_updates``.
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure

from .config import COMPONENT_SYNONYMS, EARLIEST_DATA_START, MIN_WINDOW_DAYS, PlotConfig
from .options import (
    GROUPS,
    GROUPS_CHOSEN_TOGETHER,
    OPTIONS,
    Kind,
    Option,
    choices_from_config,
    config_from_choices,
    format_dollar_bound,
    options_in,
    options_of,
)

# The start date written out before it became automatic (Terry, 2026-09-29):
# a remembered Start with this text was the old default, not a choice, and is
# read as blank, the automatic start.
_FORMER_DEFAULT_START = "1966-01-01"
# Same PNG metadata as tools/verify_charts.py, so saved files hash identically.
PNG_METADATA = {"Software": None}
TAB_TITLES = {"cdn": "Canada", "us": "United States"}
# The "By level" check boxes: letter and caption (one control for both countries,
# since --debt: and --interest: take the same letters).
LEVEL_BOXES = (("f", "Federal"), ("n", "Non-federal"), ("p", "Provincial / state"), ("m", "Municipal / local"))
# The format of the window's remembered-settings file; a file of another version is ignored.
STATE_VERSION = 1

Choices = tuple[dict[str, str], dict[str, bool]]


# ---------------------------------------------------------------------------
# Remembered settings (the window's; the web page keeps a chart's choices in
# its address instead) and the starting choices
# ---------------------------------------------------------------------------


def state_path() -> Path:
    """Return where the window's settings are remembered: ``%APPDATA%\\ratesplot`` on Windows."""
    base = os.environ.get("APPDATA")
    return (Path(base) if base else Path.home() / ".config") / "ratesplot" / "gui_state.json"


def load_state() -> tuple[dict, str | None]:
    """Return the remembered state (empty if none) and a note to log if it could not be read."""
    path = state_path()
    if not path.exists():
        return {}, None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, f"Remembered settings in {path} could not be read ({exc}); starting from defaults."
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        return {}, f"Remembered settings in {path} are from another version; starting from defaults."
    return state, None


def save_state(state: dict) -> str | None:
    """Write ``state`` atomically (temporary file, then replace); return an error message or None."""
    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        return f"Settings could not be remembered ({exc})."
    return None


def starting_choices(
    config: PlotConfig, cli_payloads: dict[str, str], cli_flags: dict[str, bool], state: dict
) -> tuple[Choices, list[str]]:
    """Combine defaults, remembered choices and the command line; return the choices and notes to log.

    ``config`` is what the command line alone describes; ``cli_payloads`` and
    ``cli_flags`` are the options it named. If the combination is not valid
    (e.g. a remembered start date after a new command-line end date), the
    remembered choices are dropped and the reason logged.
    """
    notes: list[str] = []
    cli_config_payloads, cli_config_flags = choices_from_config(config)
    if state.get("payloads", {}).get("start") == _FORMER_DEFAULT_START:
        state = {**state, "payloads": {**state["payloads"], "start": ""}}
        notes.append(f"The remembered Start {_FORMER_DEFAULT_START} was the former default: now blank, the automatic start.")

    def combine(use_state: bool) -> Choices:
        payloads, flags = choices_from_config(PlotConfig())
        if use_state:
            for name, text in state.get("payloads", {}).items():
                if name in payloads and isinstance(text, str):
                    payloads[name] = text
            for name, value in state.get("flags", {}).items():
                if name in flags and isinstance(value, bool):
                    flags[name] = value
        for name, text in cli_payloads.items():
            if name in payloads:
                payloads[name] = text
        # The levels (--debt:LETTERS, --interest:LETTERS) have one control, the
        # debt one; and giving them names their curves, as --debt would.
        levels_given = any(name in cli_payloads for name in ("debt-parts", "interest-parts"))
        if levels_given:
            payloads["debt-parts"] = cli_config_payloads["debt-parts"]
        # Curves, and countries, are overridden as a group: on the command line
        # "--gdp" means "GDP only", which the resolved config already reflects.
        for group_name in GROUPS_CHOSEN_TOGETHER:
            group = [option.name for option in options_in(group_name) if option.name in flags]
            if any(name in cli_flags for name in group) or (group_name == "curves" and levels_given):
                for name in group:
                    flags[name] = cli_config_flags[name]
        for option in OPTIONS:
            if option.group in GROUPS_CHOSEN_TOGETHER or option.kind is Kind.VALUE:
                continue
            if option.name in flags and option.name in cli_flags:
                flags[option.name] = cli_flags[option.name]
        # A value given on the command line that turns its flag on (--reg:TOL)
        # does so over the remembered flag too.
        for option in options_of(Kind.VALUE):
            if option.turns_on in flags and option.name in cli_payloads:
                flags[option.turns_on] = cli_config_flags[option.turns_on]
        # Remembered Top/Bottom are in the remembered measure's units (dollars,
        # percent, dollars per person). If the command line chose a different
        # measure they no longer apply, unless it gave them too; the other
        # remembered settings are kept.
        if use_state:
            remembered = state.get("flags", {})
            measure = [option.name for option in options_in("units") if option.name in flags]
            if any(flags[name] != bool(remembered.get(name, False)) for name in measure):
                for bound in ("top", "bottom"):
                    if bound in payloads and bound not in cli_payloads:
                        payloads[bound] = ""
        return payloads, flags

    choices = combine(use_state=bool(state))
    if state:
        try:
            config_from_choices({k: v for k, v in choices[0].items() if v.strip()}, choices[1])
            notes.append(f"Restored the settings remembered in {state_path()}.")
        except ValueError as exc:
            notes.append(f"Remembered settings not used ({exc}); starting from the command line and defaults.")
            choices = combine(use_state=False)
    return choices, notes


# ---------------------------------------------------------------------------
# Levels, help text, and the drawn chart as a file
# ---------------------------------------------------------------------------


def level_letters(text: str) -> set[str]:
    """Return the levels a ``--debt:`` text names, synonyms resolved: ``"fs"`` -> ``{"f", "p"}``.

    Lenient, for ticking the level boxes: letters that are not levels are
    ignored here (the option's parser reports them when the chart is drawn).
    """
    return {COMPONENT_SYNONYMS.get(letter, letter) for letter in text.strip().lower()}


def option_help_lines(option: Option) -> tuple[str, str, str]:
    """Return an option's group notes, its help text and its command-line spelling."""
    return (
        f"{' '.join(GROUPS[option.group].split())}",
        f"{option.label}: {' '.join(option.help.split())}",
        f"Command line: {option.usage}",
    )


def png_bytes(figure: Figure) -> bytes:
    """Return ``figure`` as PNG bytes at its own DPI, exactly as the command line saves it."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=figure.dpi, metadata=PNG_METADATA)
    return buffer.getvalue()


def saved_png_name(key: str, drawn: PlotConfig | None) -> str:
    """Return the file name a chart is saved under: ``cdn_rates_1981-09-01_2026-09-29.png``.

    Named by the dates drawn (an automatic start as found); without them
    (nothing drawn yet) just ``cdn_rates.png``.
    """
    dates = "" if drawn is None else f"_{drawn.start:%Y-%m-%d}_{drawn.end:%Y-%m-%d}"
    return f"{key}_rates{dates}.png"


# ---------------------------------------------------------------------------
# Zoom and pan: chart geometry (image pixels <-> data values) and the fields' texts
# ---------------------------------------------------------------------------

# A drag shorter than this (image pixels) on an axis does not change that axis.
_MIN_DRAG_PX = 8
# The command line insists on at least seven days between start and end.
_MIN_SPAN_DAYS = MIN_WINDOW_DAYS


@dataclass(frozen=True)
class ChartGeometry:
    """Where the axes sit in a rendered chart image, and their limits.

    ``box`` is (left, top, right, bottom) in image pixels, origin top-left.
    x limits are matplotlib date numbers. "left"/"right" are the two y axes:
    the yield axis and its twin carrying the macro series (dollars, or % of GDP).
    ``window`` is the requested Start and End, also as date numbers: the axis
    is wider (a small pad each side, and with -l room for the value labels).
    """

    box: tuple[float, float, float, float]
    xlim: tuple[float, float]
    window: tuple[float, float]
    left_ylim: tuple[float, float]
    right_ylim: tuple[float, float]
    left_log: bool
    right_log: bool

    @classmethod
    def from_figure(cls, figure: Figure, config: PlotConfig) -> ChartGeometry | None:
        """Measure a figure drawn for ``config`` (axes[0] is the yield axis, axes[1] its twin)."""
        if len(figure.axes) < 2:
            return None
        left, right = figure.axes[0], figure.axes[1]
        height = figure.bbox.height
        extent = left.get_window_extent()
        return cls(
            box=(extent.x0, height - extent.y1, extent.x1, height - extent.y0),
            xlim=tuple(left.get_xlim()),
            window=(float(mdates.date2num(config.start)), float(mdates.date2num(config.end))),
            left_ylim=tuple(left.get_ylim()),
            right_ylim=tuple(right.get_ylim()),
            left_log=left.get_yscale() == "log",
            right_log=right.get_yscale() == "log",
        )

    def date_at(self, x: float) -> pd.Timestamp:
        """Return the date at image x (not clamped: pans may run past the axes)."""
        left, _top, right, _bottom = self.box
        low, high = self.xlim
        number = low + (x - left) / (right - left) * (high - low)
        return pd.Timestamp(mdates.num2date(number).replace(tzinfo=None)).normalize()

    def window_x(self) -> tuple[float, float]:
        """Return the image x of the requested Start and End (inside the axes, which are wider)."""
        left, _top, right, _bottom = self.box
        low, high = self.xlim
        return tuple(left + (number - low) / (high - low) * (right - left) for number in self.window)

    def value_at(self, y: float, side: str) -> float:
        """Return the value at image y on the ``"left"`` or ``"right"`` y axis."""
        _left, top, _right, bottom = self.box
        low, high = self.left_ylim if side == "left" else self.right_ylim
        log = self.left_log if side == "left" else self.right_log
        fraction = (bottom - y) / (bottom - top)
        if log:
            return 10 ** (math.log10(low) + fraction * (math.log10(high) - math.log10(low)))
        return low + fraction * (high - low)


def _format_percent(value: float) -> str:
    """Two decimals at most, no trailing zeros: 2.5, 3.14, 4."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_dollars(value: float) -> str:
    """Three significant figures with a k/m/b/t suffix: 1.25t, 380b."""
    rounded = float(f"{value:.3g}")
    return format_dollar_bound((rounded,))


def _format_relative(value: float) -> str:
    """A percentage of GDP (-r), three significant figures: 0.85%, 12.5%, 140%."""
    return f"{float(f'{value:.3g}'):g}%"


def _date_range_texts(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, str]:
    """Return Start/End texts for a zoom or pan.

    The span is widened about its middle to the seven-day minimum if needed.
    There is no data after today, so the end is capped at today, and an end
    of today is written blank: the "today" default then keeps moving, as it
    would if the user had never touched the field.
    """
    today = pd.Timestamp.today().normalize()
    if end < start:
        start, end = end, start
    if (end - start).days < _MIN_SPAN_DAYS:
        middle = start + (end - start) / 2
        start = (middle - pd.Timedelta(days=_MIN_SPAN_DAYS / 2)).normalize()
        end = start + pd.Timedelta(days=_MIN_SPAN_DAYS)
    if end >= today:
        end = today
        start = min(start, today - pd.Timedelta(days=_MIN_SPAN_DAYS))
    return {"start": f"{start:%Y-%m-%d}", "end": "" if end == today else f"{end:%Y-%m-%d}"}


def axis_updates(geometry: ChartGeometry, config: PlotConfig | None, top_y: float, bottom_y: float) -> dict[str, str]:
    """Return field texts pinning the drawn y axes to image rows ``top_y``..``bottom_y``.

    ``config`` is the one the chart was drawn for (None: nothing drawn, no texts).
    """
    updates: dict[str, str] = {}
    if config is None:
        return updates
    if config.include_yield:
        low = max(0.0, geometry.value_at(bottom_y, "left"))
        high = min(99.99, geometry.value_at(top_y, "left"))
        if high - low >= 0.01:
            updates.update({"min": _format_percent(low), "max": _format_percent(high)})
    if config.has_macro_series:
        low, high = geometry.value_at(bottom_y, "right"), geometry.value_at(top_y, "right")
        if 0 < low < high:
            bound_text = _format_relative if config.relative else _format_dollars
            updates.update({"bottom": bound_text(low), "top": bound_text(high)})
    return updates


def zoom_updates(
    geometry: ChartGeometry, config: PlotConfig | None, x0: float, y0: float, x1: float, y1: float
) -> dict[str, str]:
    """Return field texts that zoom to the box (x0, y0)-(x1, y1) in image pixels (x0 <= x1, y0 <= y1).

    Its width sets Start and End, its height the yield Min and Max and the
    right axis's Top and Bottom, of whichever axes are drawn. A side shorter
    than ``_MIN_DRAG_PX`` leaves its axis alone, so a flat box zooms the dates only.
    """
    updates: dict[str, str] = {}
    if x1 - x0 >= _MIN_DRAG_PX:
        updates.update(_date_range_texts(geometry.date_at(x0), geometry.date_at(x1)))
    if y1 - y0 >= _MIN_DRAG_PX:
        updates.update(axis_updates(geometry, config, y0, y1))
    return updates


def pan_updates(geometry: ChartGeometry, config: PlotConfig | None, dx: float, dy: float) -> dict[str, str]:
    """Return field texts that shift the view so the point under the pointer moves by (dx, dy) image pixels."""
    _left, top, _right, bottom = geometry.box
    updates: dict[str, str] = {}
    if abs(dx) >= _MIN_DRAG_PX:
        # The requested window moves, not the whole axis: its pad (and the
        # -l label room) would otherwise be added to the window at every pan.
        start_x, end_x = geometry.window_x()
        updates.update(_date_range_texts(geometry.date_at(start_x - dx), geometry.date_at(end_x - dx)))
    if abs(dy) >= _MIN_DRAG_PX:
        updates.update(axis_updates(geometry, config, top - dy, bottom - dy))
    return updates


def scale_dates_updates(geometry: ChartGeometry, factor: float, anchor_x: float) -> dict[str, str]:
    """Return Start and End texts for the requested window scaled by ``factor`` about image x ``anchor_x``.

    A factor below 1 zooms in, above 1 out; the date under ``anchor_x`` stays put.
    """
    start_x, end_x = geometry.window_x()  # the requested window, as for a pan
    new_left = anchor_x - (anchor_x - start_x) * factor
    new_right = anchor_x + (end_x - anchor_x) * factor
    return _date_range_texts(geometry.date_at(new_left), geometry.date_at(new_right))


def move_dates_updates(start: pd.Timestamp, end: pd.Timestamp, *, shift: float = 0.0, scale: float = 1.0) -> dict[str, str]:
    """Return Start and End texts for the window ``start``..``end`` moved or rescaled (the web page's buttons).

    ``shift`` moves the window by that fraction of its span (negative:
    earlier); ``scale`` multiplies the span about its middle (2: zoom out).
    The window is kept between the first data (``EARLIEST_DATA_START``) and
    today by moving it back inside rather than cutting it, so "Later" on a
    window that nearly reaches today keeps its span, where a pan by pixels
    would shorten it (``_date_range_texts`` caps the end at today). Whole
    days throughout: half a window of an odd number of days would otherwise
    lose its half day to the date's format, and the span a day per step.
    """
    today = pd.Timestamp.today().normalize()
    days = (end - start).days
    # Never longer than all the data (a pandas Timedelta also ends at 292 years).
    new_days = min(round(days * scale), (today - EARLIEST_DATA_START).days)
    new_start = start + pd.Timedelta(days=round(days * shift) - (new_days - days) // 2)
    new_end = new_start + pd.Timedelta(days=new_days)
    if new_end > today:
        new_start, new_end = new_start - (new_end - today), today
    if new_start < EARLIEST_DATA_START:
        new_start, new_end = EARLIEST_DATA_START, min(today, new_end + (EARLIEST_DATA_START - new_start))
    return _date_range_texts(new_start, new_end)
