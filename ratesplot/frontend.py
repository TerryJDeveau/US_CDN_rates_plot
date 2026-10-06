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
* each nation's own controls (``NATION_OPTIONS``, under its tab in the window
  and in its own sidebar section on the page), keyed "NATION:OPTION";
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

from .config import COMPONENT_SYNONYMS, EARLIEST_DATA_START, MIN_WINDOW_DAYS, NATIONS, Nation, PlotConfig
from .options import (
    GROUPS,
    GROUPS_CHOSEN_TOGETHER,
    OPTIONS,
    Kind,
    Option,
    by_name,
    choices_from_config,
    config_from_choices,
    format_dollar_bound,
    nation_choice,
    nation_view,
    options_in,
    options_of,
    per_nation,
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
# The options each nation has its own control for (batch 2, Terry 2026-10-06:
# the tick boxes of a nation's terms, and its axis limits, under its tab). The
# window and the page show none of them for both charts at once.
NATION_OPTIONS = tuple(option for option in OPTIONS if per_nation(option) and option.in_gui)


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


def nation_text(nation: Nation, option: Option, text: str) -> str:
    """Return a value for both charts as ``nation``'s own field shows it: its own terms and pairs only.

    Blank stays blank (the default); a text that does not parse is kept as
    it is, for the field to report.
    """
    if not text.strip() or option.fields[0] not in ("yield_terms", "mortgage_terms", "spread_pairs"):
        return text
    try:
        value = nation_view(nation, option, option.parse(text))
    except ValueError:
        return text
    return option.format((value,), PlotConfig()) if value else ""


def spread_to_nations(payloads: dict[str, str]) -> dict[str, str]:
    """Return ``payloads`` with each value for both charts moved into every nation's own field.

    Settings remembered before there were nations ("max": "8", "yields":
    "3m,2y,7y") become "cdn:max" and "us:max", and so on (``nation_text``);
    a nation's own field already there is kept.
    """
    moved = {name: text for name, text in payloads.items() if name not in {option.name for option in NATION_OPTIONS}}
    for option in NATION_OPTIONS:
        for nation in NATIONS:
            key = nation_choice(nation, option)
            if key in payloads:
                moved[key] = payloads[key]
            elif option.name in payloads:
                moved[key] = nation_text(nation, option, payloads[option.name])
    return moved


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
            remembered = {name: text for name, text in state.get("payloads", {}).items() if isinstance(text, str)}
            for name, text in spread_to_nations(remembered).items():
                if name in payloads:
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
        # A list taken out of a field (--no-yields:30y) has no control of its
        # own: the field's control shows what the command line left.
        for option in options_of(Kind.VALUE):
            if option.removes and option.name in cli_payloads:
                for shown in options_of(Kind.VALUE):
                    if shown.in_gui and shown.fields == option.fields and shown.name in cli_config_payloads:
                        payloads[shown.name] = cli_config_payloads[shown.name]
        # Each nation's own fields: what the command line gave for both charts
        # (--max:8, --no-y:30), or for that nation (--us:max:8), as the
        # resolved config has it for that nation.
        for option in NATION_OPTIONS:
            given = [other for other in options_of(Kind.VALUE) if other.fields == option.fields]
            for nation in NATIONS:
                key = nation_choice(nation, option)
                if any(other.name in cli_payloads or nation_choice(nation, other) in cli_payloads for other in given):
                    payloads[key] = cli_config_payloads[key]
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
            named = [option.name] + [nation_choice(nation, option) for nation in NATIONS]
            if option.turns_on in flags and any(name in cli_payloads for name in named):
                flags[option.turns_on] = cli_config_flags[option.turns_on]
        # Remembered Top/Bottom are in the remembered measure's units (dollars,
        # percent, dollars per person). If the command line chose a different
        # measure they no longer apply, unless it gave them too; the other
        # remembered settings are kept.
        if use_state:
            remembered = state.get("flags", {})
            measure = [option.name for option in options_in("units") if option.name in flags]
            if any(flags[name] != bool(remembered.get(name, False)) for name in measure):
                for bound in bound_keys():
                    if bound in payloads and bound not in cli_payloads and bound.split(":")[-1] not in cli_payloads:
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


def bound_keys() -> list[str]:
    """Return the keys of the right-axis limits, whose units the measure sets: each nation's Top and Bottom."""
    return [nation_choice(nation, by_name(bound)) for bound in ("top", "bottom") for nation in NATIONS]


def option_of(key: str) -> Option:
    """Return the option a choice's key names: "us:top" and "top" are both --top."""
    return by_name(key.split(":")[-1])


def nation_of(key: str) -> Nation | None:
    """Return the nation a choice's key is for ("us:top"), or None for both charts."""
    code = key.partition(":")[0] if ":" in key else ""
    return next((nation for nation in NATIONS if nation.key == code), None)


def level_letters(text: str) -> set[str]:
    """Return the levels a ``--debt:`` text names, synonyms resolved: ``"fs"`` -> ``{"f", "p"}``.

    Lenient, for ticking the level boxes: letters that are not levels are
    ignored here (the option's parser reports them when the chart is drawn).
    """
    return {COMPONENT_SYNONYMS.get(letter, letter) for letter in text.strip().lower()}


# ---------------------------------------------------------------------------
# Tick boxes for a list option (Option.editor == "choices")
# ---------------------------------------------------------------------------
# The option's text (its command-line value, "10y-2y,10y-3m") stays the one
# source of truth, as for the levels and the size: the boxes are a view of it
# and write it back. The window and the web page share these rules.


def choice_values(option: Option, text: str, nation: Nation | None = None) -> list[str]:
    """Return the list items ``text`` names, as the option writes them ("2", "10" -> ["2y", "10y"]).

    A blank text is the option's default; for a nation's own field, as far as
    its chart draws it (``nation_view``: Canada's mortgage terms by default
    are "5"). Lenient, for ticking the boxes: a text that does not parse
    names nothing here (the option's parser reports it when the chart is
    drawn).
    """
    default = PlotConfig()
    try:
        value = option.parse(text) if text.strip() else getattr(default, option.fields[0])
    except ValueError:
        return []
    if nation is not None and not text.strip():
        value = nation_view(nation, option, value)
    written = option.format((value,), default)
    return [item for item in written.split(",") if item]


def _offers(nation: Nation, option: Option, value: str) -> bool:
    """True when ``nation``'s chart can draw a box's item: a term it has, a spread of two of them."""
    if option.fields[0] == "mortgage_terms":
        return value in nation.mortgage_terms
    terms = value.split("-") if option.fields[0] == "spread_pairs" else [value]
    return all(term in nation.yield_terms for term in terms)


def choice_rows(option: Option, text: str, nation: Nation | None = None) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return the option's rows of boxes, ``(caption, [(value, box caption), ...])``.

    The offered rows (``Option.choices``), then a row "other" for any item
    ``text`` names that none of them offers (a spread typed on the command
    line), so nothing chosen is hidden. For a nation's own field, only the
    boxes its chart can draw, without the rows' captions ("U.S. only",
    "Canada"), which say whose they are.
    """
    rows = [(caption, list(boxes)) for caption, boxes in option.choices]
    if nation is not None:
        rows = [("", [box for box in boxes if _offers(nation, option, box[0])]) for _caption, boxes in rows]
        rows = [row for row in rows if row[1]]
    offered = {value for _caption, boxes in rows for value, _box in boxes}
    others = [(value, value.replace("-", "–")) for value in choice_values(option, text, nation) if value not in offered]
    return rows + ([("other", others)] if others else [])


def choices_text(option: Option, ticked: list[str]) -> str:
    """Return the option's text for the ticked items, written as the command line writes it.

    The yield terms in term order ("3m,2y,7y,10y", wherever 7y's box is), the
    spreads in box order (which sets their colours); "" when none is ticked.
    """
    joined = ",".join(ticked)
    return ",".join(choice_values(option, joined)) if joined else ""


def choice_refusal(option: Option) -> str:
    """Say why a list's last ticked box stays ticked, and which box draws none of it.

    An empty list would read as the default, so the last box stays; drawing
    none is the job of the curve's own box (Terry, 2026-10-06: "that is what
    --no-y is for"): the flag a list belongs to, or for the yield terms the
    yield curve's.
    """
    switch = by_name(option.turns_on or "yield")
    return f"At least one of the {option.label.lower()} stays ticked; to draw none, untick “{switch.label}”."


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
    """Two decimals at most, no trailing zeros: 2.5, 3.14, 4, -0.5 (and 0, never "-0")."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


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
    if config.has_left_axis_series:
        # Below zero too: a spread (--spreads) is negative where the curve inverts.
        low = max(-99.99, geometry.value_at(bottom_y, "left"))
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
