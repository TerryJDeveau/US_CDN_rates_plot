"""The web page: the charts in a browser, for anyone with the link (Streamlit).

Run it with ``streamlit run streamlit_app.py`` from the project root; on
Streamlit Community Cloud the same file is the entry point.

What it is
    A third front end beside the command line and the window, over the same
    option table and the same drawing. Every chart is
    ``plotting.build_figure`` rendered to PNG (``frontend.png_bytes``): on the
    same machine, byte for byte what the window shows and saves. The controls
    are generated from ``options.OPTIONS`` as the window's are (options with
    ``in_gui=False`` are left out), grouped as in ``--help``, and follow the
    window's rules: at least one country stays ticked; one measure at a time,
    and changing it clears Top and Bottom; GDP greyed out under -r; each
    nation's own choices (``frontend.NATION_OPTIONS``: its yield terms,
    spreads and mortgage terms, only those its chart can draw, and its axis
    limits) in a sidebar section of its own; a value
    that belongs to a flag (the regression tolerance) greyed while the flag
    is off, and ignored then. A new option in the table appears here with no
    change to this module.

The chart's address
    The page's address carries the chart's command line: ``?chart=-c+-r+--reg``
    holds the same tokens ``python ratesplot.py -c -r --reg`` takes
    (``options.command_line_tokens``). Opening such a link draws that chart;
    after each drawing the address is updated, so a chart can be bookmarked
    or sent to someone. An address the program cannot read is reported, and
    the defaults are drawn instead. This replaces the window's remembered
    settings.

Visitors
    Streamlit runs each visitor's page in its own thread, all in one process.
    - Downloads are shared by every visitor and kept for
      ``DOWNLOAD_MAX_AGE_SECONDS`` (``http.enable_download_cache``); the
      intraday quotes are fetched for every drawing, as in the window.
    - Charts are drawn one at a time (``_DRAWING_LOCK``). Matplotlib does not
      promise that drawing in several threads at once is safe, and a free
      server has two processors; a visitor who asks during another's drawing
      waits for it, a few seconds.
    - What a drawing prints (progress, warnings, the start found) goes to
      that visitor's log only (``console``).
    - "Today", the default end, is the markets' date (Toronto) on a server,
      whatever the server's own time zone (``_use_market_time_zone``).

Zoom and dates
    As in the window, the mouse and the buttons write the fields a visitor
    would type in, so the fields stay the single source of truth (the
    address, the command line and the downloads all follow). Drag a box on
    a chart to zoom to it: its width sets Start and End, its height the
    yield Min and Max and the right axis's Top and Bottom (the window's
    rules, ``frontend.zoom_updates``). The chart is a small in-page component
    (``_chart_component``) because ``st.image`` reports no mouse events.
    "◀ Earlier" and "Later ▶" move the dates by half the window, "Zoom out"
    doubles their span (``frontend.move_dates_updates``: between 1680 and
    today, keeping the span). "Back" undoes one zoom, move or Reset;
    "Unzoom" returns to the charts before the first. The axis limits are
    each nation's own, so zooming one chart's axes leaves the other's; the
    dates are shared, as on the command line.

Dates
    Beside each date field a 📅 button opens Streamlit's calendar (between
    1680 and today) and a "Blank (default)" button, as the window's does:
    picking a day writes ``YYYY-MM-DD`` in the field, and typing ``YYYY`` or
    ``YYYY-MM`` still works (``_date_control``).

Size
    Width and height boxes and a "Preserve aspect ratio" lock, as in the
    window: with the lock on, changing one box sets the other, so a chart
    whose shape suits can be drawn larger or smaller by typing one number
    (``_size_control``). The ``WxH`` text behind them stays the source of truth.

Not here (the window has them)
    The wheel's zoom and the right-drag pan (the buttons move the dates),
    and remembered settings (the address keeps the chart instead).
"""

from __future__ import annotations

import datetime
import functools
import os
import struct
import threading
import time
import traceback
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # before pyplot is imported (by plotting): a server has no display

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from . import http  # noqa: E402
from .cli import sort_tokens  # noqa: E402
from .config import (  # noqa: E402
    COMPONENT_LETTERS,
    DEFAULT_REGRESSION_TOLERANCE_PCT,
    EARLIEST_DATA_START,
    MARKET_TIMEZONE,
    NATIONS,
    Nation,
    PlotConfig,
)
from .console import this_thread_output_to  # noqa: E402
from .frontend import (  # noqa: E402
    LEVEL_BOXES,
    NATION_OPTIONS,
    TAB_TITLES,
    ChartGeometry,
    Choices,
    bound_keys,
    choice_refusal,
    choice_rows,
    choice_values,
    choices_text,
    level_letters,
    move_dates_updates,
    option_help_lines,
    option_of,
    png_bytes,
    saved_png_name,
    starting_choices,
    zoom_updates,
)
from .options import (  # noqa: E402
    GROUPS,
    OPTIONS,
    Kind,
    Option,
    choices_from_config,
    command_line_tokens,
    config_from_choices,
    group_title,
    help_epilog,
    nation_choice,
    options_in,
    parse_date_spec,
)
from .plotting import COUNTRIES, build_figure, resolve_start  # noqa: E402

PAGE_TITLE = "Bond Yields and Public Debt"
PAGE_SUMMARY = (
    "Government bond yields against public debt, GDP and interest outlays, for Canada and the "
    "United States, from Statistics Canada, the Bank of Canada, FRED, the U.S. Treasury and the "
    "U.S. Census Bureau; and, when ticked, for the United Kingdom, from the Bank of England and the "
    "Office for National Statistics. Choose the charts in the panel on the left (the » button on a phone); "
    "each change redraws them."
)
_SCRIPT_NAME = "ratesplot.py"
# The query parameter that holds the chart's command line.
ADDRESS_KEY = "chart"
# How long a download is kept for every visitor. The regular sources publish
# at most once a day (yields in the afternoon, Debt to the Penny the next
# morning), so an hour shows each new figure within an hour at most, and the
# visitor who comes after a quiet hour waits for the downloads (about 10 s).
DOWNLOAD_MAX_AGE_SECONDS = 3600
# What a blank field means, shown in it. The other blank fields mean
# "automatic" (the axis limits) or are filled with their default.
_PLACEHOLDERS = {
    "start": "automatic",
    "end": "today",
    "regression-tolerance": f"{DEFAULT_REGRESSION_TOLERANCE_PCT:g}",
}
_DEFAULT_PLACEHOLDER = "automatic"
_DRAWING_LOCK = threading.Lock()
# How many zoom and pan steps "Back" can undo (as in the window).
_HISTORY_LIMIT = 50
# The buttons move the dates by this fraction of the window, or scale it by this factor.
_DATE_STEP_FRACTION = 0.5
_ZOOM_OUT_FACTOR = 2.0
_ZOOM_HINT = (
    "Drag a box on a chart with the mouse to zoom to it: its width sets the dates, its height the "
    "axes' limits. The buttons move the dates, and Back undoes a zoom."
)

# Session-state keys: one per control, and the page's own records.
_FLAG = "flag:"      # + option name: a check box
_VALUE = "value:"    # + option name, or a nation's choice key ("us:max"): a text field
_LEVEL = "level:"    # + letter: one of the "By level" boxes
_CHOICE = "choice:"  # + choice key + ":" + value: one tick box of a list option (Option.choices)
_CHOICE_OTHERS = "choice_others:"  # + choice key: the values given that no offered box has
_READY = "ready"     # the controls have been set from the address
_ADDRESS_PROBLEMS = "address_problems"
_NOTICE = "notice"   # a message from a control's callback, shown once
_DRAWING = "drawing"  # the last Drawing
_FORCE = "force_redraw"
_START_HINT = "start_hint"  # what the Start field's hint says now
_HISTORY = "history"  # the choices before each zoom or pan, for Back and Unzoom
_CHART = "chart:"     # + country key + drawing number: a chart (its dragged box)
_CALENDAR = "calendar:"  # + option name: a date field's calendar
_SIZE = "size:"          # + "width", "height", "lock", "ratio": the Size boxes, their lock and kept shape
# The calendar's first day: no series starts earlier (config.EARLIEST_DATA_START).
_CALENDAR_FIRST = EARLIEST_DATA_START.date()


@dataclass(frozen=True)
class Drawing:
    """One drawing of the charts, kept in the visitor's session."""

    config: PlotConfig  # as the controls described it (an automatic start is None)
    drawn: PlotConfig | None  # with the start found; None if the drawing failed
    pngs: dict[str, bytes]  # Country.key -> PNG bytes
    geometry: dict[str, ChartGeometry | None]  # Country.key -> where its axes are, for zooming
    tokens: tuple[str, ...]  # the chart's command line (after the script name)
    log: str
    error: str | None
    seconds: float
    finished: pd.Timestamp

    def chart_key(self, key: str) -> str:
        """The session key of ``key``'s chart in this drawing (a new drawing, a new chart: no stale box)."""
        return f"{_CHART}{key}:{self.finished.value}"


# ---------------------------------------------------------------------------
# The chart on the page: a picture that reports a box dragged over it
# ---------------------------------------------------------------------------

# Streamlit's own st.image reports no mouse events, so each chart is shown by
# this small component (Streamlit's components v2: it runs in the page
# itself, not in a frame, so the picture resizes with the page as st.image
# does). It shows the PNG bytes and, while the mouse button is held down,
# a dashed box as the window does. On release it sends the box back once,
# as fractions of the picture (0 to 1 from the left and from the top),
# whatever size the browser shows it at; _box_dragged turns that into the
# fields' texts. A click, or a drag shorter than 8 screen pixels both ways,
# sends nothing. Touch is left to scrolling the page: a finger drag on a
# chart that fills a phone's screen would otherwise trap it there. On a
# phone the buttons above the charts move the dates.
#
# Streamlit builds the component's element afresh on every run of the page
# (measured with Streamlit 1.64: a new parent element each run, and the
# function returned below called just before), so nothing is kept between
# runs: the picture's object URL is released by that function, and the
# picture is given its size from the PNG's header, so that the browser keeps
# its place while decoding it and the page does not jump.
_CHART_JS = """
const MIN_DRAG = 8;

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const frame = document.createElement("div");
  frame.className = "chart";
  frame.innerHTML = '<img alt="Chart" draggable="false"><div class="band"></div>';
  parentElement.appendChild(frame);
  const img = frame.querySelector("img");
  const band = frame.querySelector(".band");

  const header = new DataView(data.buffer, data.byteOffset, 24);  // PNG: IHDR width, height at 16, 20
  img.width = header.getUint32(16);
  img.height = header.getUint32(20);
  const url = URL.createObjectURL(new Blob([data], { type: "image/png" }));
  img.src = url;

  let start = null;
  const point = (event) => {
    const rect = img.getBoundingClientRect();
    return {
      x: Math.min(Math.max(event.clientX - rect.left, 0), rect.width),
      y: Math.min(Math.max(event.clientY - rect.top, 0), rect.height),
      width: rect.width,
      height: rect.height,
    };
  };
  const showBand = (a, b) => {
    band.style.left = Math.min(a.x, b.x) + "px";
    band.style.top = Math.min(a.y, b.y) + "px";
    band.style.width = Math.abs(b.x - a.x) + "px";
    band.style.height = Math.abs(b.y - a.y) + "px";
    band.style.display = "block";
  };
  img.onpointerdown = (event) => {
    if (event.pointerType === "touch" || event.button !== 0) return;
    event.preventDefault();
    img.setPointerCapture(event.pointerId);
    start = point(event);
    showBand(start, start);
  };
  img.onpointermove = (event) => {
    if (start) showBand(start, point(event));
  };
  img.onpointerup = (event) => {
    if (!start) return;
    const a = start;
    const b = point(event);
    start = null;
    band.style.display = "none";
    if (Math.abs(b.x - a.x) < MIN_DRAG && Math.abs(b.y - a.y) < MIN_DRAG) return;
    setTriggerValue("box", {
      x0: Math.min(a.x, b.x) / b.width,
      x1: Math.max(a.x, b.x) / b.width,
      y0: Math.min(a.y, b.y) / b.height,
      y1: Math.max(a.y, b.y) / b.height,
    });
  };
  img.onpointercancel = () => {
    start = null;
    band.style.display = "none";
  };
  return () => URL.revokeObjectURL(url);
}
"""
_CHART_CSS = """
.chart { position: relative; line-height: 0; }
.chart img { width: 100%; height: auto; cursor: crosshair; user-select: none; -webkit-user-select: none; }
.band {
  position: absolute; display: none; box-sizing: border-box; pointer-events: none;
  border: 2px dashed #1f5fbf; background: rgba(31, 95, 191, 0.08);
}
"""
def _chart_component():
    """Register the chart component with the running Streamlit and return the command that shows a chart.

    Registration belongs to Streamlit's runtime, not to this module, so it is
    done on every run rather than once at import: a process may start a
    second runtime (Streamlit's AppTest does, one per test), which would not
    know the component. Registering the same definition again is silent.
    """
    return st.components.v2.component("ratesplot_chart", js=_CHART_JS, css=_CHART_CSS)


def _png_size(png: bytes) -> tuple[int, int]:
    """Return a PNG's width and height in pixels, read from its header (the IHDR chunk)."""
    width, height = struct.unpack(">II", png[16:24])
    return width, height


# ---------------------------------------------------------------------------
# Process-wide setup (once per server process: imported modules are cached)
# ---------------------------------------------------------------------------


def _use_market_time_zone() -> None:
    """Make the process's local time Toronto's, so "today" is the markets' date.

    A cloud server usually runs on UTC, where "today" turns over at 20:00 in
    Toronto: the default end would then be tomorrow's date for four hours each
    evening. ``time.tzset`` exists only on Unix, so on Windows (running the
    page on Terry's own machine) the machine's own zone is kept, as the
    command line and the window keep it.
    """
    if hasattr(time, "tzset"):
        os.environ["TZ"] = MARKET_TIMEZONE
        time.tzset()


_use_market_time_zone()
http.enable_download_cache(max_age_seconds=DOWNLOAD_MAX_AGE_SECONDS)


# ---------------------------------------------------------------------------
# Choices <-> controls
# ---------------------------------------------------------------------------


def _gui_options(kind_is_value: bool) -> list[Option]:
    """The options that have a control here, VALUE ones or the others."""
    return [option for option in OPTIONS if option.in_gui and (option.kind is Kind.VALUE) == kind_is_value]


def _value_keys() -> list[tuple[str, Option, Nation | None]]:
    """Every value control: ``(choice key, option, nation)``; a nation's own (``NATION_OPTIONS``) once for each nation."""
    shared = [(option.name, option, None) for option in _gui_options(True) if option not in NATION_OPTIONS]
    own = [(nation_choice(nation, option), option, nation) for nation in NATIONS for option in NATION_OPTIONS]
    return shared + own


def _choices_from_address() -> tuple[Choices, list[str]]:
    """Return the starting choices from the address's command line, and any problems with it.

    The address is read as the command line is (``cli.sort_tokens``, then
    ``options.config_from_choices``), and laid over the defaults as the window
    lays a command line over them (``frontend.starting_choices``, with nothing
    remembered). If it cannot be read, the defaults are used.
    """
    text = st.query_params.get(ADDRESS_KEY, "")
    payloads, flags, unmatched, misspelled = sort_tokens(text.split())
    problems = [f"{token}: not an option" for token in unmatched] + misspelled
    config = PlotConfig()
    if not problems:
        try:
            config = config_from_choices(payloads, flags)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        payloads, flags = {}, {}
    choices, _notes = starting_choices(config, payloads, flags, {})
    return choices, problems


def _load_choices(choices: Choices) -> None:
    """Set every control from ``choices``."""
    payloads, flags = choices
    for key, option, nation in _value_keys():
        if option.editor == "levels":
            letters = level_letters(payloads.get(key, ""))
            for letter in COMPONENT_LETTERS:
                st.session_state[_LEVEL + letter] = letter in letters
        elif option.editor == "choices":
            text = payloads.get(key, "")
            rows = choice_rows(option, text, nation)
            st.session_state[_CHOICE_OTHERS + key] = [value for value, _box in rows[-1][1]] if rows[-1][0] == "other" else []
            ticked = choice_values(option, text, nation)
            for _caption, boxes in rows:
                for value, _box in boxes:
                    st.session_state[_CHOICE + key + ":" + value] = value in ticked
        else:
            st.session_state[_VALUE + key] = payloads.get(key, "")
            if option.editor == "size":
                _size_boxes_from_text(key)
    for option in _gui_options(kind_is_value=False):
        st.session_state[_FLAG + option.name] = bool(flags.get(option.name, False))


def _snapshot() -> Choices:
    """Return every control's value, blanks included, as the window's ``_snapshot`` does."""
    payloads: dict[str, str] = {}
    for key, option, nation in _value_keys():
        if option.editor == "levels":
            payloads[key] = "".join(letter for letter in COMPONENT_LETTERS if st.session_state.get(_LEVEL + letter, False))
        elif option.editor == "choices":
            ticked = [
                value for value in _choice_values_offered(key) if st.session_state.get(_CHOICE + key + ":" + value, False)
            ]
            payloads[key] = choices_text(option, ticked)
        else:
            payloads[key] = st.session_state.get(_VALUE + key, "")
    flags = {option.name: bool(st.session_state.get(_FLAG + option.name, False)) for option in _gui_options(False)}
    return payloads, flags


def _config_for(choices: Choices) -> tuple[PlotConfig | None, list[str]]:
    """Return the PlotConfig the controls describe, or None and the problems, one per bad field first."""
    payloads = {name: text.strip() for name, text in choices[0].items() if text.strip()}
    flags = choices[1]
    # A value whose flag is off has no effect and its field is greyed: a bad
    # one left there must not stop the drawing (as in the window).
    for key in list(payloads):
        if option_of(key).turns_on in flags and not flags[option_of(key).turns_on]:
            payloads.pop(key, None)
    problems: list[str] = []
    for key, option, nation in _value_keys():
        if key in payloads:
            try:
                option.parse(payloads[key])
            except ValueError as exc:
                problems.append(f"{nation.name + ', ' if nation else ''}{option.label}: {exc}")
    try:
        return config_from_choices(payloads, flags), problems
    except ValueError as exc:
        return None, problems or [str(exc)]


# ---------------------------------------------------------------------------
# Control callbacks (they run before the page is drawn again)
# ---------------------------------------------------------------------------


def _country_changed(name: str) -> None:
    """Keep at least one country ticked: the command line's "none means both" would look like a bug."""
    countries = [option.name for option in options_in("country") if option.in_gui]
    if not any(st.session_state[_FLAG + country] for country in countries):
        st.session_state[_FLAG + name] = True
        st.session_state[_NOTICE] = "At least one country must stay selected."


def _measure_changed(name: str) -> None:
    """One measure at a time; Top and Bottom are in the measure's units, so they are cleared."""
    units = [option.name for option in options_in("units") if option.in_gui]
    if st.session_state[_FLAG + name]:
        for other in units:
            if other != name:
                st.session_state[_FLAG + other] = False
    for bound in bound_keys():
        st.session_state[_VALUE + bound] = ""


def _remember_view() -> None:
    """Keep the controls' choices for Back and Unzoom (the last ``_HISTORY_LIMIT``)."""
    history = st.session_state.setdefault(_HISTORY, [])
    history.append(_snapshot())
    del history[:-_HISTORY_LIMIT]


def _choice_values_offered(key: str) -> list[str]:
    """Every value a list control has a box for now: those offered, then any others given (``_CHOICE_OTHERS``)."""
    _key, option, nation = next(entry for entry in _value_keys() if entry[0] == key)
    offered = [value for _caption, boxes in choice_rows(option, "", nation) for value, _box in boxes]
    return offered + st.session_state.get(_CHOICE_OTHERS + key, [])


def _choice_changed(key: str, value: str) -> None:
    """Keep at least one box of a list ticked: an empty list would read as the default (``choice_refusal``)."""
    if not any(st.session_state.get(_CHOICE + key + ":" + item, False) for item in _choice_values_offered(key)):
        st.session_state[_CHOICE + key + ":" + value] = True
        st.session_state[_NOTICE] = choice_refusal(option_of(key))


def _reset() -> None:
    _remember_view()  # as in the window, Back undoes a Reset
    _load_choices(choices_from_config(PlotConfig()))
    st.query_params.clear()


def _force_redraw() -> None:
    st.session_state[_FORCE] = True


def _apply_view(updates: dict[str, str], key: str | None = None) -> None:
    """Write a zoom's or a date move's texts into the fields, undoably (Back), as the window does.

    Nothing is kept for Back when the fields already say it (e.g. Zoom out
    from 1680 to today). The axis limits go into the fields of the chart's
    own nation (``key``); the dates are the same for every chart.
    """
    nation = next((nation for nation in NATIONS if nation.key == key), None)
    own = {option.name for option in NATION_OPTIONS}
    updates = {nation_choice(nation, option_of(name)) if nation and name in own else name: text for name, text in updates.items()}
    payloads = _snapshot()[0]
    if all(payloads.get(name, "") == text for name, text in updates.items()):
        return
    _remember_view()
    for name, text in updates.items():
        st.session_state[_VALUE + name] = text


def _box_dragged(key: str) -> None:
    """A box was dragged on ``key``'s chart: zoom to it, by the window's rules (``frontend.zoom_updates``)."""
    drawing: Drawing | None = st.session_state.get(_DRAWING)
    if drawing is None or drawing.geometry.get(key) is None:
        return
    box = (st.session_state.get(drawing.chart_key(key)) or {}).get("box")
    if not box:
        return
    width, height = _png_size(drawing.pngs[key])
    _apply_view(
        zoom_updates(
            drawing.geometry[key],
            drawing.drawn.for_nation(key) if drawing.drawn is not None else None,
            box["x0"] * width,
            box["y0"] * height,
            box["x1"] * width,
            box["y1"] * height,
        ),
        key,
    )


def _move_dates(shift: float, scale: float) -> None:
    """Move or rescale the dates drawn (``frontend.move_dates_updates``): the buttons over the charts."""
    drawing: Drawing | None = st.session_state.get(_DRAWING)
    if drawing is not None and drawing.drawn is not None:
        _apply_view(move_dates_updates(drawing.drawn.start, drawing.drawn.end, shift=shift, scale=scale))


def _back() -> None:
    history = st.session_state.get(_HISTORY)
    if history:
        _load_choices(history.pop())


def _unzoom() -> None:
    history = st.session_state.get(_HISTORY)
    if history:
        first = history[0]
        history.clear()
        _load_choices(first)


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------


def _help(option: Option, extra: str = "") -> str:
    """The option's help, as the window's tooltip, one paragraph per line (Markdown)."""
    return "\n\n".join([*option_help_lines(option), *([extra] if extra else [])])


def _flag_control(where, option: Option) -> None:
    callbacks = {"country": _country_changed, "units": _measure_changed}
    callback = callbacks.get(option.group)
    relative = st.session_state.get(_FLAG + "relative", False)
    where.checkbox(
        option.label,
        key=_FLAG + option.name,
        help=_help(option),
        on_change=callback,
        args=(option.name,) if callback else None,
        # GDP is only the denominator under -r and is not drawn.
        disabled=option.name == "gdp" and relative,
    )


def _value_control(where, option: Option, placeholder: str | None = None, key: str | None = None) -> None:
    """A text field; ``key`` names a nation's own field ("us:max"), else the option's name."""
    flag_off = option.turns_on is not None and not st.session_state.get(_FLAG + option.turns_on, False)
    where.text_input(
        option.label,
        key=_VALUE + (key or option.name),
        help=_help(option, "Blank = default."),
        placeholder=placeholder or _PLACEHOLDERS.get(option.name, _DEFAULT_PLACEHOLDER),
        disabled=flag_off,
    )


def _field_date(option: Option) -> datetime.date | None:
    """The date a date field names now, for its calendar; None if blank, unreadable or outside the calendar's range."""
    text = st.session_state.get(_VALUE + option.name, "").strip()
    try:
        date = parse_date_spec(text, kind=option.name).date() if text else None
    except ValueError:
        return None
    return date if date is not None and _CALENDAR_FIRST <= date <= _today() else None


def _today() -> datetime.date:
    return pd.Timestamp.today().date()


def _date_picked(name: str) -> None:
    """A day was picked in a field's calendar: write it in the field (a cleared calendar changes nothing)."""
    picked = st.session_state.get(_CALENDAR + name)
    if picked is not None:
        st.session_state[_VALUE + name] = f"{picked:%Y-%m-%d}"


def _date_blanked(name: str) -> None:
    st.session_state[_VALUE + name] = ""


def _date_control(where, option: Option, placeholder: str | None) -> None:
    """A date field with a calendar button beside it, as in the window.

    Picking a day writes ``YYYY-MM-DD`` in the field, which stays the single
    source of truth (typing ``YYYY`` or ``YYYY-MM`` still works); "Blank"
    empties it, meaning the option's default (today for the end; for the
    start, ``config.DEFAULT_START_FLOOR`` or the first date on which every
    chosen curve has data, if later). The
    calendar opens at the field's date: it is set from the field on every
    run, before it is drawn.
    """
    # One row that never wraps (st.columns would put the button under the
    # field on a phone); the field takes the room the button leaves.
    row = where.container(horizontal=True, wrap=False, vertical_alignment="bottom")
    _value_control(row, option, placeholder)
    with row.popover("📅", help="Pick a date from a calendar"):
        st.session_state[_CALENDAR + option.name] = _field_date(option)
        st.date_input(
            option.label,
            key=_CALENDAR + option.name,
            min_value=_CALENDAR_FIRST,
            max_value=_today(),
            format="YYYY-MM-DD",
            on_change=_date_picked,
            args=(option.name,),
        )
        st.button(
            "Blank (default)",
            key=f"blank:{option.name}",
            on_click=_date_blanked,
            args=(option.name,),
            help=_PLACEHOLDERS.get(option.name, _DEFAULT_PLACEHOLDER).capitalize() + " when blank.",
        )


def _pixels(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def _remember_ratio() -> None:
    """Take the boxes' shape as the one the lock keeps (width / height), if both are numbers."""
    width = _pixels(st.session_state.get(_SIZE + "width", ""))
    height = _pixels(st.session_state.get(_SIZE + "height", ""))
    if width and height:
        st.session_state[_SIZE + "ratio"] = width / height


def _size_boxes_from_text(name: str) -> None:
    """Show a ``WxH`` text set from elsewhere (the address, Back, Reset) in the boxes, and take its shape.

    As in the window: one dimension alone means 4:3 on the command line, so
    the other box shows the size that implies.
    """
    text = st.session_state.get(_VALUE + name, "").strip().lower()
    width, height = (part.strip() for part in text.split("x", 1)) if "x" in text else (text, "")
    if width.isdigit() and not height:
        height = str(round(int(width) * 3 / 4))
    elif height.isdigit() and not width:
        width = str(round(int(height) * 4 / 3))
    st.session_state[_SIZE + "width"] = width
    st.session_state[_SIZE + "height"] = height
    _remember_ratio()


def _size_box_changed(name: str, which: str) -> None:
    """A Size box changed: with the lock on, set the other from the kept shape; then write the ``WxH`` text."""
    state = st.session_state
    ratio = state.get(_SIZE + "ratio")
    if state.get(_SIZE + "lock", True) and ratio:
        other = "height" if which == "width" else "width"
        value = _pixels(state[_SIZE + which])
        if value is not None:
            state[_SIZE + other] = str(max(1, round(value / ratio if which == "width" else value * ratio)))
        elif not state[_SIZE + which].strip():
            state[_SIZE + other] = ""
    else:
        _remember_ratio()
    width, height = state[_SIZE + "width"].strip(), state[_SIZE + "height"].strip()
    state[_VALUE + name] = f"{width}x{height}" if width and height else width or (f"x{height}" if height else "")


def _size_lock_changed() -> None:
    # Ticking the lock keeps the shape the boxes have now.
    if st.session_state[_SIZE + "lock"]:
        _remember_ratio()


def _size_control(where, option: Option) -> None:
    """Width and height boxes and a "Preserve aspect ratio" lock, as in the window (``gui._SizeEditor``).

    Terry, 2026-09-30: "if you have a chart shape you like, and have saved to
    PNG, but you need an upscaled or downscaled version of it, you can just
    change one number and the other automatically adjusts." The option's own
    ``WxH`` text (``_VALUE`` + name, the command-line form) stays the single
    source of truth for drawing, the address, Back and Reset; the boxes are a
    view of it and write it back. With the lock on (the default), changing
    one box sets the other from the shape the pair had when the lock was
    ticked or the size was loaded, so repeated edits do not drift; with it
    off, each box changes alone. A blank box with the other filled keeps the
    command line's meaning: 4:3 from the one given.
    """
    st.session_state.setdefault(_SIZE + "lock", True)
    tip = _help(option, "Width and height in pixels. Blank = default.")
    row = where.container(horizontal=True, wrap=False)
    for which, placeholder in (("width", PlotConfig().width_px), ("height", PlotConfig().height_px)):
        row.text_input(
            f"{which.capitalize()} (px)",
            key=_SIZE + which,
            help=tip,
            placeholder=str(placeholder),
            on_change=_size_box_changed,
            args=(option.name, which),
        )
    where.checkbox(
        "Preserve aspect ratio",
        key=_SIZE + "lock",
        on_change=_size_lock_changed,
        help="When ticked, changing one box changes the other to keep the current shape.",
    )


def _levels_control(where, option: Option) -> None:
    where.markdown(f"**{option.label}**", help=_help(option, "None ticked = debt and interest in total."))
    columns = where.columns(2)
    for index, (letter, caption) in enumerate(LEVEL_BOXES):
        columns[index % 2].checkbox(caption, key=_LEVEL + letter)


def _choices_control(where, option: Option, nation: Nation | None = None) -> None:
    """A list option as tick boxes (``Option.choices``): a row of boxes per row offered, with its caption.

    One that belongs to a flag (mortgage terms, spreads) is greyed while
    the flag is off, as its field was. A nation's own list has its label
    always, and only the boxes its chart can draw (``frontend.choice_rows``).
    """
    key = nation_choice(nation, option) if nation else option.name
    disabled = option.turns_on is not None and not st.session_state.get(_FLAG + option.turns_on, False)
    if option.turns_on is None or nation is not None:
        where.markdown(f"**{option.label}**", help=_help(option))
    others = st.session_state.get(_CHOICE_OTHERS + key, [])
    rows = [row for row in choice_rows(option, "", nation)]
    if others:
        rows.append(("other", [(value, value.replace("-", "–")) for value in others]))
    width = max(len(boxes) for _caption, boxes in rows)  # boxes line up from row to row
    for caption, boxes in rows:
        if caption:
            where.caption(caption)
        columns = where.columns(width)
        for column, (value, text) in zip(columns, boxes):
            column.checkbox(
                text,
                key=_CHOICE + key + ":" + value,
                help=_help(option),
                disabled=disabled,
                on_change=_choice_changed,
                args=(key, value),
            )


def _start_placeholder() -> str | None:
    """"automatic: 1981-09-01" once a drawing has found the start (it shows in no field otherwise)."""
    drawing: Drawing | None = st.session_state.get(_DRAWING)
    if drawing is not None and drawing.drawn is not None and drawing.config.start is None:
        return f"automatic: {drawing.drawn.start:%Y-%m-%d}"
    return None


def _controls() -> None:
    """The sidebar: two buttons, then one section per option group, all from the table."""
    bar = st.sidebar
    buttons = bar.columns(2)
    buttons[0].button("Redraw", on_click=_force_redraw, help="Draw again, with fresh intraday quotes.", width="stretch")
    buttons[1].button("Reset", on_click=_reset, help="Back to the default charts.", width="stretch")
    for group in GROUPS:
        members = [
            option
            for option in OPTIONS
            if (option.panel or option.group) == group and option.in_gui and option not in NATION_OPTIONS
        ]
        if not members:
            continue
        bar.subheader(group_title(group), divider="gray")
        # A value that belongs to a flag goes right under it (the tolerance
        # under "Regression segments"), as on the window's row.
        attached = {option.turns_on: option for option in members if option.kind is Kind.VALUE and option.turns_on}
        values = [option for option in members if option.kind is Kind.VALUE and option not in attached.values()]
        for option in members:
            if option.kind is Kind.VALUE:
                continue
            _flag_control(bar, option)
            if option.name in attached and attached[option.name].editor == "choices":
                # Straight in the sidebar: Streamlit allows no columns inside its columns there.
                _choices_control(bar, attached[option.name])
            elif option.name in attached:
                _value_control(bar.columns([1, 12])[1], attached[option.name])
        # Dates one to a row (the Start field's hint names the date found);
        # other text fields two to a row (Min and Max, Top and Bottom).
        for option in values:
            if option.editor == "date":
                placeholder = None
                if option.name == "start":
                    placeholder = st.session_state[_START_HINT] = _start_placeholder()
                _date_control(bar, option, placeholder)
        paired = [option for option in values if option.editor not in ("date", "levels", "size", "choices")]
        for index in range(0, len(paired), 2):
            for column, option in zip(bar.columns(2), paired[index:index + 2]):
                _value_control(column, option)
        for option in values:
            if option.editor == "choices":
                _choices_control(bar, option)
            elif option.editor == "levels":
                _levels_control(bar, option)
            elif option.editor == "size":
                _size_control(bar, option)
    for nation in NATIONS:
        _nation_controls(bar, nation)


def _nation_controls(bar, nation: Nation) -> None:
    """A nation's own section (``frontend.NATION_OPTIONS``), as the panel under its tab in the window.

    Its yield terms, spreads and mortgage terms (only those its chart can
    draw), then its yield-axis Min and Max and right-axis Top and Bottom.
    """
    bar.subheader(f"{nation.name} only", divider="gray")
    order = ("yield_terms", "spread_pairs", "mortgage_terms")
    for option in sorted((o for o in NATION_OPTIONS if o.editor == "choices"), key=lambda o: order.index(o.fields[0])):
        _choices_control(bar, option, nation)
    limits = [option for option in NATION_OPTIONS if option.editor != "choices"]
    for index in range(0, len(limits), 2):
        for column, option in zip(bar.columns(2), limits[index:index + 2]):
            _value_control(column, option, key=nation_choice(nation, option))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw(config: PlotConfig, tokens: tuple[str, ...]) -> Drawing:
    """Draw every chosen country's chart for ``config``; never raises (a failure is in the Drawing)."""
    lines: list[str] = []
    began = time.perf_counter()
    source = "the automatic start" if config.start is None else f"{config.start:%Y-%m-%d}"
    lines.append(f"--- drawing {config.width_px}x{config.height_px}, {source} to {config.end:%Y-%m-%d}\n")
    drawn: PlotConfig | None = None
    pngs: dict[str, bytes] = {}
    geometry: dict[str, ChartGeometry | None] = {}
    error: str | None = None
    with this_thread_output_to(lines.append), _DRAWING_LOCK:
        try:
            drawn = resolve_start(config)
            for country in COUNTRIES:
                if getattr(drawn, country.show_field):
                    figure = build_figure(country, drawn)
                    pngs[country.key] = png_bytes(figure)
                    # Measured after savefig, so the axes positions are the final rendered ones (as in the window).
                    geometry[country.key] = ChartGeometry.from_figure(figure, drawn)
        except Exception as exc:  # shown on the page and in the log, never swallowed
            error = f"{type(exc).__name__}: {exc}"
            lines.append(traceback.format_exc())
            drawn = None
    return Drawing(
        config=config,
        drawn=drawn,
        pngs=pngs,
        geometry=geometry,
        tokens=tokens,
        log="".join(lines),
        error=error,
        seconds=time.perf_counter() - began,
        finished=pd.Timestamp.now(tz=MARKET_TIMEZONE),
    )


def _set_address(tokens: tuple[str, ...]) -> None:
    """Write the chart's command line into the page's address (nothing at all for the defaults)."""
    text = " ".join(tokens)
    if st.query_params.get(ADDRESS_KEY, "") == text:
        return
    if text:
        st.query_params[ADDRESS_KEY] = text
    else:
        st.query_params.pop(ADDRESS_KEY, None)


def _view_buttons(drawing: Drawing) -> None:
    """Back, Unzoom and the date buttons, in one row over the charts, and how to zoom."""
    history = st.session_state.get(_HISTORY, [])
    drawn = drawing.drawn
    at_first = drawn is None or drawn.start <= EARLIEST_DATA_START
    at_today = drawn is None or drawn.end >= pd.Timestamp.today().normalize()
    buttons = (
        ("◀ Back", _back, (), not history, "Undo the last zoom or move of the dates."),
        ("Unzoom", _unzoom, (), not history, "Back to the charts before the first zoom or move."),
        ("◀ Earlier", _move_dates, (-_DATE_STEP_FRACTION, 1.0), at_first, "Move the dates back by half the window."),
        ("Later ▶", _move_dates, (_DATE_STEP_FRACTION, 1.0), at_today, "Move the dates on by half the window, up to today."),
        ("Zoom out", _move_dates, (0.0, _ZOOM_OUT_FACTOR), at_first and at_today, "Twice the span of dates, about its middle."),
    )
    # A row that wraps (not st.columns, which a phone stacks one per line,
    # pushing the charts below the screen).
    row = st.container(horizontal=True, wrap=True)
    for label, callback, args, disabled, tip in buttons:
        row.button(label, on_click=callback, args=args, disabled=disabled, help=tip)
    st.caption(_ZOOM_HINT)


def _show(drawing: Drawing) -> None:
    """The charts, one tab per country, each with its download button; then the command line and log."""
    if drawing.error is not None:
        st.error(f"Drawing failed: {drawing.error} (details in the log below)")
    if drawing.pngs:
        _view_buttons(drawing)
        show_chart = _chart_component()
        keys = [key for key in TAB_TITLES if key in drawing.pngs]
        for tab, key in zip(st.tabs([TAB_TITLES[key] for key in keys]), keys):
            with tab:
                show_chart(
                    key=drawing.chart_key(key),
                    data=drawing.pngs[key],
                    on_box_change=functools.partial(_box_dragged, key),
                )
                st.download_button(
                    "Download this chart (PNG)",
                    data=drawing.pngs[key],
                    file_name=saved_png_name(key, drawing.drawn),
                    mime="image/png",
                    on_click="ignore",
                    key=f"download:{key}",
                )
        st.caption(
            f"Drawn at {drawing.finished:%Y-%m-%d %H:%M %Z} in {drawing.seconds:.1f} s. "
            "The page's address holds this chart's settings: copy it from the address bar to share the chart."
        )
    st.markdown("**The same chart from the command line**")
    st.code(" ".join(["python", _SCRIPT_NAME, *drawing.tokens]), language=None, wrap_lines=True)
    with st.expander("Log"):
        st.code(drawing.log.rstrip() or "(nothing printed)", language=None, wrap_lines=True)


def main() -> None:
    """Draw the page (Streamlit runs this from the top on every change of a control)."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📈", layout="wide")
    state = st.session_state
    if not state.get(_READY):
        choices, problems = _choices_from_address()
        _load_choices(choices)
        state[_ADDRESS_PROBLEMS] = problems
        state[_READY] = True

    _controls()
    st.header(PAGE_TITLE)
    st.caption(PAGE_SUMMARY)
    if state.get(_ADDRESS_PROBLEMS):
        st.warning(
            "The chart named in this page's address could not be drawn ("
            + "; ".join(state[_ADDRESS_PROBLEMS])
            + "). The default charts are shown instead."
        )
    if state.get(_NOTICE):
        st.info(state.pop(_NOTICE))

    choices = _snapshot()
    config, problems = _config_for(choices)
    previous: Drawing | None = state.get(_DRAWING)
    forced = state.pop(_FORCE, False)
    if config is None:
        st.error("Not drawn: " + "; ".join(problems))
    elif previous is None or previous.config != config or forced:
        tokens = tuple(command_line_tokens(*choices))
        if _DRAWING_LOCK.locked():
            note = "Another visitor's chart is being drawn; yours is next …"
        elif previous is None:
            note = "Drawing … (after a quiet hour the data are downloaded first: about 15 s)"
        else:
            note = "Drawing …"
        with st.spinner(note, show_time=True):
            drawing = _draw(config, tokens)
        state[_DRAWING] = drawing
        if previous is not None:
            state[_ADDRESS_PROBLEMS] = []  # the visitor has moved on from the address's chart
        if drawing.error is None:
            _set_address(tokens)
        if _start_placeholder() != state.get(_START_HINT):
            # The controls were laid out before this drawing found the start:
            # lay them out again (no new drawing) so the Start field shows it.
            st.rerun()
    drawing = state.get(_DRAWING)
    if drawing is not None:
        _show(drawing)

    with st.expander("Every option, as the command line spells it"):
        st.code(help_epilog(), language=None)
