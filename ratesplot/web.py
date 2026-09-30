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
    and changing it clears Top and Bottom; GDP greyed out under -r; a value
    that belongs to a flag (the regression tolerance) greyed while the flag
    is off, and ignored then. A new option in the table appears here with no
    change to this module.

The chart's address
    The page's address carries the chart's command line: ``?chart=-c+-r+--reg``
    holds the same tokens ``python US_CDN_rates_plot.py -c -r --reg`` takes
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

Not here (the window has them)
    Mouse zoom and pan (type Start, End, Min, Max, Top and Bottom instead), the
    calendar button, the aspect-ratio lock (Size is typed as WxH), and
    remembered settings (the address keeps the chart instead).
"""

from __future__ import annotations

import os
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
    MARKET_TIMEZONE,
    PlotConfig,
)
from .console import this_thread_output_to  # noqa: E402
from .frontend import (  # noqa: E402
    LEVEL_BOXES,
    TAB_TITLES,
    Choices,
    level_letters,
    option_help_lines,
    png_bytes,
    saved_png_name,
    starting_choices,
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
    options_in,
    options_of,
)
from .plotting import COUNTRIES, build_figure, resolve_start  # noqa: E402

PAGE_TITLE = "Canadian and U.S. Bond Yields and Public Debt"
PAGE_SUMMARY = (
    "Government bond yields against public debt, GDP and interest outlays, for Canada and the "
    "United States, from Statistics Canada, the Bank of Canada, FRED, the U.S. Treasury and the "
    "U.S. Census Bureau. Choose the charts in the panel on the left (the » button on a phone); each change redraws them."
)
_SCRIPT_NAME = "US_CDN_rates_plot.py"
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

# Session-state keys: one per control, and the page's own records.
_FLAG = "flag:"      # + option name: a check box
_VALUE = "value:"    # + option name: a text field
_LEVEL = "level:"    # + letter: one of the "By level" boxes
_READY = "ready"     # the controls have been set from the address
_ADDRESS_PROBLEMS = "address_problems"
_NOTICE = "notice"   # a message from a control's callback, shown once
_DRAWING = "drawing"  # the last Drawing
_FORCE = "force_redraw"
_START_HINT = "start_hint"  # what the Start field's hint says now


@dataclass(frozen=True)
class Drawing:
    """One drawing of the charts, kept in the visitor's session."""

    config: PlotConfig  # as the controls described it (an automatic start is None)
    drawn: PlotConfig | None  # with the start found; None if the drawing failed
    pngs: dict[str, bytes]  # Country.key -> PNG bytes
    tokens: tuple[str, ...]  # the chart's command line (after the script name)
    log: str
    error: str | None
    seconds: float
    finished: pd.Timestamp


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
    for option in _gui_options(kind_is_value=True):
        if option.editor == "levels":
            letters = level_letters(payloads.get(option.name, ""))
            for letter in COMPONENT_LETTERS:
                st.session_state[_LEVEL + letter] = letter in letters
        else:
            st.session_state[_VALUE + option.name] = payloads.get(option.name, "")
    for option in _gui_options(kind_is_value=False):
        st.session_state[_FLAG + option.name] = bool(flags.get(option.name, False))


def _snapshot() -> Choices:
    """Return every control's value, blanks included, as the window's ``_snapshot`` does."""
    payloads: dict[str, str] = {}
    for option in _gui_options(kind_is_value=True):
        if option.editor == "levels":
            payloads[option.name] = "".join(
                letter for letter in COMPONENT_LETTERS if st.session_state.get(_LEVEL + letter, False)
            )
        else:
            payloads[option.name] = st.session_state.get(_VALUE + option.name, "")
    flags = {option.name: bool(st.session_state.get(_FLAG + option.name, False)) for option in _gui_options(False)}
    return payloads, flags


def _config_for(choices: Choices) -> tuple[PlotConfig | None, list[str]]:
    """Return the PlotConfig the controls describe, or None and the problems, one per bad field first."""
    payloads = {name: text.strip() for name, text in choices[0].items() if text.strip()}
    flags = choices[1]
    # A value whose flag is off has no effect and its field is greyed: a bad
    # one left there must not stop the drawing (as in the window).
    for option in options_of(Kind.VALUE):
        if option.turns_on in flags and not flags[option.turns_on]:
            payloads.pop(option.name, None)
    problems: list[str] = []
    for option in options_of(Kind.VALUE):
        if option.name in payloads:
            try:
                option.parse(payloads[option.name])
            except ValueError as exc:
                problems.append(f"{option.label}: {exc}")
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
    for bound in ("top", "bottom"):
        st.session_state[_VALUE + bound] = ""


def _reset() -> None:
    _load_choices(choices_from_config(PlotConfig()))
    st.query_params.clear()


def _force_redraw() -> None:
    st.session_state[_FORCE] = True


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


def _value_control(where, option: Option, placeholder: str | None = None) -> None:
    flag_off = option.turns_on is not None and not st.session_state.get(_FLAG + option.turns_on, False)
    where.text_input(
        option.label,
        key=_VALUE + option.name,
        help=_help(option, "Blank = default."),
        placeholder=placeholder or _PLACEHOLDERS.get(option.name, _DEFAULT_PLACEHOLDER),
        disabled=flag_off,
    )


def _levels_control(where, option: Option) -> None:
    where.markdown(f"**{option.label}**", help=_help(option, "None ticked = debt and interest in total."))
    columns = where.columns(2)
    for index, (letter, caption) in enumerate(LEVEL_BOXES):
        columns[index % 2].checkbox(caption, key=_LEVEL + letter)


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
        members = [option for option in OPTIONS if option.group == group and option.in_gui]
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
            if option.name in attached:
                _value_control(bar.columns([1, 12])[1], attached[option.name])
        # Dates one to a row (the Start field's hint names the date found);
        # other text fields two to a row (Min and Max, Top and Bottom).
        for option in values:
            if option.editor == "date":
                placeholder = None
                if option.name == "start":
                    placeholder = st.session_state[_START_HINT] = _start_placeholder()
                _value_control(bar, option, placeholder)
        paired = [option for option in values if option.editor not in ("date", "levels")]
        for index in range(0, len(paired), 2):
            for column, option in zip(bar.columns(2), paired[index:index + 2]):
                _value_control(column, option)
        for option in values:
            if option.editor == "levels":
                _levels_control(bar, option)


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
    error: str | None = None
    with this_thread_output_to(lines.append), _DRAWING_LOCK:
        try:
            drawn = resolve_start(config)
            for country in COUNTRIES:
                if getattr(drawn, country.show_field):
                    pngs[country.key] = png_bytes(build_figure(country, drawn))
        except Exception as exc:  # shown on the page and in the log, never swallowed
            error = f"{type(exc).__name__}: {exc}"
            lines.append(traceback.format_exc())
            drawn = None
    return Drawing(
        config=config,
        drawn=drawn,
        pngs=pngs,
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


def _show(drawing: Drawing) -> None:
    """The charts, one tab per country, each with its download button; then the command line and log."""
    if drawing.error is not None:
        st.error(f"Drawing failed: {drawing.error} (details in the log below)")
    if drawing.pngs:
        keys = [key for key in TAB_TITLES if key in drawing.pngs]
        for tab, key in zip(st.tabs([TAB_TITLES[key] for key in keys]), keys):
            with tab:
                st.image(drawing.pngs[key], width="stretch")
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
