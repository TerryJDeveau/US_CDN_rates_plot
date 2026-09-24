"""The interactive window: every chart option adjustable, charts redrawn on the fly.

Opened by default (``--gui``); ``--no-gui`` gives the plain matplotlib windows.

Controls
    Generated from ``options.OPTIONS``: a check box for every on/off option, a
    text field for every VALUE option (with a calendar button where the option
    has ``picker="date"``), grouped as in ``--help``, each with the option's
    help text and command-line spelling as a tooltip. A new option in the table
    appears here with no change to this module (``in_gui=False`` opts out). The
    choices become a ``PlotConfig`` through ``options.config_from_choices``, the
    same code the command line uses, so the two cannot disagree.

Remembered settings
    The choices of the last successful drawing, the preview mode, the selected
    tab and the window size are saved to ``gui_state.json`` (see
    ``state_path``) after every drawing and on closing, and restored at the
    next start. Options given on the command line override the remembered ones
    they name: a value option individually; the curves, or the countries, as a
    group, because naming one curve on the command line means "only this one".

Zoom and pan
    The mouse edits the same fields a user would type in, so the fields remain
    the single source of truth (and the command line, the remembered settings
    and "Save PNG" all follow). Left-drag a box to zoom to it: its width sets
    Start/End, its height sets the yield Min/Max and the dollar Top/Bottom of
    whichever axes are drawn. Right- or middle-drag pans. The wheel zooms the
    dates about the pointer. "Back" undoes one zoom or pan; "Unzoom" returns to
    the view before the first one. Axis limits are shared options, so zooming
    one country's chart applies them to the other too, as on the command line.

Drawing
    Each chart is drawn exactly as the command line draws it (the same
    ``plotting.draw_country``) on an off-screen Agg figure at the configured
    pixel size, rendered to PNG, and shown scaled to fit the window (or at
    actual size). "Save PNG" writes those same bytes, so a saved chart is the
    real output, not a screenshot of the preview.

Threading
    Fetching and drawing run on one worker thread at a time so the window stays
    responsive during downloads. The worker never touches Tk; it posts messages
    to a queue that the window polls. Its printed progress and warnings are
    captured into the log pane. If the choices change while a chart is being
    drawn, the newest choices are drawn as soon as the worker is free.

Downloads are cached for the session (``http.enable_download_cache``), so only
the first drawing waits for the network; "Reload data" clears the cache.
"""

from __future__ import annotations

import calendar
import io
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from . import http
from .config import PlotConfig
from .options import (
    GROUPS,
    OPTIONS,
    Kind,
    Option,
    by_name,
    choices_from_config,
    command_line_tokens,
    config_from_choices,
    format_dollar_bound,
    group_title,
    parse_date_spec,
)
from .plotting import COUNTRIES, Country, build_figure

WINDOW_TITLE = "Rates plot: yields, public debt, GDP & interest"
_SCRIPT_NAME = "US_CDN_rates_plot.py"
# Pause after the last keystroke in a text field before the chart is redrawn,
# so typing "2001" does not try to draw 2, 20 and 200 on the way.
_TEXT_REDRAW_DELAY_MS = 700
# How often the window collects messages from the worker thread.
_POLL_MS = 100
# Pause after a window resize before the preview is rescaled.
_RESIZE_DELAY_MS = 150
# Pause after the last wheel notch before the zoom is applied (one redraw per gesture).
_WHEEL_DELAY_MS = 350
# Date range multiplier per wheel notch (towards the pointer).
_WHEEL_ZOOM_FACTOR = 0.8
# A drag shorter than this (image pixels) on an axis does not change that axis.
_MIN_DRAG_PX = 8
# How many zoom/pan steps "Back" can undo.
_HISTORY_LIMIT = 50
# The command line insists on at least seven days between start and end.
_MIN_SPAN_DAYS = 7
# Initial window size as a fraction of the screen (when nothing is remembered).
_SCREEN_FRACTION = 0.9
# Same PNG metadata as tools/verify_charts.py, so saved files hash identically.
_PNG_METADATA = {"Software": None}
_ERROR_FOREGROUND = "#b00020"
_TAB_TITLES = {"cdn": "Canada", "us": "United States"}
_STATE_VERSION = 1
_GEOMETRY_PATTERN = re.compile(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$")
_ZOOM_HINT = "Drag: zoom to box · right/middle-drag: pan · wheel: zoom dates"

Choices = tuple[dict[str, str], dict[str, bool]]


# ---------------------------------------------------------------------------
# Remembered settings
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
    if not isinstance(state, dict) or state.get("version") != _STATE_VERSION:
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
        # Curves, and countries, are overridden as a group: on the command line
        # "--gdp" means "GDP only", which the resolved config already reflects.
        for kind in (Kind.TOGGLE, Kind.SWITCH):
            group = [option.name for option in OPTIONS if option.kind is kind and option.name in flags]
            if any(name in cli_flags for name in group):
                for name in group:
                    flags[name] = cli_config_flags[name]
        for option in OPTIONS:
            if option.kind is Kind.EXACT and option.name in flags and option.name in cli_flags:
                flags[option.name] = cli_flags[option.name]
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
# Chart geometry: image pixels <-> data values, for zoom and pan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartGeometry:
    """Where the axes sit in a rendered chart image, and their limits.

    ``box`` is (left, top, right, bottom) in image pixels, origin top-left.
    x limits are matplotlib date numbers. "left"/"right" are the two y axes:
    the yield axis and its twin carrying the dollar series.
    """

    box: tuple[float, float, float, float]
    xlim: tuple[float, float]
    left_ylim: tuple[float, float]
    right_ylim: tuple[float, float]
    left_log: bool
    right_log: bool

    @classmethod
    def from_figure(cls, figure: Figure) -> ChartGeometry | None:
        """Measure a drawn figure (axes[0] is the yield axis, axes[1] its twin)."""
        if len(figure.axes) < 2:
            return None
        left, right = figure.axes[0], figure.axes[1]
        height = figure.bbox.height
        extent = left.get_window_extent()
        return cls(
            box=(extent.x0, height - extent.y1, extent.x1, height - extent.y0),
            xlim=tuple(left.get_xlim()),
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


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------


class _WorkerStdout(io.TextIOBase):
    """Stand-in for ``sys.stdout`` that sends the worker thread's output to the log pane.

    The fetchers report progress and warnings with ``print``. Swapping
    ``sys.stdout`` around the worker's run (``contextlib.redirect_stdout``)
    would capture every thread's output for that time, since ``sys.stdout`` is
    process-wide. This routes by thread instead: writes from the worker go to
    the window's message queue, everything else to the original stream.
    """

    def __init__(self, messages: queue.Queue, original) -> None:
        super().__init__()
        self._messages = messages
        self.original = original
        self.worker_ident: int | None = None

    def write(self, text: str) -> int:
        if threading.get_ident() == self.worker_ident:
            if text:
                self._messages.put(("log", text))
            return len(text)
        return self.original.write(text) if self.original is not None else len(text)

    def flush(self) -> None:
        if self.original is not None:
            self.original.flush()


class _Tooltip:
    """Show ``text`` in a small borderless window while the pointer is over ``widget``."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: tk.Event) -> None:
        if self.window is not None:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            self.window, text=self.text, justify="left", background="#ffffe0", relief="solid", borderwidth=1, padding=4
        ).pack()

    def _hide(self, _event: tk.Event) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


class DatePicker:
    """A small month calendar under a date field. Picking a day writes ``YYYY-MM-DD`` into it.

    Year and month can be typed or stepped, so distant decades are two clicks
    away. "Blank" empties the field, which means the option's default (today,
    for the end date).
    """

    _SELECTED = "#cfe3ff"
    _TODAY = "#fff2b3"

    def __init__(self, anchor: tk.Widget, variable: tk.StringVar, kind: str, on_pick) -> None:
        self.variable = variable
        self.on_pick = on_pick
        today = pd.Timestamp.today().normalize()
        try:
            self.selected = parse_date_spec(variable.get(), kind=kind) if variable.get().strip() else None
        except ValueError:
            self.selected = None
        shown = self.selected or today
        self.today = today
        self.year = tk.IntVar(value=shown.year)
        self.month = tk.StringVar(value=calendar.month_name[shown.month])

        self.top = tk.Toplevel(anchor)
        self.top.title("Pick a date")
        self.top.resizable(False, False)
        self.top.transient(anchor.winfo_toplevel())
        self.top.geometry(f"+{anchor.winfo_rootx()}+{anchor.winfo_rooty() + anchor.winfo_height() + 2}")
        self.top.bind("<Escape>", lambda _event: self.top.destroy())

        header = ttk.Frame(self.top, padding=4)
        header.pack(fill="x")
        ttk.Button(header, text="◀", width=3, command=lambda: self._step(-1)).pack(side="left")
        months = ttk.Combobox(
            header, textvariable=self.month, values=list(calendar.month_name)[1:], state="readonly", width=10
        )
        months.pack(side="left", padx=4)
        months.bind("<<ComboboxSelected>>", lambda _event: self._draw_days())
        years = ttk.Spinbox(header, from_=1900, to=2100, textvariable=self.year, width=6, command=self._draw_days)
        years.pack(side="left")
        years.bind("<Return>", lambda _event: self._draw_days())
        years.bind("<FocusOut>", lambda _event: self._draw_days())
        ttk.Button(header, text="▶", width=3, command=lambda: self._step(1)).pack(side="left", padx=(4, 0))

        self.days = ttk.Frame(self.top, padding=(4, 0))
        self.days.pack()
        footer = ttk.Frame(self.top, padding=4)
        footer.pack(fill="x")
        ttk.Button(footer, text="Today", command=lambda: self._pick(self.today)).pack(side="left")
        ttk.Button(footer, text="Blank (default)", command=self._blank).pack(side="left", padx=4)
        ttk.Button(footer, text="Cancel", command=self.top.destroy).pack(side="right")

        self._draw_days()
        self.top.focus_set()
        self.top.grab_set()

    def _month_number(self) -> int:
        return list(calendar.month_name).index(self.month.get())

    def _year_number(self) -> int | None:
        try:
            year = int(self.year.get())
        except (tk.TclError, ValueError):
            return None
        return year if 1000 <= year <= 9999 else None

    def _step(self, months: int) -> None:
        year = self._year_number()
        if year is None:
            return
        index = year * 12 + self._month_number() - 1 + months
        self.year.set(index // 12)
        self.month.set(calendar.month_name[index % 12 + 1])
        self._draw_days()

    def _draw_days(self) -> None:
        year = self._year_number()
        if year is None:
            return
        month = self._month_number()
        for child in self.days.winfo_children():
            child.destroy()
        for column, name in enumerate(("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")):
            ttk.Label(self.days, text=name, width=4, anchor="center").grid(row=0, column=column)
        for row, week in enumerate(calendar.monthcalendar(year, month), start=1):
            for column, day in enumerate(week):
                if day == 0:
                    continue
                date = pd.Timestamp(year=year, month=month, day=day)
                background = (
                    self._SELECTED if date == self.selected else self._TODAY if date == self.today else "white"
                )
                tk.Button(
                    self.days, text=str(day), width=3, relief="flat", background=background,
                    command=lambda picked=date: self._pick(picked),
                ).grid(row=row, column=column, padx=1, pady=1)

    def _pick(self, date: pd.Timestamp) -> None:
        self.variable.set(f"{date:%Y-%m-%d}")
        self.top.destroy()
        self.on_pick()

    def _blank(self) -> None:
        self.variable.set("")
        self.top.destroy()
        self.on_pick()


def _tooltip_text(option: Option) -> str:
    """Return an option's group notes, its help text and its command-line spelling."""
    return (
        f"{' '.join(GROUPS[option.group].split())}\n"
        f"{option.label}: {' '.join(option.help.split())}\n"
        f"Command line: {option.usage}"
    )


def _enable_windows_dpi_awareness() -> None:
    """Ask Windows to render this process at native resolution.

    Without it, on a scaled display (e.g. 150 %) Windows stretches the whole
    window as a bitmap and the chart preview turns blurry. Purely cosmetic, so
    if the call is unavailable (older Windows) the window simply stays as it
    would otherwise be, and a note is printed.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError) as exc:
        print(f"Note: could not enable high-DPI rendering ({exc}); the preview may look soft.")


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


class RatesPlotApp:
    """The main window.

    Build with a Tk root, the starting choices (see ``starting_choices``),
    notes to show in the log, and the remembered window state.
    """

    def __init__(self, root: tk.Tk, choices: Choices, notes: list[str] | None = None, state: dict | None = None) -> None:
        self.root = root
        self.messages: queue.Queue = queue.Queue()
        self.stdout = _WorkerStdout(self.messages, sys.stdout)
        sys.stdout = self.stdout
        state = state or {}

        # Controls, keyed by option name.
        self.payload_vars: dict[str, tk.StringVar] = {}
        self.flag_vars: dict[str, tk.BooleanVar] = {}
        self.value_labels: dict[str, ttk.Label] = {}

        # Per-country preview state, keyed by Country.key.
        self.tabs: dict[str, ttk.Frame] = {}
        self.canvases: dict[str, tk.Canvas] = {}
        self.scrollbars: dict[str, tuple[ttk.Scrollbar, ttk.Scrollbar]] = {}
        self.pngs: dict[str, bytes] = {}
        self.images: dict[str, Image.Image] = {}
        self.photos: dict[str, ImageTk.PhotoImage] = {}  # Tk shows an image only while a reference exists
        self.geometry: dict[str, ChartGeometry | None] = {}
        # Where each preview image sits on its canvas: (scale, left, top) in canvas coordinates.
        self.placement: dict[str, tuple[float, float, float]] = {}

        # Drawing state.
        self.rendering = False
        self.data_cached = False  # True once a drawing has filled the session download cache
        self.pending: tuple[PlotConfig, Choices] | None = None
        self.shown_config: PlotConfig | None = None
        self.target_config: PlotConfig | None = None  # what the worker is drawing now
        self.rendering_choices: Choices | None = None
        self.drawn_choices: Choices | None = None
        self._text_after_id: str | None = None
        self._resize_after_ids: dict[str, str] = {}
        self._loading = False

        # Mouse state.
        self.history: list[Choices] = []
        self._drag: dict | None = None
        self._wheel: dict | None = None

        self.zoom = tk.StringVar(value=state.get("zoom") if state.get("zoom") in ("fit", "actual") else "fit")
        self.command = tk.StringVar()
        self.status = tk.StringVar(value="Starting …")

        self._build_layout()
        self._load_choices(choices)
        if state.get("tab") in self.tabs:
            self.notebook.select(self.tabs[state["tab"]])
        for note in notes or []:
            self._append_log(note + "\n")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._poll_id = self.root.after(_POLL_MS, self._poll)
        self.request_redraw()

    # -- layout -------------------------------------------------------------

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        panel = ttk.Frame(self.root, padding=8)
        panel.grid(row=0, column=0, sticky="nsew")
        self._build_option_controls(panel)
        self._build_actions(panel)
        self._build_log(panel)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        for country in COUNTRIES:
            self._build_tab(country)

        bar = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="Command line:").grid(row=0, column=0, sticky="w")
        ttk.Entry(bar, textvariable=self.command, state="readonly").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(bar, text="Copy", command=self._copy_command).grid(row=0, column=2)
        ttk.Label(bar, textvariable=self.status, anchor="w").grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        self.root.bind("<F5>", lambda _event: self.request_redraw(force=True))
        self.root.bind("<Control-s>", lambda _event: self._save_png())
        self.root.bind("<Alt-Left>", lambda _event: self._back())

    def _build_option_controls(self, panel: ttk.Frame) -> None:
        """One labelled frame per option group, one control per GUI option, all from the table."""
        style = ttk.Style(self.root)
        style.configure("Error.TLabel", foreground=_ERROR_FOREGROUND)

        for group in GROUPS:
            members = [option for option in OPTIONS if option.group == group and option.in_gui]
            if not members:
                continue
            frame = ttk.LabelFrame(panel, text=group_title(group), padding=(8, 4))
            frame.pack(fill="x", pady=(0, 6))
            frame.columnconfigure(1, weight=1)
            for row, option in enumerate(members):
                if option.kind is Kind.VALUE:
                    self._add_value_control(frame, row, option)
                else:
                    self._add_flag_control(frame, row, option)

    def _add_flag_control(self, frame: ttk.LabelFrame, row: int, option: Option) -> None:
        variable = tk.BooleanVar(value=False)
        self.flag_vars[option.name] = variable
        box = ttk.Checkbutton(
            frame, text=option.label, variable=variable, command=lambda name=option.name: self._flag_changed(name)
        )
        box.grid(row=row, column=0, columnspan=3, sticky="w")
        _Tooltip(box, _tooltip_text(option))

    def _add_value_control(self, frame: ttk.LabelFrame, row: int, option: Option) -> None:
        variable = tk.StringVar()
        self.payload_vars[option.name] = variable
        label = ttk.Label(frame, text=option.label)
        label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
        entry = ttk.Entry(frame, textvariable=variable, width=16)
        entry.grid(row=row, column=1, sticky="ew", pady=1)
        entry.bind("<Return>", lambda _event: self.request_redraw())
        variable.trace_add("write", lambda *_args: self._text_changed())
        self.value_labels[option.name] = label
        for widget in (label, entry):
            _Tooltip(widget, _tooltip_text(option) + "\nBlank = default.")
        if option.picker == "date":
            button = ttk.Button(
                frame, text="📅", width=3,
                command=lambda: DatePicker(entry, variable, option.name, on_pick=self.request_redraw),
            )
            button.grid(row=row, column=2, padx=(2, 0), pady=1)
            _Tooltip(button, "Pick a date from a calendar")

    def _build_actions(self, panel: ttk.Frame) -> None:
        preview = ttk.LabelFrame(panel, text="Preview", padding=(8, 4))
        preview.pack(fill="x", pady=(0, 6))
        preview.columnconfigure((0, 1), weight=1)
        for column, (value, text) in enumerate((("fit", "Fit to window"), ("actual", "Actual size"))):
            ttk.Radiobutton(preview, text=text, value=value, variable=self.zoom, command=self._refresh_all_views).grid(
                row=0, column=column, sticky="w"
            )
        ttk.Button(preview, text="◀ Back (Alt+←)", command=self._back).grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(preview, text="Unzoom", command=self._unzoom).grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        ttk.Label(preview, text=_ZOOM_HINT, wraplength=300, foreground="#555555").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", pady=(0, 6))
        actions = (
            ("Redraw (F5)", lambda: self.request_redraw(force=True)),
            ("Save PNG… (Ctrl+S)", self._save_png),
            ("Reload data", self._reload_data),
            ("Reset to defaults", self._reset),
        )
        for index, (text, command) in enumerate(actions):
            ttk.Button(buttons, text=text, command=command).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=1, pady=1
            )
        buttons.columnconfigure((0, 1), weight=1)

    def _build_log(self, panel: ttk.Frame) -> None:
        frame = ttk.LabelFrame(panel, text="Log", padding=4)
        frame.pack(fill="both", expand=True)
        self.log = tk.Text(frame, width=44, height=8, wrap="word", state="disabled", font=("Consolas", 9))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_tab(self, country: Country) -> None:
        tab = ttk.Frame(self.notebook)
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        canvas = tk.Canvas(tab, background="white", highlightthickness=0, cursor="crosshair")
        x_scroll = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
        y_scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        key = country.key
        canvas.bind("<Configure>", lambda _event: self._schedule_view_refresh(key))
        canvas.bind("<ButtonPress-1>", lambda event: self._drag_start(key, event, "zoom"))
        canvas.bind("<B1-Motion>", lambda event: self._drag_motion(key, event))
        canvas.bind("<ButtonRelease-1>", lambda event: self._drag_end(key, event))
        for button in (2, 3):
            canvas.bind(f"<ButtonPress-{button}>", lambda event: self._drag_start(key, event, "pan"))
            canvas.bind(f"<B{button}-Motion>", lambda event: self._drag_motion(key, event))
            canvas.bind(f"<ButtonRelease-{button}>", lambda event: self._drag_end(key, event))
        canvas.bind("<MouseWheel>", lambda event: self._wheel_turned(key, event))
        self.notebook.add(tab, text=_TAB_TITLES.get(country.key, country.key))
        self.tabs[key] = tab
        self.canvases[key] = canvas
        self.scrollbars[key] = (x_scroll, y_scroll)

    # -- choices ------------------------------------------------------------

    def _load_choices(self, choices: Choices) -> None:
        """Set every control from ``choices`` without triggering a redraw per control."""
        payloads, flags = choices
        self._loading = True
        try:
            for name, text in payloads.items():
                if name in self.payload_vars:
                    self.payload_vars[name].set(text)
            for name, value in flags.items():
                if name in self.flag_vars:
                    self.flag_vars[name].set(value)
        finally:
            self._loading = False

    def _snapshot(self) -> Choices:
        """Return every control's current value, blanks included."""
        return (
            {name: var.get() for name, var in self.payload_vars.items()},
            {name: var.get() for name, var in self.flag_vars.items()},
        )

    def _flag_changed(self, name: str) -> None:
        # The command-line rule "no country means both" would make unticking
        # the last country draw both, which would look like a bug here.
        countries = [option.name for option in OPTIONS if option.kind is Kind.SWITCH and option.in_gui]
        if name in countries and not any(self.flag_vars[country].get() for country in countries):
            self.flag_vars[name].set(True)
            self.status.set("At least one country must stay selected.")
            return
        self.request_redraw()

    def _text_changed(self) -> None:
        if self._loading:
            return
        if self._text_after_id is not None:
            self.root.after_cancel(self._text_after_id)
        self._text_after_id = self.root.after(_TEXT_REDRAW_DELAY_MS, self.request_redraw)

    def _update_command_line(self) -> None:
        tokens = command_line_tokens(*self._snapshot())
        self.command.set(subprocess.list2cmdline(["python", _SCRIPT_NAME, *tokens]))

    def _mark_invalid_fields(self, payloads: dict[str, str]) -> list[str]:
        """Colour the label of every text field whose value its parser rejects; return the messages."""
        problems: list[str] = []
        for option in OPTIONS:
            if option.kind is not Kind.VALUE or option.name not in self.value_labels:
                continue
            style = "TLabel"
            if option.name in payloads:
                try:
                    option.parse(payloads[option.name])
                except ValueError as exc:
                    style = "Error.TLabel"
                    problems.append(str(exc))
            self.value_labels[option.name].configure(style=style)
        return problems

    # -- drawing ------------------------------------------------------------

    def request_redraw(self, force: bool = False) -> None:
        """Validate the controls and, if they describe a new chart, draw it."""
        if self._text_after_id is not None:
            self.root.after_cancel(self._text_after_id)
        self._text_after_id = None
        self._update_command_line()
        snapshot = self._snapshot()
        payloads = {name: text.strip() for name, text in snapshot[0].items() if text.strip()}
        problems = self._mark_invalid_fields(payloads)
        try:
            config = config_from_choices(payloads, snapshot[1])
        except ValueError as exc:
            # Field-level problems are listed first; a cross-field one (e.g.
            # end before start) has no single field to colour.
            self.status.set(f"Not drawn: {problems[0] if problems else exc}")
            return

        if force:
            self.shown_config = None
        if self.rendering:
            # Compare with what is being drawn, not what is on screen: going
            # back to the displayed view while a new one is being drawn must
            # still queue a drawing, or the old view would replace it.
            if config == self.target_config and not force:
                self.pending = None
                return
            self.pending = (config, snapshot)
            self.status.set("Will redraw with the new settings when the current drawing finishes …")
            return
        if config == self.shown_config:
            return
        self._start_render(config, snapshot)

    def _start_render(self, config: PlotConfig, choices: Choices) -> None:
        self.rendering = True
        self.target_config = config
        self.rendering_choices = choices
        countries = [country for country in COUNTRIES if getattr(config, country.show_field)]
        for country in COUNTRIES:
            self.notebook.tab(self.tabs[country.key], state="normal" if country in countries else "hidden")
        self.status.set("Drawing …")
        # The fetchers print "Downloading …" either way; say when the session
        # cache means nothing is actually downloaded.
        source = "reusing downloaded data" if self.data_cached else "downloading data"
        self._append_log(
            f"--- drawing {config.width_px}x{config.height_px}, "
            f"{config.start:%Y-%m-%d} to {config.end:%Y-%m-%d} ({source})\n"
        )
        threading.Thread(target=self._render_worker, args=(config, countries), daemon=True).start()

    def _render_worker(self, config: PlotConfig, countries: list[Country]) -> None:
        """Worker thread: draw each country to PNG bytes and post them. Never touches Tk."""
        self.stdout.worker_ident = threading.get_ident()
        try:
            for country in countries:
                figure = build_figure(country, config)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=figure.dpi, metadata=_PNG_METADATA)
                # Measured after savefig, so the axes positions are the final rendered ones.
                geometry = ChartGeometry.from_figure(figure)
                self.messages.put(("chart", country.key, buffer.getvalue(), geometry))
        except Exception as exc:  # reported in the window and log, never swallowed
            self.messages.put(("failed", config, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        else:
            self.messages.put(("done", config))

    def _poll(self) -> None:
        """Handle every message the worker has posted, then check again shortly."""
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]
                if kind == "log":
                    self._append_log(message[1])
                elif kind == "chart":
                    self._show_chart(message[1], message[2], message[3])
                elif kind == "done":
                    self._render_finished(message[1], None)
                elif kind == "failed":
                    self._append_log(message[3])
                    self._render_finished(message[1], message[2])
        except queue.Empty:
            pass
        self._poll_id = self.root.after(_POLL_MS, self._poll)

    def _show_chart(self, key: str, png: bytes, geometry: ChartGeometry | None) -> None:
        self.pngs[key] = png
        self.geometry[key] = geometry
        image = Image.open(io.BytesIO(png))
        image.load()
        self.images[key] = image
        self._refresh_view(key)

    def _render_finished(self, config: PlotConfig, error: str | None) -> None:
        self.rendering = False
        if error is None:
            self.shown_config = config
            self.drawn_choices = self.rendering_choices
            self.data_cached = True
            drawn = [_TAB_TITLES.get(c.key, c.key) for c in COUNTRIES if getattr(config, c.show_field)]
            self.status.set(f"Drawn: {' and '.join(drawn)} at {config.width_px} x {config.height_px} px.")
            self._remember()
        else:
            self.status.set(f"Drawing failed: {error} (details in the log)")
        pending, self.pending = self.pending, None
        if pending is not None and pending[0] != self.shown_config:
            self._start_render(*pending)

    # -- preview ------------------------------------------------------------

    def _schedule_view_refresh(self, key: str) -> None:
        previous = self._resize_after_ids.pop(key, None)
        if previous is not None:
            self.root.after_cancel(previous)
        self._resize_after_ids[key] = self.root.after(_RESIZE_DELAY_MS, lambda: self._refresh_view(key))

    def _refresh_all_views(self) -> None:
        for key in self.images:
            self._refresh_view(key)

    def _refresh_view(self, key: str) -> None:
        """Show the chart scaled to fit its tab (never enlarged), or at actual size with scroll bars."""
        self._resize_after_ids.pop(key, None)
        image = self.images.get(key)
        canvas = self.canvases[key]
        if image is None:
            return
        # Scroll bars only mean something at actual size.
        for scrollbar in self.scrollbars[key]:
            if self.zoom.get() == "fit":
                scrollbar.grid_remove()
            else:
                scrollbar.grid()
        canvas.update_idletasks()
        view_width, view_height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        canvas.delete("all")
        if self.zoom.get() == "fit":
            scale = min(view_width / image.width, view_height / image.height, 1.0)
            shown = image
            if scale < 1.0:
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                shown = image.resize(size, Image.Resampling.LANCZOS)
                scale = shown.width / image.width
            left, top = (view_width - shown.width) // 2, (view_height - shown.height) // 2
            canvas.configure(scrollregion=(0, 0, view_width, view_height))
        else:
            shown, scale, left, top = image, 1.0, 0, 0
            canvas.configure(scrollregion=(0, 0, image.width, image.height))
        self.photos[key] = ImageTk.PhotoImage(shown)
        canvas.create_image(left, top, image=self.photos[key], anchor="nw", tags="chart")
        self.placement[key] = (scale, left, top)

    # -- zoom and pan -------------------------------------------------------

    def _image_point(self, key: str, event: tk.Event) -> tuple[float, float] | None:
        """Return the event position in chart-image pixels, or None if no chart is shown."""
        if key not in self.placement or self.geometry.get(key) is None:
            return None
        canvas = self.canvases[key]
        scale, left, top = self.placement[key]
        return (canvas.canvasx(event.x) - left) / scale, (canvas.canvasy(event.y) - top) / scale

    def _drag_start(self, key: str, event: tk.Event, mode: str) -> None:
        point = self._image_point(key, event)
        if point is None:
            return
        canvas = self.canvases[key]
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        self._drag = {"key": key, "mode": mode, "image": point, "canvas": (x, y), "last": (x, y), "band": None}
        if mode == "zoom":
            self._drag["band"] = canvas.create_rectangle(x, y, x, y, outline="#1f5fbf", dash=(4, 2), width=2)
        else:
            canvas.configure(cursor="fleur")

    def _drag_motion(self, key: str, event: tk.Event) -> None:
        drag = self._drag
        if drag is None or drag["key"] != key:
            return
        canvas = self.canvases[key]
        x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
        if drag["mode"] == "zoom":
            x0, y0 = drag["canvas"]
            canvas.coords(drag["band"], x0, y0, x, y)
        else:
            # Move the picture with the pointer for immediate feedback; the real
            # redraw happens on release.
            last_x, last_y = drag["last"]
            canvas.move("chart", x - last_x, y - last_y)
            drag["last"] = (x, y)

    def _drag_end(self, key: str, event: tk.Event) -> None:
        drag, self._drag = self._drag, None
        if drag is None or drag["key"] != key:
            return
        canvas = self.canvases[key]
        canvas.configure(cursor="crosshair")
        if drag["band"] is not None:
            canvas.delete(drag["band"])
        end = self._image_point(key, event)
        if end is None:
            return
        (x0, y0), (x1, y1) = drag["image"], end
        if drag["mode"] == "zoom":
            self._zoom_to_box(key, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        else:
            self._pan(key, x1 - x0, y1 - y0)

    def _axis_updates(self, key: str, top_y: float, bottom_y: float) -> dict[str, str]:
        """Return field texts pinning the drawn y axes to image rows ``top_y``..``bottom_y``."""
        geometry, config = self.geometry[key], self.shown_config
        updates: dict[str, str] = {}
        if config is None or geometry is None:
            return updates
        if config.include_yield:
            low = max(0.0, geometry.value_at(bottom_y, "left"))
            high = min(99.99, geometry.value_at(top_y, "left"))
            if high - low >= 0.01:
                updates.update({"min": _format_percent(low), "max": _format_percent(high)})
        if config.has_dollar_series:
            low, high = geometry.value_at(bottom_y, "right"), geometry.value_at(top_y, "right")
            if 0 < low < high:
                updates.update({"bottom": _format_dollars(low), "top": _format_dollars(high)})
        return updates

    def _zoom_to_box(self, key: str, x0: float, y0: float, x1: float, y1: float) -> None:
        geometry = self.geometry.get(key)
        if geometry is None:
            return
        updates: dict[str, str] = {}
        if x1 - x0 >= _MIN_DRAG_PX:
            updates.update(_date_range_texts(geometry.date_at(x0), geometry.date_at(x1)))
        if y1 - y0 >= _MIN_DRAG_PX:
            updates.update(self._axis_updates(key, y0, y1))
        if updates:
            self._apply_view(updates, "Zoomed")

    def _pan(self, key: str, dx: float, dy: float) -> None:
        """Shift the view so the point under the pointer moves by (dx, dy) image pixels."""
        geometry = self.geometry.get(key)
        if geometry is None:
            return
        left, top, right, bottom = geometry.box
        updates: dict[str, str] = {}
        if abs(dx) >= _MIN_DRAG_PX:
            updates.update(_date_range_texts(geometry.date_at(left - dx), geometry.date_at(right - dx)))
        if abs(dy) >= _MIN_DRAG_PX:
            updates.update(self._axis_updates(key, top - dy, bottom - dy))
        if updates:
            self._apply_view(updates, "Panned")
        else:
            self._refresh_view(key)  # put the picture back where it was

    def _wheel_turned(self, key: str, event: tk.Event) -> None:
        """Collect wheel notches; zoom the dates about the pointer once the wheel stops."""
        point = self._image_point(key, event)
        if point is None:
            return
        if self._wheel is None or self._wheel["key"] != key:
            self._wheel = {"key": key, "notches": 0.0, "x": point[0], "after": None}
        wheel = self._wheel
        wheel["notches"] += event.delta / 120.0  # Windows: one notch = 120, positive = away = zoom in
        wheel["x"] = point[0]
        if wheel["after"] is not None:
            self.root.after_cancel(wheel["after"])
        wheel["after"] = self.root.after(_WHEEL_DELAY_MS, self._apply_wheel)
        self.status.set(f"Zoom ×{_WHEEL_ZOOM_FACTOR ** -wheel['notches']:.2f} (applies when the wheel stops) …")

    def _apply_wheel(self) -> None:
        wheel, self._wheel = self._wheel, None
        if wheel is None or self.geometry.get(wheel["key"]) is None:
            return
        geometry = self.geometry[wheel["key"]]
        left, _top, right, _bottom = geometry.box
        factor = _WHEEL_ZOOM_FACTOR ** wheel["notches"]
        anchor = wheel["x"]
        new_left = anchor - (anchor - left) * factor
        new_right = anchor + (right - anchor) * factor
        self._apply_view(_date_range_texts(geometry.date_at(new_left), geometry.date_at(new_right)), "Zoomed")

    def _apply_view(self, updates: dict[str, str], verb: str) -> None:
        """Write zoom/pan results into the fields (undoably) and redraw."""
        self.history.append(self._snapshot())
        del self.history[:-_HISTORY_LIMIT]
        self._loading = True
        try:
            for name, text in updates.items():
                self.payload_vars[name].set(text)
        finally:
            self._loading = False
        changed = ", ".join(f"{by_name(name).label} {text or '(today)'}" for name, text in updates.items())
        self._append_log(f"--- {verb.lower()}: {changed}\n")
        self.request_redraw()

    def _back(self) -> None:
        if not self.history:
            self.status.set("Nothing to go back to.")
            return
        self._load_choices(self.history.pop())
        self.request_redraw()

    def _unzoom(self) -> None:
        if not self.history:
            self.status.set("Not zoomed.")
            return
        first = self.history[0]
        self.history.clear()
        self._load_choices(first)
        self.request_redraw()

    # -- actions ------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")
        last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        if last_line and self.rendering:
            self.status.set(last_line)

    def _current_tab_key(self) -> str | None:
        selected = self.notebook.select()
        for key, tab in self.tabs.items():
            if str(tab) == selected:
                return key
        return None

    def _save_png(self) -> None:
        key = self._current_tab_key()
        if key is None or key not in self.pngs:
            self.status.set("Nothing drawn in this tab yet.")
            return
        config = self.shown_config or PlotConfig()
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save chart as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"{key}_rates_{config.start:%Y-%m-%d}_{config.end:%Y-%m-%d}.png",
        )
        if not path:
            return
        with open(path, "wb") as output:
            output.write(self.pngs[key])
        self.status.set(f"Saved {path}")

    def _reload_data(self) -> None:
        http.clear_download_cache()
        self.data_cached = False
        self._append_log("--- download cache cleared\n")
        self.request_redraw(force=True)

    def _reset(self) -> None:
        self.history.append(self._snapshot())
        self._load_choices(choices_from_config(PlotConfig()))
        self.request_redraw()

    def _copy_command(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.command.get())
        self.status.set("Command line copied.")

    def _remember(self) -> None:
        """Save the last drawn choices and the window layout for the next session."""
        if self.drawn_choices is None:
            return
        payloads, flags = self.drawn_choices
        error = save_state(
            {
                "version": _STATE_VERSION,
                "payloads": payloads,
                "flags": flags,
                "zoom": self.zoom.get(),
                "tab": self._current_tab_key(),
                "geometry": self.root.geometry(),
            }
        )
        if error is not None:
            self._append_log(error + "\n")

    def _close(self) -> None:
        self._remember()
        sys.stdout = self.stdout.original
        self.root.after_cancel(self._poll_id)  # or it fires once more after the window is gone
        self.root.destroy()


def _initial_geometry(root: tk.Tk, state: dict) -> str:
    """Return the remembered window geometry if it still fits on screen, else 90 % of the screen."""
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    match = _GEOMETRY_PATTERN.match(str(state.get("geometry", "")))
    if match:
        width, height, x, y = map(int, match.groups())
        if width <= screen_width and height <= screen_height and 0 <= x < screen_width - 100 and 0 <= y < screen_height - 100:
            return f"{width}x{height}+{x}+{y}"
    return f"{round(screen_width * _SCREEN_FRACTION)}x{round(screen_height * _SCREEN_FRACTION)}+20+20"


def run_gui(
    config: PlotConfig, cli_payloads: dict[str, str] | None = None, cli_flags: dict[str, bool] | None = None
) -> None:
    """Open the window and run until it is closed.

    ``config`` is what the command line described; ``cli_payloads`` and
    ``cli_flags`` are the options it named, which override the remembered
    settings (see ``starting_choices``).
    """
    _enable_windows_dpi_awareness()
    http.enable_download_cache()
    state, note = load_state()
    choices, notes = starting_choices(config, cli_payloads or {}, cli_flags or {}, state)
    if note:
        notes.insert(0, note)
    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry(_initial_geometry(root, state))
    RatesPlotApp(root, choices, notes, state)
    root.mainloop()
