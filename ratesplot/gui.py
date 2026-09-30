"""The interactive window: every chart option adjustable, charts redrawn on the fly.

Opened by default (``--gui``); ``--no-gui`` gives the plain matplotlib windows.

Controls
    Generated from ``options.OPTIONS``: a check box for every on/off option, a
    text field for every VALUE option (``editor="date"`` adds a calendar
    button; ``editor="size"`` gives width and height boxes with an
    aspect-ratio lock; ``editor="levels"`` gives one check box per level of
    government, the single "By level" control for ``--debt:`` and
    ``--interest:``), grouped as in ``--help``, each with the option's
    help text and command-line spelling as a tooltip. A new option in the table
    appears here with no change to this module (``in_gui=False`` opts out). The
    choices become a ``PlotConfig`` through ``options.config_from_choices``, the
    same code the command line uses, so the two cannot disagree. The measure
    switches (-r, -p) exclude each other: ticking one unticks the other.
    Changing the measure clears Top and Bottom, whose units it changes
    (dollars, percent, dollars per person), and under -r the GDP box is
    greyed out, GDP being only the denominator.

Remembered settings
    The choices of the last successful drawing, the preview mode, the selected
    tab and the window size are saved to ``gui_state.json`` (see
    ``frontend.state_path``) after every drawing and on closing, and restored at the
    next start. Options given on the command line override the remembered ones
    they name: a value option individually; the curves, the countries or the
    measure each as a group, because naming one curve on the command line
    means "only this one".

Zoom and pan
    The mouse edits the same fields a user would type in, so the fields remain
    the single source of truth (and the command line, the remembered settings
    and "Save PNG" all follow). Left-drag a box to zoom to it: its width sets
    Start/End, its height sets the yield Min/Max and the right-axis Top/Bottom of
    whichever axes are drawn. Right- or middle-drag pans. The wheel zooms the
    dates about the pointer. "Back" undoes one zoom or pan; "Unzoom" returns to
    the view before the first one. Axis limits are shared options, so zooming
    one country's chart applies them to the other too, as on the command line.

Drawing
    Each chart is drawn exactly as the command line draws it (the same
    ``plotting.draw_country``) on an off-screen Agg figure at the configured
    pixel size, rendered to PNG, and shown scaled to fit the window (or at
    actual size). "Save PNG" writes those same bytes, so a saved chart is the
    real output, not a screenshot of the preview. A blank Start is the
    automatic start, found for the charts drawn together
    (``plotting.resolve_start``) each time they are drawn, so it follows the
    chosen curves; the log names it.

Threading
    Fetching and drawing run on one worker thread at a time so the window stays
    responsive during downloads. The worker never touches Tk; it posts messages
    to a queue that the window polls. Its printed progress and warnings are
    captured into the log pane (``console`` routes them by thread). If the choices change while a chart is being
    drawn, the newest choices are drawn as soon as the worker is free.

Downloads are cached for the session (``http.enable_download_cache``), so only
the first drawing waits for the network; "Reload data" clears the cache.
"""

from __future__ import annotations

import calendar
import io
import math
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from tkinter import filedialog, ttk

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from . import http
from .config import COMPONENT_LETTERS, MIN_WINDOW_DAYS, PlotConfig
from .console import this_thread_output_to
from .frontend import (
    LEVEL_BOXES,
    STATE_VERSION,
    TAB_TITLES,
    Choices,
    level_letters,
    load_state,
    option_help_lines,
    png_bytes,
    save_state,
    saved_png_name,
    starting_choices,
)
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
    options_in,
    options_of,
    parse_date_spec,
)
from .plotting import COUNTRIES, Country, build_figure, resolve_start

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
_MIN_SPAN_DAYS = MIN_WINDOW_DAYS
# Initial window size as a fraction of the screen (when nothing is remembered).
_SCREEN_FRACTION = 0.9
_ERROR_FOREGROUND = "#b00020"
_GEOMETRY_PATTERN = re.compile(r"^(\d+)x(\d+)\+(-?\d+)\+(-?\d+)$")
_ZOOM_HINT = "Drag: zoom to box · right/middle-drag: pan · wheel: zoom dates"


# ---------------------------------------------------------------------------
# Chart geometry: image pixels <-> data values, for zoom and pan
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------


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
    for the end date; for the start, the first date on which every chosen
    curve has data).
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


class _LevelsEditor:
    """Keeps one check box per level of government and the option's letters (e.g. ``"fp"``) in step.

    See ``RatesPlotApp._add_levels_control``. The letters variable stays the
    single source of truth (for drawing, the command line, Back and the
    remembered settings); the boxes are a view of it. ``syncing`` stops the
    two sides' updates from triggering each other.
    """

    def __init__(self, payload: tk.StringVar, on_change) -> None:
        self.payload = payload
        self.on_change = on_change
        self.boxes = {letter: tk.BooleanVar(value=False) for letter in COMPONENT_LETTERS}
        self.syncing = False
        self.payload.trace_add("write", lambda *_args: self._payload_changed())

    def box_changed(self) -> None:
        if self.syncing:
            return
        self.syncing = True
        try:
            self.payload.set("".join(letter for letter in COMPONENT_LETTERS if self.boxes[letter].get()))
        finally:
            self.syncing = False
        self.on_change()

    def _payload_changed(self) -> None:
        """Show letters set from elsewhere (loading, Back, Reset) in the boxes."""
        if self.syncing:
            return
        letters = level_letters(self.payload.get())
        self.syncing = True
        try:
            for letter, box in self.boxes.items():
                box.set(letter in letters)
        finally:
            self.syncing = False


class _SizeEditor:
    """Keeps two boxes (width, height) and one ``WxH`` text variable in step.

    See ``RatesPlotApp._add_size_control``. ``syncing`` stops the three
    variables' write traces from triggering each other.
    """

    def __init__(self, payload: tk.StringVar, keep_ratio: bool) -> None:
        self.payload = payload
        self.width = tk.StringVar()
        self.height = tk.StringVar()
        self.keep_ratio = tk.BooleanVar(value=keep_ratio)
        self.ratio: float | None = None  # width / height, used while the lock is on
        self.syncing = False
        self.width.trace_add("write", lambda *_args: self._box_changed("width"))
        self.height.trace_add("write", lambda *_args: self._box_changed("height"))
        self.payload.trace_add("write", lambda *_args: self._payload_changed())

    @staticmethod
    def _number(text: str) -> int | None:
        text = text.strip()
        return int(text) if text.isdigit() and int(text) > 0 else None

    def _remember_ratio(self) -> None:
        width, height = self._number(self.width.get()), self._number(self.height.get())
        if width and height:
            self.ratio = width / height

    def lock_changed(self) -> None:
        # Ticking the lock freezes the shape the boxes have now.
        if self.keep_ratio.get():
            self._remember_ratio()

    def _box_changed(self, which: str) -> None:
        if self.syncing:
            return
        self.syncing = True
        try:
            if self.keep_ratio.get() and self.ratio:
                changed = self.width if which == "width" else self.height
                other = self.height if which == "width" else self.width
                value = self._number(changed.get())
                if value is not None:
                    other.set(str(max(1, round(value / self.ratio if which == "width" else value * self.ratio))))
                elif not changed.get().strip():
                    other.set("")
            else:
                self._remember_ratio()
            width, height = self.width.get().strip(), self.height.get().strip()
            self.payload.set(f"{width}x{height}" if width and height else width or (f"x{height}" if height else ""))
        finally:
            self.syncing = False

    def _payload_changed(self) -> None:
        """Show a ``WxH`` text set from elsewhere (loading, Back, Reset) in the boxes."""
        if self.syncing:
            return
        text = self.payload.get().strip().lower()
        if "x" in text:
            width, height = (part.strip() for part in text.split("x", 1))
        else:
            width, height = text, ""
        # One dimension alone means 4:3 on the command line; show the height it implies.
        if width.isdigit() and not height:
            height = str(round(int(width) * 3 / 4))
        elif height.isdigit() and not width:
            width = str(round(int(height) * 4 / 3))
        self.syncing = True
        try:
            self.width.set(width)
            self.height.set(height)
        finally:
            self.syncing = False
        self._remember_ratio()


def _tooltip_text(option: Option) -> str:
    """Return an option's group notes, its help text and its command-line spelling, one per line."""
    return "\n".join(option_help_lines(option))


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
        state = state or {}

        # Controls, keyed by option name.
        self.payload_vars: dict[str, tk.StringVar] = {}
        self.flag_vars: dict[str, tk.BooleanVar] = {}
        self.flag_boxes: dict[str, ttk.Checkbutton] = {}
        self.value_labels: dict[str, ttk.Label] = {}
        self.value_entries: dict[str, ttk.Entry] = {}
        self.size_editors: dict[str, _SizeEditor] = {}
        self.level_editors: dict[str, _LevelsEditor] = {}

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
        # The shown charts' config with the automatic start filled in (the
        # dates they were drawn for); shown_config keeps a blank start blank.
        self.drawn_config: PlotConfig | None = None
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
        for name, keep in state.get("keep_ratio", {}).items():
            if name in self.size_editors and isinstance(keep, bool):
                self.size_editors[name].keep_ratio.set(keep)
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

        panel = self._build_control_column()
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

    def _build_control_column(self) -> ttk.Frame:
        """Return the frame the controls go in: a column that scrolls when the window is too short for it.

        On a laptop screen (a 972 px window) the controls filled the column
        exactly, with the log out of sight, and one more row would have hidden
        the Redraw and Save buttons. So the column is a canvas holding the
        controls' frame: when the window is taller than the controls need, the
        frame is stretched to the window and the log takes the rest, as
        before; when it is shorter, a scroll bar appears, and the mouse wheel
        scrolls the column while the pointer is over it (over the log, the log).
        """
        outer = ttk.Frame(self.root)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(0, weight=1)
        background = ttk.Style(self.root).lookup("TFrame", "background") or None
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, background=background)
        canvas.grid(row=0, column=0, sticky="ns")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        panel = ttk.Frame(canvas, padding=8)
        window = canvas.create_window(0, 0, window=panel, anchor="nw")

        def fit(_event=None) -> None:
            width, needed = panel.winfo_reqwidth(), panel.winfo_reqheight()
            height = max(needed, canvas.winfo_height())
            canvas.configure(width=width, scrollregion=(0, 0, width, height))
            canvas.itemconfigure(window, width=width, height=height)
            if needed > canvas.winfo_height():
                scroll.grid(row=0, column=1, sticky="ns")
            else:
                scroll.grid_remove()
                canvas.yview_moveto(0)

        def wheel(event) -> None:
            # Only with the pointer over the column; the charts zoom with their
            # own binding, and the log scrolls itself.
            try:
                widget = self.root.winfo_containing(event.x_root, event.y_root)
            except (KeyError, tk.TclError):  # Tk's own pop-ups have no tkinter widget
                return
            if isinstance(widget, tk.Text) or not scroll.winfo_ismapped():
                return
            while widget is not None and widget is not outer:
                widget = widget.master
            if widget is outer:
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        canvas.bind("<Configure>", fit)
        panel.bind("<Configure>", fit)
        self.root.bind_all("<MouseWheel>", wheel, add="+")
        return panel

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
            # A value that belongs to a flag (the regression tolerance) goes on
            # the flag's own row: on a 972 px window one more row would put the
            # Redraw and Save buttons below the fold.
            attached = {option.turns_on: option for option in members if option.kind is Kind.VALUE and option.turns_on}
            rows = [option for option in members if option not in attached.values()]
            for row, option in enumerate(rows):
                if option.kind is Kind.VALUE and option.editor == "size":
                    self._add_size_control(frame, row, option)
                elif option.kind is Kind.VALUE and option.editor == "levels":
                    self._add_levels_control(frame, row, option)
                elif option.kind is Kind.VALUE:
                    self._add_value_control(frame, row, option)
                else:
                    self._add_flag_control(frame, row, option, attached.get(option.name))

    def _add_flag_control(self, frame: ttk.LabelFrame, row: int, option: Option, value: Option | None = None) -> None:
        """A check box; with ``value``, that option's field at the right of the same row."""
        variable = tk.BooleanVar(value=False)
        self.flag_vars[option.name] = variable
        box = ttk.Checkbutton(
            frame, text=option.label, variable=variable, command=lambda name=option.name: self._flag_changed(name)
        )
        box.grid(row=row, column=0, columnspan=3 if value is None else 1, sticky="w")
        self.flag_boxes[option.name] = box
        _Tooltip(box, _tooltip_text(option))
        if value is not None:
            field = ttk.Frame(frame)
            field.grid(row=row, column=1, columnspan=2, sticky="e")
            # No padding: the row stays as tall as a check box.
            self._add_value_control(field, 0, value, entry_width=5, pady=0)

    def _add_value_control(self, frame: ttk.Frame, row: int, option: Option, entry_width: int = 16, pady: int = 1) -> None:
        variable = tk.StringVar()
        self.payload_vars[option.name] = variable
        label = ttk.Label(frame, text=option.label)
        label.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=pady)
        entry = ttk.Entry(frame, textvariable=variable, width=entry_width)
        entry.grid(row=row, column=1, sticky="ew", pady=pady)
        entry.bind("<Return>", lambda _event: self.request_redraw())
        variable.trace_add("write", lambda *_args: self._text_changed())
        self.value_labels[option.name] = label
        self.value_entries[option.name] = entry
        for widget in (label, entry):
            _Tooltip(widget, _tooltip_text(option) + "\nBlank = default.")
        if option.editor == "date":
            button = ttk.Button(
                frame, text="📅", width=3,
                command=lambda: DatePicker(entry, variable, option.name, on_pick=self.request_redraw),
            )
            button.grid(row=row, column=2, padx=(2, 0), pady=1)
            _Tooltip(button, "Pick a date from a calendar")

    def _add_levels_control(self, frame: ttk.LabelFrame, row: int, option: Option) -> None:
        """One check box per level of government, writing the option's letters ("fp" …).

        Ticking none gives the aggregate lines. The one control serves debt and
        interest alike (``--debt:`` and ``--interest:`` take the same letters).
        """
        payload = tk.StringVar()
        self.payload_vars[option.name] = payload
        label = ttk.Label(frame, text=option.label)
        label.grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=(3, 1))
        self.value_labels[option.name] = label
        editor = _LevelsEditor(payload, on_change=self.request_redraw)
        self.level_editors[option.name] = editor
        box = ttk.Frame(frame)
        box.grid(row=row, column=1, columnspan=2, sticky="w", pady=1)
        tip = _tooltip_text(option) + "\nNone ticked = debt and interest in total."
        for index, (letter, text) in enumerate(LEVEL_BOXES):
            check = ttk.Checkbutton(box, text=text, variable=editor.boxes[letter], command=editor.box_changed)
            check.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 8))
            _Tooltip(check, tip)
        _Tooltip(label, tip)

    def _add_size_control(self, frame: ttk.LabelFrame, row: int, option: Option) -> None:
        """Width and height boxes, "x" between them, and a "Preserve aspect ratio" lock.

        The option's own text variable (``WxH``, the command-line form) stays
        the single source of truth for drawing, the command line, Back/Unzoom
        and the remembered settings; the two boxes are a view of it and write
        it back. With the lock on, changing one box recomputes the other from
        the ratio the pair had when the lock was last set (or the pair was
        loaded); with it off, each box changes alone. A blank box with the
        other filled keeps the command line's meaning: 4:3 from the one given.
        """
        payload = tk.StringVar()
        self.payload_vars[option.name] = payload
        label = ttk.Label(frame, text=option.label)
        label.grid(row=row, column=0, sticky="nw", padx=(0, 6), pady=(3, 1))
        self.value_labels[option.name] = label

        box = ttk.Frame(frame)
        box.grid(row=row, column=1, columnspan=2, sticky="w", pady=1)
        editor = _SizeEditor(payload, keep_ratio=True)
        self.size_editors[option.name] = editor
        width_entry = ttk.Entry(box, textvariable=editor.width, width=7, justify="right")
        width_entry.grid(row=0, column=0)
        ttk.Label(box, text=" x ").grid(row=0, column=1)
        height_entry = ttk.Entry(box, textvariable=editor.height, width=7, justify="right")
        height_entry.grid(row=0, column=2)
        ttk.Label(box, text=" px").grid(row=0, column=3)
        lock = ttk.Checkbutton(box, text="Preserve aspect ratio", variable=editor.keep_ratio, command=editor.lock_changed)
        lock.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

        payload.trace_add("write", lambda *_args: self._text_changed())
        for entry in (width_entry, height_entry):
            entry.bind("<Return>", lambda _event: self.request_redraw())
        tip = _tooltip_text(option) + "\nWidth x height in pixels. Blank = default."
        for widget in (label, width_entry, height_entry):
            _Tooltip(widget, tip)
        _Tooltip(lock, "When ticked, changing one box changes the other to keep the current shape.")

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
        self.notebook.add(tab, text=TAB_TITLES.get(country.key, country.key))
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
        self._update_dependent_controls()

    def _snapshot(self) -> Choices:
        """Return every control's current value, blanks included."""
        return (
            {name: var.get() for name, var in self.payload_vars.items()},
            {name: var.get() for name, var in self.flag_vars.items()},
        )

    def _flag_changed(self, name: str) -> None:
        # The command-line rule "no country means both" would make unticking
        # the last country draw both, which would look like a bug here.
        countries = [option.name for option in options_in("country") if option.in_gui]
        if name in countries and not any(self.flag_vars[country].get() for country in countries):
            self.flag_vars[name].set(True)
            self.status.set("At least one country must stay selected.")
            return
        units = [option.name for option in options_in("units") if option.name in self.flag_vars]
        if name in units:
            # One measure at a time: ticking one unticks the others.
            if self.flag_vars[name].get():
                for other in units:
                    if other != name:
                        self.flag_vars[other].set(False)
            # Top and Bottom are in the measure's units (dollars, or percent
            # under -r), so values typed or zoomed for one measure are wrong
            # for another: clear them without a redraw per field.
            self._loading = True
            try:
                for bound in ("top", "bottom"):
                    if bound in self.payload_vars:
                        self.payload_vars[bound].set("")
            finally:
                self._loading = False
        self._update_dependent_controls()
        self.request_redraw()

    def _update_dependent_controls(self) -> None:
        """Grey out controls that have no effect: GDP under -r, where it is the
        denominator and is not drawn, and a value whose flag is off (the
        regression tolerance without --reg)."""
        relative = "relative" in self.flag_vars and self.flag_vars["relative"].get()
        if "gdp" in self.flag_boxes:
            self.flag_boxes["gdp"].state(["disabled"] if relative else ["!disabled"])
        for option in options_of(Kind.VALUE):
            if option.turns_on in self.flag_vars and option.name in self.value_entries:
                state = ["!disabled"] if self.flag_vars[option.turns_on].get() else ["disabled"]
                self.value_entries[option.name].state(state)
                self.value_labels[option.name].state(state)

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
        # A value whose flag is off has no effect and its field is greyed:
        # a bad one left there must not stop the drawing.
        for option in options_of(Kind.VALUE):
            if option.turns_on in snapshot[1] and not snapshot[1][option.turns_on]:
                payloads.pop(option.name, None)
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
        start = "the automatic start" if config.start is None else f"{config.start:%Y-%m-%d}"
        self._append_log(
            f"--- drawing {config.width_px}x{config.height_px}, "
            f"{start} to {config.end:%Y-%m-%d} ({source})\n"
        )
        threading.Thread(target=self._render_worker, args=(config, countries), daemon=True).start()

    def _render_worker(self, config: PlotConfig, countries: list[Country]) -> None:
        """Worker thread: draw each country to PNG bytes and post them. Never touches Tk.

        A blank Start is found first, once for all the charts (``resolve_start``).
        What this thread prints goes to the log pane (``console``); the main
        thread's output goes to the console as before.
        """
        with this_thread_output_to(lambda text: self.messages.put(("log", text))):
            try:
                drawn = resolve_start(config)
                for country in countries:
                    figure = build_figure(country, drawn)
                    png = png_bytes(figure)
                    # Measured after savefig, so the axes positions are the final rendered ones.
                    geometry = ChartGeometry.from_figure(figure, drawn)
                    self.messages.put(("chart", country.key, png, geometry))
            except Exception as exc:  # reported in the window and log, never swallowed
                self.messages.put(("failed", config, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
            else:
                self.messages.put(("done", config, drawn))

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
                    self._render_finished(message[1], None, message[2])
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

    def _render_finished(self, config: PlotConfig, error: str | None, drawn: PlotConfig | None = None) -> None:
        self.rendering = False
        if error is None:
            self.shown_config = config
            self.drawn_config = drawn
            self.drawn_choices = self.rendering_choices
            self.data_cached = True
            drawn = [TAB_TITLES.get(c.key, c.key) for c in COUNTRIES if getattr(config, c.show_field)]
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
        if config.has_macro_series:
            low, high = geometry.value_at(bottom_y, "right"), geometry.value_at(top_y, "right")
            if 0 < low < high:
                bound_text = _format_relative if config.relative else _format_dollars
                updates.update({"bottom": bound_text(low), "top": bound_text(high)})
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
        _left, top, _right, bottom = geometry.box
        updates: dict[str, str] = {}
        if abs(dx) >= _MIN_DRAG_PX:
            # The requested window moves, not the whole axis: its pad (and the
            # -l label room) would otherwise be added to the window at every pan.
            start_x, end_x = geometry.window_x()
            updates.update(_date_range_texts(geometry.date_at(start_x - dx), geometry.date_at(end_x - dx)))
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
        start_x, end_x = geometry.window_x()  # the requested window, as for a pan
        factor = _WHEEL_ZOOM_FACTOR ** wheel["notches"]
        anchor = wheel["x"]
        new_left = anchor - (anchor - start_x) * factor
        new_right = anchor + (end_x - anchor) * factor
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
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save chart as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=saved_png_name(key, self.drawn_config),
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
                "version": STATE_VERSION,
                "payloads": payloads,
                "flags": flags,
                "zoom": self.zoom.get(),
                "tab": self._current_tab_key(),
                "geometry": self.root.geometry(),
                "keep_ratio": {name: editor.keep_ratio.get() for name, editor in self.size_editors.items()},
            }
        )
        if error is not None:
            self._append_log(error + "\n")

    def _close(self) -> None:
        self._remember()
        # Every callback still waiting, or it fires after the window is gone
        # ("invalid command name ..."): the poll, and a view refresh after a
        # resize, a redraw after typing, or a zoom after the wheel stops.
        pending = [self._poll_id, self._text_after_id, *self._resize_after_ids.values()]
        if self._wheel is not None:
            pending.append(self._wheel["after"])
        for after_id in pending:
            if after_id is not None:
                self.root.after_cancel(after_id)
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
