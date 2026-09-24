"""The interactive window: every chart option adjustable, charts redrawn on the fly.

Opened by default (``--gui``); any other command-line options only set the
window's starting values. ``--no-gui`` gives the plain matplotlib windows.

The controls are generated from ``options.OPTIONS``: a check box for every
on/off option, a text field for every VALUE option, grouped as in ``--help``,
each with the option's help text and command-line spelling as a tooltip. A new
option in the table appears here with no change to this module (set
``in_gui=False`` on options that should not). The choices are turned into a
``PlotConfig`` by ``options.config_from_choices`` - the same code the command
line uses - so the window and the command line cannot disagree.

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

import contextlib
import io
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from . import http
from .config import PlotConfig
from .options import (
    GROUPS,
    OPTIONS,
    Kind,
    Option,
    choices_from_config,
    command_line_tokens,
    config_from_choices,
    group_title,
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
# Initial window size as a fraction of the screen.
_SCREEN_FRACTION = 0.9
# Same PNG metadata as tools/verify_charts.py, so saved files hash identically.
_PNG_METADATA = {"Software": None}
_ERROR_FOREGROUND = "#b00020"
_TAB_TITLES = {"cdn": "Canada", "us": "United States"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


class _QueueWriter(io.TextIOBase):
    """A text stream that forwards everything written to it to the window's message queue."""

    def __init__(self, messages: queue.Queue) -> None:
        super().__init__()
        self._messages = messages

    def write(self, text: str) -> int:
        if text:
            self._messages.put(("log", text))
        return len(text)


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
    """The main window. Build with a Tk root and the starting PlotConfig."""

    def __init__(self, root: tk.Tk, config: PlotConfig) -> None:
        self.root = root
        self.messages: queue.Queue = queue.Queue()

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

        # Drawing state.
        self.rendering = False
        self.data_cached = False  # True once a drawing has filled the session download cache
        self.pending: PlotConfig | None = None
        self.shown_config: PlotConfig | None = None
        self._text_after_id: str | None = None
        self._resize_after_ids: dict[str, str] = {}
        self._loading = False

        self.zoom = tk.StringVar(value="fit")
        self.command = tk.StringVar()
        self.status = tk.StringVar(value="Starting …")

        self._build_layout()
        self._load_choices(config)
        self.root.after(_POLL_MS, self._poll)
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
        box.grid(row=row, column=0, columnspan=2, sticky="w")
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

    def _build_actions(self, panel: ttk.Frame) -> None:
        zoom = ttk.LabelFrame(panel, text="Preview", padding=(8, 4))
        zoom.pack(fill="x", pady=(0, 6))
        for value, text in (("fit", "Fit to window"), ("actual", "Actual size")):
            ttk.Radiobutton(zoom, text=text, value=value, variable=self.zoom, command=self._refresh_all_views).pack(
                anchor="w"
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
        self.log = tk.Text(frame, width=44, height=10, wrap="word", state="disabled", font=("Consolas", 9))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_tab(self, country: Country) -> None:
        tab = ttk.Frame(self.notebook)
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        canvas = tk.Canvas(tab, background="white", highlightthickness=0)
        x_scroll = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
        y_scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        canvas.bind("<Configure>", lambda _event, key=country.key: self._schedule_view_refresh(key))
        self.notebook.add(tab, text=_TAB_TITLES.get(country.key, country.key))
        self.tabs[country.key] = tab
        self.canvases[country.key] = canvas
        self.scrollbars[country.key] = (x_scroll, y_scroll)

    # -- choices ------------------------------------------------------------

    def _load_choices(self, config: PlotConfig) -> None:
        """Set every control from ``config`` without triggering a redraw per control."""
        payloads, flags = choices_from_config(config)
        self._loading = True
        try:
            for name, text in payloads.items():
                self.payload_vars[name].set(text)
            for name, value in flags.items():
                self.flag_vars[name].set(value)
        finally:
            self._loading = False

    def _choices(self) -> tuple[dict[str, str], dict[str, bool]]:
        """Return the controls' current ``(payloads, flags)``; blank text fields are left out (= default)."""
        payloads = {name: var.get().strip() for name, var in self.payload_vars.items() if var.get().strip()}
        flags = {name: var.get() for name, var in self.flag_vars.items()}
        return payloads, flags

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
        payloads = {name: var.get() for name, var in self.payload_vars.items()}
        flags = {name: var.get() for name, var in self.flag_vars.items()}
        tokens = command_line_tokens(payloads, flags)
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
        self._text_after_id = None
        self._update_command_line()
        payloads, flags = self._choices()
        problems = self._mark_invalid_fields(payloads)
        try:
            config = config_from_choices(payloads, flags)
        except ValueError as exc:
            # Field-level problems are listed first; a cross-field one (e.g.
            # end before start) has no single field to colour.
            self.status.set(f"Not drawn: {problems[0] if problems else exc}")
            return

        if force:
            self.shown_config = None
        if config == self.shown_config:
            return
        if self.rendering:
            self.pending = config
            self.status.set("Will redraw with the new settings when the current drawing finishes …")
            return
        self._start_render(config)

    def _start_render(self, config: PlotConfig) -> None:
        self.rendering = True
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
        try:
            with contextlib.redirect_stdout(_QueueWriter(self.messages)):
                for country in countries:
                    figure = build_figure(country, config)
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", dpi=figure.dpi, metadata=_PNG_METADATA)
                    self.messages.put(("chart", country.key, buffer.getvalue()))
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
                    self._show_chart(message[1], message[2])
                elif kind == "done":
                    self._render_finished(message[1], None)
                elif kind == "failed":
                    self._append_log(message[3])
                    self._render_finished(message[1], message[2])
        except queue.Empty:
            pass
        self.root.after(_POLL_MS, self._poll)

    def _show_chart(self, key: str, png: bytes) -> None:
        self.pngs[key] = png
        image = Image.open(io.BytesIO(png))
        image.load()
        self.images[key] = image
        self._refresh_view(key)

    def _render_finished(self, config: PlotConfig, error: str | None) -> None:
        self.rendering = False
        if error is None:
            self.shown_config = config
            self.data_cached = True
            drawn = [_TAB_TITLES.get(c.key, c.key) for c in COUNTRIES if getattr(config, c.show_field)]
            self.status.set(f"Drawn: {' and '.join(drawn)} at {config.width_px} x {config.height_px} px.")
        else:
            self.status.set(f"Drawing failed: {error} (details in the log)")
        pending, self.pending = self.pending, None
        if pending is not None and pending != self.shown_config:
            self._start_render(pending)

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
            self.photos[key] = ImageTk.PhotoImage(shown)
            canvas.create_image(view_width // 2, view_height // 2, image=self.photos[key], anchor="center")
            canvas.configure(scrollregion=(0, 0, view_width, view_height))
        else:
            self.photos[key] = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, image=self.photos[key], anchor="nw")
            canvas.configure(scrollregion=(0, 0, image.width, image.height))

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
        self._load_choices(PlotConfig())
        self.request_redraw()

    def _copy_command(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.command.get())
        self.status.set("Command line copied.")


def run_gui(config: PlotConfig) -> None:
    """Open the window with ``config`` as the starting choices and run until it is closed."""
    _enable_windows_dpi_awareness()
    http.enable_download_cache()
    root = tk.Tk()
    root.title(WINDOW_TITLE)
    width = round(root.winfo_screenwidth() * _SCREEN_FRACTION)
    height = round(root.winfo_screenheight() * _SCREEN_FRACTION)
    root.geometry(f"{width}x{height}+20+20")
    RatesPlotApp(root, config)
    root.mainloop()
