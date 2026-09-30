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
  under (``saved_png_name``); the tab titles.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from matplotlib.figure import Figure

from .config import COMPONENT_SYNONYMS, PlotConfig
from .options import (
    GROUPS,
    GROUPS_CHOSEN_TOGETHER,
    OPTIONS,
    Kind,
    Option,
    choices_from_config,
    config_from_choices,
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
