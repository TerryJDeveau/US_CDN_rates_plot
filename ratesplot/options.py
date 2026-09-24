"""The table of command-line options: one declaration per option.

Every option is described once, in ``OPTIONS``: how it is recognised on the
command line, which ``PlotConfig`` field(s) it sets, how its value is parsed
and checked, and its line in ``--help``. ``cli.parse_args`` is driven entirely
by this table, and the ``--help`` epilog is generated from it. A future GUI
should read the same table (kinds, fields, parsers, help text) so that adding
an option here adds it everywhere.

How a token is matched (see ``cli.parse_args``), in this order:

1. ``EXACT`` options: the whole name after the dashes, case-insensitive, with
   an optional ``no-`` prefix. No abbreviation. Checked first so that a new
   exact name can never be swallowed by the leading-letter rules below
   (``--gui`` would otherwise mean ``--gdp``).
2. ``VALUE`` options (``KEY:VALUE``): the key must start with one of the
   option's ``prefixes``. Only the leading letter(s) matter, so the many
   historical spellings (``--dim:``, ``--dimensions:``, ``--D:`` …) all work.
3. ``TOGGLE`` options (curves, no colon): the first letter after an optional
   ``no-`` must be one of the ``prefixes``.
4. ``SWITCH`` options go to argparse. A switch with ``initial`` set is also
   reached by any remaining token whose first letter (after dashes) is that
   letter, so ``Canada``, ``-c`` and ``--cdn`` all mean ``--C``. argparse
   would also accept unique prefixes of a long switch name, so any option
   that must not be abbreviated belongs in ``EXACT`` instead.

Anything still unmatched reaches argparse, which reports it as unrecognised.

Value parsers raise ``ValueError`` with a user-facing message; ``cli`` turns
that into an argparse error (exit status 2).
"""

from __future__ import annotations

import calendar
import dataclasses
import re
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Mapping

import pandas as pd

from .config import MIN_CANVAS_PX, PlotConfig

_DOLLAR_SUFFIX_MULTIPLIERS = {"t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3}
_DEFAULT_DOLLAR_MULTIPLIER = 1e9

_DATE_SPEC = re.compile(r"^(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?$")


# ---------------------------------------------------------------------------
# Value parsers (each raises ValueError with a user-facing message)
# ---------------------------------------------------------------------------


def parse_date_spec(spec: str, *, kind: str) -> pd.Timestamp:
    """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` (``/`` also accepted) to a Timestamp.

    Omitted month/day default to 1. ``kind`` ("start"/"end") is used in messages.
    """
    value = spec.strip()
    if not value:
        raise ValueError(f"empty {kind} date spec")

    match = _DATE_SPEC.match(value)
    if not match:
        raise ValueError(f"invalid {kind} date {spec!r}: expected YYYY, YYYY-MM or YYYY-MM-DD")

    year = int(match.group(1))
    month = int(match.group(2) or 1)
    day = int(match.group(3) or 1)
    if not 1 <= month <= 12:
        raise ValueError(f"invalid {kind} month {month}: must be 1–12")
    max_day = calendar.monthrange(year, month)[1]
    if not 1 <= day <= max_day:
        raise ValueError(f"invalid {kind} day {day} for {year}-{month:02d}: month has only {max_day} days")
    return pd.Timestamp(year=year, month=month, day=day)


def parse_dimensions_spec(spec: str) -> tuple[int, int]:
    """Parse ``W``, ``xH`` or ``WxH`` into ``(width, height)`` pixels, assuming 4:3 if needed."""
    value = spec.strip().lower()
    if not value:
        raise ValueError("empty dimensions spec")

    if value.startswith("x"):
        if not value[1:].isdigit():
            raise ValueError(f"dimensions height must be digits only, got {spec!r}")
        height = int(value[1:])
        width = round(height * 4 / 3)
    elif "x" in value:
        width_text, height_text = value.split("x", 1)
        if not (width_text.isdigit() and height_text.isdigit()):
            raise ValueError(f"dimensions must be digits around 'x', got {spec!r}")
        width, height = int(width_text), int(height_text)
    else:
        if not value.isdigit():
            raise ValueError(f"dimensions width must be digits only, got {spec!r}")
        width = int(value)
        height = round(width * 3 / 4)

    if min(width, height) < MIN_CANVAS_PX:
        raise ValueError(f"canvas {width}x{height} px is below minimum size of {MIN_CANVAS_PX} px")
    return width, height


def parse_yield_bound(spec: str, *, kind: str) -> float:
    """Parse a yield-axis bound in percent, allowed range ``[0, 100)``."""
    value = spec.strip()
    if not value:
        raise ValueError(f"empty {kind} yield bound")
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"invalid {kind} yield bound {spec!r}: must be a decimal number") from None
    if not 0 <= parsed < 100:
        raise ValueError(f"{kind} yield bound must be non-negative and < 100, got {parsed}")
    return parsed


def parse_dollar_bound(spec: str, *, kind: str) -> float:
    """Parse a dollar-axis bound; ``k/m/b/t`` suffixes scale, unsuffixed means billions."""
    value = spec.strip().lower()
    if not value:
        raise ValueError(f"empty {kind} dollar limit")

    multiplier = _DOLLAR_SUFFIX_MULTIPLIERS.get(value[-1])
    if multiplier is not None:
        value = value[:-1].strip()
    else:
        multiplier = _DEFAULT_DOLLAR_MULTIPLIER

    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"invalid {kind} dollar limit {spec!r}: must be a positive decimal value") from None
    if parsed <= 0:
        raise ValueError(f"{kind} dollar limit must be positive, got {parsed}")
    return parsed * multiplier


# ---------------------------------------------------------------------------
# Value formatters: field value(s) -> text the matching parser accepts
# ---------------------------------------------------------------------------
# Used to fill the GUI's text fields from a PlotConfig. Each receives the
# option's field values as a tuple and returns "" for "not set".


def format_date(values: tuple) -> str:
    """Format a date field as ``YYYY-MM-DD``."""
    return f"{values[0]:%Y-%m-%d}"


def format_dimensions(values: tuple) -> str:
    """Format ``(width_px, height_px)`` as ``WxH``."""
    return f"{values[0]}x{values[1]}"


def format_yield_bound(values: tuple) -> str:
    """Format a yield bound in percent; blank when unset."""
    return "" if values[0] is None else f"{values[0]:g}"


def format_dollar_bound(values: tuple) -> str:
    """Format a dollar bound with the largest k/m/b/t suffix that keeps it ≥ 1; blank when unset."""
    value = values[0]
    if value is None:
        return ""
    for suffix, multiplier in _DOLLAR_SUFFIX_MULTIPLIERS.items():  # t, b, m, k: largest first
        if value >= multiplier:
            return f"{value / multiplier:g}{suffix}"
    return f"{value / _DOLLAR_SUFFIX_MULTIPLIERS['k']:g}k"


# ---------------------------------------------------------------------------
# Checks that involve more than one option
# ---------------------------------------------------------------------------
# Each receives the field values gathered so far (PlotConfig defaults plus
# everything parsed up to this option) and raises ValueError if they conflict.


def check_date_range(values: Mapping[str, object]) -> None:
    """Reject reversed ranges and windows shorter than seven days."""
    start, end = values["start"], values["end"]
    if end < start:
        raise ValueError(f"end date {end.date()} is before start date {start.date()}")
    if (end - start).days < 7:
        raise ValueError(f"end date {end.date()} is less than 7 days after start date {start.date()}")


def check_yield_bounds(values: Mapping[str, object]) -> None:
    """Require min < max when both yield bounds are given."""
    low, high = values["yield_ymin"], values["yield_ymax"]
    if low is not None and high is not None and low >= high:
        raise ValueError(f"yield min ({low}) must be less than max ({high})")


def check_dollar_bounds(values: Mapping[str, object]) -> None:
    """Require bottom < top when both dollar bounds are given."""
    top, bottom = values["macro_top"], values["macro_bottom"]
    if top is not None and bottom is not None and bottom >= top:
        raise ValueError(f"bottom limit ({bottom}) must be less than top limit ({top})")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class Kind(Enum):
    EXACT = "exact"    # on/off flag, whole name only, optional "no-" prefix
    VALUE = "value"    # KEY:VALUE, recognised by leading letters of KEY
    TOGGLE = "toggle"  # curve on/off, recognised by first letter, optional "no-" prefix
    SWITCH = "switch"  # handed to argparse as a store_true flag


@dataclass(frozen=True)
class Option:
    """One command-line option. See the module docstring for how ``kind`` is matched."""

    name: str  # canonical name, used by the GUI and in messages
    kind: Kind
    fields: tuple[str, ...]  # PlotConfig field(s) the option sets
    group: str  # key into GROUPS, for the help listing
    usage: str  # left-hand column of the help listing: the spellings
    help: str  # right-hand column; "\n" starts a continuation line
    prefixes: tuple[str, ...] = ()  # VALUE / TOGGLE: accepted leading letters (lower case)
    parse: Callable[[str], object] | None = None  # VALUE: payload -> value (a tuple if several fields)
    check: Callable[[Mapping[str, object]], None] | None = None  # VALUE: run after this option is parsed
    flag: str | None = None  # SWITCH: the argparse flag
    flag_help: str | None = None  # SWITCH: argparse's own one-line help
    initial: str | None = None  # SWITCH: legacy first-letter alias (upper case)
    # GUI. ``label`` is the caption beside the control. ``format`` turns the
    # field value(s) back into text a parser accepts (VALUE only). Options with
    # ``in_gui=False`` get no control: they choose the interface or run
    # maintenance rather than shape the chart.
    label: str = ""
    format: Callable[[tuple], str] | None = None
    in_gui: bool = True
    picker: str | None = None  # GUI: "date" adds a calendar button beside the text field


# Help-listing sections, in display order: key -> heading (and any notes).
# The GUI titles its panels with the heading up to the first " (" or ":".
GROUPS: dict[str, str] = {
    "country": "Country selection (case-insensitive; only the first letter matters):",
    "curves": (
        "Curve selection (case-insensitive; only the first letter matters). Naming any\n"
        "curve positively shows *only* the named curves; ``--no-`` forms hide curves\n"
        "from the default set of all four:"
    ),
    "dates": "Date window (YYYY, YYYY-MM or YYYY-MM-DD; '/' also accepted):",
    "canvas": "Canvas size in pixels (4:3 assumed when only one dimension is given):",
    "yield": "Yield-axis limits (percent):",
    "dollar": "Dollar-axis limits (suffixes k/m/b/t; unsuffixed values are billions):",
    "interface": "Interface (spelled in full; no abbreviation):",
    "maintenance": "Maintenance:",
}

OPTIONS: tuple[Option, ...] = (
    # Country selection: argparse store_true flags; no flag at all means both.
    Option(
        "cdn", Kind.SWITCH, ("show_cdn",), "country", "--C / -C / --canada / --cdn", "Canadian chart only",
        flag="--C", flag_help="Canadian chart only", initial="C", label="Canada",
    ),
    Option(
        "us", Kind.SWITCH, ("show_us",), "country", "--U / -U / --us / --usa", "U.S. chart only",
        flag="--U", flag_help="U.S. chart only", initial="U", label="United States",
    ),
    # Curves. The listing order here is the help order; the selection rule is
    # in ``config_from_choices``.
    Option(
        "gdp", Kind.TOGGLE, ("include_gdp",), "curves", "--GDP / --no-GDP", "TTM nominal GDP",
        prefixes=("g",), label="TTM nominal GDP",
    ),
    Option(
        "debt", Kind.TOGGLE, ("include_debt",), "curves", "--debt / --no-debt", "aggregate public debt",
        prefixes=("d",), label="Aggregate public debt",
    ),
    Option(
        "interest", Kind.TOGGLE, ("include_interest",), "curves", "--interest / --no-interest", "TTM interest outlays",
        prefixes=("i",), label="TTM interest outlays",
    ),
    Option(
        "yield", Kind.TOGGLE, ("include_yield",), "curves", "--yield / --no-yield", "bond yields",
        prefixes=("y",), label="Bond yields",
    ),
    # Values. Table order is parse order, so it decides which error is reported
    # first when several values are bad; each ``check`` runs once both of its
    # fields are known.
    Option(
        "dimensions", Kind.VALUE, ("width_px", "height_px"), "canvas", "--dimensions:WxH / --d:W / --d:xH",
        f"(minimum {MIN_CANVAS_PX} px each way)", prefixes=("d",), parse=parse_dimensions_spec,
        label="W x H px", format=format_dimensions,
    ),
    Option(
        "start", Kind.VALUE, ("start",), "dates", "--start:DATE / --s:DATE", "first date (default 1966-01-01)",
        prefixes=("s",), parse=partial(parse_date_spec, kind="start"), label="Start", format=format_date,
        picker="date",
    ),
    Option(
        "end", Kind.VALUE, ("end",), "dates", "--end:DATE / --e:DATE", "last date (default today)",
        prefixes=("e",), parse=partial(parse_date_spec, kind="end"), check=check_date_range,
        label="End", format=format_date, picker="date",
    ),
    Option(
        "min", Kind.VALUE, ("yield_ymin",), "yield", "--min:VAL / --mn:VAL", "lower bound",
        prefixes=("mn", "mi"), parse=partial(parse_yield_bound, kind="min"), label="Min %", format=format_yield_bound,
    ),
    Option(
        "max", Kind.VALUE, ("yield_ymax",), "yield", "--max:VAL / --mx:VAL", "upper bound",
        prefixes=("mx", "ma"), parse=partial(parse_yield_bound, kind="max"), check=check_yield_bounds,
        label="Max %", format=format_yield_bound,
    ),
    Option(
        "top", Kind.VALUE, ("macro_top",), "dollar", "--top:VAL / --t:VAL", "upper bound",
        prefixes=("t",), parse=partial(parse_dollar_bound, kind="top"), label="Top", format=format_dollar_bound,
    ),
    Option(
        "bottom", Kind.VALUE, ("macro_bottom",), "dollar", "--bottom:VAL / --b:VAL", "lower bound",
        prefixes=("b",), parse=partial(parse_dollar_bound, kind="bottom"), check=check_dollar_bounds,
        label="Bottom", format=format_dollar_bound,
    ),
    # Interface. EXACT, so "--g", "--gu" and "--guix" still mean the GDP curve.
    Option(
        "gui", Kind.EXACT, ("gui",), "interface", "--gui / --no-gui",
        "interactive window (default); --no-gui\ndraws plain matplotlib windows instead",
        in_gui=False,
    ),
    # Maintenance. EXACT, so it must be spelled in full: as an argparse switch,
    # "--b" (a bottom bound missing its ":VAL") silently rewrote the archive
    # data module. Kept for a possible repurposed use.
    Option(
        "bake-archives", Kind.EXACT, ("bake_archives",), "maintenance", "--bake-archives",
        "refresh ratesplot/cdn_archive_data.py from\nthe archived StatCan tables, then exit",
        in_gui=False,
    ),
)


def options_of(kind: Kind) -> list[Option]:
    """Return the options of one kind, in table order."""
    return [option for option in OPTIONS if option.kind is kind]


def by_name(name: str) -> Option:
    """Return the option called ``name``."""
    for option in OPTIONS:
        if option.name == name:
            return option
    raise KeyError(name)


def group_title(group: str) -> str:
    """Return a group's short title: its help heading up to the first " (" or ":"."""
    return GROUPS[group].split(" (")[0].split(":")[0]


# ---------------------------------------------------------------------------
# Choices <-> PlotConfig (shared by the command line and the GUI)
# ---------------------------------------------------------------------------
# "Choices" are what a user expressed, keyed by option name:
#   payloads: VALUE option -> its text (absent = not given, so the default)
#   flags:    EXACT / TOGGLE / SWITCH option -> True or False (absent = not given)


def config_from_choices(payloads: Mapping[str, str], flags: Mapping[str, bool]) -> PlotConfig:
    """Build the PlotConfig the choices describe; raise ValueError naming the first problem.

    Starts from PlotConfig's defaults (the default end date is today). VALUE
    options are parsed in table order and each option's ``check`` runs straight
    after it, against everything gathered so far. Two rules span several options:

    * Curves: if any curve is chosen positively, only the positively chosen
      curves are drawn; otherwise all four are, minus any switched off. (With
      every curve given explicitly, as the GUI does, this is simply "draw the
      ones switched on".)
    * Countries: no country chosen means both charts.
    """
    values = dataclasses.asdict(PlotConfig())
    for option in options_of(Kind.VALUE):
        if option.name in payloads:
            parsed = option.parse(payloads[option.name])
            values.update(zip(option.fields, parsed if len(option.fields) > 1 else (parsed,)))
        if option.check is not None:
            option.check(values)

    for option in options_of(Kind.EXACT):
        if option.name in flags:
            values[option.fields[0]] = flags[option.name]

    curves = {option.fields[0]: flags.get(option.name) for option in options_of(Kind.TOGGLE)}
    if any(chosen is True for chosen in curves.values()):
        values.update({field: chosen is True for field, chosen in curves.items()})
    else:
        values.update({field: chosen is not False for field, chosen in curves.items()})

    for option in options_of(Kind.SWITCH):
        values[option.fields[0]] = flags.get(option.name, False)
    if not (values["show_cdn"] or values["show_us"]):
        values["show_cdn"] = values["show_us"] = True

    return PlotConfig(**values)


def choices_from_config(config: PlotConfig) -> tuple[dict[str, str], dict[str, bool]]:
    """Return ``(payloads, flags)`` describing ``config``, for filling the GUI's controls.

    Every GUI option is included. An unset value (a blank yield or dollar
    bound) comes back as "", which the GUI treats as "use the default".
    So does a value equal to a *moving* default (one PlotConfig computes
    afresh, such as the end date "today"): written out as a date it would be
    remembered and frozen, and tomorrow's chart would stop at yesterday.
    """
    defaults = PlotConfig()
    moving = {field.name for field in dataclasses.fields(PlotConfig) if field.default_factory is not dataclasses.MISSING}
    payloads: dict[str, str] = {}
    flags: dict[str, bool] = {}
    for option in OPTIONS:
        if not option.in_gui:
            continue
        field_values = tuple(getattr(config, field) for field in option.fields)
        if option.kind is Kind.VALUE:
            at_moving_default = any(field in moving for field in option.fields) and field_values == tuple(
                getattr(defaults, field) for field in option.fields
            )
            payloads[option.name] = "" if at_moving_default else option.format(field_values)
        else:
            flags[option.name] = bool(field_values[0])
    return payloads, flags


def command_line_tokens(payloads: Mapping[str, str], flags: Mapping[str, bool]) -> list[str]:
    """Return the shortest command line (after the script name) that reproduces the GUI choices.

    Options left at their defaults are omitted. The GUI passes every option,
    blanks included; blank values are omitted too.
    """
    defaults, _ = choices_from_config(PlotConfig())
    tokens: list[str] = []

    countries = [option for option in options_of(Kind.SWITCH) if option.in_gui]
    chosen = [option for option in countries if flags.get(option.name)]
    if 0 < len(chosen) < len(countries):
        tokens += [option.flag for option in chosen]

    # Curves: name the ones on, or negate the ones off, whichever is shorter
    # (both mean the same under the curve rule; all off needs every "--no-").
    curves = options_of(Kind.TOGGLE)
    on = [option for option in curves if flags.get(option.name, True)]
    off = [option for option in curves if option not in on]
    if off and (not on or len(off) <= len(on)):
        tokens += [f"--no-{option.name}" for option in off]
    elif off:
        tokens += [f"--{option.name}" for option in on]

    for option in options_of(Kind.VALUE):
        text = payloads.get(option.name, "").strip()
        if text and text != defaults.get(option.name):
            tokens.append(f"--{option.name}:{text}")
    return tokens


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_INDENT = "    "
_HELP_COLUMN = 34  # width of the spellings column
_HELP_MIN_GAP = 3  # a spelling longer than the column still gets this many spaces


def help_epilog() -> str:
    """Return the option reference printed after argparse's own ``--help`` output."""
    lines = ["Run with no flags to open the interactive window with both charts (Canadian,", "then U.S.). Flags set the window's starting values.", ""]
    for group, heading in GROUPS.items():
        lines.append(heading)
        for option in OPTIONS:
            if option.group != group:
                continue
            width = max(_HELP_COLUMN, len(option.usage) + _HELP_MIN_GAP)
            first, *rest = option.help.split("\n")
            lines.append(f"{_HELP_INDENT}{option.usage:<{width}}{first}")
            lines.extend(f"{_HELP_INDENT}{'':<{width}}{line}" for line in rest)
        lines.append("")
    return "\n".join(lines)
