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
   letter, so ``Canada``, ``-c`` and ``--cdn`` all mean ``--C``. Other switches
   keep argparse's unique-prefix abbreviation (``--bake`` for
   ``--bake-archives``).

Anything still unmatched reaches argparse, which reports it as unrecognised.

Value parsers raise ``ValueError`` with a user-facing message; ``cli`` turns
that into an argparse error (exit status 2).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Mapping

import pandas as pd

from .config import MIN_CANVAS_PX

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


# Help-listing sections, in display order: key -> heading (and any notes).
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
    "maintenance": "Maintenance:",
}

OPTIONS: tuple[Option, ...] = (
    # Country selection: argparse store_true flags; no flag at all means both.
    Option(
        "cdn", Kind.SWITCH, ("show_cdn",), "country", "--C / -C / --canada / --cdn", "Canadian chart only",
        flag="--C", flag_help="Canadian chart only", initial="C",
    ),
    Option(
        "us", Kind.SWITCH, ("show_us",), "country", "--U / -U / --us / --usa", "U.S. chart only",
        flag="--U", flag_help="U.S. chart only", initial="U",
    ),
    # Curves. The listing order here is the help order; resolution rules are in cli.
    Option("gdp", Kind.TOGGLE, ("include_gdp",), "curves", "--GDP / --no-GDP", "TTM nominal GDP", prefixes=("g",)),
    Option("debt", Kind.TOGGLE, ("include_debt",), "curves", "--debt / --no-debt", "aggregate public debt", prefixes=("d",)),
    Option(
        "interest", Kind.TOGGLE, ("include_interest",), "curves", "--interest / --no-interest", "TTM interest outlays",
        prefixes=("i",),
    ),
    Option("yield", Kind.TOGGLE, ("include_yield",), "curves", "--yield / --no-yield", "bond yields", prefixes=("y",)),
    # Values. Table order is parse order, so it decides which error is reported
    # first when several values are bad; each ``check`` runs once both of its
    # fields are known.
    Option(
        "dimensions", Kind.VALUE, ("width_px", "height_px"), "canvas", "--dimensions:WxH / --d:W / --d:xH",
        f"(minimum {MIN_CANVAS_PX} px each way)", prefixes=("d",), parse=parse_dimensions_spec,
    ),
    Option(
        "start", Kind.VALUE, ("start",), "dates", "--start:DATE / --s:DATE", "first date (default 1966-01-01)",
        prefixes=("s",), parse=partial(parse_date_spec, kind="start"),
    ),
    Option(
        "end", Kind.VALUE, ("end",), "dates", "--end:DATE / --e:DATE", "last date (default today)",
        prefixes=("e",), parse=partial(parse_date_spec, kind="end"), check=check_date_range,
    ),
    Option(
        "min", Kind.VALUE, ("yield_ymin",), "yield", "--min:VAL / --mn:VAL", "lower bound",
        prefixes=("mn", "mi"), parse=partial(parse_yield_bound, kind="min"),
    ),
    Option(
        "max", Kind.VALUE, ("yield_ymax",), "yield", "--max:VAL / --mx:VAL", "upper bound",
        prefixes=("mx", "ma"), parse=partial(parse_yield_bound, kind="max"), check=check_yield_bounds,
    ),
    Option(
        "top", Kind.VALUE, ("macro_top",), "dollar", "--top:VAL / --t:VAL", "upper bound",
        prefixes=("t",), parse=partial(parse_dollar_bound, kind="top"),
    ),
    Option(
        "bottom", Kind.VALUE, ("macro_bottom",), "dollar", "--bottom:VAL / --b:VAL", "lower bound",
        prefixes=("b",), parse=partial(parse_dollar_bound, kind="bottom"), check=check_dollar_bounds,
    ),
    # Maintenance. Note argparse abbreviation: "--b" (no colon) also means this.
    Option(
        "bake-archives", Kind.SWITCH, ("bake_archives",), "maintenance", "--bake-archives",
        "refresh ratesplot/cdn_archive_data.py from\nthe archived StatCan tables, then exit",
        flag="--bake-archives",
        flag_help="Download the archived Canadian StatCan tables once and embed them in ratesplot/cdn_archive_data.py.",
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


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_INDENT = "    "
_HELP_COLUMN = 34  # width of the spellings column
_HELP_MIN_GAP = 3  # a spelling longer than the column still gets this many spaces


def help_epilog() -> str:
    """Return the option reference printed after argparse's own ``--help`` output."""
    lines = ["Run with no flags to produce both charts (Canadian, then U.S.).", ""]
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
