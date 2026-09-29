"""The table of command-line options: one declaration per option.

Every option is described once, in ``OPTIONS``: how it is recognised on the
command line, which ``PlotConfig`` field(s) it sets, how its value is parsed
and checked, and its line in ``--help``. ``cli.parse_args`` is driven entirely
by this table, the ``--help`` epilog is generated from it, and the GUI builds
its controls from it, so adding an option here adds it everywhere.

How a token is matched (``cli.match_token``). Leading dashes are optional
and case does not matter. A token spells an option when the word after the
dashes (and after ``no-``, for the options that have an off form) is a
leading part of one of the option's ``names``, at least ``shortest`` letters
long; hyphens inside a name may be left out (``--percapita``). So ``-r``,
``--rel`` and ``--ratio`` are all ``--relative``, and ``--reg`` is the
regression. Nothing is matched by its first letter alone any more (Terry,
2026-09-29: "--reg must never invoke -r"): ``--yes`` is not the yield curve,
``--dept`` is not debt, and ``--dimensions`` without a value is not debt.

* A ``KEY:VALUE`` token is matched against the ``VALUE`` options only, and
  any other token against the rest. A value given to an option that takes
  none (``-r:1``, ``--cur:1``) is an error, never another option that shares
  its first letter; so is a ``VALUE`` option named without a value.
* ``no-`` is accepted by ``FLAG`` and ``TOGGLE`` options only.
* A token that spells two options is an error (``--m:5``: --min or --max).
  ``shortest`` keeps apart the names that share a first letter: "c" is
  Canada and "cur" the latest values, "r" the measure and "reg" the
  regression, "g" the GDP curve and "gui" (in full) the window.

Anything unmatched goes to argparse, which prints ``--help`` or reports it
as unrecognised.

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

from .config import (
    COMPONENT_LETTERS,
    COMPONENT_SYNONYMS,
    DEFAULT_REGRESSION_TOLERANCE_PCT,
    MIN_CANVAS_PX,
    MIN_REGRESSION_TOLERANCE_PCT,
    PlotConfig,
)

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


def parse_regression_tolerance(spec: str) -> float:
    """Parse the regression tolerance, a percentage of the right axis's height: ``1.5`` or ``1.5%``."""
    value = spec.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    if not value:
        raise ValueError("empty --reg tolerance: give a percentage of the axis height, e.g. --reg:1.5")
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"invalid --reg tolerance {spec!r}: must be a percentage of the axis height, e.g. 1.5") from None
    if not 0 < parsed <= 100:
        raise ValueError(f"--reg tolerance must be above 0 and at most 100 (% of the axis height), got {spec.strip()}")
    return parsed


def is_percent_bound(spec: str) -> bool:
    """True when a right-axis bound is written as a percentage (``150%``), as ``-r`` requires."""
    return spec.strip().endswith("%")


def parse_macro_bound(spec: str, *, kind: str) -> float:
    """Parse a right-axis bound: dollars, or with a ``%`` sign a percentage (for ``-r``).

    Dollars: ``k/m/b/t`` suffixes scale, unsuffixed means billions. A
    percentage is returned as the number of percent (``150%`` -> 150.0).
    Which of the two the chart needs is checked in ``config_from_choices``,
    once ``-r`` is known.
    """
    value = spec.strip().lower()
    if not value:
        raise ValueError(f"empty {kind} limit")

    if value.endswith("%"):
        try:
            parsed = float(value[:-1].strip())
        except ValueError:
            raise ValueError(f"invalid {kind} percentage {spec!r}: must be a positive decimal value") from None
        if parsed <= 0:
            raise ValueError(f"{kind} percentage must be positive, got {parsed}")
        return parsed

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
# option's field values as a tuple, and the whole config (a right-axis bound
# is a percentage under -r), and returns "" for "not set".


def format_date(values: tuple, _config: PlotConfig) -> str:
    """Format a date field as ``YYYY-MM-DD``."""
    return f"{values[0]:%Y-%m-%d}"


def format_dimensions(values: tuple, _config: PlotConfig) -> str:
    """Format ``(width_px, height_px)`` as ``WxH``."""
    return f"{values[0]}x{values[1]}"


def format_yield_bound(values: tuple, _config: PlotConfig) -> str:
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


def format_macro_bound(values: tuple, config: PlotConfig) -> str:
    """Format a right-axis bound: a percentage under -r, otherwise dollars; blank when unset."""
    if values[0] is None:
        return ""
    return f"{values[0]:g}%" if config.relative else format_dollar_bound(values)


def format_regression_tolerance(values: tuple, _config: PlotConfig) -> str:
    """Format the regression tolerance in percent (the window's label gives the unit); blank when unset."""
    return "" if values[0] is None else f"{values[0]:g}"


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


def check_macro_bounds(values: Mapping[str, object]) -> None:
    """Require bottom < top when both right-axis bounds are given."""
    top, bottom = values["macro_top"], values["macro_bottom"]
    if top is not None and bottom is not None and bottom >= top:
        raise ValueError(f"bottom limit ({bottom}) must be less than top limit ({top})")


def parse_components(spec: str, *, kind: str) -> str:
    """Parse ``--debt:``/``--interest:`` sub-option letters into ``COMPONENT_LETTERS`` order.

    Any of f, n, p, m in any order and combination; "s" (state) means "p";
    repeats are harmless. ``kind`` names the option in messages.
    """
    value = spec.strip().lower()
    if not value:
        raise ValueError(f"empty --{kind} sub-options: use letters from f, n, p (or s), m")
    unknown = sorted(set(value) - set(COMPONENT_LETTERS) - set(COMPONENT_SYNONYMS))
    if unknown:
        # A size written the old way ("--d:1100", "--d:x600") gets pointed at --dim.
        hint = "; the canvas size is --dim:WxH" if kind == "debt" and re.match(r"x?\d", value) else ""
        raise ValueError(
            f"invalid --{kind} sub-option {spec!r}: use letters f federal, n non-federal, "
            f"p or s provincial/state, m municipal{hint}"
        )
    letters = {COMPONENT_SYNONYMS.get(letter, letter) for letter in value}
    return "".join(letter for letter in COMPONENT_LETTERS if letter in letters)


def format_components(values: tuple, _config: PlotConfig) -> str:
    """Format the component letters (already in canonical order); blank for the aggregates."""
    return values[0]


def check_single_measure(values: Mapping[str, object]) -> None:
    """Reject -r with -p: a curve is either a share of GDP or an amount per person."""
    if values["relative"] and values["per_capita"]:
        raise ValueError("-r (% of GDP) and -p (per capita) cannot be combined; choose one")


def check_macro_bound_units(payloads: Mapping[str, str], values: Mapping[str, object]) -> None:
    """Require right-axis bounds as percentages under -r, and in dollars otherwise.

    Run once the flags are known: the same text means different things in
    the two measures, so it is rejected rather than reinterpreted.
    """
    for name in ("top", "bottom"):
        if name not in payloads:
            continue
        spec = payloads[name].strip()
        if values["relative"] and not is_percent_bound(spec):
            raise ValueError(f"with -r the --{name} limit is a percentage of GDP, e.g. --{name}:150%")
        if not values["relative"] and is_percent_bound(spec):
            raise ValueError(f"--{name}:{spec} is a percentage, which needs -r (debt and interest as % of GDP)")


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class Kind(Enum):
    FLAG = "flag"      # on/off with a "no-" form, sets its field directly (--cur, --reg, --gui)
    VALUE = "value"    # KEY:VALUE
    TOGGLE = "toggle"  # curve on/off with a "no-" form, under the curve rule
    SWITCH = "switch"  # on only; also an argparse store_true flag (``flag``), for usage and --help


@dataclass(frozen=True)
class Option:
    """One command-line option. See the module docstring for how a token is matched."""

    name: str  # canonical name, used by the GUI and in messages
    kind: Kind
    fields: tuple[str, ...]  # PlotConfig field(s) the option sets
    group: str  # key into GROUPS, for the help listing
    usage: str  # left-hand column of the help listing: the spellings
    help: str  # right-hand column; "\n" starts a continuation line
    names: tuple[str, ...] = ()  # lower case; any leading part of one spells the option
    shortest: int = 1  # the fewest letters a spelling may have
    parse: Callable[[str], object] | None = None  # VALUE: payload -> value (a tuple if several fields)
    check: Callable[[Mapping[str, object]], None] | None = None  # VALUE: run after this option is parsed
    turns_on: str | None = None  # VALUE: the FLAG option that giving this value turns on (--reg:TOL is --reg)
    flag: str | None = None  # SWITCH: the argparse flag
    flag_help: str | None = None  # SWITCH: argparse's own one-line help
    # GUI. ``label`` is the caption beside the control. ``format`` turns the
    # field value(s) back into text a parser accepts, given the rest of the
    # config (VALUE only). Options with ``in_gui=False`` get no control: they
    # choose the interface rather than shape the chart.
    label: str = ""
    format: Callable[[tuple, PlotConfig], str] | None = None
    in_gui: bool = True
    # GUI editor for a VALUE option: None = a plain text field; "date" = text
    # field plus calendar button; "size" = width and height boxes with an
    # aspect-ratio lock (for a "WxH" option such as the canvas size).
    editor: str | None = None


# Help-listing sections, in display order: key -> heading (and any notes).
# The GUI titles its panels with the heading up to the first " (" or ":".
GROUPS: dict[str, str] = {
    "country": "Country selection:",
    "curves": (
        "Curve selection (naming any curve positively shows *only* the named curves;\n"
        "``--no-`` forms hide curves from the default set of all four):"
    ),
    "units": "Measure of debt, GDP and interest:",
    "labels": "Line labels and regression segments (--reg needs at least \"reg\"):",
    "dates": "Date window (YYYY, YYYY-MM or YYYY-MM-DD; '/' also accepted):",
    "latest": "Latest data (at least \"cur\"):",
    "canvas": "Canvas size in pixels (4:3 assumed when only one dimension is given):",
    "yield": "Yield-axis limits (percent):",
    "dollar": (
        "Right-axis limits (suffixes k/m/b/t; unsuffixed values are billions;\n"
        "with -r, percentages such as 150%):"
    ),
    "interface": "Interface (spelled in full; no abbreviation):",
}

OPTIONS: tuple[Option, ...] = (
    # Country selection: argparse store_true flags; no flag at all means both.
    Option(
        "cdn", Kind.SWITCH, ("show_cdn",), "country", "--C / -C / --canada / --cdn", "Canadian chart only",
        names=("canada", "cdn"), flag="--C", flag_help="Canadian chart only", label="Canada",
    ),
    Option(
        "us", Kind.SWITCH, ("show_us",), "country", "--U / -U / --us / --usa", "U.S. chart only",
        names=("us", "usa"), flag="--U", flag_help="U.S. chart only", label="United States",
    ),
    # Curves. The listing order here is the help order; the selection rule is
    # in ``config_from_choices``.
    Option(
        "gdp", Kind.TOGGLE, ("include_gdp",), "curves", "--GDP / --no-GDP", "TTM nominal GDP",
        names=("gdp",), label="TTM nominal GDP",
    ),
    Option(
        "debt", Kind.TOGGLE, ("include_debt",), "curves", "--debt / --no-debt", "aggregate public debt",
        names=("debt",), label="Aggregate public debt",
    ),
    Option(
        "interest", Kind.TOGGLE, ("include_interest",), "curves", "--interest / --no-interest", "TTM interest outlays",
        names=("interest",), label="TTM interest outlays",
    ),
    Option(
        "yield", Kind.TOGGLE, ("include_yield",), "curves", "--yield / --no-yield", "bond yields",
        names=("yield", "yields"), label="Bond yields",
    ),
    # Measure of the right-axis curves: argparse store_true flags, like the
    # countries.
    Option(
        "relative", Kind.SWITCH, ("relative",), "units", "--R / -r / --relative",
        "debt and interest as % of TTM GDP (log\npercent axis); GDP itself is not drawn",
        names=("relative",), flag="--R", flag_help="debt and interest as a percentage of GDP", label="As % of GDP",
    ),
    Option(
        "per-capita", Kind.SWITCH, ("per_capita",), "units", "--P / -p / --per-capita",
        "GDP, debt and interest per person (log\ndollar axis); not with -r",
        names=("per-capita",), flag="--P", flag_help="GDP, debt and interest per person", label="Per capita",
    ),
    # Value labels at the line ends: a switch like the measures, but in a
    # group of its own, since it combines with either.
    Option(
        "label", Kind.SWITCH, ("end_labels",), "labels", "--L / -l / --label",
        "each line's last value at its end, in\nthe line's colour; the date axis is\nwidened to make room",
        names=("label", "labels"), flag="--L", flag_help="label each line's end with its last value",
        label="Last value at each line's end",
    ),
    # Regression segments on the right-axis curves (ratesplot.regression), in
    # the same group, so the window keeps one panel for both. At least "reg",
    # so it is never -r: "-r", "--re" and "--rel" are --relative.
    Option(
        "regression", Kind.FLAG, ("regression",), "labels", "--reg / --regression / --no-reg",
        "each right-axis curve fitted by the fewest\nstraight pieces on its log axis, each\n"
        "labelled with its growth in %/yr:\ncompounded on a piece of a year or more,\n"
        "the log slope on a shorter one",
        names=("regression",), shortest=3, label="Regression segments",
    ),
    # Its tolerance. Giving it turns --reg on, as --debt:LETTERS names the
    # debt curve (``config_from_choices``); in the window the field is greyed
    # while --reg is off.
    Option(
        "regression-tolerance", Kind.VALUE, ("regression_tolerance",), "labels", "--reg:TOL / --regression:TOL",
        "how far a piece may stray from the\ncurve's points, in % of the axis height\n"
        f"(default {DEFAULT_REGRESSION_TOLERANCE_PCT:g}; never less than {MIN_REGRESSION_TOLERANCE_PCT:g} % of the\n"
        "value); turns --reg on",
        names=("regression",), shortest=3, parse=parse_regression_tolerance, turns_on="regression",
        label="tolerance, % of axis", format=format_regression_tolerance,
    ),
    # Values. Table order is parse order, so it decides which error is reported
    # first when several values are bad; each ``check`` runs once both of its
    # fields are known.
    Option(
        "dimensions", Kind.VALUE, ("width_px", "height_px"), "canvas", "--dimensions:WxH / --dim:W / --dim:xH",
        f"(minimum {MIN_CANVAS_PX} px each way)", names=("dimensions",), shortest=3, parse=parse_dimensions_spec,
        label="Size", format=format_dimensions, editor="size",
    ),
    # Curve sub-options: debt and interest by level of government ("--d:" is
    # debt; the size needs at least "dim"). They set one field; see
    # ``config_from_choices`` for how they name their curves. The window has
    # one control for both.
    Option(
        "debt-parts", Kind.VALUE, ("components",), "curves", "--debt:LETTERS / --d:LETTERS",
        "debt and interest by level instead of\nin total: f federal, n non-federal,\n"
        "p or s provincial/state, m municipal",
        names=("debt",), parse=partial(parse_components, kind="debt"),
        label="By level", format=format_components, editor="levels",
    ),
    Option(
        "interest-parts", Kind.VALUE, ("components",), "curves", "--interest:LETTERS / --i:LETTERS",
        "the same letters; given on both --debt\nand --interest they must agree",
        names=("interest",), parse=partial(parse_components, kind="interest"), in_gui=False,
    ),
    Option(
        "start", Kind.VALUE, ("start",), "dates", "--start:DATE / --s:DATE", "first date (default 1966-01-01)",
        names=("start",), parse=partial(parse_date_spec, kind="start"), label="Start", format=format_date,
        editor="date",
    ),
    Option(
        "end", Kind.VALUE, ("end",), "dates", "--end:DATE / --e:DATE", "last date (default today)",
        names=("end",), parse=partial(parse_date_spec, kind="end"), check=check_date_range,
        label="End", format=format_date, editor="date",
    ),
    # "--m:" could be either, so it is refused as ambiguous.
    Option(
        "min", Kind.VALUE, ("yield_ymin",), "yield", "--min:VAL / --mn:VAL", "lower bound",
        names=("minimum", "mn"), parse=partial(parse_yield_bound, kind="min"), label="Min %", format=format_yield_bound,
    ),
    Option(
        "max", Kind.VALUE, ("yield_ymax",), "yield", "--max:VAL / --mx:VAL", "upper bound",
        names=("maximum", "mx"), parse=partial(parse_yield_bound, kind="max"), check=check_yield_bounds,
        label="Max %", format=format_yield_bound,
    ),
    Option(
        "top", Kind.VALUE, ("macro_top",), "dollar", "--top:VAL / --t:VAL", "upper bound",
        names=("top",), parse=partial(parse_macro_bound, kind="top"), label="Top", format=format_macro_bound,
    ),
    Option(
        "bottom", Kind.VALUE, ("macro_bottom",), "dollar", "--bottom:VAL / --b:VAL", "lower bound",
        names=("bottom",), parse=partial(parse_macro_bound, kind="bottom"), check=check_macro_bounds,
        label="Bottom", format=format_macro_bound,
    ),
    # The latest values (ratesplot.latest), on by default. At least "cur", so
    # it is never Canada ("-c", "--ca", "--cdn").
    Option(
        "current", Kind.FLAG, ("current",), "latest", "--cur / --current / --no-cur",
        "on by default: U.S. Treasury daily yields\nand debt, and the day's intraday yield\n"
        "quotes (CNBC), past the regular sources",
        names=("current",), shortest=3, label="Latest values, with intraday quotes",
    ),
    # Interface. Spelled in full (Terry, 2026-09-24): "--g" is the GDP curve,
    # and "--gu" nothing.
    Option(
        "gui", Kind.FLAG, ("gui",), "interface", "--gui / --no-gui",
        "interactive window (default); --no-gui\ndraws plain matplotlib windows instead",
        names=("gui",), shortest=3, in_gui=False,
    ),
    # Baking the historical data into the code is not an option of the program:
    # it is a maintenance step run from tools/bake_archives.py when an
    # extraction changes (Terry, 2026-09-27).
)


def options_of(kind: Kind) -> list[Option]:
    """Return the options of one kind, in table order."""
    return [option for option in OPTIONS if option.kind is kind]


def options_in(group: str) -> list[Option]:
    """Return the options of one help group (a key of ``GROUPS``), in table order."""
    return [option for option in OPTIONS if option.group == group]


# Groups whose options are chosen together: naming one curve on the command
# line means "only the named curves", naming one country "only the named
# countries", and a measure is one choice however many switches offer it (see
# ``config_from_choices``). So the GUI lets the command line override its
# remembered choices a whole group at a time.
GROUPS_CHOSEN_TOGETHER = ("curves", "country", "units")


def by_name(name: str) -> Option:
    """Return the option called ``name``."""
    for option in OPTIONS:
        if option.name == name:
            return option
    raise KeyError(name)


def group_title(group: str) -> str:
    """Return a group's short title: its help heading up to the first " (" or ":"."""
    return GROUPS[group].split(" (")[0].split(":")[0]


def _letters(text: str) -> str:
    """Return ``text`` without hyphens or underscores, which may be left out of a name."""
    return text.replace("-", "").replace("_", "")


def spells(option: Option, word: str) -> bool:
    """True when ``word`` (lower case) is a leading part of one of the option's names.

    It must have at least ``option.shortest`` letters. ``per``, ``percap``
    and ``per_capita`` all spell ``per-capita``; ``pizza`` spells nothing.
    """
    letters = _letters(word)
    return len(letters) >= option.shortest and any(_letters(name).startswith(letters) for name in option.names)


# ---------------------------------------------------------------------------
# Choices <-> PlotConfig (shared by the command line and the GUI)
# ---------------------------------------------------------------------------

# The curve sub-options and the curve each belongs to (and is written as).
_PART_OPTIONS = {"debt-parts": "debt", "interest-parts": "interest"}
# "Choices" are what a user expressed, keyed by option name:
#   payloads: VALUE option -> its text (absent = not given, so the default)
#   flags:    FLAG / TOGGLE / SWITCH option -> True or False (absent = not given)


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
    * Curve sub-options: --debt:LETTERS and --interest:LETTERS set the same
      levels, so given on both they must agree. Each also names its curve,
      as --debt and --interest do, unless that curve's flag was given
      explicitly (the window gives every flag, so there it names nothing).
    * A value that belongs to a flag (``turns_on``: --reg:TOL) turns it on,
      unless the flag was given explicitly (--reg:2 --no-reg is off).
    * Measure: -r and -p exclude each other (``check_single_measure``).
    * Right-axis limits are percentages under -r and dollars otherwise
      (``check_macro_bound_units``, once the flags are known).
    """
    values = dataclasses.asdict(PlotConfig())
    for option in options_of(Kind.VALUE):
        if option.name in payloads:
            parsed = option.parse(payloads[option.name])
            values.update(zip(option.fields, parsed if len(option.fields) > 1 else (parsed,)))
        if option.check is not None:
            option.check(values)

    given_parts = {curve: by_name(name).parse(payloads[name]) for name, curve in _PART_OPTIONS.items() if name in payloads}
    if len(set(given_parts.values())) > 1:
        raise ValueError(
            f"--debt:{given_parts['debt']} and --interest:{given_parts['interest']} choose different levels; "
            "give the letters once, or the same on both"
        )
    turned_on = {option.turns_on: True for option in options_of(Kind.VALUE) if option.turns_on and option.name in payloads}
    flags = {**{curve: True for curve in given_parts}, **turned_on, **flags}

    for option in options_of(Kind.FLAG):
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

    check_single_measure(values)
    check_macro_bound_units(payloads, values)
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
            payloads[option.name] = "" if at_moving_default else option.format(field_values, config)
        else:
            flags[option.name] = bool(field_values[0])
    return payloads, flags


def command_line_tokens(payloads: Mapping[str, str], flags: Mapping[str, bool]) -> list[str]:
    """Return the shortest command line (after the script name) that reproduces the GUI choices.

    Options left at their defaults are omitted. The GUI passes every option,
    blanks included; blank values are omitted too.
    """
    defaults, default_flags = choices_from_config(PlotConfig())
    tokens: list[str] = []

    countries = [option for option in options_in("country") if option.in_gui]
    chosen = [option for option in countries if flags.get(option.name)]
    if 0 < len(chosen) < len(countries):
        tokens += [option.flag for option in chosen]

    # Curves: name the ones on, or negate the ones off, whichever is shorter
    # (both mean the same under the curve rule; all off needs every "--no-").
    # With levels chosen every curve is written out, because --debt:LETTERS
    # would otherwise name debt and hide the curves left unnamed.
    curves = options_of(Kind.TOGGLE)
    on = [option for option in curves if flags.get(option.name, True)]
    off = [option for option in curves if option not in on]
    with_levels = any(payloads.get(name, "").strip() for name in _PART_OPTIONS)
    if with_levels:
        tokens += [f"--{option.name}" if option in on else f"--no-{option.name}" for option in curves]
    elif off and (not on or len(off) <= len(on)):
        tokens += [f"--no-{option.name}" for option in off]
    elif off:
        tokens += [f"--{option.name}" for option in on]

    # The measure and the line labels: each switch that is on, spelled out.
    tokens += [
        f"--{option.name}" for option in options_of(Kind.SWITCH) if option.group != "country" and flags.get(option.name)
    ]
    # Values, each by the first spelling the help shows (--dimensions:,
    # --reg:). The levels are written on a curve that is shown: --debt:fp, or
    # --interest:fp when debt is off (both mean the same). A value that turns
    # a flag on (--reg:TOL) is left out while that flag is off, where it has
    # no effect, and otherwise stands for the flag too.
    level_key = "interest" if not flags.get("debt", True) and flags.get("interest", True) else "debt"
    values: list[str] = []
    stood_for: set[str] = set()
    for option in options_of(Kind.VALUE):
        text = payloads.get(option.name, "").strip()
        if not text or text == defaults.get(option.name):
            continue
        if option.turns_on:
            if not flags.get(option.turns_on, default_flags.get(option.turns_on)):
                continue
            stood_for.add(option.turns_on)
        key = f"--{level_key}" if option.name in _PART_OPTIONS else option.usage.split(":")[0]
        values.append(f"{key}:{text}")

    # On/off flags in the window (--cur, --reg): only when not at their default.
    for option in options_of(Kind.FLAG):
        if option.name in stood_for:
            continue
        if option.in_gui and option.name in flags and flags[option.name] != default_flags[option.name]:
            tokens.append(f"--{option.name}" if flags[option.name] else f"--no-{option.name}")
    return tokens + values


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_INDENT = "    "
_HELP_COLUMN = 36  # width of the spellings column (fits "--interest:LETTERS / --i:LETTERS")
_HELP_MIN_GAP = 3  # a spelling longer than the column still gets this many spaces
_HELP_INTRO = """\
Run with no flags to open the interactive window with both charts (Canadian,
then U.S.). Flags set the window's starting values.

Names are case-insensitive, the dashes in front are optional, and any leading
part of a name will do: -r, --rel and --relative are the same. A name is never
matched by its first letter alone (--reg is not -r, --yes is not --yield). An
option that takes a value is written NAME:VALUE, and one that takes none must
not be given one (-r:1 is an error).
"""


def help_epilog() -> str:
    """Return the option reference printed after argparse's own ``--help`` output."""
    lines = _HELP_INTRO.split("\n")
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
