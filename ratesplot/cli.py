"""Command-line parsing and the program entry point.

Canadian and/or U.S. benchmark bond yields vs public debt, TTM nominal GDP and
TTM interest outlays, on a two-axis chart per country.

Usage summary (also printed by ``--help``)
------------------------------------------
Run with no flags to produce both charts (Canadian, then U.S.).

Country selection (case-insensitive; only the first letter matters):
    --C / -C / --canada / --cdn       Canadian chart only
    --U / -U / --us / --usa           U.S. chart only

Curve selection (case-insensitive; only the first letter matters). Naming any
curve positively shows *only* the named curves; ``--no-`` forms hide curves
from the default set of all four:
    --GDP / --no-GDP                  TTM nominal GDP
    --debt / --no-debt                aggregate public debt
    --interest / --no-interest        TTM interest outlays
    --yield / --no-yield              bond yields

Date window (YYYY, YYYY-MM or YYYY-MM-DD; '/' also accepted):
    --start:DATE / --s:DATE           first date (default 1966-01-01)
    --end:DATE / --e:DATE             last date (default today)

Canvas size in pixels (4:3 assumed when only one dimension is given):
    --dimensions:WxH / --d:W / --d:xH   (minimum 800 px each way)

Yield-axis limits (percent):
    --min:VAL / --mn:VAL              lower bound
    --max:VAL / --mx:VAL              upper bound

Dollar-axis limits (suffixes k/m/b/t; unsuffixed values are billions):
    --top:VAL / --t:VAL               upper bound
    --bottom:VAL / --b:VAL            lower bound

Maintenance:
    --bake-archives                   refresh ratesplot/cdn_archive_data.py from
                                      the archived StatCan tables, then exit
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys

import pandas as pd

from .bake import bake_canadian_archives
from .config import DEFAULT_CANVAS_PX, DEFAULT_START, MIN_CANVAS_PX, PlotConfig
from .plotting import run_cdn, run_us

# Curve flags keyed by first letter: ``--g``, ``--debt``, ``--i``, ``--yield`` …
_CURVE_BY_INITIAL = {"g": "gdp", "d": "debt", "i": "interest", "y": "yield"}
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


def validate_date_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Reject reversed ranges and windows shorter than seven days."""
    if end < start:
        raise ValueError(f"end date {end.date()} is before start date {start.date()}")
    if (end - start).days < 7:
        raise ValueError(f"end date {end.date()} is less than 7 days after start date {start.date()}")


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
# Token-level CLI handling
# ---------------------------------------------------------------------------


def consume_custom_cli_token(token: str, state: dict[str, object]) -> bool:
    """Interpret one legacy-style token, storing its raw payload in ``state``.

    Returns True when the token was recognised (and must not reach argparse).
    Recognition is deliberately loose — only the leading letter(s) matter — so
    the many historical spellings (``--dim:``, ``--dimensions:``, ``--GDP``,
    ``--no-gdp`` …) all keep working.
    """
    core = token.lstrip("-")
    if not core:
        return False

    if ":" in core:
        key, _, payload = core.partition(":")
        key = key.lower()
        initial = key[:1]
        if initial in {"d", "s", "e", "t", "b"}:
            state[{"d": "dimensions", "s": "start", "e": "end", "t": "top", "b": "bottom"}[initial]] = payload
            return True
        if key[:2] in {"mx", "ma"}:
            state["ymax"] = payload
            return True
        if key[:2] in {"mn", "mi"}:
            state["ymin"] = payload
            return True
        return False

    lower = core.lower()
    is_negative = lower.startswith("no-")
    option = lower[3:] if is_negative else lower
    curve = _CURVE_BY_INITIAL.get(option[:1])
    if curve is None:
        return False
    state[curve] = not is_negative
    return True


def normalize_country_flag(token: str) -> str | None:
    """Map any ``-c``/``--canada``/``--U``/``--usa`` spelling to argparse's ``--C``/``--U``."""
    initial = token.lstrip("-")[:1].upper()
    return {"C": "--C", "U": "--U"}.get(initial)


def _resolve_curve_selection(custom: dict[str, object]) -> dict[str, bool]:
    """Turn the raw curve flags into four booleans.

    If any curve was named positively, only positively named curves are shown.
    Otherwise every curve is shown except those explicitly negated.
    """
    requests = {curve: custom[curve] for curve in _CURVE_BY_INITIAL.values()}
    if any(value is True for value in requests.values()):
        return {curve: value is True for curve, value in requests.items()}
    return {curve: value is not False for curve, value in requests.items()}


def parse_args(argv: list[str] | None = None) -> PlotConfig:
    """Parse the command line into an immutable :class:`PlotConfig`."""
    parser = argparse.ArgumentParser(
        description="Plot Canadian and/or U.S. benchmark yields vs public debt, TTM GDP, and interest.",
        epilog=__doc__.split("Usage summary", 1)[1].split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--C", dest="show_cdn", action="store_true", default=False, help="Canadian chart only")
    parser.add_argument("--U", dest="show_us", action="store_true", default=False, help="U.S. chart only")
    parser.add_argument(
        "--bake-archives",
        action="store_true",
        default=False,
        help="Download the archived Canadian StatCan tables once and embed them in ratesplot/cdn_archive_data.py.",
    )

    raw_tokens = list(argv) if argv is not None else sys.argv[1:]
    custom: dict[str, object] = dict.fromkeys(
        ("dimensions", "start", "end", "ymin", "ymax", "top", "bottom", *_CURVE_BY_INITIAL.values())
    )
    argparse_tokens: list[str] = []
    for token in raw_tokens:
        if not consume_custom_cli_token(token, custom):
            argparse_tokens.append(normalize_country_flag(token) or token)

    args = parser.parse_args(argparse_tokens)
    curves = _resolve_curve_selection(custom)

    try:
        width_px, height_px = (
            parse_dimensions_spec(str(custom["dimensions"])) if custom["dimensions"] is not None else DEFAULT_CANVAS_PX
        )
        start = parse_date_spec(str(custom["start"]), kind="start") if custom["start"] is not None else DEFAULT_START
        end = (
            parse_date_spec(str(custom["end"]), kind="end")
            if custom["end"] is not None
            else pd.Timestamp.today().normalize()
        )
        validate_date_range(start, end)

        yield_ymin = parse_yield_bound(str(custom["ymin"]), kind="min") if custom["ymin"] is not None else None
        yield_ymax = parse_yield_bound(str(custom["ymax"]), kind="max") if custom["ymax"] is not None else None
        if yield_ymin is not None and yield_ymax is not None and yield_ymin >= yield_ymax:
            raise ValueError(f"yield min ({yield_ymin}) must be less than max ({yield_ymax})")

        macro_top = parse_dollar_bound(str(custom["top"]), kind="top") if custom["top"] is not None else None
        macro_bottom = parse_dollar_bound(str(custom["bottom"]), kind="bottom") if custom["bottom"] is not None else None
        if macro_top is not None and macro_bottom is not None and macro_bottom >= macro_top:
            raise ValueError(f"bottom limit ({macro_bottom}) must be less than top limit ({macro_top})")
    except ValueError as exc:
        parser.error(str(exc))

    # No country flag means both charts.
    show_cdn, show_us = args.show_cdn, args.show_us
    if not (show_cdn or show_us):
        show_cdn = show_us = True

    return PlotConfig(
        start=start,
        end=end,
        width_px=width_px,
        height_px=height_px,
        yield_ymin=yield_ymin,
        yield_ymax=yield_ymax,
        macro_bottom=macro_bottom,
        macro_top=macro_top,
        include_yield=curves["yield"],
        include_debt=curves["debt"],
        include_gdp=curves["gdp"],
        include_interest=curves["interest"],
        show_cdn=show_cdn,
        show_us=show_us,
        bake_archives=args.bake_archives,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the requested charts (or bake), and report completion."""
    config = parse_args(argv)

    if config.bake_archives:
        bake_canadian_archives()
        return

    if config.show_cdn:
        run_cdn(config)
    if config.show_us:
        run_us(config)
    print("All requested charts finished.")
