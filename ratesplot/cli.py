"""
US_CDN_rates_plot_9.py
======================
Canadian and/or U.S. benchmark bond yields vs public debt, TTM nominal GDP,
and TTM interest outlays (1996–present).

This version keeps the original command-line behavior while separating:
    1. configuration and CLI parsing,
    2. data acquisition/transformation, and
    3. chart construction/formatting.

CLI compatibility
-----------------
Run with no flags to produce both charts (Canadian, then U.S.).

Country selection (case-insensitive; first character is significant):
    --C / -C / --canada / --cdn   -> Canadian chart only
    --U / -U / --us / --usa       -> U.S. chart only

Curve selection (case-insensitive; first character is significant):
    --GDP / --no-GDP              -> include/exclude TTM Nominal GDP
    --debt / --no-debt            -> include/exclude Public Debt
    --interest / --no-interest    -> include/exclude TTM Interest
    --yield / --no-yield          -> include/exclude Bond Yields

Yield-axis limits:
    --min:VAL / --mn:VAL / --minimum:VAL -> lower bound
    --max:VAL / --mx:VAL / --maxim:VAL    -> upper bound

Dollar-axis limits:
    --top:VAL / --t:VAL             -> upper bound
    --bottom:VAL / --b:VAL          -> lower bound

Dollar values accept b/B, t/T, m/M, and k/K suffixes. Unsuffixed values
are interpreted as billions, matching the original script.
"""

from __future__ import annotations

import argparse
import datetime
import sys

import pandas as pd

from .bake import bake_canadian_archives
from .config import DEFAULT_CANVAS_PX, DEFAULT_START, MIN_CANVAS_PX, PlotConfig
from .plotting import run_cdn, run_us


def days_in_month(year: int, month: int) -> int:
    """Return the number of days in a calendar month without external helpers."""
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    is_leap_year = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return 29 if is_leap_year else 28


def parse_date_spec(spec: str, *, kind: str) -> pd.Timestamp:
    """Parse YYYY, YYYY-MM, or YYYY-MM-DD (also accepting '/')."""
    value = spec.strip()
    if not value:
        raise ValueError(f"empty {kind} date spec")

    parts = value.replace("/", "-").split("-")
    if len(parts) > 3:
        raise ValueError(f"invalid {kind} date {spec!r}: too many components")

    year_text = parts[0]
    if not (year_text.isdigit() and len(year_text) == 4):
        raise ValueError(f"invalid {kind} year {year_text!r}: must be exactly 4 digits")
    year = int(year_text)

    if len(parts) >= 2:
        month_text = parts[1]
        if not month_text.isdigit() or int(month_text) == 0:
            raise ValueError(f"invalid {kind} month {month_text!r}: must be 1–12")
        month = int(month_text)
        if not 1 <= month <= 12:
            raise ValueError(f"invalid {kind} month {month}: must be 1–12")
    else:
        month = 1

    if len(parts) == 3:
        day_text = parts[2]
        if not day_text.isdigit() or int(day_text) == 0:
            raise ValueError(f"invalid {kind} day {day_text!r}: must be a positive integer")
        day = int(day_text)
        max_day = days_in_month(year, month)
        if day > max_day:
            raise ValueError(
                f"invalid {kind} day {day} for {year}-{month:02d}: month has only {max_day} days"
            )
    else:
        day = 1

    return pd.Timestamp(year=year, month=month, day=day)


def validate_date_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Reject reversed ranges and windows shorter than seven days."""
    if end < start:
        raise ValueError(f"end date {end.date()} is before start date {start.date()}")
    if (end - start).days < 7:
        raise ValueError(
            f"end date {end.date()} is less than 7 days after start date {start.date()}"
        )


def parse_dimensions_spec(spec: str) -> tuple[int, int]:
    """Parse width, height, or aspect-preserving xN/NxN dimensions."""
    value = spec.strip().lower()
    if not value:
        raise ValueError("empty dimensions spec")

    if value.startswith("x"):
        if not value[1:].isdigit():
            raise ValueError(f"dimensions height must be digits only, got {spec!r}")
        height = int(value[1:])
        width = int(round(height * 4 / 3))
    elif "x" in value:
        width_text, height_text = value.split("x", 1)
        if not width_text.isdigit() or not height_text.isdigit():
            raise ValueError(f"dimensions must be digits around 'x', got {spec!r}")
        width, height = int(width_text), int(height_text)
    else:
        if not value.isdigit():
            raise ValueError(f"dimensions width must be digits only, got {spec!r}")
        width = int(value)
        height = int(round(width * 3 / 4))

    if width < MIN_CANVAS_PX or height < MIN_CANVAS_PX:
        raise ValueError(
            f"canvas {width}x{height} px is below minimum size of {MIN_CANVAS_PX} px"
        )
    return width, height


def parse_yield_bound(spec: str, *, kind: str) -> float:
    """Parse a yield axis bound as a value in [0, 100)."""
    value = spec.strip()
    if not value:
        raise ValueError(f"empty {kind} yield bound")
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(
            f"invalid {kind} yield bound {spec!r}: must be a decimal number"
        ) from None
    if parsed < 0 or parsed >= 100:
        raise ValueError(f"{kind} yield bound must be non-negative and < 100, got {parsed}")
    return parsed


def parse_dollar_bound(spec: str, *, kind: str) -> float:
    """Parse a dollar-axis bound, defaulting unsuffixed values to billions."""
    value = spec.strip()
    if not value:
        raise ValueError(f"empty {kind} dollar limit")

    multipliers = {
        "t": 1e12,
        "b": 1e9,
        "m": 1e6,
        "k": 1e3,
    }
    normalized = value.lower()
    suffix = normalized[-1] if normalized and normalized[-1] in multipliers else None
    multiplier = multipliers.get(suffix, 1e9)
    numeric_part = normalized[:-1].strip() if suffix else normalized

    try:
        parsed = float(numeric_part)
    except ValueError:
        raise ValueError(
            f"invalid {kind} dollar limit {spec!r}: must be a positive decimal value"
        ) from None
    if parsed <= 0:
        raise ValueError(f"{kind} dollar limit must be positive, got {parsed}")
    return parsed * multiplier


def consume_custom_cli_token(
    token: str,
    state: dict[str, object],
) -> bool:
    """Consume one custom token and store its parsed payload in ``state``.

    The original CLI intentionally accepts many aliases by inspecting only the
    first character(s), so this helper preserves that compatibility in one place.
    """
    core = token.lstrip("-")
    if not core:
        return False

    if ":" in core:
        key, _, payload = core.partition(":")
        key_lower = key.lower()
        first_letter = key_lower[:1]

        if first_letter == "d":
            state["dimensions"] = payload
            return True
        if first_letter == "s":
            state["start"] = payload
            return True
        if first_letter == "e":
            state["end"] = payload
            return True
        if first_letter == "t":
            state["top"] = payload
            return True
        if first_letter == "b":
            state["bottom"] = payload
            return True
        if first_letter == "m" and len(key_lower) >= 2:
            prefix = key_lower[:2]
            if prefix in {"mx", "ma"}:
                state["ymax"] = payload
                return True
            if prefix in {"mn", "mi"}:
                state["ymin"] = payload
                return True

    lower = core.lower()
    is_negative = lower.startswith("no-")
    base_option = lower[3:] if is_negative else lower
    if base_option:
        first = base_option[0]
        if first in {"g", "d", "i", "y"}:
            state[{"g": "gdp", "d": "debt", "i": "interest", "y": "yield"}[first]] = not is_negative
            return True

    return False


def normalize_country_flag(token: str) -> str | None:
    """Normalize legacy country spellings to argparse's ``--C`` or ``--U``."""
    core = token.lstrip("-")
    if not core:
        return None
    first = core[0].upper()
    if first == "C":
        return "--C"
    if first == "U":
        return "--U"
    return None


def parse_args(argv: list[str] | None = None) -> PlotConfig:
    """Parse CLI arguments and return an immutable runtime configuration."""
    parser = argparse.ArgumentParser(
        description="Plot Canadian and/or U.S. benchmark yields vs public debt, TTM GDP, and interest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--C", dest="show_cdn", action="store_true", default=False)
    parser.add_argument("--U", dest="show_us", action="store_true", default=False)
    parser.add_argument("--bake-archives", action="store_true", default=False,
                        help="Download inactive Canadian StatCan archives once and embed them in ratesplot/cdn_archive_data.py.")

    raw_tokens = list(argv) if argv is not None else sys.argv[1:]
    normalized_tokens: list[str] = []
    custom: dict[str, object] = {
        "dimensions": None,
        "start": None,
        "end": None,
        "ymin": None,
        "ymax": None,
        "top": None,
        "bottom": None,
        "gdp": None,
        "debt": None,
        "interest": None,
        "yield": None,
    }

    for token in raw_tokens:
        if consume_custom_cli_token(token, custom):
            continue
        country_flag = normalize_country_flag(token)
        normalized_tokens.append(country_flag or token)

    args = parser.parse_args(normalized_tokens)

    # A positive selection means only explicitly enabled curves are shown;
    # otherwise the absence of --no-* leaves that curve enabled by default.
    curve_requests = [custom["gdp"], custom["debt"], custom["interest"], custom["yield"]]
    any_positive = any(value is True for value in curve_requests)
    if any_positive:
        include_gdp = custom["gdp"] is True
        include_debt = custom["debt"] is True
        include_interest = custom["interest"] is True
        include_yield = custom["yield"] is True
    else:
        include_gdp = custom["gdp"] is not False
        include_debt = custom["debt"] is not False
        include_interest = custom["interest"] is not False
        include_yield = custom["yield"] is not False

    try:
        if custom["dimensions"] is not None:
            width_px, height_px = parse_dimensions_spec(str(custom["dimensions"]))
        else:
            width_px, height_px = DEFAULT_CANVAS_PX

        start = (
            parse_date_spec(str(custom["start"]), kind="start")
            if custom["start"] is not None
            else DEFAULT_START
        )
        end = (
            parse_date_spec(str(custom["end"]), kind="end")
            if custom["end"] is not None
            else pd.Timestamp(datetime.date.today())
        )
        validate_date_range(start, end)

        yield_ymin = (
            parse_yield_bound(str(custom["ymin"]), kind="min")
            if custom["ymin"] is not None
            else None
        )
        yield_ymax = (
            parse_yield_bound(str(custom["ymax"]), kind="max")
            if custom["ymax"] is not None
            else None
        )
        if yield_ymin is not None and yield_ymax is not None and yield_ymin >= yield_ymax:
            raise ValueError(f"yield min ({yield_ymin}) must be less than max ({yield_ymax})")

        macro_top = (
            parse_dollar_bound(str(custom["top"]), kind="top")
            if custom["top"] is not None
            else None
        )
        macro_bottom = (
            parse_dollar_bound(str(custom["bottom"]), kind="bottom")
            if custom["bottom"] is not None
            else None
        )
        if macro_top is not None and macro_bottom is not None and macro_bottom >= macro_top:
            raise ValueError(f"bottom limit ({macro_bottom}) must be less than top limit ({macro_top})")
    except ValueError as exc:
        parser.error(str(exc))

    show_cdn = args.show_cdn
    show_us = args.show_us
    if not show_cdn and not show_us:
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
        include_yield=include_yield,
        include_debt=include_debt,
        include_gdp=include_gdp,
        include_interest=include_interest,
        show_cdn=show_cdn,
        show_us=show_us,
        bake_archives=args.bake_archives,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, run the requested charts, and report completion."""
    config = parse_args(argv)

    if config.bake_archives:
        bake_canadian_archives()
        return

    if config.show_cdn:
        run_cdn(config)
    if config.show_us:
        run_us(config)

    print("All requested charts finished.")
