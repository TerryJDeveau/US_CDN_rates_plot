"""Command-line parsing and the program entry point.

Canadian and/or U.S. benchmark bond yields vs public debt, TTM nominal GDP and
TTM interest outlays, on a two-axis chart per country.

Every option is declared once, in ``options.OPTIONS``; this module only walks
that table (the matching rules are described in the ``options`` docstring).
``--help`` prints the reference generated from the same table. The two rules
here that involve several options at once are the curve-selection rule and
"no country flag means both charts".
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .bake import bake_canadian_archives
from .config import PlotConfig
from .options import Kind, Option, help_epilog, options_of
from .plotting import run_cdn, run_us


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------


def match_token(token: str) -> tuple[Option, str | bool] | None:
    """Return the option a token names, with its raw value, or None to hand it to argparse.

    The raw value is the text after the colon for ``VALUE`` options, and
    True/False (False when prefixed ``no-``) for ``EXACT`` and ``TOGGLE``.
    ``SWITCH`` options are never returned here; argparse handles them.
    """
    core = token.lstrip("-")
    if not core:
        return None

    lower = core.lower()
    is_negative = lower.startswith("no-")
    name = lower[3:] if is_negative else lower

    for option in options_of(Kind.EXACT):
        if name == option.name:
            return option, not is_negative

    if ":" in core:
        key = lower.partition(":")[0]
        payload = core.partition(":")[2]
        for option in options_of(Kind.VALUE):
            if key.startswith(option.prefixes):
                return option, payload
        return None

    for option in options_of(Kind.TOGGLE):
        if name.startswith(option.prefixes):
            return option, not is_negative
    return None


def to_argparse_token(token: str) -> str:
    """Map a legacy first-letter spelling (``-c``, ``Canada``, ``--usa`` …) to its argparse flag.

    Tokens that are not such a spelling are returned unchanged, so argparse
    can apply its own prefix matching or report them as unrecognised.
    """
    initial = token.lstrip("-")[:1].upper()
    for option in options_of(Kind.SWITCH):
        if option.initial is not None and initial == option.initial:
            return option.flag
    return token


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the ``SWITCH`` options, with the generated reference as epilog."""
    parser = argparse.ArgumentParser(
        description="Plot Canadian and/or U.S. benchmark yields vs public debt, TTM GDP, and interest.",
        epilog=help_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for option in options_of(Kind.SWITCH):
        parser.add_argument(option.flag, dest=option.fields[0], action="store_true", default=False, help=option.flag_help)
    return parser


def _resolve_curve_selection(requests: dict[str, bool | None]) -> dict[str, bool]:
    """Turn the curve flags (True, False or not given) into one boolean per field.

    If any curve was named positively, only positively named curves are shown.
    Otherwise every curve is shown except those explicitly negated.
    """
    if any(value is True for value in requests.values()):
        return {field: value is True for field, value in requests.items()}
    return {field: value is not False for field, value in requests.items()}


def parse_args(argv: list[str] | None = None) -> PlotConfig:
    """Parse the command line into an immutable :class:`PlotConfig`."""
    parser = build_parser()
    raw_tokens = list(argv) if argv is not None else sys.argv[1:]

    # Sort tokens: table-matched ones are recorded (last occurrence wins),
    # everything else goes to argparse.
    payloads: dict[str, str] = {}
    flags: dict[str, bool] = {}
    argparse_tokens: list[str] = []
    for token in raw_tokens:
        matched = match_token(token)
        if matched is None:
            argparse_tokens.append(to_argparse_token(token))
        elif matched[0].kind is Kind.VALUE:
            payloads[matched[0].name] = str(matched[1])
        else:
            flags[matched[0].name] = bool(matched[1])

    # argparse first, so unrecognised tokens are reported before bad values.
    args = parser.parse_args(argparse_tokens)

    # Start from PlotConfig's defaults (the default end date is today) and
    # overwrite what was given. Values are parsed in table order, and each
    # option's cross-check runs straight after it, against everything so far.
    values = dataclasses.asdict(PlotConfig())
    try:
        for option in options_of(Kind.VALUE):
            if option.name in payloads:
                parsed = option.parse(payloads[option.name])
                values.update(zip(option.fields, parsed if len(option.fields) > 1 else (parsed,)))
            if option.check is not None:
                option.check(values)
    except ValueError as exc:
        parser.error(str(exc))

    for option in options_of(Kind.EXACT):
        if option.name in flags:
            values[option.fields[0]] = flags[option.name]

    curve_requests = {option.fields[0]: flags.get(option.name) for option in options_of(Kind.TOGGLE)}
    values.update(_resolve_curve_selection(curve_requests))

    for option in options_of(Kind.SWITCH):
        values[option.fields[0]] = getattr(args, option.fields[0])
    # No country flag means both charts.
    if not (values["show_cdn"] or values["show_us"]):
        values["show_cdn"] = values["show_us"] = True

    return PlotConfig(**values)


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
