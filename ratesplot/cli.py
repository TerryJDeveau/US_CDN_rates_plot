"""Command-line parsing and the program entry point.

Canadian and/or U.S. benchmark bond yields vs public debt, TTM nominal GDP and
TTM interest outlays, on a two-axis chart per country.

Every option is declared once, in ``options.OPTIONS``; this module only walks
that table (the matching rules are described in the ``options`` docstring).
``--help`` prints the reference generated from the same table. Turning the
recognised choices into a PlotConfig, including the rules that span several
options, is ``options.config_from_choices``, which the GUI shares.
"""

from __future__ import annotations

import argparse
import sys

from .bake import bake_canadian_archives
from .config import PlotConfig
from .options import Kind, Option, config_from_choices, help_epilog, options_of
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


def parse_choices(argv: list[str] | None = None) -> tuple[dict[str, str], dict[str, bool], argparse.ArgumentParser]:
    """Return what the command line explicitly chose, as ``(payloads, flags, parser)``.

    Only options actually given appear (see ``options`` for the shapes). The
    GUI needs this, not just the resulting PlotConfig, to lay the command line
    over the remembered settings. Unrecognised tokens exit via argparse.
    """
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
    # A switch argparse saw as absent counts as "not given".
    args = parser.parse_args(argparse_tokens)
    for option in options_of(Kind.SWITCH):
        if getattr(args, option.fields[0]):
            flags[option.name] = True
    return payloads, flags, parser


def parse_args(argv: list[str] | None = None) -> PlotConfig:
    """Parse the command line into an immutable :class:`PlotConfig`."""
    payloads, flags, parser = parse_choices(argv)
    try:
        return config_from_choices(payloads, flags)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, then open the GUI, bake, or draw the charts in matplotlib windows."""
    payloads, flags, parser = parse_choices(argv)
    try:
        config = config_from_choices(payloads, flags)
    except ValueError as exc:
        parser.error(str(exc))

    if config.bake_archives:
        bake_canadian_archives()
        return

    if config.gui:
        # Imported here so --no-gui runs (and the harnesses) never load tkinter.
        from .gui import run_gui

        run_gui(config, payloads, flags)
        return

    if config.show_cdn:
        run_cdn(config)
    if config.show_us:
        run_us(config)
    print("All requested charts finished.")
