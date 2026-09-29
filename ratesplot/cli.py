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

from .config import PlotConfig
from .options import OPTIONS, Kind, Option, config_from_choices, help_epilog, options_of, spells
from .http import enable_download_cache
from .plotting import resolve_start, run_cdn, run_us


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------

_WITHOUT_VALUE = (Kind.FLAG, Kind.TOGGLE, Kind.SWITCH)
_WITH_OFF_FORM = (Kind.FLAG, Kind.TOGGLE)  # the kinds that take "no-"


def _spelled(token: str, word: str, kinds: tuple[Kind, ...]) -> Option | None:
    """Return the option of one of ``kinds`` that ``word`` spells, or None; raise if two do."""
    found = [option for option in OPTIONS if option.kind in kinds and spells(option, word)]
    if len(found) > 1:
        raise ValueError(f"{token} is ambiguous: it could be " + " or ".join(f"--{option.name}" for option in found))
    return found[0] if found else None


def match_token(token: str) -> tuple[Option, str | bool] | None:
    """Return the option a token spells, with its raw value, or None to hand it to argparse.

    The raw value is the text after the colon for ``VALUE`` options, and
    True/False (False when prefixed ``no-``) for the others. Raises
    ValueError for a token that spells an option in the wrong form: a value
    for one that takes none (``-r:1``), none for one that needs it
    (``--dimensions``), or a leading part two options share (``--m:5``).
    The rules are in the ``options`` docstring.
    """
    core = token.lstrip("-")
    if not core:
        return None
    lower = core.lower()

    if ":" in lower:
        key = lower.partition(":")[0]
        option = _spelled(token, key, (Kind.VALUE,))
        if option is not None:
            return option, core.partition(":")[2]
        # Never another option with the same first letter: "-r:1" is not "--reg:1".
        takes_none = _spelled(token, key, _WITHOUT_VALUE)
        if takes_none is not None:
            raise ValueError(f"{token}: --{takes_none.name} takes no value")
        return None

    is_negative = lower.startswith("no-")
    name = lower[3:] if is_negative else lower
    option = _spelled(token, name, _WITH_OFF_FORM if is_negative else _WITHOUT_VALUE)
    if option is not None:
        return option, not is_negative
    if not is_negative:
        needs_value = _spelled(token, name, (Kind.VALUE,))
        if needs_value is not None:
            raise ValueError(f"{token} needs a value: {needs_value.usage}")
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser, with the generated reference as epilog.

    The ``SWITCH`` options are declared to it for its usage line and
    ``--help`` only: ``match_token`` recognises them, so argparse never
    receives one (nor gets to apply its own abbreviation rules to them).
    """
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
    over the remembered settings. Unrecognised and misspelled tokens exit
    via argparse.
    """
    parser = build_parser()
    raw_tokens = list(argv) if argv is not None else sys.argv[1:]

    # Sort tokens: table-matched ones are recorded (last occurrence wins),
    # everything else goes to argparse.
    payloads: dict[str, str] = {}
    flags: dict[str, bool] = {}
    argparse_tokens: list[str] = []
    misspelled: list[str] = []
    for token in raw_tokens:
        try:
            matched = match_token(token)
        except ValueError as exc:
            misspelled.append(str(exc))
            continue
        if matched is None:
            argparse_tokens.append(token)
        elif matched[0].kind is Kind.VALUE:
            payloads[matched[0].name] = str(matched[1])
        else:
            flags[matched[0].name] = bool(matched[1])

    # argparse sees only what the table did not match: it prints --help, or
    # reports unrecognised tokens, before misspelled ones and bad values.
    parser.parse_args(argparse_tokens)
    if misspelled:
        parser.error(misspelled[0])
    return payloads, flags, parser


def parse_args(argv: list[str] | None = None) -> PlotConfig:
    """Parse the command line into an immutable :class:`PlotConfig`."""
    payloads, flags, parser = parse_choices(argv)
    try:
        return config_from_choices(payloads, flags)
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, then open the GUI or draw the charts in matplotlib windows."""
    payloads, flags, parser = parse_choices(argv)
    try:
        config = config_from_choices(payloads, flags)
    except ValueError as exc:
        parser.error(str(exc))

    if config.gui:
        # Imported here so --no-gui runs (and the harnesses) never load tkinter.
        from .gui import run_gui

        run_gui(config, payloads, flags)
        return

    if config.start is None:
        # The start is found from the data (plotting.resolve_start), which are
        # then prepared again from it: keep each download for this run, so no
        # source is fetched twice.
        enable_download_cache()
        config = resolve_start(config)
    if config.show_cdn:
        run_cdn(config)
    if config.show_us:
        run_us(config)
    print("All requested charts finished.")
