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

from .config import NATIONS, Nation, PlotConfig, nation_by_code
from .options import (
    OPTIONS,
    Kind,
    Option,
    config_from_choices,
    help_epilog,
    nation_choice,
    options_in,
    options_of,
    per_nation,
    spells,
)
from .http import enable_download_cache
from .plotting import resolve_start, run_cdn, run_us


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------

_WITHOUT_VALUE = (Kind.FLAG, Kind.TOGGLE, Kind.SWITCH)
_WITH_OFF_FORM = (Kind.FLAG, Kind.TOGGLE)  # the kinds that take "no-"


def _spelled(token: str, word: str, kinds: tuple[Kind, ...], *, removes: bool = False) -> Option | None:
    """Return the option of one of ``kinds`` that ``word`` spells, or None; raise if two do.

    ``removes`` picks the VALUE options spelled after "no-" (``Option.removes``)
    instead of the others; it never matters for the other kinds.
    """
    found = [
        option for option in OPTIONS if option.kind in kinds and option.removes == removes and spells(option, word)
    ]
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
        # "no-" before a value: an option that takes its list out (--no-yields:30).
        if key.startswith("no-"):
            option = _spelled(token, key[3:], (Kind.VALUE,), removes=True)
            if option is not None:
                return option, core.partition(":")[2]
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


def match_prefixed(token: str) -> tuple[Nation, Option, str] | None:
    """Return ``(nation, option, raw value)`` for a value given for one nation (``--us:top:20t``), or None.

    The part before the first colon must be a nation's code exactly; the rest
    is matched as a token of its own (``match_token``), and must be a value
    option that can differ by nation (``options.per_nation``). None when the
    token has no nation's code in front (a country written otherwise,
    ``--u:top:20t``, is an error), or when the rest spells nothing and
    has no value (``--cdn:x``, which ``match_token`` then reports as before:
    --cdn takes no value). Raises ValueError for anything else after a code.
    """
    code, colon, rest = token.lstrip("-").partition(":")
    nation = nation_by_code(code) if colon else None
    if nation is None and ":" in rest and _spelled(token, code.lower(), (Kind.SWITCH,)) in options_in("country"):
        codes = ", ".join(code for nation in NATIONS for code in nation.codes)
        raise ValueError(f"{token}: a nation's code is written in full ({codes})")
    if nation is None or not rest:
        return None
    try:
        matched = match_token(f"--{rest}")
    except ValueError as exc:
        raise ValueError(f"{token}: {exc}") from None
    if matched is None:
        if ":" not in rest:
            return None
        raise ValueError(f"{token}: {rest.partition(':')[0]} is not an option")
    option, value = matched
    if not per_nation(option):
        allowed = ", ".join(other.usage.split(" / ")[0] for other in OPTIONS if per_nation(other))
        raise ValueError(f"{token}: --{option.name} is the same for every chart; a nation's code goes only on {allowed}")
    return nation, option, str(value)


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


def sort_tokens(raw_tokens: list[str]) -> tuple[dict[str, str], dict[str, bool], list[str], list[str]]:
    """Sort command-line tokens into ``(payloads, flags, unmatched, misspelled)``; never exits.

    Table-matched tokens are recorded (the last occurrence wins; a value for
    one nation under "NATION:OPTION", ``match_prefixed``); ``unmatched``
    are the tokens the table does not know (``--help``, or unrecognised), for
    argparse; ``misspelled`` holds a message for each token that spells an
    option in the wrong form (``match_token``). The web page reads a chart's
    command line from its address with this, where exiting is not an option.
    """
    payloads: dict[str, str] = {}
    flags: dict[str, bool] = {}
    unmatched: list[str] = []
    misspelled: list[str] = []
    for token in raw_tokens:
        try:
            prefixed = match_prefixed(token)
            matched = None if prefixed else match_token(token)
        except ValueError as exc:
            misspelled.append(str(exc))
            continue
        if prefixed:
            nation, option, value = prefixed
            payloads[nation_choice(nation, option)] = value
        elif matched is None:
            unmatched.append(token)
        elif matched[0].kind is Kind.VALUE:
            payloads[matched[0].name] = str(matched[1])
        else:
            flags[matched[0].name] = bool(matched[1])
    return payloads, flags, unmatched, misspelled


def parse_choices(argv: list[str] | None = None) -> tuple[dict[str, str], dict[str, bool], argparse.ArgumentParser]:
    """Return what the command line explicitly chose, as ``(payloads, flags, parser)``.

    Only options actually given appear (see ``options`` for the shapes). The
    GUI needs this, not just the resulting PlotConfig, to lay the command line
    over the remembered settings. Unrecognised and misspelled tokens exit
    via argparse.
    """
    parser = build_parser()
    raw_tokens = list(argv) if argv is not None else sys.argv[1:]
    payloads, flags, argparse_tokens, misspelled = sort_tokens(raw_tokens)

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
