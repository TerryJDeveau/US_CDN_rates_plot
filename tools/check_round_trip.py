#!/usr/bin/env python3
"""Check that the window's and the page's command line draws what the command line it came from draws; offline, in seconds.

The window and the web page read a command line into their controls
(``frontend.starting_choices``) and write their controls back as a command
line (``options.command_line_tokens``): the one they show, and the page's
address. For each combination of nations and lists below, the written line
must draw the same charts (``options.drawn_config`` for each nation drawn)
as the line it came from. No data is fetched and nothing is drawn.

Found wanting on 2026-10-07: 91 of 192 combinations did not hold. A list of
which a nation could draw nothing (Canada's view of ``--yields:7,20``) was
read back as that nation's default; such a list was then written after a
nation's code, which the parser refuses; and a nation with its own default
(Germany's 2 5 10 30) had its list shortened to a removal counted from the
default for every chart. The last two were fixed first. The first was
Terry's to decide (2026-10-07: "plot no yields in that case, but issue a
warning message"): that nation's field now reads "none", as its chart draws
(``options.NONE_ITEM``), and every combination must hold.

Usage (from the project root)::

    python tools/check_round_trip.py

Exit code 1 if any combination does not hold.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot.cli import sort_tokens  # noqa: E402
from ratesplot.config import NATIONS  # noqa: E402
from ratesplot.frontend import NATION_OPTIONS, choice_values, choices_text, starting_choices  # noqa: E402
from ratesplot.options import command_line_tokens, config_from_choices, drawn_config, nation_choice  # noqa: E402

# The lists a nation may be left none of, and the flag that draws each.
_LISTS = {"yield_terms": "include_yield", "mortgage_terms": "mortgages", "spread_pairs": "spreads"}

# The nations drawn, and the lists and values given: for every chart, for
# one nation, both; lists some nation has none of; removals; limits.
NATION_SETS = ("-c", "-u", "-c -u", "--na:gb", "--na:de", "--na:ca,gb", "--na:us,de", "--na:ca,us,gb,de")
VALUES = (
    "", "--mo", "--sp", "--y:7,20", "--y:20", "--y:3m", "--y:1m,7", "--y:2,10", "--no-y:30", "--yields:3m,2,5,10,30",
    "--mo:30", "--mo:5", "--mo:svr", "--mo:1-5", "--mo:30,svr", "--mo:30,5",
    "--sp:10-3m", "--sp:20-5", "--sp:7-1m", "--sp:30-2", "--sp:7-1m,10-2", "--sp:10-2,10-3m,30-10",
    "--us:y:7 --y:20", "--cdn:y:2 --y:7", "--y:7,20 --uk:y:10", "--y:1,7 --cdn:y:3m --uk:y:5",
    "--us:no-y:30 --y:2,30", "--uk:no-y:20", "--ger:no-y:2 --us:y:1m",
    "--uk:mo:2f --mo:30", "--mo:15 --cdn:mo:prime --ger:mo:over10", "--y:20,30 --mo:svr,5-10",
    "--ger:sp:30-2 --sp:10-3m", "--sp:20-10 --cdn:sp:10-3m", "--sp:30-10 --us:sp:7-1m", "--y:7 --mo:15 --sp:7-1m",
    "--max:6 --us:max:8", "--uk:top:5t --ger:top:4t", "-r --uk:top:150%", "--d:fpm --y:7",
    # A nation's own list of none (2026-10-07), alone, with every nation's, and with a list for every chart.
    "--cdn:y:none", "--cdn:y:none --us:y:none", "--uk:y:none --y:7", "--ger:mo:none --mo", "--us:mo:none --mo:30,5",
    "--cdn:sp:none --sp:10-2", "--cdn:mo:none --us:mo:none --uk:mo:none --ger:mo:none --mo",
)


def round_trip(line: list[str]) -> tuple[list[str], list[str]] | None:
    """Return (the written line, the nations it draws otherwise), or None for a line the parser refuses.

    Written twice: from the fields as the window keeps them, and from the
    fields as the page rebuilds each list from its ticked boxes
    (``through_boxes``); both must draw what the line drew. The written line
    returned is the page's where they differ.
    """
    payloads, flags, unmatched, misspelled = sort_tokens(line)
    if unmatched or misspelled:
        return None
    try:
        config = config_from_choices(payloads, flags)
    except ValueError:
        return None
    (choice_payloads, choice_flags), _notes = starting_choices(config, payloads, flags, {})
    differ: list[str] = []
    for fields in (choice_payloads, through_boxes(choice_payloads)):
        written = command_line_tokens(fields, choice_flags)
        try:
            again = config_from_choices(*sort_tokens(written)[:2])
        except ValueError as exc:
            return written, [f"refused: {exc}"]
        differ += [
            nation.key for nation in NATIONS
            if (getattr(config, nation.show_field) != getattr(again, nation.show_field)
                or (getattr(config, nation.show_field) and _drawn(config, nation.key) != _drawn(again, nation.key)))
            and nation.key not in differ
        ]
    return written, differ


def through_boxes(payloads: dict[str, str]) -> dict[str, str]:
    """Return ``payloads`` with each nation's list as the page's boxes give it back: the items ticked, "none" for none."""
    boxed = dict(payloads)
    for option in NATION_OPTIONS:
        if option.editor == "choices":
            for nation in NATIONS:
                key = nation_choice(nation, option)
                boxed[key] = choices_text(option, choice_values(option, payloads[key], nation))
    return boxed


def _drawn(config, key: str):
    """``drawn_config`` for one nation, with each list whose curves are off set aside (it draws nothing either way)."""
    drawn = drawn_config(config, key)
    hidden = {field: None for field, flag in _LISTS.items() if not getattr(drawn, flag)}
    return replace(drawn, **hidden) if hidden else drawn


def main() -> int:
    checked, failed = 0, 0
    for nations, values in itertools.product(NATION_SETS, VALUES):
        line = f"{nations} {values}".split()
        result = round_trip(line)
        if result is None:
            continue
        checked += 1
        written, differ = result
        if differ:
            failed += 1
            print(f"FAIL {' '.join(line)}  ->  {' '.join(written)}  [{', '.join(differ)}]")
    print(f"[round trip] {checked - failed} of {checked} command lines come back drawing the same charts; {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
