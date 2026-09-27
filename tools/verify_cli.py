#!/usr/bin/env python3
"""Record how the command line parses, so a parser change can be proved behaviour-neutral.

Each case in ``CASES`` is passed to ``ratesplot.cli.parse_args``. What gets
recorded is the resulting ``PlotConfig`` (the fields that differ from the
defaults), or else the exit code and the text written to stdout/stderr
(``--help``, argparse errors). No data is downloaded and no chart is drawn.

Usage (from the project root)::

    python tools/verify_cli.py out/cli_before.txt
    <change the parser>
    python tools/verify_cli.py out/cli_after.txt --compare out/cli_before.txt

With ``--compare`` each case is reported only if it differs; the exit code is
1 if any case differs. Cases added at the end of ``CASES`` since the earlier
record (for a new option) are counted as NEW, not as differences. The
default end date (today) is a default like any other, so records made on
different days still compare.

The case list is deliberately heavy on odd spellings. The legacy parser
matches on leading letters, so ``--dimensions`` without a colon is the *debt*
curve and ``Canada`` needs no dashes, while ``--bake-archives`` must be spelled
in full. Those behaviours are part of what must not change by accident.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot.cli import parse_args  # noqa: E402
from ratesplot.config import PlotConfig  # noqa: E402

CASES: list[list[str]] = [
    [],
    # Country selection: first letter only, dashes optional.
    ["--C"], ["-C"], ["--canada"], ["--cdn"], ["-c"], ["c"], ["Canada"],
    ["--U"], ["-u"], ["--usa"], ["--us"], ["-c", "-u"], ["--cdn:x"], ["--u:1"],
    # Curves: first letter after an optional "no-".
    ["--GDP"], ["--gdp"], ["-g"], ["--g"], ["gdp"], ["--no-GDP"], ["--no-g"],
    ["--debt"], ["--dept"], ["-d"], ["--dimensions"], ["--no-debt"],
    ["--interest"], ["--i"], ["--no-interest"], ["--yield"], ["--y"], ["--yes"], ["--no-yield"],
    ["--gdp", "--no-debt"], ["--no-gdp", "--no-debt", "--no-interest", "--no-yield"],
    ["--gdp", "--no-gdp"], ["--no-gdp", "--gdp"], ["--gui"], ["--no-gui"],
    # --gui is EXACT: these must not fall through to the GDP curve's first-letter rule.
    ["--GUI"], ["--No-Gui"], ["gui"], ["--gu"], ["--guix"], ["--gui:1"], ["--no-gui", "--gui"],
    ["--no-c"], ["--no-"], ["--no"],
    # Dates.
    ["--start:2001"], ["--s:2001-05"], ["--s:2001/5/7"], ["--START:2001-05-07"], ["--s:"],
    ["--s:20011"], ["--s:2001-13"], ["--s:2001-02-30"], ["--s:2000-02-29"], ["--e:2026-09-01"],
    ["--end:2026"], ["--s:2020", "--e:2020-01-05"], ["--s:2021", "--e:2020"], ["--s:2001", "--s:2005"],
    # Canvas.
    ["--d:1100"], ["--dim:1100x900"], ["--dimensions:x900"], ["--d:800"], ["--D:1100X1000"],
    ["--d:abc"], ["--d:"], ["--d:1100x"], ["--d:4096x3072"], ["--d:x600"],
    # Yield bounds.
    ["--min:1"], ["--mn:0.5"], ["--mi:2"], ["--max:5"], ["--mx:5.5"], ["--ma:6"],
    ["--min:5", "--max:3"], ["--min:-1"], ["--max:100"], ["--min:abc"], ["--m:5"], ["--max"],
    ["--min:"], ["--min:2", "--max:4"],
    # Dollar bounds.
    ["--top:5t"], ["--t:500"], ["--t:2.5T"], ["--bottom:10b"], ["--b:100m"], ["--b:5k"],
    ["--b:0"], ["--b:-1"], ["--b:5t", "--t:1t"], ["--t:x"], ["--t:"], ["--t:k"],
    # Flags handled by argparse, and unknown tokens.
    ["--bake-archives"], ["--bake"], ["--b"], ["--ba"], ["--bottom"], ["--h:1"],
    ["--Bake-Archives"], ["bake-archives"], ["--bake-archive"], ["--no-bake-archives"],
    ["--x"], ["foo"], ["--"], ["-"], ["--help"], ["-h"], ["--h"], ["--he"],
    # A realistic combination.
    ["-c", "--gdp", "--debt", "-s:1990", "--e:2026-09-01", "--d:1100", "--min:1", "--max:5", "--t:5t", "--b:10b"],
    ["-u", "--no-interest", "--s:1980", "--e:1995", "--D:1400x1000"],
]


def record(case: list[str]) -> str:
    """Return a one-block text description of what parsing ``case`` produced."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            config = parse_args(case)
    except SystemExit as exc:
        result = f"exit {exc.code}"
    else:
        # Only fields that differ from PlotConfig's defaults are recorded, so a
        # new option (a new field at its default) leaves earlier records as
        # they were. The moving default end date (today) is a default too.
        defaults = PlotConfig()
        fields = []
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if value != getattr(defaults, field.name):
                fields.append(f"{field.name}={value!r}")
        result = "config " + (", ".join(fields) or "(all defaults)")
    lines = [f"### {case!r}", result]
    if out.getvalue():
        lines.append("stdout:\n" + out.getvalue().rstrip())
    if err.getvalue():
        lines.append("stderr:\n" + err.getvalue().rstrip())
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outfile", type=Path, help="file to write the records into")
    parser.add_argument("--compare", type=Path, help="earlier record file to compare against")
    args = parser.parse_args(argv)

    blocks = [record(case) for case in CASES]
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"[verify_cli] {len(blocks)} cases written to {args.outfile}")
    if args.compare is None:
        return 0

    # Split into cases; the file's final newline belongs to no case, so a case
    # that was last in the earlier record still matches when more follow it.
    previous = args.compare.read_text(encoding="utf-8").rstrip("\n").split("\n\n### ")
    current = args.outfile.read_text(encoding="utf-8").rstrip("\n").split("\n\n### ")
    if len(current) < len(previous):
        print(f"case count shrank: {len(previous)} before, {len(current)} now")
        return 1
    # Cases are only ever appended (for new options), so the earlier record
    # is compared case by case with the start of this one; the rest are new.
    differing = 0
    for before, after in zip(previous, current):
        if before != after:
            differing += 1
            print(f"DIFFER\n--- before\n{before}\n--- after\n{after}\n")
    new = len(current) - len(previous)
    print(f"[verify_cli] {len(previous) - differing} IDENTICAL, {differing} DIFFER, {new} NEW")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
