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

The case list is deliberately heavy on odd spellings: ``Canada`` needs no
dashes, ``--rel`` is ``-r``. Those behaviours are part of what must not
change by accident. Until 2026-09-29 the parser matched on first letters,
so ``--dimensions`` without a colon was the *debt* curve and ``--reg:1`` was
``-r``; since then a token must be a leading part of an option's name, and
the records of those cases changed with it. (The ``--bake-archives`` cases
stay: since the bake moved to ``tools/bake_archives.py`` they record that
the option is gone.)
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
    # Country selection: any leading part of a name, dashes optional.
    ["--C"], ["-C"], ["--canada"], ["--cdn"], ["-c"], ["c"], ["Canada"],
    ["--U"], ["-u"], ["--usa"], ["--us"], ["-c", "-u"], ["--cdn:x"], ["--u:1"],
    # Curves: any leading part of a name, after an optional "no-".
    ["--GDP"], ["--gdp"], ["-g"], ["--g"], ["gdp"], ["--no-GDP"], ["--no-g"],
    ["--debt"], ["--dept"], ["-d"], ["--dimensions"], ["--no-debt"],
    ["--interest"], ["--i"], ["--no-interest"], ["--yield"], ["--y"], ["--yes"], ["--no-yield"],
    ["--gdp", "--no-debt"], ["--no-gdp", "--no-debt", "--no-interest", "--no-yield"],
    ["--gdp", "--no-gdp"], ["--no-gdp", "--gdp"], ["--gui"], ["--no-gui"],
    # --gui is spelled in full: these must not fall through to the GDP curve.
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
    # Appended cases (new options). Earlier records compare against the cases above.
    # -r: debt and interest as % of GDP; a switch, like the countries.
    ["-r"], ["--r"], ["--R"], ["--relative"], ["Relative"], ["--rel"], ["--no-relative"],
    ["-r", "--no-gdp"], ["-r", "--gdp"], ["-r", "-c", "--debt", "-s:1926"],
    # Right-axis limits: percentages with -r, dollars without.
    ["-r", "--top:150%"], ["-r", "--top:150"], ["-r", "--b:5%", "--t:200%"], ["-r", "--b:200%", "--t:5%"],
    ["--top:150%"], ["-r", "--t:x%"], ["-r", "--t:0%"], ["-r", "--t: 80 %"],
    # -p: per capita; not with -r.
    ["-p"], ["--p"], ["--P"], ["--per-capita"], ["--percapita"], ["Per"], ["--no-per-capita"],
    ["-p", "-r"], ["-r", "-p"], ["-p", "--top:100k", "--b:100"], ["-p", "--top:50%"], ["-p", "-c", "-s:1867"],
    # Canvas size needs at least "dim" since 2026-09-27: "--d:" is left for the debt sub-options.
    ["--dim:1100"], ["--DIM:1100X1000"], ["--dims:x900"], ["-dim:800x800"], ["--dimension:1400"], ["--di:1100"],
    # Debt and interest by level: letters f n p/s m in any order; the option names its curve.
    ["--d:fp"], ["--debt:PF"], ["--debt:s"], ["--d:ps"], ["--d:fnpsm"], ["--d:ff"], ["-i:m"], ["--interest:nm", "--gdp"],
    ["--d:fp", "--i:pf"], ["--d:fp", "--i:m"], ["--d:fp", "--no-debt", "--interest"], ["--d:"], ["--d:fx"], ["--d:1100"],
    ["--d:fp", "-r"], ["--d:fp", "-p", "-u"], ["--dept:f"],
    # -l: each line's last value at its end; a switch like the measures, and combines with them.
    ["-l"], ["--l"], ["--L"], ["--label"], ["--labels"], ["Label"], ["--lab"], ["--no-label"], ["--l:1"],
    ["-l", "-r", "-c"], ["-l", "-p", "-u", "--d:fp"], ["-L", "--gdp"],
    # --cur: on by default; at least "cur", so never Canada; --no-cur turns it off.
    ["--cur"], ["--curr"], ["--current"], ["--CURRENT"], ["--no-cur"], ["--no-current"], ["--No-Cur"],
    ["--cu"], ["--cur:1"], ["current"], ["-c", "--no-cur"], ["--no-cur", "--cur"], ["--no-cur", "-l", "-u"],
    # --reg: off by default; at least "reg", so never -r; "--re" and "--rel" are still -r.
    ["--reg"], ["--regr"], ["--regression"], ["--REGRESSION"], ["reg"], ["-reg"], ["--no-reg"], ["--no-regression"],
    ["--re"], ["--rel"], ["--reg:1"], ["--reg", "-r"], ["--reg", "-l", "-p", "-c"], ["--no-reg", "--reg"], ["--reg", "--no-reg"],
    # 2026-09-29: a token must be a leading part of a name (Terry: "--reg must never invoke -r";
    # "-r:yyy is an error, not a shorthand for --reg:yyy"). A value on a flag is an error, a
    # value option needs its value, and two options sharing a leading part is an error.
    ["-r:1"], ["-r:yyy"], ["--r:1"], ["--rel:1"], ["--relative:1"], ["--ra"], ["--ratio"], ["--relatives"],
    ["--regx"], ["--regressions"], ["--curx"], ["--currents"], ["--c:1"], ["-c:1"], ["--g:1"], ["--y:1"],
    ["--canadas"], ["--cdnx"], ["--usb"], ["pizza"], ["--per"], ["--per_capita"], ["--percap"], ["--pc"],
    ["--yields"], ["--no-yields"], ["--yieldx"], ["--gdpx"], ["--debts"], ["--interests"], ["--lab"], ["--labelx"],
    ["--deb:fp"], ["--de:fp"], ["--int:m"], ["--dime:1100"], ["--no-dim"], ["--no-d:fp"],
    ["--st:2001"], ["--starts:2001"], ["--en:2026"], ["--ends:2026"], ["--start"], ["--s"], ["--e"],
    ["--minimum:2"], ["--maximum:5"], ["--minimums:2"], ["--mix:2"], ["--to:5t"], ["--bot:10b"], ["--tops:5t"],
    # --reg:TOL: the regression's tolerance in % of the axis height; turns --reg on unless --no-reg is given.
    ["--reg:1.5"], ["--reg:1.5%"], ["--regression:2"], ["--REG: 0.5 %"], ["--regr:3"], ["--reg:1e-3"],
    ["--reg:2", "--no-reg"], ["--no-reg", "--reg:2"], ["--reg", "--reg:2"], ["--reg:2", "-r", "-c"],
    ["--reg:0"], ["--reg:-1"], ["--reg:100"], ["--reg:101"], ["--reg:"], ["--reg:x"], ["--reg:%"], ["--reg:nan"],
    ["--re:1"], ["-r:1.5"],
    # --yields:LIST / --no-yields:LIST (2026-10-06): the yield terms; years unless m. "--y:" is now
    # --yields: (it was "--yield takes no value"); "no-" before a value is new to the matcher.
    ["--yields:3m,10"], ["--y:2,10y"], ["--yield:30"], ["--yields:1m,6m,1,7,20"], ["--yields:10,3m,10"], ["--Y:10Y"],
    ["--yields:4"], ["--yields:3x"], ["--yields:"], ["--yields:,"], ["--yields:12m"], ["--yields: 10 , 2 "],
    ["--no-yields:30"], ["--no-y:30y,3m"], ["--no-yield:30"], ["--NO-Y:2"], ["--no-y:7"], ["--no-y:3m,2,5,10,30"],
    ["--no-y:"], ["--no-y:x"], ["--yields:7,1m,10", "--no-y:10"], ["--yields:7", "--no-y:7"],
    ["--no-y:30", "--yields:10"], ["--y:10", "--no-yield"], ["--y:10", "-c"], ["--y:10", "--gdp"],
    ["--no-debt:fp"], ["--no-dim:1100"], ["--no-reg:2"], ["--no-s:2001"], ["--yieldsx:10"],
    # --policy: a flag, off by default, adds its curve; at least "po", so "-p" stays per capita.
    ["--policy"], ["--po"], ["--pol"], ["--policy-rates"], ["--policyrates"], ["policy"], ["--no-policy"], ["--no-po"],
    ["--policy", "--no-policy"], ["--policy", "--gdp"], ["--policy", "-c", "--no-yield"], ["--policy", "-p"],
    ["--po:1"], ["--policyx"], ["--no-p"], ["--p"], ["--pe"],
    # --mortgages[:LIST]: a flag and its terms (turning it on); at least "mo", so "--m:" stays ambiguous.
    ["--mortgages"], ["--mo"], ["--mort"], ["mortgage"], ["--no-mortgages"], ["--no-mo"], ["--mo:30"], ["--mort:15,30"],
    ["--mortgages:5,3,1,5v"], ["--mo:30y,5"], ["--mo:5V"], ["--mo:5,5"], ["--mo:20"], ["--mo:x"], ["--mo:"], ["--mo:,"],
    ["--mo:30", "--no-mo"], ["--no-mo", "--mo:30"], ["--mo", "--mo:1"], ["--mo:1", "--gdp"], ["--mo:5", "-c", "--no-yield"],
    ["--m"], ["--m:5"], ["--mo:1", "--min:1"], ["--mortgagesx"], ["--no-mo:30"],
    # --spreads[:LIST]: a flag and its pairs (turning it on); at least "sp", so "--s:" stays the start.
    ["--spreads"], ["--sp"], ["--spr"], ["spreads"], ["--no-spreads"], ["--no-sp"], ["--sp:10-2"], ["--spreads:10y-3m,30-10"],
    ["--sp:2-10"], ["--sp:10-2,10y-2y"], ["--sp:10-10"], ["--sp:10"], ["--sp:10-2-1"], ["--sp:10-4"], ["--sp:"], ["--sp:-"],
    ["--sp:10-2", "--no-sp"], ["--sp", "--no-yield"], ["--sp:7-1m", "-u"], ["--sp", "--gdp"], ["--s"], ["--s:2001", "--sp"],
    ["--spreadsx"], ["--no-sp:10-2"], ["--po", "--mo", "--sp"],
    # Yield-axis bounds may be negative since 2026-10-06 (spreads invert), above -100.
    ["--min:-0.5"], ["--min:-1", "--max:2"], ["--min:-100"], ["--max:-1"], ["--min:1", "--max:-1"],
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
