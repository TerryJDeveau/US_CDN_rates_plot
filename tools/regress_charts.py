#!/usr/bin/env python3
"""Render the whole regression set of charts headless and compare it with an earlier run.

Every case in ``CASES`` goes through ``tools/verify_charts.run`` (``plt.show``
replaced by a PNG writer), all in this one process, into ``OUTDIR/NAME/chart_N.png``.
``--e:2026-09-01`` is put first on every command line so live data cannot drift
between runs; a case's own ``--e:`` still wins, as the last occurrence always does.

Downloads are cached on disk (``--cache``, default ``out/download_cache``), one
pickle per request, so a "before" and an "after" run see byte-identical data and
only the first run touches the network. Delete the folder to fetch fresh data.

--cur is on by default. With the end pinned in the past, the only newer values
it adds are the U.S. Treasury's daily federal debt after FRED's last quarter
(cached like the rest); intraday quotes are never fetched for such a window,
and nothing is projected to today. ``--extra=--no-cur`` renders every case
without it. The quotes and projections need a live run ending today.

Usage (from the project root)::

    python tools/regress_charts.py out/before --commit HEAD     # the last commit, working tree untouched
    <make a change>
    python tools/regress_charts.py out/after --compare out/before

    python tools/regress_charts.py out/x --only default,c_r     # a few cases
    python tools/regress_charts.py --list                       # the case names

``--commit REV`` exports that revision with ``git archive`` into ``out/tree_REV``
and renders from it; ``--root DIR`` renders from any other tree. With
``--compare`` each chart is reported IDENTICAL, DIFFER or MISSING, and the exit
code is 1 unless all are identical.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_END = "--e:2026-09-01"

# Name -> command-line arguments. Short and long spans, single series, small and
# large canvases, the history before 1961, and each right-axis measure.
CASES: dict[str, list[str]] = {
    "default": [],
    "u2001": ["-u", "-s:2001"],
    "u2001i": ["-u", "-s:2001", "--interest"],
    "c_gd_1990": ["-c", "--gdp", "--debt", "-s:1990"],
    "u_y2019": ["-u", "--yield", "-s:2019"],
    "c2015": ["-c", "-s:2015"],
    "u1980_95": ["-u", "-s:1980", "--e:1995"],
    "u2007_10": ["-u", "-s:2007", "--e:2010"],
    "u1985_90": ["-u", "-s:1985", "--e:1990"],
    "c1100": ["-c", "-dim:1100", "-s:2005"],
    "c1100sq": ["-c", "-dim:1100x1100", "-s:1966"],
    "u1400": ["-u", "-dim:1400x1000", "-s:2001"],
    "u800": ["-u", "-dim:800x800", "-s:2001"],
    "c4096": ["-c", "-dim:4096x3072", "-s:2015"],
    "c1867": ["-c", "-s:1867"],
    "c1926": ["-c", "-s:1926"],
    "c1900_40": ["-c", "-s:1900", "--e:1940"],
    "c1945_85": ["-c", "-s:1945", "--e:1985"],
    "c_r": ["-c", "-r"],
    "u_r": ["-u", "-r"],
    "c_r1926": ["-c", "-r", "-s:1926"],
    "c_r1867_di": ["-c", "-r", "--debt", "--interest", "-s:1867"],
    "u_r_lim": ["-u", "-r", "--t:200%", "--b:1%"],
    "c_r1100": ["-c", "-r", "-dim:1100", "-s:2005"],
    "c_p": ["-c", "-p"],
    "u_p": ["-u", "-p"],
    "c_p1867": ["-c", "-p", "-s:1867"],
    "c_p_lim": ["-c", "-p", "--t:200k", "--b:1k", "-s:1990"],
    # Debt and interest by level of government (--debt:LETTERS / --interest:LETTERS).
    "c_lv_fn": ["-c", "--d:fn", "-i"],
    "c_lv_fpm1926": ["-c", "--debt:fpm", "--interest", "-s:1926"],
    "c_lv_all1990": ["-c", "--d:fnpm", "--gdp", "--i", "--yield", "-s:1990"],
    "c_lv_r": ["-c", "-r", "--d:fpm", "-i"],
    "c_lv_p1867": ["-c", "-p", "--d:f", "-s:1867"],
    "u_lv_fn": ["-u", "--d:fn", "-i"],
    "u_lv_psm": ["-u", "--i:psm", "-d"],
    "u_lv_fpm1955": ["-u", "--i:fpm", "-s:1955"],
    "u_lv_d_fsm": ["-u", "--d:fsm", "-i", "-s:1990"],
    "u_lv_r": ["-u", "-r", "--i:fnpm"],
    "u_lv_dpsm1950": ["-u", "--d:psm", "-s:1950"],
    "u_lv_r_all1970": ["-u", "-r", "--d:fnpm", "-i", "-s:1970"],
    # -l: last values at the line ends (the date axis widens to fit them).
    "l_default": ["-l"],
    "c_l1867": ["-c", "-l", "-s:1867"],
    "c_l1900_40": ["-c", "-l", "-s:1900", "--e:1940"],
    "u_l_y2019": ["-u", "-l", "--yield", "-s:2019"],
    "u_l_r": ["-u", "-l", "-r"],
    "c_l_p": ["-c", "-l", "-p"],
    "c_l_lv": ["-c", "-l", "--d:fpm", "-i", "-s:1990"],
    "u_l800": ["-u", "-l", "-dim:800x800", "-s:2001"],
    "u_l_top": ["-u", "-l", "--t:20t", "-s:2001"],
    # A flat yield curve on a long axis: the yield labels crowd (smaller type, leader lines).
    "u_l2019": ["-u", "-l", "--e:2019-09-01"],
    "c_l2019": ["-c", "-l", "--e:2019-09-01"],
}


def install_disk_cache(http_module, cache_dir: Path) -> bool:
    """Replace the program's in-memory download cache with one pickle per request in ``cache_dir``.

    Returns False (and changes nothing) for a tree older than the cache, which
    then downloads live on every run.
    """
    if not hasattr(http_module, "_cached"):
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)

    def disk_cached(key, download):
        path = cache_dir / (hashlib.sha1(repr(key).encode()).hexdigest() + ".pkl")
        if path.exists():
            return pickle.loads(path.read_bytes())
        result = download()
        path.write_bytes(pickle.dumps(result))
        return result

    http_module._cached = disk_cached
    return True


def export_commit(revision: str) -> Path:
    """Export ``revision`` with ``git archive`` into ``out/tree_REV`` (reused if already there)."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", revision], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    target = ROOT / "out" / f"tree_{sha}"
    if not (target / "ratesplot").exists():
        archive = subprocess.run(["git", "archive", "--format=tar", sha], cwd=ROOT, capture_output=True, check=True).stdout
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter="data")
    return target


def sha256(path: Path) -> bytes:
    return hashlib.sha256(path.read_bytes()).digest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=Path, nargs="?", help="directory to write NAME/chart_N.png into")
    parser.add_argument("--compare", type=Path, help="directory of an earlier run to hash against")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--commit", help="render from this git revision (exported, working tree untouched)")
    source.add_argument("--root", type=Path, help="render from this project tree")
    parser.add_argument("--only", help="comma-separated case names")
    parser.add_argument(
        "--extra", default="", help="tokens added to every case, e.g. --extra=--no-cur to prove a default-on option neutral"
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "out" / "download_cache", help="download cache folder")
    parser.add_argument("--verbose", action="store_true", help="show the program's own output")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, case in CASES.items():
            print(f"{name:12} {' '.join(case)}")
        return 0
    if args.outdir is None:
        parser.error("OUTDIR is required")
    cases = CASES if not args.only else {name: CASES[name] for name in args.only.split(",")}

    tree = export_commit(args.commit) if args.commit else (args.root.resolve() if args.root else ROOT)
    sys.path[:0] = [str(tree / "tools"), str(tree)]
    verify_charts = importlib.import_module("verify_charts")
    http = importlib.import_module("ratesplot.http")
    if not install_disk_cache(http, args.cache.resolve()):
        print(f"[regress] {tree} has no download cache; downloading live")
    print(f"[regress] rendering {len(cases)} case(s) from {tree}")

    all_same = True
    for name, case in cases.items():
        output = io.StringIO()
        with contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(output):
            written = verify_charts.run(args.outdir / name, [PINNED_END, *case, *args.extra.split()])
        status = []
        if args.compare is not None:
            for path in written:
                other = args.compare / name / path.name
                verdict = "MISSING" if not other.exists() else "IDENTICAL" if sha256(other) == sha256(path) else "DIFFER"
                status.append(f"{path.name}:{verdict}")
                all_same &= verdict == "IDENTICAL"
        print(f"{name:12} {' '.join(case):40} {len(written)} chart(s) {' '.join(status)}")
        warnings = [line.strip() for line in output.getvalue().splitlines() if line.strip().startswith("Warning:") and "requested start" not in line]
        for line in warnings:
            print(f"    {line}")
    if args.compare is not None:
        print("[regress] ALL IDENTICAL" if all_same else "[regress] SOME DIFFER")
        return 0 if all_same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
