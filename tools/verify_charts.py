#!/usr/bin/env python3
"""Render the charts headless to PNG so a change can be checked by image hash.

``plt.show()`` blocks on a GUI window, so this harness switches to the Agg
backend, patches ``ratesplot.plotting.plt.show`` to save the current figure
instead, and runs the normal CLI. Pin the end date (``--e:YYYY-MM-DD``) so
live data does not drift between two runs you intend to compare.

Usage (from the project root)::

    python tools/verify_charts.py OUTDIR [--compare OTHERDIR] [-- CLI ARGS...]

    python tools/verify_charts.py out/before -- --e:2026-09-01
    <make a change>
    python tools/verify_charts.py out/after --compare out/before -- --e:2026-09-01

With ``--compare`` each ``chart_N.png`` is hashed against the same file in the
other directory and reported IDENTICAL or DIFFER; the exit code is 1 if any differ.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot import plotting  # noqa: E402
from ratesplot.cli import main  # noqa: E402


def run(outdir: Path, cli_args: list[str]) -> list[Path]:
    """Run the CLI with ``plt.show`` replaced by a PNG writer; return the files written."""
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save_instead_of_show(*_args, **_kwargs) -> None:
        path = outdir / f"chart_{len(written) + 1}.png"
        # Strip the matplotlib version from the PNG metadata so hashes stay comparable
        # across library upgrades that do not change the rendered pixels.
        plt.gcf().savefig(path, dpi=plt.gcf().dpi, metadata={"Software": None})
        plt.close("all")
        written.append(path)
        print(f"[verify] saved {path}")

    plotting.plt.show = save_instead_of_show
    main(cli_args)
    return written


def compare(written: list[Path], other: Path) -> bool:
    """Hash each written PNG against its namesake in ``other``; return True if all match."""
    all_same = True
    for path in written:
        counterpart = other / path.name
        if not counterpart.exists():
            print(f"{path.name}: MISSING in {other}")
            all_same = False
            continue
        same = hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(counterpart.read_bytes()).digest()
        print(f"{path.name}: {'IDENTICAL' if same else 'DIFFER'}")
        all_same &= same
    return all_same


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=Path, help="directory to write chart_N.png into")
    parser.add_argument("--compare", type=Path, help="directory holding a previous run to hash against")
    if "--" in argv:
        split = argv.index("--")
        own, cli = argv[:split], argv[split + 1 :]
    else:
        own, cli = argv, []
    return parser.parse_args(own), cli


if __name__ == "__main__":
    args, cli_args = parse_args(sys.argv[1:])
    files = run(args.outdir, cli_args)
    if args.compare is not None and not compare(files, args.compare):
        sys.exit(1)
