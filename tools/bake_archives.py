#!/usr/bin/env python3
"""Bake the historical data into the code: rewrite the generated blocks of the archive data modules.

Downloads the terminated and historical sources (see ``ratesplot/bake.py``),
extracts the rows the charts need and writes them as Python literals between
the BEGIN/END markers of

* ``ratesplot/cdn_archive_data.py``: the Canadian history (StatCan archived
  tables, *Historical Statistics of Canada*, Bank of Canada term bands);
* ``ratesplot/us_archive_data.py``: U.S. state and local government debt apart
  (Census Bureau). Needs ``pyodbc`` and the Microsoft Access ODBC driver, to
  read the Census historical database, and ``xlrd`` for one old ``.xls``
  table (both ``pip install``-able; only this bake uses them);
* ``ratesplot/uk_archive_data.py``: the UK history (the Bank of England's
  "A millennium of macroeconomic data", no longer updated). Needs
  ``openpyxl``. ``--uk-workbook FILE`` reads a copy already downloaded
  instead of fetching the 27.5 MB workbook.
* ``ratesplot/de_archive_data.py``: Germany's 10-year before 1972 (the
  Bundesbank's frozen yield on public debt securities, from 1956).
  ``--de-csv FILE`` reads a download already saved (tools/de_fixtures).

Nothing is written for a country unless every one of its sources is extracted
and the new module compiles.

This is a maintenance step, not an option of the program (it was
``--bake-archives`` until 2026-09-27): run it when an extraction changes or a
source is added (the Census publishes a new fiscal year each summer), check
the charts, and commit the regenerated module with the change that needed it.
Normal runs only read the modules.

Usage (from the project root)::

    python tools/bake_archives.py            # every nation
    python tools/bake_archives.py --only us  # or cdn, uk, de
    python tools/bake_archives.py --only uk --uk-workbook a-millennium-of-macroeconomic-data-for-the-uk.xlsx
    python tools/bake_archives.py --only de --de-csv tools/de_fixtures/raw/bbk01_WU0004.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot.bake import bake_canadian_archives, bake_german_archives, bake_uk_archives, bake_us_archives  # noqa: E402

BAKES = {"cdn": bake_canadian_archives, "us": bake_us_archives, "uk": bake_uk_archives, "de": bake_german_archives}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(BAKES), help="bake one nation only")
    parser.add_argument("--uk-workbook", type=Path, help="the UK millennium workbook already on disk (not downloaded)")
    parser.add_argument("--de-csv", type=Path, help="the Bundesbank's WU0004 download already on disk (not downloaded)")
    args = parser.parse_args(argv)
    for key, bake in BAKES.items():
        if args.only in (None, key):
            if key == "uk":
                bake(workbook_path=args.uk_workbook)
            elif key == "de":
                bake(saved_csv=args.de_csv)
            else:
                bake()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
