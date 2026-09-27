#!/usr/bin/env python3
"""Bake the historical data into the code: rewrite the generated block of the archive data module.

Downloads the terminated and historical sources (see ``ratesplot/bake.py``),
extracts the rows the charts need and writes them as Python literals into
``ratesplot/cdn_archive_data.py``, between its BEGIN/END markers. Nothing is
written unless every source is extracted and the new module compiles.

This is a maintenance step, not an option of the program (it was
``--bake-archives`` until 2026-09-27): run it when an extraction changes or a
source is added, check the charts, and commit the regenerated module with the
change that needed it. Normal runs only read the module.

Usage (from the project root)::

    python tools/bake_archives.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot.bake import bake_canadian_archives  # noqa: E402


def main() -> int:
    bake_canadian_archives()
    return 0


if __name__ == "__main__":
    sys.exit(main())
