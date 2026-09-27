#!/usr/bin/env python3
"""Check the transcribed Bank of Canada yield history against the source PDFs' own text.

``ratesplot/cdn_hist_yields.py`` holds five monthly tables transcribed by hand
from Bank of Canada PDFs. This downloads each PDF (cached in ``out/pdf_cache``),
extracts its text layer with ``pdftotext -raw`` and compares every transcribed
month with the printed value. Missing and differing months are listed; the
exit code is 1 if any differ.

A year the PDF prints twice (the long-bond table prints 1936 in both its
1919-1936 and its 1936-1948 block, across a change of series) matches if
either printing does; the report says which block the transcription follows.

``pdftotext`` comes with Git for Windows (``C:\\Program Files\\Git\\mingw64\\bin``,
the xpdf build); ``-raw`` keeps each printed row on one line, where ``-layout``
scrambles this table's first block.

Usage (from the project root)::

    python tools/check_hist_yields.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratesplot.cdn_hist_yields import build_cdn_hist_yields  # noqa: E402

PDF_URL = "https://www.bankofcanada.ca/wp-content/uploads/2010/09/selected_historical_{vector}.pdf"
# Chart column -> the Bank of Canada vector whose PDF it was transcribed from.
SOURCE_VECTORS = {
    "3-Month": "v122541",
    "2-Year": "v122538",
    "5-Year": "v122540",
    "10-Year": "v122486",
    "30-Year": "v122487",
}
# A printed row: a year, then up to twelve values (a dash or ** marks a missing month).
ROW = re.compile(r"^(1[89]\d\d|20\d\d)((?:\s+(?:-?\d+\.\d+|-|\*\*))+)\s*$")
TOLERANCE = 0.0005


def pdf_text(vector: str, cache: Path) -> str:
    """Return the PDF's text layer, downloading it into ``cache`` the first time."""
    path = cache / f"{vector}.pdf"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        response = requests.get(PDF_URL.format(vector=vector), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        path.write_bytes(response.content)
    result = subprocess.run(["pdftotext", "-raw", str(path), "-"], capture_output=True, check=True)
    return result.stdout.decode("latin-1")


def printed_rows(text: str) -> dict[int, list[list[float | None]]]:
    """Return year -> each printing of that year's twelve months (None where blank or missing)."""
    rows: dict[int, list[list[float | None]]] = {}
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        values = [float(v) if re.fullmatch(r"-?\d+\.\d+", v) else None for v in match.group(2).split()]
        # A year that starts part-way through is printed right-aligned.
        rows.setdefault(int(match.group(1)), []).append([None] * (12 - len(values)) + values)
    return rows


def main() -> int:
    transcribed = build_cdn_hist_yields()
    cache = ROOT / "out" / "pdf_cache"
    all_match = True
    for column, vector in SOURCE_VECTORS.items():
        series = transcribed[column]
        rows = printed_rows(pdf_text(vector, cache))
        compared, problems, notes = 0, [], []
        for year, printings in sorted(rows.items()):
            dates = [pd.Timestamp(year=year, month=month, day=1) for month in range(1, 13)]
            if dates[0] not in series.index:
                continue  # beyond the transcription (it ends 2000-12; live data after)
            ours = [series.loc[date] for date in dates]

            def agrees(printed: list[float | None]) -> bool:
                return all(
                    value is None or (not pd.isna(mine) and abs(mine - value) <= TOLERANCE)
                    for mine, value in zip(ours, printed)
                )

            matching = [index for index, printed in enumerate(printings) if agrees(printed)]
            compared += sum(value is not None for value in printings[0])
            if len(printings) > 1 and matching:
                notes.append(f"{year} is printed {len(printings)} times; the transcription follows printing {matching[0] + 1}")
            if not matching:
                for month, (mine, value) in enumerate(zip(ours, printings[-1]), start=1):
                    if value is not None and (pd.isna(mine) or abs(mine - value) > TOLERANCE):
                        problems.append(f"{year}-{month:02d}: transcribed {mine}, printed {value}")
        verdict = "OK" if not problems else f"{len(problems)} DIFFER"
        print(f"{column:8} {vector}: {compared} printed months compared, {verdict}")
        for line in notes + problems:
            print(f"    {line}")
        all_match &= not problems
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
