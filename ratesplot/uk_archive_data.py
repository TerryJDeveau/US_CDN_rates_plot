"""Baked historical UK observations: the Bank of England's "A millennium of macroeconomic data".

The dataset (version 3.1, data to 2016 or 2017) is no longer updated, so its
columns the UK chart needs before the live sources begin are written here
once, as the Canadian terminated tables are: debt and GDP from about 1700,
the population from 1707, the 10-year gilt yield from 1935 and the 20-year
from 1963 (monthly averages), Bank Rate from 1833 (the days it changed),
and the variable mortgage rate from 1939. ``uk_data`` and ``rates`` join
them to the live series at run time; each list's comment names its source.

The block between the BEGIN/END markers is regenerated in place by
``tools/bake_archives.py --only uk`` (see :mod:`ratesplot.bake`). Do not
hand-edit it.
"""

# BEGIN AUTO-GENERATED UK ARCHIVE DATA
EMBEDDED_UK_DEBT_HISTORY = []
EMBEDDED_UK_GDP_HISTORY = []
EMBEDDED_UK_POPULATION_HISTORY = []
EMBEDDED_UK_10Y_HISTORY = []
EMBEDDED_UK_20Y_HISTORY = []
EMBEDDED_UK_BANK_RATE_HISTORY = []
EMBEDDED_UK_MORTGAGE_HISTORY = []
# END AUTO-GENERATED UK ARCHIVE DATA
