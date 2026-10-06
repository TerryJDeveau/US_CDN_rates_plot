#!/usr/bin/env python3
"""Command-line entry point for the ratesplot package.

Run ``python ratesplot.py --help`` for options; the implementation lives in
the ``ratesplot/`` package beside this file. (It was ``US_CDN_rates_plot.py``
until batch 3, 2026-10-06; Python imports the package, not this file, for
``import ratesplot``: a package directory wins over a module of its name.)
"""

from ratesplot.cli import main

if __name__ == "__main__":
    main()
