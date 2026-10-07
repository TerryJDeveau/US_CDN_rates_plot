# ratesplot: bond yields and public debt

Government bond yields against public debt, TTM GDP and interest outlays, for Canada and the
United States, and when asked for the United Kingdom and Germany, on one two-axis chart per
country. The data come from their sources each time: Statistics Canada, the Bank of Canada, FRED,
the U.S. Treasury and the U.S. Census Bureau; the Bank of England and the Office for National
Statistics for the UK; the Deutsche Bundesbank, the ECB and Eurostat for Germany. The history
before the sources' own series (back to 1867 for Canadian federal debt, to about 1700 for the UK's
national debt and GDP, to 1956 for Germany's 10-year yield) is built into the code.

## Three ways to run it

- **The web page** (`ratesplot/web.py`): open <https://ratesplot.streamlit.app> and choose
  the charts in the panel on the left. Drag a box on a chart to zoom to it; buttons move the
  dates and undo a zoom; each date has a calendar; the size keeps its shape when you change
  one side. The page's address holds the chart's settings, so a chart can be bookmarked or
  sent.
  To run it on your own machine: `streamlit run streamlit_app.py`.
- **The desktop window** (`ratesplot/gui.py`): `python ratesplot.py`. It adds the
  mouse wheel and a right-drag pan, and remembers your last settings.
- **The command line**: `python ratesplot.py --no-gui [options]`, or any options with the
  window. `python ratesplot.py --help` lists them all; the web page shows the same list.

All three draw the same charts from the same code (`plotting.build_figure`).

## Setting up

Python 3.12 or later, then:

```
python -m pip install -r requirements.txt
```

## A few options

| | |
|---|---|
| `-c`, `-u`, `--nations:gb,de` | Canada or the United States only (Canada and the U.S. by default); the United Kingdom and Germany are named only by their codes, `--nations:gb`, `--nations:de` or with others `--nations:ca,us,gb,de` (`-c` and `-u` may join them) |
| `-r`, `-p` | debt and interest as % of GDP, or GDP, debt and interest per person |
| `--debt:fnpm` | debt and interest by level of government |
| `--yields:2,10`, `--no-yields:30` | the yield terms drawn (U.S. also 1m, 6m, 1, 7 and 20 years; the UK 5, 10 and 20; Germany 1, 2, 5, 7, 10, 20 and 30), or those dropped; a chart that has none of the terms named draws no yield, with a warning (`-c --yields:7,20`) |
| `--policy`, `--mortgages`, `--spreads` | policy rates, mortgage rates (`--mo:30,15`; the UK's `2f`, `3f`, `5f`, `svr`; Germany's fixation bands `1-5`, `5-10`, `over10`), yield spreads (`--sp:10-2`), on the yield axis |
| `--us:max:8`, `--cdn:yields:2,10` | a value for one country's chart only (its code in front: `ca` or `cdn`, `us`, `gb` or `uk`, `ger` or `deu`; Germany's `de` names it only in `--nations`, since `--de` is `--debt`); a country's own list may be `none` (`--cdn:yields:none`); in the window and on the page each country's terms and axis limits have their own panel |
| `-l` | each line's last value at its end |
| `--reg` | each right-axis curve fitted by straight pieces, labelled with its growth in %/yr |
| `-s:1990`, `-e:2020` | the date window (without `-s` it starts in 2000, or where the chosen curves all have data if later; the year is `DEFAULT_START_FLOOR` in `ratesplot/config.py`) |
| `--no-cur` | without the latest daily and intraday values |

Intraday yield quotes (from CNBC's quote feed) and the projection of each curve to today are
provisional, and drawn as such.

## Layout

`ratesplot/` is the program (its `__init__.py` has the module map); `tools/` holds the checks
used when it is changed (`regress_charts.py`, `verify_cli.py`, `verify_web.py`, `check_sources.py`)
and the bake of the historical data (`bake_archives.py`).
