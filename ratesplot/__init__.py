"""ratesplot: Canadian/U.S./UK bond yields vs public debt, GDP and interest charts.

Package layout
--------------
config            constants, per-country metadata, HTTP session, PlotConfig; the
                  nations (Nation, NATIONS) and the choices each may have its own of
frames            small DataFrame helpers shared by the layers below
http              retrying GET helper, FRED CSV reader, BoC CSV parser, session download cache
console           printed output routed by thread (the window's log; the trial pass discarded)
axes              y-axis configuration, limit padding, currency and percent formatters
legend            legend auto-placement, title/subtitle/layout finishing
occupancy         every drawn line sampled into a grid of the axes, for finding free room
endlabels         -l: each line's last value at its end; widens the date axis to fit
regression        --reg: the right-axis curves fitted by straight pieces on their log
                  axis, each labelled with its growth in %/yr
joins             a nation's sources joined into one series (baked history to live
                  tables), and the series laid on the chart's dates
cdn_data          Bank of Canada / Statistics Canada fetch, splice and align
cdn_archive_data  baked historical CDN rows: debt, GDP, interest, population, yield
                  stand-ins (rewritten by tools/bake_archives.py)
cdn_hist_yields   transcribed Bank of Canada historical yield tables (1919-2000)
bake              the bake of the historical data (run by tools/bake_archives.py)
us_data           FRED fetch for U.S. yields and macro series
latest            --cur: newer values than the regular series (Treasury daily yields and
                  debt, intraday quotes)
us_archive_data   baked Census counts of U.S. state and local debt apart
                  (rewritten by tools/bake_archives.py)
uk_data           Bank of England database / ONS fetch for the UK's yields and macro
                  series (batch 3)
uk_archive_data   baked UK history from the Bank of England's "A millennium of
                  macroeconomic data" (rewritten by tools/bake_archives.py)
rates             the yield axis's other curves, each only when chosen: policy rates,
                  mortgage rates, yield spreads
measures          the right-axis measure: dollars, % of GDP (-r) or per person (-p)
plotting          line drawing, draw_country, build_figure (no pyplot), COUNTRIES, run_cdn / run_us / run_uk
options           the option table, value parsers/formatters, choices <-> PlotConfig, --help text
frontend          what the window and the web page share without a GUI toolkit: starting
                  choices, remembered settings, each nation's own controls, option
                  help, PNG bytes and file names
gui               the interactive window (default; --no-gui for plain matplotlib windows)
web               the web page (Streamlit; streamlit_app.py): the same charts in a browser
cli               argument parsing (driven by the options table) and main()

Dependency direction is strictly downward: cli -> gui -> {frontend, options,
plotting, http}; web -> {cli (its token matching), frontend, options,
plotting, http}; cli -> {options, plotting, http}; frontend -> {options,
config}; plotting -> {axes, legend, endlabels, regression, measures, rates,
cdn_data, us_data, uk_data, joins, latest} -> {frames, http} -> config;
{cdn_data, uk_data} -> joins; rates -> {cdn_data, uk_data, http};
plotting, gui, web -> console;
us_data -> latest; legend -> {axes, endlabels, regression, occupancy};
regression -> occupancy; options -> config. Data modules never import
plotting. Nothing imports web or gui, and cli imports gui only when the
window is opened, so --no-gui and the web page never load tkinter.
"""
