"""ratesplot: Canadian/U.S. bond yields vs public debt, GDP and interest charts.

Package layout
--------------
config            constants, per-country metadata, HTTP session, PlotConfig
frames            small DataFrame helpers shared by the layers below
http              retrying GET helper, FRED CSV reader, BoC CSV parser, session download cache
axes              y-axis configuration, limit padding, currency formatter
legend            legend auto-placement, title/subtitle/layout finishing
cdn_data          Bank of Canada / Statistics Canada fetch, splice and align
cdn_archive_data  baked historical CDN debt/interest rows (rewritten by --bake-archives)
cdn_hist_yields   transcribed Bank of Canada historical yield tables (1919-2000)
bake              --bake-archives implementation
us_data           FRED fetch for U.S. yields and macro series
plotting          line drawing, draw_country, build_figure (no pyplot), COUNTRIES, run_cdn / run_us
options           the option table, value parsers/formatters, choices <-> PlotConfig, --help text
gui               the interactive window (default; --no-gui for plain matplotlib windows)
cli               argument parsing (driven by the options table) and main()

Dependency direction is strictly downward: cli -> gui -> {options, plotting,
http}; cli -> {options, plotting}; plotting -> {axes, legend, cdn_data,
us_data} -> {frames, http} -> config; options -> config. Data modules never
import plotting, and nothing imports cli or gui (cli imports gui only when the
window is opened, so --no-gui never loads tkinter).
"""
