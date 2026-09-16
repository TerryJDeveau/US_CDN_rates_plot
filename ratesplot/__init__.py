"""ratesplot: Canadian/U.S. bond yields vs public debt, GDP and interest charts.

Package layout
--------------
config            constants, per-country metadata, HTTP session, PlotConfig
frames            small DataFrame helpers shared by the layers below
http              retrying GET helper, FRED CSV reader, BoC CSV parser
axes              y-axis configuration, limit padding, currency formatter
legend            legend auto-placement, title/subtitle/layout finishing
cdn_data          Bank of Canada / Statistics Canada fetch, splice and align
cdn_archive_data  baked historical CDN debt/interest rows (rewritten by --bake-archives)
cdn_hist_yields   transcribed Bank of Canada historical yield tables (1919-2000)
bake              --bake-archives implementation
us_data           FRED fetch for U.S. yields and macro series
plotting          line drawing, plot_country, run_cdn / run_us
cli               argument parsing and main()

Dependency direction is strictly downward: cli -> plotting -> {axes, legend,
cdn_data, us_data} -> {frames, http} -> config. Data modules never import
plotting, and nothing imports cli.
"""
