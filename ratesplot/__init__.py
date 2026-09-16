"""ratesplot: Canadian/U.S. bond yields vs public debt, GDP and interest charts.

Package layout
--------------
config            constants, COUNTRY_METADATA, HTTP sessions, PlotConfig
http              retrying GET helpers, FRED CSV reader, BoC CSV parser
axes              y-axis configuration, limit padding, currency formatter
legend            legend auto-placement, title/subtitle/layout finishing
cdn_data          Bank of Canada / Statistics Canada fetch, splice and align
cdn_archive_data  baked historical CDN debt/interest rows (rewritten by --bake-archives)
cdn_hist_yields   transcribed Bank of Canada historical yield tables (1919-2000)
bake              --bake-archives implementation
us_data           FRED fetch for U.S. yields and macro series
plotting          line drawing, plot_cdn / plot_us, run_cdn / run_us
cli               argument parsing and main()
"""
