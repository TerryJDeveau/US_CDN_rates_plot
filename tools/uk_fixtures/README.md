# UK sources for batch 3, measured on the laptop (Ambergris), 2026-10-06

Cloud sessions cannot reach the data sources, so the UK sources were measured from the laptop
before batch 3 moved to the cloud. Every file in this folder is a real response (or a faithful
extract of one) saved that day, for writing and testing parsers offline.

**Before the pull request is merged:** keep only the small files a test actually reads (move
them wherever the tests live) and delete the rest. The batch pull requests are squash-merged,
so files deleted on the branch never reach `main`'s history. The laptop runs the real fetches
(`tools/check_sources.py`) and the bake when it verifies the branch.

Spans below are what the sources served on 2026-10-06. "q" quarterly, "m" monthly, "d" daily.

## 1. Bank of England database (IADB): yields, Bank Rate, mortgage rates

`https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1975&Dateto=now&SeriesCodes=A,B,C&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N`

- **Needs a browser User-Agent**; without one it refuses.
- `CSVF=TN` (`boe_iadb_daily.csv`, `boe_iadb_monthly.csv`): header `DATE,CODE1,CODE2,...`, then one
  row per date, dates like `01 Sep 2026`, an empty cell where a series has no value that day. A
  monthly series is dated on the month's last day (`31 Aug 2026`).
- `CSVF=CT` (`boe_iadb_described.csv`): `SERIES,DESCRIPTION` lines, a blank line, then
  `DATE,SERIES,VALUE` rows. Use it to read the official titles.
- `Datefrom=01/Jan/1950` or `01/Jan/1960` answers `302` to `ErrorPage.asp?ei=905`, even for one
  series (`boe_iadb_error_905.txt`); `01/Jan/1975` works. An unknown series code answers
  `ei=1131`. Treat a redirect as a failure, not as data.

| code | what (official title, shortened) | span |
|---|---|---|
| IUDSNZC | 5-year nominal zero-coupon gilt yield | d 1982-01-04 → 2026-10-02 |
| IUDMNZC | 10-year nominal zero-coupon | d 1982-01-04 → 2026-10-02 |
| IUDLNZC | 20-year nominal zero-coupon | d 1992-02-11 → 2026-10-02 |
| IUDSNPY | 5-year nominal par yield | d 1993-12-01 → 2026-10-02 |
| IUDMNPY | 10-year nominal par yield (5.3714 on 2026-09-30) | d 1993-11-01 → 2026-10-02 |
| IUDLNPY | 20-year nominal par yield | d 2000-01-04 → 2026-10-02 |
| IUMAMNPY | 10-year par yield, monthly average | m 1993-12 → 2026-09 |
| IUDBEDR | Official Bank Rate (3.75 on 2026-10-05) | d 1975-01-02 → 2026-10-05 |
| IUDSOIA | SONIA (overnight) | d 1997-01-02 → 2026-10-02 |
| IUMAJNB | 3-month Treasury bill discount rate, month end | m 1975-01 → **2017-06 (ended)** |
| IUMBV34 | quoted rate, 2-year fixed mortgage, 75% LTV | m 1995-01 → 2026-08 |
| IUMBV37 | quoted rate, 3-year fixed mortgage, 75% LTV | m 1995-01 → 2026-08 |
| IUMBV42 | quoted rate, 5-year fixed mortgage, 75% LTV | m 1995-01 → 2026-08 |
| IUMTLMV | quoted revert-to rate (the standard variable rate, SVR) | m 1995-01 → 2026-08 |

The mortgage figures are the Bank's *quoted* rates (advertised for new loans at 75% loan to
value), so label them "quoted", as Canada's are labelled "posted". The par and zero-coupon
yields are the Bank's fitted curves, not one benchmark gilt's yield.

## 2. Bank of England fitted gilt curve (spreadsheets): every term, 1979 on

- Archive, 39 MB: `https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalddata.zip`,
  eight workbooks `GLC Nominal daily data_1979 to 1984.xlsx` … `_2016 to 2024.xlsx`,
  `_2025 to present.xlsx` (1.7 MB; updated 2026-10-02, data to 2026-09-30).
- This month, 220 KB: `https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/latest-yield-curve-data.zip`,
  holding `GLC Nominal daily data current month.xlsx` (saved here as
  `glc_nominal_daily_current_month.xlsx`: 2026-10-01 → 2026-10-05), plus real, inflation and
  OIS workbooks.
- Layout: sheets `info`, `1. fwds, short end`, `2. fwd curve`, `3. spot, short end`, `4. spot curve`.
  ⚠ Before 2005 the names carry "nominal": `4. nominal spot curve`, `3. nominal spot, short end`.
  Row 4 (0-based 3) is `years:` then the maturities; row 6 on, column A the date and then the
  yields in percent. The spot curve runs 0.5 to 25 years until 2015 and 0.5 to 40 from 2016; the
  short end runs in months, 1/12 to 5 years, written as floats (`0.24999999` is 3 months).
- Daily values in the archive, spot (zero-coupon) curve: 2, 5 and 10 years 12,068 days from
  1979-01-02; 20 years 10,337 (patchy until 1985); 25 years from 1992; **30 years only from
  2016-01-04**; 3 months (short end) from 1979-08-01 but only 4,209 days.
- `glc_nominal_spot_month_end.csv`: the month-end value of each term from the archive (573
  months), to show the shapes without the 39 MB download.

## 3. ONS time series (JSON): debt, interest, GDP, population

`https://www.ons.gov.uk<uri>/data`, for example
`https://www.ons.gov.uk/economy/governmentpublicsectorandtaxes/publicsectorfinance/timeseries/bkpx/pusf/data`.
JSON with `years`, `quarters`, `months` lists of `{"date", "value", "year", "quarter", "month",
"sourceDataset", "updateDate", ...}`; values are strings and `""` where there is none. Dates read
`1975 Q1`, `1975 MAR`, `1993`. `description` holds `title`, `unit`, `preUnit`, `nextRelease`.
The `ons_*.json` files here keep each list's first 4 and last 10 entries only. Units differ by
series: £m for most, **£bn for HF6W**, thousands for EBAQ. PUSF is monthly (next release
21 October 2026); UKEA and EDP are quarterly (next 22 December 2026).

| cdid / dataset | what | span |
|---|---|---|
| BKPX / pusf | General government consolidated gross debt (Maastricht), £m | q 1975Q1 → 2026Q2, m 1975-03 → 2026-08; £3,195,774m Aug 2026 |
| BKPW / pusf | Central government total gross debt, £m | q 1975Q1 → 2026Q2, m from 1975-03; £3,181,031m Aug 2026 |
| MDYT / edp | Local government gross debt, £m | q 1966Q1 → 2026Q2; £138,883m 2026Q2 |
| HF6W / pusf | Public sector net debt ex public sector banks (the UK's headline), **£bn** | q 1975Q1 → 2026Q2, m from 1975-03; £2,985.5bn Aug 2026 |
| HF6X / pusf | the same as % of GDP | q from 1975Q1, m from 1993-03; 93.8 Aug 2026 |
| A3PW / pusf | General government gross debt as % of GDP | q from 1997Q2 |
| NMFX / pusf | Central government net interest payable, £m | **q 1946Q1** → 2026Q2, m from 1997-04 |
| NUGW / pusf | Local government interest, £m | q 1946Q1 → 2026Q2 |
| NMYX / pusf | General government interest and dividends paid to the private sector and the rest of the world (consolidated), £m | q 1946Q1 → 2026Q2; £98,975m in 2025 |
| YBHA / ukea | GDP at market prices, current prices, seasonally adjusted, £m | q 1955Q1 → 2026Q2, years from 1948 |
| EBAQ / ukea | UK resident population, mid-year estimates interpolated quarterly, thousands | q 1955Q1 → 2026Q2 |
| UKPOP / pop | UK population mid-year estimate (annual) | 1971 → 2025 |

Also seen, not recommended: NRKB (general government D.41 interest, from 1987 only); RNEV and
RNTI (central and local interest, CP SA, from 1987; RNEV's £15bn in 2025 is not the debt
interest); JMEQ (public sector gross debt, from 1997 only); MDK2 (general government net debt).

## 4. The long history: the Bank of England's "A millennium of macroeconomic data" (v3.1)

`https://www.bankofengland.co.uk/-/media/boe/files/statistics/research-datasets/a-millennium-of-macroeconomic-data-for-the-uk.xlsx`,
27.5 MB, 109 sheets, last modified 2024-09-26, data to 2016 or 2017. It is no longer updated:
bake it, as Canada's terminated tables are baked. `millennium_columns.csv` lists the columns
extracted here (sheet, 1-based column, header, first and last period, first data row);
`millennium_values.csv` holds their values (sheet code, column, row, period, value).

| sheet, column | what | span |
|---|---|---|
| A29, 36 | National debt, spliced: consolidated debt incl. terminable annuities, PSND ex public sector banks from 2007/8, £m, end of financial year | 1690/91 → 2016/17 |
| A29, 41 / 42 | the same spliced measure / as % of nominal GDP | 1690/91 → 2015/16 / 1699/00 → 2015/16 |
| A29, 35 | Central government gross debt (BKPW less coin) | 1974/75 → 2015/16 |
| A9, 35 | UK nominal GDP at market prices, composite, £m, annual | 1700 → 2016 |
| A18, 2 / 3 / 4 | Population, thousands: Great Britain 1707-1801 / UK incl. Ireland 1801-1922 / UK 1922-2016 | |
| M10, 31 | Spliced 10-year gilt yield, monthly average | 1935 Jan → 2017 Mar |
| M10, 23 / 24 | 10- and 20-year redemption yields, monthly average | 1963 → 2002 / 1963 → 2010 |
| M10, 20 | Long-dated (20-year from 1957), month end | 1945 → 1969 |
| M10, 11 | Spliced consol yield, corrected for Goschen's conversion | 1753 Aug → 2017 Mar |
| M12, 17 | Spliced variable mortgage rate | 1939 Sep → 2017 Mar |
| M12, 15 | SVR (= IADB IUMTLMV) | 1995 → 2017 |
| M12, 18 | Bank Rate, monthly | 1929 → 2017 |
| D1, 3 | Bank Rate (1833-1972), Minimum Lending Rate (1972-81), ..., daily; extracted only on the 821 days it changed | 1833-01-01 → 2017-11-02 |

⚠ Debt is at the end of a financial year: 31 March in recent centuries, other months earlier
(column 2 of A29 names the month of each row). In M10, columns 26, 27 and 31 run to 2017 Mar and
then hold one more value in row 3178 with no period in columns A-B: skip it. Where a column
holds text notes in some rows, skip non-numbers.

## 5. Latest values (`--cur`): CNBC's quote feed

`cnbc_gb_quotes.json`, from the same feed `latest.py` reads (`config.CNBC_QUOTE_URL`): symbols
`GB2Y-GB`, `GB5Y-GB`, `GB10Y-GB` (5.3971% at 19:23 UTC), `GB20Y-GB`, `GB30Y-GB`, named "British
N Year Gilt". ⚠ `GB3M-GB` is "Gilt 3 Month Repo Rate", not a bill yield: do not use it as a
3-month yield. ⚠ `last_time` offsets differ between symbols (`+0000` and `-0400`): parse the
offset. On 2026-10-06 the database reached 2026-10-02 and the current-month workbook 2026-10-05, so
only the quotes reach today. No UK market-debt series for the projection to today was looked for (the Debt
Management Office's gilts outstanding would be the candidate).

## 6. What is not here yet

- A 3-month yield since the Treasury bill series ended in 2017: the fitted curve's 3 months
  (sparse early) or SONIA, labelled for what it is.
- A daily 2-year before 1979 or a 30-year before 2016 (the fitted curve has neither).
- A UK market-debt figure for `--cur` (above), and whether the "levels" make sense for the UK:
  central and local only (the devolved governments are inside central government in the UK
  accounts), and **BKPW + MDYT exceed BKPX** because general government debt is consolidated
  (debt one level holds of another is netted out). The bake's "levels add up" check needs a
  rule for that.
- FRED's UK series answered from the laptop: IRLTLT01GBM156N (10-year, monthly, OECD) 1960-01 →
  2026-08; IR3TIB01GBM156N (3-month interbank) 1957 → 2026-01; INTDSRGBM193N (discount rate) to
  2013 only.

## 7. Suggestions from the measuring session (judgement calls for the building session)

- Debt: BKPX (general government gross) is the counterpart of Canada's StatCan general
  government gross debt; HF6W (net) is the UK's own headline, so it is worth offering too.
- Yields: the fitted spot curve gives 2, 5, 10, 20 and 30 years on one basis. IADB serves 5, 10
  and 20 cheaply and daily; 2 and 30 need the workbooks (bake the archive, fetch "2025 to present"
  and "current month" live, or bake to the last full month).
- History before 1975-79: the millennium dataset, baked; the 10-year from 1935, debt and GDP
  from 1700, Bank Rate from 1833, mortgages from 1939.
- Interest: NMFX + NUGW by level, NMYX for the consolidated total; summed over four quarters as
  the other nations' interest is.
