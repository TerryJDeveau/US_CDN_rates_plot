# German sources for batch 4, measured on the laptop (Ambergris), 2026-10-06/07

Cloud sessions cannot reach the data sources, so the German sources were fetched from the laptop
first, as the UK's were (`tools/uk_fixtures`). `raw/` holds each response **as served**. `raw/_index.tsv` lists each request kept: file, HTTP
status, bytes, content type, first and last data line, URL.

**Pruned 2026-10-07** (the build of batch 4) to the 33 responses `tools/check_de_parsers.py`
reads, as batch 3 did: it serves them to the program's own fetchers by URL and query, so each
must stay as served. The other 23 (the 3- and 15-year terms, the old database's other series,
the failed `UMR` keys, €STR, the ECB's other key rates and quarterly debt, the convergence
yield, the APRC, the ECB's GDP, and the failed ECB keys) and the FRED rows are in commit
`b90210e`, with this README as it was.

Fetched by a scratch script with a browser User-Agent, 8 at a time. Spans below are what the
sources served on 2026-10-06 evening (Toronto).

## Answered

### Bundesbank SDMX REST (`https://api.statistiken.bundesbank.de/rest/download/{flow}/{key}?format=csv&lang=en`)

- Layout: a header block (title, unit, last update), then `date,value,flag` rows; "." with
  "No value available" on days without a value (weekends included in daily series).
- `/rest/download/...` returns the whole series; `/rest/data/...?startPeriod=` a window. One
  `/rest/download` with a browser User-Agent once broke off (IncompleteRead) and worked on retry:
  use `http._with_retries`.
- **Svensson term structure of listed Federal securities** (`BBSIS.{D|M}.I.ZST.ZI.EUR.S1311.B.A604.{term}.R.A.A._Z._Z.A`,
  terms `R01XX` … `R30XX` = 1 … 30 years; `R00X5` (6 months) does not exist):
  - daily (`bbk_ts_d_*`): 1–20 years from **1997-08-07**, 30 years from 2000-08-01, to 2026-10-06
    (10y 3.52 %);
  - monthly (`bbk_ts_m_*`): 1, 2, 3, 5, 7, 10 years from **1972-09**; 15 and 20 from 1986-06;
    30 from 2000-01; to 2026-09 (10y 3.64 %).
- The old `BBK01` codes through `https://www.bundesbank.de/statistic-rmi/StatisticDownload?tsId=BBK01.{code}&its_csvFormat=en&its_fileFormat=csv&mode=its`
  answer only for some codes and **end 2020-04** (frozen; the SDMX keys replaced them):
  - `WU0004`: yields on debt securities outstanding, public debt securities, monthly from
    **1955-01** (first value later; see file);
  - `WU0017`: from 1955-08 (6.1 %);
  - `WZ9826`: 10-year term structure, monthly, 1972-09 (8.08 %) — the same as `bbk_ts_m_R10XX`;
  - `WX4260`: from 1990;
  - `SU0112` (discount rate), `SU0115` (Lombard), `SU0202`, `WT1010`: "not valid or not present".
  The Bundesbank discount rate 1948–1998 still needs its SDMX key (flow `BBIN1` or `BBIM1`);
  not found yet.

### ECB Data Portal (`https://data-api.ecb.europa.eu/service/data/{flow}/{key}?format=csvdata`)

- `ecb_mro_fixed` (MRO fixed rate), `ecb_mro_minbid` (minimum bid rate, 2000–2008),
  `ecb_deposit` (deposit facility), `ecb_marginal` (marginal lending): ECB key rates from 1999,
  on their change days.
- `ecb_estr`: €STR, daily from 2019-10.
- `ecb_irs_de_10y`: Germany's 10-year convergence yield, monthly.
- `ecb_gfs_q_debt` / `ecb_gfs_a_debt`: Germany's general government gross debt (Maastricht),
  quarterly and annual, national currency.
- `ecb_mna_q_gdp`: GDP at current prices, quarterly, not seasonally adjusted, € millions.
- `ecb_mir_house_*`: MFI interest rates on new housing loans to households, Germany, by initial
  rate fixation: `1_5` (over 1 and up to 5 years), `5_10`, `10plus`, `aprc` (annual percentage
  rate of charge); from 2003-01. `float` (floating and up to 1 year) answered 500: key wrong.
- Not found (404): `GFS ... D41` (interest; key wrong), `ENA ... POP`.

### Eurostat (`https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?...`)

JSON-stat 2.0. `est_ggdebt_q` (gov_10q_ggdebt, GD, all sectors and subsectors: S13 general
government, S1311 central, S1312 state (Länder), S1313 local, S1314 social security),
`est_ggnfa_q` (gov_10q_ggnfa, D41PAY interest payable, all sectors), `est_gdp_q` (namq_10_gdp,
current prices, NSA), `est_pop_q` (namq_10_pe, population, thousands), `est_edpt1_a`
(gov_10dd_edpt1, annual EDP table), `est_pjan_a` (demo_pjan, population on 1 January).

### CNBC quotes (`cnbc_de_quotes.json`)

The feed `latest.py` reads; needs the browser User-Agent (403 without). Symbols tried:
DE3M, DE6M, DE1Y, DE2Y, DE5Y, DE7Y, DE10Y, DE20Y, DE30Y (`-DE`). `DE3M-DE` answered `code 1`
(unknown); see the file for which others exist. `DE2Y-DE` "Germany 2 Year Bond", 3.1094 % at
2026-10-07 02:48 CEST.

## Not answered

- **FRED** timed out or reset with the browser User-Agent (the program fetches FRED without the
  session for this reason: `http.fetch_fred_csv`). To retry: IRLTLT01DEM156N (10y from 1956),
  INTDSRDEM193N (discount rate), IR3TIB01DEM156N, IRSTCI01DEM156N, CPMNACSCAB1GQDE (GDP),
  GGGDTADEA188N (debt % GDP), POPTOTDEA647NWDB, DEUPOPNDQ.

## Used by the program (batch 4, built 2026-10-07)

Yields `bbk_ts_{d,m}_*` (1 2 5 7 10 20 30 years) and `bbk01_WU0004` (baked, the 10-year before
1972-09); policy `ecb_mro_fixed` + `ecb_mro_minbid`; mortgages `ecb_mir_house_{1_5,5_10,10plus}`;
debt `ecb_gfs_a_debt` (year-ends 1991-1999) + `est_ggdebt_q`; interest `est_edpt1_a` (1995-2001)
+ `est_ggnfa_q`; GDP `est_gdp_q`; population `est_pjan_a` + `est_pop_q`; quotes `cnbc_de_quotes`.
`bbk01_WZ9826` shows the monthly term structure is end-of-month values; the error answers
(`bbk01_SU0112`, `bbk_ts_m_R00X5`, `ecb_mir_house_float`) are kept as answers to refuse.

## Still to find (on the laptop)

- The Bundesbank discount and Lombard rates, 1948–1998, as SDMX keys.
- German GDP, debt, interest and population **before 1991** (West Germany): Destatis or the
  Bundesbank's long series; candidates for a bake, like the UK's millennium dataset.
- Mortgage rates before 2003 (the Bundesbank's old interest rate statistics, to 2003-06).
