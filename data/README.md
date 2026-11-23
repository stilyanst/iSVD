# Data

`prices_etf.csv` is the ETF price panel of Section 5 of the paper. It is not part of
this repository: the prices come from Yahoo Finance (a third party) and are not
redistributed. Rebuild it with the script in `experiments/`.

## Contents

- 61 columns: `date` (trading day, `YYYY-MM-DD`) and the adjusted close price of 60 US-listed
  ETFs (equity indices and styles, regions, GICS sectors, rates and credit, commodities and real
  assets). The tickers and their order are listed in `experiments/prepare_etf_data.py`.
- Download window 2010-01-01 to 2024-12-31; only days on which all 60 prices exist are kept.
  This leaves 1644 trading days, 2018-06-19 to 2024-12-30 (the youngest ETF, XLC, starts in
  June 2018).
- The experiments turn the prices into 1643 daily log-returns and center each asset
  (`experiments/finance.py`, `load_price_data_and_returns`).

## Rebuild

```bash
pip install -e ".[experiments,data]"     # yfinance is only needed here
python experiments/prepare_etf_data.py   # writes data/prices_etf.csv
```

The file used for the paper has SHA-256
`2b0fb4786f0f4b61a99a970f76452046edc9455d55ac065d7b6ab35ee4f67c58`.
Yahoo Finance revises adjusted closes after dividends and splits, so a file downloaded later can
differ slightly from it, and the Section 5 numbers computed from it can differ in the last digits.
