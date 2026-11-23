"""Download the ETF price panel of the paper's Sec. 5 and save it to data/prices_etf.csv.

Universe (60 tickers, liquid US-listed ETFs):

- Broad US equity / style: SPY, IVV, VTI, QQQ, DIA, IWM, IWN, IWO, VTV, VUG
- International / regional equity: VEA, VWO, EFA, EEM, EWJ, EWU, EWG, EWQ, EWL,
  EWA, EWC, EWH, EWY, EWZ, EWW, EWD
- US equity sectors (GICS): XLF, XLK, XLE, XLY, XLP, XLV, XLI, XLB, XLRE, XLU, XLC
- Fixed income / rates / credit: TLT, IEF, SHY, BND, AGG, LQD, HYG, JNK, TIP, EMB
- Commodities / real assets / thematics: GLD, SLV, GDX, GDXJ, USO, DBC, DBA, VNQ,
  IYR, XME, XOP, KRE, SMH

Daily adjusted close prices (adjusted for splits and dividends) from Yahoo
Finance via ``yfinance``, requested for 2010-01-01 to 2024-12-31. Days with a
missing price for any ticker are dropped, so the panel is a T x N matrix with
no NaNs: 1644 trading days, 2018-06-19 to 2024-12-30, for the file used in the
paper (see data/README.md).

Output columns: ``date`` (YYYY-MM-DD), then one column per ticker in the order
above. experiments/finance.py turns the prices into log-returns.

Usage::

    python experiments/prepare_etf_data.py [--out data/prices_etf.csv]
"""

import argparse
import os
from typing import List, Optional

import pandas as pd
import yfinance as yf

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prices_etf.csv")


def build_etf_prices_csv(
    tickers: Optional[List[str]] = None,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    outfile: str = DATA_PATH,
) -> None:
    """Download daily adjusted close prices and save the panel to CSV.

    Parameters
    ----------
    tickers : list of str, optional
        ETF tickers; default the 60-ETF universe of the module docstring.
    start, end : str
        Date range (YYYY-MM-DD) of the download.
    outfile : str
        Output CSV; default data/prices_etf.csv in the repository. Its folder
        is created if needed.
    """
    if tickers is None:
        tickers = [
            # --- Broad US equity / style ---
            "SPY",  # S&P 500
            "IVV",  # S&P 500 (iShares)
            "VTI",  # Total US market
            "QQQ",  # Nasdaq 100
            "DIA",  # Dow Jones
            "IWM",  # Russell 2000
            "IWN",  # Russell 2000 Value
            "IWO",  # Russell 2000 Growth
            "VTV",  # US Value
            "VUG",  # US Growth

            # --- International / regional equity ---
            "VEA",  # Developed ex-US
            "VWO",  # Emerging markets
            "EFA",  # Developed ex-US (iShares)
            "EEM",  # Emerging markets (iShares)
            "EWJ",  # Japan
            "EWU",  # UK
            "EWG",  # Germany
            "EWQ",  # France
            "EWL",  # Switzerland
            "EWA",  # Australia
            "EWC",  # Canada
            "EWH",  # Hong Kong
            "EWY",  # South Korea
            "EWZ",  # Brazil
            "EWW",  # Mexico
            "EWD",  # Sweden

            # --- US sector ETFs (GICS sectors) ---
            "XLF",  # Financials
            "XLK",  # Technology
            "XLE",  # Energy
            "XLY",  # Consumer Discretionary
            "XLP",  # Consumer Staples
            "XLV",  # Health Care
            "XLI",  # Industrials
            "XLB",  # Materials
            "XLRE", # Real Estate
            "XLU",  # Utilities
            "XLC",  # Communication Services

            # --- Fixed income / rates / credit ---
            "TLT",  # Long-term Treasuries
            "IEF",  # 7–10Y Treasuries
            "SHY",  # 1–3Y Treasuries
            "BND",  # US aggregate bond
            "AGG",  # US aggregate bond (iShares)
            "LQD",  # Investment-grade credit
            "HYG",  # High-yield credit
            "JNK",  # High-yield credit (SPDR)
            "TIP",  # TIPS
            "EMB",  # EM sovereign bonds (USD)

            # --- Commodities / real assets / thematics ---
            "GLD",  # Gold
            "SLV",  # Silver
            "GDX",  # Gold miners
            "GDXJ", # Junior gold miners
            "USO",  # WTI oil
            "DBC",  # Broad commodities
            "DBA",  # Agriculture
            "VNQ",  # US REITs
            "IYR",  # Real estate
            "XME",  # Metals & mining
            "XOP",  # Oil & gas exploration
            "KRE",  # Regional banks
            "SMH",  # Semiconductors
        ]

    print(f"[prepare_etf_data] Downloading {len(tickers)} ETFs from {start} to {end}")
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)

    # data has columns like ('Adj Close', 'SPY') in a MultiIndex from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        adj = data["Adj Close"].copy()
    else:
        # Some yfinance versions may already flatten columns; be defensive
        adj = data.copy()

    # Drop rows with any missing prices across the selected ETFs
    adj = adj.dropna(how="any")

    # Put date into a column and sort
    adj = adj.reset_index().rename(columns={"Date": "date"})
    adj = adj.sort_values("date").reset_index(drop=True)

    # Ensure we only keep the tickers we care about (and in that order)
    cols = ["date"] + [t for t in tickers if t in adj.columns]
    adj = adj[cols]

    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    adj.to_csv(outfile, index=False)
    print(f"[prepare_etf_data] Saved {outfile} with shape {adj.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the ETF price panel of Sec. 5.")
    parser.add_argument("--out", default=DATA_PATH, help="output CSV (default: data/prices_etf.csv)")
    args = parser.parse_args()
    build_etf_prices_csv(outfile=args.out)
