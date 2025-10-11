from __future__ import annotations
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def fetch_prices(tickers: list[str], start: str, end: str, n_attempts: int = 5) -> pd.DataFrame:
    import time
    import random
    
    # Download each ticker individually with delays
    dfs = []
    for i, ticker in enumerate(tickers):
        for attempt in range(n_attempts):  # 1 initial try + 2 retries
            try:
                print(f"Downloading {ticker} ({i+1}/{len(tickers)})... (attempt {attempt+1}/3)")
                df = pdr.DataReader(ticker, "stooq", start=start, end=end)["Close"]
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                df.columns = [ticker]
                dfs.append(df)
                
                # Add delay between requests to avoid rate limiting
                if i < len(tickers) - 1:  # Don't delay after the last ticker
                    time.sleep(random.uniform(1, 3))  # Random delay between 1-3 seconds
                print(f"Downloaded {ticker} ({i+1}/{len(tickers)})... (attempt {attempt+1}/3)")
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Failed to download {ticker} on attempt {attempt+1}: {e}")
                if attempt == 2:
                    print(f"Giving up on {ticker} after 3 attempts.")
                    continue
                else:
                    time.sleep(random.uniform(2, 5))  # Wait a bit longer before retrying
    
    if not dfs:
        raise ValueError("No data could be downloaded")
    
    result = pd.concat(dfs, axis=1)
    return result.dropna(how="all").ffill()

def cache_prices(df: pd.DataFrame, name: str = "prices.parquet") -> Path:
    path = DATA_DIR / name
    df.to_parquet(path)
    return path

def load_prices(name: str = "prices.parquet") -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_parquet(path)
