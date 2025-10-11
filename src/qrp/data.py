from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def _generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 7,
) -> pd.DataFrame:
    """Create a deterministic price history when remote data is unavailable."""

    index = pd.date_range(start=start, end=end, freq="B")
    if len(index) == 0:
        raise ValueError("Synthetic data requires a valid date range")

    rng = np.random.default_rng(seed)

    # Base drifts encourage modest growth while keeping series correlated.
    base_trend = rng.normal(0.00015, 0.00005, size=len(tickers))
    base_vol = rng.uniform(0.008, 0.015, size=len(tickers))

    shocks = rng.normal(size=(len(index), len(tickers)))
    returns = base_trend + shocks * base_vol

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    df = pd.DataFrame(prices, index=index, columns=tickers)
    return df.round(4)


def fetch_prices(tickers: list[str], start: str, end: str, n_attempts: int = 3) -> pd.DataFrame:
    import time
    import random
    
    # Download each ticker individually with delays
    dfs = []
    failed = []
    for i, ticker in enumerate(tickers):
        for attempt in range(1, n_attempts + 1):
            try:
                print(
                    f"Downloading {ticker} ({i+1}/{len(tickers)})... "
                    f"(attempt {attempt}/{n_attempts})"
                )
                df = pdr.DataReader(ticker, "stooq", start=start, end=end)["Close"]
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                df.columns = [ticker]
                dfs.append(df)

                if i < len(tickers) - 1:
                    time.sleep(random.uniform(1, 3))
                print(
                    f"Downloaded {ticker} ({i+1}/{len(tickers)})... "
                    f"(attempt {attempt}/{n_attempts})"
                )
                break
            except Exception as e:
                print(f"Failed to download {ticker} on attempt {attempt}: {e}")
                if attempt == n_attempts:
                    failed.append(ticker)
                    break
                time.sleep(random.uniform(2, 5))

    if not dfs:
        print("Falling back to synthetic data for all tickers.")
        return _generate_synthetic_prices(tickers, start, end)

    result = pd.concat(dfs, axis=1)
    result = result.dropna(how="all").ffill()

    if failed:
        print(
            "Generating synthetic series for tickers without remote data: "
            + ", ".join(failed)
        )
        synth = _generate_synthetic_prices(failed, start, end)
        result = result.join(synth, how="outer")

    return result.sort_index()

def cache_prices(df: pd.DataFrame, name: str = "prices.parquet") -> Path:
    path = DATA_DIR / name
    df.to_parquet(path)
    return path

def load_prices(name: str = "prices.parquet") -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_parquet(path)
