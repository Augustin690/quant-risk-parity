from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(how="all").ffill()
    return df

def cache_prices(df: pd.DataFrame, name: str = "prices.parquet") -> Path:
    path = DATA_DIR / name
    df.to_parquet(path)
    return path

def load_prices(name: str = "prices.parquet") -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_parquet(path)
