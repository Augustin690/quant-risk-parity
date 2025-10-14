from __future__ import annotations
from pandas_datareader import data as pdr
from typing import Callable, Iterable, Optional
import pandas as pd
from pathlib import Path
from . import data as D
from .paths import DATA_DIR
import typer

## DATA_DIR centralized in qrp.paths

# Baseline configuration constants
BASELINE_CONFIG = {
    "equity_ticker": "SPY",
    "bond_ticker": "IEF", 
    "equity_weight": 0.6,
    "bond_weight": 0.4,
    "rebalance_freq": "M"
}

def force_fetch_and_cache(
    fetch_func: Callable[..., pd.DataFrame],
    cache_file: str,
    start: str,
    end: str,
    fetch_args: Optional[Iterable] = None,
) -> Path:
    if fetch_args is None:
        fetch_args = []
    data = fetch_func(start, end, *fetch_args)
    path = D.cache_prices(data, cache_file)
    typer.echo(f"Saved data to: {path}")
    return path

def load_or_fetch_and_cache(
    fetch_func: Callable[..., pd.DataFrame],
    cache_file: str,
    start: str,
    end: str,
    fetch_args: Optional[Iterable] = None,
) -> pd.DataFrame:
    if fetch_args is None:
        fetch_args = []
    path = DATA_DIR / cache_file
    if path.exists():
        typer.echo(f"Loading cached data from {path}")
        return D.load_prices(cache_file)
    typer.echo(f"Fetching new data for {cache_file}")
    data = fetch_func(start, end, *fetch_args)
    D.cache_prices(data, cache_file)
    typer.echo(f"Saved to {path}")
    return data

def fetch_prices(start: str, end: str, tickers: list[str], n_attempts: int = 5) -> pd.DataFrame:
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

def fetch_baseline_data(start: str, end: str) -> pd.DataFrame:
    """
    Fetch baseline portfolio data (SPY and IEF) for 60/40 allocation benchmark.
    
    Args:
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with SPY and IEF price data
        
    Raises:
        ValueError: If baseline tickers cannot be fetched or data is invalid
    """
    baseline_tickers = [BASELINE_CONFIG["equity_ticker"], BASELINE_CONFIG["bond_ticker"]]
    
    try:
        # Validate date inputs
        pd.to_datetime(start)
        pd.to_datetime(end)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD format: {e}")
    
    if pd.to_datetime(start) >= pd.to_datetime(end):
        raise ValueError("Start date must be before end date")
    
    try:
        print(f"Fetching baseline data for tickers: {baseline_tickers}")
        baseline_data = fetch_prices(start, end, baseline_tickers)
        
        # Validate that we have data for both baseline tickers
        missing_tickers = [ticker for ticker in baseline_tickers if ticker not in baseline_data.columns]
        if missing_tickers:
            raise ValueError(f"Failed to fetch data for baseline tickers: {missing_tickers}")
        
        # Validate data quality - check for sufficient data points
        min_data_points = 30  # Minimum 30 data points for meaningful analysis
        for ticker in baseline_tickers:
            valid_data_points = baseline_data[ticker].dropna().shape[0]
            if valid_data_points < min_data_points:
                raise ValueError(f"Insufficient data for {ticker}: only {valid_data_points} valid data points (minimum {min_data_points} required)")
        
        # Validate that data covers the requested date range reasonably
        data_start = baseline_data.index.min()
        data_end = baseline_data.index.max()
        requested_start = pd.to_datetime(start)
        requested_end = pd.to_datetime(end)
        
        # Allow some tolerance for weekends/holidays
        if data_start > requested_start + pd.Timedelta(days=7):
            print(f"Warning: Baseline data starts later than requested ({data_start} vs {requested_start})")
        
        if data_end < requested_end - pd.Timedelta(days=7):
            print(f"Warning: Baseline data ends earlier than requested ({data_end} vs {requested_end})")
        
        print(f"Successfully fetched baseline data: {baseline_data.shape[0]} rows, {baseline_data.shape[1]} tickers")
        print(f"Date range: {data_start.strftime('%Y-%m-%d')} to {data_end.strftime('%Y-%m-%d')}")

        # make sure data is sorted by date
        baseline_data = baseline_data.sort_index()
        
        return baseline_data
        
    except Exception as e:
        if "Failed to fetch data for baseline tickers" in str(e) or "Insufficient data" in str(e):
            # Re-raise validation errors as-is
            raise
        else:
            # Wrap other errors with context
            raise ValueError(f"Failed to fetch baseline data: {e}") from e
