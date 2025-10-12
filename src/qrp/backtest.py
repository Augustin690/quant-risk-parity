from __future__ import annotations
import numpy as np
import pandas as pd
from .weights import erc_weights, target_leverage
from .data import BASELINE_CONFIG
from typing import Tuple

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def rebalance_schedule(index: pd.DatetimeIndex, freq: str="M") -> pd.DatetimeIndex:
    return pd.date_range(index.min(), index.max(), freq=freq).intersection(index)

def run_backtest(
    prices: pd.DataFrame,
    rebalance: str="M",
    ewma_window: int=60,
    target_vol_annual: float=0.10,
    cost_bps_per_trade: float=2.0,
    slippage_bps_per_turnover: float=5.0,
    risk_estim_strat: str="ewma",
    ewma_halflife_days: int=60,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rets = compute_returns(prices)
    dates = rets.index
    cols = rets.columns
    n = len(cols)
    # fill with NaNs
    weights_hist = pd.DataFrame(np.nan, index=dates, columns=cols)
    lev_hist = pd.Series(np.nan, index=dates)
    turnover = pd.Series(np.nan, index=dates)

    if risk_estim_strat == "rolling":
        roll_cov = rets.rolling(ewma_window).cov().dropna()
    elif risk_estim_strat == "ewma":
        roll_cov = rets.ewm(halflife=ewma_halflife_days).cov().dropna()
    else:
        raise ValueError(f"Invalid risk estimation strategy: {risk_estim_strat}")

    prev_w = np.zeros(n)
    for t in rebalance_schedule(dates, rebalance):
        if t not in roll_cov.index.get_level_values(0):
            continue
        cov_t = roll_cov.loc[t].values.reshape(n, n)
        w = erc_weights(cov_t)
        lev = target_leverage(w, cov_t, target_vol_annual)
        weights_hist.loc[t] = w * lev
        lev_hist.loc[t] = lev
        turnover.loc[t] = np.abs(w * lev - prev_w).sum()
        prev_w = w * lev

    weights_hist = weights_hist.ffill().fillna(0.0)
    lev_hist = lev_hist.ffill().fillna(0.0)
    turnover = turnover.fillna(0.0)

    daily_cost = turnover * (cost_bps_per_trade + slippage_bps_per_turnover) / 1e4
    port_rets = (weights_hist.shift().fillna(0.0) * rets).sum(axis=1) - daily_cost
    return weights_hist, port_rets, turnover.to_frame("turnover")


def run_baseline(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run 60/40 baseline portfolio backtest with static allocation.
    
    Args:
        prices: DataFrame with SPY and IEF price data
        
    Returns:
        Tuple of (weights_history, portfolio_returns)
        - weights_history: DataFrame with daily portfolio weights (constant 60/40)
        - portfolio_returns: Series of daily portfolio returns
        
    Raises:
        ValueError: If required baseline tickers are missing from prices DataFrame
    """
    # Validate input data
    equity_ticker = BASELINE_CONFIG["equity_ticker"]
    bond_ticker = BASELINE_CONFIG["bond_ticker"]
    required_tickers = [equity_ticker, bond_ticker]
    
    missing_tickers = [ticker for ticker in required_tickers if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError(f"Missing required baseline tickers in prices DataFrame: {missing_tickers}")
    
    if prices.empty:
        raise ValueError("Prices DataFrame is empty")
    
    # Calculate returns
    rets = compute_returns(prices[required_tickers])
    if rets.empty:
        raise ValueError("No valid return data available after computing returns")
    
    dates = rets.index
    
    # Static 60/40 allocation - no rebalancing needed
    equity_weight = BASELINE_CONFIG["equity_weight"]
    bond_weight = BASELINE_CONFIG["bond_weight"]
    
    # Create weights DataFrame with constant static allocation
    weights_hist = pd.DataFrame(index=dates, columns=required_tickers)
    weights_hist[equity_ticker] = equity_weight
    weights_hist[bond_ticker] = bond_weight
    
    # Calculate portfolio returns using static weights
    # Use shifted weights to avoid look-ahead bias (same pattern as main strategy)
    port_rets = (weights_hist.shift().fillna(0.0) * rets).sum(axis=1)
    
    # Handle any remaining NaN values
    port_rets = port_rets.fillna(0.0)

    # make sure both are sorted in chronological order
    port_rets = port_rets.sort_index()
    weights_hist = weights_hist.sort_index()

    return weights_hist, port_rets
