from __future__ import annotations
import numpy as np
import pandas as pd
from .weights import erc_weights, target_leverage
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
