from __future__ import annotations
import numpy as np
import pandas as pd

def sharpe(series: pd.Series, periods_per_year: int=252) -> float:
    mu = series.mean() * periods_per_year
    sig = series.std(ddof=1) * (periods_per_year**0.5)
    return float(mu / sig) if sig > 0 else 0.0

def max_drawdown(series: pd.Series) -> float:
    equity = (1+series).cumprod()
    peak = equity.cummax()
    return float((equity/peak - 1.0).min())

def sortino(series: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio."""
    mu = series.mean() * periods_per_year
    downside = series[series < 0].std(ddof=1) * (periods_per_year ** 0.5)
    return float(mu / downside) if downside > 0 else 0.0

def calmar(series: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio: annualized return / abs(max drawdown)."""
    mu = series.mean() * periods_per_year
    mdd = abs(max_drawdown(series))
    return float(mu / mdd) if mdd > 0 else 0.0

def summarize(port_rets: pd.Series, split_date: str="2022-01-01") -> pd.DataFrame:
    out = []
    segs = {
        "IS": port_rets[port_rets.index < split_date],
        "OOS": port_rets[port_rets.index >= split_date],
    }
    for k, s in segs.items():
        if s.empty:
            out.append((k, 0, 0, 0, 0, 0, 0))
        else:
            out.append((k, sharpe(s), s.mean()*252, s.std(ddof=1)*(252**0.5), max_drawdown(s), sortino(s), calmar(s)))
    return pd.DataFrame(out, columns=["Segment","Sharpe","AnnReturn","AnnVol","MaxDD","Sortino","Calmar"]).set_index("Segment")

def summarize_with_baseline(strategy_returns: pd.Series, baseline_returns: pd.Series, split_date: str="2022-01-01") -> pd.DataFrame:
    """
    Generate comparative performance summary between strategy and baseline.
    
    Args:
        strategy_returns: Daily returns of the main strategy
        baseline_returns: Daily returns of the 60/40 baseline
        split_date: Date to split in-sample vs out-of-sample periods
        
    Returns:
        DataFrame with strategy vs baseline comparison showing performance metrics
        and comparative analysis (excess returns, Sharpe difference, etc.)
    """
    # Align returns to same date range
    common_dates = strategy_returns.index.intersection(baseline_returns.index)
    if common_dates.empty:
        raise ValueError("No overlapping dates between strategy and baseline returns")
    
    strategy_aligned = strategy_returns.loc[common_dates]
    baseline_aligned = baseline_returns.loc[common_dates]
    
    # Calculate excess returns
    excess_returns = strategy_aligned - baseline_aligned
    
    out = []
    segs = {
        "IS": (strategy_aligned[strategy_aligned.index < split_date], 
               baseline_aligned[baseline_aligned.index < split_date],
               excess_returns[excess_returns.index < split_date]),
        "OOS": (strategy_aligned[strategy_aligned.index >= split_date],
                baseline_aligned[baseline_aligned.index >= split_date], 
                excess_returns[excess_returns.index >= split_date]),
    }
    
    for k, (strat_seg, base_seg, excess_seg) in segs.items():
    
        # Strategy metrics
        strat_sharpe = sharpe(strat_seg)
        strat_ret = strat_seg.mean() * 252
        strat_vol = strat_seg.std(ddof=1) * (252**0.5)
        strat_dd = max_drawdown(strat_seg)
        strat_sortino = sortino(strat_seg)
        strat_calmar = calmar(strat_seg)

        # Baseline metrics
        base_sharpe = sharpe(base_seg)
        base_ret = base_seg.mean() * 252
        base_vol = base_seg.std(ddof=1) * (252**0.5)
        base_dd = max_drawdown(base_seg)
        base_sortino = sortino(base_seg)
        base_calmar = calmar(base_seg)

        # Comparative metrics
        excess_ret = excess_seg.mean() * 252
        sharpe_diff = strat_sharpe - base_sharpe
        excess_sharpe = sharpe(excess_seg)
        dd_diff = strat_dd - base_dd

        out.append((k, strat_sharpe, strat_ret, strat_vol, strat_dd, strat_sortino, strat_calmar,
                    base_sharpe, base_ret, base_vol, base_dd, base_sortino, base_calmar,
                    excess_ret, sharpe_diff, excess_sharpe))

    columns = [
        "Segment", "Strategy_Sharpe", "Strategy_AnnReturn", "Strategy_AnnVol", "Strategy_MaxDD",
        "Strategy_Sortino", "Strategy_Calmar",
        "Baseline_Sharpe", "Baseline_AnnReturn", "Baseline_AnnVol", "Baseline_MaxDD",
        "Baseline_Sortino", "Baseline_Calmar",
        "Excess_AnnReturn", "Sharpe_Difference", "Excess_Sharpe"
    ]
    
    return pd.DataFrame(out, columns=columns).set_index("Segment")
