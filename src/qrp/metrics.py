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

def summarize(port_rets: pd.Series, split_date: str="2022-01-01") -> pd.DataFrame:
    out = []
    segs = {
        "IS": port_rets[port_rets.index < split_date],
        "OOS": port_rets[port_rets.index >= split_date],
    }
    for k, s in segs.items():
        if s.empty:
            out.append((k, 0, 0, 0, 0))
        else:
            out.append((k, sharpe(s), s.mean()*252, s.std(ddof=1)*(252**0.5), max_drawdown(s)))
    return pd.DataFrame(out, columns=["Segment","Sharpe","AnnReturn","AnnVol","MaxDD"]).set_index("Segment")
