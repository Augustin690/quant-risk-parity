from pydantic import BaseModel

class Config(BaseModel):
    tickers: list[str] = ["SPY","IEFA","EEM","IEF","TLT","LQD","DBC","GLD"]
    start: str = "2016-01-01"
    end: str = "2025-09-01"
    rebalance: str = "M"
    target_vol_annual: float = 0.10
    ewma_halflife_days: int = 60
    cost_bps_per_trade: float = 2.0
    slippage_bps_per_turnover: float = 5.0
