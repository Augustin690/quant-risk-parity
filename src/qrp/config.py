from pydantic import BaseModel

class Config(BaseModel):
    # Original 8-ETF universe
    tickers: list[str] = ["SPY","IEFA","EEM","IEF","TLT","LQD","DBC","GLD"]

    # Expanded 14-ETF universe (maps to paper's 26 futures)
    tickers_expanded: list[str] = [
        "SPY", "FEZ", "EWJ",           # Equity
        "IEI", "IEF", "TLT", "BWX",    # Fixed Income
        "USO", "UNG", "UGA", "DBC",     # Commodities - Energy
        "GLD", "SLV", "DBA",            # Commodities - Metals & Agriculture
    ]

    # Asset class mappings
    asset_classes: dict[str, list[str]] = {
        "Equity": ["SPY", "IEFA", "EEM"],
        "Fixed Income": ["IEF", "TLT", "LQD"],
        "Commodities": ["DBC", "GLD"],
    }
    asset_classes_expanded: dict[str, list[str]] = {
        "Equity": ["SPY", "FEZ", "EWJ"],
        "Fixed Income": ["IEI", "IEF", "TLT", "BWX"],
        "Commodities": ["USO", "UNG", "UGA", "DBC", "GLD", "SLV", "DBA"],
    }

    start: str = "2016-01-01"
    end: str = "2025-09-01"
    rebalance: str = "ME"
    target_vol_annual: float = 0.10
    ewma_halflife_days: int = 60
    rolling_window_days: int = 100
    risk_estim_strat: str = "ewma"
    cost_bps_per_trade: float = 2.0
    slippage_bps_per_turnover: float = 5.0
    leverage_cost_annual: float = 0.02  # 2% default financing cost

    # SP3-specific
    min_lookback_days: int = 252
    max_lookback_days: int = 252 * 15
    weight_method: str = "erc"  # "erc" or "sp3"
