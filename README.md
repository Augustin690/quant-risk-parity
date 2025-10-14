# Quant Risk Parity (ERC) — Target Vol, Costs, OOS Evaluation

This repo implements a risk parity (equal risk contribution) portfolio with target volatility and basic trading costs.

## Quickstart
```bash
pip install -e .
qrp fetch
qrp run --start 2016-01-01 --end 2025-09-01
qrp report
```

Next steps:
    - implement 60/40 baseline done
    - use longer-term lookback window for risk estimation - to be done
        - need to download 10 years of data prior to investment start
    - fix the plot for rolling vol by asset class 
    