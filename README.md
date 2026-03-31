# Quant Risk Parity

A risk parity portfolio construction and backtesting framework implementing both the **ERC (Equal Risk Contribution)** and **S&P 3-step** methodologies with target volatility, realistic cost modeling, and robustness testing.

## Features

- **Two weight methods**: ERC iterative optimizer and S&P 3-step inverse-vol method
- **Target volatility**: Dynamic leverage to hit 10% (configurable) annualized vol
- **Realistic costs**: Transaction costs, slippage, and leverage financing costs
- **Two universes**: Original 8-ETF and expanded 14-ETF (mapping to paper's 26 futures)
- **9-panel dashboard**: Equity curve, drawdowns, rolling Sharpe, weights, risk contribution, turnover, returns distribution, rolling vol
- **IS/OOS evaluation**: In-sample / out-of-sample performance split
- **Robustness testing**: Regime stress tests, parameter sensitivity, bootstrap CIs (in analysis/)

## Quickstart

```bash
pip install -e .
qrp fetch --flag all
qrp run
qrp report
```

## Weight Methods

### ERC (Equal Risk Contribution)
Iterative optimizer that finds weights where each asset contributes equally to portfolio risk. Applied via the covariance matrix (EWMA or rolling window).

### S&P 3-Step (from the paper)
Based on "Indexing Risk Parity Strategies" (S&P Dow Jones Indices, 2018):
1. Inverse-vol weight each instrument to target vol
2. Asset-class multiplier to equalize risk across equity, fixed income, commodities
3. Portfolio multiplier to hit overall target volatility

## Asset Universes

| Universe | Tickers | Description |
|---|---|---|
| Original (8) | SPY, IEFA, EEM, IEF, TLT, LQD, DBC, GLD | Default |
| Expanded (14) | SPY, FEZ, EWJ, IEI, IEF, TLT, BWX, USO, UNG, UGA, DBC, GLD, SLV, DBA | Maps to paper's 26 futures |

## Project Structure

```
src/qrp/
├── __init__.py
├── paths.py          # Centralized file paths and constants
├── config.py         # Pydantic configuration with defaults
├── data.py           # Data fetching (Stooq) and caching (parquet)
├── weights.py        # ERC optimizer + S&P 3-step + risk contributions
├── backtest.py       # Portfolio simulation with cost modeling
├── metrics.py        # Sharpe, Sortino, Calmar, max drawdown, IS/OOS
├── visualize.py      # 9-panel dashboard generator
└── cli.py            # Typer CLI (fetch, run, report)

analysis/             # Standalone research scripts from paper replication
├── risk_parity_backtest_v2.py      # Cost modeling + variant comparison
├── risk_parity_v3_robustness.py    # Regime stress tests + sensitivity + bootstrap
├── risk_parity_v5_expanded.py      # 14-ETF expanded universe
├── risk_parity_pie_charts.py       # Risk contribution visualization
└── risk_parity_paper_period.py     # Paper-period direct comparison

tests/
├── __init__.py
└── test_core.py      # Unit tests for weights, metrics
```

## References

- Liu, B. et al. (2018). *Indexing Risk Parity Strategies*. S&P Dow Jones Indices.
- Dalio, R. et al. (2015). *Our Thoughts about Risk Parity and All Weather*. Bridgewater.
- Hurst, B. et al. (2010). *Understanding Risk Parity*. AQR Capital Management.
