# Design Document

## Overview

The 60/40 portfolio baseline feature will integrate seamlessly with the existing quantitative risk parity system to provide a traditional allocation benchmark. The implementation will follow the established patterns in the codebase, creating dedicated functions for baseline data fetching, portfolio construction, and performance comparison.

The baseline will allocate 60% to equity indices (SPY for US large-cap exposure) and 40% to bond indices (IEF for intermediate-term Treasury exposure), representing a classic institutional portfolio allocation that serves as an industry-standard benchmark.

## Architecture

### Integration Points

The baseline functionality will integrate with existing modules:

- **Data Module (`data.py`)**: New `fetch_baseline_data()` function following existing data fetching patterns
- **Backtest Module (`backtest.py`)**: New `run_baseline()` function that creates static 60/40 allocation
- **Visualize Module (`visualize.py`)**: Enhanced plotting functions to include baseline comparison
- **Metrics Module (`metrics.py`)**: Baseline metrics calculation using existing performance functions
- **CLI Module (`cli.py`)**: Enhanced commands to include baseline in reports

### Data Flow

```
User Request → CLI Command → Baseline Data Fetch → 60/40 Allocation → Performance Calculation → Visualization with Comparison
```

## Components and Interfaces

### 1. Baseline Data Fetching (`data.py`)

**Function**: `fetch_baseline_data(start: str, end: str) -> pd.DataFrame`

- **Purpose**: Fetch equity and bond index data for baseline calculation
- **Tickers**: 
  - SPY (SPDR S&P 500 ETF) - 60% allocation for broad US equity exposure
  - IEF (iShares 7-10 Year Treasury Bond ETF) - 40% allocation for intermediate-term bond exposure
- **Integration**: Uses existing `fetch_prices()` infrastructure with error handling and caching
- **Output**: DataFrame with SPY and IEF price data aligned to same date range as main strategy

### 2. Baseline Portfolio Construction (`backtest.py`)

**Function**: `run_baseline(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]`

- **Purpose**: Create 60/40 static allocation with periodic rebalancing
- **Logic**:
  - Static weights: SPY=0.6, IEF=0.4
  - Returns weights history and portfolio returns
- **Integration**: Follows same return signature as `run_backtest()` for consistency

### 3. Enhanced Visualization (`visualize.py`)

**Enhanced Functions**:

- `plot_equity_curve()`: Add baseline comparison line
- `create_dashboard()`: Include baseline in relevant charts
- New function: `plot_strategy_comparison()`: Side-by-side performance metrics

**Baseline Styling**:
- Color: Gray (#808080) for neutral benchmark appearance
- Line style: Dashed line to distinguish from main strategy
- Label: "60/40 Baseline" for clear identification

### 4. Performance Comparison (`metrics.py`)

**Enhanced Function**: `summarize_with_baseline(strategy_returns: pd.Series, baseline_returns: pd.Series, split_date: str) -> pd.DataFrame`

- **Purpose**: Generate comparative performance summary
- **Metrics**: Sharpe ratio, annual return, volatility, max drawdown for both strategy and baseline
- **Output**: Combined DataFrame with strategy vs baseline comparison

## Data Models

### Baseline Configuration

```python
BASELINE_CONFIG = {
    "equity_ticker": "SPY",
    "bond_ticker": "IEF", 
    "equity_weight": 0.6,
    "bond_weight": 0.4,
    "rebalance_freq": "M"
}
```

### Return Data Structure

The baseline will return data structures consistent with the main strategy:

- **Weights**: DataFrame with columns ["SPY", "IEF"] and static allocations
- **Returns**: Series of daily portfolio returns
- **Performance**: Dictionary with standard metrics (Sharpe, volatility, etc.)

## Error Handling

### Data Fetching Errors

- **Missing Data**: Graceful degradation if baseline tickers unavailable
- **Date Misalignment**: Automatic alignment with strategy date range
- **Network Issues**: Retry logic consistent with existing data fetching

## Implementation Considerations

### Ticker Selection Rationale

- **SPY**: Most liquid and representative US equity ETF, widely used as benchmark
- **IEF**: Intermediate-term Treasury exposure balances duration risk and yield
- **Alternative Options**: Could support VTI (total stock market) or AGG (aggregate bonds) as configuration options

### Performance Attribution

The baseline will enable analysis of:
- **Risk-Adjusted Returns**: Strategy Sharpe vs baseline Sharpe
- **Volatility Comparison**: Risk parity volatility targeting vs static allocation
- **Drawdown Analysis**: Maximum drawdown comparison during stress periods
- **Rolling Performance**: Time-varying outperformance/underperformance analysis

### Extensibility

The design supports future enhancements:
- **Multiple Baselines**: Easy addition of other benchmark allocations (e.g., 70/30, 80/20)
- **International Exposure**: Addition of international equity/bond components
- **Sector Allocation**: More granular baseline construction with sector ETFs
- **Dynamic Allocation**: Time-varying baseline weights based on market conditions