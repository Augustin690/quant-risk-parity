from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple
from scipy import stats

# Asset class mapping
ASSET_CLASSES = {
    'Equity': ['SPY', 'IEFA', 'EEM'],
    'Fixed Income': ['IEF', 'TLT', 'LQD'],
    'Commodities': ['DBC', 'GLD']
}

# Color palette for asset classes
CLASS_COLORS = {
    'Equity': '#1f77b4',      # Blue
    'Fixed Income': '#2ca02c', # Green
    'Commodities': '#ff7f0e'   # Orange
}

def get_asset_class(ticker: str) -> str:
    """Map ticker to asset class."""
    for class_name, tickers in ASSET_CLASSES.items():
        if ticker in tickers:
            return class_name
    return 'Other'

def aggregate_by_class(weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weights by asset class."""
    class_weights = pd.DataFrame(index=weights.index)
    
    for class_name, tickers in ASSET_CLASSES.items():
        class_tickers = [t for t in tickers if t in weights.columns]
        if class_tickers:
            class_weights[class_name] = weights[class_tickers].sum(axis=1)
    
    return class_weights

def compute_risk_contribution(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate risk contribution by asset class over time."""
    # Align data
    common_dates = weights.index.intersection(returns.index)
    weights_aligned = weights.loc[common_dates]
    returns_aligned = returns.loc[common_dates]
    
    risk_contrib = pd.DataFrame(index=common_dates)
    
    # Calculate rolling covariance (252-day window)
    window = min(252, len(returns_aligned) // 4)  # Use 1/4 of data or 252 days
    
    for class_name, tickers in ASSET_CLASSES.items():
        class_tickers = [t for t in tickers if t in weights.columns and t in returns.columns]
        if not class_tickers:
            continue
            
        class_risk_contrib = pd.Series(0.0, index=common_dates)
        
        for i in range(window, len(common_dates)):
            # Get data for this window
            start_idx = i - window
            end_idx = i + 1
            
            w = weights_aligned.iloc[i][class_tickers].values
            ret_window = returns_aligned.iloc[start_idx:end_idx][class_tickers]
            
            if len(ret_window) < 2:
                continue
                
            # Calculate covariance matrix
            cov_matrix = ret_window.cov().values
            
            # Portfolio weights for this class only
            class_weights = weights_aligned.iloc[i][class_tickers].values
            
            # Skip if all weights are zero
            if np.sum(np.abs(class_weights)) == 0:
                continue
            
            # Calculate marginal contribution to variance for this class
            class_var = np.dot(class_weights, np.dot(cov_matrix, class_weights))
            
            if class_var > 0:
                # Risk contribution for this class
                class_contrib = np.sum(class_weights * np.dot(cov_matrix, class_weights))
                class_risk_contrib.iloc[i] = class_contrib / class_var * 100
        
        risk_contrib[class_name] = class_risk_contrib
    
    return risk_contrib

def plot_equity_curve(returns: pd.Series, split_date: str, ax: plt.Axes) -> None:
    """Plot equity curve with IS/OOS split."""
    equity = (1 + returns).cumprod()
    
    # Plot equity curve
    ax.plot(equity.index, equity.values, linewidth=2, color='black', label='Portfolio')
    
    # Add split line
    split_dt = pd.to_datetime(split_date)
    if split_dt in equity.index:
        ax.axvline(x=split_dt, color='red', linestyle='--', alpha=0.7, label='IS/OOS Split')
        
        # Shade regions
        ax.axvspan(equity.index[0], split_dt, alpha=0.1, color='blue', label='In-Sample')
        ax.axvspan(split_dt, equity.index[-1], alpha=0.1, color='red', label='Out-of-Sample')
    
    ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_drawdowns(returns: pd.Series, ax: plt.Axes) -> None:
    """Plot drawdown chart."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100
    
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    
    # Highlight max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.axhline(y=max_dd_val, color='black', linestyle='--', alpha=0.5)
    ax.annotate(f'Max DD: {max_dd_val:.1f}%', 
                xy=(max_dd_idx, max_dd_val), xytext=(10, 10),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('Drawdowns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_rolling_sharpe(returns: pd.Series, ax: plt.Axes, window: int = 252) -> None:
    """Plot rolling Sharpe ratio."""
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='purple')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
    
    ax.set_title(f'Rolling Sharpe ({window}d)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_individual_weights(weights: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot individual asset weights as stacked area chart."""
    # Use distinct colors for each asset
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights.columns)))
    
    ax.stackplot(weights.index, *[weights[col] for col in weights.columns],
                 labels=weights.columns, colors=colors, alpha=0.8)
    
    ax.set_title('Portfolio Weights by Asset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_class_weights(weights: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot asset class weights as stacked area chart."""
    class_weights = aggregate_by_class(weights)
      # make the bars a bit thicker
    class_weights = class_weights.mul(100)
    ordered_columns = sorted(class_weights.columns, key=lambda c: list(ASSET_CLASSES.keys()).index(c))
    colors = [CLASS_COLORS[col] for col in ordered_columns]
    
    # make sure it is stacked

    # Use fill_between with step='post' for thicker, bar-like stacked areas
    y_bottom = np.zeros(len(class_weights.index))
    for i, col in enumerate(ordered_columns):
        y_top = y_bottom + class_weights[col]
        ax.fill_between(class_weights.index, y_bottom, y_top, step='post', color=colors[i], alpha=0.8, label=col)
        y_bottom = y_top
    
    ax.set_title('Portfolio Weights by Asset Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_risk_contribution(weights: pd.DataFrame, returns: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot risk contribution by asset class."""
    risk_contrib = compute_risk_contribution(weights, returns)
    
    # Handle case where risk_contrib is empty (no asset returns available)
    if risk_contrib.empty or len(risk_contrib.columns) == 0:
        # throw error
        raise ValueError("Risk contribution is empty")

    
    colors = [CLASS_COLORS[col] for col in risk_contrib.columns]
    
    ax.stackplot(risk_contrib.index, *[risk_contrib[col] for col in risk_contrib.columns],
                 labels=risk_contrib.columns, colors=colors, alpha=0.8)
    
    ax.set_title('Risk Contribution by Asset Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Contribution (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_turnover(turnover: pd.Series, ax: plt.Axes) -> None:
    """Plot turnover over time."""
    # Filter to non-zero turnover (rebalance dates)
    turnover_nonzero = turnover[turnover > 0]
    
    if len(turnover_nonzero) > 0:
        ax.bar(turnover_nonzero.index, turnover_nonzero.values, 
               alpha=0.7, color='steelblue', width=20)
        
        # Add average line
        avg_turnover = turnover_nonzero.mean()
        ax.axhline(y=avg_turnover, color='red', linestyle='--', 
                  label=f'Avg: {avg_turnover:.1%}')
        ax.legend(fontsize=9)
    
    ax.set_title('Portfolio Turnover', fontsize=12, fontweight='bold')
    ax.set_ylabel('Turnover')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def plot_returns_distribution(returns: pd.Series, ax: plt.Axes) -> None:
    """Plot returns distribution with normal overlay."""
    # Remove NaN values
    clean_returns = returns.dropna()
    
    # Histogram
    ax.hist(clean_returns, bins=50, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5)
    
    # Normal distribution overlay
    mu, sigma = clean_returns.mean(), clean_returns.std()
    x = np.linspace(clean_returns.min(), clean_returns.max(), 100)
    normal_dist = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, normal_dist, 'r-', linewidth=2, label='Normal')
    
    # Statistics
    skewness = stats.skew(clean_returns)
    kurtosis = stats.kurtosis(clean_returns)
    
    stats_text = f'Mean: {mu:.3f}\nStd: {sigma:.3f}\nSkew: {skewness:.2f}\nKurt: {kurtosis:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

def plot_rolling_volatility_by_class(weights: pd.DataFrame, returns: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot rolling volatility contribution by asset class."""
    class_weights = aggregate_by_class(weights)
    window = min(252, len(returns) // 4)
    
    # Calculate rolling volatility for each class
    rolling_vol = pd.DataFrame(index=returns.index)
    
    for class_name, tickers in ASSET_CLASSES.items():
        class_tickers = [t for t in tickers if t in returns.columns]
        if class_tickers:
            class_returns = returns[class_tickers].mean(axis=1)  # Equal weight within class
            rolling_vol[class_name] = class_returns.rolling(window).std() * np.sqrt(252)
    
    # Plot
    colors = [CLASS_COLORS[col] for col in rolling_vol.columns]
    for i, col in enumerate(rolling_vol.columns):
        ax.plot(rolling_vol.index, rolling_vol[col], 
               label=col, color=colors[i], linewidth=2)
    
    ax.set_title(f'Rolling Volatility by Asset Class ({window}d)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Volatility')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

def create_dashboard(weights: pd.DataFrame, returns: pd.Series, turnover: pd.Series, 
                    split_date: str = "2022-01-01") -> plt.Figure:
    """Create comprehensive dashboard with 9 panels."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Try to load individual asset returns for risk contribution calculation
    asset_returns = None
    try:
        from pathlib import Path
        data_path = Path("data/prices.parquet")
        if data_path.exists():
            prices = pd.read_parquet(data_path)
            # Calculate returns from prices
            asset_returns = prices.pct_change().dropna()
    except Exception:
        # If we can't load asset returns, we'll handle it in the plot function
        pass
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Risk Parity Strategy Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot 1: Equity Curve
    plot_equity_curve(returns, split_date, axes_flat[0])
    
    # Plot 2: Drawdowns
    plot_drawdowns(returns, axes_flat[1])
    
    # Plot 3: Rolling Sharpe
    plot_rolling_sharpe(returns, ax=axes_flat[2])
    
    # Plot 4: Individual Asset Weights
    plot_individual_weights(weights, axes_flat[3])
    
    # Plot 5: Asset Class Weights
    plot_class_weights(weights, axes_flat[4])
    
    # Plot 6: Risk Contribution by Asset Class

    plot_risk_contribution(weights, asset_returns, axes_flat[5])
    
    # Plot 7: Turnover
    plot_turnover(turnover, axes_flat[6])
    
    # Plot 8: Returns Distribution
    plot_returns_distribution(returns, axes_flat[7])
    
    # Plot 9: Rolling Volatility by Asset Class
    plot_rolling_volatility_by_class(weights, returns.to_frame(), axes_flat[8])
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig
