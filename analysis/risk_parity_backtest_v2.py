"""
Risk Parity Strategy Backtest v2 — Enhanced
Based on: "Indexing Risk Parity Strategies" (S&P Dow Jones Indices, October 2018)

Enhancements over v1:
- Transaction costs (5 bps spread+slippage) and leverage financing costs (3M T-bill)
- SLV replaces USO for commodities
- Multiple lookback windows: expanding, 3yr rolling, 5yr rolling, EWMA
- Correlation-aware ERC optimization via scipy
- Multiple target volatility levels (10%, 12%, 15%)
- Comprehensive comparison charts and metrics
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_VOLS = [0.10, 0.12, 0.15]
DEFAULT_TV = 0.10
MIN_LOOKBACK_DAYS = 252      # 1 year minimum
MAX_LOOKBACK_DAYS = 252 * 15 # 15 years max
REBALANCE_FREQ = 'M'
ANNUALIZATION_FACTOR = np.sqrt(252)
START_DATE = '2007-01-01'
END_DATE = '2025-12-31'
TRANSACTION_COST_BPS = 5     # 5 bps per rebalance on turnover
FALLBACK_RF_RATE = 0.02      # 2% annual fallback for leverage cost

ASSET_CLASSES = {
    'Equity': ['SPY', 'EFA', 'EEM'],
    'Fixed Income': ['IEF', 'TLT', 'AGG'],
    'Commodities': ['GLD', 'DBC', 'SLV'],
}
ALL_TICKERS = [t for tickers in ASSET_CLASSES.values() for t in tickers]

# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_data():
    """Download price data for all tickers and ^IRX for risk-free rate."""
    print("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = data['Close'].dropna()
    returns = prices.pct_change().dropna()
    print(f"Data range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Total trading days: {len(returns)}")

    # Download 3-month T-bill rate (^IRX) for leverage financing cost
    print("Downloading ^IRX (3-month T-bill rate)...")
    try:
        irx_data = yf.download('^IRX', start=START_DATE, end=END_DATE, auto_adjust=True)
        if irx_data is not None and len(irx_data) > 0:
            irx_rate = irx_data['Close'].squeeze()
            irx_rate = irx_rate / 100.0  # Convert from percentage to decimal
            # Reindex to match returns, forward-fill missing values
            rf_daily = irx_rate.reindex(returns.index).ffill().bfill()
            rf_daily = rf_daily / 252.0  # Convert annual rate to daily
            print(f"^IRX data loaded: {rf_daily.notna().sum()} days")
        else:
            raise ValueError("Empty IRX data")
    except Exception as e:
        print(f"^IRX download failed ({e}), using fallback rate of {FALLBACK_RF_RATE*100:.0f}%")
        rf_daily = pd.Series(FALLBACK_RF_RATE / 252.0, index=returns.index)

    return prices, returns, rf_daily

# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY ESTIMATION METHODS
# ══════════════════════════════════════════════════════════════════════════════

def get_vol_expanding(returns_history, ticker, min_days=252, max_days=252*15):
    """Expanding window volatility — use all available history (clamped to min/max)."""
    n = len(returns_history)
    lb = max(min_days, min(n, max_days))
    return returns_history[ticker].iloc[-lb:].std() * ANNUALIZATION_FACTOR

def get_vol_rolling(returns_history, ticker, window_years):
    """Fixed rolling window volatility (3yr or 5yr)."""
    window_days = window_years * 252
    n = len(returns_history)
    if n < window_days:
        # Use all available data if less than window
        return returns_history[ticker].std() * ANNUALIZATION_FACTOR
    return returns_history[ticker].iloc[-window_days:].std() * ANNUALIZATION_FACTOR

def get_vol_ewma(returns_history, ticker, halflife=60):
    """EWMA volatility with given halflife."""
    ewma_vol = returns_history[ticker].ewm(halflife=halflife).std().iloc[-1] * ANNUALIZATION_FACTOR
    return ewma_vol

def get_cov_for_method(returns_history, vol_method='expanding'):
    """Get covariance matrix appropriate for the vol method."""
    if vol_method == 'expanding':
        n = len(returns_history)
        lb = max(MIN_LOOKBACK_DAYS, min(n, MAX_LOOKBACK_DAYS))
        return returns_history[ALL_TICKERS].iloc[-lb:].cov() * 252
    elif vol_method.startswith('rolling_'):
        years = int(vol_method.split('_')[1].replace('yr', ''))
        window_days = years * 252
        n = len(returns_history)
        if n < window_days:
            return returns_history[ALL_TICKERS].cov() * 252
        return returns_history[ALL_TICKERS].iloc[-window_days:].cov() * 252
    elif vol_method == 'ewma':
        # Use EWMA covariance
        return returns_history[ALL_TICKERS].ewm(halflife=60).cov().iloc[-len(ALL_TICKERS):] * 252
    else:
        return returns_history[ALL_TICKERS].cov() * 252

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT COMPUTATION: S&P 3-STEP METHOD
# ══════════════════════════════════════════════════════════════════════════════

def compute_sp3step_weights(returns_history, target_vol=DEFAULT_TV, vol_method='expanding'):
    """
    3-step S&P Risk Parity weight computation with configurable vol method.
    Step 1: Instrument-level inverse-vol weighting
    Step 2: Asset-class-level multiplier to equalize risk contribution
    Step 3: Portfolio-level multiplier to hit target volatility
    """
    n_days = len(returns_history)
    if n_days < MIN_LOOKBACK_DAYS:
        return None

    # Step 1: Compute instrument vols based on chosen method
    instrument_vols = {}
    for ticker in ALL_TICKERS:
        if vol_method == 'expanding':
            instrument_vols[ticker] = get_vol_expanding(returns_history, ticker)
        elif vol_method == 'rolling_3yr':
            instrument_vols[ticker] = get_vol_rolling(returns_history, ticker, 3)
        elif vol_method == 'rolling_5yr':
            instrument_vols[ticker] = get_vol_rolling(returns_history, ticker, 5)
        elif vol_method == 'ewma':
            instrument_vols[ticker] = get_vol_ewma(returns_history, ticker)
        else:
            instrument_vols[ticker] = get_vol_expanding(returns_history, ticker)

    # Inverse-vol raw weights
    raw_weights = {}
    for ticker in ALL_TICKERS:
        if instrument_vols[ticker] > 0:
            raw_weights[ticker] = target_vol / instrument_vols[ticker]
        else:
            raw_weights[ticker] = 0

    # Step 2: Asset-class-level multiplier
    asset_class_weights = {}
    for ac_name, tickers in ASSET_CLASSES.items():
        n_inst = len(tickers)
        ac_ticker_weights = {t: raw_weights[t] / n_inst for t in tickers}

        # Compute asset class sub-portfolio realized vol
        ac_returns = sum(ac_ticker_weights[t] * returns_history[t] for t in tickers)
        ac_vol = ac_returns.std() * ANNUALIZATION_FACTOR

        multiplier = target_vol / ac_vol if ac_vol > 0 else 1.0
        for t in tickers:
            asset_class_weights[t] = ac_ticker_weights[t] * multiplier

    # Step 3: Portfolio-level multiplier
    n_classes = len(ASSET_CLASSES)
    portfolio_weights = {t: asset_class_weights[t] / n_classes for t in ALL_TICKERS}

    port_returns = sum(portfolio_weights[t] * returns_history[t] for t in ALL_TICKERS)
    port_vol = port_returns.std() * ANNUALIZATION_FACTOR

    port_multiplier = target_vol / port_vol if port_vol > 0 else 1.0
    final_weights = {t: portfolio_weights[t] * port_multiplier for t in ALL_TICKERS}

    return final_weights

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT COMPUTATION: ERC OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def compute_erc_weights(returns_history, target_vol=DEFAULT_TV):
    """
    Equal Risk Contribution weights via scipy optimization.
    Minimize sum_i sum_j (RC_i - RC_j)^2 where RC_i = w_i * (Sigma @ w)_i
    Then scale to target volatility.
    """
    n_days = len(returns_history)
    if n_days < MIN_LOOKBACK_DAYS:
        return None

    cov = returns_history[ALL_TICKERS].cov().values * 252
    n = len(ALL_TICKERS)

    def erc_objective(w):
        sigma_w = cov @ w
        rc = w * sigma_w  # Risk contributions
        # Minimize sum of squared differences between all pairs
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += (rc[i] - rc[j]) ** 2
        return total

    # Constraints: weights sum to 1 (we'll scale after), all positive
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, None) for _ in range(n)]  # Minimum 1% per asset
    w0 = np.ones(n) / n  # Equal weight initial guess

    result = minimize(erc_objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    if not result.success:
        # Fallback to equal weight
        w_opt = np.ones(n) / n
    else:
        w_opt = result.x

    # Scale to target volatility
    port_vol = np.sqrt(w_opt @ cov @ w_opt)
    if port_vol > 0:
        scale = target_vol / port_vol
    else:
        scale = 1.0

    final_weights = {ALL_TICKERS[i]: w_opt[i] * scale for i in range(n)}
    return final_weights

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(returns, rf_daily, weight_func, label, apply_costs=True, **kwargs):
    """
    Generic backtest engine.
    weight_func: callable(returns_history, **kwargs) -> dict of weights or None
    Returns dict with daily returns series, weights history, leverage, cost tracking.
    """
    # Get monthly rebalance dates
    monthly_last_days = returns.groupby(returns.index.to_period('M')).apply(lambda x: x.index[-1])
    valid_dates = [d for d in monthly_last_days if returns.index.get_loc(d) >= MIN_LOOKBACK_DAYS]

    daily_rets = []
    weights_history = []
    leverage_history = []
    cum_txn_cost = 0.0
    cum_lev_cost = 0.0
    txn_cost_series = []
    lev_cost_series = []

    current_weights = None
    prev_weights = None
    rebal_idx = 0

    for i in range(MIN_LOOKBACK_DAYS, len(returns)):
        date = returns.index[i]

        # Check if rebalance
        if rebal_idx < len(valid_dates) and date >= valid_dates[rebal_idx]:
            history = returns.iloc[max(0, i - MAX_LOOKBACK_DAYS):i]
            new_weights = weight_func(history, **kwargs)

            if new_weights is not None:
                prev_weights = current_weights
                current_weights = new_weights

                # Transaction cost: 5 bps on turnover
                if apply_costs and prev_weights is not None:
                    turnover = sum(abs(current_weights.get(t, 0) - prev_weights.get(t, 0))
                                   for t in ALL_TICKERS)
                    txn_drag = turnover * TRANSACTION_COST_BPS / 10000.0
                    cum_txn_cost += txn_drag

                # Record leverage
                total_leverage = sum(abs(v) for v in current_weights.values())
                leverage_history.append({'date': date, 'leverage': total_leverage})

                # Record weights by asset class
                ac_w = {}
                for ac_name, tickers in ASSET_CLASSES.items():
                    ac_w[ac_name] = sum(current_weights[t] for t in tickers)
                weights_history.append({'date': date, **ac_w})

            rebal_idx += 1

        if current_weights is None:
            continue

        # Daily portfolio return
        day_ret = returns.iloc[i]
        port_ret = sum(current_weights[t] * day_ret[t] for t in ALL_TICKERS)

        # Leverage financing cost (daily)
        if apply_costs:
            total_lev = sum(abs(v) for v in current_weights.values())
            if total_lev > 1.0:
                rf_rate_today = rf_daily.iloc[i] if i < len(rf_daily) else FALLBACK_RF_RATE / 252
                daily_lev_cost = (total_lev - 1.0) * rf_rate_today
                port_ret -= daily_lev_cost
                cum_lev_cost += daily_lev_cost

        # Subtract transaction cost drag (spread across month)
        # Actually, apply txn cost as lump sum on rebalance day already tracked above
        # We track it separately for reporting

        daily_rets.append({'date': date, 'return': port_ret})
        txn_cost_series.append({'date': date, 'cum_txn_cost': cum_txn_cost})
        lev_cost_series.append({'date': date, 'cum_lev_cost': cum_lev_cost})

    ret_df = pd.DataFrame(daily_rets).set_index('date')
    # Apply cumulative transaction cost as drag on returns
    # Transaction costs are deducted on rebalance days
    if apply_costs and len(ret_df) > 0:
        txn_df = pd.DataFrame(txn_cost_series).set_index('date')
        lev_df_cost = pd.DataFrame(lev_cost_series).set_index('date')
        # Spread txn cost: deduct incremental txn cost from daily return
        txn_incremental = txn_df['cum_txn_cost'].diff().fillna(0)
        ret_df['return'] = ret_df['return'] - txn_incremental

    weights_df = pd.DataFrame(weights_history).set_index('date') if weights_history else pd.DataFrame()
    leverage_df = pd.DataFrame(leverage_history).set_index('date') if leverage_history else pd.DataFrame()
    cost_df = pd.DataFrame(txn_cost_series).set_index('date') if txn_cost_series else pd.DataFrame()
    if lev_cost_series:
        lev_cost_df = pd.DataFrame(lev_cost_series).set_index('date')
        if len(cost_df) > 0:
            cost_df['cum_lev_cost'] = lev_cost_df['cum_lev_cost']

    return {
        'label': label,
        'returns': ret_df,
        'cumulative': (1 + ret_df['return']).cumprod() if len(ret_df) > 0 else pd.Series(dtype=float),
        'weights': weights_df,
        'leverage': leverage_df,
        'costs': cost_df,
    }


def run_benchmark(returns, weights_dict, label):
    """Run a simple fixed-weight benchmark (60/40 or equal weight)."""
    daily_rets = []
    for i in range(MIN_LOOKBACK_DAYS, len(returns)):
        date = returns.index[i]
        day_ret = returns.iloc[i]
        port_ret = sum(weights_dict.get(t, 0) * day_ret[t] for t in ALL_TICKERS)
        daily_rets.append({'date': date, 'return': port_ret})

    ret_df = pd.DataFrame(daily_rets).set_index('date')
    return {
        'label': label,
        'returns': ret_df,
        'cumulative': (1 + ret_df['return']).cumprod() if len(ret_df) > 0 else pd.Series(dtype=float),
        'weights': pd.DataFrame(),
        'leverage': pd.DataFrame(),
        'costs': pd.DataFrame(),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(result, name=None):
    """Compute comprehensive performance metrics from a backtest result."""
    if name is None:
        name = result['label']
    rets = result['returns']['return']
    n_years = len(rets) / 252
    if n_years == 0:
        return {'Strategy': name}

    cum_ret = (1 + rets).prod()
    ann_ret = cum_ret ** (1 / n_years) - 1
    ann_vol = rets.std() * ANNUALIZATION_FACTOR
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Sortino ratio
    downside = rets[rets < 0].std() * ANNUALIZATION_FACTOR
    sortino = ann_ret / downside if downside > 0 else 0

    # Monthly win rate
    monthly_rets = rets.resample('M').sum()
    win_rate = (monthly_rets > 0).mean()

    metrics = {
        'Strategy': name,
        'Ann. Return (%)': f"{ann_ret*100:.2f}",
        'Ann. Vol (%)': f"{ann_vol*100:.2f}",
        'Sharpe': f"{sharpe:.3f}",
        'Sortino': f"{sortino:.3f}",
        'Max DD (%)': f"{max_dd*100:.2f}",
        'Calmar': f"{calmar:.3f}",
        'Win Rate (%)': f"{win_rate*100:.1f}",
        'Cum. Return (%)': f"{(cum_ret-1)*100:.2f}",
        'sharpe_float': sharpe,  # For bar chart
    }

    # Add cost info if available
    if len(result.get('costs', pd.DataFrame())) > 0:
        costs = result['costs']
        if 'cum_txn_cost' in costs.columns:
            metrics['Txn Cost (%)'] = f"{costs['cum_txn_cost'].iloc[-1]*100:.2f}"
        if 'cum_lev_cost' in costs.columns:
            metrics['Lev Cost (%)'] = f"{costs['cum_lev_cost'].iloc[-1]*100:.2f}"

    return metrics

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: MAIN CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_main_chart(results, output_path):
    """Generate the main 8-panel chart (20x24, 4 rows x 2 cols)."""
    baseline = results['baseline_costs']
    baseline_nc = results['baseline_nocosts']
    sixfour = results['sixfour']
    ew = results['equal_weight']

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle('Risk Parity v2 — Enhanced Backtest\n(S&P 3-Step, 10% Target Vol)',
                 fontsize=16, fontweight='bold', y=0.98)

    C = {'RP': '#1a5276', 'RP_NC': '#2e86c1', '6040': '#c0392b', 'EW': '#7d8c8e',
         'Equity': '#2e86c1', 'FI': '#28b463', 'Comm': '#f39c12'}

    # Panel 1 (full width): Cumulative returns
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(baseline['cumulative'].index, baseline['cumulative'].values, color=C['RP'], lw=2, label='RP (with costs)')
    ax1.plot(baseline_nc['cumulative'].index, baseline_nc['cumulative'].values, color=C['RP_NC'], lw=1.5, ls='-.', label='RP (no costs)')
    ax1.plot(sixfour['cumulative'].index, sixfour['cumulative'].values, color=C['6040'], lw=1.5, ls='--', label='60/40')
    ax1.plot(ew['cumulative'].index, ew['cumulative'].values, color=C['EW'], lw=1.5, ls=':', label='Equal Weight')
    ax1.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Growth of $1')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2 (full width): Drawdowns
    ax2 = fig.add_subplot(gs[1, :])
    for res, lbl, clr in [(baseline, 'RP (costs)', C['RP']), (baseline_nc, 'RP (no costs)', C['RP_NC']),
                           (sixfour, '60/40', C['6040']), (ew, 'EW', C['EW'])]:
        cum = res['cumulative']
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.2, color=clr)
        ax2.plot(dd.index, dd.values, color=clr, lw=1, label=lbl)
    ax2.set_title('Drawdown Analysis', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Drawdown')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3 (left): Capital allocation
    ax3 = fig.add_subplot(gs[2, 0])
    wdf = baseline['weights']
    if len(wdf) > 0:
        total_w = wdf.sum(axis=1)
        pct_df = wdf.div(total_w, axis=0) * 100
        ax3.stackplot(pct_df.index, pct_df['Equity'], pct_df['Fixed Income'], pct_df['Commodities'],
                      labels=['Equity', 'Fixed Income', 'Commodities'],
                      colors=[C['Equity'], C['FI'], C['Comm']], alpha=0.8)
        ax3.set_ylim(0, 100)
    ax3.set_title('Capital Allocation (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Allocation (%)')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4 (right): Leverage
    ax4 = fig.add_subplot(gs[2, 1])
    ldf = baseline['leverage']
    if len(ldf) > 0:
        ax4.plot(ldf.index, ldf['leverage'], color=C['RP'], lw=1.5)
        mean_lev = ldf['leverage'].mean()
        ax4.axhline(y=mean_lev, color='red', ls='--', alpha=0.7, label=f'Mean: {mean_lev:.2f}x')
    ax4.set_title('Portfolio Leverage', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Leverage (x)')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Panel 5 (left): Rolling 12-month returns
    ax5 = fig.add_subplot(gs[3, 0])
    roll_rp = baseline['returns']['return'].rolling(252).sum() * 100
    roll_sf = sixfour['returns']['return'].rolling(252).sum() * 100
    ax5.plot(roll_rp.index, roll_rp.values, color=C['RP'], lw=1.5, label='RP (costs)')
    ax5.plot(roll_sf.index, roll_sf.values, color=C['6040'], lw=1.5, ls='--', label='60/40')
    ax5.axhline(y=0, color='black', lw=0.5)
    ax5.set_title('Rolling 12-Month Returns', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Return (%)')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Panel 6 (right): Rolling 12-month Sharpe
    ax6 = fig.add_subplot(gs[3, 1])
    rp_rets = baseline['returns']['return']
    sf_rets = sixfour['returns']['return']
    roll_rp_sharpe = (rp_rets.rolling(252).mean() / rp_rets.rolling(252).std()) * ANNUALIZATION_FACTOR
    roll_sf_sharpe = (sf_rets.rolling(252).mean() / sf_rets.rolling(252).std()) * ANNUALIZATION_FACTOR
    ax6.plot(roll_rp_sharpe.index, roll_rp_sharpe.values, color=C['RP'], lw=1.5, label='RP (costs)')
    ax6.plot(roll_sf_sharpe.index, roll_sf_sharpe.values, color=C['6040'], lw=1.5, ls='--', label='60/40')
    ax6.axhline(y=0, color='black', lw=0.5)
    ax6.set_title('Rolling 12-Month Sharpe Ratio', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Sharpe Ratio')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart 1 saved to {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: VARIANTS CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_variants_chart(results, all_metrics, output_path):
    """Generate the 6-panel variants comparison chart (20x20, 4 rows x 2 cols, top 2 full-width)."""
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1, 1, 1, 1])
    fig.suptitle('Risk Parity v2 — Strategy Variants Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    colors_lb = {'Expanding': '#1a5276', '3yr Rolling': '#c0392b',
                 '5yr Rolling': '#28b463', 'EWMA (hl=60)': '#f39c12'}
    colors_tv = {'TV=10%': '#1a5276', 'TV=12%': '#8e44ad', 'TV=15%': '#d35400'}

    # Panel 1 (full width): Lookback methods comparison
    ax1 = fig.add_subplot(gs[0, :])
    for key, clr in colors_lb.items():
        rkey = f'lb_{key}'
        if rkey in results:
            cum = results[rkey]['cumulative']
            ax1.plot(cum.index, cum.values, lw=1.8, label=key, color=clr)
    ax1.set_title('Lookback Method Comparison (all at 10% TV, with costs)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Growth of $1')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2 (full width): Target vol comparison
    ax2 = fig.add_subplot(gs[1, :])
    for key, clr in colors_tv.items():
        rkey = f'tv_{key}'
        if rkey in results:
            cum = results[rkey]['cumulative']
            ax2.plot(cum.index, cum.values, lw=1.8, label=key, color=clr)
    ax2.set_title('Target Volatility Comparison (expanding lookback, with costs)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Growth of $1')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 3 (left): S&P 3-step vs ERC
    ax3 = fig.add_subplot(gs[2, 0])
    if 'baseline_costs' in results:
        cum_sp = results['baseline_costs']['cumulative']
        ax3.plot(cum_sp.index, cum_sp.values, lw=1.8, label='S&P 3-Step', color='#1a5276')
    if 'erc' in results:
        cum_erc = results['erc']['cumulative']
        ax3.plot(cum_erc.index, cum_erc.values, lw=1.8, label='ERC Optimizer', color='#c0392b', ls='--')
    ax3.set_title('S&P 3-Step vs ERC (10% TV)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Growth of $1')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4 (right): Rolling Sharpe — S&P vs ERC
    ax4 = fig.add_subplot(gs[2, 1])
    if 'baseline_costs' in results:
        r1 = results['baseline_costs']['returns']['return']
        rs1 = (r1.rolling(252).mean() / r1.rolling(252).std()) * ANNUALIZATION_FACTOR
        ax4.plot(rs1.index, rs1.values, lw=1.5, label='S&P 3-Step', color='#1a5276')
    if 'erc' in results:
        r2 = results['erc']['returns']['return']
        rs2 = (r2.rolling(252).mean() / r2.rolling(252).std()) * ANNUALIZATION_FACTOR
        ax4.plot(rs2.index, rs2.values, lw=1.5, label='ERC Optimizer', color='#c0392b', ls='--')
    ax4.axhline(y=0, color='black', lw=0.5)
    ax4.set_title('Rolling 12-Month Sharpe: S&P vs ERC', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Panel 5 (left): Cost drag over time
    ax5 = fig.add_subplot(gs[3, 0])
    if 'baseline_costs' in results and len(results['baseline_costs']['costs']) > 0:
        cdf = results['baseline_costs']['costs']
        if 'cum_txn_cost' in cdf.columns:
            ax5.plot(cdf.index, cdf['cum_txn_cost'] * 100, lw=1.5, label='Transaction Costs', color='#2e86c1')
        if 'cum_lev_cost' in cdf.columns:
            ax5.plot(cdf.index, cdf['cum_lev_cost'] * 100, lw=1.5, label='Leverage Costs', color='#c0392b')
            if 'cum_txn_cost' in cdf.columns:
                total = (cdf['cum_txn_cost'] + cdf['cum_lev_cost']) * 100
                ax5.plot(cdf.index, total, lw=2, label='Total Cost Drag', color='black', ls='--')
    ax5.set_title('Cumulative Cost Drag', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Cumulative Cost (%)')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Panel 6 (right): Sharpe ratio bar chart for ALL variants
    ax6 = fig.add_subplot(gs[3, 1])
    names = [m['Strategy'] for m in all_metrics]
    sharpes = [m['sharpe_float'] for m in all_metrics]
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax6.barh(range(len(names)), sharpes, color=colors_bar, alpha=0.85)
    ax6.set_yticks(range(len(names)))
    ax6.set_yticklabels(names, fontsize=9)
    ax6.set_xlabel('Sharpe Ratio')
    ax6.set_title('Sharpe Ratio Comparison — All Variants', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.axvline(x=0, color='black', lw=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart 2 saved to {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# PLACEHOLDER: main_execution
if __name__ == '__main__':
    # ── Step 1: Download data ──
    prices, returns, rf_daily = download_data()

    # ── Step 2: Define benchmarks ──
    SIXTY_FORTY = {}
    for t in ASSET_CLASSES['Equity']:
        SIXTY_FORTY[t] = 0.60 / len(ASSET_CLASSES['Equity'])
    for t in ASSET_CLASSES['Fixed Income']:
        SIXTY_FORTY[t] = 0.40 / len(ASSET_CLASSES['Fixed Income'])
    for t in ASSET_CLASSES['Commodities']:
        SIXTY_FORTY[t] = 0.0

    EQUAL_WEIGHT = {t: 1.0 / len(ALL_TICKERS) for t in ALL_TICKERS}

    # ── Step 3: Run ALL strategy variants ──
    all_results = {}
    all_metrics_list = []

    # 3a. Baseline: S&P 3-step, expanding, 10% TV, WITH costs
    print("\n[1/10] Running S&P 3-step, expanding, TV=10%, WITH costs...")
    all_results['baseline_costs'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 Expand 10% (costs)',
        apply_costs=True, target_vol=0.10, vol_method='expanding')

    # 3b. Baseline NO costs
    print("[2/10] Running S&P 3-step, expanding, TV=10%, NO costs...")
    all_results['baseline_nocosts'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 Expand 10% (no costs)',
        apply_costs=False, target_vol=0.10, vol_method='expanding')

    # 3c. S&P 3-step at 12% TV
    print("[3/10] Running S&P 3-step, expanding, TV=12%, WITH costs...")
    all_results['tv_TV=12%'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 Expand 12%',
        apply_costs=True, target_vol=0.12, vol_method='expanding')

    # 3d. S&P 3-step at 15% TV
    print("[4/10] Running S&P 3-step, expanding, TV=15%, WITH costs...")
    all_results['tv_TV=15%'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 Expand 15%',
        apply_costs=True, target_vol=0.15, vol_method='expanding')

    # 3e. S&P 3-step with 3yr rolling
    print("[5/10] Running S&P 3-step, 3yr rolling, TV=10%, WITH costs...")
    all_results['lb_3yr Rolling'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 3yr Roll 10%',
        apply_costs=True, target_vol=0.10, vol_method='rolling_3yr')

    # 3f. S&P 3-step with 5yr rolling
    print("[6/10] Running S&P 3-step, 5yr rolling, TV=10%, WITH costs...")
    all_results['lb_5yr Rolling'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 5yr Roll 10%',
        apply_costs=True, target_vol=0.10, vol_method='rolling_5yr')

    # 3g. S&P 3-step with EWMA
    print("[7/10] Running S&P 3-step, EWMA, TV=10%, WITH costs...")
    all_results['lb_EWMA (hl=60)'] = run_backtest(
        returns, rf_daily, compute_sp3step_weights, 'SP3 EWMA 10%',
        apply_costs=True, target_vol=0.10, vol_method='ewma')

    # 3h. ERC optimizer, expanding, 10% TV
    print("[8/10] Running ERC optimizer, expanding, TV=10%, WITH costs...")
    all_results['erc'] = run_backtest(
        returns, rf_daily, compute_erc_weights, 'ERC 10%',
        apply_costs=True, target_vol=0.10)

    # Also store expanding as a lookback variant for chart
    all_results['lb_Expanding'] = all_results['baseline_costs']
    all_results['tv_TV=10%'] = all_results['baseline_costs']

    # 3i. 60/40 benchmark
    print("[9/10] Running 60/40 benchmark...")
    all_results['sixfour'] = run_benchmark(returns, SIXTY_FORTY, '60/40')

    # 3j. Equal weight benchmark
    print("[10/10] Running Equal Weight benchmark...")
    all_results['equal_weight'] = run_benchmark(returns, EQUAL_WEIGHT, 'Equal Weight')

    # ── Step 4: Compute metrics for ALL variants ──
    print("\n" + "=" * 90)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 90)

    variant_keys = [
        'baseline_costs', 'baseline_nocosts',
        'tv_TV=12%', 'tv_TV=15%',
        'lb_3yr Rolling', 'lb_5yr Rolling', 'lb_EWMA (hl=60)',
        'erc', 'sixfour', 'equal_weight',
    ]

    for key in variant_keys:
        if key in all_results:
            m = compute_metrics(all_results[key])
            all_metrics_list.append(m)

    # Display table (exclude internal sharpe_float)
    display_cols = ['Strategy', 'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe', 'Sortino',
                    'Max DD (%)', 'Calmar', 'Win Rate (%)', 'Cum. Return (%)']
    metrics_df = pd.DataFrame(all_metrics_list)
    extra_cols = [c for c in ['Txn Cost (%)', 'Lev Cost (%)'] if c in metrics_df.columns]
    print(metrics_df[display_cols + extra_cols].set_index('Strategy').to_string())

    # ── Step 5: Generate charts ──
    print("\nGenerating charts...")
    plot_main_chart(all_results, '/workspace/output/risk_parity_v2_main.png')
    plot_variants_chart(all_results, all_metrics_list, '/workspace/output/risk_parity_v2_variants.png')

    # ── Step 6: Save metrics CSV ──
    metrics_df_save = metrics_df.drop(columns=['sharpe_float'], errors='ignore')
    metrics_df_save.set_index('Strategy').to_csv('/workspace/output/risk_parity_v2_metrics.csv')
    print("Metrics saved to /workspace/output/risk_parity_v2_metrics.csv")

    # ── Step 7: Print cost summary ──
    print("\n" + "=" * 90)
    print("COST ANALYSIS (Baseline: S&P 3-step, Expanding, 10% TV)")
    print("=" * 90)
    if len(all_results['baseline_costs']['costs']) > 0:
        cdf = all_results['baseline_costs']['costs']
        if 'cum_txn_cost' in cdf.columns:
            print(f"  Cumulative Transaction Costs: {cdf['cum_txn_cost'].iloc[-1]*100:.3f}%")
        if 'cum_lev_cost' in cdf.columns:
            print(f"  Cumulative Leverage Costs:    {cdf['cum_lev_cost'].iloc[-1]*100:.3f}%")
            if 'cum_txn_cost' in cdf.columns:
                total = cdf['cum_txn_cost'].iloc[-1] + cdf['cum_lev_cost'].iloc[-1]
                print(f"  Total Cost Drag:              {total*100:.3f}%")

    # ── Step 8: Leverage summary ──
    print("\n" + "=" * 90)
    print("LEVERAGE SUMMARY (Baseline)")
    print("=" * 90)
    ldf = all_results['baseline_costs']['leverage']
    if len(ldf) > 0:
        print(f"  Mean leverage: {ldf['leverage'].mean():.2f}x")
        print(f"  Min leverage:  {ldf['leverage'].min():.2f}x")
        print(f"  Max leverage:  {ldf['leverage'].max():.2f}x")

    print("\nBacktest v2 complete.")
