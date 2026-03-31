"""
Risk Parity Strategy Backtest v5 — Expanded Universe
Based on: "Indexing Risk Parity Strategies" (S&P Dow Jones Indices, October 2018)

Expanded to 14 ETFs mapping to the paper's 26 futures as closely as possible.
Compares: v5 SP3 Expanded (14 ETFs), v2 SP3 Original (9 ETFs), ERC Expanded,
60/40 benchmark, Equal Weight benchmark.
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

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TARGET_VOL = 0.10
MIN_LOOKBACK_DAYS = 252
MAX_LOOKBACK_DAYS = 252 * 15
ANNUALIZATION_FACTOR = np.sqrt(252)
START_DATE = '2007-01-01'
END_DATE = '2025-12-31'
TRANSACTION_COST_BPS = 5
FALLBACK_RF_RATE = 0.02

# v5 Expanded Universe (14 ETFs)
ASSET_CLASSES_V5 = {
    'Equity': ['SPY', 'FEZ', 'EWJ'],
    'Fixed Income': ['IEI', 'IEF', 'TLT', 'BWX'],
    'Commodities': ['USO', 'UNG', 'UGA', 'DBC', 'GLD', 'SLV', 'DBA'],
}
ALL_TICKERS_V5 = ['SPY', 'FEZ', 'EWJ', 'IEI', 'IEF', 'TLT', 'BWX',
                   'USO', 'UNG', 'UGA', 'DBC', 'GLD', 'SLV', 'DBA']

# v2 Original Universe (9 ETFs)
ASSET_CLASSES_V2 = {
    'Equity': ['SPY', 'EFA', 'EEM'],
    'Fixed Income': ['IEF', 'TLT', 'AGG'],
    'Commodities': ['GLD', 'DBC', 'SLV'],
}
ALL_TICKERS_V2 = ['SPY', 'EFA', 'EEM', 'IEF', 'TLT', 'AGG', 'GLD', 'DBC', 'SLV']

# All tickers needed for download
ALL_DOWNLOAD_TICKERS = sorted(set(ALL_TICKERS_V5 + ALL_TICKERS_V2))

# Color scheme
C_V5 = '#1a5276'
C_V2 = '#8e44ad'
C_6040 = '#c0392b'
C_EW = '#7d8c8e'
C_ERC = '#e67e22'
C_NOCOST = '#2ecc71'
C_EQUITY = '#2e86c1'
C_FI = '#28b463'
C_COMM = '#f39c12'

# ==============================================================================
# DATA DOWNLOAD
# ==============================================================================

def download_data():
    """Download price data for all tickers and ^IRX for risk-free rate."""
    print("Downloading price data...")
    tickers_to_dl = ALL_DOWNLOAD_TICKERS + ['^IRX']
    data = yf.download(tickers_to_dl, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices_all = data['Close']

    # Extract IRX before dropping NaN
    irx_raw = prices_all['^IRX'].copy() if '^IRX' in prices_all.columns else None
    prices_etf = prices_all.drop(columns=['^IRX'], errors='ignore')

    # Drop rows with any NaN across ALL ETFs to get common date range
    prices_etf = prices_etf.dropna()
    returns_all = prices_etf.pct_change().dropna()
    print(f"Common data range: {returns_all.index[0].date()} to {returns_all.index[-1].date()}")
    print(f"Total trading days: {len(returns_all)}")

    # Build risk-free rate series
    if irx_raw is not None and irx_raw.notna().sum() > 0:
        rf_annual = irx_raw / 100.0
        rf_daily = rf_annual.reindex(returns_all.index).ffill().bfill() / 252.0
        rf_daily = rf_daily.fillna(FALLBACK_RF_RATE / 252.0)
        print(f"^IRX data loaded: {rf_daily.notna().sum()} days")
    else:
        print(f"^IRX unavailable, using fallback rate of {FALLBACK_RF_RATE*100:.0f}%")
        rf_daily = pd.Series(FALLBACK_RF_RATE / 252.0, index=returns_all.index)

    # Build v2 returns (subset of columns, may have different common start if all v2 tickers present earlier)
    returns_v5 = returns_all[ALL_TICKERS_V5].copy()
    returns_v2 = returns_all[ALL_TICKERS_V2].copy()

    return returns_all, returns_v5, returns_v2, rf_daily

# ==============================================================================
# WEIGHT COMPUTATION: S&P 3-STEP METHOD
# ==============================================================================

def compute_sp3_weights(returns_history, asset_classes, all_tickers, target_vol=TARGET_VOL):
    """
    3-step S&P Risk Parity weight computation.
    Step 1: Instrument-level inverse-vol weighting
    Step 2: Asset-class-level multiplier to equalize risk contribution
    Step 3: Portfolio-level multiplier to hit target volatility
    """
    n_days = len(returns_history)
    if n_days < MIN_LOOKBACK_DAYS:
        return None

    lb = max(MIN_LOOKBACK_DAYS, min(n_days, MAX_LOOKBACK_DAYS))
    hist = returns_history[all_tickers].iloc[-lb:]

    # Step 1: Compute instrument vols and inverse-vol raw weights
    instrument_vols = {}
    raw_weights = {}
    for ticker in all_tickers:
        vol = hist[ticker].std() * ANNUALIZATION_FACTOR
        instrument_vols[ticker] = vol
        raw_weights[ticker] = target_vol / vol if vol > 0 else 0

    # Step 2: Asset-class-level multiplier
    asset_class_weights = {}
    for ac_name, tickers in asset_classes.items():
        n_inst = len(tickers)
        ac_ticker_weights = {t: raw_weights[t] / n_inst for t in tickers}
        ac_returns = sum(ac_ticker_weights[t] * hist[t] for t in tickers)
        ac_vol = ac_returns.std() * ANNUALIZATION_FACTOR
        multiplier = target_vol / ac_vol if ac_vol > 0 else 1.0
        for t in tickers:
            asset_class_weights[t] = ac_ticker_weights[t] * multiplier

    # Step 3: Portfolio-level multiplier
    n_classes = len(asset_classes)
    portfolio_weights = {t: asset_class_weights[t] / n_classes for t in all_tickers}
    port_returns = sum(portfolio_weights[t] * hist[t] for t in all_tickers)
    port_vol = port_returns.std() * ANNUALIZATION_FACTOR
    port_multiplier = target_vol / port_vol if port_vol > 0 else 1.0
    final_weights = {t: portfolio_weights[t] * port_multiplier for t in all_tickers}

    return final_weights

# ==============================================================================
# WEIGHT COMPUTATION: ERC OPTIMIZER
# ==============================================================================

def compute_erc_weights(returns_history, all_tickers, target_vol=TARGET_VOL):
    """
    Equal Risk Contribution weights via scipy SLSQP optimization.
    Minimize sum of squared differences in risk contributions, then scale to target vol.
    """
    n_days = len(returns_history)
    if n_days < MIN_LOOKBACK_DAYS:
        return None

    lb = max(MIN_LOOKBACK_DAYS, min(n_days, MAX_LOOKBACK_DAYS))
    hist = returns_history[all_tickers].iloc[-lb:]
    cov = hist.cov().values * 252
    n = len(all_tickers)

    def erc_objective(w):
        sigma_w = cov @ w
        rc = w * sigma_w
        total = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                total += (rc[i] - rc[j]) ** 2
        return total

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, None) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(erc_objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    w_opt = result.x if result.success else np.ones(n) / n

    # Scale to target volatility
    port_vol = np.sqrt(w_opt @ cov @ w_opt)
    scale = target_vol / port_vol if port_vol > 0 else 1.0
    final_weights = {all_tickers[i]: w_opt[i] * scale for i in range(n)}

    return final_weights

# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================

def run_single_backtest(returns, all_tickers, asset_classes, weight_func, rf_daily,
                        txn_bps=TRANSACTION_COST_BPS, include_costs=True, label=''):
    """
    Generic backtest engine.
    weight_func: callable(returns_history, asset_classes_or_tickers, all_tickers, target_vol) or similar
    Returns dict with daily_returns (pd.Series), weights_history, leverage_history, costs.
    """
    # Monthly rebalance dates
    monthly_last_days = returns.groupby(returns.index.to_period('M')).apply(lambda x: x.index[-1])
    valid_dates = [d for d in monthly_last_days if returns.index.get_loc(d) >= MIN_LOOKBACK_DAYS]

    daily_rets = []
    weights_history = []       # per-ETF weights at each rebalance
    leverage_history = []
    cum_txn_cost = 0.0
    cum_lev_cost = 0.0
    txn_increments = {}        # date -> txn cost increment

    current_weights = None
    prev_weights = None
    rebal_idx = 0

    for i in range(MIN_LOOKBACK_DAYS, len(returns)):
        date = returns.index[i]

        # Check if rebalance
        if rebal_idx < len(valid_dates) and date >= valid_dates[rebal_idx]:
            history = returns.iloc[max(0, i - MAX_LOOKBACK_DAYS):i]
            new_weights = weight_func(history, asset_classes, all_tickers, TARGET_VOL)

            if new_weights is not None:
                prev_weights = current_weights
                current_weights = new_weights

                # Transaction cost
                if include_costs and prev_weights is not None:
                    turnover = sum(abs(current_weights.get(t, 0) - prev_weights.get(t, 0))
                                   for t in all_tickers)
                    txn_drag = turnover * txn_bps / 10000.0
                    cum_txn_cost += txn_drag
                    txn_increments[date] = txn_drag

                # Record leverage
                total_leverage = sum(abs(v) for v in current_weights.values())
                leverage_history.append({'date': date, 'leverage': total_leverage})

                # Record per-ETF weights
                w_record = {'date': date}
                for t in all_tickers:
                    w_record[t] = current_weights.get(t, 0)
                weights_history.append(w_record)

            rebal_idx += 1

        if current_weights is None:
            continue

        # Daily portfolio return
        day_ret = returns.iloc[i]
        port_ret = sum(current_weights.get(t, 0) * day_ret[t] for t in all_tickers)

        # Leverage financing cost (daily)
        if include_costs:
            total_lev = sum(abs(v) for v in current_weights.values())
            if total_lev > 1.0:
                rf_rate_today = rf_daily.iloc[i] if i < len(rf_daily) else FALLBACK_RF_RATE / 252
                daily_lev_cost = (total_lev - 1.0) * float(rf_rate_today)
                port_ret -= daily_lev_cost
                cum_lev_cost += daily_lev_cost

        # Subtract txn cost on rebalance day
        if date in txn_increments:
            port_ret -= txn_increments[date]

        daily_rets.append({'date': date, 'return': port_ret})

    ret_series = pd.DataFrame(daily_rets).set_index('date')['return'] if daily_rets else pd.Series(dtype=float)
    weights_df = pd.DataFrame(weights_history).set_index('date') if weights_history else pd.DataFrame()
    leverage_df = pd.DataFrame(leverage_history).set_index('date') if leverage_history else pd.DataFrame()

    return {
        'label': label,
        'daily_returns': ret_series,
        'weights_history': weights_df,
        'leverage_history': leverage_df,
        'cum_txn_cost': cum_txn_cost,
        'cum_lev_cost': cum_lev_cost,
    }

def run_benchmark(returns, weights_dict, all_tickers, label=''):
    """Run a simple fixed-weight benchmark (60/40 or equal weight)."""
    daily_rets = []
    for i in range(MIN_LOOKBACK_DAYS, len(returns)):
        date = returns.index[i]
        day_ret = returns.iloc[i]
        port_ret = sum(weights_dict.get(t, 0) * day_ret[t] for t in all_tickers)
        daily_rets.append({'date': date, 'return': port_ret})

    ret_series = pd.DataFrame(daily_rets).set_index('date')['return'] if daily_rets else pd.Series(dtype=float)
    return {
        'label': label,
        'daily_returns': ret_series,
        'weights_history': pd.DataFrame(),
        'leverage_history': pd.DataFrame(),
        'cum_txn_cost': 0.0,
        'cum_lev_cost': 0.0,
    }

# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================

def compute_metrics(daily_returns_series, name, txn_cost=0, lev_cost=0):
    """Compute comprehensive performance metrics from a daily returns series."""
    rets = daily_returns_series
    n_years = len(rets) / 252
    if n_years == 0:
        return {'Strategy': name, 'sharpe_float': 0}

    cum_ret = (1 + rets).prod()
    ann_ret = cum_ret ** (1 / n_years) - 1
    ann_vol = rets.std() * ANNUALIZATION_FACTOR
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    downside = rets[rets < 0].std() * ANNUALIZATION_FACTOR
    sortino = ann_ret / downside if downside > 0 else 0
    monthly_rets = rets.resample('ME').sum()
    win_rate = (monthly_rets > 0).mean()

    return {
        'Strategy': name,
        'Ann. Return (%)': f"{ann_ret*100:.2f}",
        'Ann. Vol (%)': f"{ann_vol*100:.2f}",
        'Sharpe': f"{sharpe:.3f}",
        'Sortino': f"{sortino:.3f}",
        'Max DD (%)': f"{max_dd*100:.2f}",
        'Calmar': f"{calmar:.3f}",
        'Win Rate (%)': f"{win_rate*100:.1f}",
        'Cum. Return (%)': f"{(cum_ret-1)*100:.2f}",
        'Txn Cost (%)': f"{txn_cost*100:.3f}",
        'Lev Cost (%)': f"{lev_cost*100:.3f}",
        'sharpe_float': sharpe,
    }

# ==============================================================================
# CHART 1: MAIN CHART
# ==============================================================================

def plot_main_chart(results, output_path):
    """
    Chart 1: 20x28, GridSpec 5 rows x 2 cols.
    Row 0 full: Cumulative returns (v5, v5 no costs, v2, 60/40, EW)
    Row 1 full: Drawdowns (v5 vs 60/40 vs v2)
    Row 2 left: Capital allocation over time (v5 stacked area)
    Row 2 right: Leverage over time (v5)
    Row 3 left: Rolling 12m returns (v5 vs 60/40)
    Row 3 right: Rolling 12m Sharpe (v5 vs 60/40)
    Row 4 left: Rolling 3m realized vol (v5) with 10% target line
    Row 4 right: Annual returns bar chart (v5 vs v2 vs 60/40)
    """
    v5 = results['v5_costs']
    v5nc = results['v5_nocosts']
    v2 = results['v2_costs']
    sf = results['sixfour']
    ew = results['equal_weight']

    # Cumulative series
    cum_v5 = (1 + v5['daily_returns']).cumprod()
    cum_v5nc = (1 + v5nc['daily_returns']).cumprod()
    cum_v2 = (1 + v2['daily_returns']).cumprod()
    cum_sf = (1 + sf['daily_returns']).cumprod()
    cum_ew = (1 + ew['daily_returns']).cumprod()

    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle('Risk Parity v5 — Expanded Universe Backtest (14 ETFs, 10% Target Vol)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Row 0: Cumulative returns
    ax = fig.add_subplot(gs[0, :])
    ax.plot(cum_v5.index, cum_v5.values, color=C_V5, lw=2, label='v5 Expanded (costs)')
    ax.plot(cum_v5nc.index, cum_v5nc.values, color=C_NOCOST, lw=1.5, ls='-.', label='v5 Expanded (no costs)')
    ax.plot(cum_v2.index, cum_v2.values, color=C_V2, lw=1.5, ls='--', label='v2 Original (costs)')
    ax.plot(cum_sf.index, cum_sf.values, color=C_6040, lw=1.5, ls='--', label='60/40')
    ax.plot(cum_ew.index, cum_ew.values, color=C_EW, lw=1.5, ls=':', label='Equal Weight')
    ax.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 1: Drawdowns
    ax = fig.add_subplot(gs[1, :])
    for cum, lbl, clr in [(cum_v5, 'v5 Expanded', C_V5), (cum_v2, 'v2 Original', C_V2),
                           (cum_sf, '60/40', C_6040)]:
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ax.fill_between(dd.index, dd.values, 0, alpha=0.15, color=clr)
        ax.plot(dd.index, dd.values, color=clr, lw=1, label=lbl)
    ax.set_title('Drawdown Analysis', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2 left: Capital allocation (v5 stacked area)
    ax = fig.add_subplot(gs[2, 0])
    wdf = v5['weights_history']
    if len(wdf) > 0:
        ac_weights = pd.DataFrame(index=wdf.index)
        for ac_name, tickers in ASSET_CLASSES_V5.items():
            ac_weights[ac_name] = wdf[[t for t in tickers if t in wdf.columns]].sum(axis=1)
        total_w = ac_weights.sum(axis=1)
        pct_df = ac_weights.div(total_w, axis=0) * 100
        ax.stackplot(pct_df.index, pct_df['Equity'], pct_df['Fixed Income'], pct_df['Commodities'],
                      labels=['Equity', 'Fixed Income', 'Commodities'],
                      colors=[C_EQUITY, C_FI, C_COMM], alpha=0.8)
        ax.set_ylim(0, 100)
    ax.set_title('Capital Allocation — v5 Expanded (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Allocation (%)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2 right: Leverage over time (v5)
    ax = fig.add_subplot(gs[2, 1])
    ldf = v5['leverage_history']
    if len(ldf) > 0:
        ax.plot(ldf.index, ldf['leverage'], color=C_V5, lw=1.5)
        mean_lev = ldf['leverage'].mean()
        ax.axhline(y=mean_lev, color='red', ls='--', alpha=0.7, label=f'Mean: {mean_lev:.2f}x')
    ax.set_title('Portfolio Leverage — v5 Expanded', fontsize=13, fontweight='bold')
    ax.set_ylabel('Leverage (x)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 3 left: Rolling 12m returns
    ax = fig.add_subplot(gs[3, 0])
    roll_v5 = v5['daily_returns'].rolling(252).sum() * 100
    roll_sf = sf['daily_returns'].rolling(252).sum() * 100
    ax.plot(roll_v5.index, roll_v5.values, color=C_V5, lw=1.5, label='v5 Expanded')
    ax.plot(roll_sf.index, roll_sf.values, color=C_6040, lw=1.5, ls='--', label='60/40')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_title('Rolling 12-Month Returns', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 3 right: Rolling 12m Sharpe
    ax = fig.add_subplot(gs[3, 1])
    r_v5 = v5['daily_returns']
    r_sf = sf['daily_returns']
    rs_v5 = (r_v5.rolling(252).mean() / r_v5.rolling(252).std()) * ANNUALIZATION_FACTOR
    rs_sf = (r_sf.rolling(252).mean() / r_sf.rolling(252).std()) * ANNUALIZATION_FACTOR
    ax.plot(rs_v5.index, rs_v5.values, color=C_V5, lw=1.5, label='v5 Expanded')
    ax.plot(rs_sf.index, rs_sf.values, color=C_6040, lw=1.5, ls='--', label='60/40')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_title('Rolling 12-Month Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 4 left: Rolling 3m realized vol with 10% target
    ax = fig.add_subplot(gs[4, 0])
    roll_vol = r_v5.rolling(63).std() * ANNUALIZATION_FACTOR * 100
    ax.plot(roll_vol.index, roll_vol.values, color=C_V5, lw=1.5, label='Realized Vol')
    ax.axhline(y=10, color='red', ls='--', lw=1.5, label='10% Target')
    ax.set_title('Rolling 3-Month Realized Volatility — v5', fontsize=13, fontweight='bold')
    ax.set_ylabel('Annualized Vol (%)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 4 right: Annual returns bar chart
    ax = fig.add_subplot(gs[4, 1])
    ann_v5 = v5['daily_returns'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
    ann_v2 = v2['daily_returns'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
    ann_sf = sf['daily_returns'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
    years = ann_v5.index.year
    x = np.arange(len(years))
    w = 0.25
    ax.bar(x - w, ann_v5.values, w, color=C_V5, label='v5 Expanded', alpha=0.85)
    ax.bar(x, ann_v2.values[:len(years)], w, color=C_V2, label='v2 Original', alpha=0.85)
    ax.bar(x + w, ann_sf.values[:len(years)], w, color=C_6040, label='60/40', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_title('Annual Returns Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart 1 saved to {output_path}")

# ==============================================================================
# CHART 2: COMPARISON CHART
# ==============================================================================

def plot_comparison_chart(results, all_metrics, output_path):
    """
    Chart 2: 20x16, GridSpec 3 rows x 2 cols.
    Row 0 full: Cumulative returns (v5 vs v2 vs ERC, all with costs)
    Row 1 left: Asset class allocation grouped bar (v5 vs v2)
    Row 1 right: Per-ETF weight breakdown for v5 (horizontal bar, 14 bars)
    Row 2 left: Rolling Sharpe (v5 vs v2)
    Row 2 right: Bar chart of Sharpe ratios for all 6 strategies
    """
    v5 = results['v5_costs']
    v2 = results['v2_costs']
    erc = results['erc_costs']

    cum_v5 = (1 + v5['daily_returns']).cumprod()
    cum_v2 = (1 + v2['daily_returns']).cumprod()
    cum_erc = (1 + erc['daily_returns']).cumprod()

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('Risk Parity v5 — Expanded vs Original Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    # Row 0: Cumulative returns
    ax = fig.add_subplot(gs[0, :])
    ax.plot(cum_v5.index, cum_v5.values, color=C_V5, lw=2, label='v5 SP3 Expanded (14 ETFs)')
    ax.plot(cum_v2.index, cum_v2.values, color=C_V2, lw=1.5, ls='--', label='v2 SP3 Original (9 ETFs)')
    ax.plot(cum_erc.index, cum_erc.values, color=C_ERC, lw=1.5, ls='-.', label='v5 ERC Expanded (14 ETFs)')
    ax.set_title('Cumulative Returns — All with Costs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 1 left: Asset class allocation grouped bar (v5 vs v2 mean allocations)
    ax = fig.add_subplot(gs[1, 0])
    # Compute mean allocations for v5
    wdf_v5 = v5['weights_history']
    v5_ac_mean = {}
    if len(wdf_v5) > 0:
        for ac_name, tickers in ASSET_CLASSES_V5.items():
            cols = [t for t in tickers if t in wdf_v5.columns]
            v5_ac_mean[ac_name] = wdf_v5[cols].sum(axis=1).mean()
        total_v5 = sum(v5_ac_mean.values())
        v5_ac_pct = {k: v / total_v5 * 100 for k, v in v5_ac_mean.items()}
    else:
        v5_ac_pct = {'Equity': 33, 'Fixed Income': 33, 'Commodities': 33}

    wdf_v2 = v2['weights_history']
    v2_ac_mean = {}
    if len(wdf_v2) > 0:
        for ac_name, tickers in ASSET_CLASSES_V2.items():
            cols = [t for t in tickers if t in wdf_v2.columns]
            v2_ac_mean[ac_name] = wdf_v2[cols].sum(axis=1).mean()
        total_v2 = sum(v2_ac_mean.values())
        v2_ac_pct = {k: v / total_v2 * 100 for k, v in v2_ac_mean.items()}
    else:
        v2_ac_pct = {'Equity': 33, 'Fixed Income': 33, 'Commodities': 33}

    ac_names = ['Equity', 'Fixed Income', 'Commodities']
    x = np.arange(len(ac_names))
    w = 0.35
    ax.bar(x - w/2, [v5_ac_pct.get(a, 0) for a in ac_names], w, color=C_V5, label='v5 Expanded', alpha=0.85)
    ax.bar(x + w/2, [v2_ac_pct.get(a, 0) for a in ac_names], w, color=C_V2, label='v2 Original', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ac_names)
    ax.set_ylabel('Mean Allocation (%)')
    ax.set_title('Asset Class Allocation Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 1 right: Per-ETF weight breakdown for v5 (horizontal bar)
    ax = fig.add_subplot(gs[1, 1])
    if len(wdf_v5) > 0:
        etf_means = {}
        etf_colors = {}
        ac_color_map = {'Equity': C_EQUITY, 'Fixed Income': C_FI, 'Commodities': C_COMM}
        for ac_name, tickers in ASSET_CLASSES_V5.items():
            for t in tickers:
                if t in wdf_v5.columns:
                    etf_means[t] = wdf_v5[t].mean() * 100
                    etf_colors[t] = ac_color_map[ac_name]
        etfs = list(etf_means.keys())
        vals = [etf_means[t] for t in etfs]
        colors = [etf_colors[t] for t in etfs]
        y_pos = np.arange(len(etfs))
        ax.barh(y_pos, vals, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(etfs, fontsize=9)
        ax.set_xlabel('Mean Weight (%)')
        # Legend for asset classes
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=C_EQUITY, label='Equity'),
                           Patch(facecolor=C_FI, label='Fixed Income'),
                           Patch(facecolor=C_COMM, label='Commodities')]
        ax.legend(handles=legend_elements, fontsize=9)
    ax.set_title('Per-ETF Mean Weights — v5 Expanded', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Row 2 left: Rolling Sharpe (v5 vs v2)
    ax = fig.add_subplot(gs[2, 0])
    r_v5 = v5['daily_returns']
    r_v2 = v2['daily_returns']
    rs_v5 = (r_v5.rolling(252).mean() / r_v5.rolling(252).std()) * ANNUALIZATION_FACTOR
    rs_v2 = (r_v2.rolling(252).mean() / r_v2.rolling(252).std()) * ANNUALIZATION_FACTOR
    ax.plot(rs_v5.index, rs_v5.values, color=C_V5, lw=1.5, label='v5 Expanded')
    ax.plot(rs_v2.index, rs_v2.values, color=C_V2, lw=1.5, ls='--', label='v2 Original')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.set_title('Rolling 12-Month Sharpe Ratio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Row 2 right: Sharpe ratios bar chart for all 6 strategies
    ax = fig.add_subplot(gs[2, 1])
    names = [m['Strategy'] for m in all_metrics]
    sharpes = [m['sharpe_float'] for m in all_metrics]
    strat_colors = [C_V5, C_NOCOST, C_V2, C_ERC, C_6040, C_EW]
    # Pad colors if needed
    while len(strat_colors) < len(names):
        strat_colors.append('#555555')
    bars = ax.barh(range(len(names)), sharpes, color=strat_colors[:len(names)], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio — All Strategies', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='black', lw=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart 2 saved to {output_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    # ── Step 1: Download data ──
    returns_all, returns_v5, returns_v2, rf_daily = download_data()

    # ── Step 2: Run all strategies ──
    results = {}

    # 1. v5 SP3 Expanded (14 ETFs, with costs)
    print("\n[1/6] Running v5 SP3 Expanded (14 ETFs, with costs)...")
    results['v5_costs'] = run_single_backtest(
        returns_v5, ALL_TICKERS_V5, ASSET_CLASSES_V5, compute_sp3_weights, rf_daily,
        txn_bps=TRANSACTION_COST_BPS, include_costs=True,
        label='v5 SP3 Expanded (costs)')

    # 2. v5 SP3 Expanded (14 ETFs, no costs)
    print("[2/6] Running v5 SP3 Expanded (14 ETFs, no costs)...")
    results['v5_nocosts'] = run_single_backtest(
        returns_v5, ALL_TICKERS_V5, ASSET_CLASSES_V5, compute_sp3_weights, rf_daily,
        txn_bps=0, include_costs=False,
        label='v5 SP3 Expanded (no costs)')

    # 3. v2 SP3 Original (9 ETFs, with costs)
    print("[3/6] Running v2 SP3 Original (9 ETFs, with costs)...")
    results['v2_costs'] = run_single_backtest(
        returns_v2, ALL_TICKERS_V2, ASSET_CLASSES_V2, compute_sp3_weights, rf_daily,
        txn_bps=TRANSACTION_COST_BPS, include_costs=True,
        label='v2 SP3 Original (costs)')

    # 4. v5 ERC Expanded (14 ETFs, with costs)
    # ERC weight_func has different signature — wrap it
    def erc_weight_func(returns_history, asset_classes, all_tickers, target_vol):
        return compute_erc_weights(returns_history, all_tickers, target_vol)

    print("[4/6] Running v5 ERC Expanded (14 ETFs, with costs)...")
    results['erc_costs'] = run_single_backtest(
        returns_v5, ALL_TICKERS_V5, ASSET_CLASSES_V5, erc_weight_func, rf_daily,
        txn_bps=TRANSACTION_COST_BPS, include_costs=True,
        label='v5 ERC Expanded (costs)')

    # 5. 60/40 benchmark (expanded universe)
    print("[5/6] Running 60/40 benchmark...")
    SIXTY_FORTY = {}
    for t in ASSET_CLASSES_V5['Equity']:
        SIXTY_FORTY[t] = 0.60 / len(ASSET_CLASSES_V5['Equity'])
    for t in ASSET_CLASSES_V5['Fixed Income']:
        SIXTY_FORTY[t] = 0.40 / len(ASSET_CLASSES_V5['Fixed Income'])
    for t in ASSET_CLASSES_V5['Commodities']:
        SIXTY_FORTY[t] = 0.0
    results['sixfour'] = run_benchmark(returns_v5, SIXTY_FORTY, ALL_TICKERS_V5, label='60/40')

    # 6. Equal Weight benchmark
    print("[6/6] Running Equal Weight benchmark...")
    EW_WEIGHTS = {t: 1.0 / len(ALL_TICKERS_V5) for t in ALL_TICKERS_V5}
    results['equal_weight'] = run_benchmark(returns_v5, EW_WEIGHTS, ALL_TICKERS_V5, label='Equal Weight')

    # ── Step 3: Compute metrics ──
    all_metrics = []
    strategy_order = ['v5_costs', 'v5_nocosts', 'v2_costs', 'erc_costs', 'sixfour', 'equal_weight']
    for key in strategy_order:
        r = results[key]
        m = compute_metrics(r['daily_returns'], r['label'], r['cum_txn_cost'], r['cum_lev_cost'])
        all_metrics.append(m)

    # ── Step 4: Print comparison table ──
    print("\n" + "=" * 110)
    print("FULL COMPARISON TABLE")
    print("=" * 110)
    display_cols = ['Strategy', 'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe', 'Sortino',
                    'Max DD (%)', 'Calmar', 'Win Rate (%)', 'Cum. Return (%)',
                    'Txn Cost (%)', 'Lev Cost (%)']
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df[display_cols].set_index('Strategy').to_string())

    # ── Step 5: Capital allocation comparison ──
    print("\n" + "=" * 110)
    print("CAPITAL ALLOCATION COMPARISON (mean % in each asset class)")
    print("=" * 110)

    def print_ac_breakdown(wdf, ac_dict, label):
        if len(wdf) == 0:
            print(f"  {label}: No weight data")
            return
        ac_mean = {}
        for ac_name, tickers in ac_dict.items():
            cols = [t for t in tickers if t in wdf.columns]
            ac_mean[ac_name] = wdf[cols].sum(axis=1).mean()
        total = sum(ac_mean.values())
        print(f"  {label}:")
        for ac_name, val in ac_mean.items():
            print(f"    {ac_name:20s}: {val/total*100:6.2f}%  (raw weight: {val:.4f})")
        print(f"    {'Total leverage':20s}: {total:.4f}x")

    print_ac_breakdown(results['v5_costs']['weights_history'], ASSET_CLASSES_V5, 'v5 Expanded')
    print_ac_breakdown(results['v2_costs']['weights_history'], ASSET_CLASSES_V2, 'v2 Original')

    # ── Step 6: Per-ETF weight breakdown for v5 ──
    print("\n" + "=" * 110)
    print("WITHIN-CLASS WEIGHT BREAKDOWN — v5 EXPANDED (mean weight to each ETF)")
    print("=" * 110)
    wdf_v5 = results['v5_costs']['weights_history']
    if len(wdf_v5) > 0:
        for ac_name, tickers in ASSET_CLASSES_V5.items():
            print(f"  {ac_name}:")
            for t in tickers:
                if t in wdf_v5.columns:
                    mean_w = wdf_v5[t].mean()
                    print(f"    {t:6s}: {mean_w*100:7.3f}%")

    # ── Step 7: Generate charts ──
    print("\nGenerating charts...")
    plot_main_chart(results, '/workspace/output/risk_parity_v5_main.png')
    plot_comparison_chart(results, all_metrics, '/workspace/output/risk_parity_v5_comparison.png')

    # ── Step 8: Save CSV ──
    metrics_df_save = metrics_df.drop(columns=['sharpe_float'], errors='ignore')
    metrics_df_save.set_index('Strategy').to_csv('/workspace/output/risk_parity_v5_metrics.csv')
    print("Metrics saved to /workspace/output/risk_parity_v5_metrics.csv")

    print("\nBacktest v5 complete.")
