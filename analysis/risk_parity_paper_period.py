"""
Risk Parity Backtest — Paper Period Comparison
Backtests over the S&P paper period (Dec 2003 – May 2018) using v2 universe.
Compares our SP3 implementation against the paper's published numbers.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
DOWNLOAD_START = '1998-01-01'
DOWNLOAD_END = '2018-06-30'
TRANSACTION_COST_BPS = 5
FALLBACK_RF_RATE = 0.02

# v2 Universe — 9 ETFs
ASSET_CLASSES_V2 = {
    'Equity': ['SPY', 'EFA', 'EEM'],
    'Fixed Income': ['IEF', 'TLT', 'AGG'],
    'Commodities': ['GLD', 'DBC', 'SLV'],
}
ALL_TICKERS_V2 = ['SPY', 'EFA', 'EEM', 'IEF', 'TLT', 'AGG', 'GLD', 'DBC', 'SLV']

# 6-ETF Universe — Equity + Fixed Income only
ASSET_CLASSES_6ETF = {
    'Equity': ['SPY', 'EFA', 'EEM'],
    'Fixed Income': ['IEF', 'TLT', 'AGG'],
}
ALL_TICKERS_6ETF = ['SPY', 'EFA', 'EEM', 'IEF', 'TLT', 'AGG']

# Paper's published numbers (Dec 2003 – May 2018)
PAPER_RP_10 = {'return': 7.30, 'vol': 8.34, 'sharpe': 0.73, 'max_dd': -28.17}
PAPER_6040 = {'return': 6.31, 'vol': 9.90, 'sharpe': 0.52, 'max_dd': -36.42}
PAPER_HFR = {'return': 7.36, 'vol': 8.34, 'sharpe': 0.74, 'max_dd': -22.43}

# Colors
C_PAPER = '#d4a017'
C_RP_NOCOST = '#1a5276'
C_RP_COST = '#2e86c1'
C_6040 = '#c0392b'
C_SPY = '#7d8c8e'
C_6ETF = '#8e44ad'

# ==============================================================================
# CORE FUNCTIONS (from v5)
# ==============================================================================

def compute_sp3_weights(returns_history, asset_classes, all_tickers, target_vol=TARGET_VOL):
    """3-step S&P Risk Parity weight computation."""
    n_days = len(returns_history)
    if n_days < MIN_LOOKBACK_DAYS:
        return None
    lb = max(MIN_LOOKBACK_DAYS, min(n_days, MAX_LOOKBACK_DAYS))
    hist = returns_history[all_tickers].iloc[-lb:]

    # Step 1: Inverse-vol raw weights
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


def run_single_backtest(returns, all_tickers, asset_classes, weight_func, rf_daily,
                        txn_bps=TRANSACTION_COST_BPS, include_costs=True, label=''):
    """Generic backtest engine with monthly rebalancing."""
    monthly_last_days = returns.groupby(returns.index.to_period('M')).apply(lambda x: x.index[-1])
    valid_dates = [d for d in monthly_last_days if returns.index.get_loc(d) >= MIN_LOOKBACK_DAYS]

    daily_rets = []
    weights_history = []
    leverage_history = []
    cum_txn_cost = 0.0
    cum_lev_cost = 0.0
    txn_increments = {}
    current_weights = None
    prev_weights = None
    rebal_idx = 0

    for i in range(MIN_LOOKBACK_DAYS, len(returns)):
        date = returns.index[i]
        if rebal_idx < len(valid_dates) and date >= valid_dates[rebal_idx]:
            history = returns.iloc[max(0, i - MAX_LOOKBACK_DAYS):i]
            new_weights = weight_func(history, asset_classes, all_tickers, TARGET_VOL)
            if new_weights is not None:
                prev_weights = current_weights
                current_weights = new_weights
                if include_costs and prev_weights is not None:
                    turnover = sum(abs(current_weights.get(t, 0) - prev_weights.get(t, 0))
                                   for t in all_tickers)
                    txn_drag = turnover * txn_bps / 10000.0
                    cum_txn_cost += txn_drag
                    txn_increments[date] = txn_drag
                total_leverage = sum(abs(v) for v in current_weights.values())
                leverage_history.append({'date': date, 'leverage': total_leverage})
                w_record = {'date': date}
                for t in all_tickers:
                    w_record[t] = current_weights.get(t, 0)
                weights_history.append(w_record)
            rebal_idx += 1

        if current_weights is None:
            continue

        day_ret = returns.iloc[i]
        port_ret = sum(current_weights.get(t, 0) * day_ret[t] for t in all_tickers)

        if include_costs:
            total_lev = sum(abs(v) for v in current_weights.values())
            if total_lev > 1.0:
                rf_rate_today = rf_daily.iloc[i] if i < len(rf_daily) else FALLBACK_RF_RATE / 252
                daily_lev_cost = (total_lev - 1.0) * float(rf_rate_today)
                port_ret -= daily_lev_cost
                cum_lev_cost += daily_lev_cost

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
    """Run a simple fixed-weight benchmark."""
    daily_rets = []
    for i in range(len(returns)):
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


def compute_metrics(daily_returns_series, name, txn_cost=0, lev_cost=0):
    """Compute performance metrics from daily returns."""
    rets = daily_returns_series
    n_years = len(rets) / 252
    if n_years == 0:
        return {'Strategy': name, 'sharpe_float': 0, 'ann_ret': 0, 'ann_vol': 0, 'max_dd': 0}
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
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
    }

# ==============================================================================
# DATA DOWNLOAD
# ==============================================================================

def download_data():
    """Download price data for all needed tickers."""
    print("Downloading price data...")
    all_tickers = sorted(set(ALL_TICKERS_V2 + ['SPY']))  # SPY for benchmark
    tickers_to_dl = all_tickers + ['^IRX']
    data = yf.download(tickers_to_dl, start=DOWNLOAD_START, end=DOWNLOAD_END, auto_adjust=True)
    prices_all = data['Close']

    # Extract IRX
    irx_raw = prices_all['^IRX'].copy() if '^IRX' in prices_all.columns else None
    prices_etf = prices_all.drop(columns=['^IRX'], errors='ignore')

    # Print data availability for each ETF
    print("\nETF data availability:")
    for t in ALL_TICKERS_V2:
        if t in prices_etf.columns:
            valid = prices_etf[t].dropna()
            if len(valid) > 0:
                print(f"  {t:5s}: {valid.index[0].strftime('%Y-%m-%d')} to {valid.index[-1].strftime('%Y-%m-%d')} ({len(valid)} days)")
            else:
                print(f"  {t:5s}: NO DATA")
        else:
            print(f"  {t:5s}: NOT FOUND")

    # 9-ETF universe: drop rows where any of the 9 ETFs has NaN
    prices_9 = prices_etf[ALL_TICKERS_V2].dropna()
    returns_9 = prices_9.pct_change().dropna()
    common_start_9 = returns_9.index[0]
    print(f"\n9-ETF common data start: {common_start_9.strftime('%Y-%m-%d')}")
    print(f"9-ETF common data end:   {returns_9.index[-1].strftime('%Y-%m-%d')}")

    # Find when we have at least 1 year of common data for 9-ETF backtest
    backtest_start_9_idx = MIN_LOOKBACK_DAYS
    backtest_start_9 = returns_9.index[backtest_start_9_idx]
    print(f"9-ETF backtest starts:   {backtest_start_9.strftime('%Y-%m-%d')} (after 1yr lookback)")

    # 6-ETF universe: drop rows where any of the 6 ETFs has NaN
    prices_6 = prices_etf[ALL_TICKERS_6ETF].dropna()
    returns_6 = prices_6.pct_change().dropna()
    common_start_6 = returns_6.index[0]
    print(f"\n6-ETF common data start: {common_start_6.strftime('%Y-%m-%d')}")
    backtest_start_6_idx = MIN_LOOKBACK_DAYS
    backtest_start_6 = returns_6.index[backtest_start_6_idx]
    print(f"6-ETF backtest starts:   {backtest_start_6.strftime('%Y-%m-%d')} (after 1yr lookback)")

    # SPY-only returns (from earliest available)
    spy_prices = prices_etf['SPY'].dropna()
    spy_returns = spy_prices.pct_change().dropna()

    # Build risk-free rate series (use the union of all indices)
    all_idx = returns_9.index.union(returns_6.index).union(spy_returns.index)
    if irx_raw is not None and irx_raw.notna().sum() > 0:
        rf_annual = irx_raw / 100.0
        rf_daily = rf_annual.reindex(all_idx).ffill().bfill() / 252.0
        rf_daily = rf_daily.fillna(FALLBACK_RF_RATE / 252.0)
        print(f"\n^IRX data loaded: {rf_daily.notna().sum()} days")
    else:
        print(f"\n^IRX unavailable, using fallback rate of {FALLBACK_RF_RATE*100:.0f}%")
        rf_daily = pd.Series(FALLBACK_RF_RATE / 252.0, index=all_idx)

    return returns_9, returns_6, spy_returns, rf_daily

# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_chart(results, all_metrics, output_path):
    """
    GridSpec 3 rows x 2 cols, 20x20, dpi=150.
    Row 0 full: Cumulative returns with paper annotations
    Row 1 left: Sharpe bar chart, Row 1 right: Max DD bar chart
    Row 2 left: Ann Return bar chart, Row 2 right: Ann Vol bar chart
    """
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('Risk Parity — Paper Period Comparison (Dec 2003 - May 2018)',
                 fontsize=16, fontweight='bold', y=0.98)

    # ------ Row 0: Cumulative returns (full width) ------
    ax = fig.add_subplot(gs[0, :])
    for key, clr, ls, lw in [('rp9_nocost', C_RP_NOCOST, '-', 2.0),
                               ('rp9_cost', C_RP_COST, '--', 1.8),
                               ('rp6_nocost', C_6ETF, '-.', 1.5),
                               ('sixfour', C_6040, '--', 1.5),
                               ('spy', C_SPY, ':', 1.5)]:
        if key in results and len(results[key]['daily_returns']) > 0:
            cum = (1 + results[key]['daily_returns']).cumprod()
            ax.plot(cum.index, cum.values, color=clr, ls=ls, lw=lw, label=results[key]['label'])

    # Annotate paper results on the right side
    # Paper period is ~14.42 years (Dec 2003 to May 2018)
    paper_years = 14.42
    paper_rp_cum = (1 + PAPER_RP_10['return']/100) ** paper_years
    paper_6040_cum = (1 + PAPER_6040['return']/100) ** paper_years
    ax.axhline(y=paper_rp_cum, color=C_PAPER, ls='--', lw=1.0, alpha=0.7)
    ax.axhline(y=paper_6040_cum, color=C_PAPER, ls=':', lw=1.0, alpha=0.7)
    ax.annotate(f"Paper RP 10% TV: {PAPER_RP_10['return']}% ann. -> ${paper_rp_cum:.2f}",
                xy=(0.98, paper_rp_cum), xycoords=('axes fraction', 'data'),
                fontsize=9, color=C_PAPER, ha='right', va='bottom', fontweight='bold')
    ax.annotate(f"Paper 60/40: {PAPER_6040['return']}% ann. -> ${paper_6040_cum:.2f}",
                xy=(0.98, paper_6040_cum), xycoords=('axes fraction', 'data'),
                fontsize=9, color=C_PAPER, ha='right', va='bottom', fontweight='bold')

    ax.set_title('Cumulative Returns — Our Strategies vs Paper Benchmarks', fontsize=13, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ------ Helper for bar charts ------
    # Build comparison data: Paper RP, Paper 60/40, our strategies
    bar_labels = ['Paper RP\n10% TV', 'Paper 60/40',
                  'Our RP 9-ETF\n(no costs)', 'Our RP 9-ETF\n(with costs)',
                  'Our 6-ETF\n(no costs)', 'Our 60/40', 'SPY']
    bar_colors = [C_PAPER, C_PAPER, C_RP_NOCOST, C_RP_COST, C_6ETF, C_6040, C_SPY]

    # Extract metrics
    def get_m(key):
        for m in all_metrics:
            if m.get('_key') == key:
                return m
        return None

    m_rp9nc = get_m('rp9_nocost')
    m_rp9c = get_m('rp9_cost')
    m_rp6nc = get_m('rp6_nocost')
    m_6040 = get_m('sixfour')
    m_spy = get_m('spy')

    sharpe_vals = [PAPER_RP_10['sharpe'], PAPER_6040['sharpe'],
                   m_rp9nc['sharpe_float'] if m_rp9nc else 0,
                   m_rp9c['sharpe_float'] if m_rp9c else 0,
                   m_rp6nc['sharpe_float'] if m_rp6nc else 0,
                   m_6040['sharpe_float'] if m_6040 else 0,
                   m_spy['sharpe_float'] if m_spy else 0]

    maxdd_vals = [PAPER_RP_10['max_dd'], PAPER_6040['max_dd'],
                  m_rp9nc['max_dd']*100 if m_rp9nc else 0,
                  m_rp9c['max_dd']*100 if m_rp9c else 0,
                  m_rp6nc['max_dd']*100 if m_rp6nc else 0,
                  m_6040['max_dd']*100 if m_6040 else 0,
                  m_spy['max_dd']*100 if m_spy else 0]

    annret_vals = [PAPER_RP_10['return'], PAPER_6040['return'],
                   m_rp9nc['ann_ret']*100 if m_rp9nc else 0,
                   m_rp9c['ann_ret']*100 if m_rp9c else 0,
                   m_rp6nc['ann_ret']*100 if m_rp6nc else 0,
                   m_6040['ann_ret']*100 if m_6040 else 0,
                   m_spy['ann_ret']*100 if m_spy else 0]

    annvol_vals = [PAPER_RP_10['vol'], PAPER_6040['vol'],
                   m_rp9nc['ann_vol']*100 if m_rp9nc else 0,
                   m_rp9c['ann_vol']*100 if m_rp9c else 0,
                   m_rp6nc['ann_vol']*100 if m_rp6nc else 0,
                   m_6040['ann_vol']*100 if m_6040 else 0,
                   m_spy['ann_vol']*100 if m_spy else 0]

    def draw_bar(ax, vals, title, ylabel, fmt='{:.2f}'):
        x = np.arange(len(bar_labels))
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8, rotation=0, ha='center')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', lw=0.5)
        for bar, val in zip(bars, vals):
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, val, fmt.format(val),
                    ha='center', va=va, fontsize=9, fontweight='bold')

    # Row 1 left: Sharpe
    ax = fig.add_subplot(gs[1, 0])
    draw_bar(ax, sharpe_vals, 'Sharpe Ratio Comparison', 'Sharpe Ratio', fmt='{:.3f}')

    # Row 1 right: Max DD
    ax = fig.add_subplot(gs[1, 1])
    draw_bar(ax, maxdd_vals, 'Maximum Drawdown Comparison', 'Max Drawdown (%)', fmt='{:.1f}%')

    # Row 2 left: Ann Return
    ax = fig.add_subplot(gs[2, 0])
    draw_bar(ax, annret_vals, 'Annualized Return Comparison', 'Ann. Return (%)', fmt='{:.2f}%')

    # Row 2 right: Ann Vol
    ax = fig.add_subplot(gs[2, 1])
    draw_bar(ax, annvol_vals, 'Annualized Volatility Comparison', 'Ann. Volatility (%)', fmt='{:.2f}%')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nChart saved to {output_path}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    # ── Step 1: Download data ──
    returns_9, returns_6, spy_returns, rf_daily = download_data()

    results = {}
    all_metrics = []

    # ── Step 2: Run 9-ETF strategies ──
    # Trim returns_9 to end at 2018-05-31
    end_date = pd.Timestamp('2018-05-31')
    returns_9_trimmed = returns_9[returns_9.index <= end_date]

    print(f"\n9-ETF backtest period: {returns_9_trimmed.index[MIN_LOOKBACK_DAYS].strftime('%Y-%m-%d')} to {returns_9_trimmed.index[-1].strftime('%Y-%m-%d')}")

    # Strategy 1: SP3 9-ETF no costs
    print("\n[1/5] Running SP3 9-ETF (no costs)...")
    results['rp9_nocost'] = run_single_backtest(
        returns_9_trimmed, ALL_TICKERS_V2, ASSET_CLASSES_V2, compute_sp3_weights, rf_daily,
        txn_bps=0, include_costs=False, label='SP3 9-ETF (no costs)')

    # Strategy 2: SP3 9-ETF with costs
    print("[2/5] Running SP3 9-ETF (with costs)...")
    results['rp9_cost'] = run_single_backtest(
        returns_9_trimmed, ALL_TICKERS_V2, ASSET_CLASSES_V2, compute_sp3_weights, rf_daily,
        txn_bps=TRANSACTION_COST_BPS, include_costs=True, label='SP3 9-ETF (with costs)')

    # ── Step 3: Run 6-ETF strategy ──
    returns_6_trimmed = returns_6[returns_6.index <= end_date]
    print(f"\n6-ETF backtest period: {returns_6_trimmed.index[MIN_LOOKBACK_DAYS].strftime('%Y-%m-%d')} to {returns_6_trimmed.index[-1].strftime('%Y-%m-%d')}")

    # Strategy 3: SP3 6-ETF no costs
    print("[3/5] Running SP3 6-ETF Equity+FI (no costs)...")
    results['rp6_nocost'] = run_single_backtest(
        returns_6_trimmed, ALL_TICKERS_6ETF, ASSET_CLASSES_6ETF, compute_sp3_weights, rf_daily,
        txn_bps=0, include_costs=False, label='SP3 6-ETF Eq+FI (no costs)')

    # ── Step 4: Run benchmarks ──
    # 60/40 benchmark using 6-ETF universe
    print("[4/5] Running 60/40 Portfolio...")
    SIXTY_FORTY = {}
    for t in ASSET_CLASSES_6ETF['Equity']:
        SIXTY_FORTY[t] = 0.60 / len(ASSET_CLASSES_6ETF['Equity'])
    for t in ASSET_CLASSES_6ETF['Fixed Income']:
        SIXTY_FORTY[t] = 0.40 / len(ASSET_CLASSES_6ETF['Fixed Income'])
    # Use 6-ETF returns for 60/40 to match same period
    results['sixfour'] = run_benchmark(returns_6_trimmed, SIXTY_FORTY, ALL_TICKERS_6ETF, label='60/40 Portfolio')

    # SPY buy-and-hold — use raw series directly
    print("[5/5] Running SPY buy-and-hold...")
    spy_trimmed = spy_returns[(spy_returns.index >= returns_6_trimmed.index[0]) &
                               (spy_returns.index <= end_date)]
    results['spy'] = {
        'label': 'SPY Buy-and-Hold',
        'daily_returns': spy_trimmed,
        'weights_history': pd.DataFrame(),
        'leverage_history': pd.DataFrame(),
        'cum_txn_cost': 0.0,
        'cum_lev_cost': 0.0,
    }

    # ── Step 5: Compute metrics ──
    strategy_keys = ['rp9_nocost', 'rp9_cost', 'rp6_nocost', 'sixfour', 'spy']
    for key in strategy_keys:
        r = results[key]
        m = compute_metrics(r['daily_returns'], r['label'], r['cum_txn_cost'], r['cum_lev_cost'])
        m['_key'] = key
        all_metrics.append(m)

    # ── Step 6: Print comparison table ──
    print("\n" + "=" * 120)
    print("PAPER PERIOD COMPARISON: Dec 2003 - May 2018")
    print("=" * 120)

    print("\nPaper's Published Results:")
    print(f"  S&P RP 10% TV:    {PAPER_RP_10['return']:.2f}% return, {PAPER_RP_10['vol']:.2f}% vol, "
          f"{PAPER_RP_10['sharpe']:.2f} Sharpe, {PAPER_RP_10['max_dd']:.2f}% max DD")
    print(f"  60/40 Portfolio:  {PAPER_6040['return']:.2f}% return, {PAPER_6040['vol']:.2f}% vol, "
          f"{PAPER_6040['sharpe']:.2f} Sharpe, {PAPER_6040['max_dd']:.2f}% max DD")
    print(f"  HFR RP Vol 10:   {PAPER_HFR['return']:.2f}% return, {PAPER_HFR['vol']:.2f}% vol, "
          f"{PAPER_HFR['sharpe']:.2f} Sharpe, {PAPER_HFR['max_dd']:.2f}% max DD")

    print("\nOur Backtest Results:")
    display_cols = ['Strategy', 'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe', 'Sortino',
                    'Max DD (%)', 'Calmar', 'Win Rate (%)', 'Cum. Return (%)',
                    'Txn Cost (%)', 'Lev Cost (%)']
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df[display_cols].set_index('Strategy').to_string())

    # Side-by-side summary
    print("\n" + "-" * 80)
    print("SIDE-BY-SIDE SUMMARY")
    print("-" * 80)
    print(f"{'Metric':<22} {'Paper RP':>12} {'Our RP 9-ETF':>14} {'Our RP 9-ETF':>14} {'Paper 60/40':>12} {'Our 60/40':>12} {'SPY':>10}")
    print(f"{'':22} {'(10% TV)':>12} {'(no costs)':>14} {'(with costs)':>14} {'':>12} {'':>12} {'':>10}")
    print("-" * 106)

    m_nc = next(m for m in all_metrics if m['_key'] == 'rp9_nocost')
    m_c = next(m for m in all_metrics if m['_key'] == 'rp9_cost')
    m_sf = next(m for m in all_metrics if m['_key'] == 'sixfour')
    m_spy = next(m for m in all_metrics if m['_key'] == 'spy')

    print(f"{'Ann. Return (%)':<22} {PAPER_RP_10['return']:>12.2f} {m_nc['ann_ret']*100:>14.2f} {m_c['ann_ret']*100:>14.2f} {PAPER_6040['return']:>12.2f} {m_sf['ann_ret']*100:>12.2f} {m_spy['ann_ret']*100:>10.2f}")
    print(f"{'Ann. Vol (%)':<22} {PAPER_RP_10['vol']:>12.2f} {m_nc['ann_vol']*100:>14.2f} {m_c['ann_vol']*100:>14.2f} {PAPER_6040['vol']:>12.2f} {m_sf['ann_vol']*100:>12.2f} {m_spy['ann_vol']*100:>10.2f}")
    print(f"{'Sharpe':<22} {PAPER_RP_10['sharpe']:>12.3f} {m_nc['sharpe_float']:>14.3f} {m_c['sharpe_float']:>14.3f} {PAPER_6040['sharpe']:>12.3f} {m_sf['sharpe_float']:>12.3f} {m_spy['sharpe_float']:>10.3f}")
    print(f"{'Max DD (%)':<22} {PAPER_RP_10['max_dd']:>12.2f} {m_nc['max_dd']*100:>14.2f} {m_c['max_dd']*100:>14.2f} {PAPER_6040['max_dd']:>12.2f} {m_sf['max_dd']*100:>12.2f} {m_spy['max_dd']*100:>10.2f}")

    # ── Step 7: Generate chart ──
    print("\nGenerating chart...")
    plot_chart(results, all_metrics, '/workspace/output/risk_parity_paper_period.png')

    # ── Step 8: Save CSV ──
    csv_df = metrics_df.drop(columns=['sharpe_float', 'ann_ret', 'ann_vol', 'max_dd', '_key'], errors='ignore')
    csv_df.set_index('Strategy').to_csv('/workspace/output/risk_parity_paper_period_metrics.csv')
    print(f"Metrics saved to /workspace/output/risk_parity_paper_period_metrics.csv")

    print("\nPaper period backtest complete.")
