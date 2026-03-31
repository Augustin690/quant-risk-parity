"""
Risk Parity Strategy Robustness Testing v3
Builds on v2 backtest: regime stress tests, parameter sensitivity, bootstrap CIs.
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
# CONFIGURATION (mirrored from v2)
# ══════════════════════════════════════════════════════════════════════════════

ANNUALIZATION_FACTOR = np.sqrt(252)
START_DATE = '2007-01-01'
END_DATE = '2025-12-31'
TRANSACTION_COST_BPS = 5
FALLBACK_RF_RATE = 0.02
MIN_LOOKBACK_DAYS = 252

ASSET_CLASSES = {
    'Equity': ['SPY', 'EFA', 'EEM'],
    'Fixed Income': ['IEF', 'TLT', 'AGG'],
    'Commodities': ['GLD', 'DBC', 'SLV'],
}
ALL_TICKERS = [t for tickers in ASSET_CLASSES.values() for t in tickers]

REGIMES = {
    '2008 GFC': ('2007-10-01', '2009-03-31'),
    '2010 Euro Crisis': ('2010-03-01', '2010-06-30'),
    '2011 US Debt': ('2011-08-01', '2011-11-30'),
    '2013 Taper': ('2013-05-01', '2013-09-30'),
    '2015 China': ('2015-05-01', '2015-09-30'),
    '2018 Vol Shock': ('2018-01-01', '2018-03-31'),
    '2020 COVID': ('2020-02-01', '2020-03-31'),
    '2022 Rate Hike': ('2022-01-01', '2022-10-31'),
    '2023-25 Recovery': ('2023-01-01', '2025-12-31'),
}

TARGET_VOLS_GRID = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
LOOKBACK_MONTHS_GRID = [6, 12, 24, 36, 60, 120, 180]
REBAL_FREQS_GRID = ['W', 'M', 'Q']

C_RP = '#1a5276'
C_6040 = '#c0392b'

# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD (same as v2)
# ══════════════════════════════════════════════════════════════════════════════

def download_data():
    """Download price data for all tickers and ^IRX for risk-free rate."""
    print("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = data['Close'].dropna()
    returns = prices.pct_change().dropna()
    print(f"Data range: {returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Total trading days: {len(returns)}")

    print("Downloading ^IRX (3-month T-bill rate)...")
    try:
        irx_data = yf.download('^IRX', start=START_DATE, end=END_DATE, auto_adjust=True)
        if irx_data is not None and len(irx_data) > 0:
            irx_rate = irx_data['Close'].squeeze()
            irx_rate = irx_rate / 100.0
            rf_daily = irx_rate.reindex(returns.index).ffill().bfill()
            rf_daily = rf_daily / 252.0
            print(f"^IRX data loaded: {rf_daily.notna().sum()} days")
        else:
            raise ValueError("Empty IRX data")
    except Exception as e:
        print(f"^IRX download failed ({e}), using fallback rate of {FALLBACK_RF_RATE*100:.0f}%")
        rf_daily = pd.Series(FALLBACK_RF_RATE / 252.0, index=returns.index)

    return prices, returns, rf_daily

# ══════════════════════════════════════════════════════════════════════════════
# S&P 3-STEP WEIGHT FUNCTION (copied from v2)
# ══════════════════════════════════════════════════════════════════════════════

def get_vol(returns_history, ticker, lookback_days=None):
    """Compute annualized vol for a ticker using given lookback (expanding if None)."""
    if lookback_days is None:
        lb = max(MIN_LOOKBACK_DAYS, len(returns_history))
    else:
        lb = min(lookback_days, len(returns_history))
    return returns_history[ticker].iloc[-lb:].std() * ANNUALIZATION_FACTOR


def compute_sp3step_weights(returns_history, target_vol=0.10, lookback_days=None):
    """
    3-step S&P Risk Parity weight computation.
    Step 1: Instrument-level inverse-vol weighting
    Step 2: Asset-class-level multiplier to equalize risk contribution
    Step 3: Portfolio-level multiplier to hit target volatility
    """
    n_days = len(returns_history)
    min_lb = max(MIN_LOOKBACK_DAYS, 126)  # at least ~6 months
    if n_days < min_lb:
        return None

    # Step 1: Compute instrument vols
    instrument_vols = {}
    for ticker in ALL_TICKERS:
        instrument_vols[ticker] = get_vol(returns_history, ticker, lookback_days)

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
# BASELINE BACKTEST ENGINE (simplified from v2)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_backtest(returns, rf_daily, target_vol=0.10, lookback_days=None,
                          rebal_freq='M', apply_costs=True, leverage_rate=None):
    """
    Simplified backtest engine for the S&P 3-step strategy.
    Returns a dict with daily returns Series.
    leverage_rate: if given, use constant annual rate instead of rf_daily.
    """
    # Determine rebalance dates
    if rebal_freq == 'W':
        rebal_dates = returns.groupby(returns.index.to_period('W')).apply(lambda x: x.index[-1])
    elif rebal_freq == 'Q':
        rebal_dates = returns.groupby(returns.index.to_period('Q')).apply(lambda x: x.index[-1])
    else:
        rebal_dates = returns.groupby(returns.index.to_period('M')).apply(lambda x: x.index[-1])

    min_start = max(MIN_LOOKBACK_DAYS, lookback_days if lookback_days else MIN_LOOKBACK_DAYS)
    valid_dates = [d for d in rebal_dates if returns.index.get_loc(d) >= min_start]

    daily_rets = []
    current_weights = None
    prev_weights = None
    rebal_idx = 0
    cum_txn_cost = 0.0

    for i in range(min_start, len(returns)):
        date = returns.index[i]

        # Check if rebalance
        if rebal_idx < len(valid_dates) and date >= valid_dates[rebal_idx]:
            lb = lookback_days if lookback_days else (i + 1)
            history = returns.iloc[max(0, i - lb):i]
            new_weights = compute_sp3step_weights(history, target_vol=target_vol,
                                                   lookback_days=lookback_days)
            if new_weights is not None:
                prev_weights = current_weights
                current_weights = new_weights

                if apply_costs and prev_weights is not None:
                    turnover = sum(abs(current_weights.get(t, 0) - prev_weights.get(t, 0))
                                   for t in ALL_TICKERS)
                    txn_drag = turnover * TRANSACTION_COST_BPS / 10000.0
                    cum_txn_cost += txn_drag

            rebal_idx += 1

        if current_weights is None:
            continue

        day_ret = returns.iloc[i]
        port_ret = sum(current_weights[t] * day_ret[t] for t in ALL_TICKERS)

        if apply_costs:
            total_lev = sum(abs(v) for v in current_weights.values())
            if total_lev > 1.0:
                if leverage_rate is not None:
                    daily_lev_cost = (total_lev - 1.0) * (leverage_rate / 252.0)
                else:
                    rf_rate_today = rf_daily.iloc[i] if i < len(rf_daily) else FALLBACK_RF_RATE / 252
                    daily_lev_cost = (total_lev - 1.0) * rf_rate_today
                port_ret -= daily_lev_cost

        daily_rets.append({'date': date, 'return': port_ret})

    ret_df = pd.DataFrame(daily_rets).set_index('date')

    # Apply transaction cost as incremental drag
    if apply_costs and len(ret_df) > 0 and cum_txn_cost > 0:
        # Recompute incremental txn costs
        txn_costs = []
        cum = 0.0
        current_w = None
        prev_w = None
        rebal_idx2 = 0
        min_s = max(MIN_LOOKBACK_DAYS, lookback_days if lookback_days else MIN_LOOKBACK_DAYS)
        for i in range(min_s, len(returns)):
            date = returns.index[i]
            inc = 0.0
            if rebal_idx2 < len(valid_dates) and date >= valid_dates[rebal_idx2]:
                lb2 = lookback_days if lookback_days else (i + 1)
                hist2 = returns.iloc[max(0, i - lb2):i]
                nw = compute_sp3step_weights(hist2, target_vol=target_vol,
                                              lookback_days=lookback_days)
                if nw is not None:
                    prev_w = current_w
                    current_w = nw
                    if prev_w is not None:
                        to = sum(abs(current_w.get(t, 0) - prev_w.get(t, 0)) for t in ALL_TICKERS)
                        inc = to * TRANSACTION_COST_BPS / 10000.0
                rebal_idx2 += 1
            if current_w is not None:
                txn_costs.append({'date': date, 'inc': inc})

        if txn_costs:
            txn_df = pd.DataFrame(txn_costs).set_index('date')
            txn_df = txn_df.reindex(ret_df.index, fill_value=0.0)
            ret_df['return'] = ret_df['return'] - txn_df['inc']

    return ret_df['return']


def run_6040_benchmark(returns):
    """Run 60/40 benchmark, return daily returns Series."""
    SIXTY_FORTY = {}
    for t in ASSET_CLASSES['Equity']:
        SIXTY_FORTY[t] = 0.60 / len(ASSET_CLASSES['Equity'])
    for t in ASSET_CLASSES['Fixed Income']:
        SIXTY_FORTY[t] = 0.40 / len(ASSET_CLASSES['Fixed Income'])
    for t in ASSET_CLASSES['Commodities']:
        SIXTY_FORTY[t] = 0.0

    daily_rets = []
    for i in range(MIN_LOOKBACK_DAYS, len(returns)):
        date = returns.index[i]
        day_ret = returns.iloc[i]
        port_ret = sum(SIXTY_FORTY.get(t, 0) * day_ret[t] for t in ALL_TICKERS)
        daily_rets.append({'date': date, 'return': port_ret})

    ret_df = pd.DataFrame(daily_rets).set_index('date')
    return ret_df['return']

# ══════════════════════════════════════════════════════════════════════════════
# REGIME METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_period_metrics(rets_series, label=''):
    """Compute metrics for a given period's daily returns Series."""
    if len(rets_series) < 2:
        return {'label': label, 'cum_ret': np.nan, 'max_dd': np.nan,
                'vol': np.nan, 'sharpe': np.nan}
    cum = (1 + rets_series).cumprod()
    cum_ret = cum.iloc[-1] / cum.iloc[0] - 1 if cum.iloc[0] != 0 else np.nan
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    n_years = len(rets_series) / 252
    ann_vol = rets_series.std() * ANNUALIZATION_FACTOR
    ann_ret = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {'label': label, 'cum_ret': cum_ret, 'max_dd': max_dd,
            'vol': ann_vol, 'sharpe': sharpe}


def run_regime_stress_tests(rp_rets, sf_rets):
    """Run regime stress tests. Returns a DataFrame with metrics per regime."""
    rows = []
    for regime_name, (start, end) in REGIMES.items():
        rp_slice = rp_rets.loc[start:end]
        sf_slice = sf_rets.loc[start:end]

        rp_m = compute_period_metrics(rp_slice, f'RP - {regime_name}')
        sf_m = compute_period_metrics(sf_slice, f'60/40 - {regime_name}')

        rows.append({
            'Regime': regime_name,
            'RP Cum Return (%)': rp_m['cum_ret'] * 100 if not np.isnan(rp_m['cum_ret']) else np.nan,
            'RP Max DD (%)': rp_m['max_dd'] * 100 if not np.isnan(rp_m['max_dd']) else np.nan,
            'RP Vol (%)': rp_m['vol'] * 100 if not np.isnan(rp_m['vol']) else np.nan,
            'RP Sharpe': rp_m['sharpe'],
            '60/40 Cum Return (%)': sf_m['cum_ret'] * 100 if not np.isnan(sf_m['cum_ret']) else np.nan,
            '60/40 Max DD (%)': sf_m['max_dd'] * 100 if not np.isnan(sf_m['max_dd']) else np.nan,
            '60/40 Vol (%)': sf_m['vol'] * 100 if not np.isnan(sf_m['vol']) else np.nan,
            '60/40 Sharpe': sf_m['sharpe'],
            'Spread Cum Ret (%)': (rp_m['cum_ret'] - sf_m['cum_ret']) * 100 if not (np.isnan(rp_m['cum_ret']) or np.isnan(sf_m['cum_ret'])) else np.nan,
            'Spread Max DD (%)': (rp_m['max_dd'] - sf_m['max_dd']) * 100 if not (np.isnan(rp_m['max_dd']) or np.isnan(sf_m['max_dd'])) else np.nan,
            'Spread Vol (%)': (rp_m['vol'] - sf_m['vol']) * 100 if not (np.isnan(rp_m['vol']) or np.isnan(sf_m['vol'])) else np.nan,
            'Spread Sharpe': (rp_m['sharpe'] - sf_m['sharpe']) if not (np.isnan(rp_m['sharpe']) or np.isnan(sf_m['sharpe'])) else np.nan,
        })

    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def run_parameter_sensitivity(returns, rf_daily):
    """
    Grid search over target_vol x lookback_months x rebal_freq.
    Returns dict of DataFrames keyed by rebal_freq, each with
    rows=lookback_months, cols=target_vol, values=Sharpe.
    """
    results = {freq: pd.DataFrame(index=LOOKBACK_MONTHS_GRID, columns=TARGET_VOLS_GRID,
                                   dtype=float)
               for freq in REBAL_FREQS_GRID}

    total = len(TARGET_VOLS_GRID) * len(LOOKBACK_MONTHS_GRID) * len(REBAL_FREQS_GRID)
    count = 0

    for freq in REBAL_FREQS_GRID:
        for lb_months in LOOKBACK_MONTHS_GRID:
            lb_days = int(lb_months * 21)  # approximate trading days per month
            for tv in TARGET_VOLS_GRID:
                count += 1
                if count % 20 == 0:
                    print(f"  Sensitivity grid: {count}/{total}")
                try:
                    rets = run_baseline_backtest(
                        returns, rf_daily, target_vol=tv, lookback_days=lb_days,
                        rebal_freq=freq, apply_costs=True, leverage_rate=0.03)
                    if len(rets) < 252:
                        results[freq].loc[lb_months, tv] = np.nan
                        continue
                    n_years = len(rets) / 252
                    cum_ret = (1 + rets).prod()
                    ann_ret = cum_ret ** (1 / n_years) - 1
                    ann_vol = rets.std() * ANNUALIZATION_FACTOR
                    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
                    results[freq].loc[lb_months, tv] = sharpe
                except Exception:
                    results[freq].loc[lb_months, tv] = np.nan

    return results

# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════

def run_bootstrap_ci(rp_rets, sf_rets, n_bootstrap=1000, block_size=21):
    """
    Block bootstrap confidence intervals for key metrics.
    Returns dict with 'rp' and 'sf' keys, each containing arrays of metrics.
    """
    def bootstrap_metrics(rets, n_bootstrap, block_size):
        n = len(rets)
        rets_arr = rets.values
        n_blocks = n // block_size + 1

        ann_returns = []
        ann_vols = []
        sharpes = []
        max_dds = []

        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
            sample = np.concatenate([rets_arr[s:s + block_size] for s in block_starts])[:n]

            cum = np.cumprod(1 + sample)
            cum_ret = cum[-1] - 1
            n_years = n / 252
            ann_ret = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
            ann_vol = np.std(sample, ddof=1) * ANNUALIZATION_FACTOR
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_dd = np.min(dd)

            ann_returns.append(ann_ret)
            ann_vols.append(ann_vol)
            sharpes.append(sharpe)
            max_dds.append(max_dd)

        return {
            'ann_return': np.array(ann_returns),
            'ann_vol': np.array(ann_vols),
            'sharpe': np.array(sharpes),
            'max_dd': np.array(max_dds),
        }

    print("  Running bootstrap for RP...")
    rp_boot = bootstrap_metrics(rp_rets, n_bootstrap, block_size)
    print("  Running bootstrap for 60/40...")
    sf_boot = bootstrap_metrics(sf_rets, n_bootstrap, block_size)

    return {'rp': rp_boot, 'sf': sf_boot}


def bootstrap_summary_table(boot_results):
    """Create summary table of bootstrap CIs."""
    rows = []
    for strat_key, strat_name in [('rp', 'Risk Parity'), ('sf', '60/40')]:
        data = boot_results[strat_key]
        for metric_key, metric_name, mult in [
            ('ann_return', 'Ann. Return (%)', 100),
            ('ann_vol', 'Ann. Vol (%)', 100),
            ('sharpe', 'Sharpe Ratio', 1),
            ('max_dd', 'Max Drawdown (%)', 100),
        ]:
            vals = data[metric_key] * mult
            rows.append({
                'Strategy': strat_name,
                'Metric': metric_name,
                '5th pct': np.percentile(vals, 5),
                'Median': np.percentile(vals, 50),
                '95th pct': np.percentile(vals, 95),
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(regime_df, sensitivity_results, boot_results, output_path):
    """Generate the single 20x28 composite chart."""
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False

    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.25,
                  height_ratios=[1, 1, 1.2, 1, 0.1])
    fig.suptitle('Risk Parity v3 — Robustness Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # ── Row 0: Regime cumulative returns (grouped bar) ──
    ax0 = fig.add_subplot(gs[0, :])
    regimes = regime_df['Regime'].values
    x = np.arange(len(regimes))
    width = 0.35
    rp_vals = regime_df['RP Cum Return (%)'].values.astype(float)
    sf_vals = regime_df['60/40 Cum Return (%)'].values.astype(float)
    ax0.bar(x - width / 2, rp_vals, width, label='Risk Parity', color=C_RP, alpha=0.85)
    ax0.bar(x + width / 2, sf_vals, width, label='60/40', color=C_6040, alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels(regimes, rotation=30, ha='right', fontsize=9)
    ax0.set_ylabel('Cumulative Return (%)')
    ax0.set_title('Regime Stress Test — Cumulative Returns', fontsize=14, fontweight='bold')
    ax0.legend(fontsize=11)
    ax0.grid(True, alpha=0.3, axis='y')
    ax0.axhline(y=0, color='black', lw=0.5)

    # ── Row 1: Regime max drawdown (grouped bar) ──
    ax1 = fig.add_subplot(gs[1, :])
    rp_dd = regime_df['RP Max DD (%)'].values.astype(float)
    sf_dd = regime_df['60/40 Max DD (%)'].values.astype(float)
    ax1.bar(x - width / 2, rp_dd, width, label='Risk Parity', color=C_RP, alpha=0.85)
    ax1.bar(x + width / 2, sf_dd, width, label='60/40', color=C_6040, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Max Drawdown (%)')
    ax1.set_title('Regime Stress Test — Max Drawdown', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Row 2: Parameter sensitivity heatmaps (3 panels) ──
    gs_heat = gs[2, :].subgridspec(1, 3, wspace=0.3)
    freq_labels = {'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly'}

    for idx, freq in enumerate(REBAL_FREQS_GRID):
        ax_h = fig.add_subplot(gs_heat[0, idx])
        heat_data = sensitivity_results[freq].astype(float)

        # Row labels = lookback months, col labels = target vol %
        row_labels = [str(m) for m in heat_data.index]
        col_labels = [f'{v*100:.0f}%' for v in heat_data.columns]

        if HAS_SEABORN:
            sns.heatmap(heat_data.values, annot=True, fmt='.2f', cmap='RdYlGn',
                        xticklabels=col_labels, yticklabels=row_labels,
                        ax=ax_h, cbar=True, center=0.4,
                        annot_kws={'size': 8})
        else:
            im = ax_h.imshow(heat_data.values.astype(float), cmap='RdYlGn', aspect='auto')
            ax_h.set_xticks(range(len(col_labels)))
            ax_h.set_xticklabels(col_labels, fontsize=8)
            ax_h.set_yticks(range(len(row_labels)))
            ax_h.set_yticklabels(row_labels, fontsize=8)
            for r in range(heat_data.shape[0]):
                for c in range(heat_data.shape[1]):
                    val = heat_data.values[r, c]
                    if not np.isnan(val):
                        ax_h.text(c, r, f'{val:.2f}', ha='center', va='center', fontsize=7)
            fig.colorbar(im, ax=ax_h, shrink=0.8)

        ax_h.set_title(f'Sharpe — {freq_labels[freq]} Rebal', fontsize=11, fontweight='bold')
        ax_h.set_xlabel('Target Vol')
        if idx == 0:
            ax_h.set_ylabel('Lookback (months)')

    # ── Row 3 left: Bootstrap Sharpe distribution ──
    ax3l = fig.add_subplot(gs[3, 0])
    rp_sharpe = boot_results['rp']['sharpe']
    sf_sharpe = boot_results['sf']['sharpe']
    ax3l.hist(rp_sharpe, bins=50, alpha=0.6, color=C_RP, label='Risk Parity', density=True)
    ax3l.hist(sf_sharpe, bins=50, alpha=0.6, color=C_6040, label='60/40', density=True)
    # Median and CI lines
    for vals, clr, lbl in [(rp_sharpe, C_RP, 'RP'), (sf_sharpe, C_6040, '60/40')]:
        med = np.median(vals)
        p5, p95 = np.percentile(vals, [5, 95])
        ax3l.axvline(med, color=clr, lw=2, ls='-', alpha=0.9)
        ax3l.axvline(p5, color=clr, lw=1, ls='--', alpha=0.7)
        ax3l.axvline(p95, color=clr, lw=1, ls='--', alpha=0.7)
    ax3l.set_title('Bootstrap Distribution — Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3l.set_xlabel('Sharpe Ratio')
    ax3l.set_ylabel('Density')
    ax3l.legend(fontsize=10)
    ax3l.grid(True, alpha=0.3)

    # ── Row 3 right: Bootstrap Max DD distribution ──
    ax3r = fig.add_subplot(gs[3, 1])
    rp_dd_boot = boot_results['rp']['max_dd'] * 100
    sf_dd_boot = boot_results['sf']['max_dd'] * 100
    ax3r.hist(rp_dd_boot, bins=50, alpha=0.6, color=C_RP, label='Risk Parity', density=True)
    ax3r.hist(sf_dd_boot, bins=50, alpha=0.6, color=C_6040, label='60/40', density=True)
    for vals, clr in [(rp_dd_boot, C_RP), (sf_dd_boot, C_6040)]:
        med = np.median(vals)
        p5, p95 = np.percentile(vals, [5, 95])
        ax3r.axvline(med, color=clr, lw=2, ls='-', alpha=0.9)
        ax3r.axvline(p5, color=clr, lw=1, ls='--', alpha=0.7)
        ax3r.axvline(p95, color=clr, lw=1, ls='--', alpha=0.7)
    ax3r.set_title('Bootstrap Distribution — Max Drawdown', fontsize=12, fontweight='bold')
    ax3r.set_xlabel('Max Drawdown (%)')
    ax3r.set_ylabel('Density')
    ax3r.legend(fontsize=10)
    ax3r.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart saved to {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Step 1: Download data ──
    prices, returns, rf_daily = download_data()

    # ── Step 2: Run baseline strategies ──
    print("\n[1/4] Running baseline Risk Parity backtest (SP3 Expand 10% with costs)...")
    rp_rets = run_baseline_backtest(returns, rf_daily, target_vol=0.10,
                                     lookback_days=None, rebal_freq='M',
                                     apply_costs=True)
    print(f"  RP returns: {len(rp_rets)} days")

    print("[2/4] Running 60/40 benchmark...")
    sf_rets = run_6040_benchmark(returns)
    print(f"  60/40 returns: {len(sf_rets)} days")

    # Align indices
    common_idx = rp_rets.index.intersection(sf_rets.index)
    rp_rets = rp_rets.loc[common_idx]
    sf_rets = sf_rets.loc[common_idx]

    # ── Step 3: Regime Stress Tests ──
    print("\n[3/4] Running regime stress tests...")
    regime_df = run_regime_stress_tests(rp_rets, sf_rets)

    print("\n" + "=" * 120)
    print("REGIME STRESS TEST RESULTS")
    print("=" * 120)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 200)
    print(regime_df.set_index('Regime').to_string(float_format=lambda x: f'{x:.2f}'))

    # ── Step 4: Parameter Sensitivity ──
    print("\n[4/4] Running parameter sensitivity grid search...")
    sensitivity_results = run_parameter_sensitivity(returns, rf_daily)

    # Print top 10 and bottom 10 by Sharpe
    all_combos = []
    for freq in REBAL_FREQS_GRID:
        df = sensitivity_results[freq]
        for lb in df.index:
            for tv in df.columns:
                val = df.loc[lb, tv]
                if not np.isnan(val):
                    all_combos.append({
                        'Rebal': freq, 'Lookback (mo)': lb,
                        'Target Vol': f'{tv*100:.0f}%', 'Sharpe': val
                    })
    combos_df = pd.DataFrame(all_combos).sort_values('Sharpe', ascending=False)

    print("\n" + "=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS BY SHARPE")
    print("=" * 80)
    print(combos_df.head(10).to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    print("\n" + "=" * 80)
    print("BOTTOM 10 PARAMETER COMBINATIONS BY SHARPE")
    print("=" * 80)
    print(combos_df.tail(10).to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # ── Step 5: Bootstrap Confidence Intervals ──
    print("\nRunning block bootstrap (1000 samples, block=21 days)...")
    boot_results = run_bootstrap_ci(rp_rets, sf_rets, n_bootstrap=1000, block_size=21)

    boot_table = bootstrap_summary_table(boot_results)
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    print(boot_table.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # ── Step 6: Generate chart ──
    print("\nGenerating robustness chart...")
    plot_all(regime_df, sensitivity_results, boot_results,
             '/workspace/output/risk_parity_v3_robustness.png')

    # ── Step 7: Save metrics CSV ──
    regime_df.to_csv('/workspace/output/risk_parity_v3_robustness_metrics.csv', index=False)
    print("Regime metrics saved to /workspace/output/risk_parity_v3_robustness_metrics.csv")

    print("\nRobustness analysis v3 complete.")
