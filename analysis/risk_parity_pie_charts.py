"""
Risk Parity Pie Chart Snapshots
Generates a single figure with pie charts showing portfolio allocation
at key moments: Dec 2008, Dec 2015, Dec 2022.
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
# CONFIGURATION (copied from risk_parity_v5_expanded.py)
# ==============================================================================

TARGET_VOL = 0.10
MIN_LOOKBACK_DAYS = 252
MAX_LOOKBACK_DAYS = 252 * 15
ANNUALIZATION_FACTOR = np.sqrt(252)
START_DATE = '2007-01-01'
END_DATE = '2025-12-31'
FALLBACK_RF_RATE = 0.02

ASSET_CLASSES = {
    'Equity': ['SPY', 'FEZ', 'EWJ'],
    'Fixed Income': ['IEI', 'IEF', 'TLT', 'BWX'],
    'Commodities': ['USO', 'UNG', 'UGA', 'DBC', 'GLD', 'SLV', 'DBA'],
}
ALL_TICKERS = ['SPY', 'FEZ', 'EWJ', 'IEI', 'IEF', 'TLT', 'BWX',
               'USO', 'UNG', 'UGA', 'DBC', 'GLD', 'SLV', 'DBA']

# Per-ETF colors by asset class
TICKER_COLORS = {
    'SPY': '#1a5276', 'FEZ': '#2e86c1', 'EWJ': '#5dade2',
    'IEI': '#1e8449', 'IEF': '#28b463', 'TLT': '#82e0aa', 'BWX': '#abebc6',
    'USO': '#d35400', 'UNG': '#e67e22', 'UGA': '#f39c12', 'DBC': '#f5b041',
    'GLD': '#f7dc6f', 'SLV': '#fad7a0', 'DBA': '#fdebd0',
}

# Asset class level colors
C_EQUITY = '#2e86c1'
C_FI = '#28b463'
C_COMM = '#f39c12'

# Snapshot dates (year, month)
SNAPSHOTS = [
    (2009, 12, 'Dec 2009'),
    (2015, 12, 'Dec 2015'),
    (2022, 12, 'Dec 2022'),
]

# ==============================================================================
# DATA DOWNLOAD
# ==============================================================================

def download_data():
    """Download price data for all 14 ETFs + ^IRX."""
    tickers_to_dl = ALL_TICKERS + ['^IRX']
    data = yf.download(tickers_to_dl, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices_all = data['Close']

    irx_raw = prices_all['^IRX'].copy() if '^IRX' in prices_all.columns else None
    prices_etf = prices_all.drop(columns=['^IRX'], errors='ignore')
    prices_etf = prices_etf.dropna()
    returns = prices_etf.pct_change().dropna()
    print(f"Data range: {returns.index[0].date()} to {returns.index[-1].date()}, {len(returns)} days")

    if irx_raw is not None and irx_raw.notna().sum() > 0:
        rf_annual = irx_raw / 100.0
        rf_daily = rf_annual.reindex(returns.index).ffill().bfill() / 252.0
        rf_daily = rf_daily.fillna(FALLBACK_RF_RATE / 252.0)
    else:
        rf_daily = pd.Series(FALLBACK_RF_RATE / 252.0, index=returns.index)

    return returns, rf_daily

# ==============================================================================
# WEIGHT COMPUTATION: S&P 3-STEP METHOD
# ==============================================================================

def compute_sp3_weights(returns_history):
    """3-step S&P Risk Parity weight computation for 14 ETFs."""
    n_days = len(returns_history)
    if n_days < MIN_LOOKBACK_DAYS:
        return None

    lb = max(MIN_LOOKBACK_DAYS, min(n_days, MAX_LOOKBACK_DAYS))
    hist = returns_history[ALL_TICKERS].iloc[-lb:]

    # Step 1: inverse-vol raw weights
    raw_weights = {}
    for ticker in ALL_TICKERS:
        vol = hist[ticker].std() * ANNUALIZATION_FACTOR
        raw_weights[ticker] = TARGET_VOL / vol if vol > 0 else 0

    # Step 2: asset-class-level multiplier
    asset_class_weights = {}
    for ac_name, tickers in ASSET_CLASSES.items():
        n_inst = len(tickers)
        ac_ticker_weights = {t: raw_weights[t] / n_inst for t in tickers}
        ac_returns = sum(ac_ticker_weights[t] * hist[t] for t in tickers)
        ac_vol = ac_returns.std() * ANNUALIZATION_FACTOR
        multiplier = TARGET_VOL / ac_vol if ac_vol > 0 else 1.0
        for t in tickers:
            asset_class_weights[t] = ac_ticker_weights[t] * multiplier

    # Step 3: portfolio-level multiplier
    n_classes = len(ASSET_CLASSES)
    portfolio_weights = {t: asset_class_weights[t] / n_classes for t in ALL_TICKERS}
    port_returns = sum(portfolio_weights[t] * hist[t] for t in ALL_TICKERS)
    port_vol = port_returns.std() * ANNUALIZATION_FACTOR
    port_multiplier = TARGET_VOL / port_vol if port_vol > 0 else 1.0
    final_weights = {t: portfolio_weights[t] * port_multiplier for t in ALL_TICKERS}

    return final_weights

# ==============================================================================
# RISK CONTRIBUTION COMPUTATION
# ==============================================================================

def compute_risk_contributions(weights_dict, returns_history, tickers=None):
    """
    Compute marginal risk contribution for each instrument as % of portfolio vol.
    Returns dict {ticker: pct_contribution}. Negative contributions clamped to 0.
    """
    if tickers is None:
        tickers = ALL_TICKERS
    w = np.array([weights_dict.get(t, 0.0) for t in tickers])
    lb = max(MIN_LOOKBACK_DAYS, min(len(returns_history), MAX_LOOKBACK_DAYS))
    hist = returns_history[tickers].iloc[-lb:]
    cov = hist.cov().values * 252
    port_var = w @ cov @ w
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10
    mrc = cov @ w
    rc = w * mrc
    # Clamp negatives to zero for pie chart display
    rc = np.maximum(rc, 0)
    total_rc = rc.sum()
    if abs(total_rc) < 1e-12:
        return {t: 0.0 for t in tickers}
    pct = {tickers[i]: rc[i] / total_rc * 100 for i in range(len(tickers))}
    return pct

# ==============================================================================
# HELPER: find snapshot dates and compute weights/RC
# ==============================================================================

def prepare_snapshot_data(returns):
    """
    For each snapshot date, find the last trading day of that month,
    compute SP3 weights and risk contributions.
    Returns list of dicts with keys: label, weights, capital_pct, risk_pct.
    """
    monthly_last = returns.groupby(returns.index.to_period('M')).apply(lambda x: x.index[-1])

    results = []
    for year, month, label in SNAPSHOTS:
        target_period = pd.Period(f'{year}-{month:02d}', freq='M')
        if target_period not in monthly_last.index:
            print(f"WARNING: {label} not found in data, skipping")
            continue
        snap_date = monthly_last[target_period]
        snap_loc = returns.index.get_loc(snap_date)
        history = returns.iloc[:snap_loc + 1]

        weights = compute_sp3_weights(history)
        if weights is None:
            print(f"WARNING: Could not compute weights for {label}")
            continue

        # Capital allocation: % of total absolute weight
        total_abs = sum(abs(v) for v in weights.values())
        capital_pct = {t: abs(weights[t]) / total_abs * 100 for t in ALL_TICKERS}

        # Risk contribution
        risk_pct = compute_risk_contributions(weights, history)

        results.append({
            'label': label,
            'date': snap_date,
            'weights': weights,
            'capital_pct': capital_pct,
            'risk_pct': risk_pct,
            'history': history,
        })

    return results

# ==============================================================================
# PIE CHART HELPERS
# ==============================================================================

def draw_pie(ax, pct_dict, tickers, title, min_label_pct=3.0):
    """Draw a single pie chart on the given axes."""
    values = [max(pct_dict.get(t, 0), 0) for t in tickers]
    colors = [TICKER_COLORS[t] for t in tickers]
    labels = [f"{t}\n{pct_dict.get(t, 0):.1f}%" if pct_dict.get(t, 0) >= min_label_pct else ''
              for t in tickers]

    total = sum(values)
    if total <= 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return

    # Normalize
    values_norm = [v / total * 100 for v in values]

    def autopct_func(pct):
        return f'{pct:.1f}%' if pct >= min_label_pct else ''

    wedges, texts, autotexts = ax.pie(
        values_norm, labels=labels, autopct=autopct_func,
        pctdistance=0.75, startangle=90, colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 8},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title(title, fontsize=11, fontweight='bold')


def draw_ac_pie(ax, pct_dict_or_values, title, ac_names=None):
    """Draw an asset-class-level pie (3 slices)."""
    if ac_names is None:
        ac_names = ['Equity', 'Fixed Income', 'Commodities']
    colors = [C_EQUITY, C_FI, C_COMM]

    if isinstance(pct_dict_or_values, dict):
        values = [max(pct_dict_or_values.get(n, 0), 0) for n in ac_names]
    else:
        values = [max(v, 0) for v in pct_dict_or_values]

    total = sum(values)
    if total <= 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return

    values_norm = [v / total * 100 for v in values]
    labels = [f"{n}\n{v:.1f}%" for n, v in zip(ac_names, values_norm)]

    wedges, texts, autotexts = ax.pie(
        values_norm, labels=labels, autopct=lambda p: f'{p:.1f}%',
        pctdistance=0.75, startangle=90, colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontsize=11, fontweight='bold')

# ==============================================================================
# MAIN FIGURE GENERATION
# ==============================================================================

def aggregate_by_ac(pct_dict):
    """Aggregate per-ticker percentages into asset class totals."""
    ac_totals = {}
    for ac_name, tickers in ASSET_CLASSES.items():
        ac_totals[ac_name] = sum(pct_dict.get(t, 0) for t in tickers)
    return ac_totals


def plot_all_pies(snapshot_data, returns):
    """Generate the full 4-row x 3-col pie chart figure."""
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('Risk Parity Portfolio — Pie Chart Snapshots',
                 fontsize=18, fontweight='bold', y=0.98)

    # ── Row 0: Capital Allocation at 3 snapshots ──
    for col, snap in enumerate(snapshot_data):
        ax = fig.add_subplot(gs[0, col])
        draw_pie(ax, snap['capital_pct'], ALL_TICKERS,
                 f"Capital Allocation — {snap['label']}")

    # ── Row 1: Risk Contribution at 3 snapshots ──
    for col, snap in enumerate(snapshot_data):
        ax = fig.add_subplot(gs[1, col])
        draw_pie(ax, snap['risk_pct'], ALL_TICKERS,
                 f"Risk Contribution — {snap['label']}")

    # ── Row 2: Asset Class Level comparison ──
    # Use Dec 2015 snapshot (index 1) for the AC-level capital & risk pies
    # Left: Capital by AC, Middle: Risk by AC, Right: Ideal target
    mid_snap = snapshot_data[1]  # Dec 2015
    ac_capital = aggregate_by_ac(mid_snap['capital_pct'])
    ac_risk = aggregate_by_ac(mid_snap['risk_pct'])

    ax = fig.add_subplot(gs[2, 0])
    draw_ac_pie(ax, ac_capital, f"Capital Allocation by Class — {mid_snap['label']}")

    ax = fig.add_subplot(gs[2, 1])
    draw_ac_pie(ax, ac_risk, f"Risk Contribution by Class — {mid_snap['label']}")

    ax = fig.add_subplot(gs[2, 2])
    draw_ac_pie(ax, [33.33, 33.33, 33.33], "Ideal Target (Equal Risk)")

    # ── Row 3: 60/40 vs Risk Parity risk contribution (Dec 2015) ──
    # 60/40 weights
    weights_6040 = {}
    for t in ALL_TICKERS:
        weights_6040[t] = 0.0
    weights_6040['SPY'] = 0.20
    weights_6040['FEZ'] = 0.20
    weights_6040['EWJ'] = 0.20
    weights_6040['IEI'] = 0.10
    weights_6040['IEF'] = 0.10
    weights_6040['TLT'] = 0.10
    weights_6040['BWX'] = 0.10

    rc_6040 = compute_risk_contributions(weights_6040, mid_snap['history'])
    ac_rc_6040 = aggregate_by_ac(rc_6040)

    ac_rc_rp = aggregate_by_ac(mid_snap['risk_pct'])

    ax = fig.add_subplot(gs[3, 0])
    draw_ac_pie(ax, ac_rc_6040, "60/40 Risk Contribution — Dec 2015")

    ax = fig.add_subplot(gs[3, 1])
    draw_ac_pie(ax, ac_rc_rp, "Risk Parity Risk Contribution — Dec 2015")

    # Row 3 col 2: leave empty or add annotation
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    ax.text(0.5, 0.6,
            "Key Insight from the Paper:\n\n"
            "A traditional 60/40 portfolio\n"
            "concentrates nearly all risk\n"
            "in equities, despite the\n"
            "40% bond allocation.\n\n"
            "Risk Parity equalizes risk\n"
            "contributions across asset\n"
            "classes, producing a truly\n"
            "diversified portfolio.",
            ha='center', va='center', fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', alpha=0.9))

    plt.savefig('/workspace/output/risk_parity_pie_snapshots.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Figure saved to /workspace/output/risk_parity_pie_snapshots.png")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    print("Downloading data...")
    returns, rf_daily = download_data()

    print("Computing snapshot data...")
    snapshot_data = prepare_snapshot_data(returns)

    print("Generating pie chart figure...")
    plot_all_pies(snapshot_data, returns)

    print("Done. Output: /workspace/output/risk_parity_pie_snapshots.png")
