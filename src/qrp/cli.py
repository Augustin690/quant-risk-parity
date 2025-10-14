from __future__ import annotations
import typer
from pathlib import Path
import pandas as pd
from .config import Config
from . import data as D
from . import backtest as BT
from . import metrics as M
from .paths import *
from .data import load_or_fetch_and_cache, force_fetch_and_cache

app = typer.Typer(help="Risk Parity (ERC) backtester with target vol and costs.")

def validate_flag(value: str):
    if value is None:
        raise typer.BadParameter("Flag is required")
    if value not in ["baseline", "portfolio", "all"]:
        raise typer.BadParameter(f"Invalid flag: {value}. Must be 'baseline', 'portfolio', or 'all'.")
    return value

## caching helpers moved to cache_utils

@app.command()
def fetch(start: str = None, end: str = None, flag: str = typer.Option(None, "--flag", callback=validate_flag)):
    """Fetches price data for configured tickers and saves it to disk.

        This function retrieves price data for the specified date range and stores it in a cache file.

        Args:
            start: Optional start date for fetching prices. If not provided, uses config default.
            end: Optional end date for fetching prices. If not provided, uses config default.
            flag: Optional flag for data fetching, either "baseline", "portfolio", "all"

        Returns:
            None
    """

    cfg = Config()
    start = start or cfg.start
    end = end or cfg.end
    if flag == "baseline":
        force_fetch_and_cache(D.fetch_baseline_data, BASELINE_CACHE, start, end)
    elif flag == "portfolio":
        force_fetch_and_cache(D.fetch_prices, PRICES_CACHE, start, end, fetch_args=[cfg.tickers])
    elif flag == "all":
        force_fetch_and_cache(D.fetch_baseline_data, BASELINE_CACHE, start, end)
        force_fetch_and_cache(D.fetch_prices, PRICES_CACHE, start, end, fetch_args=[cfg.tickers])

@app.command()
def run(
    start: str = None,
    end: str = None,
    rebalance: str = None,
    target_vol: float = None,
    risk_estim_strat: str = None,
    rolling_window: int = None,
    ewma_halflife_days: int = None,
):
    cfg = Config()
    start = start or cfg.start
    end = end or cfg.end
    rebalance = rebalance or cfg.rebalance
    target_vol = target_vol or cfg.target_vol_annual
    ewma_halflife_days = ewma_halflife_days or cfg.ewma_halflife_days
    risk_estim_strat = risk_estim_strat or cfg.risk_estim_strat
    rolling_window = rolling_window or cfg.rolling_window_days

    print(f"Running backtest with:")
    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Rebalance: {rebalance}")
    print(f"Target Vol: {target_vol}")
    print(f"Risk Estim Strat: {risk_estim_strat}")
    print(f"Ewma Halflife Days: {ewma_halflife_days}")
    print(f"Rolling Window: {rolling_window}")

    prices = load_or_fetch_and_cache(D.fetch_prices, PRICES_CACHE, start, end, fetch_args=[cfg.tickers])
    outputs = Path(OUTPUT_DIR); outputs.mkdir(exist_ok=True, parents=True)

    # Run main strategy backtest
    W, R, T = BT.run_backtest(
        prices.loc[start:end],
        rebalance=rebalance,
        rolling_window=rolling_window,
        target_vol_annual=target_vol,
        cost_bps_per_trade=cfg.cost_bps_per_trade,
        slippage_bps_per_turnover=cfg.slippage_bps_per_turnover,
        risk_estim_strat=risk_estim_strat,
        ewma_halflife_days=ewma_halflife_days,
    )
    
    # Fetch and run baseline portfolio
    print("Running 60/40 baseline...")
    try:
        baseline_prices = load_or_fetch_and_cache(D.fetch_baseline_data, BASELINE_CACHE, start, end)
        # Ensure proper date filtering - baseline data should already be in the right range
        baseline_W, baseline_R = BT.run_baseline(baseline_prices)
        
        # Save baseline results
        baseline_W.to_csv(outputs / BASELINE_WEIGHTS_FILE)
        baseline_R.to_csv(outputs / BASELINE_RETURNS_FILE, header=["ret"])
        print("Baseline calculation completed successfully")
    except Exception as e:
        print(f"Warning: Baseline calculation failed: {e}")
        baseline_R = None
    
    # Save main strategy results
    W.to_csv(outputs / WEIGHTS_FILE)
    R.to_csv(outputs / PORTFOLIO_RETURNS_FILE, header=["ret"])
    T.to_csv(outputs / TURNOVER_FILE)
    typer.echo(f"Saved outputs to {outputs}")

@app.command()
def report(split_date: str = "2022-01-01"):
    import matplotlib.pyplot as plt
    from . import visualize as V
    
    outputs = Path(OUTPUT_DIR)
    R = pd.read_csv(outputs / PORTFOLIO_RETURNS_FILE, index_col=0, parse_dates=True)["ret"]
    W = pd.read_csv(outputs / WEIGHTS_FILE, index_col=0, parse_dates=True)
    T = pd.read_csv(outputs / TURNOVER_FILE, index_col=0, parse_dates=True)["turnover"]
    
    # Load baseline data if available
    baseline_R = None
    baseline_path = outputs / BASELINE_RETURNS_FILE
    if baseline_path.exists():
        baseline_R = pd.read_csv(baseline_path, index_col=0, parse_dates=True)["ret"]
        typer.echo("Baseline data found - including in analysis")
    else:
        typer.echo("No baseline data found - generating strategy-only report")
    
    # Generate performance summary with baseline comparison if available
    if baseline_R is not None:
        summary = M.summarize_with_baseline(R, baseline_R, split_date)
        summary.to_csv(outputs / BASELINE_SUMMARY_FILE)
        typer.echo("Strategy vs Baseline Performance Summary:")
        typer.echo(summary.to_string())
    else:
        summary = M.summarize(R, split_date)
        summary.to_csv(outputs / SUMMARY_FILE)
        typer.echo("Strategy Performance Summary:")
        typer.echo(summary.to_string())

    # Create comprehensive dashboard with baseline if available
    if baseline_R is not None:
        fig = V.create_dashboard(W, R, T, split_date, baseline_returns=baseline_R)
    else:
        fig = V.create_dashboard(W, R, T, split_date)
    fig.savefig(outputs / DASHBOARD_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved dashboard to {outputs / DASHBOARD_FILE}")

    # Keep simple equity curve for backward compatibility with baseline if available
    fig_simple = plt.figure(figsize=(10, 6))
    
    if baseline_R is not None:
        # Align both series to common date range
        common_dates = R.index.intersection(baseline_R.index)
        if len(common_dates) == 0:
            typer.echo("Warning: No overlapping dates between strategy and baseline")
            eq = (1+R).cumprod()
            eq.plot(title="Equity Curve", label="Strategy")
        else:
            # Both start at 1.0 on the first common date
            eq = (1+R.loc[common_dates]).cumprod()
            baseline_eq = (1+baseline_R.loc[common_dates]).cumprod()
            
            eq.plot(title="Equity Curve", label="Strategy")
            baseline_eq.plot(label="60/40 Baseline", 
                            linestyle="--", color="gray", alpha=0.8)
            plt.legend()
    else:
        eq = (1+R).cumprod()
        eq.plot(title="Equity Curve", label="Strategy")
    
    fig_simple.savefig(outputs / EQUITY_CURVE_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig_simple)

def main():
    app()

if __name__ == "__main__":
    main()
