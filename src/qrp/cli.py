from __future__ import annotations
import typer
from pathlib import Path
import pandas as pd
from .config import Config
from . import data as D
from . import backtest as BT
from . import metrics as M

app = typer.Typer(help="Risk Parity (ERC) backtester with target vol and costs.")

@app.command()
def fetch(start: str = None, end: str = None):
    cfg = Config()
    start = start or cfg.start
    end = end or cfg.end
    prices = D.fetch_prices(cfg.tickers, start, end)
    path = D.cache_prices(prices)
    typer.echo(f"Saved: {path}")

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

    prices_path = Path('data/prices.parquet')
    prices = D.load_prices() if prices_path.exists() else D.fetch_prices(cfg.tickers, start, end)
    outputs = Path("outputs"); outputs.mkdir(exist_ok=True, parents=True)

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
        baseline_prices = D.fetch_baseline_data(start, end)
        # Ensure proper date filtering - baseline data should already be in the right range
        baseline_W, baseline_R = BT.run_baseline(baseline_prices)
        
        # Save baseline results
        baseline_W.to_csv(outputs / "baseline_weights.csv")
        baseline_R.to_csv(outputs / "baseline_returns.csv", header=["ret"])
        print("Baseline calculation completed successfully")
    except Exception as e:
        print(f"Warning: Baseline calculation failed: {e}")
        baseline_R = None
    
    # Save main strategy results
    W.to_csv(outputs / "weights.csv")
    R.to_csv(outputs / "portfolio_returns.csv", header=["ret"])
    T.to_csv(outputs / "turnover.csv")
    typer.echo(f"Saved outputs to {outputs}")

@app.command()
def report(split_date: str = "2022-01-01"):
    import matplotlib.pyplot as plt
    from . import visualize as V
    
    outputs = Path("outputs")
    R = pd.read_csv(outputs / "portfolio_returns.csv", index_col=0, parse_dates=True)["ret"]
    W = pd.read_csv(outputs / "weights.csv", index_col=0, parse_dates=True)
    T = pd.read_csv(outputs / "turnover.csv", index_col=0, parse_dates=True)["turnover"]
    
    # Load baseline data if available
    baseline_R = None
    baseline_path = outputs / "baseline_returns.csv"
    if baseline_path.exists():
        baseline_R = pd.read_csv(baseline_path, index_col=0, parse_dates=True)["ret"]
        typer.echo("Baseline data found - including in analysis")
    else:
        typer.echo("No baseline data found - generating strategy-only report")
    
    # Generate performance summary with baseline comparison if available
    if baseline_R is not None:
        summary = M.summarize_with_baseline(R, baseline_R, split_date)
        summary.to_csv(outputs / "baseline_summary.csv")
        typer.echo("Strategy vs Baseline Performance Summary:")
        typer.echo(summary.to_string())
    else:
        summary = M.summarize(R, split_date)
        summary.to_csv(outputs / "summary.csv")
        typer.echo("Strategy Performance Summary:")
        typer.echo(summary.to_string())

    # Create comprehensive dashboard with baseline if available
    if baseline_R is not None:
        fig = V.create_dashboard(W, R, T, split_date, baseline_returns=baseline_R)
    else:
        fig = V.create_dashboard(W, R, T, split_date)
    fig.savefig(outputs / "dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved dashboard to {outputs / 'dashboard.png'}")

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
    
    fig_simple.savefig(outputs / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig_simple)

def main():
    app()

if __name__ == "__main__":
    main()
