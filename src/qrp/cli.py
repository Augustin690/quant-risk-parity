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
    ewma_window: int = 60,
):
    cfg = Config()
    start = start or cfg.start
    end = end or cfg.end
    rebalance = rebalance or cfg.rebalance
    target_vol = target_vol or cfg.target_vol_annual

    prices_path = Path('data/prices.parquet')
    prices = D.load_prices() if prices_path.exists() else D.fetch_prices(cfg.tickers, start, end)
    outputs = Path("outputs"); outputs.mkdir(exist_ok=True, parents=True)

    W, R, T = BT.run_backtest(
        prices.loc[start:end],
        rebalance=rebalance,
        ewma_window=ewma_window,
        target_vol_annual=target_vol,
        cost_bps_per_trade=cfg.cost_bps_per_trade,
        slippage_bps_per_turnover=cfg.slippage_bps_per_turnover,
    )
    W.to_csv(outputs / "weights.csv")
    R.to_csv(outputs / "portfolio_returns.csv", header=["ret"])
    T.to_csv(outputs / "turnover.csv")
    typer.echo(f"Saved outputs to {outputs}")

@app.command()
def report(split_date: str = "2022-01-01"):
    import matplotlib.pyplot as plt
    outputs = Path("outputs")
    R = pd.read_csv(outputs / "portfolio_returns.csv", index_col=0, parse_dates=True)["ret"]
    summary = M.summarize(R, split_date)
    summary.to_csv(outputs / "summary.csv")
    typer.echo(summary.to_string())

    eq = (1+R).cumprod()
    fig = plt.figure()
    eq.plot(title="Equity Curve")
    fig.savefig(outputs / "equity_curve.png", dpi=150, bbox_inches="tight")

def main():
    app()

if __name__ == "__main__":
    main()
