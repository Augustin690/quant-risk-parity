from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_DIR = "output"

PRICES_CACHE = "prices.parquet"
BASELINE_CACHE = "baseline_prices.parquet"

WEIGHTS_FILE = "weights.csv"
PORTFOLIO_RETURNS_FILE = "portfolio_returns.csv"
TURNOVER_FILE = "turnover.csv"
BASELINE_WEIGHTS_FILE = "baseline_weights.csv"
BASELINE_RETURNS_FILE = "baseline_returns.csv"
BASELINE_SUMMARY_FILE = "baseline_summary.csv"
SUMMARY_FILE = "summary.csv"
DASHBOARD_FILE = "dashboard.png"
EQUITY_CURVE_FILE = "equity_curve.png"
