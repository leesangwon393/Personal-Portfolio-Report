from __future__ import annotations

import argparse
from pathlib import Path

from yfinance_metrics import (
    collect_yfinance_metrics,
    load_nasdaq100_tickers,
    load_sp500_tickers,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SP500_OUTPUT = ROOT_DIR / "data" / "S&P500_metrics.csv"
NASDAQ100_OUTPUT = ROOT_DIR / "train_and_inference" / "NASDAQ100_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update S&P 500 and Nasdaq-100 financial metrics in one run."
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per index.")
    args = parser.parse_args()

    collect_yfinance_metrics(
        SP500_OUTPUT,
        label="S&P 500",
        ticker_loader=load_sp500_tickers,
        workers=args.workers,
        limit=args.limit,
        fallback_tickers_csv=SP500_OUTPUT,
    )
    collect_yfinance_metrics(
        NASDAQ100_OUTPUT,
        label="Nasdaq-100",
        ticker_loader=load_nasdaq100_tickers,
        workers=args.workers,
        limit=args.limit,
        fallback_tickers_csv=NASDAQ100_OUTPUT,
    )


if __name__ == "__main__":
    main()
