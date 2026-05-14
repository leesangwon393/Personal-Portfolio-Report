from __future__ import annotations

import argparse
from pathlib import Path

from yfinance_metrics import collect_yfinance_metrics, load_sp500_tickers


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "S&P500_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update S&P 500 fundamental metrics with yfinance."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()

    collect_yfinance_metrics(
        Path(args.output),
        label="S&P 500",
        ticker_loader=load_sp500_tickers,
        workers=args.workers,
        limit=args.limit,
        fallback_tickers_csv=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    main()
