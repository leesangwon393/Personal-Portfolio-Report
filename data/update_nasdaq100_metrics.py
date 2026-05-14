from __future__ import annotations

import argparse
from pathlib import Path

from yfinance_metrics import collect_yfinance_metrics, load_nasdaq100_tickers


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "train_and_inference" / "NASDAQ100_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update Nasdaq-100 fundamental metrics with yfinance."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()

    collect_yfinance_metrics(
        Path(args.output),
        label="Nasdaq-100",
        ticker_loader=load_nasdaq100_tickers,
        workers=args.workers,
        limit=args.limit,
        fallback_tickers_csv=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    main()
