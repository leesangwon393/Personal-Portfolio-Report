#!/usr/bin/env python3
"""CLI entrypoint for the SEC 13F risk-clustering pipeline.

Example:
    python scripts/run_13f_clustering.py \\
        --quarter 2026-06-30 \\
        --num-portfolios 100 \\
        --lookback-days 252 \\
        --output-dir artifacts/13f_clustering

See README's "13F 위험 운용성향 군집화" section (or docs/13F_CLUSTERING.md)
for the full methodology, environment variables, and result-interpretation
caveats.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sec13f import config
from sec13f.pipeline import run_pipeline
from sec13f.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quarter", default="2026-06-30",
                         help="Target 13F report period (YYYY-MM-DD). Falls back to the newest "
                              "fully-published quarter if this one isn't available yet.")
    parser.add_argument("--num-portfolios", type=int, default=config.NUM_PORTFOLIOS)
    parser.add_argument("--lookback-days", type=int, default=config.DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--output-dir", default=str(config.DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--selection-method", choices=config.SELECTION_METHODS, default=config.SELECTION_TOP_VALUE,
                         help="top_value = largest reported total value (default). "
                              "aum_stratified = sample proportionally across AUM quantile bins.")
    parser.add_argument("--candidate-pool-multiplier", type=int, default=3,
                         help="How many extra ranked candidates (as a multiple of --num-portfolios) "
                              "to keep on hand so portfolios excluded for low mapping/price coverage "
                              "can be backfilled from the next-ranked candidate.")
    parser.add_argument("--scaler", choices=["robust", "standard"], default="robust")
    parser.add_argument("--k", type=int, default=config.PREFERRED_K,
                         help="Number of clusters for the FINAL saved model (all of K=2..6 are still "
                              "evaluated and reported regardless of this value).")
    args = parser.parse_args()

    t0 = time.time()
    result = run_pipeline(
        quarter=args.quarter,
        num_portfolios=args.num_portfolios,
        lookback_days=args.lookback_days,
        output_dir=Path(args.output_dir),
        selection_method=args.selection_method,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
        scaler_kind=args.scaler,
        fit_k=args.k,
    )
    report_path = write_report(result, args)
    elapsed = time.time() - t0

    print(f"\n[13f] Done in {elapsed/60:.1f} min. {result.n_selected} portfolios clustered "
          f"into {result.chosen_k} clusters ({result.chosen_method}).")
    print(f"[13f] Report: {report_path}")


if __name__ == "__main__":
    main()
