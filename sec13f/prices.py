"""Price history fetching (Section 4-5): adjusted-close daily prices for
every mapped ticker plus the SPY benchmark, cached per-ticker so a re-run
(or a later --lookback-days change within the same downloaded range) never
re-hits Yahoo Finance for a ticker already on disk.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sec13f import config

_BATCH_SIZE = 80
_INTER_BATCH_SLEEP = 1.5


def _cache_file(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return Path(cache_dir) / f"{safe}.parquet"


def _load_cached(cache_dir: Path, ticker: str) -> pd.Series | None:
    path = _cache_file(cache_dir, ticker)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df["close"]


def _save_cache(cache_dir: Path, ticker: str, close: pd.Series) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    close.to_frame("close").to_parquet(_cache_file(cache_dir, ticker))


def _covers_window(close: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if close is None or close.empty:
        return False
    return close.index.min() <= start and close.index.max() >= end


def fetch_adjusted_close(
    tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path,
    progress: bool = True,
) -> dict[str, pd.Series]:
    """Returns {ticker: adjusted-close Series indexed by date}. Tickers
    already cached with data covering [start, end] are served from disk;
    everything else is fetched from Yahoo Finance in batches via yfinance
    and then cached.
    """
    result: dict[str, pd.Series] = {}
    to_fetch: list[str] = []
    for t in tickers:
        cached = _load_cached(cache_dir, t)
        if _covers_window(cached, start, end):
            result[t] = cached
        else:
            to_fetch.append(t)

    if progress and to_fetch:
        print(f"[prices] fetching {len(to_fetch)} tickers from Yahoo Finance "
              f"(batches of {_BATCH_SIZE})...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(to_fetch), _BATCH_SIZE):
            batch = to_fetch[i:i + _BATCH_SIZE]
            try:
                data = yf.download(
                    batch, start=(start - pd.Timedelta(days=15)).date(),
                    end=(end + pd.Timedelta(days=1)).date(),
                    auto_adjust=True, progress=False, threads=True, group_by="ticker",
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"[prices] batch fetch failed ({batch[0]}..{batch[-1]}): {exc}")
                continue

            for t in batch:
                try:
                    if len(batch) == 1:
                        close = data["Close"] if "Close" in data else None
                    else:
                        close = data[t]["Close"] if t in data.columns.get_level_values(0) else None
                except Exception:  # noqa: BLE001
                    close = None
                if close is None or close.dropna().empty:
                    continue
                close = close.dropna()
                close.index = pd.to_datetime(close.index)
                _save_cache(cache_dir, t, close)
                result[t] = close

            if progress:
                print(f"[prices]   {min(i + _BATCH_SIZE, len(to_fetch))}/{len(to_fetch)}")
            time.sleep(_INTER_BATCH_SLEEP)

    return result


def build_returns_matrix(
    tickers: list[str],
    report_period: pd.Timestamp,
    lookback_days: int,
    cache_dir: Path,
    min_coverage: float = config.MIN_PRICE_HISTORY_COVERAGE,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 4: daily simple returns for every ticker (plus SPY) over the
    `lookback_days` trading days immediately preceding `report_period`.

    Returns (returns_df, coverage_df):
      returns_df   — date-indexed, one column per ticker that met
                     min_coverage of the requested lookback window.
      coverage_df  — one row per requested ticker: n_trading_days found,
                     coverage ratio, and whether it was kept or dropped.
    """
    all_tickers = sorted(set(tickers) | {config.BENCHMARK_TICKER})
    # Fetch a wide buffer of calendar days so lookback_days *trading* days
    # are available even accounting for weekends/holidays.
    calendar_buffer = int(lookback_days * 1.6) + 15
    start = report_period - pd.Timedelta(days=calendar_buffer)

    raw = fetch_adjusted_close(all_tickers, start, report_period, cache_dir, progress=progress)

    coverage_rows = []
    kept_returns = {}
    for t in all_tickers:
        close = raw.get(t)
        if close is None or close.empty:
            coverage_rows.append({"ticker": t, "n_days": 0, "coverage": 0.0, "kept": False})
            continue
        window = close[close.index <= report_period].tail(lookback_days + 1)  # +1 for pct_change lag
        n_days = max(len(window) - 1, 0)
        coverage = n_days / lookback_days
        kept = coverage >= min_coverage
        coverage_rows.append({"ticker": t, "n_days": n_days, "coverage": coverage, "kept": kept})
        if kept:
            kept_returns[t] = window.pct_change().dropna()

    coverage_df = pd.DataFrame(coverage_rows)
    if not kept_returns:
        return pd.DataFrame(), coverage_df

    returns_df = pd.DataFrame(kept_returns)
    return returns_df, coverage_df
