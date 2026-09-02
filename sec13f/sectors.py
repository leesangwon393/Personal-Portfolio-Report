"""Sector lookup per ticker (Section 5, Sector HHI).

Built on the same yfinance `.info["sector"]` pattern already used in
`data/yfinance_metrics.py`, but with a 3-way outcome instead of a 2-way one:

  - a real sector string (equity with a GICS-style sector)
  - FUND_SECTOR_LABEL — a *confirmed* non-equity holding (ETF/mutual fund/
    closed-end fund), which genuinely has no GICS sector in yfinance; this
    is real information, not a failure, and is tracked as its own bucket
    rather than dumped into "Unknown"
  - UNKNOWN_SECTOR — yfinance returned a real (non-empty) response with no
    sector and no recognizable fund quoteType (e.g. some ADRs/preferreds/
    SPACs), OR the ticker could not be resolved at all after retries

The distinction matters: an EMPTY `.info` response from yfinance is Yahoo's
rate-limiting/blocking signature, not "this ticker has no sector" — the
first version of this module conflated the two and, under sustained
ThreadPoolExecutor load across ~7,000 tickers, ended up caching real
equities (confirmed independently to have a sector, e.g. AAON ->
Industrials) as permanently "Unknown" because a transient block was taken
as a final answer. This version never caches an empty/failed response —
only a genuine non-empty API response is written to the cache — so a
transient failure is retried (with real exponential backoff) rather than
poisoning the cache forever.
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yfinance as yf

UNKNOWN_SECTOR = "Unknown"
FUND_SECTOR_LABEL = "ETF/Fund (no GICS sector)"
_FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "CLOSEDEND", "INDEX"}

_FAILED = "__LOOKUP_FAILED__"  # never written to the cache; signals "retry me"


def _cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "ticker_sector_map.json"


def _load_cache(cache_dir: Path) -> dict:
    path = _cache_path(cache_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache_dir: Path, cache: dict) -> None:
    path = _cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")


def _fetch_one_sector(ticker: str, retries: int = 5, base_pause: float = 1.0) -> str:
    """Returns a real sector, FUND_SECTOR_LABEL, UNKNOWN_SECTOR, or _FAILED.
    Only the first three are safe to cache; _FAILED means "still couldn't
    get a real response after retries" and must be retried on a later run,
    not written to disk as if it were confirmed.
    """
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(ticker).info
        except Exception:  # noqa: BLE001
            info = None
        if info:  # a genuinely non-empty response — trust it, don't retry
            sector = info.get("sector")
            if sector:
                return sector
            quote_type = (info.get("quoteType") or "").upper()
            if quote_type in _FUND_QUOTE_TYPES:
                return FUND_SECTOR_LABEL
            return UNKNOWN_SECTOR
        # Empty dict / exception: Yahoo's rate-limit/block signature, not a
        # real "no sector" answer — back off (exponential + jitter) and retry.
        if attempt < retries:
            time.sleep(base_pause * (2 ** attempt) + random.uniform(0, 0.5))
    return _FAILED


def lookup_sectors(
    tickers: list[str], cache_dir: Path, workers: int = 4, progress: bool = True
) -> dict[str, str]:
    """Returns {ticker: sector}. Tickers that could not be resolved after
    retries fall back to UNKNOWN_SECTOR for THIS run's HHI computation (so
    the pipeline can still proceed), but are deliberately left out of the
    cache so a future run retries them instead of trusting a transient
    failure forever.
    """
    unique = sorted(set(tickers))
    cache = _load_cache(cache_dir)
    to_fetch = [t for t in unique if t not in cache]

    if progress and to_fetch:
        print(f"[sectors] looking up {len(to_fetch)} tickers via yfinance "
              f"({workers} workers, exponential backoff on empty/failed responses)...")

    n_failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_fetch_one_sector, t): t for t in to_fetch}
        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            result = future.result()
            done += 1
            if result == _FAILED:
                n_failed += 1
            else:
                cache[ticker] = result
            if progress and (done == 1 or done % 25 == 0 or done == len(to_fetch)):
                print(f"[sectors]   {done}/{len(to_fetch)}"
                      + (f" ({n_failed} still failed, will retry next run)" if n_failed else ""))

    if to_fetch:
        _save_cache(cache_dir, cache)
    if progress and n_failed:
        print(f"[sectors] {n_failed} tickers still unresolved after retries "
              f"(treated as {UNKNOWN_SECTOR!r} for this run's HHI, NOT cached — re-run to retry them)")

    return {t: cache.get(t, UNKNOWN_SECTOR) for t in unique}
