"""CUSIP -> ticker mapping for 13F holdings (Section 3).

Priority order (per spec):
  1. A trusted mapping already cached in this repo's artifacts (reused
     across runs so we never re-ask OpenFIGI for a CUSIP we've resolved
     before) — this repo had no pre-existing CUSIP->ticker mapping data
     before this pipeline (checked: no cusip-keyed CSV/JSON anywhere in the
     repo), so this reduces to "our own cache", not a pre-shipped dataset.
  2. OpenFIGI /v3/mapping (with OPENFIGI_API_KEY if set, else the public
     unauthenticated endpoint, which is real and works but is rate-limited
     harder).
  3. (no further fallback implemented) — CUSIPs OpenFIGI can't resolve are
     recorded as unmapped, never guessed from issuer-name similarity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from secrets_config import get_openfigi_api_key

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
# OpenFIGI enforces different limits per request depending on whether a key
# is presented: unauthenticated requests are capped at 10 mapping jobs per
# call (empirically confirmed: "413 Request may only contain 10 mapping
# jobs"); with an API key the cap is 100.
OPENFIGI_BATCH_SIZE_NO_KEY = 10
OPENFIGI_BATCH_SIZE_WITH_KEY = 100
# OpenFIGI's documented anonymous rate limit is 25 requests/minute; with an
# API key it's 25 requests/6 seconds. Staying comfortably under either.
_RATE_LIMIT_SLEEP_NO_KEY = 2.6
_RATE_LIMIT_SLEEP_WITH_KEY = 0.3
MAX_RETRIES = 6

# Preference order for which OpenFIGI result row to use when a CUSIP maps to
# several listings (composite vs. individual exchanges): a primary US
# composite listing is what a US-market price lookup (yfinance) needs.
_PREFERRED_EXCH_CODES = ["US", "UN", "UW", "UQ"]


def _cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "cusip_ticker_map.json"


def _load_cache(cache_dir: Path) -> dict:
    path = _cache_path(cache_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache_dir: Path, cache: dict) -> None:
    path = _cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")


def _pick_best_row(rows: list[dict]) -> dict | None:
    equities = [r for r in rows if r.get("marketSector") == "Equity" and r.get("ticker")]
    if not equities:
        return None
    for exch in _PREFERRED_EXCH_CODES:
        for r in equities:
            if r.get("exchCode") == exch:
                return r
    return equities[0]


def _query_openfigi_batch(cusips: list[str], api_key: str | None) -> dict[str, dict]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    jobs = [{"idType": "ID_CUSIP", "idValue": c} for c in cusips]

    last_exc: Exception | None = None
    results = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPENFIGI_MAPPING_URL, headers=headers, json=jobs, timeout=45)
        except requests.exceptions.RequestException as exc:
            # Transient network failure (read timeout, connection reset, DNS
            # blip, ...) — retry with backoff rather than crash the whole
            # pipeline mid-mapping. Real, observed in practice: OpenFIGI's
            # API occasionally times out under sustained unauthenticated
            # traffic.
            last_exc = exc
            time.sleep(3 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code >= 500:
            last_exc = RuntimeError(f"OpenFIGI HTTP {resp.status_code}")
            time.sleep(3 * (attempt + 1))
            continue
        resp.raise_for_status()
        results = resp.json()
        break
    if results is None:
        raise RuntimeError(
            f"OpenFIGI unreachable/rate-limited after {MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    out: dict[str, dict] = {}
    for cusip, result in zip(cusips, results):
        if "data" in result and result["data"]:
            best = _pick_best_row(result["data"])
            if best:
                out[cusip] = {
                    "ticker": best.get("ticker"),
                    "figi": best.get("figi"),
                    "name": best.get("name"),
                    "exch_code": best.get("exchCode"),
                    "mapped": True,
                    "unmapped_reason": None,
                }
            else:
                out[cusip] = {
                    "ticker": None, "figi": None, "name": None, "exch_code": None,
                    "mapped": False, "unmapped_reason": "no_equity_result",
                }
        else:
            out[cusip] = {
                "ticker": None, "figi": None, "name": None, "exch_code": None,
                "mapped": False,
                "unmapped_reason": result.get("error", "not_found"),
            }
    return out


def map_cusips_to_tickers(cusips: list[str], cache_dir: Path, progress: bool = True) -> pd.DataFrame:
    """Resolves each unique CUSIP to a ticker via OpenFIGI, using a local
    JSON cache so repeat pipeline runs never re-query a CUSIP already
    resolved. Returns one row per input CUSIP.
    """
    unique_cusips = sorted({c for c in cusips if isinstance(c, str) and c.strip()})
    cache = _load_cache(cache_dir)
    api_key = get_openfigi_api_key()
    sleep_s = _RATE_LIMIT_SLEEP_WITH_KEY if api_key else _RATE_LIMIT_SLEEP_NO_KEY
    batch_size = OPENFIGI_BATCH_SIZE_WITH_KEY if api_key else OPENFIGI_BATCH_SIZE_NO_KEY

    to_fetch = [c for c in unique_cusips if c not in cache]
    if progress and to_fetch:
        print(
            f"[mapping] {len(to_fetch)} CUSIPs not in cache, querying OpenFIGI "
            f"({'with' if api_key else 'without'} API key, batches of {batch_size})..."
        )

    failed_batches = 0
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        try:
            result = _query_openfigi_batch(batch, api_key)
        except Exception as exc:  # noqa: BLE001
            # A single batch failing after every retry (e.g. a sustained
            # OpenFIGI outage) shouldn't discard the CUSIPs resolved so far
            # in this run — record these as unmapped-for-this-reason and
            # keep going; a re-run will retry them since they're absent
            # from the cache.
            failed_batches += 1
            if progress:
                print(f"[mapping]   batch {i}-{i+len(batch)} failed permanently: {exc}")
            continue
        cache.update(result)
        _save_cache(cache_dir, cache)  # persist incrementally in case of interruption
        if progress:
            done = min(i + batch_size, len(to_fetch))
            print(f"[mapping]   {done}/{len(to_fetch)}")
        time.sleep(sleep_s)

    if progress and failed_batches:
        print(f"[mapping] {failed_batches} batch(es) failed permanently after retries "
              f"(their CUSIPs are 'not_queried' below; re-running will retry them).")

    rows = []
    for cusip in unique_cusips:
        entry = cache.get(cusip, {
            "ticker": None, "figi": None, "name": None, "exch_code": None,
            "mapped": False, "unmapped_reason": "not_queried",
        })
        rows.append({"cusip": cusip, **entry})
    return pd.DataFrame(rows)
