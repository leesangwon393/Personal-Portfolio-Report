from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secrets_config import get_fmp_api_key


BASE_URL = "https://financialmodelingprep.com/stable"

COLUMNS = [
    "Ticker",
    "Sector",
    "Industry",
    "Profit Margin",
    "Operating Margin (ttm)",
    "Return on Assets(ROA) (ttm)",
    "Return on Equity(ROE) (ttm)",
    "Revenue (ttm)",
    "Revenue Per Share (ttm)",
    "Quarterly Revenue Growth (yoy)",
    "Gross Profit (ttm)",
    "EBITDA",
    "Net Income Avi to Common (ttm)",
    "Diluted EPS (ttm)",
    "Quarterly Earnings Growth (yoy)",
    "Total Cash (mrq)",
    "Total Cash Per Share (mrq)",
    "Total Debt (mrq)",
    "Total Debt/Equity (mrq)",
    "Current Ratio (mrq)",
    "Book Value Per Share (mrq)",
    "Operating Cash Flow (ttm)",
    "Levered Free Cash Flow (ttm)",
]


def _record(payload):
    if isinstance(payload, list):
        return payload[0] if payload else {}
    return payload if isinstance(payload, dict) else {}


def _pick(*records, keys: list[str] | tuple[str, ...]):
    for record in records:
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
    return None


def _ratio(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if abs(value) > 3:
        return value / 100.0
    return value


def _per_share(numerator, shares):
    try:
        if numerator is None or not shares:
            return None
        return float(numerator) / float(shares)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def fmp_get(session: requests.Session, endpoint: str, api_key: str, **params):
    params = {k: v for k, v in params.items() if v is not None}
    params["apikey"] = api_key
    response = session.get(f"{BASE_URL}/{endpoint.lstrip('/')}", params=params, timeout=12)
    if response.status_code >= 400:
        raise RuntimeError(f"FMP {endpoint} HTTP {response.status_code}: {response.text[:220]}")
    payload = response.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(f"FMP {endpoint}: {payload['Error Message']}")
    return payload


def fmp_get_optional(session: requests.Session, endpoint: str, api_key: str, **params):
    try:
        return fmp_get(session, endpoint, api_key, **params)
    except Exception:
        return []


def yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def load_index_tickers(index: str, api_key: str) -> list[str]:
    endpoint_by_index = {
        "sp500": "sp500-constituent",
        "nasdaq100": "nasdaq-constituent",
    }
    endpoint = endpoint_by_index[index]
    with requests.Session() as session:
        payload = fmp_get(session, endpoint, api_key)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected FMP {endpoint} response: {type(payload).__name__}")
    tickers = []
    for row in payload:
        symbol = row.get("symbol") if isinstance(row, dict) else None
        if symbol:
            tickers.append(yahoo_symbol(symbol))
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise RuntimeError(f"FMP {endpoint} returned no symbols.")
    return tickers


def load_tickers_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise RuntimeError(f"Ticker column not found in fallback CSV: {path}")
    tickers = [yahoo_symbol(ticker) for ticker in df["Ticker"].dropna().astype(str).tolist()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        raise RuntimeError(f"No tickers found in fallback CSV: {path}")
    return tickers


def _fetch_one(session: requests.Session, ticker: str, api_key: str, retries: int = 0) -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            profile = _record(fmp_get_optional(session, "profile", api_key, symbol=ticker))
            ratios = _record(fmp_get_optional(session, "ratios-ttm", api_key, symbol=ticker))
            metrics = _record(fmp_get_optional(session, "key-metrics-ttm", api_key, symbol=ticker))
            income = _record(
                fmp_get_optional(session, "income-statement", api_key, symbol=ticker, period="annual", limit=1)
            )
            balance = _record(
                fmp_get_optional(session, "balance-sheet-statement", api_key, symbol=ticker, period="annual", limit=1)
            )
            cashflow = _record(
                fmp_get_optional(session, "cash-flow-statement", api_key, symbol=ticker, period="annual", limit=1)
            )
            growth = _record(
                fmp_get_optional(session, "financial-growth", api_key, symbol=ticker, period="quarter", limit=1)
            )

            if not any([profile, ratios, metrics, income, balance, cashflow, growth]):
                raise RuntimeError("all FMP endpoints returned no usable data")

            shares = _pick(
                income,
                balance,
                metrics,
                profile,
                keys=("weightedAverageShsOutDil", "weightedAverageShsOut", "sharesOutstanding", "weightedAverageSharesDiluted"),
            )
            revenue = _pick(income, metrics, keys=("revenue", "revenueTTM"))
            total_cash = _pick(
                balance,
                metrics,
                keys=("cashAndCashEquivalents", "cashAndShortTermInvestments", "cashAndCashEquivalentsTTM"),
            )
            total_debt = _pick(balance, metrics, keys=("totalDebt", "totalDebtTTM"))
            total_equity = _pick(balance, keys=("totalStockholdersEquity", "totalEquity", "totalStockholdersEquityAndTotalLiabilities"))

            return {
                "Ticker": ticker,
                "Sector": _pick(profile, keys=("sector",)),
                "Industry": _pick(profile, keys=("industry",)),
                "Profit Margin": _ratio(
                    _pick(ratios, metrics, keys=("netProfitMarginTTM", "netProfitMargin", "profitMarginTTM"))
                ),
                "Operating Margin (ttm)": _ratio(
                    _pick(ratios, metrics, keys=("operatingProfitMarginTTM", "operatingProfitMargin"))
                ),
                "Return on Assets(ROA) (ttm)": _ratio(
                    _pick(ratios, metrics, keys=("returnOnAssetsTTM", "returnOnAssets"))
                ),
                "Return on Equity(ROE) (ttm)": _ratio(
                    _pick(ratios, metrics, keys=("returnOnEquityTTM", "returnOnEquity"))
                ),
                "Revenue (ttm)": revenue,
                "Revenue Per Share (ttm)": _pick(
                    metrics, keys=("revenuePerShareTTM", "revenuePerShare")
                )
                or _per_share(revenue, shares),
                "Quarterly Revenue Growth (yoy)": _ratio(
                    _pick(growth, keys=("revenueGrowth", "growthRevenue"))
                ),
                "Gross Profit (ttm)": _pick(income, keys=("grossProfit", "grossProfitTTM")),
                "EBITDA": _pick(income, metrics, keys=("ebitda", "ebitdaTTM")),
                "Net Income Avi to Common (ttm)": _pick(
                    income, keys=("netIncome", "netIncomeCommonStockholders", "netIncomeTTM")
                ),
                "Diluted EPS (ttm)": _pick(income, metrics, keys=("epsdiluted", "epsDiluted", "netIncomePerShareTTM")),
                "Quarterly Earnings Growth (yoy)": _ratio(
                    _pick(growth, keys=("netIncomeGrowth", "growthNetIncome"))
                ),
                "Total Cash (mrq)": total_cash,
                "Total Cash Per Share (mrq)": _pick(
                    metrics, keys=("cashPerShareTTM", "cashPerShare")
                )
                or _per_share(total_cash, shares),
                "Total Debt (mrq)": total_debt,
                "Total Debt/Equity (mrq)": _pick(
                    ratios, metrics, keys=("debtEquityRatioTTM", "debtToEquityTTM", "debtToEquity")
                )
                or _per_share(total_debt, total_equity),
                "Current Ratio (mrq)": _pick(ratios, metrics, keys=("currentRatioTTM", "currentRatio")),
                "Book Value Per Share (mrq)": _pick(
                    metrics, keys=("bookValuePerShareTTM", "bookValuePerShare")
                )
                or _per_share(total_equity, shares),
                "Operating Cash Flow (ttm)": _pick(
                    cashflow, keys=("netCashProvidedByOperatingActivities", "operatingCashFlow", "operatingCashFlowTTM")
                ),
                "Levered Free Cash Flow (ttm)": _pick(
                    cashflow, metrics, ratios, keys=("freeCashFlow", "freeCashFlowTTM", "freeCashFlowPerShareTTM")
                ),
            }
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    return {"Ticker": ticker, "_error": str(last_error)}


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def collect_fmp_metrics(
    output: Path,
    index: str,
    label: str,
    workers: int = 4,
    limit: int | None = None,
    api_key: str | None = None,
    fallback_tickers_csv: Path | None = None,
) -> pd.DataFrame:
    api_key = api_key or get_fmp_api_key(required=True)
    try:
        tickers = load_index_tickers(index, api_key)
    except Exception as exc:
        if not fallback_tickers_csv:
            raise
        print(f"FMP constituent endpoint unavailable ({exc}); using fallback tickers: {fallback_tickers_csv}")
        tickers = load_tickers_from_csv(fallback_tickers_csv)
    if limit:
        tickers = tickers[: int(limit)]
    print(f"{label} tickers loaded from FMP: {len(tickers)}")

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        with requests.Session() as session:
            futures = {
                executor.submit(_fetch_one, session, ticker, api_key): ticker for ticker in tickers
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                ticker = futures[future]
                row = future.result()
                if "_error" in row:
                    failures.append(row)
                    print(f"[{idx}/{len(tickers)}] failed {ticker}: {row['_error']}")
                else:
                    rows.append(row)
                    if idx == 1 or idx % 25 == 0 or idx == len(tickers):
                        print(f"[{idx}/{len(tickers)}] collected {ticker}")

    df = pd.DataFrame(rows)
    if df.empty:
        existing_path = fallback_tickers_csv if fallback_tickers_csv and fallback_tickers_csv.exists() else output
        if existing_path.exists():
            print("No new FMP metrics were collected; keeping existing CSV unchanged.")
            if failures:
                failed_path = output.with_name(f"{output.stem}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                pd.DataFrame(failures).to_csv(failed_path, index=False)
                print(f"Failures saved: {failed_path} ({len(failures)} rows)")
            return pd.read_csv(existing_path)
        raise RuntimeError("No FMP metrics were collected; existing CSV was not overwritten.")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS].sort_values("Ticker").reset_index(drop=True)

    existing_path = fallback_tickers_csv if fallback_tickers_csv and fallback_tickers_csv.exists() else output
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        if "Ticker" in existing.columns:
            for col in COLUMNS:
                if col not in existing.columns:
                    existing[col] = None
            existing = existing[COLUMNS]
            missing = existing[~existing["Ticker"].astype(str).isin(df["Ticker"].astype(str))]
            if not missing.empty:
                df = (
                    pd.concat([df, missing], ignore_index=True)
                    .drop_duplicates(subset=["Ticker"], keep="first")
                    .sort_values("Ticker")
                    .reset_index(drop=True)
                )
                print(f"Kept existing rows for {len(missing)} tickers blocked by FMP plan/rate limits.")

    output.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_existing(output)
    if backup:
        print(f"Backup saved: {backup}")
    df.to_csv(output, index=False)
    print(f"Saved: {output} ({len(df)} rows)")

    if failures:
        failed_path = output.with_name(f"{output.stem}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        pd.DataFrame(failures).to_csv(failed_path, index=False)
        print(f"Failures saved: {failed_path} ({len(failures)} rows)")

    return df
