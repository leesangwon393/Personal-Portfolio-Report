from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


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

FIELD_MAP = {
    "Sector": "sector",
    "Industry": "industry",
    "Profit Margin": "profitMargins",
    "Operating Margin (ttm)": "operatingMargins",
    "Return on Assets(ROA) (ttm)": "returnOnAssets",
    "Return on Equity(ROE) (ttm)": "returnOnEquity",
    "Revenue (ttm)": "totalRevenue",
    "Revenue Per Share (ttm)": "revenuePerShare",
    "Quarterly Revenue Growth (yoy)": "revenueGrowth",
    "Gross Profit (ttm)": "grossProfits",
    "EBITDA": "ebitda",
    "Net Income Avi to Common (ttm)": "netIncomeToCommon",
    "Diluted EPS (ttm)": "trailingEps",
    "Quarterly Earnings Growth (yoy)": "earningsGrowth",
    "Total Cash (mrq)": "totalCash",
    "Total Cash Per Share (mrq)": "totalCashPerShare",
    "Total Debt (mrq)": "totalDebt",
    "Total Debt/Equity (mrq)": "debtToEquity",
    "Current Ratio (mrq)": "currentRatio",
    "Book Value Per Share (mrq)": "bookValue",
    "Operating Cash Flow (ttm)": "operatingCashflow",
    "Levered Free Cash Flow (ttm)": "freeCashflow",
}


def yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def _table_by_id(url: str, table_id: str):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": table_id})
    if table is None:
        raise RuntimeError(f"Could not find table {table_id} at {url}")
    return table


def load_sp500_tickers() -> list[str]:
    table = _table_by_id("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "constituents")
    tickers = []
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if cells:
            tickers.append(yahoo_symbol(cells[0].get_text(strip=True)))
    return _dedupe_or_raise(tickers, "S&P 500")


def load_nasdaq100_tickers() -> list[str]:
    html = requests.get(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    ).text
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        ticker_idx = next((i for i, h in enumerate(headers) if h in {"ticker", "ticker symbol"}), None)
        if ticker_idx is None:
            continue
        tickers = []
        for row in table.select("tbody tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) > ticker_idx:
                tickers.append(yahoo_symbol(cells[ticker_idx].get_text(strip=True)))
        return _dedupe_or_raise(tickers, "Nasdaq-100")
    raise RuntimeError("Could not find Nasdaq-100 ticker table.")


def load_tickers_from_csv(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise RuntimeError(f"Ticker column not found: {path}")
    return _dedupe_or_raise(df["Ticker"].dropna().astype(str).tolist(), str(path))


def _dedupe_or_raise(tickers, label: str) -> list[str]:
    blocked = {"SYMBOL", "TICKER", "TICKER SYMBOL", "COMPANY", "SECURITY"}
    values = [yahoo_symbol(t) for t in tickers if str(t).strip() and str(t).strip().upper() not in blocked]
    values = list(dict.fromkeys(values))
    if not values:
        raise RuntimeError(f"No tickers loaded for {label}.")
    return values


def clean_value(value):
    if isinstance(value, (list, tuple, dict)):
        return None
    return value


def fetch_one(ticker: str, retries: int = 1, pause: float = 0.35) -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(ticker).info or {}
            if not info:
                raise RuntimeError("empty yfinance info")
            row = {"Ticker": ticker}
            for out_col, info_key in FIELD_MAP.items():
                row[out_col] = clean_value(info.get(info_key))
            return row
        except Exception as exc:
            last_error = exc
            time.sleep(pause * (attempt + 1))
    return {"Ticker": ticker, "_error": str(last_error)}


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def collect_yfinance_metrics(
    output: Path,
    label: str,
    ticker_loader,
    workers: int = 8,
    limit: int | None = None,
    fallback_tickers_csv: Path | None = None,
) -> pd.DataFrame:
    try:
        tickers = ticker_loader()
    except Exception as exc:
        if not fallback_tickers_csv:
            raise
        print(f"{label} ticker source failed ({exc}); using fallback tickers: {fallback_tickers_csv}")
        tickers = load_tickers_from_csv(fallback_tickers_csv)
    if limit:
        tickers = tickers[: int(limit)]
    print(f"{label} tickers loaded: {len(tickers)}")

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_one, ticker): ticker for ticker in tickers}
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
        raise RuntimeError("No yfinance metrics were collected; existing CSV was not overwritten.")

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
                print(f"Kept existing rows for {len(missing)} tickers that failed to refresh.")

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
