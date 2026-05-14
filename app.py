from __future__ import annotations

import json
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from srisk_result.analyze_portfolio_risk import (
    compute_srisk_from_portfolio,
    extract_number,
)


ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "srisk_result" / "us_market_metrics_sp500_nasdaq100.csv"
WALLSTREET_PATH = ROOT_DIR / "srisk_result" / "wallstreet_srisk_results.csv"
FINANCIAL_METRICS_PATH = ROOT_DIR / "train_and_inference" / "NASDAQ100_metrics.csv"
NEWS_DB_PATH = ROOT_DIR / "db" / "news.db"

DEFAULT_PORTFOLIO = "AAPL 10, MSFT 8, NVDA 4, AMZN 6"
PRICE_CACHE_TTL = 60
DATA_CACHE_TTL = 15 * 60
PRICE_CACHE: dict[str, tuple[float, float]] = {}
FINANCIAL_CACHE: dict[str, tuple[dict, float]] = {}
NEWS_CACHE: dict[str, tuple[list[dict], float]] = {}
DEMO_PRICES = {
    "AAPL": 190.0,
    "MSFT": 420.0,
    "NVDA": 120.0,
    "AMZN": 185.0,
}

app = Flask(__name__)


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_PATH)
    for col in ["Sigma", "MDD", "Beta"]:
        df[col] = df[col].apply(extract_number)
    return df.dropna(subset=["Ticker", "Sigma", "MDD", "Beta"]).copy()


def parse_portfolio(raw: str) -> dict[str, float]:
    positions: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.match(r"^([A-Za-z.\-]+)\s+([0-9]*\.?[0-9]+)$", item)
        if not match:
            raise ValueError(f"Could not parse item: {item}")
        ticker, quantity = match.groups()
        positions[ticker.upper().replace(".", "-")] = float(quantity)

    if not positions:
        raise ValueError("Enter at least one ticker and quantity.")

    if sum(positions.values()) <= 0:
        raise ValueError("Portfolio quantities must be greater than zero.")

    return positions


def _fetch_latest_price(ticker: str) -> float:
    cached = PRICE_CACHE.get(ticker)
    now = time.time()
    if cached and now - cached[1] < PRICE_CACHE_TTL:
        return cached[0]

    quote = yf.Ticker(ticker)
    price = None

    try:
        fast_info = quote.fast_info
        price = fast_info.get("last_price") or fast_info.get("regular_market_price")
    except Exception:
        price = None

    if price is None:
        try:
            history = quote.history(period="1d", interval="1m")
            if not history.empty:
                price = float(history["Close"].dropna().iloc[-1])
        except Exception:
            price = None

    if price is None or float(price) <= 0:
        raise ValueError(f"Could not fetch live price for {ticker}.")

    price = float(price)
    PRICE_CACHE[ticker] = (price, now)
    return price


def get_latest_prices(tickers: list[str], allow_demo: bool = False) -> tuple[dict[str, float], bool]:
    prices: dict[str, float] = {}
    used_demo = False
    missing = []

    for ticker in tickers:
        try:
            prices[ticker] = _fetch_latest_price(ticker)
        except Exception:
            if allow_demo and ticker in DEMO_PRICES:
                prices[ticker] = DEMO_PRICES[ticker]
                used_demo = True
            else:
                missing.append(ticker)

    if missing:
        raise ValueError(
            "Could not fetch live prices for: "
            + ", ".join(missing)
            + ". Check internet/Yahoo Finance access."
        )
    return prices, used_demo


def positions_to_weights(
    positions: dict[str, float], prices: dict[str, float]
) -> tuple[dict[str, float], list[dict], float]:
    values = {
        ticker: float(quantity) * float(prices[ticker])
        for ticker, quantity in positions.items()
    }
    total_value = sum(values.values())
    if total_value <= 0:
        raise ValueError("Portfolio market value must be greater than zero.")

    weights = {ticker: value / total_value for ticker, value in values.items()}
    allocation = [
        {
            "Ticker": ticker,
            "Quantity": positions[ticker],
            "Price": round(prices[ticker], 4),
            "MarketValue": round(values[ticker], 2),
            "Weight": round(weights[ticker], 4),
        }
        for ticker in sorted(weights, key=lambda t: weights[t], reverse=True)
    ]
    return weights, allocation, total_value


def format_result(
    portfolio: dict[str, float],
    metrics_df: pd.DataFrame,
    allocation: list[dict] | None = None,
    total_value: float | None = None,
) -> dict:
    missing = sorted(set(portfolio) - set(metrics_df["Ticker"].astype(str)))
    if missing:
        raise ValueError(f"Missing tickers in metrics CSV: {', '.join(missing)}")

    result = compute_srisk_from_portfolio(portfolio, metrics_df)
    holdings = metrics_df[metrics_df["Ticker"].isin(portfolio)].copy()
    holdings["Weight"] = holdings["Ticker"].map(portfolio)
    holdings = holdings.sort_values("Weight", ascending=False)

    sectors = [
        {"sector": sector, "weight": round(weight, 4)}
        for sector, weight in sorted(
            result["Sectors"].items(), key=lambda item: item[1], reverse=True
        )
    ]

    return {
        "srisk": round(float(result["Srisk"]), 4),
        "category": result["Category"],
        "sigma": round(float(result["Sigma_p"]), 4),
        "mdd": round(float(result["MDD_p"]), 4),
        "beta": round(float(result["Beta_p"]), 4),
        "hhi": round(float(result["HHI"]), 4),
        "total_value": round(float(total_value or 0), 2),
        "allocation": allocation or [],
        "sectors": sectors,
        "holdings": holdings[
            ["Ticker", "Weight", "Sector", "Sigma", "MDD", "Beta"]
        ].round(4).to_dict(orient="records"),
    }


def load_wallstreet() -> list[dict]:
    if not WALLSTREET_PATH.exists():
        return []
    df = pd.read_csv(WALLSTREET_PATH)
    return df.to_dict(orient="records")


def load_financial_metrics() -> pd.DataFrame:
    if not FINANCIAL_METRICS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(FINANCIAL_METRICS_PATH)
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper()
    return df


def _clean_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", unescape(str(value)))
    return re.sub(r"\s+", " ", text).strip()


def fetch_live_financials(ticker: str) -> dict:
    cached = FINANCIAL_CACHE.get(ticker)
    now = time.time()
    if cached and now - cached[1] < DATA_CACHE_TTL:
        return cached[0]

    info = yf.Ticker(ticker).get_info()
    snapshot = {
        "Ticker": ticker,
        "Company": info.get("longName") or info.get("shortName") or ticker,
        "Sector": info.get("sector") or "Unknown",
        "Market Cap": info.get("marketCap"),
        "Diluted EPS (ttm)": info.get("trailingEps"),
        "Profit Margin": info.get("profitMargins"),
        "Quarterly Earnings Growth (yoy)": info.get("earningsQuarterlyGrowth"),
        "Revenue Growth (yoy)": info.get("revenueGrowth"),
        "Total Debt/Equity (mrq)": info.get("debtToEquity"),
        "Current Ratio (mrq)": info.get("currentRatio"),
        "Levered Free Cash Flow (ttm)": info.get("freeCashflow"),
    }
    FINANCIAL_CACHE[ticker] = (snapshot, now)
    return snapshot


def fetch_live_news(ticker: str, limit: int = 3) -> list[dict]:
    cached = NEWS_CACHE.get(ticker)
    now = time.time()
    if cached and now - cached[1] < DATA_CACHE_TTL:
        return cached[0][:limit]

    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={quote_plus(ticker)}&region=US&lang=en-US"
    )
    response = requests.get(url, timeout=8)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    items: list[dict] = []
    for item in root.findall("./channel/item"):
        title = _clean_html(item.findtext("title"))
        summary = _clean_html(item.findtext("description"))
        pubdate = _clean_html(item.findtext("pubDate"))
        if title:
            items.append(
                {
                    "headline": title,
                    "summary": summary,
                    "pubdate": pubdate,
                    "source": "Yahoo Finance",
                }
            )
        if len(items) >= limit:
            break

    NEWS_CACHE[ticker] = (items, now)
    return items


def load_recent_news_from_db(tickers: list[str], limit_per_ticker: int = 3) -> dict[str, list[dict]]:
    news: dict[str, list[dict]] = {ticker: [] for ticker in tickers}
    if not NEWS_DB_PATH.exists() or not tickers:
        return news

    conn = sqlite3.connect(NEWS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        for ticker in tickers:
            rows = conn.execute(
                """
                SELECT headline, summary, pubdate
                FROM articles
                WHERE ticker = ?
                  AND summary IS NOT NULL
                  AND LENGTH(TRIM(summary)) > 0
                ORDER BY COALESCE(pubdate, id) DESC
                LIMIT ?
                """,
                (ticker, limit_per_ticker),
            ).fetchall()
            news[ticker] = [dict(row) for row in rows]
    finally:
        conn.close()
    return news


def load_recent_news(tickers: list[str], limit_per_ticker: int = 3) -> dict[str, list[dict]]:
    news: dict[str, list[dict]] = {ticker: [] for ticker in tickers}
    if not tickers:
        return news

    db_news = load_recent_news_from_db(tickers, limit_per_ticker)
    for ticker in tickers:
        try:
            news[ticker] = fetch_live_news(ticker, limit_per_ticker)
        except Exception:
            news[ticker] = db_news.get(ticker, [])
    return news


def _fmt_metric(value, percent: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{number * 100:.1f}%"
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    return f"{number:.2f}"


def _financial_notes(row: pd.Series | None) -> list[str]:
    if row is None:
        return ["재무 지표 데이터가 없어 뉴스와 리스크 지표 중심으로 해석합니다."]

    notes: list[str] = []
    eps = row.get("Diluted EPS (ttm)")
    profit_margin = row.get("Profit Margin")
    earnings_growth = row.get("Quarterly Earnings Growth (yoy)")
    debt_equity = row.get("Total Debt/Equity (mrq)")
    current_ratio = row.get("Current Ratio (mrq)")
    fcf = row.get("Levered Free Cash Flow (ttm)")

    if not pd.isna(eps):
        notes.append(f"EPS는 {_fmt_metric(eps)}로 수익성의 기준점 역할을 합니다.")
    if not pd.isna(profit_margin):
        notes.append(f"순이익률은 {_fmt_metric(profit_margin, percent=True)}입니다.")
    if not pd.isna(earnings_growth):
        direction = "성장" if float(earnings_growth) >= 0 else "둔화"
        notes.append(f"전년 대비 이익은 {_fmt_metric(earnings_growth, percent=True)} {direction} 흐름입니다.")
    if not pd.isna(debt_equity):
        risk = "레버리지 부담은 낮은 편" if float(debt_equity) < 1 else "부채 부담 점검이 필요"
        notes.append(f"부채비율은 {_fmt_metric(debt_equity)}로 {risk}합니다.")
    if not pd.isna(current_ratio):
        liquidity = "단기 유동성은 양호" if float(current_ratio) >= 1 else "단기 유동성은 보수적으로 볼 필요"
        notes.append(f"유동비율은 {_fmt_metric(current_ratio)}로 {liquidity}합니다.")
    if not pd.isna(fcf):
        notes.append(f"레버드 FCF는 {_fmt_metric(fcf)}입니다.")

    return notes[:5] or ["사용 가능한 핵심 재무 지표가 제한적입니다."]


def build_reports(portfolio: dict[str, float], risk_result: dict) -> list[dict]:
    tickers = list(portfolio.keys())
    finance_df = load_financial_metrics()
    finance_by_ticker = (
        finance_df.set_index("Ticker") if not finance_df.empty and "Ticker" in finance_df else pd.DataFrame()
    )
    news_by_ticker = load_recent_news(tickers)
    holding_by_ticker = {row["Ticker"]: row for row in risk_result["holdings"]}

    reports = []
    for ticker in tickers:
        finance_row = None
        if not finance_by_ticker.empty and ticker in finance_by_ticker.index:
            finance_row = finance_by_ticker.loc[ticker]
        try:
            live_finance = fetch_live_financials(ticker)
            if live_finance:
                finance_row = live_finance
        except Exception:
            pass
        holding = holding_by_ticker.get(ticker, {})
        news_items = news_by_ticker.get(ticker, [])
        sector = holding.get("Sector") or (finance_row.get("Sector") if finance_row is not None else "Unknown")

        news_lines = [
            {
                "headline": item.get("headline") or "Untitled",
                "summary": item.get("summary") or "",
                "pubdate": str(item.get("pubdate") or "")[:10],
            }
            for item in news_items
        ]

        if news_lines:
            news_impact = "최신 뉴스는 실적, 수요, 정책 또는 산업 변화가 해당 종목의 단기 리스크 해석에 반영될 수 있음을 보여줍니다."
        else:
            news_impact = "뉴스 조회 결과가 없어 현재는 재무 지표와 포트폴리오 리스크 중심으로 판단합니다."

        reports.append(
            {
                "ticker": ticker,
                "weight": round(portfolio[ticker], 4),
                "sector": sector,
                "risk": {
                    "sigma": holding.get("Sigma"),
                    "mdd": holding.get("MDD"),
                    "beta": holding.get("Beta"),
                },
                "overview": f"{ticker}는 포트폴리오의 {portfolio[ticker] * 100:.1f}%를 차지하며 {sector} 섹터 노출을 만듭니다.",
                "financial": _financial_notes(finance_row),
                "news_impact": news_impact,
                "news": news_lines,
                "outlook": "포트폴리오 전체 성향과 함께 변동성, 베타, 섹터 집중도를 같이 확인하는 것이 좋습니다.",
            }
        )
    return reports


@app.route("/")
def index():
    metrics_df = load_metrics()
    default_positions = parse_portfolio(DEFAULT_PORTFOLIO)
    prices = {ticker: DEMO_PRICES[ticker] for ticker in default_positions}
    price_warning = "Initial dashboard uses demo prices. Click Analyze to fetch live prices."
    default_weights, allocation, total_value = positions_to_weights(default_positions, prices)
    result = format_result(default_weights, metrics_df, allocation, total_value)
    result["price_warning"] = price_warning
    reports = build_reports(default_weights, result)
    top_market = (
        metrics_df.sort_values("Sigma", ascending=False)
        .head(12)[["Ticker", "Sector", "Sigma", "MDD", "Beta"]]
        .round(4)
        .to_dict(orient="records")
    )
    tickers = metrics_df["Ticker"].sort_values().tolist()
    ticker_options = (
        metrics_df[["Ticker", "Sector"]]
        .drop_duplicates()
        .sort_values("Ticker")
        .to_dict(orient="records")
    )
    return render_template(
        "index.html",
        default_portfolio=DEFAULT_PORTFOLIO,
        result=result,
        result_json=json.dumps(result),
        reports=reports,
        reports_json=json.dumps(reports),
        ticker_options_json=json.dumps(ticker_options),
        wallstreet=load_wallstreet(),
        top_market=top_market,
        ticker_count=len(tickers),
    )


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    raw = str(payload.get("portfolio", ""))
    try:
        positions = parse_portfolio(raw)
        prices, used_demo = get_latest_prices(list(positions), allow_demo=False)
        portfolio, allocation, total_value = positions_to_weights(positions, prices)
        result = format_result(portfolio, load_metrics(), allocation, total_value)
        result["price_warning"] = (
            "Demo prices were used." if used_demo else "Live prices updated."
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    result["reports"] = build_reports(portfolio, result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
