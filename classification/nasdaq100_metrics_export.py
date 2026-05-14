import requests
import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm

# ===== 1. 나스닥100 티커 리스트 가져오기 (SlickCharts 크롤링) =====
url = "https://www.slickcharts.com/nasdaq100"
html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
dfs = pd.read_html(html)
nasdaq100 = dfs[0]  # 첫 번째 테이블
tickers = nasdaq100["Symbol"].tolist()

print("NASDAQ 100 종목 수:", len(tickers))
print("예시:", tickers[:10])

# ===== 2. 가격 시계열 가져오기 (Close 또는 Adj Close fallback) =====
def get_price_series(tickers, benchmark="^NDX", start="2023-01-01", end="2024-01-01"):
    data = yf.download([*tickers, benchmark], start=start, end=end).dropna()
    if "Adj Close" in data.columns.get_level_values(0):
        return data["Adj Close"]
    elif "Close" in data.columns.get_level_values(0):
        return data["Close"]
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' found")

# ===== 3. 개별 티커 지표 계산 =====
def calc_metrics(ticker, prices, benchmark="^NDX"):
    if ticker not in prices or benchmark not in prices:
        return None

    stock = prices[ticker].dropna()
    market = prices[benchmark].dropna()

    # 수익률
    stock_ret = stock.pct_change().dropna()
    market_ret = market.pct_change().dropna()

    # σ (연간 변동성)
    sigma = stock_ret.std() * np.sqrt(252)

    # MDD (최대 낙폭)
    cummax = stock.cummax()
    drawdown = (stock - cummax) / cummax
    mdd = drawdown.min()

    # β (시장 베타)
    X = sm.add_constant(market_ret)
    y = stock_ret.reindex_like(X).dropna()
    X = X.loc[y.index]
    try:
        beta = sm.OLS(y, X).fit().params[1]
    except Exception:
        beta = np.nan

    # 섹터 정보
    try:
        sector = yf.Ticker(ticker).info.get("sector", "Unknown")
    except Exception:
        sector = "Unknown"

    return {
        "Ticker": ticker,
        "Sigma": sigma,
        "MDD": mdd,
        "Beta": beta,
        "Sector": sector
    }

# ===== 4. 실행 & 저장 =====
prices = get_price_series(tickers)
results = []

for t in tickers:
    try:
        metrics = calc_metrics(t, prices)
        if metrics:
            results.append(metrics)
    except Exception as e:
        print(f"Error fetching {t}: {e}")

df = pd.DataFrame(results)

# CSV 저장
df.to_csv("nasdaq100_metrics.csv", index=False, encoding="utf-8-sig")
print("✅ CSV 저장 완료: nasdaq100_metrics.csv")
