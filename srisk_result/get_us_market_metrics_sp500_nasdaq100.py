# get_us_market_metrics.py (수정본)
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LinearRegression


SCRIPT_DIR = Path(__file__).resolve().parent

def get_us_market_metrics():
    print("📈 S&P500 + Nasdaq100 리스크 및 섹터 지표 수집 중...")

    # ===== 1️⃣ S&P500 목록 =====
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        tables = pd.read_html(requests.get(sp500_url, headers=headers).text)
        sp500_table = tables[0]
        sp500_tickers = [t.replace('.', '-') for t in sp500_table["Symbol"].tolist()]
        sp500_sector_map = sp500_table.set_index("Symbol")["GICS Sector"].to_dict()
        print(f"✅ S&P500 종목 {len(sp500_tickers)}개 로드 완료")
    except Exception as e:
        print("⚠️ S&P500 로드 실패:", e)
        sp500_tickers, sp500_sector_map = [], {}

    # ===== 2️⃣ Nasdaq100 목록 =====
    nasdaq_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(requests.get(nasdaq_url, headers=headers).text)
        nasdaq_table = tables[4] if len(tables) > 4 else tables[0]
        nasdaq_tickers = [t.replace('.', '-') for t in nasdaq_table["Ticker"].dropna().tolist()]
        print(f"✅ Nasdaq100 종목 {len(nasdaq_tickers)}개 로드 완료")
    except Exception as e:
        print("⚠️ Nasdaq100 로드 실패:", e)
        nasdaq_tickers = []

    # ===== 3️⃣ 합집합 =====
    all_tickers = sorted(set(sp500_tickers + nasdaq_tickers))
    print(f"📊 전체 종목 수 (합집합): {len(all_tickers)}개")
    if not all_tickers:
        print("❌ 종목 목록을 가져오지 못해 기존 CSV를 덮어쓰지 않고 종료합니다.")
        return

    # ===== 4️⃣ 기간 설정 =====
    end = datetime.now()
    start = end - timedelta(days=365 * 3)

    # ===== 5️⃣ S&P500 벤치마크 =====
    print("\n📥 S&P500 (^GSPC) 지수 데이터 다운로드 중...")
    spy_data = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)

    # ✅ yfinance 버전별로 안전 처리
    if isinstance(spy_data.columns, pd.MultiIndex):
        if "Adj Close" in spy_data.columns.levels[0]:
            spy = spy_data["Adj Close"]
        else:
            spy = spy_data["Close"]
    else:
        if "Adj Close" in spy_data.columns:
            spy = spy_data["Adj Close"]
        elif "Close" in spy_data.columns:
            spy = spy_data["Close"]
        else:
            raise KeyError("❌ 'Close' 혹은 'Adj Close' 컬럼이 없습니다.")

    spy_returns = np.log(spy / spy.shift(1)).dropna()

    # ===== 6️⃣ 리스크 계산 함수 =====
    def calc_metrics(ticker):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if data.empty:
                return None
            price = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
            price = price.dropna()
            if len(price) < 200:
                return None

            ret = np.log(price / price.shift(1)).dropna()
            sigma = ret.std() * np.sqrt(252)
            cum = (1 + ret).cumprod()
            mdd = (cum / cum.cummax() - 1).min()

            aligned = pd.concat([ret, spy_returns], axis=1).dropna()
            aligned.columns = ["r", "spy"]
            beta = LinearRegression().fit(aligned["spy"].values.reshape(-1, 1), aligned["r"].values).coef_[0]

            sector = sp500_sector_map.get(ticker.replace('-', '.'), "Unknown")
            return {
                "Ticker": ticker,
                "Sector": sector,
                "Sigma": round(float(sigma), 6),
                "MDD": round(float(mdd), 6),
                "Beta": round(float(beta), 6)
            }
        except Exception:
            return None

    # ===== 7️⃣ 크롤링 진행 =====
    print("\n📊 리스크 계산 중...")
    results = []
    for t in tqdm(all_tickers):
        r = calc_metrics(t)
        if r:
            results.append(r)

    # ===== 8️⃣ 저장 =====
    df = pd.DataFrame(results)
    if df.empty:
        print("❌ 계산된 결과가 없어 기존 CSV를 덮어쓰지 않고 종료합니다.")
        return

    out_path = SCRIPT_DIR / "us_market_metrics_sp500_nasdaq100.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 저장 완료: {out_path} ({len(df)}개 종목)")
    print(df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Collect S&P500 + Nasdaq100 risk metrics with yfinance."
    )
    parser.parse_args()
    get_us_market_metrics()


if __name__ == "__main__":
    main()
