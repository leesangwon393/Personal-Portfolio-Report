# compute_wallstreet_srisk_v2.py
# ==================================================
# 🏦 월가 10인 포트폴리오 Srisk 계산 (안정화 버전)
# ==================================================
import pandas as pd
import numpy as np
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# === 문자열에서 숫자만 추출 ===
def extract_number(x):
    if isinstance(x, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", x)
        return float(match.group()) if match else np.nan
    return x


# === Robust Z-score (IQR 기준 + 클리핑) ===
def robust_zscore(val, series):
    series = series.dropna()
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    med = series.median()
    if iqr == 0:
        return 0
    z = (val - med) / iqr
    return np.clip(z, -3, 3)


# === Srisk 계산 함수 ===
def compute_srisk_from_portfolio(portfolio, full_df):
    df = full_df[full_df["Ticker"].isin(portfolio.keys())].copy()
    if df.empty:
        raise ValueError("❌ 포트폴리오 종목들이 CSV 파일에 없습니다.")

    # === 가중 평균 ===
    df["Weight"] = df["Ticker"].map(portfolio)
    sigma_p = np.average(df["Sigma"], weights=df["Weight"])
    mdd_p   = np.average(df["MDD"],  weights=df["Weight"])
    beta_p  = np.average(df["Beta"], weights=df["Weight"])

    # === 섹터 기반 HHI ===
    sector_map = df.set_index("Ticker")["Sector"].to_dict()
    sector_weights = {}
    for ticker, w in portfolio.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + w
    hhi_p = sum([w**2 for w in sector_weights.values()])

    # === 시장 전체 기준 robust Z-score 계산 ===
    S_sigma =  robust_zscore(sigma_p, full_df["Sigma"])              # 변동성 ↑ → 위험 ↑
    S_mdd   = -robust_zscore(abs(mdd_p), abs(full_df["MDD"]))        # 낙폭 ↑ → 위험 ↑
    S_beta  =  robust_zscore(beta_p, full_df["Beta"])                # 민감도 ↑ → 위험 ↑
    S_hhi   =  robust_zscore(hhi_p, pd.Series([(1/len(portfolio))**2]*len(full_df)))

    # === Srisk 계산 (조정된 가중치)
    Srisk = 0.35*S_sigma + 0.25*S_mdd + 0.20*S_beta + 0.10*S_hhi

    # === 구간 분류
    if Srisk < 0.33:
        category = "SAFE"
    elif Srisk < 0.66:
        category = "NEUTRAL"
    else:
        category = "AGGRESSIVE"

    return {
        "Srisk": Srisk,
        "Category": category,
        "Sigma_p": sigma_p,
        "MDD_p": mdd_p,
        "Beta_p": beta_p,
        "HHI": hhi_p
    }


# === 월가 10인 포트폴리오 ===
wallstreet_portfolios = {
    "Warren Buffett": {"AAPL":0.45, "KO":0.10, "AXP":0.09, "KHC":0.08, "CVX":0.07, "BAC":0.06, "HPQ":0.05, "OXY":0.10},
    "Cathie Wood": {"TSLA":0.10, "NVDA":0.06, "ROKU":0.05, "PATH":0.05, "CRSP":0.05, "SQ":0.04, "SHOP":0.04, "COIN":0.05, "TDOC":0.04, "PLTR":0.03},
    "Ray Dalio": {"SPY":0.20, "GLD":0.10, "XOM":0.10, "BABA":0.05, "TLT":0.10, "KO":0.05, "VWO":0.10, "FXI":0.05, "IVV":0.10, "XLK":0.15},
    "Bill Ackman": {"CMG":0.15, "HLT":0.15, "QSR":0.15, "LMT":0.10, "LOW":0.10, "SPY":0.10, "GOOG":0.10, "HHC":0.15},
    "Peter Lynch": {"PG":0.10, "JNJ":0.10, "COST":0.10, "AAPL":0.10, "MCD":0.10, "HD":0.10, "DIS":0.10, "KO":0.10, "WMT":0.10, "PEP":0.10},
    "Ken Griffin": {"NVDA":0.08, "AMZN":0.08, "META":0.07, "MSFT":0.06, "GOOG":0.06, "TSLA":0.05, "NFLX":0.05, "AVGO":0.05, "AMD":0.05, "SPY":0.45},
    "Michael Burry": {"XLE":0.15, "XLF":0.10, "TSLA":0.05, "SPY":0.20, "GLD":0.10, "XLB":0.10, "TLT":0.10, "XLP":0.10, "OXY":0.10},
    "Stan Druckenmiller": {"NVDA":0.10, "MSFT":0.08, "AMZN":0.07, "META":0.07, "SNOW":0.05, "SPY":0.20, "GOOG":0.08, "CRM":0.05, "AAPL":0.05, "INTU":0.05},
    "George Soros": {"SPY":0.15, "TLT":0.10, "AAPL":0.08, "MSFT":0.07, "NVDA":0.05, "GOOG":0.05, "AMZN":0.05, "GLD":0.10, "META":0.05, "BABA":0.10},
    "David Tepper": {"META":0.12, "NVDA":0.10, "AMZN":0.08, "MSFT":0.08, "AAPL":0.06, "GOOG":0.06, "SPY":0.10, "QQQ":0.15, "NFLX":0.10, "TSLA":0.05}
}


# === 실행 ===
if __name__ == "__main__":
    full_df = pd.read_csv(SCRIPT_DIR / "us_market_metrics_sp500_nasdaq100.csv")
    for col in ["Sigma", "MDD", "Beta"]:
        full_df[col] = full_df[col].apply(extract_number)

    results = []
    for name, port in wallstreet_portfolios.items():
        try:
            r = compute_srisk_from_portfolio(port, full_df)
            results.append({
                "Investor": name,
                "Srisk": round(r["Srisk"], 4),
                "Category": r["Category"],
                "Sigma_p": round(r["Sigma_p"], 4),
                "MDD_p": round(r["MDD_p"], 4),
                "Beta_p": round(r["Beta_p"], 4),
                "HHI": round(r["HHI"], 4)
            })
        except Exception as e:
            results.append({"Investor": name, "Error": str(e)})

    out = pd.DataFrame(results).sort_values("Srisk", ascending=False)
    print("\n🏦 월가 10인 Srisk 결과표\n", out)
    out_path = SCRIPT_DIR / "wallstreet_srisk_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\n💾 저장 완료: {out_path}")
