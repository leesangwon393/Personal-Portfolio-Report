# compute_my_srisk.py
# ==================================================
# 💼 개인 포트폴리오 Srisk 계산기 (최종 안정화 버전)
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
        "HHI": hhi_p,
        "Sectors": sector_weights
    }


# === 실행 ===
if __name__ == "__main__":
    print("\n💼 개인 포트폴리오 Srisk 계산기 (시장 전체 기준)\n")

    # CSV 로드
    full_df = pd.read_csv(SCRIPT_DIR / "us_market_metrics_sp500_nasdaq100.csv")
    for col in ["Sigma", "MDD", "Beta"]:
        full_df[col] = full_df[col].apply(extract_number)

    # 사용자 포트폴리오 입력
    print("👉 종목과 비중을 입력하세요 (쉼표로 구분)")
    print("   예시: AAPL 0.3, MSFT 0.25, NVDA 0.15, TSLA 0.1, AMZN 0.2\n")
    raw = input("> ")

    portfolio = {}
    for pair in raw.split(","):
        parts = pair.strip().split()
        if len(parts) == 2:
            ticker, w = parts
            try:
                portfolio[ticker.upper()] = float(w)
            except ValueError:
                pass

    if not portfolio:
        print("❌ 포트폴리오 입력이 비어있습니다.")
        exit()

    # 계산 실행
    r = compute_srisk_from_portfolio(portfolio, full_df)

    print("\n✅ [결과 요약]")
    print(f"   Srisk = {r['Srisk']:.4f} → {r['Category']}")
    print(f"   σ_p={r['Sigma_p']:.4f}, MDD_p={r['MDD_p']:.4f}, β_p={r['Beta_p']:.4f}, HHI={r['HHI']:.4f}")
    print(f"   섹터 비중: {r['Sectors']}")

    out_path = SCRIPT_DIR / "my_srisk_result.csv"
    pd.DataFrame({
        "Sigma_p": [r["Sigma_p"]],
        "MDD_p": [r["MDD_p"]],
        "Beta_p": [r["Beta_p"]],
        "HHI": [r["HHI"]],
        "Srisk": [r["Srisk"]],
        "Category": [r["Category"]]
    }).to_csv(out_path, index=False)

    print(f"\n💾 저장 완료: {out_path}")
