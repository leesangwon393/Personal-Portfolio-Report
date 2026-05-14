import pandas as pd
import numpy as np
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_CANDIDATES = [
    Path.cwd() / "nasdaq100_metrics.csv",
    Path.cwd() / "us_market_metrics_sp500_nasdaq100.csv",
    ROOT_DIR / "classification" / "nasdaq100_metrics.csv",
    ROOT_DIR / "srisk_result" / "us_market_metrics_sp500_nasdaq100.csv",
]


def load_risk_metrics():
    required = {"Ticker", "Sigma", "MDD", "Beta", "Sector"}
    for path in METRICS_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if required.issubset(df.columns):
            print(f"Using metrics CSV: {path}")
            return df
    tried = "\n".join(str(p) for p in METRICS_CANDIDATES)
    raise FileNotFoundError(
        "No risk metrics CSV found with columns "
        f"{sorted(required)}. Tried:\n{tried}"
    )

# ===== 1. 내꺼포트폴리오 (나스닥 중심) =====
weights = {
    "MSFT": 0.20,   # 마이크로소프트 (안정적 빅테크)
    "AAPL": 0.15,   # 애플 (대형 소비·테크)
    "COST": 0.10,   # 코스트코 (소비 디펜시브)
    "PEP": 0.10,    # 펩시 (소비 디펜시브)
    "AMGN": 0.10,   # 암젠 (헬스케어 안정주)
    "GILD": 0.10,   # 길리어드 (헬스케어 안정주)
    "AEP": 0.05,    # 아메리칸 일렉트릭 파워 (유틸리티)
    "XEL": 0.05,    # 엑셀 에너지 (유틸리티)
    "CASH": 0.15    # 현금 버퍼
}


# ===== 2. CSV 불러오기 =====
df = load_risk_metrics()
df.set_index("Ticker", inplace=True)

results = []
for ticker, w in weights.items():
    if ticker == "CASH":
        results.append({
            "Ticker": "CASH",
            "Sigma": 0.0,
            "MDD": 0.0,
            "Beta": 0.0,
            "Sector": "Cash",
            "Weight": w
        })
    elif ticker in df.index:
        row = df.loc[ticker]
        results.append({
            "Ticker": ticker,
            "Sigma": row["Sigma"],
            "MDD": row["MDD"],
            "Beta": row["Beta"],
            "Sector": row["Sector"],
            "Weight": w
        })

pf = pd.DataFrame(results)

# ===== 3. 업종 집중도 HHI =====
sector_weights = pf.groupby("Sector")["Weight"].sum()
hhi_p = (sector_weights ** 2).sum()

# ===== 4. 포트폴리오 전체 지표 (가중평균) =====
sigma_p = np.average(pf["Sigma"], weights=pf["Weight"])
mdd_p   = abs(np.average(pf["MDD"], weights=pf["Weight"]))
beta_p  = np.average(pf["Beta"], weights=pf["Weight"])
cash_p  = pf.loc[pf["Ticker"]=="CASH", "Weight"].sum()

print("📊 Raw 지표")
print(f"σ: {sigma_p:.3f}, MDD: {mdd_p:.3f}, β: {beta_p:.3f}, HHI: {hhi_p:.3f}, Cash: {cash_p:.3f}")

# ===== 5. z-score 표준화 =====
def zscore(val, series):
    return (val - series.mean()) / series.std() if series.std() > 0 else 0

# Nasdaq100 전체 지표 분포 기준
S_sigma = zscore(sigma_p, df["Sigma"])
S_mdd   = zscore(mdd_p, df["MDD"].abs())
S_beta  = zscore(beta_p, df["Beta"])
S_hhi   = zscore(hhi_p, (df.groupby("Sector").size() / len(df))**2)  # 업종 집중도 분포 근사
S_cash  = cash_p  # Cash는 그대로 %

print("\n📊 Z-score 지표")
print(f"S_sigma: {S_sigma:.3f}, S_mdd: {S_mdd:.3f}, S_beta: {S_beta:.3f}, S_hhi: {S_hhi:.3f}, S_cash: {S_cash:.3f}")

# ===== 6. Srisk 계산 (Cash 비중 축소 반영) =====
Srisk = (
    0.30 * S_sigma +
    0.25 * S_mdd +
    0.20 * S_beta +
    0.15 * S_hhi -
    0.05 * S_cash
)


# ===== 7. 성향 분류 =====
if Srisk < 0.33:
    category = "SAFE"
elif Srisk < 0.66:
    category = "NEUTRAL"
else:
    category = "AGGRESSIVE"

# ===== 8. 최종 출력 =====
print("\n✅ Srisk 점수:", round(Srisk, 4))
print("🎯 투자자 성향 분류:", category)
