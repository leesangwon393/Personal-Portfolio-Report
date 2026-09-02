# 13F 공개 주식 포트폴리오 기반 위험 운용성향 — 방법론

`scripts/run_13f_clustering.py`가 SEC EDGAR의 13F-HR 공개 데이터로부터
기관투자자 100곳의 롱 포지션 포트폴리오를 수집하고, 4개 위험지표
(Volatility / MDD / Beta / Sector HHI)를 계산한 뒤 비지도 군집화로
DEFENSIVE / BALANCED / AGGRESSIVE 세 군집으로 사후 해석하는 파이프라인이다.

이 문서는 실제 실행 결과가 아니라 **방법론과 재현 방법**을 설명한다.
실제 실행 결과(확보한 포트폴리오 수, 매핑 커버리지, 군집 평가지표, 대표
투자자 배정 등)는 매 실행마다 `artifacts/13f_clustering/report.md`에
그 실행 자체의 산출물로 새로 생성된다.

> **용어 주의**: 이 파이프라인이 산출하는 라벨은 사람의 전체 투자성향이
> 아니라 **13F 공개 주식 포트폴리오 기반 위험 운용성향**이다. 13F는
> 미국 상장 지분증권 롱 포지션(및 일부 옵션)만 보고 대상이므로, 채권·
> 현금·비상장자산·해외직접상장주식·숏 포지션은 반영되지 않는다.

---

## 1. 데이터 출처

- **SEC EDGAR Form 13F 구조화 데이터셋**
  (https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets):
  분기(정확히는 ~3개월 롤링 윈도우)별로 그 기간에 제출된 모든 13F 제출을
  `SUBMISSION.tsv`(제출 메타), `COVERPAGE.tsv`(필터 기관 정보), `INFOTABLE.tsv`
  (실제 보유 종목별 행)로 묶어 배포한다. 개별 필링을 하나씩 스크레이핑하는
  대신 이 벌크 데이터셋을 사용해 SEC에 수천 번의 개별 요청을 보내지 않는다.
- **가격 데이터**: Yahoo Finance(`yfinance`), 매핑된 ticker + SPY의 수정
  종가.
- **업종(Sector)**: Yahoo Finance `yf.Ticker(x).info["sector"]`
  (`data/yfinance_metrics.py`가 이미 쓰던 것과 동일한 소스/패턴).
- **CUSIP → ticker 매핑**: OpenFIGI `/v3/mapping`.

## 2. 분기(기준일) 선택 로직

SEC는 "달력 분기"가 아니라 **~3개월 롤링 제출일 윈도우**(예:
`01mar2026-31may2026`)로 데이터셋을 배포한다. 파이프라인은:

1. 데이터셋 목록 페이지에서 최신 윈도우부터 확인한다.
2. 각 윈도우의 `COVERPAGE.REPORTCALENDARORQUARTER` 최빈값을 그 윈도우가
   대표하는 "완전히 제출이 끝난 공통 분기"로 판단한다(45일 신고 마감이
   그 윈도우 안에 들어오는 분기가 압도적 다수를 차지하기 때문).
3. `--quarter`로 요청한 분기가 최빈값과 일치하는 윈도우를 찾으면 사용;
   찾지 못하면(신고 마감이 아직 안 지났거나 SEC가 아직 발행하지 않음)
   가장 최신 윈도우의 최빈 분기로 자동 대체하고 **그 사실을 로그와
   report.md에 명시**한다 — 조용히 다른 분기로 바꾸지 않는다.

## 3. 포트폴리오 선정

1. `SUBMISSIONTYPE`이 `13F-HR`/`13F-HR/A`인 제출만 사용(`13F-NT`는 보유
   내역이 없는 통지용 제출이라 제외).
2. 동일 (CIK, PERIODOFREPORT)에서 `FILING_DATE`가 가장 늦은 제출만 남긴다
   (원본/수정본 중복 계산 방지). **알려진 단순화**: `AMENDMENTTYPE`이
   "NEW HOLDINGS"인 부분 수정은 원본과 병합하지 않고 최신 제출본만
   사용한다.
3. `PUTCALL`이 비어있지 않은 행(PUT, CALL 모두)은 기본 포트폴리오에서
   제외한다(옵션 행의 VALUE는 기초 주식 수량 기준으로 보고되어 매수
   포지션과 경제적으로 동일하지 않음).
4. 유효 롱 포지션(고유 CUSIP) 10개 이상인 필터만 후보로 남긴다.
5. **선정 방식** (`--selection-method`):
   - `top_value`(기본값): 신고가액 총합 기준 상위. **주의**: 실제 실행에서
     이 방식은 BlackRock/State Street/Morgan Stanley/Goldman Sachs 같은
     초대형 커스터디언·브로커딜러 결합 필링(단일 전략이 아니라 수천 개
     고객 계좌를 하나의 13F로 합산 제출)에 심하게 편중되는 것을 실제로
     확인했다 — 예: 상위 후보의 CUSIP 수가 수천~수만 개, position이
     BlackRock 5,606개/Morgan Stanley 8,287개 수준. 결과에서 이 편중을
     반드시 점검한다.
   - `aum_stratified`: 유효 후보를 신고가액 분위수(기본 5구간)로 나눠
     구간별로 비례 샘플링. 특정 초대형 필터에 편중되지 않고, CUSIP
     매핑 대상도 훨씬 작아 실행이 더 빠르다.
   - 두 방식 모두 CLI에서 선택 가능하며, `report.md`에 실제 두 방식의
     차이(후보 수, 고유 CUSIP 수, position count 분포)를 기록한다.
6. `--candidate-pool-multiplier`로 목표 개수보다 넉넉한 후보 풀을 확보해,
   매핑/가격 커버리지 미달로 제외된 포트폴리오를 다음 순위 후보로 자동
   대체한다.

## 4. Ticker 매핑

CUSIP → ticker는 OpenFIGI `/v3/mapping`을 사용한다(우선순위: 저장소 내
기존 매핑 재사용 → OpenFIGI → 그 외 신뢰 가능한 방법 → 확인 불가 시
unmapped). Issuer 이름 유사도로 임의 추정하지 않는다. 결과는
`artifacts/13f_clustering/cache/ticker_mapping/cusip_ticker_map.json`에
캐시되어 재실행 시 재조회하지 않는다.

`OPENFIGI_API_KEY` 유무에 따른 실측 차이(둘 다 실제로 확인함):

| | 배치당 CUSIP 수 | 요청 간 대기 |
|---|---|---|
| API 키 없음 | 10 (무인증 시 초과하면 HTTP 413) | 2.6초 |
| API 키 있음 | 100 | 0.3초 |

포트폴리오당:

- 전체 신고가액
- ticker 매핑 성공 금액 / 커버리지
- 제외된 종목 수 / 가치 비중

을 기록하고, 매핑 커버리지 80% 미만 포트폴리오는 다음 순위 후보로 교체한다.

## 5. 포트폴리오 수익률 구성

기준일(분기 말) 시점 종목별 신고가액으로 비중을 계산하고
(`w_i = Value_i / sum_j Value_j`), 기준일 직전 `--lookback-days`(기본
252) 거래일의 수정종가 수익률을 사용한다. 가격 이력이 부족한 종목은
제외 후 남은 비중을 다시 합 1로 정규화한다.

**알려진 근사**: 기준일의 고정비중을 과거 lookback 기간 내내 유지했다고
가정한다(실제로는 그 기간 매매가 있었을 것). 이는 스냅샷 기반 분석의
근본적 한계로, report.md에도 매번 명시한다.

## 6. 위험지표 정의

| 지표 | 정의 | 방향 |
|---|---|---|
| Volatility | `std(R_p) * sqrt(252)` | 높을수록 위험 |
| MDD | `max_t(1 - V_t / max_{s<=t} V_s)`, 양수로 저장 | 높을수록 위험 |
| Beta | `Cov(R_p, R_m) / Var(R_m)`, R_m = SPY | 높을수록(시장보다 민감) 위험 |
| Sector HHI | 업종(개별 종목 아님)별 비중 제곱합 `sum_s w_s^2` | 높을수록(집중) 위험 |

업종을 확인하지 못한 종목은 제거하지 않고 "Unknown" 섹터로 묶어 HHI에
포함하며, 그 비중을 `unknown_sector_weight`로 함께 기록한다.

### 기존 코드에서 발견한 계산 오류 (참고)

이 프로젝트에는 기존에 사용자 개인 포트폴리오 성향을 분류하는 별도의
분류기(`srisk_result/analyze_portfolio_risk.py`,
`srisk_result/wallstreet_srisk_test.py`)가 있다. 점검 결과, 두 파일 모두
HHI를 정규화하는 `robust_zscore(hhi_p, pd.Series([(1/len(portfolio))**2] *
len(full_df)))` 호출에서 비교 대상 Series가 **동일한 값을 반복한
상수 Series**로 구성되어 있다. `robust_zscore`는 IQR(사분위 범위)이
0이면 무조건 0을 반환하므로, 이 HHI 항은 항상 0이 되어 최종 `Srisk`
점수에 어떤 포트폴리오를 넣어도 집중도가 전혀 반영되지 않는다(재현:
`python3 -c "from srisk_result.analyze_portfolio_risk import robust_zscore; import pandas as pd; print(robust_zscore(0.5, pd.Series([0.04]*10)))"`
→ `0`).

이 13F 파이프라인은 같은 실수를 반복하지 않도록, HHI를 포함한 4개 지표
모두 **실제 수집한 100개 포트폴리오의 횡단면 분포**에 `RobustScaler`/
`StandardScaler`를 `fit`해 정규화한다(상수 비교 Series를 쓰지 않음).
다만 이 발견은 별도 리포트용이며, 이번 작업은 기존 단일 포트폴리오
분류기(`srisk_result/*.py`, `app.py`가 사용)의 동작 자체는 변경하지
않았다 — 그 파일들의 수정 여부는 별도로 판단할 사안이다.

## 7. 전처리

- 결측치/무한값 점검(`processed/missing_infinite_report.csv`)
- 1%/99% winsorization, 적용 전/후 모두 `processed/portfolio_metrics.csv`
  (원본) / `portfolio_metrics_winsorized.csv`(winsorized)로 보존
- `StandardScaler`와 `RobustScaler`를 모두 `fit`해 두고, 이상치에 강건한
  `RobustScaler`를 기본 후보(`--scaler robust`)로 사용
- 4개 지표 모두 "값이 클수록 위험"으로 방향 통일(별도 부호 반전 불필요)

## 8. 군집화 모델 선택 근거

K-means(`n_init=50`, `random_state=42`)와 Gaussian Mixture Model을
K=2..6 전 구간에서 비교하고 Silhouette / Calinski-Harabasz /
Davies-Bouldin / 군집별 표본 수 / bootstrap 기반 안정성(Adjusted Rand
Index)을 모두 `clustering/clustering_evaluation.csv`에 기록한다.

**K=3 선택 근거와 한계**: 서비스가 DEFENSIVE/BALANCED/AGGRESSIVE 3분류를
요구하므로 K=3을 우선 검토한다. 그러나 K=3이 다른 K보다 지표상 우수하다는
보장은 없다 — 실제 평가표(`report.md` 6절)를 숨기지 않고 그대로
제공하므로, K=3이 차선이었다면 그 사실도 report.md에서 확인할 수 있다.
최종 저장 모델(`clustering/model.joblib`)은 `--k`로 바꿀 수 있다.

`ClusterRisk_k = mean(Z_Vol, Z_MDD, Z_Beta, Z_HHI)`(스케일러 출력을 Z로
사용)가 낮은 군집부터 DEFENSIVE → BALANCED → AGGRESSIVE로 이름을 붙인다.
**이것은 학습용 라벨링이 아니라 비지도 군집 결과에 대한 사후 해석이다.**

## 9. 대표 투자자 Sanity Check

Berkshire Hathaway, ARK Investment Management 등(`sec13f/config.py`의
`REPRESENTATIVE_INVESTORS`)이 최종 100개 포트폴리오에 포함되면 어느
군집에 배정됐는지 별도로 출력한다. 이 라벨은 군집화에 전혀 사용되지
않으며, 결과가 직관과 다르더라도(예: 집중된 포트폴리오 때문에
Berkshire가 BALANCED/AGGRESSIVE로 분류) 군집을 강제로 수정하지 않고
volatility/mdd/beta/sector_hhi 값으로 이유를 설명한다.

## 10. 재현 실행

```bash
python scripts/run_13f_clustering.py \
  --quarter 2026-06-30 \
  --num-portfolios 100 \
  --lookback-days 252 \
  --selection-method aum_stratified \
  --output-dir artifacts/13f_clustering
```

```bash
python -m pytest test_13f_clustering.py -v
```

## 11. 환경변수

```env
SEC_USER_AGENT="YourAppName your-email@example.com"   # SEC 요청에 필수 권장
OPENFIGI_API_KEY=                                      # 선택 — 없으면 매핑이 더 느림
```

## 12. 신규 사용자 포트폴리오 분류

```python
from sec13f.clustering import classify_portfolio

result = classify_portfolio(
    volatility=0.24, mdd=0.31, beta=1.08, sector_hhi=0.22,
    model_path="artifacts/13f_clustering/clustering/model.joblib",
)
# {"risk_profile": "...", "cluster_id": ..., "distance_or_probability": ..., "metrics": {...}}
```

K-means로 학습된 모델이면 `distance_or_probability`는 배정된 군집
중심까지의 (스케일된 feature 공간에서의) 유클리드 거리이고, GMM이면
배정된 컴포넌트의 확률이다 — K-means 거리를 확률처럼 표현하지 않는다.

## 13. 결과 해석 시 주의사항

- 이 결과는 **13F 공개 주식 포트폴리오 기반 위험 운용성향**이며, 투자자의
  전체 자산배분·투자철학을 대표하지 않는다.
- 13F는 분기 말 스냅샷이며, 분기 중 매매·공매도·채권/현금/비상장자산은
  반영되지 않는다.
- 고정비중을 과거로 투영하는 근사이므로 실제 수익률과 다를 수 있다.
- 군집 이름은 사후 해석이며, K=3이 데이터에 자연스러운 구조가 아닐 경우
  경계에 있는 포트폴리오의 분류는 민감할 수 있다(K별 평가표로 확인).
