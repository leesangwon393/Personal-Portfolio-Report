# 13F 공개 주식 포트폴리오 기반 위험 운용성향 — 군집화 리포트

생성 시각(UTC): 2026-09-02T13:54:58+00:00

이 리포트가 분류하는 것은 **개인·기관 투자자의 전체 투자성향이 아니라**, SEC EDGAR에 공개된 13F 롱 포지션 스냅샷 하나에서 계산한 **정량적 위험 특성**입니다. 13F는 미국 상장 지분증권(및 관련 옵션)만 보고 대상이며, 채권·현금·비상장자산·해외직접상장주식·공매도(숏) 포지션은 13F에 나타나지 않습니다. 따라서 동일 기관이라도 실제 총자산 배분과 이 리포트의 위험 프로파일은 다를 수 있습니다.

## 1. 분석 기준 (Section 1)

- 요청한 분기: `2026-06-30`
- 실제 사용한 분기(기준일): **2026-03-31**
- 요청 분기 사용 가능 여부: False
- 대체 사유: --quarter 2026-06-30 is not the dominant REPORTCALENDARORQUARTER in any published SEC 13F bulk dataset window yet (its 45-day filing deadline may not have passed, or SEC hasn't published that window's structured data set). Falling back to the newest fully-published complete quarter: 2026-03-31 (dataset window 01mar2026-31may2026).
- 사용한 SEC 13F 데이터셋 윈도우: `01mar2026-31may2026` (https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2026-31may2026_form13f.zip)
- 가격 분석기간(lookback): 252 거래일, 모든 포트폴리오 동일 기준일/동일 기간 적용
- 판단 로직: SEC가 발행하는 분기별 Form 13F 구조화 데이터셋 목록에서 가장 최신 윈도우부터 COVERPAGE.tsv의 `REPORTCALENDARORQUARTER` 최빈값을 그 윈도우의 '완전히 제출이 끝난 공통 분기'로 판단하고, 요청한 분기와 일치하는 윈도우를 찾을 때까지 더 오래된 윈도우로 내려간다(요청 분기의 45일 신고 마감이 아직 지나지 않았거나 SEC가 해당 윈도우를 아직 발행하지 않은 경우 자동으로 대체됨).

## 2. 데이터 출처 및 포트폴리오 선정 (Section 2-3)

- 출처: SEC EDGAR 공식 Form 13F 구조화 데이터셋 (https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets), `SUBMISSION.tsv`/`COVERPAGE.tsv`/`INFOTABLE.tsv`.
- 선정 방식: `aum_stratified` (AUM 구간별 층화표본)
- 스캔한 후보(랭킹된 필터 통과 기관) 수: 107
- 최종 확보한 포트폴리오 수: **99** / 목표 100
- 제외된 포트폴리오 수: 5

제외 사유별 건수:

- `mapping_coverage_below_threshold`: 4건
- `price_history_coverage_below_threshold`: 1건

- 13F-HR/A 수정공시 처리: 동일 (CIK, PERIODOFREPORT)에서 FILING_DATE가 가장 최근인 제출만 사용. **알려진 단순화**: AMENDMENTTYPE이 'NEW HOLDINGS'(원본에 누락된 보유를 추가하는 부분 수정)인 경우도 '최신 제출본 단독 사용'으로 처리하며, 원본과 병합하지 않는다 — 원본을 완전히 대체하는 'RESTATEMENT' 성격의 수정에는 정확하지만, 드문 부분 수정 케이스에서는 일부 보유가 과소집계될 수 있다.
- PUT/옵션 포지션 처리: INFOTABLE의 `PUTCALL`이 비어있지 않은 행(PUT, CALL 모두)은 기본 포트폴리오에서 제외했다. CALL도 함께 제외한 이유는, 13F 규정상 옵션 행의 VALUE가 옵션 자체의 시가가 아니라 '기초 주식 수량에 대한 가치'로 보고되어 실제 매수 포지션과 경제적으로 동일하지 않기 때문이다(해석이 불명확한 포지션에 해당).

## 3. Ticker 매핑 (Section 3)

- 평균 매핑 커버리지: 97.6%
- 최소 매핑 커버리지: 83.0%
- 매핑 커버리지 80% 미만 포트폴리오는 최종 군집화에서 제외하고 다음 순위 후보로 대체했다(대체 여력을 위해 목표치의 1배까지 후보 풀을 확장).
- CUSIP -> ticker 매핑은 OpenFIGI `/v3/mapping`을 사용했고, 결과는 `artifacts/13f_clustering/cache/ticker_mapping/cusip_ticker_map.json`에 캐시되어 재실행 시 재조회하지 않는다. Issuer 이름 유사도로 임의 추정한 매핑은 없다 — 실패한 CUSIP은 unmapped로 기록만 한다.

## 4. 4개 위험지표 요약통계 (Section 5-6)

|       |   volatility |        mdd |      beta |   sector_hhi |
|:------|-------------:|-----------:|----------:|-------------:|
| count |   99         | 99         | 99        |    99        |
| mean  |    0.181878  |  0.120475  |  0.876397 |     0.47755  |
| std   |    0.0591619 |  0.0352623 |  0.225063 |     0.289247 |
| min   |    0.0983937 |  0.0692146 |  0.401841 |     0.122215 |
| 25%   |    0.149284  |  0.104283  |  0.746093 |     0.221584 |
| 50%   |    0.169625  |  0.114327  |  0.862493 |     0.408013 |
| 75%   |    0.197632  |  0.126936  |  0.993495 |     0.76445  |
| max   |    0.477503  |  0.323849  |  1.71541  |     0.975365 |

- **Volatility**: 포트폴리오 일별 수익률 표준편차 x sqrt(252)
- **MDD**: 누적수익률 곡선의 최고점 대비 최대 하락폭, 양의 크기로 저장
- **Beta**: SPY 대비 Cov(R_p, R_m) / Var(R_m)
- **Sector HHI**: 업종별(개별 종목이 아닌 GICS 유사 sector 단위) 비중 제곱합. ETF/펀드처럼 yfinance에 GICS sector가 원래 없는 보유는 별도 'ETF/Fund' 버킷으로, 재시도 후에도 끝내 응답을 받지 못한 종목만 'Unknown' 버킷으로 묶어 HHI에 포함했다(제거하지 않음). 이번 실행에서 포트폴리오당 평균 `unknown_sector_weight`(진짜 미확인 비중)는 1.6%로 낮다(`processed/portfolio_metrics.csv` 참고).

- **알려진 근사**: 기준일 시점 13F 스냅샷의 고정비중을 기준일 직전 252거래일에 그대로 적용해 과거 수익률을 재구성한다. 실제로는 그 기간 동안 기관이 매매를 했을 것이므로, 이는 '기준일의 보유 구성이 분석기간 내내 유지되었다면'이라는 근사다.

## 5. 전처리 (Section 6)

- Winsorization: 하위/상위 1%/99% 분위수로 clip, 적용 전/후 버전 모두 `processed/portfolio_metrics.csv` / `portfolio_metrics_winsorized.csv`로 보존
- 최종 사용 스케일러: **robust**(이상치에 상대적으로 강건)
- 네 지표 모두 '값이 클수록 위험'으로 방향을 통일했다(HHI가 낮을수록 분산 = 저위험, 높을수록 집중 = 고위험이므로 별도 부호 반전 없이 그대로 사용).

## 6. 군집화 모델 비교 (Section 7)

| method   |   k |   silhouette |   calinski_harabasz |   davies_bouldin | cluster_sizes                                        |   min_cluster_size |   stability_ari |
|:---------|----:|-------------:|--------------------:|-----------------:|:-----------------------------------------------------|-------------------:|----------------:|
| kmeans   |   2 |    0.495884  |             89.3419 |         0.865042 | {"0": 74, "1": 25}                                   |                 25 |        0.630569 |
| kmeans   |   3 |    0.421227  |            128.264  |         0.727378 | {"0": 54, "1": 4, "2": 41}                           |                  4 |        0.721288 |
| kmeans   |   4 |    0.388178  |            143.635  |         0.780595 | {"0": 24, "1": 4, "2": 28, "3": 43}                  |                  4 |        0.873304 |
| kmeans   |   5 |    0.346671  |            131.769  |         0.841959 | {"0": 37, "1": 4, "2": 28, "3": 14, "4": 16}         |                  4 |        0.558619 |
| kmeans   |   6 |    0.362328  |            131.709  |         0.702271 | {"0": 30, "1": 14, "2": 17, "3": 2, "4": 34, "5": 2} |                  2 |        0.642611 |
| gmm      |   2 |    0.390113  |             57.8608 |         1.05548  | {"0": 67, "1": 32}                                   |                 32 |        0.53661  |
| gmm      |   3 |    0.356112  |             91.6319 |         0.895427 | {"0": 30, "1": 65, "2": 4}                           |                  4 |        0.849592 |
| gmm      |   4 |    0.398739  |             84.2522 |         0.579766 | {"0": 58, "1": 2, "2": 37, "3": 2}                   |                  2 |        0.517755 |
| gmm      |   5 |    0.0877146 |             57.3544 |         1.28438  | {"0": 4, "1": 4, "2": 30, "3": 57, "4": 4}           |                  4 |        0.624613 |
| gmm      |   6 |    0.0815805 |             40.9955 |         1.59773  | {"0": 53, "1": 13, "2": 2, "3": 25, "4": 4, "5": 2}  |                  2 |        0.516471 |

- **최종 선택 모델**: KMEANS, K=3

K=3(서비스에 필요한 DEFENSIVE/BALANCED/AGGRESSIVE 3분류)의 실제 평가지표는 위 표의 k=3 행을 참고. Silhouette/Calinski-Harabasz/Davies-Bouldin이 다른 K보다 항상 더 좋다는 보장은 없으며, 표에 K=2..6 전체를 숨기지 않고 보고했다 — K=3이 다른 K보다 지표상 열세라면 이 표에서 그대로 드러난다.

## 7. 군집별 표본 수와 중심값 (Section 7)

|   cluster_id |   volatility |      mdd |    beta |   sector_hhi |   cluster_risk_scaled | risk_profile   |
|-------------:|-------------:|---------:|--------:|-------------:|----------------------:|:---------------|
|            0 |     0.146353 | 0.102526 | 0.73476 |     0.639396 |             -0.273097 | DEFENSIVE      |
|            1 |     0.370809 | 0.238325 | 1.45797 |     0.264164 |              2.94421  | AGGRESSIVE     |
|            2 |     0.207991 | 0.131378 | 1.00826 |     0.285151 |              0.477272 | BALANCED       |

- **DEFENSIVE**: 54개 포트폴리오
- **BALANCED**: 41개 포트폴리오
- **AGGRESSIVE**: 4개 포트폴리오

**중요**: DEFENSIVE/BALANCED/AGGRESSIVE는 사람이 직접 라벨링한 정답이 아니라, 4개 위험지표로 비지도 군집화(K-means)를 수행한 뒤 각 군집 중심의 `ClusterRisk_k = mean(Z_Vol, Z_MDD, Z_Beta, Z_HHI)`가 낮은 순서대로 이름을 붙인 **사후 해석**이다.

## 8. 대표 투자자 Sanity Check (Section 8)

| label                                      |     cik | in_final_100   | manager_name                             |   cluster_id | risk_profile   |   volatility |      mdd |     beta |   sector_hhi |
|:-------------------------------------------|--------:|:---------------|:-----------------------------------------|-------------:|:---------------|-------------:|---------:|---------:|-------------:|
| Berkshire Hathaway Inc                     | 1067983 | True           | Berkshire Hathaway Inc                   |            2 | BALANCED       |     0.197212 | 0.157827 | 0.903344 |     0.222707 |
| ARK Investment Management LLC              | 1697748 | True           | ARK Investment Management LLC            |            1 | AGGRESSIVE     |     0.3772   | 0.209386 | 1.71541  |     0.190631 |
| Duquesne Family Office LLC (Druckenmiller) | 1536411 | True           | Duquesne Family Office LLC               |            2 | BALANCED       |     0.272881 | 0.143694 | 1.22703  |     0.252583 |
| Bridgewater Associates, LP                 | 1350694 | True           | Bridgewater Associates, LP               |            2 | BALANCED       |     0.232924 | 0.133606 | 1.1761   |     0.168899 |
| Renaissance Technologies LLC               | 1037389 | True           | RENAISSANCE TECHNOLOGIES LLC             |            2 | BALANCED       |     0.203738 | 0.13368  | 1.0026   |     0.122215 |
| Baupost Group LLC/MA                       | 1061768 | True           | BAUPOST GROUP LLC/MA                     |            2 | BALANCED       |     0.194813 | 0.113775 | 0.873878 |     0.228109 |
| Pershing Square Capital Management, L.P.   | 1336528 | True           | Pershing Square Capital Management, L.P. |            2 | BALANCED       |     0.228459 | 0.189979 | 1.08531  |     0.237545 |

대표 투자자는 군집을 학습시키는 정답 라벨이 아니라 결과를 검증하는 참고용이다. 예를 들어 Berkshire Hathaway가 소수 대형 종목에 집중된 포트폴리오 특성 때문에 BALANCED/AGGRESSIVE로 분류되더라도 군집을 강제로 수정하지 않았다 — 표의 volatility/mdd/beta/sector_hhi 값이 그 이유를 설명한다. 이는 공개 지분증권 포트폴리오의 정량적 위험 특성이 그 기관의 투자철학이나 전체 자산배분과 다를 수 있음을 보여줄 뿐, 군집화 로직의 오류가 아니다.

## 9. 결과 파일

```text
artifacts/13f_clustering/
├── raw/{filings,holdings}/
├── cache/{ticker_mapping,prices,sectors}/
├── processed/{portfolios_100,portfolio_metrics,excluded_portfolios}.csv
├── clustering/{cluster_assignments,cluster_centroids,clustering_evaluation}.csv, model.joblib
├── figures/*.png
└── report.md
```

## 10. 재현 실행 명령

```bash
python scripts/run_13f_clustering.py \
  --quarter 2026-06-30 \
  --num-portfolios 100 \
  --lookback-days 252 \
  --selection-method aum_stratified \
  --output-dir artifacts/13f_clustering
```

환경변수:

```env
SEC_USER_AGENT="YourAppName your-email@example.com"
OPENFIGI_API_KEY=  # 선택 — 없어도 동작하지만 rate limit이 더 낮음
```

## 11. 결과 해석 시 주의사항

- 이 결과는 **13F 공개 주식 포트폴리오 기반 위험 운용성향**이며, 투자자의 전체 자산배분이나 투자철학을 대표하지 않는다.
- 13F는 분기 말 스냅샷이며 분기 중 매매, 공매도, 파생상품(옵션 제외분), 채권/현금/비상장자산은 반영되지 않는다.
- 고정비중을 과거로 투영하는 근사이므로, 실제 그 기간 동안의 진짜 수익률과 다를 수 있다.
- CUSIP -> ticker 매핑 실패분은 포트폴리오에서 제외 후 재정규화했으므로, 매핑되지 않은 자산의 위험 특성은 반영되지 않는다.
- 군집 이름(DEFENSIVE/BALANCED/AGGRESSIVE)은 사후 해석이며, 특히 K=3이 데이터에 자연스러운 군집 구조가 아닐 경우(위 K별 평가표 참고) 경계선에 있는 포트폴리오의 분류는 민감할 수 있다.
