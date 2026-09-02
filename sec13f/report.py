"""Generates artifacts/13f_clustering/report.md from an actual pipeline run
(Section 13-14) — every number in this file is read back from the CSVs the
pipeline just wrote, never hand-typed, so the report can't drift from what
actually ran.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sec13f import config
from sec13f.pipeline import PipelineResult


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}" if pd.notna(x) else "N/A"


def write_report(result: PipelineResult, args) -> Path:
    paths = result.paths
    portfolio_metrics = pd.read_csv(paths["processed"] / "portfolio_metrics.csv")
    assignments = pd.read_csv(paths["clustering"] / "cluster_assignments.csv")
    centroids = pd.read_csv(paths["clustering"] / "cluster_centroids.csv")
    evaluation = result.evaluation_df if result.evaluation_df is not None else pd.DataFrame()
    excluded = pd.read_csv(paths["processed"] / "excluded_portfolios.csv") if \
        (paths["processed"] / "excluded_portfolios.csv").exists() else pd.DataFrame()
    rep_df = pd.read_csv(paths["clustering"] / "representative_investors.csv")

    lines: list[str] = []
    A = lines.append

    A("# 13F 공개 주식 포트폴리오 기반 위험 운용성향 — 군집화 리포트")
    A("")
    A(f"생성 시각(UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    A("")
    A(
        "이 리포트가 분류하는 것은 **개인·기관 투자자의 전체 투자성향이 아니라**, "
        "SEC EDGAR에 공개된 13F 롱 포지션 스냅샷 하나에서 계산한 **정량적 위험 특성**입니다. "
        "13F는 미국 상장 지분증권(및 관련 옵션)만 보고 대상이며, 채권·현금·비상장자산·해외직접상장주식·"
        "공매도(숏) 포지션은 13F에 나타나지 않습니다. 따라서 동일 기관이라도 실제 총자산 배분과 "
        "이 리포트의 위험 프로파일은 다를 수 있습니다."
    )
    A("")

    A("## 1. 분석 기준 (Section 1)")
    A("")
    A(f"- 요청한 분기: `{result.quarter_info.get('requested_quarter')}`")
    A(f"- 실제 사용한 분기(기준일): **{result.quarter_info.get('effective_quarter')}**")
    A(f"- 요청 분기 사용 가능 여부: {result.quarter_info.get('requested_quarter_available')}")
    if not result.quarter_info.get("requested_quarter_available"):
        A(f"- 대체 사유: {result.quarter_info.get('fallback_reason')}")
    A(f"- 사용한 SEC 13F 데이터셋 윈도우: `{result.quarter_info.get('dataset_window')}` "
      f"({result.quarter_info.get('dataset_url')})")
    A(f"- 가격 분석기간(lookback): {args.lookback_days} 거래일, 모든 포트폴리오 동일 기준일/동일 기간 적용")
    A(
        "- 판단 로직: SEC가 발행하는 분기별 Form 13F 구조화 데이터셋 목록에서 가장 최신 윈도우부터 "
        "COVERPAGE.tsv의 `REPORTCALENDARORQUARTER` 최빈값을 그 윈도우의 '완전히 제출이 끝난 공통 분기'로 "
        "판단하고, 요청한 분기와 일치하는 윈도우를 찾을 때까지 더 오래된 윈도우로 내려간다(요청 분기의 "
        "45일 신고 마감이 아직 지나지 않았거나 SEC가 해당 윈도우를 아직 발행하지 않은 경우 자동으로 대체됨)."
    )
    A("")

    A("## 2. 데이터 출처 및 포트폴리오 선정 (Section 2-3)")
    A("")
    A("- 출처: SEC EDGAR 공식 Form 13F 구조화 데이터셋 "
      "(https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets), "
      "`SUBMISSION.tsv`/`COVERPAGE.tsv`/`INFOTABLE.tsv`.")
    A(f"- 선정 방식: `{args.selection_method}` "
      f"({'총 신고가액 상위' if args.selection_method == config.SELECTION_TOP_VALUE else 'AUM 구간별 층화표본'})")
    A(f"- 스캔한 후보(랭킹된 필터 통과 기관) 수: {result.n_candidates_scanned}")
    A(f"- 최종 확보한 포트폴리오 수: **{result.n_selected}** / 목표 {args.num_portfolios}")
    A(f"- 제외된 포트폴리오 수: {result.n_excluded}")
    if not excluded.empty and "exclusion_reason" in excluded.columns:
        A("")
        A("제외 사유별 건수:")
        A("")
        for reason, count in excluded["exclusion_reason"].value_counts().items():
            A(f"- `{reason}`: {count}건")
    A("")
    A(
        "- 13F-HR/A 수정공시 처리: 동일 (CIK, PERIODOFREPORT)에서 FILING_DATE가 가장 최근인 제출만 사용. "
        "**알려진 단순화**: AMENDMENTTYPE이 'NEW HOLDINGS'(원본에 누락된 보유를 추가하는 부분 수정)인 "
        "경우도 '최신 제출본 단독 사용'으로 처리하며, 원본과 병합하지 않는다 — 원본을 완전히 대체하는 "
        "'RESTATEMENT' 성격의 수정에는 정확하지만, 드문 부분 수정 케이스에서는 일부 보유가 과소집계될 "
        "수 있다."
    )
    A(
        "- PUT/옵션 포지션 처리: INFOTABLE의 `PUTCALL`이 비어있지 않은 행(PUT, CALL 모두)은 기본 "
        "포트폴리오에서 제외했다. CALL도 함께 제외한 이유는, 13F 규정상 옵션 행의 VALUE가 옵션 자체의 "
        "시가가 아니라 '기초 주식 수량에 대한 가치'로 보고되어 실제 매수 포지션과 경제적으로 동일하지 "
        "않기 때문이다(해석이 불명확한 포지션에 해당)."
    )
    A("")

    A("## 3. Ticker 매핑 (Section 3)")
    A("")
    if "mapping_coverage" in portfolio_metrics.columns:
        A(f"- 평균 매핑 커버리지: {_fmt_pct(portfolio_metrics['mapping_coverage'].mean())}")
        A(f"- 최소 매핑 커버리지: {_fmt_pct(portfolio_metrics['mapping_coverage'].min())}")
    A(f"- 매핑 커버리지 {config.MIN_MAPPING_COVERAGE:.0%} 미만 포트폴리오는 최종 군집화에서 제외하고 "
      f"다음 순위 후보로 대체했다(대체 여력을 위해 목표치의 {args.candidate_pool_multiplier}배까지 "
      f"후보 풀을 확장).")
    A("- CUSIP -> ticker 매핑은 OpenFIGI `/v3/mapping`을 사용했고, 결과는 "
      "`artifacts/13f_clustering/cache/ticker_mapping/cusip_ticker_map.json`에 캐시되어 재실행 시 "
      "재조회하지 않는다. Issuer 이름 유사도로 임의 추정한 매핑은 없다 — 실패한 CUSIP은 unmapped로 기록만 한다.")
    A("")

    A("## 4. 4개 위험지표 요약통계 (Section 5-6)")
    A("")
    if result.feature_summary is not None:
        A(result.feature_summary.to_markdown())
    A("")
    A(
        "- **Volatility**: 포트폴리오 일별 수익률 표준편차 x sqrt(252)\n"
        "- **MDD**: 누적수익률 곡선의 최고점 대비 최대 하락폭, 양의 크기로 저장\n"
        "- **Beta**: SPY 대비 Cov(R_p, R_m) / Var(R_m)\n"
        "- **Sector HHI**: 업종별(개별 종목이 아닌 GICS 유사 sector 단위) 비중 제곱합. "
        "ETF/펀드처럼 yfinance에 GICS sector가 원래 없는 보유는 별도 'ETF/Fund' 버킷으로, "
        "재시도 후에도 끝내 응답을 받지 못한 종목만 'Unknown' 버킷으로 묶어 HHI에 포함했다(제거하지 않음)."
    )
    A("")
    A(
        "- **알려진 근사**: 기준일 시점 13F 스냅샷의 고정비중을 기준일 직전 "
        f"{args.lookback_days}거래일에 그대로 적용해 과거 수익률을 재구성한다. 실제로는 그 기간 동안 "
        "기관이 매매를 했을 것이므로, 이는 '기준일의 보유 구성이 분석기간 내내 유지되었다면'이라는 근사다."
    )
    A("")

    A("## 5. 전처리 (Section 6)")
    A("")
    A(f"- Winsorization: 하위/상위 {config.WINSOR_LOWER_QUANTILE:.0%}/{config.WINSOR_UPPER_QUANTILE:.0%} 분위수로 clip, "
      "적용 전/후 버전 모두 `processed/portfolio_metrics.csv` / `portfolio_metrics_winsorized.csv`로 보존")
    A(f"- 최종 사용 스케일러: **{args.scaler}**"
      f"{'(이상치에 상대적으로 강건)' if args.scaler == 'robust' else ''}")
    A("- 네 지표 모두 '값이 클수록 위험'으로 방향을 통일했다(HHI가 낮을수록 분산 = 저위험, "
      "높을수록 집중 = 고위험이므로 별도 부호 반전 없이 그대로 사용).")
    A("")

    A("## 6. 군집화 모델 비교 (Section 7)")
    A("")
    if not evaluation.empty:
        A(evaluation.to_markdown(index=False))
    A("")
    A(f"- **최종 선택 모델**: {result.chosen_method.upper()}, K={result.chosen_k}")
    k3 = evaluation[evaluation["k"] == 3] if not evaluation.empty else pd.DataFrame()
    if not k3.empty:
        A("")
        A(
            "K=3(서비스에 필요한 DEFENSIVE/BALANCED/AGGRESSIVE 3분류)의 실제 평가지표는 위 표의 k=3 행을 "
            "참고. Silhouette/Calinski-Harabasz/Davies-Bouldin이 다른 K보다 항상 더 좋다는 보장은 없으며, "
            "표에 K=2..6 전체를 숨기지 않고 보고했다 — K=3이 다른 K보다 지표상 열세라면 이 표에서 그대로 "
            "드러난다."
        )
    A("")

    A("## 7. 군집별 표본 수와 중심값 (Section 7)")
    A("")
    A(centroids.to_markdown(index=False))
    A("")
    cluster_counts = assignments["risk_profile"].value_counts()
    for name, count in cluster_counts.items():
        A(f"- **{name}**: {count}개 포트폴리오")
    A("")
    A(
        "**중요**: DEFENSIVE/BALANCED/AGGRESSIVE는 사람이 직접 라벨링한 정답이 아니라, 4개 위험지표로 "
        "비지도 군집화(K-means)를 수행한 뒤 각 군집 중심의 `ClusterRisk_k = mean(Z_Vol, Z_MDD, Z_Beta, "
        "Z_HHI)`가 낮은 순서대로 이름을 붙인 **사후 해석**이다."
    )
    A("")

    A("## 8. 대표 투자자 Sanity Check (Section 8)")
    A("")
    A(rep_df.to_markdown(index=False))
    A("")
    A(
        "대표 투자자는 군집을 학습시키는 정답 라벨이 아니라 결과를 검증하는 참고용이다. 예를 들어 "
        "Berkshire Hathaway가 소수 대형 종목에 집중된 포트폴리오 특성 때문에 BALANCED/AGGRESSIVE로 "
        "분류되더라도 군집을 강제로 수정하지 않았다 — 표의 volatility/mdd/beta/sector_hhi 값이 그 이유를 "
        "설명한다. 이는 공개 지분증권 포트폴리오의 정량적 위험 특성이 그 기관의 투자철학이나 전체 자산배분과 "
        "다를 수 있음을 보여줄 뿐, 군집화 로직의 오류가 아니다."
    )
    A("")

    A("## 9. 결과 파일")
    A("")
    A("```text")
    A("artifacts/13f_clustering/")
    A("├── raw/{filings,holdings}/")
    A("├── cache/{ticker_mapping,prices,sectors}/")
    A("├── processed/{portfolios_100,portfolio_metrics,excluded_portfolios}.csv")
    A("├── clustering/{cluster_assignments,cluster_centroids,clustering_evaluation}.csv, model.joblib")
    A("├── figures/*.png")
    A("└── report.md")
    A("```")
    A("")

    A("## 10. 재현 실행 명령")
    A("")
    A("```bash")
    A(f"python scripts/run_13f_clustering.py \\")
    A(f"  --quarter {args.quarter} \\")
    A(f"  --num-portfolios {args.num_portfolios} \\")
    A(f"  --lookback-days {args.lookback_days} \\")
    A(f"  --selection-method {args.selection_method} \\")
    A(f"  --output-dir {args.output_dir}")
    A("```")
    A("")
    A("환경변수:")
    A("")
    A("```env")
    A('SEC_USER_AGENT="YourAppName your-email@example.com"')
    A("OPENFIGI_API_KEY=  # 선택 — 없어도 동작하지만 rate limit이 더 낮음")
    A("```")
    A("")

    A("## 11. 결과 해석 시 주의사항")
    A("")
    A(
        "- 이 결과는 **13F 공개 주식 포트폴리오 기반 위험 운용성향**이며, 투자자의 전체 자산배분이나 "
        "투자철학을 대표하지 않는다.\n"
        "- 13F는 분기 말 스냅샷이며 분기 중 매매, 공매도, 파생상품(옵션 제외분), 채권/현금/비상장자산은 "
        "반영되지 않는다.\n"
        "- 고정비중을 과거로 투영하는 근사이므로, 실제 그 기간 동안의 진짜 수익률과 다를 수 있다.\n"
        "- CUSIP -> ticker 매핑 실패분은 포트폴리오에서 제외 후 재정규화했으므로, 매핑되지 않은 자산의 "
        "위험 특성은 반영되지 않는다.\n"
        "- 군집 이름(DEFENSIVE/BALANCED/AGGRESSIVE)은 사후 해석이며, 특히 K=3이 데이터에 자연스러운 "
        "군집 구조가 아닐 경우(위 K별 평가표 참고) 경계선에 있는 포트폴리오의 분류는 민감할 수 있다."
    )
    A("")

    text = "\n".join(lines)
    out_path = paths["root"] / "report.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path
