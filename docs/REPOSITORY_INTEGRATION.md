# 두 저장소 비교와 뉴스 DB 통합

## 비교 기준

- 원본: [molba2see/Personal-Port-Color](https://github.com/molba2see/Personal-Port-Color),
  `main`, 커밋 `4fceec97d534fa7b4a404e0b8c4b42c6521e9eac`.
- 반영 대상: [leesangwon393/Personal-Portfolio-Report](https://github.com/leesangwon393/Personal-Portfolio-Report),
  `feature/13f-risk-clustering`, 작업 시작 커밋 `5cba6eed1277d0307132003275d334d1255a54b5`.
- 성향별 검색 구현 출처: 사용자 저장소의 `feature/style-aware-rag-retrieval`,
  커밋 `7924f1631fa170772117632a6c168ea40ad87aaa`.

원본 저장소 전체 파일 트리의 Git blob 해시를 로컬 파일과 비교했다.
이번 통합은 뉴스 DB와 성향별 검색에 필요한 파일에 한정했다.

## 가져온 항목과 유지한 항목

| 항목 | 출처 | 처리 |
|---|---|---|
| `db/news.db` | 원본 저장소 | 원본 바이트 그대로 추가. 기사·요약 1,113건, 101개 종목, 기존 검색 벡터 1,113건 |
| `retrieval.py`, `rag_config.py` | 사용자 저장소 검색 브랜치 | 공통 쿼리 + 성향별 쿼리 검색, 병합·중복 제거·관련성/최신성 순위 로직 재사용 |
| `db/db.py` | 사용자 저장소 검색 브랜치 | 기사별 임베딩, 종목 연결, 이벤트 그룹 기능 재사용. 기존 스키마 자동 확장 및 공유 경로 보완 |
| `model_inference.py` | 사용자 저장소 검색 브랜치 | 선별 기사를 프롬프트에 연결. RSS 원문 날짜·제목 보존, 모델 import 지연, 과거 뉴스 취급 지침 추가 |
| 검색·DB·프롬프트 테스트 | 사용자 저장소 검색 브랜치 | 재사용하고 새 대시보드 연결·기존 DB 준비·장애 처리 테스트 추가 |
| FIQA/TFNS LoRA 가중치·토크나이저 | 양쪽 저장소 동일 | Git blob 비교에서 동일하므로 기존 파일 유지 |
| 위험 계산용 시장 CSV, New_data 스크립트 등 | 양쪽 저장소 동일 | 기존 파일 유지 |
| 기존 위험 계산·앱·13F 코드 | 사용자 저장소 | 과거 저장소 코드로 덮어쓰지 않음. 앱의 뉴스 연결 부분만 수정 |
| 과거 학습·추론·평가 노트북, 과거 재무 CSV | 원본 저장소 | 현재 뉴스 연결에 필요한 실행 코드·데이터가 아니므로 미추가 |

## 새로 작성한 연결 코드

- `news_paths.py`: 앱·추론·검색·수집기의 DB 경로를 통일한다.
- `scripts/prepare_news_db.py`: 원본을 작업 DB에 복사하고 컬럼·종목 연결·임베딩을 준비한다.
  기존 작업 DB가 있으면 그 내용을 유지하며, 변환 성공 후 파일을 교체한다.
- `scripts/validate_news_retrieval.py`: 실제 원본/작업 DB 데이터 보존과 세 성향의 검색,
  Flask 분석 API 연결을 검증하고 JSON을 남긴다.
- 대시보드: 계산된 `category`를 검색에 전달하고, 의미 검색 실패나 과거 기사 사용 여부를 표시한다.
- RSS/구형 DB: 저장 임베딩이 없어도 기사 후보를 임베딩해 성향별로 선별한다.
- `.gitignore`: 원본 `db/news.db`만 예외로 추적한다. `local_data/` 작업 DB는 제외한다.

## 원본 DB 보존

- 원본 경로: [`db/news.db`](https://github.com/molba2see/Personal-Port-Color/blob/4fceec97d534fa7b4a404e0b8c4b42c6521e9eac/db/news.db)
- 크기: 11,632,640바이트
- SHA-256: `f4dd0d8554b3cdc6cee197ee37be2498c0fb4b1f5e411c00e0b07cf5d5cce496`
- 기사 날짜: 2025-10-31 ~ 2025-11-13
- 작업 DB: `local_data/news/db/news.db` (`NEWS_DB_BASE` 설정 시 변경)

기존 `integrated_index`는 텍스트·종목·시간을 결합한 256차원 투영 벡터다.
새 검색기의 384차원 MiniLM 텍스트 벡터와 직접 비교할 수 없으므로 작업 DB의
`article_embeddings`에 새로 생성한다. 기존 256차원 벡터도 작업 DB에 보존한다.
기사·요약을 재수집하거나 다시 요약하지 않았다.

## 검증 결과

- 오프라인 자동 테스트 54개 통과.
- 기존 13F·earnings-call·MDD 회귀 검사에서 pytest 테스트 26개 통과.
- 실제 캐시된 `all-MiniLM-L6-v2` 모델로 기사 1,113건의 새 임베딩 생성.
- 원본 기사 ID·제목·종목·날짜·요약과 기존 벡터가 작업 DB에서도 전부 동일함을 확인.
- NVDA/AAPL/MSFT에 SAFE/NEUTRAL/AGGRESSIVE를 각각 적용해 3건씩 검색.
- SAFE와 AGGRESSIVE의 기사 집합 Jaccard 겹침: NVDA 0.5, AAPL 1.0, MSFT 0.5.
  NVDA/MSFT는 3건 중 2건이 겹치며 AAPL은 같은 3건을 선택했다.
- 실제 DB를 사용한 Flask `/api/analyze` 요청에서 HTTP 200, 뉴스 3건과 과거 뉴스 안내를 확인했다.
  가격·재무 외부 조회는 모의 응답을 사용하므로 시세 API 검증 결과는 아니다.

상세 기사·점수·DB 검증 결과: [news_retrieval_validation.json](news_retrieval_validation.json).
검색 결과 차이는 연결이 동작한다는 증거이며, 관련성의 우수성이나 투자 효용 검증은 아니다.
스냅샷은 2025년 데이터이므로 현재 사건으로 해석하지 않는다.

## 재현

```bash
python3 scripts/prepare_news_db.py
python3 -m unittest discover -s tests -v
python3 scripts/validate_news_retrieval.py
python3 retrieval.py --ticker NVDA --compare-styles --lookback-days 400
```

모델이 캐시돼 있으면 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`로 네트워크 없이 검증할 수 있다.
