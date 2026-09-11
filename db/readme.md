# 뉴스 DB

`news.db`는 molba2see/Personal-Port-Color에서 가져온 원본 스냅샷입니다.
2025-10-31~2025-11-13 기사·요약 1,113건, 101개 종목과 기존 검색 벡터를 포함합니다.
출처와 체크섬은 [통합 기록](../docs/REPOSITORY_INTEGRATION.md)에 있습니다.

프로젝트 루트에서 작업 DB를 준비합니다.

```bash
python3 scripts/prepare_news_db.py
python3 retrieval.py --ticker NVDA --compare-styles --lookback-days 400
python3 scripts/validate_news_retrieval.py
```

원본은 그대로 두고 `local_data/news/db/news.db`에 컬럼·종목 연결과
384차원 MiniLM 텍스트 임베딩을 추가합니다. 기존 256차원 결합 벡터는 보존하지만
새 검색기가 텍스트 벡터로 사용하지 않습니다. 앱과 모델 추론은 준비된 작업 DB를 우선 읽습니다.
준비 명령은 재실행 시에도 원본의 새 기사를 작업 DB에 합칩니다. 같은 기사 ID는 중복 추가하지 않고,
작업 DB에만 있는 기사는 보존합니다. 같은 ID의 내용이 다르면 파일 교체 없이 중단합니다.

이후 데이터 갱신도 같은 작업 DB에 기록합니다.

```bash
python3 db/db.py crawl --use-dynamic --refresh-tickers
python3 db/db.py preprocess
python3 db/db.py summarize
python3 db/db.py build-embeddings
python3 db/db.py assign-events
```

요약에는 `OPENAI_API_KEY`가 필요합니다. 기존 요약을 재사용하는 준비·검색에는 필요하지 않습니다.
`NEWS_DB_BASE`를 설정하면 앱·추론·수집기의 작업 디렉터리가 함께 변경됩니다.
첫 임베딩 모델 실행에는 모델 다운로드가 필요할 수 있습니다.
 
