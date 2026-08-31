# Interview Q&A — RAG 개인화 뉴스 검색 개선

이 문서는 `retrieval.py` / `db/db.py` / `model_inference.py`에 실제로 구현된
내용을 근거로 작성한 예상 질문/답변입니다. 각 답변은 코드의 실제 함수/구조를
가리키므로, 면접에서 꼬리질문이 오면 해당 파일을 열어 바로 보여줄 수 있습니다.

---

### 왜 최신 뉴스 10개를 그냥 넣지 않았나요?

기존 `model_inference.py`의 `load_news()`는 해당 ticker의 기사를
`ORDER BY pubdate DESC LIMIT 10`으로만 가져왔습니다. 이 방식은 사용자
투자성향과 무관하게 항상 같은 뉴스를 넣고, 실적 발표처럼 특정 이벤트가
터진 주에는 Top-10이 전부 같은 사건의 재탕 기사로 채워질 수 있습니다.
그래서 recency만 보던 로딩 로직을, 질의(query)와의 의미적 관련성 +
투자성향 + 최신성 + 이벤트 다양성을 함께 고려하는 retrieval 파이프라인
(`retrieval.py`의 `retrieve_news()`)으로 바꿨습니다. 기존 recency 로직은
버리지 않고 `fetch_recent_articles_fallback()`으로 남겨, RAG가 실패했을 때
안전망(fallback)으로 재사용합니다.

### 왜 RAG를 사용했나요?

리포트 생성에 필요한 사실(최근 실적, 리스크, 신제품 등)은 LLM이 학습 시점
이후의 정보를 모르기 때문에 외부 지식(뉴스 DB)에서 가져와야 하고, 이 값이
매번 바뀌므로 모델을 재학습시키는 대신 검색으로 최신 컨텍스트를 주입하는
것이 합리적입니다. 또한 "어떤 뉴스를 넣을지"를 검색 단계에서 결정하면,
투자성향별로 다른 정보를 우선하는 개인화가 LLM의 지시문(prompt)에만
의존하지 않고 실제로 입력되는 근거 자체가 달라지도록 만들 수 있습니다.
`retrieve_news()`가 반환하는 근거(headline/summary/date)만 프롬프트에
넣고 "제공된 사실 외에는 만들어내지 말라"는 지시를 System Prompt에 유지해,
hallucination을 줄이는 근거로도 RAG를 씁니다.

### 왜 SAFE / NEUTRAL / AGGRESSIVE별 Retrieval Query를 다르게 했나요?

개인화가 Generation(LLM 프롬프트) 단계에서만 일어나면, LLM에 넣어주는 뉴스
자체는 모든 사용자에게 동일하고 "해석"만 달라집니다. 하지만 SAFE 투자자는
부채·마진 악화·규제 리스크 같은 하방 정보를, AGGRESSIVE 투자자는 성장
촉매·신제품·가이던스 상향 같은 상방 정보를 더 우선적으로 볼 필요가
있습니다. 그래서 `build_retrieval_query(ticker, investor_style)`가
`STYLE_QUERY_KEYWORDS`(SAFE/NEUTRAL/AGGRESSIVE별 키워드 리스트)를 이용해
서로 다른 검색 쿼리 문자열을 만들고, 이 쿼리의 MiniLM 임베딩으로 유사도를
계산하기 때문에 Retrieval 단계에서부터 성향별로 다른 기사가 상위로
올라옵니다. 다만 어느 한쪽 정보를 완전히 배제하지는 않도록 "우선순위만
다르게" 설계했습니다(SAFE에게 호재를 숨기거나 AGGRESSIVE에게 악재를
숨기지 않음).

### 왜 Ticker를 Embedding하지 않고 Metadata Filter로 사용했나요?

Ticker를 임베딩 벡터로 만들어 텍스트 임베딩과 합치면(과거 `integrated_index`의
Random Projection 방식처럼) "NVDA와 관련 있다"는 사실이 벡터 공간의 방향으로
녹아들어가고, 그 결합 비율(가중치)을 조정하지 않는 한 검색 결과를 설명하기
어려워집니다. 반면 ticker는 사실 이진적인 조건(이 기사가 이 종목과 관련
있는가/없는가)이라서, `article_tickers` 테이블을 조인해 후보를 먼저 좁히는
것이 훨씬 단순하고 정확합니다. 하나의 기사가 NVDA와 MSFT 모두와 관련 있으면
양쪽 ticker의 후보 집합에 각각 포함되어야 하는데, 이는 다대다 관계
테이블로 자연스럽게 표현되지만 벡터 결합 방식으로는 표현하기 까다롭습니다.
그래서 `fetch_ticker_candidates()`가 SQL `WHERE ticker = ?` 조건으로 먼저
후보를 걸러낸 뒤에만 semantic similarity를 계산하도록 했습니다.

### 왜 Recency를 Embedding에 넣지 않고 Ranking Score로 사용했나요?

시간(recency)을 임베딩에 섞으면 "오늘 발행된, 별로 관련 없는 기사"가
"1주일 전 발행된, 매우 관련 있는 기사"보다 벡터 거리상 더 가까워지는 등
의미적 유사도와 최신성이 서로를 오염시킬 수 있고, 그 비율을 조정해도
결과를 설명하기 어렵습니다. 그래서 `calculate_recency_score()`가
`exp(-ln2 * days_old / HALF_LIFE_DAYS)`로 독립적인 recency score를 계산하고,
`final_score = SEMANTIC_WEIGHT * semantic_score + RECENCY_WEIGHT * recency_score`
(`rag_config.py`의 `SEMANTIC_WEIGHT=0.8`, `RECENCY_WEIGHT=0.2`)처럼 두 점수를
사후에 선형 결합합니다. 이렇게 하면 "이 기사가 왜 상위에 올라왔는지"를
의미적 유사도와 최신성으로 분리해서 설명할 수 있고, 가중치도 코드 수정
없이 config 값 하나로 조정할 수 있습니다.

### 왜 뉴스 중복 제거가 필요한가요?

중복 뉴스는 두 가지 문제를 만듭니다. 첫째, 같은 기사가 ticker마다 별도
row로 저장되면 DB가 불필요하게 커지고 임베딩도 중복 계산됩니다. 둘째,
Top-K 결과가 사실상 같은 내용의 기사로 도배되면 LLM에게 주는 정보의
다양성이 줄어들어 리포트가 한쪽 사실만 반복해서 언급하게 됩니다. 그래서
①원문이 같은 기사는 애초에 한 번만 저장하고(`upsert_articles_normalized`),
②표현이 다르지만 같은 사건을 다루는 기사는 검색 결과에서 한 개만
선택되도록(`assign_event_groups` + retrieval의 event dedup) 두 단계로
나눠서 처리했습니다.

### Exact duplicate와 Semantic/Event duplicate의 차이는 무엇인가요?

Exact duplicate는 "같은 URL" 또는 "같은 content_hash(정규화한
headline+summary의 SHA-256)"처럼 문자열 수준에서 완전히 같은 원문을
말합니다 — 이건 ingestion 시점(`upsert_articles_normalized`)에 DB에
저장하기도 전에 걸러냅니다. Semantic/event duplicate는 문자열은 다르지만
같은 사건을 다룬 기사입니다. 예를 들어 "Nvidia beats earnings
expectations"와 "Nvidia quarterly results top estimates"는 URL도
headline도 다르지만 같은 실적 발표를 가리킵니다. 이건 문자열 비교로는
잡을 수 없기 때문에 MiniLM 임베딩의 cosine similarity(threshold 0.90) +
발행일 차이(±2일) + 관련 ticker 일치라는 세 조건을 함께 써서
`assign_event_groups()`가 사후에 `event_group_id`로 묶습니다. 저장은
그대로 두고(삭제하지 않고) 태깅만 하는 것이 핵심 차이입니다.

### 왜 동일 이벤트 기사 중 하나만 Top-K에 넣나요?

Top-K는 슬롯이 제한적입니다(기본 6개). 만약 6개 중 4개가 같은 실적 발표를
다룬 기사라면, 사용자는 사실상 "실적 발표"라는 정보 하나만 4번 반복해서
받는 셈이고, 다른 중요한 뉴스(리스크, 신제품, 규제 등)가 밀려납니다.
그래서 `retrieve_news()`는 semantic+recency로 점수를 매겨 정렬한 뒤,
같은 `event_group_id`를 가진 기사들 중 점수가 가장 높은 1개만 남기고
나머지는 건너뛰는 event deduplication을 Top-K 선택 직전에 적용합니다.
완벽한 이벤트 클러스터링이 목표가 아니라, Top-K가 한 사건으로 도배되는
것을 막아 정보 다양성을 확보하는 것이 목표이기 때문에 이 정도의 단순한
규칙(유사도+날짜창)으로 충분하다고 판단했습니다.

### Random Projection은 학습인가요?

아닙니다. 기존 `db/db.py`의 `integrated_index` 경로는 text embedding(384차원)
+ ticker vector(32차원) + recency vector(32차원)를 이어붙인 뒤, 고정된
시드(`PROJ_SEED`)로 생성한 랜덤 가우시안 행렬을 곱해 256차원으로 축소하는
방식입니다. 이 행렬은 데이터로부터 최적화(gradient descent 등)되지 않고
난수로 한 번 고정되어 재사용되는 값이라서, 학습(learning) 과정이 아니라
비학습(non-learned) 선형 차원 축소 기법입니다. Johnson-Lindenstrauss
lemma에 근거해 "무작위 투영도 거리를 대략 보존한다"는 성질을 이용한
것뿐이지, 파라미터를 데이터에 맞춰 조정하는 과정은 전혀 없습니다.

### 왜 Random Projection을 최종 Retrieval에서 제거했나요?

두 가지 이유입니다. 첫째, ticker/recency를 벡터에 섞고 축소하면 최종
유사도 점수가 "의미적 유사도인지 ticker 일치인지 최신성인지"를 분리해서
설명할 수 없어, 왜 이 기사가 상위에 나왔는지 디버깅하기 어렵습니다.
Ticker는 metadata filter로, recency는 별도 ranking score로 빼면 각
요소의 기여도를 독립적으로 확인하고 조정할 수 있습니다. 둘째, ticker를
벡터에 섞으면 "하나의 기사가 여러 ticker와 연결된다"는 요구사항을
표현하기 까다로워지는데(어느 ticker 벡터를 합칠지 애매해짐), metadata
filter는 다대다 관계 테이블(`article_tickers`)로 자연스럽게 이를
지원합니다. 그래서 새 RAG 경로(`retrieval.py`)는 Random Projection을
전혀 쓰지 않고, 기존 코드는 삭제하지 않은 채 `db/db.py`에 "LEGACY /
DEPRECATED" 주석과 함께 하위 호환용으로만 남겨뒀습니다.
