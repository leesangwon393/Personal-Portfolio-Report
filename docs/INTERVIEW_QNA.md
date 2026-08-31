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

### 왜 하나의 Style Query를 사용하지 않았나요?

처음에는 `build_style_query(ticker, investor_style)`가
`"NVDA financial stability debt cash flow stability downside risk
regulatory risk ..."`처럼 한 성향의 모든 관심사를 한 문장으로 이어붙여
embedding했습니다. 하지만 "재무 안정성"과 "하방위험"과 "규제 리스크"는
서로 다른 검색 의도인데, 이걸 하나의 벡터로 합치면 각 의도가 서로
희석되면서 특정 의도 하나에 유독 강하게 매칭되는 기사도, 세 의도 모두에
약하게 걸치는 기사도 구분 없이 비슷한 점수를 받게 됩니다. 또 "이 기사가
왜 검색됐는지"를 설명하려면 어떤 개념이 점수에 기여했는지가 필요한데, 문장
하나로 합쳐진 쿼리에서는 그걸 분리할 수 없습니다. 그래서 지금은
`build_style_facet_queries(ticker, investor_style)`가 성향별로 2~3개의
좁은 Facet Query(예: SAFE의 stability/downside_risk/external_risk)를
따로 만들고, 각각을 독립적으로 embedding·검색한 뒤 결과를 병합합니다.

### 왜 Core Query를 별도로 두었나요?

개인화가 Style Query에만 의존하면, SAFE 사용자에게는 위험 관련 기사만,
AGGRESSIVE 사용자에게는 성장 관련 기사만 검색되는 정보 편향이 생길 수
있습니다. 하지만 실적 발표처럼 투자성향과 무관하게 모든 투자자가 반드시
봐야 하는 뉴스도 있습니다. 그래서 `build_core_query(ticker)`가
`investor_style` 파라미터 없이 `"earnings revenue profitability cash flow
guidance financial performance"`라는 고정 쿼리를 만들고, 이 쿼리는 항상
검색 대상에 포함됩니다. `retrieve_news()`가 반환하는 `core_hit_count`는
Top-K 중 Core Query 덕분에 뽑힌 기사 수를 보여주는 diagnostic이라서,
"핵심 실적 뉴스가 실제로 Top-K에 살아남고 있는지"를 수치로 확인할 수
있습니다.

### 왜 Facet Query를 여러 개 사용했나요?

한 성향의 관심사도 사실 여러 갈래입니다. SAFE는 "재무 안정성"뿐 아니라
"하방위험(실적 미스, 마진 악화)"과 "외부 리스크(규제, 경쟁, 공급망)"를
모두 신경 씁니다. 이 세 가지를 하나의 문장으로 합치는 대신
`STYLE_FACET_KEYWORDS["SAFE"]`처럼 facet마다 별도 키워드 그룹을 두고,
`build_style_facet_queries()`가 이를 각각 독립된 쿼리로 만들어 따로
검색합니다(SAFE/AGGRESSIVE 모두 3개, NEUTRAL도 3개). 결과적으로 한 종목에
대해 최대 `1(core) + N(facet)`개의 쿼리가 각자 Top-N(`CANDIDATES_PER_QUERY`,
기본 5)을 검색하고, `retrieve_news()`가 이를 article_id 기준으로 병합하며
어떤 기사가 어떤 query(들)에서 뽑혔는지 `matched_queries`에 기록합니다.
같은 기사가 여러 query의 Top-N에 동시에 들면 병합 시 1개 row로 합쳐지고
`matched_queries`만 누적되므로, 여러 관점에서 일관되게 중요한 기사일수록
자연스럽게 눈에 띕니다(뒤이은 reranking에서 유리해짐).

### 왜 검색 결과를 바로 쓰지 않고 Reranking했나요?

Multi-query 단계는 "후보를 넉넉히 모으는" 단계이지 최종 순위를 정하는
단계가 아닙니다. 같은 기사가 여러 query의 Top-N에 들어도 merge 직후에는
순서 정보가 없고, recency도 전혀 반영되지 않았습니다. 그래서 병합된 후보
전체에 대해 `semantic_score = max(cosine(query_vec, article_vec) for 해당
(ticker, investor_style)에서 검색한 모든 query_vec)`를 계산합니다 — 어느
한 query의 Top-N에 든 것으로 끝나지 않고, core를 포함한 모든 query
중에서 이 기사와 가장 가까운 값을 최종 semantic 신호로 씁니다. 여기에
recency_score를 더해 `final_score = SEMANTIC_WEIGHT(0.8) * semantic_score
+ RECENCY_WEIGHT(0.2) * recency_score`로 재정렬한 뒤에야 Event
Deduplication과 Top-K 선택이 이어집니다. 이 reranking이 없으면 어떤 query가
그 기사를 "먼저" 검색했는지 같은 우연에 최종 순위가 좌우됩니다.

### 투자성향별 정보 편향은 어떻게 방지했나요?

세 겹의 안전장치를 둡니다. 첫째, Core Query가 investor_style과 무관하게
항상 검색되므로 실적처럼 모두가 봐야 할 뉴스가 특정 성향에서만 사라지는
일이 구조적으로 어렵습니다. 둘째, Facet Query는 "우선순위만" 바꾸고
"배제"는 하지 않도록 설계했습니다 — SAFE facet 키워드 목록에 "성장"이라는
단어가 없다고 해서 성장 뉴스가 후보 풀에서 제외되는 게 아니라, 단지 SAFE
facet들의 Top-N에 덜 뽑힐 뿐이고 Core Query를 통해서는 여전히 후보가 될 수
있습니다. 셋째, `core_hit_count`/`style_facet_hit_count`(둘 다
`evaluation/retrieval_eval.py`) 같은 diagnostic으로 Top-K 구성을 계속
관찰할 수 있게 했습니다 — Section 14에서 "Top-K 중 최소 Core 기사 N개"
같은 rigid rule을 강제하는 대신, 우선 수치로 확인 가능하게 만드는 쪽을
택했습니다(강제 규칙은 relevance를 해칠 수 있다는 판단).

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
후보를 걸러낸 뒤에만 core/facet query들과의 semantic similarity를
계산하도록 했습니다 — ticker는 어떤 query에도 섞여 들어가지 않습니다.

### 왜 Recency를 반영했나요?

금융 뉴스는 의미적 관련성만으로 평가할 수 없습니다. 6개월 전 기사가
"NVDA 실적"이라는 주제와 아무리 잘 맞아도, 지금 리포트를 받는 투자자에게는
어제 나온 관련 뉴스가 훨씬 중요합니다. 그렇다고 시간을 임베딩에 섞으면
"오늘 발행된, 별로 관련 없는 기사"가 "1주일 전 발행된, 매우 관련 있는
기사"보다 벡터 거리상 더 가까워지는 등 의미적 유사도와 최신성이 서로를
오염시킬 수 있고, 그 비율을 조정해도 결과를 설명하기 어렵습니다. 그래서
`calculate_recency_score()`가 `exp(-ln2 * days_old / HALF_LIFE_DAYS)`로
독립적인 recency score를 계산하고, `final_score = SEMANTIC_WEIGHT *
semantic_score + RECENCY_WEIGHT * recency_score`(`rag_config.py`의
`SEMANTIC_WEIGHT=0.80`, `RECENCY_WEIGHT=0.20`, `HALF_LIFE_DAYS=14`)처럼
merge 이후 두 점수를 사후에 선형 결합합니다. 이렇게 하면 "이 기사가 왜
상위에 올라왔는지"를 semantic relevance와 최신성 두 갈래로 분리해서 설명할
수 있고, 가중치도 코드 수정 없이 config 값 하나로 조정할 수 있습니다. 두
값의 합이 1이 아니면 `rag_config.py`가 import 시점에 `warnings.warn()`으로
경고하지만 retrieval 자체는 그대로 동작합니다(하드 실패시키지 않은 이유는
최종 점수 스케일이 달라질 뿐 로직이 깨지는 건 아니기 때문).

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

### Multi-query candidate merge 단계에서 exact duplicate를 왜 또 검사하나요?

Ingestion(`upsert_articles_normalized`/`_lookup_existing_article_id`)이
이미 같은 canonical URL/content_hash를 가진 article은 새 article_id를
만들지 않고 기존 row를 재사용하므로, 정상 경로에서는 candidate merge
단계에 도달하는 두 article_id가 진짜 중복일 수 없습니다. 그럼에도
`retrieve_news()`는 merge 직후 `_drop_exact_duplicates()`로 content_hash
/URL 기준 재검사를 한 번 더 합니다 — content_hash가 나중에 추가된 컬럼이라
`db.py migrate-legacy`로 backfill되기 전의 오래된 row가 섞여 있을 수
있기 때문입니다. 정상 DB에서는 이 단계가 항상 0건을 제거해야 하고,
`retrieve_news()`가 반환하는 `exact_duplicate_count`로 실제로 0인지 확인할
수 있습니다.

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

### `fiqa`/`tfns` 어댑터는 정확히 어떤 데이터로 학습됐나요? earnings-call 데이터도 쓰였나요?

아니요. `fiqa` 어댑터는 `FinGPT/fingpt-fiqa_qa`(금융 Q&A), `tfns` 어댑터는
`zeroshot/twitter-financial-news-sentiment`(금융 뉴스/트윗 감성 분류)라는
서로 다른 Hugging Face 공개 데이터셋으로 `Meta-Llama-3-8B-Instruct`를 각각
LoRA fine-tuning한 결과입니다(과거 `fiqa-peft.ipynb`/`fingpt-peft.ipynb`
기준 — 두 노트북은 "Clean up project for app release" 커밋에서 앱 배포용
정리 과정 중 삭제되고 학습된 어댑터 가중치만 저장소에 남았습니다).
Earnings-call 원문(실적발표 콜 스크립트)은 이 두 어댑터 어느 쪽 학습에도
쓰이지 않았습니다. `Crawling`/`db`가 수집하는 뉴스 중에는 "Q1 2026 Earnings
Call Transcript"처럼 실적발표를 다룬 *기사*가 섞여 있지만, 이건 RAG가
검색하는 뉴스 코퍼스일 뿐 LoRA 학습 데이터와는 무관합니다. 한편
`New_data/make_text.py`, `add_text.py`가 만드는 재무지표+뉴스 합성
리포트 데이터셋(`finetune_dataset_*.jsonl`)은 `fiqa`/`tfns`와는 완전히
별개의 실험이고, 이 데이터로 학습한 결과물은 저장소에 없습니다 — 세 가지
(FiQA, TFNS, New_data 합성 데이터)를 하나로 뭉뚱그리면 안 됩니다.

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
