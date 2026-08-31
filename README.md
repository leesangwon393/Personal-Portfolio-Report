# Personal Portfolio Report

주식 포트폴리오를 입력하면 최신 가격, 뉴스, 재무 지표를 가져와 사용자 투자 성향을 분류하고 종목별 리포트를 생성하는 end-to-end Flask 대시보드입니다.

## 주요 기능

- 보유 종목과 수량 입력
- Yahoo Finance 기반 실시간 가격 조회
- Yahoo Finance RSS 기반 최신 뉴스 조회
- yfinance 기반 종목별 재무 지표 조회
- 변동성, 베타, MDD, HHI 기반 사용자 성향 분류 (SAFE / NEUTRAL / AGGRESSIVE)
- 포트폴리오 시장 가치, 종목 비중, 섹터 집중도 계산
- **Core Query + 여러 개의 Style Facet Query를 따로 검색해 병합하는
  Multi-query RAG 파이프라인** (`retrieval.py`) — 모든 투자자에게 공통적으로
  중요한 기업 핵심정보(Core Query 1개)는 항상 검색하고, 그 위에
  SAFE/NEUTRAL/AGGRESSIVE별로 2~3개의 좁은 관점(Facet Query, 예: SAFE의
  stability/downside_risk/external_risk)을 각각 독립적으로 검색해 후보를
  병합함으로써 같은 종목이라도 사용자마다 다른 뉴스가 상위로 올라오게 함
- **뉴스 원문 중복 관리**: 같은 기사가 여러 ticker에 중복 저장되지 않도록
  article ↔ ticker를 다대다 관계로 관리하고, 같은 사건을 다룬 여러 기사는
  event_group으로 묶어 Top-K가 한 이벤트로 도배되지 않게 함
- 투자 성향과 종목별 지표, RAG로 검색한 뉴스를 반영한 맞춤형 리포트 생성
  (Meta-Llama-3-8B-Instruct + LoRA)
- 크롤링, 뉴스 DB 구축, 재무 지표 업데이트, 모델 추론, 검색/생성 평가용 Python 스크립트 포함

## Preview

<img width="1200" height="707" alt="Dashboard preview" src="https://github.com/user-attachments/assets/35510939-a17d-48a6-b2a6-065427e5c1c2" />

<img width="1140" height="590" alt="Report preview" src="https://github.com/user-attachments/assets/ce30a0b3-7aa0-49d2-acc8-4f39f6ac2b82" />

## Project Structure

```text
.
├── app.py                         # Flask end-to-end dashboard
├── Crawling/                      # Yahoo Finance news crawling helpers
├── data/                          # Financial metrics update scripts
├── db/                            # News database, dedup ingestion, embeddings, event dedup
├── classification/                # Portfolio classification utilities
├── srisk_result/                  # Risk metric calculation and market data
├── static/                        # Frontend JavaScript/CSS
├── templates/                     # Flask templates
├── train_and_inference/           # NASDAQ metrics and inference helpers
├── evaluation/                    # Retrieval (Precision/Recall/nDCG) and generation (LLM-judge) evaluation
├── tests/                         # Unit tests for DB dedup, event dedup, retrieval, inference prompt
├── rag_config.py                  # Single source of truth for RAG thresholds/weights
├── retrieval.py                   # Style-aware RAG retrieval (Section 이하 참고)
├── model_inference.py             # Optional local LoRA report inference — now RAG-backed
└── requirements.txt
```

## Architecture

```text
                    User Portfolio
                          ↓
              Volatility / Beta / MDD / HHI
                          ↓
                 Investor Style
           SAFE / NEUTRAL / AGGRESSIVE
                          ↓
                 Query Builder
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
          Core Query           Style Facet Queries
      (1개, 모든 투자자 공통,   (2~3개, SAFE/NEUTRAL/AGGRESSIVE
       investor_style 무관)     별로 다른 좁은 관점)
              │                       │
              │                       │
Yahoo Finance News                    │
        ↓                             │
News Crawling                          │
        ↓                             │
GPT Factual Summary                   │
        ↓                             │
Duplicate Removal (ingestion 시점,     │
exact + event-level)                   │
        ↓                             │
Canonical Article DB                   │
(SQLite: articles /                    │
 article_tickers /                     │
 article_embeddings)                   │
        ↓                             │
MiniLM Embedding                       │
        ↓                             │
Ticker Metadata Filter ←───────────────┘
        │
        ↓
Per-query Top-N Retrieval
(Core 1개 + Style Facet 2~3개, 각각 독립 검색)
        ↓
   Candidate Merge
(article_id로 병합 — 여러 query에 매칭돼도 1 row,
 matched_queries만 누적)
        ↓
Exact Duplicate Removal (병합 단계 defensive re-check)
        ↓
Semantic + Recency Reranking
(semantic_score = 모든 query 중 최대 유사도)
        ↓
Event Deduplication
        ↓
        Top-K
        │
        ├─────────────┐
        │             │
Financial Data   Investor Style
        │             │
        └──────┬──────┘
               ↓
      Meta-Llama-3-8B-Instruct
               +
             LoRA
               ↓
      Personalized Stock Report
```

모든 투자자에게 필요한 기업의 핵심 실적정보는 Core Query로 공통 검색하고,
투자성향별 관심정보는 여러 Facet Query로 분리해 추가 검색하였다. 각
query에서 확보한 후보를 병합한 뒤 semantic relevance와 recency를 기준으로
재정렬하고 동일 이벤트 기사를 제거하여 최종 Top-K를 구성하였다.

투자 성향(`SAFE`/`NEUTRAL`/`AGGRESSIVE`)은 **retrieval과 generation 양쪽에서
모두** 사용된다: retrieval 단계에서는 `build_style_facet_queries()`가
성향별로 2~3개의 Facet Query를 만들어 각각 독립적으로 검색하고(Core Query는
`build_core_query()`가 만들며 investor_style과 무관하게 항상 동일), generation
단계에서는 같은 성향 값이 LLM 프롬프트의 "Investor Style" 섹션과 분석 톤
조정에 그대로 쓰인다.

### 기존 구조 → 현재 구조

```text
[기존]
ticker 최신 뉴스 10개 (recency만 정렬) → LLM

[개선 1차]
투자성향별로 하나로 합친 Style Query 1개 → semantic retrieval
→ recency ranking → event dedup → Top-K → LLM

[개선 2차 — 현재]
Ticker
+ Investor Style
        ↓
Core Query(1개, 공통) + Style Facet Query(2~3개, 성향별)
        ↓
Multi-query Retrieval (query마다 별도 Top-N, 기본 5개씩)
        ↓
Candidate Merge (article_id 기준, matched_queries 기록)
        ↓
Exact Duplicate Removal
        ↓
Semantic + Recency Reranking
  (semantic_score = max(cosine(query_vec, article_vec) for 모든 query_vec)
   final_score = SEMANTIC_WEIGHT(0.8)×semantic_score + RECENCY_WEIGHT(0.2)×recency_score)
        ↓
Event Deduplication (같은 사건 기사 중 최고 Final Score 1개만)
        ↓
Top-K (기본 6개)
        ↓
LLM (retrieval 실패 시 기존 "최신 뉴스" 로딩이 fallback으로 재사용됨)
```

하나의 긴 Style Query에 여러 검색 의도(안정성 + 하방위험 + 규제리스크 등)를
몰아넣으면 embedding 상에서 각 의도가 서로 희석될 수 있고, 어떤 기사가 어떤
의도 때문에 검색됐는지도 해석하기 어렵다. 그래서 SAFE/NEUTRAL/AGGRESSIVE
각각을 2~3개의 독립된 Facet Query로 나눠 따로 검색한 뒤 후보를 병합하는
방식을 쓴다. 이 방식이 하나로 합치지 않는 이유는
[`docs/INTERVIEW_QNA.md`](docs/INTERVIEW_QNA.md)에 정리되어 있다.

**주의**: SQLite에 임베딩(BLOB)을 저장해 코사인 유사도를 직접 계산하는
구조이며, FAISS/Chroma/Pinecone 같은 별도 Vector DB 제품은 쓰지 않는다.
Random Projection(비학습 차원 축소, 구 `integrated_index`)은 더 이상 이
RAG 경로에서 쓰이지 않고 `db/db.py`에 legacy 섹션으로만 남아있다.

## 실행 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

브라우저에서 아래 주소로 접속합니다.

```text
http://127.0.0.1:5000
```

초기 화면은 데모 포트폴리오를 보여주고, `Analyze` 버튼을 누르면 입력한 포트폴리오 기준으로 최신 가격, 뉴스, 재무 지표를 다시 조회합니다.

## Environment

앱의 기본 대시보드는 별도 API 키 없이 동작합니다. OpenAI 요약, FMP 재무 데이터 업데이트, Hugging Face 기반 로컬 모델 추론을 사용할 때만 `.env`를 만듭니다.

```bash
cp .env.example .env
```

```text
OPENAI_API_KEY=
HF_TOKEN=
FMP_API_KEY=
```

`.env`, 로컬 DB, 백업 CSV는 Git에 올리지 않도록 `.gitignore`에 포함되어 있습니다. 튜닝된 LoRA 어댑터 가중치만 예외로 저장소에 포함합니다.

## Tuned Model Inference

튜닝된 LoRA 어댑터 가중치는 아래 경로에 포함되어 있습니다.

```text
train_and_inference/fiqa/model/
train_and_inference/tfns/model/
```

### Fine-tuning 데이터셋

두 어댑터는 서로 다른 공개 데이터셋으로 `Meta-Llama-3-8B-Instruct`를 각각
LoRA fine-tuning한 결과입니다(학습 노트북은 `9f20ae3` "Clean up project for
app release"에서 앱 배포용으로 정리하며 저장소에서 제거했고, 결과 가중치만
남겼습니다 — 아래는 그 이전 커밋 히스토리에 남아있던 `fiqa-peft.ipynb` /
`fingpt-peft.ipynb` 기준입니다):

| Adapter | 데이터셋 | 성격 |
| --- | --- | --- |
| `fiqa` | [`FinGPT/fingpt-fiqa_qa`](https://huggingface.co/datasets/FinGPT/fingpt-fiqa_qa) | 금융 Q&A (FiQA 기반) |
| `tfns` | [`zeroshot/twitter-financial-news-sentiment`](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 금융 뉴스/트윗 감성 분류 (Twitter Financial News Sentiment) |

**earnings-call(실적발표 콜) 원문 데이터는 두 어댑터 어디의 학습에도 쓰이지
않았습니다.** `Crawling`/`db`가 수집하는 뉴스 중에는 "Q1 2026 Earnings Call
Transcript"처럼 실적발표를 다룬 *기사*가 섞여 있지만, 이는 RAG가 검색하는
뉴스 코퍼스의 일부일 뿐 LoRA 학습 데이터가 아닙니다.

`New_data/make_text.py`, `New_data/add_text.py`는 별도의 실험으로, 재무
지표(`NASDAQ100_finance.csv`)와 뉴스(`NASDAQ100_news.csv`)를 템플릿으로
합성한 리포트 스타일 JSONL(`finetune_dataset_expanded.jsonl` /
`finetune_dataset_augmented.jsonl`)을 만듭니다. 이 스크립트로 만든 데이터는
위 두 어댑터를 만드는 데 쓰이지 않았고, 저장소에 학습 결과(가중치)도
포함되어 있지 않습니다 — 즉 `fiqa`/`tfns` 어댑터와는 독립적인, 실행되지
않은 아이디어 단계의 스크립트입니다.

로컬 추론은 Hugging Face에서 base model을 내려받아 위 어댑터를 붙여 실행합니다. `HF_TOKEN`이 필요합니다.

```bash
python3 model_inference.py --ticker AAPL --style SAFE --adapter fiqa
python3 model_inference.py --ticker TSLA --style AGGRESSIVE --adapter tfns

# LoRA 없이 base 모델만 실행 (Base vs LoRA 비교용)
python3 model_inference.py --ticker NVDA --style NEUTRAL --adapter base

# Top-K 조정 + retrieval 쿼리/점수를 stderr로 확인
python3 model_inference.py --ticker NVDA --style AGGRESSIVE --adapter fiqa --top-k 8 --debug
```

`model_inference.py`의 뉴스 섹션은 이제 `retrieval.py`의 스타일별 RAG
파이프라인(`retrieve_news_with_fallback`)에서 채워집니다. Fallback 순서는
다음과 같습니다: **① 스타일별 RAG Top-K → ② DB에 저장된 해당 ticker 최신
기사(recency만) → ③ Yahoo Finance RSS 실시간 조회 → ④ "No relevant news
found."** — 세 번째 단계까지는 기존에 있던 "ticker 최신 뉴스" 로딩 로직을
그대로 재사용합니다(Section 19: RAG가 실패해도 서비스가 깨지지 않도록).

## 뉴스 DB 구축 & RAG 파이프라인

`db/db.py`는 뉴스 크롤링부터 article 정규화 적재, GPT 요약, embedding 생성,
event dedup까지 담당합니다. 전체 순서:

```bash
cd db
python3 db.py init
python3 db.py crawl --use-dynamic --refresh-tickers
python3 db.py preprocess                 # exact duplicate 제거하며 정규화 적재 (Section 3/4)
python3 db.py cleanup-db --days-keep 14   # 오래된 기사 정리 + [legacy] 유사 기사 하드 삭제

export OPENAI_API_KEY="..."               # 또는 .env에 설정
python3 db.py summarize                   # GPT로 factual summary 생성

python3 db.py build-embeddings            # article 기준 MiniLM 임베딩 1회 생성 (Section 6)
python3 db.py assign-events               # 같은 사건 기사 event_group으로 묶기 (Section 5)

# 기존 DB(스키마 변경 이전)를 쓰고 있었다면 1회만 실행
python3 db.py migrate-legacy
```

검색만 별도로 테스트하려면(같은 ticker, 성향별 비교 포함):

```bash
python3 retrieval.py --ticker NVDA --style SAFE
python3 retrieval.py --ticker NVDA --compare-styles   # SAFE/NEUTRAL/AGGRESSIVE Top-K를 나란히 출력
```

### 뉴스 DB 스키마

```text
articles            # article 원문 메타 + summary. 1 row = 1 원문 기사.
  id, headline, ticker(legacy 대표 ticker), pubdate, summary,
  url, source, content_hash, event_group_id, created_at

article_tickers      # article ↔ ticker 다대다 관계
  article_id, ticker

article_embeddings   # article 기준 MiniLM 임베딩 (ticker/시간 결합 없음)
  article_id, embedding, embedding_model, created_at

integrated_index     # [LEGACY] Random Projection 기반 검색 벡터, 새 RAG 경로 미사용
  id, ivect
```

`articles.ticker`는 기존 앱(`app.py`, 옛 `model_inference.py`)과의 하위
호환을 위해 "최초로 연결된 ticker"를 그대로 남겨두지만, ingestion 시점의
실제 중복 판단과 retrieval의 ticker filter는 `article_tickers`를 기준으로
동작합니다. 기존 DB를 새 스키마로 완전히 재정규화하는 대신, additive한
`ALTER TABLE`과 `migrate-legacy` backfill로 호환성을 유지합니다.

## Retrieval Evaluation

```bash
# evaluation/sample_relevance_labels.json의 REPLACE_WITH_REAL_ARTICLE_ID_* 를
# retrieval.py --ticker NVDA --style SAFE 로 확인한 실제 article_id로 채운 뒤:
python3 evaluation/retrieval_eval.py --labels evaluation/sample_relevance_labels.json
```

Precision@K / Recall@K / nDCG@K를 (ticker, investor_style) 케이스별로
계산합니다. 추가로 다음 diagnostic을 케이스별로 함께 출력합니다(모두 품질
지표가 아니라 multi-query 파이프라인이 실제로 의도대로 동작하는지 확인하는
용도입니다):

- `average_semantic_score` / `average_recency_score` — Top-K가 semantic
  relevance와 recency 중 어디에 더 의존해 뽑혔는지
- `core_query_hit_count` / `style_facet_hit_count` — Top-K 중 Core Query
  때문에 뽑힌 기사 수 / Style Facet Query 때문에 뽑힌 기사 수 (핵심 실적
  정보가 Top-K에서 밀려나지 않았는지 확인하는 diagnostic, Top-K 전체가
  facet에서만 나온다면 정보 편향 신호)
- `merged_candidate_count` / `exact_duplicate_count` — multi-query
  candidate merge 이후 후보 수와, 그중 exact-duplicate로 제거된 수
- `event_diversity_count` — reranked 후보 중 서로 다른 event_group 수

같은 ticker의 SAFE vs AGGRESSIVE Top-K가 얼마나 겹치는지 보여주는 Jaccard
overlap도 함께 출력합니다. 겹침이 낮을수록 무조건 좋은 것은 아닙니다 —
중요한 실적 뉴스는 Core Query 덕분에 여러 성향에 공통으로 뽑히는 것이
오히려 정상입니다.

## Generation Evaluation & Base vs LoRA 비교

```bash
# LLM-as-a-Judge (faithfulness / personalization / relevance / coherence, 1~5)
python3 evaluation/generation_eval.py \
    --report-file report.txt --financial-file financials.txt \
    --news-file news.txt --style SAFE

# 동일 ticker/style/top-k로 base / fiqa / tfns를 순서대로 실행해 비교
python3 evaluation/compare_adapters.py --ticker NVDA --style AGGRESSIVE
```

## Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

DB dedup, event dedup, retrieval(ticker filter / core query가 모든 style에서
동일한지 / style facet query가 style별로 여러 개 다르게 생성되는지 / 각
query가 별도로 embedding되어 query마다 Top-N이 나오는지 / 동일 article이
여러 query에 매칭돼도 candidate merge 시 한 row로 병합되는지 / exact
duplicate가 content_hash 기준으로 제거되는지 / semantic score = 모든 query
중 최대 유사도인지 / recency / final score = semantic×recency 가중합 /
event dedup / fallback), inference prompt 구성(RAG Top-K가 실제로 prompt에
들어가는지, investor style 정규화, financial data 포함 여부, chat template
형태 유지)을 다룹니다. Sentence embedding 모델은 네트워크/모델 다운로드
없이 `tests/fake_embedder.py`의 결정론적(deterministic) hashing embedder로
mocking되어 있습니다.

## 이번 개선 요약 (면접용)

이번 변경의 배경/설계 결정을 정리한 Q&A는 [`docs/INTERVIEW_QNA.md`](docs/INTERVIEW_QNA.md)에 있습니다.
