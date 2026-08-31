# Personal Portfolio Report

주식 포트폴리오를 입력하면 최신 가격, 뉴스, 재무 지표를 가져와 사용자 투자 성향을 분류하고 종목별 리포트를 생성하는 end-to-end Flask 대시보드입니다.

## 주요 기능

- 보유 종목과 수량 입력
- Yahoo Finance 기반 실시간 가격 조회
- Yahoo Finance RSS 기반 최신 뉴스 조회
- yfinance 기반 종목별 재무 지표 조회
- 변동성, 베타, MDD, HHI 기반 사용자 성향 분류 (SAFE / NEUTRAL / AGGRESSIVE)
- 포트폴리오 시장 가치, 종목 비중, 섹터 집중도 계산
- **투자 성향별로 다른 뉴스를 검색하는 RAG 파이프라인** (`retrieval.py`) — 같은
  종목이라도 SAFE/NEUTRAL/AGGRESSIVE 사용자에게 다른 뉴스를 우선 검색
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
                 Style-aware Query
                          │
                          │
Yahoo Finance News        │
        ↓                 │
News Crawling              │
        ↓                 │
GPT Factual Summary       │
        ↓                 │
Duplicate Removal          │
(exact + event-level)      │
        ↓                 │
Canonical Article DB       │
(SQLite: articles /        │
 article_tickers /         │
 article_embeddings)       │
        ↓                 │
MiniLM Embedding           │
        ↓                 │
Ticker Metadata Filter ←──┘
        ↓
Semantic Similarity
        +
Recency Ranking
        ↓
Event Deduplication
        ↓
Relevant Top-K News
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

투자 성향(`SAFE`/`NEUTRAL`/`AGGRESSIVE`)은 **retrieval과 generation 양쪽에서
모두** 사용된다: retrieval 단계에서는 `build_retrieval_query()`가 성향별로
다른 키워드를 우선하는 검색 쿼리를 만들고, generation 단계에서는 같은 성향
값이 LLM 프롬프트의 "Investor Style" 섹션과 분석 톤 조정에 그대로 쓰인다.

### 기존 방식 → 개선된 방식

```text
[기존]
ticker 최신 뉴스 10개 (recency만 정렬) → LLM

[개선]
투자성향별 query (build_retrieval_query)
→ ticker filter (article_tickers, metadata filter)
→ semantic retrieval (MiniLM cosine similarity)
→ recency ranking (별도 score, 임베딩에 넣지 않음)
→ event dedup (같은 사건 기사 중 최고점 1개만)
→ Top-K (기본 6개)
→ LLM (retrieval 실패 시 기존 "최신 뉴스" 로딩이 fallback으로 재사용됨)
```

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
계산하고, 같은 ticker의 SAFE vs AGGRESSIVE Top-K가 얼마나 겹치는지 보여주는
Jaccard overlap도 함께 출력합니다(이건 품질 지표가 아니라 개인화가 실제로
적용됐는지 확인하는 diagnostic입니다).

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

DB dedup, event dedup, retrieval(ticker filter/style query/semantic
similarity/recency/final score/event dedup/fallback), inference prompt
구성(RAG Top-K가 실제로 prompt에 들어가는지, investor style 정규화,
financial data 포함 여부, chat template 형태 유지)을 다룹니다. Sentence
embedding 모델은 네트워크/모델 다운로드 없이 `tests/fake_embedder.py`의
결정론적(deterministic) hashing embedder로 mocking되어 있습니다.

## 이번 개선 요약 (면접용)

이번 변경의 배경/설계 결정을 정리한 Q&A는 [`docs/INTERVIEW_QNA.md`](docs/INTERVIEW_QNA.md)에 있습니다.
