# Personal Portfolio Report

주식 포트폴리오를 입력하면 최신 가격, 뉴스, 재무 지표를 가져와 사용자 투자 성향을 분류하고 종목별 리포트를 생성하는 end-to-end Flask 대시보드입니다.

## 주요 기능

- 보유 종목과 수량 입력
- Yahoo Finance 기반 실시간 가격 조회
- Yahoo Finance RSS 기반 최신 뉴스 조회
- 사용자 성향별 뉴스 선별: 공통 실적 주제 + SAFE/NEUTRAL/AGGRESSIVE별 관심 주제를 검색해 리포트에 반영
- yfinance 기반 종목별 재무 지표 조회
- 변동성, 베타, MDD, HHI 기반 사용자 성향 분류
- 포트폴리오 시장 가치, 종목 비중, 섹터 집중도 계산
- 투자 성향과 종목별 지표를 반영한 맞춤형 리포트 생성
- 크롤링, 뉴스 DB 구축, 재무 지표 업데이트, 모델 추론용 Python 스크립트 포함

## Preview

<img width="1200" height="707" alt="Dashboard preview" src="https://github.com/user-attachments/assets/35510939-a17d-48a6-b2a6-065427e5c1c2" />

<img width="1140" height="590" alt="Report preview" src="https://github.com/user-attachments/assets/ce30a0b3-7aa0-49d2-acc8-4f39f6ac2b82" />

## Project Structure

```text
.
├── app.py                         # Flask end-to-end dashboard
├── Crawling/                      # Yahoo Finance news crawling helpers
├── data/                          # Financial metrics update scripts
├── db/                            # News database and summarization pipeline
├── classification/                # Portfolio classification utilities
├── srisk_result/                  # Risk metric calculation and market data (single user portfolio)
├── sec13f/                        # SEC 13F risk-clustering pipeline (institutional portfolios)
├── scripts/                       # run_13f_clustering.py CLI entrypoint
├── static/                        # Frontend JavaScript/CSS
├── templates/                     # Flask templates
├── train_and_inference/           # NASDAQ metrics and inference helpers
├── model_inference.py             # Optional local LoRA report inference
└── requirements.txt
```

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

`.env`, 작업 DB, 백업 CSV는 Git에서 제외합니다. 원본 뉴스 스냅샷 `db/news.db`와 튜닝된 LoRA 어댑터 가중치는 재현을 위해 저장소에 포함합니다.

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
```

`model_inference.py`는 저장된 뉴스 DB가 없으면 Yahoo Finance RSS에서 최신 뉴스를 가져오고, CSV 지표와 yfinance 재무 스냅샷을 함께 사용해 리포트를 생성합니다.

## 사용자 성향별 뉴스 검색

대시보드는 현재 위험 분류기의 `category`를, 로컬 추론은 `--style`을
`retrieval.py`에 전달합니다. 같은 종목의 기사 후보에서 기업 공통 실적 쿼리와
성향별 세부 쿼리를 각각 검색한 후 관련성 80% + 최신성 20%로 순위를 정합니다.
이 가중치는 검증된 최적값이 아닌 초기 설정이며 `rag_config.py`에서 조정합니다.

- SAFE: 현금흐름, 안정성, 하방·외부 위험
- NEUTRAL: 성장과 수익성, 위험의 균형
- AGGRESSIVE: 성장, 신제품, 사업 확장과 촉매

DB 임베딩이 있으면 재사용합니다. 없으면 DB의 최근 기사 후보를 임베딩하고,
DB 기사도 없으면 RSS 후보를 가져와 같은 방식으로 선별합니다. 후보 수는
최소 30개를 요청하되 실제 RSS 제공량에 따라 줄어들 수 있습니다. 모델 로딩이
실패하면 최신 뉴스로 대체하고 대시보드에 성향별 선별을 사용할 수 없다고 표시합니다.
API 응답의 `reports[].news_selection.source`로도 적용 여부를 확인할 수 있습니다
(`rag`, `db_semantic`, `live_semantic`, `archive_semantic`은 성향별 검색 적용).

첫 의미 검색 시 `sentence-transformers/all-MiniLM-L6-v2` 다운로드가 필요할 수 있습니다.
원본 저장소에서 가져온 `db/news.db`는 2025-10-31~2025-11-13의 기사·요약
1,113건과 기존 인덱스를 담은 스냅샷입니다. 아래 준비 명령은 원본을 보존하고
`local_data/news/db/news.db`에 작업 복사본과 새 MiniLM 검색 인덱스를 만듭니다.
재실행하면 원본의 추가 기사를 ID 기준으로 병합하고 작업 DB의 기존 기사를 보존하며,
빠진 임베딩을 채웁니다. 같은 ID의 기사 내용이 다르면 원본이나 작업 DB를 덮어쓰지 않고 중단합니다. 앱·모델 추론·수집기는
이 작업 경로를 공유하며, `NEWS_DB_BASE` 환경변수로 기본 디렉터리를 변경할 수 있습니다.
준비 전에는 원본 스냅샷의 기사·요약을 읽어 후보를 임베딩합니다.
검색 기간 밖의 저장 기사는 `archive_semantic`으로 표시하고 대시보드에 과거 뉴스임을 명시합니다.

```bash
python3 scripts/prepare_news_db.py
python3 retrieval.py --ticker NVDA --compare-styles --lookback-days 400
python3 model_inference.py --ticker NVDA --style SAFE --debug
python3 -m unittest discover -s tests -v
```

공통 쿼리도 후보 검색에 참여하지만 최종 목록에 공통 기사가 반드시 포함되도록
할당량을 강제하지는 않습니다. 성향별 기사가 항상 달라지는 것도 아닙니다.
자동 테스트는 가짜 임베딩으로 흐름을 검증하며, 실제 뉴스 검색 품질은 별도 평가가 필요합니다.
두 저장소의 비교·가져온 항목·실제 DB 검증 결과는 [통합 기록](docs/REPOSITORY_INTEGRATION.md)에 정리합니다.
101개 종목의 성향별 검색 점검과 한계는 [검색 품질 점검](docs/NEWS_SELECTION_REVIEW.md)에 있습니다.

## Earnings-call LoRA pipeline

원문 데이터나 학습 결과물은 Git에 올리지 않습니다. Alpha Vantage의 `EARNINGS_CALL_TRANSCRIPT` API 사용 권한을 확인한 뒤, CEO/CFO 발화만 로컬 JSONL로 수집하고 LoRA 학습을 실행합니다.

```bash
# .env에 ALPHAVANTAGE_API_KEY와 필요 시 HF_TOKEN 설정
python3 train_and_inference/earnings_call_lora.py collect --year 2025
python3 train_and_inference/earnings_call_lora.py train
```

수집 결과와 어댑터는 `local_data/earnings_calls/`에 저장되며 `.gitignore`로 제외됩니다. 이 학습은 CEO/CFO 발화 기반의 금융 도메인 적응용이며, 개인화 리포트 생성에는 재무·뉴스·투자성향을 포함한 별도 instruction 데이터셋이 필요합니다.

## 13F 공개 주식 포트폴리오 기반 위험 운용성향 (기관투자자 군집화)

`app.py`의 개인 포트폴리오 분류기(`srisk_result/`)와는 별도로, SEC EDGAR에
공개된 기관투자자 13F-HR 신고 데이터로 실제 100개 포트폴리오를 수집해
Volatility / MDD / Beta / Sector HHI 4개 위험지표를 계산하고, **비지도
군집화**로 DEFENSIVE / BALANCED / AGGRESSIVE 세 군집을 사후 해석하는
파이프라인입니다. 사람이 직접 라벨링하지 않으며, 군집 이름은 학습용
정답이 아니라 군집 중심값을 사후에 해석해 붙인 이름입니다.

```bash
python scripts/run_13f_clustering.py \
  --quarter 2026-06-30 \
  --num-portfolios 100 \
  --lookback-days 252 \
  --output-dir artifacts/13f_clustering

python -m pytest test_13f_clustering.py -v
```

방법론(데이터 출처, 선정 기준, 13F의 한계, 정규화 방식, K=3 선택 근거,
환경변수, 결과 해석 시 주의사항)은
[`docs/13F_CLUSTERING.md`](docs/13F_CLUSTERING.md)에, 실제 실행 결과는
`artifacts/13f_clustering/report.md`에 정리되어 있습니다.
