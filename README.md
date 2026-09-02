# Personal Portfolio Report

주식 포트폴리오를 입력하면 최신 가격, 뉴스, 재무 지표를 가져와 사용자 투자 성향을 분류하고 종목별 리포트를 생성하는 end-to-end Flask 대시보드입니다.

## 주요 기능

- 보유 종목과 수량 입력
- Yahoo Finance 기반 실시간 가격 조회
- Yahoo Finance RSS 기반 최신 뉴스 조회
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
```

`model_inference.py`는 저장된 뉴스 DB가 없으면 Yahoo Finance RSS에서 최신 뉴스를 가져오고, CSV 지표와 yfinance 재무 스냅샷을 함께 사용해 리포트를 생성합니다.

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
