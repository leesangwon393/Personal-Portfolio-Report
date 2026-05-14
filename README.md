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
├── srisk_result/                  # Risk metric calculation and market data
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

`.env`와 로컬 DB, 모델 바이너리, 백업 CSV는 Git에 올리지 않도록 `.gitignore`에 포함되어 있습니다.
