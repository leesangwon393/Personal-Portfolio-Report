# Personal Portfolio Report

개인 주식 포트폴리오의 리스크, 섹터 집중도, 종목별 지표를 분석하고 투자 성향에 맞춘 요약 리포트를 보여주는 Flask 대시보드입니다.

## 주요 기능

- 보유 종목과 수량 입력
- Yahoo Finance 기반 실시간 가격 조회
- 포트폴리오 시장 가치와 종목별 비중 계산
- 변동성, 베타, MDD, HHI 기반 리스크 점수 산출
- 섹터 집중도와 위험 상위 종목 시각화
- 종목별 재무 지표 기반 요약 리포트 제공

## Preview

<img width="1200" height="707" alt="Dashboard preview" src="https://github.com/user-attachments/assets/35510939-a17d-48a6-b2a6-065427e5c1c2" />

<img width="1140" height="590" alt="Report preview" src="https://github.com/user-attachments/assets/ce30a0b3-7aa0-49d2-acc8-4f39f6ac2b82" />

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── static/
├── templates/
├── srisk_result/
│   ├── analyze_portfolio_risk.py
│   ├── us_market_metrics_sp500_nasdaq100.csv
│   └── wallstreet_srisk_results.csv
└── train_and_inference/
    └── NASDAQ100_metrics.csv
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

초기 화면은 데모 가격을 사용하고, `Analyze` 버튼을 누르면 Yahoo Finance에서 최신 가격을 조회합니다.
