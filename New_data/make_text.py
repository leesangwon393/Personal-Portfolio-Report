import pandas as pd
import json
import random

# === Config ===
METRICS_CSV = "NASDAQ100_finance.csv"
NEWS_CSV = "NASDAQ100_news.csv"
OUTPUT_JSONL = "finetune_dataset_expanded.jsonl"
SAMPLE_SIZE = None   # None → 전체 사용

# === Load Data ===
metrics_df = pd.read_csv(METRICS_CSV)
news_df = pd.read_csv(NEWS_CSV)

# 컬럼명 통일 (Ticker → ticker)
metrics_df = metrics_df.rename(columns={"Ticker": "ticker"})

# Merge
merged_df = news_df.merge(metrics_df, on="ticker", how="inner")

if SAMPLE_SIZE and SAMPLE_SIZE < len(merged_df):
    merged_df = merged_df.sample(SAMPLE_SIZE, random_state=42)

# === Financial Analysis ===
def generate_financial_analysis(row):
    lines = []

    # EPS
    eps = row.get("Diluted EPS (ttm)")
    if not pd.isna(eps):
        lines.append(f"With EPS at {eps}, the firm demonstrates resilient profitability.")

    # Profitability
    pm = row.get("Profit Margin")
    if not pd.isna(pm):
        lines.append(f"Profit margin of {pm:.1%} highlights efficiency in generating net income.")
    om = row.get("Operating Margin (ttm)")
    if not pd.isna(om):
        lines.append(f"Operating margin of {om:.1%} indicates cost efficiency.")

    # Growth
    g = row.get("Quarterly Earnings Growth (yoy)")
    if not pd.isna(g):
        if g >= 0:
            lines.append(f"Earnings grew {g:.1%} YoY, showing strong momentum.")
        else:
            lines.append(f"Earnings declined {g:.1%} YoY, signaling performance challenges.")

    # Scale
    rev = row.get("Revenue (ttm)")
    if not pd.isna(rev):
        lines.append(f"Revenue of {rev:,} underscores the company’s scale.")
    gp = row.get("Gross Profit (ttm)")
    if not pd.isna(gp):
        lines.append(f"Gross profit of {gp:,} provides a buffer for operations.")
    ebitda = row.get("EBITDA")
    if not pd.isna(ebitda):
        lines.append(f"EBITDA of {ebitda:,} reinforces operating strength.")

    # Efficiency
    roa = row.get("Return on Assets(ROA) (ttm)")
    if not pd.isna(roa):
        lines.append(f"ROA of {roa:.1%} shows asset utilization efficiency.")
    roe = row.get("Return on Equity(ROE) (ttm)")
    if not pd.isna(roe):
        lines.append(f"ROE of {roe:.1%} reflects shareholder return generation.")

    # Balance sheet
    d = row.get("Total Debt/Equity (mrq)")
    if not pd.isna(d):
        lines.append(f"Debt-to-equity ratio of {d:.2f} suggests {'manageable leverage' if d < 1 else 'high financial risk'}.")

    c = row.get("Current Ratio (mrq)")
    if not pd.isna(c):
        if c > 1:
            lines.append(f"Current ratio of {c:.2f} indicates short-term liquidity strength.")
        else:
            lines.append(f"Current ratio of {c:.2f} raises liquidity concerns.")

    # Cash Flow
    ocf = row.get("Operating Cash Flow (ttm)")
    if not pd.isna(ocf):
        lines.append(f"Operating cash flow of {ocf:,} highlights robust core operations.")
    fcf = row.get("Levered Free Cash Flow (ttm)")
    if not pd.isna(fcf):
        lines.append(f"Free cash flow of {fcf:,} provides flexibility for reinvestment and returns.")

    # Valuation
    bv = row.get("Book Value Per Share (mrq)")
    if not pd.isna(bv):
        lines.append(f"Book value per share of {bv} indicates intrinsic shareholder equity.")

    return "\n".join(lines)

# === News Analysis ===
def generate_news_analysis(news_text, row):
    if not isinstance(news_text, str) or news_text.strip() == "":
        return "No major news reported recently."

    text = news_text.lower()
    parts = []

    # Growth news
    if any(k in text for k in ["growth", "demand", "launch", "ai", "semiconductor", "expansion"]):
        g = row.get("Quarterly Earnings Growth (yoy)")
        if not pd.isna(g):
            parts.append(f"Product or demand-related news aligns with earnings growth of {g:.1%}, reinforcing momentum.")

    # Investment / factory
    if any(k in text for k in ["investment", "factory", "capex", "r&d", "expansion"]):
        fcf = row.get("Levered Free Cash Flow (ttm)")
        if not pd.isna(fcf):
            parts.append(f"Capital expenditure plans appear supported by free cash flow of {fcf:,}.")

    # Regulation
    if any(k in text for k in ["regulation", "lawsuit", "recall", "sanction"]):
        d = row.get("Total Debt/Equity (mrq)")
        if not pd.isna(d):
            parts.append(f"Regulatory risks combined with D/E of {d:.2f} may elevate financial vulnerability.")

    # Dividend / buyback
    if any(k in text for k in ["dividend", "buyback", "shareholder"]):
        fcf = row.get("Levered Free Cash Flow (ttm)")
        if not pd.isna(fcf):
            parts.append(f"Shareholder actions could be sustained by FCF of {fcf:,}.")

    # Sustainability
    if any(k in text for k in ["climate", "sustainability", "green", "esg"]):
        ocf = row.get("Operating Cash Flow (ttm)")
        if not pd.isna(ocf):
            parts.append(f"Sustainability initiatives may be financed by operating cash flow of {ocf:,}.")

    if not parts:
        parts.append("The news has strategic significance, though financial impact remains uncertain.")

    return " ".join(parts)

# === Outlook ===
def generate_outlook(row):
    positives = []
    negatives = []

    # Positives
    if not pd.isna(row.get("Diluted EPS (ttm)")):
        positives.append("solid profitability")
    if not pd.isna(row.get("Operating Cash Flow (ttm)")):
        positives.append("strong operating cash flow")
    if not pd.isna(row.get("Levered Free Cash Flow (ttm)")) and row["Levered Free Cash Flow (ttm)"] > 0:
        positives.append("capacity for reinvestment")

    # Negatives
    if not pd.isna(row.get("Total Debt/Equity (mrq)")) and row["Total Debt/Equity (mrq)"] > 1:
        negatives.append("high leverage")
    if not pd.isna(row.get("Current Ratio (mrq)")) and row["Current Ratio (mrq)"] < 1:
        negatives.append("weak liquidity position")

    outlook = []
    if positives:
        outlook.append(f"Positives include {', '.join(positives)}.")
    if negatives:
        outlook.append(f"Risks stem from {', '.join(negatives)}.")
    outlook.append("Overall, the company’s outlook depends on balancing financial strengths with external challenges.")

    return " ".join(outlook)

# === Build Records ===
records = []
for _, row in merged_df.iterrows():
    ticker = row.get("ticker", "Unknown Ticker")
    sector = row.get("Sector", "N/A")
    industry = row.get("Industry", "N/A")
    news_text = row.get("article", "")

    # Input
    input_text = f"Ticker: {ticker}\nSector: {sector}\nIndustry: {industry}\nMetrics: {row.to_dict()}\nNews: {news_text}"

    # Output
    output_text = f"[Company Overview]\n{ticker.upper()} operates in the {sector} sector ({industry}).\n\n"
    output_text += f"[Financial Analysis]\n{generate_financial_analysis(row)}\n\n"
    output_text += f"[News Impact]\n{generate_news_analysis(news_text, row)}\n\n"
    output_text += f"[Outlook]\n{generate_outlook(row)}"

    records.append({"input": input_text, "output": output_text})

# === Save JSONL ===
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Expanded dataset created: {OUTPUT_JSONL} ({len(records)} samples)")

# 샘플 1개 출력
print("\n=== Sample Output ===")
print(records[0]["output"])
