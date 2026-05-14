import pandas as pd
import json
import random

# === Config ===
METRICS_CSV = "NASDAQ100_finance.csv"
NEWS_CSV = "NASDAQ100_news.csv"
OUTPUT_JSONL = "finetune_dataset_augmented.jsonl"
AUG_FACTOR = 3   # 한 레코드당 몇 개 버전 만들지

# === Load Data ===
metrics_df = pd.read_csv(METRICS_CSV)
news_df = pd.read_csv(NEWS_CSV)
metrics_df = metrics_df.rename(columns={"Ticker": "ticker"})
merged_df = news_df.merge(metrics_df, on="ticker", how="inner")

# === 템플릿 풀 ===
EPS_TEMPLATES = [
    "EPS of {val} indicates solid profitability.",
    "The company reported EPS of {val}, showing earnings strength.",
    "With EPS at {val}, the firm demonstrates resilient performance.",
    "Per-share earnings reached {val}, reflecting profitability.",
    "Reported EPS stood at {val}, underlining earnings capacity."
]

OUTLOOK_TEMPLATES = [
    "Overall, the company is positioned for growth, though risks must be monitored.",
    "The outlook remains balanced, with financial strengths offset by challenges.",
    "We maintain a cautiously optimistic outlook given profitability and cash flow.",
    "The company faces risks, but fundamentals suggest resilience.",
    "Future performance will hinge on managing leverage while sustaining growth."
]

# === Functions ===
def generate_financial_analysis(row):
    lines = []
    eps = row.get("Diluted EPS (ttm)")
    if not pd.isna(eps):
        lines.append(random.choice(EPS_TEMPLATES).format(val=eps))

    g = row.get("Quarterly Earnings Growth (yoy)")
    if not pd.isna(g):
        if g >= 0:
            lines.append(f"Earnings grew {g:.1%} YoY, showing momentum.")
        else:
            lines.append(f"Earnings declined {g:.1%} YoY, raising concerns.")

    d = row.get("Total Debt/Equity (mrq)")
    if not pd.isna(d):
        lines.append(f"Debt-to-equity ratio of {d:.2f} suggests {'manageable leverage' if d < 1 else 'elevated risk'}.")

    return "\n".join(lines)

def generate_news_analysis(news_text, row):
    if not isinstance(news_text, str) or news_text.strip() == "":
        return "No major news reported recently."

    text = news_text.lower()
    parts = []

    if "growth" in text or "launch" in text or "demand" in text:
        g = row.get("Quarterly Earnings Growth (yoy)")
        if not pd.isna(g):
            parts.append(f"This aligns with earnings growth of {g:.1%}.")
    if "investment" in text or "factory" in text:
        fcf = row.get("Levered Free Cash Flow (ttm)")
        if not pd.isna(fcf):
            parts.append(f"Such investments may be funded by FCF of {fcf:,}.")
    if not parts:
        parts.append("The news may have strategic significance, though financial impact is uncertain.")

    return " ".join(parts)

def generate_outlook(row):
    positives, negatives = [], []
    if not pd.isna(row.get("Diluted EPS (ttm)")):
        positives.append("profitability")
    if not pd.isna(row.get("Operating Cash Flow (ttm)")):
        positives.append("cash generation")
    if not pd.isna(row.get("Total Debt/Equity (mrq)")) and row["Total Debt/Equity (mrq)"] > 1:
        negatives.append("high leverage")
    if not pd.isna(row.get("Current Ratio (mrq)")) and row["Current Ratio (mrq)"] < 1:
        negatives.append("weak liquidity")

    outlook = []
    if positives:
        outlook.append("Positives include " + ", ".join(positives) + ".")
    if negatives:
        outlook.append("Risks stem from " + ", ".join(negatives) + ".")
    outlook.append(random.choice(OUTLOOK_TEMPLATES))
    return " ".join(outlook)

# === Build Records with Augmentation ===
records = []
for _, row in merged_df.iterrows():
    ticker = row.get("ticker", "Unknown")
    sector = row.get("Sector", "N/A")
    industry = row.get("Industry", "N/A")
    news_text = row.get("article", "")

    input_text = f"Ticker: {ticker}\nSector: {sector}\nIndustry: {industry}\nMetrics: {row.to_dict()}\nNews: {news_text}"

    for _ in range(AUG_FACTOR):
        output_text = f"[Company Overview]\n{ticker.upper()} operates in the {sector} sector ({industry}).\n\n"
        output_text += f"[Financial Analysis]\n{generate_financial_analysis(row)}\n\n"
        output_text += f"[News Impact]\n{generate_news_analysis(news_text, row)}\n\n"
        output_text += f"[Outlook]\n{generate_outlook(row)}"
        records.append({"input": input_text, "output": output_text})

# === Save JSONL ===
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Augmented dataset created: {OUTPUT_JSONL} ({len(records)} samples)")
