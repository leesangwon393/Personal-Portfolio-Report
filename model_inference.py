from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from secrets_config import get_hf_token


ROOT_DIR = Path(__file__).resolve().parent
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTERS = {
    "fiqa": ROOT_DIR / "train_and_inference" / "fiqa" / "model",
    "tfns": ROOT_DIR / "train_and_inference" / "tfns" / "model",
}
METRICS_PATH = ROOT_DIR / "train_and_inference" / "NASDAQ100_metrics.csv"
NEWS_DB_PATH = ROOT_DIR / "db" / "news.db"


SYSTEM_PROMPT = """You are an expert financial analyst. Your mission is to write a concise, objective investment report for a client based on their specific risk profile.

ANALYSIS INSTRUCTIONS:
- Use BOTH the provided financial metrics ("Facts") and recent news ("News").
- Do not hallucinate numbers that are not in Facts.
- Adjust the focus and tone strictly based on the investor's style.
- Do NOT provide direct financial advice or buy/sell recommendations.

OUTPUT REPORT TEMPLATE
Report for: A (investor_style) Investor
Company: (company_name)

1. Executive Summary:
   (One concise paragraph aligned with the investor style.)

2. Key Analysis & Highlights:
   (5-7 bullet points. Each bullet should connect a financial metric with relevant news.)

3. Concluding Remark:
   (One or two sentences summarizing current standing for this type of investor.)
"""


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_company_data(ticker: str) -> str:
    df = pd.read_csv(METRICS_PATH)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    row = df[df["Ticker"] == ticker.upper()]
    if row.empty:
        return f"Ticker: {ticker.upper()}\nNo financial metrics found."
    return row.iloc[0].dropna().to_string()


def load_news(ticker: str, limit: int = 10) -> str:
    if not NEWS_DB_PATH.exists():
        return "No news database found."
    conn = sqlite3.connect(NEWS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT headline, summary, pubdate
            FROM articles
            WHERE ticker = ?
              AND summary IS NOT NULL
              AND LENGTH(TRIM(summary)) > 0
            ORDER BY COALESCE(pubdate, id) DESC
            LIMIT ?
            """,
            (ticker.upper(), limit),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No summarized news found."
    return "\n".join(
        f"- {str(row['pubdate'])[:10]} | {row['headline']}: {row['summary']}"
        for row in rows
    )


def build_prompt(ticker: str, investor_style: str) -> list[dict[str, str]]:
    user_prompt = f"""Analyze all the provided data and generate a report tailored to the investor's profile.

1. Investor Style: {investor_style.upper()}

2. Company Under Review, Key Data from Corporate Filings:
{load_company_data(ticker)}

3. Recent News:
{load_news(ticker)}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LoRA report inference.")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--style", default="SAFE", choices=["SAFE", "NEUTRAL", "RISKY"])
    parser.add_argument("--adapter", default="fiqa", choices=sorted(ADAPTERS))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--min-new-tokens", type=int, default=120)
    args = parser.parse_args()

    hf_token = get_hf_token(required=True)

    device = pick_device()
    adapter_path = ADAPTERS[args.adapter]

    print(f"Loading tokenizer from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {BASE_MODEL} on {device}")
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=hf_token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.to(device)

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    messages = build_prompt(args.ticker, args.style)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = encoded["input_ids"].shape[1]
    generated = output[0, input_length:]
    print("\n=== MODEL INFERENCE RESULT ===\n")
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
