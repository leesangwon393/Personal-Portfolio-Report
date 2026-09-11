from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus
import re

import pandas as pd
import requests
import yfinance as yf

from secrets_config import get_hf_token
from rag_config import TOP_K
from retrieval import normalize_investor_style, retrieve_news_with_fallback
from news_paths import get_news_db_path


ROOT_DIR = Path(__file__).resolve().parent
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTERS = {
    "fiqa": ROOT_DIR / "train_and_inference" / "fiqa" / "model",
    "tfns": ROOT_DIR / "train_and_inference" / "tfns" / "model",
}
METRICS_PATH = ROOT_DIR / "train_and_inference" / "NASDAQ100_metrics.csv"
NEWS_DB_PATH = get_news_db_path()


SYSTEM_PROMPT = """You are an expert financial analyst. Your mission is to write a concise, objective investment report for a client based on their specific risk profile.

ANALYSIS INSTRUCTIONS:
- Use BOTH the provided financial metrics ("Facts") and recent news ("News") as your only source of evidence.
- Do not hallucinate numbers that are not in Facts.
- Do not state uncertain or unverified information as fact.
- Check each article's date. Archived news must not be presented as a current event.
- Adjust the focus and tone strictly based on the investor's style.
- Consider both growth factors and risk factors in the retrieved news, even when the investor's style leans toward one side — personalization is about emphasis, not omission.
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


def clean_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", unescape(str(value)))
    return re.sub(r"\s+", " ", text).strip()


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_company_data(ticker: str) -> str:
    df = pd.read_csv(METRICS_PATH)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    row = df[df["Ticker"] == ticker.upper()]
    lines = []
    if not row.empty:
        lines.append(row.iloc[0].dropna().to_string())

    try:
        info = yf.Ticker(ticker).get_info()
        live_fields = {
            "Company": info.get("longName") or info.get("shortName"),
            "Sector": info.get("sector"),
            "Market Cap": info.get("marketCap"),
            "Trailing EPS": info.get("trailingEps"),
            "Profit Margins": info.get("profitMargins"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Earnings Growth": info.get("earningsQuarterlyGrowth"),
            "Debt To Equity": info.get("debtToEquity"),
            "Current Ratio": info.get("currentRatio"),
            "Free Cash Flow": info.get("freeCashflow"),
        }
        live_text = "\n".join(
            f"{key}: {value}" for key, value in live_fields.items() if value is not None
        )
        if live_text:
            lines.append("Live yfinance snapshot:\n" + live_text)
    except Exception:
        pass

    return "\n\n".join(lines) or f"Ticker: {ticker.upper()}\nNo financial metrics found."


def load_live_news(ticker: str, limit: int = 10) -> str:
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={quote_plus(ticker.upper())}&region=US&lang=en-US"
    )
    response = requests.get(url, timeout=8)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    items = []
    for item in root.findall("./channel/item"):
        title = clean_html(item.findtext("title"))
        summary = clean_html(item.findtext("description"))
        pubdate = clean_html(item.findtext("pubDate"))
        if title:
            items.append(f"- {pubdate[:16]} | {title}: {summary}")
        if len(items) >= limit:
            break
    return "\n".join(items) if items else "No live news found."


def _live_news_fallback(ticker: str, limit: int) -> list[dict]:
    """Preserve complete RSS headlines and timestamps for semantic ranking."""
    url = ("https://feeds.finance.yahoo.com/rss/2.0/headline"
           f"?s={quote_plus(ticker.upper())}&region=US&lang=en-US")
    response = requests.get(url, timeout=8)
    response.raise_for_status()
    items = []
    for item in ET.fromstring(response.content).findall("./channel/item"):
        headline = clean_html(item.findtext("title"))
        if headline:
            items.append({
                "headline": headline,
                "summary": clean_html(item.findtext("description")),
                "pubdate": clean_html(item.findtext("pubDate")),
                "url": item.findtext("link") or "",
                "source": "Yahoo Finance RSS",
            })
    return items[:limit]


def load_news(ticker: str, investor_style: str, top_k: int = TOP_K, debug: bool = False) -> tuple[str, dict]:
    """Section 16: style-aware RAG retrieval replaces the old "ticker 최신
    뉴스 10개" loader. Falls back per Section 19 (RAG -> DB recency -> live
    RSS -> "no relevant news") when the DB has nothing usable yet.

    Returns (formatted_news_block, retrieval_result) — the retrieval_result
    dict carries the query and per-article scores for debugging/evaluation
    (Section 17/18/20), even though the scores themselves are not injected
    into the prompt text.
    """
    result = retrieve_news_with_fallback(
        ticker=ticker,
        investor_style=investor_style,
        top_k=top_k,
        db_path=NEWS_DB_PATH,
        live_fallback_fn=_live_news_fallback,
    )
    items = result["top_k"]

    if debug:
        queries = result.get("queries") or {}
        print(
            f"[retrieval] ticker={result['ticker']} style={result['investor_style']} "
            f"source={result['source']} lookback_days={result['lookback_days']} "
            f"merged={result.get('merged_candidate_count')} "
            f"exact_dup={result.get('exact_duplicate_count')} "
            f"core_hit={result.get('core_hit_count')}\n"
            f"  core_query={queries.get('core')!r}\n"
            f"  style_facet_queries={queries.get('style_facets')!r}",
            file=sys.stderr,
        )
        for i, item in enumerate(items, 1):
            print(
                f"  [{i}] final={item.get('final_score')} semantic={item.get('semantic_score')} "
                f"rec={item.get('recency_score')} matched={item.get('matched_queries')} "
                f"event={item.get('event_group_id')} "
                f"{str(item.get('pubdate'))[:10]} | {item.get('headline')}",
                file=sys.stderr,
            )

    if not items:
        return "No relevant news found.", result

    blocks = []
    for i, item in enumerate(items, 1):
        date_str = str(item.get("pubdate") or "")[:10] or "N/A"
        headline = item.get("headline") or "Untitled"
        summary = item.get("summary") or ""
        blocks.append(f"[{i}]\nDate: {date_str}\nHeadline: {headline}\nSummary: {summary}")
    return "\n\n".join(blocks), result


def build_prompt(
    ticker: str,
    investor_style: str,
    top_k: int = TOP_K,
    debug: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    """Section 17 prompt structure. Investor style is normalized once and
    used for BOTH retrieval (Section 10) and generation personalization
    (this prompt), and the same normalized style is what gets shown to the
    model — so a legacy alias like "RISKY" reliably becomes "AGGRESSIVE" in
    both the retrieval query and the report the model is asked to write.
    """
    style = normalize_investor_style(investor_style)
    news_text, retrieval_result = load_news(ticker, style, top_k=top_k, debug=debug)

    user_prompt = f"""Analyze all the provided data and generate a report tailored to the investor's profile.

1. Investor Style
{style}

2. Company Under Review
Ticker: {ticker.upper()}

3. Key Financial Data
{load_company_data(ticker)}

4. Retrieved Relevant News
{news_text}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages, retrieval_result


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description="Run local LoRA report inference.")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument(
        "--style",
        default="SAFE",
        choices=["SAFE", "NEUTRAL", "RISKY", "AGGRESSIVE", "CONSERVATIVE"],
        help="Any alias is normalized to SAFE/NEUTRAL/AGGRESSIVE (see retrieval.normalize_investor_style).",
    )
    parser.add_argument(
        "--adapter", default="fiqa", choices=sorted(ADAPTERS) + ["base"],
        help="'base' runs Meta-Llama-3-8B-Instruct with no LoRA adapter (Section 23: base-vs-LoRA comparison).",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="RAG Top-K news items (Section 14).")
    parser.add_argument(
        "--debug", action="store_true",
        help="Print the retrieval query and per-article scores to stderr (Section 17/18/20).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--min-new-tokens", type=int, default=120)
    args = parser.parse_args()

    hf_token = get_hf_token(required=True)
    device = pick_device()

    if args.adapter == "base":
        print(f"Loading tokenizer from base model {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    else:
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

    if args.adapter == "base":
        print("Running the base model with no LoRA adapter (Section 23 comparison mode).")
        model = base_model
    else:
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(base_model, ADAPTERS[args.adapter])
    model.eval()

    messages, retrieval_result = build_prompt(
        args.ticker, args.style, top_k=args.top_k, debug=args.debug,
    )
    if args.debug:
        print(f"[prompt] investor_style={retrieval_result['investor_style']} "
              f"news_source={retrieval_result['source']}", file=sys.stderr)
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
