"""LLM-as-a-Judge generation evaluation (Section 22).

Scores a generated report against the financial data + retrieved news that
were actually used to produce it, on four 1-5 axes:

    faithfulness   — did it avoid inventing facts/numbers not present in
                      the financial data or news it was given?
    personalization — did the analysis focus match the investor style?
    relevance      — did it make good use of the retrieved news and key
                      financial metrics (coverage)?
    coherence      — is it logically organized and readable?

This replaces the original project's Evaluation_by_GPT.ipynb with a plain
script so it can run from the CLI or CI without a notebook dependency.
Requires OPENAI_API_KEY (see secrets_config.py / .env.example).

Typical usage — save a model_inference.py run's prompt sections to files,
then grade the report against them:

    python3 model_inference.py --ticker NVDA --style SAFE --adapter fiqa \\
        --debug > /tmp/nvda_safe_fiqa.txt 2> /tmp/nvda_safe_fiqa.debug.txt

    python3 evaluation/generation_eval.py \\
        --report-file /tmp/nvda_safe_fiqa.txt \\
        --financial-file /tmp/nvda_financials.txt \\
        --news-file /tmp/nvda_safe_news.txt \\
        --style SAFE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from secrets_config import get_openai_api_key  # noqa: E402

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


JUDGE_PROMPT_TMPL = """You are grading an AI-generated personalized stock report.

Investor style: {investor_style}

Financial data given to the model:
{financial_data}

News given to the model:
{news}

Report to grade:
{report}

Score the report from 1 (poor) to 5 (excellent) on each axis:
- faithfulness: did it avoid inventing facts/numbers not present in the financial data or news above?
- personalization: did the analysis focus match the {investor_style} investor style?
- relevance: did it make good use of the retrieved news and the key financial metrics?
- coherence: is it logically organized and readable?

Respond with ONLY a JSON object, e.g.:
{{"faithfulness": 4, "personalization": 5, "relevance": 4, "coherence": 4}}
"""

SCORE_KEYS = ["faithfulness", "personalization", "relevance", "coherence"]


def judge_report(
    investor_style: str,
    financial_data: str,
    news: str,
    report: str,
    model: str = "gpt-4o-mini",
    client=None,
) -> dict:
    if client is None:
        if OpenAI is None:
            raise RuntimeError("openai 패키지가 설치되어야 합니다. pip install openai")
        api_key = get_openai_api_key(required=True)
        client = OpenAI(api_key=api_key)

    prompt = JUDGE_PROMPT_TMPL.format(
        investor_style=investor_style, financial_data=financial_data, news=news, report=report,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        parsed = json.loads(text[start:end + 1])

    scores = {k: float(parsed.get(k, 0)) for k in SCORE_KEYS}
    scores["overall"] = round(sum(scores.values()) / len(scores), 2)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--financial-file", required=True)
    parser.add_argument("--news-file", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    args = parser.parse_args()

    report = Path(args.report_file).read_text(encoding="utf-8")
    financial_data = Path(args.financial_file).read_text(encoding="utf-8")
    news = Path(args.news_file).read_text(encoding="utf-8")

    scores = judge_report(args.style, financial_data, news, report, model=args.judge_model)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
