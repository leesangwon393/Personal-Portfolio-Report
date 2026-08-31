"""Section 24 Inference tests:
- 기존 최신 뉴스 10개가 아닌 RAG Top-K가 실제 prompt에 들어가는가
- investor style이 retrieval과 generation 양쪽에 모두 사용되는가
- financial data가 정상 포함되는가
- 기존 Llama chat template이 깨지지 않는가

Model/GPU loading is never exercised here — build_prompt() is pure given a
mocked retrieval call, so these tests run without transformers/peft doing
any real model work (Section 24: GPU/API 필요한 테스트는 mocking).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model_inference as mi  # noqa: E402


def _fake_retrieval_result(source: str = "rag") -> dict:
    return {
        "ticker": "NVDA",
        "investor_style": "AGGRESSIVE",
        "query": "NVDA revenue growth earnings surprise guidance raise growth catalyst",
        "lookback_days": 60,
        "source": source,
        "ranked": [],
        "top_k": [
            {
                "article_id": "abc123",
                "headline": "__SENTINEL_RAG_HEADLINE__",
                "summary": "Sentinel summary text used only by this test.",
                "pubdate": "2026-08-20",
                "source": "Yahoo Finance",
                "url": "https://example.com",
                "event_group_id": None,
                "semantic_score": 0.8,
                "recency_score": 0.9,
                "final_score": 0.82,
            }
        ],
    }


class TestBuildPromptUsesRAG(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(mi, "load_company_data", return_value="__SENTINEL_FINANCIALS__")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rag_topk_headline_in_prompt(self):
        with mock.patch.object(
            mi, "retrieve_news_with_fallback", return_value=_fake_retrieval_result()
        ) as mocked:
            messages, retrieval_result = mi.build_prompt("NVDA", "AGGRESSIVE")

        mocked.assert_called_once()
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs["ticker"], "NVDA")
        self.assertEqual(kwargs["top_k"], mi.TOP_K)

        user_content = messages[1]["content"]
        self.assertIn("__SENTINEL_RAG_HEADLINE__", user_content)
        self.assertIn("Retrieved Relevant News", user_content)
        self.assertIn("[1]", user_content)
        self.assertEqual(retrieval_result["source"], "rag")

    def test_investor_style_normalized_for_both_retrieval_and_generation(self):
        with mock.patch.object(
            mi, "retrieve_news_with_fallback", return_value=_fake_retrieval_result()
        ) as mocked:
            messages, _ = mi.build_prompt("NVDA", "RISKY")

        # RISKY (legacy alias) must reach retrieval as AGGRESSIVE ...
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs["investor_style"], "AGGRESSIVE")
        # ... and the prompt shown to the model must say AGGRESSIVE too.
        user_content = messages[1]["content"]
        self.assertIn("1. Investor Style\nAGGRESSIVE", user_content)

    def test_financial_data_included(self):
        with mock.patch.object(
            mi, "retrieve_news_with_fallback", return_value=_fake_retrieval_result()
        ):
            messages, _ = mi.build_prompt("NVDA", "SAFE")
        self.assertIn("__SENTINEL_FINANCIALS__", messages[1]["content"])
        self.assertIn("Key Financial Data", messages[1]["content"])

    def test_fallback_to_no_relevant_news_text(self):
        empty_result = _fake_retrieval_result()
        empty_result["top_k"] = []
        empty_result["source"] = "none"
        with mock.patch.object(mi, "retrieve_news_with_fallback", return_value=empty_result):
            messages, retrieval_result = mi.build_prompt("NVDA", "SAFE")
        self.assertIn("No relevant news found.", messages[1]["content"])
        self.assertEqual(retrieval_result["source"], "none")

    def test_chat_template_message_shape_unchanged(self):
        with mock.patch.object(
            mi, "retrieve_news_with_fallback", return_value=_fake_retrieval_result()
        ):
            messages, _ = mi.build_prompt("NVDA", "SAFE")
        self.assertEqual(len(messages), 2)
        self.assertEqual(set(messages[0].keys()), {"role", "content"})
        self.assertEqual(set(messages[1].keys()), {"role", "content"})
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        # System prompt's core guardrails must survive the change (Section 17).
        self.assertIn("Do not hallucinate", messages[0]["content"])
        self.assertIn("Do NOT provide direct financial advice", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
