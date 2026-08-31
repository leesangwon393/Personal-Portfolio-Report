"""Section 24 Retrieval tests:
- ticker filtering 정상 동작
- style query가 SAFE / NEUTRAL / AGGRESSIVE별로 달라지는가
- semantic similarity 정상 계산
- 최근 뉴스일수록 recency score가 높은가
- final score 정상 계산
- event dedup 이후 Top-K가 생성되는가
- fallback 정상 동작
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for p in (ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import db.db as dbmod  # noqa: E402
import retrieval as rag  # noqa: E402
from fake_embedder import FakeEmbedder  # noqa: E402


def _today(offset_days: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=offset_days)).strftime("%Y-%m-%d")


def _new_db() -> str:
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "news.db")
    dbmod.init_db(db_path=db_path)
    return db_path


def _seed_db(rows) -> str:
    db_path = _new_db()
    dbmod.upsert_articles_normalized(pd.DataFrame(rows), db_path=db_path)
    dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder())
    return db_path


class TestBuildRetrievalQuery(unittest.TestCase):
    def test_style_queries_differ(self):
        safe = rag.build_retrieval_query("NVDA", "SAFE")
        neutral = rag.build_retrieval_query("NVDA", "NEUTRAL")
        aggressive = rag.build_retrieval_query("NVDA", "AGGRESSIVE")
        self.assertNotEqual(safe, neutral)
        self.assertNotEqual(neutral, aggressive)
        self.assertIn("NVDA", safe)
        self.assertIn("debt", safe)
        self.assertIn("growth catalyst", aggressive)

    def test_alias_mapping(self):
        self.assertEqual(rag.normalize_investor_style("RISKY"), "AGGRESSIVE")
        self.assertEqual(rag.normalize_investor_style("CONSERVATIVE"), "SAFE")
        self.assertEqual(rag.normalize_investor_style("unknown-style"), "NEUTRAL")
        self.assertEqual(rag.normalize_investor_style(None), "NEUTRAL")


class TestRecencyScore(unittest.TestCase):
    def test_recent_scores_higher_than_old(self):
        recent = rag.calculate_recency_score(_today(0))
        old = rag.calculate_recency_score(_today(60))
        self.assertGreater(recent, old)

    def test_missing_date_scores_zero(self):
        self.assertEqual(rag.calculate_recency_score(None), 0.0)

    def test_half_life_halves_score(self):
        from rag_config import HALF_LIFE_DAYS
        at_zero = rag.calculate_recency_score(_today(0))
        at_half_life = rag.calculate_recency_score(_today(int(HALF_LIFE_DAYS)))
        self.assertAlmostEqual(at_half_life, at_zero / 2, places=2)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors_similarity_one(self):
        import numpy as np
        v = np.array([1.0, 2.0, 3.0], dtype="float32")
        self.assertAlmostEqual(rag.cosine_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors_similarity_zero(self):
        import numpy as np
        a = np.array([1.0, 0.0], dtype="float32")
        b = np.array([0.0, 1.0], dtype="float32")
        self.assertAlmostEqual(rag.cosine_similarity(a, b), 0.0, places=5)


class TestTickerFiltering(unittest.TestCase):
    def test_only_requested_ticker_returned_and_shared_article_both_sides(self):
        db_path = _new_db()
        dbmod.upsert_articles_normalized(pd.DataFrame([
            {"id": None, "headline": "NVDA news", "ticker": "NVDA", "pubdate": _today(1),
             "summary": "Nvidia specific news.", "primary_url": "u1"},
            {"id": None, "headline": "AAPL news", "ticker": "AAPL", "pubdate": _today(1),
             "summary": "Apple specific news.", "primary_url": "u2"},
            {"id": None, "headline": "Shared AI news", "ticker": "NVDA", "pubdate": _today(1),
             "summary": "Shared AI infrastructure news.", "primary_url": "u3"},
        ]), db_path=db_path)
        dbmod.upsert_articles_normalized(pd.DataFrame([
            {"id": None, "headline": "Shared AI news", "ticker": "AAPL", "pubdate": _today(1),
             "summary": "Shared AI infrastructure news.", "primary_url": "u3"},
        ]), db_path=db_path)
        dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder())

        nvda_candidates = rag.fetch_ticker_candidates("NVDA", db_path=db_path, lookback_days=60)
        aapl_candidates = rag.fetch_ticker_candidates("AAPL", db_path=db_path, lookback_days=60)
        self.assertEqual({c["headline"] for c in nvda_candidates}, {"NVDA news", "Shared AI news"})
        self.assertEqual({c["headline"] for c in aapl_candidates}, {"AAPL news", "Shared AI news"})


class TestRetrievePipeline(unittest.TestCase):
    def _seed_nvda(self) -> str:
        return _seed_db([
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "NVDA",
             "pubdate": _today(1), "summary": "NVDA reported quarterly revenue beating estimates.",
             "primary_url": "u1"},
            {"id": None, "headline": "Nvidia quarterly results top estimates", "ticker": "NVDA",
             "pubdate": _today(2), "summary": "NVDA reported quarterly revenue beating estimates.",
             "primary_url": "u2"},
            {"id": None, "headline": "Nvidia faces new export regulation risk", "ticker": "NVDA",
             "pubdate": _today(3),
             "summary": "Regulators proposed export controls raising debt funded compliance costs.",
             "primary_url": "u3"},
            {"id": None, "headline": "Nvidia unveils new AI chip for data centers", "ticker": "NVDA",
             "pubdate": _today(4),
             "summary": "Nvidia launched a new AI chip expanding market demand for growth.",
             "primary_url": "u4"},
        ])

    def test_final_score_matches_weighted_sum_and_is_sorted(self):
        db_path = self._seed_nvda()
        result = rag.retrieve_news("NVDA", "NEUTRAL", top_k=6, db_path=db_path, model=FakeEmbedder())
        self.assertEqual(result["source"], "rag")
        scores = [item["final_score"] for item in result["ranked"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for item in result["ranked"]:
            expected = (
                rag.SEMANTIC_WEIGHT * item["semantic_score"]
                + rag.RECENCY_WEIGHT * item["recency_score"]
            )
            self.assertAlmostEqual(item["final_score"], expected, places=6)

    def test_event_dedup_keeps_topk_events_unique(self):
        db_path = self._seed_nvda()
        dbmod.assign_event_groups(db_path=db_path, similarity_threshold=0.5, date_window_days=3)
        result = rag.retrieve_news("NVDA", "NEUTRAL", top_k=6, db_path=db_path, model=FakeEmbedder())
        event_ids = [item["event_group_id"] for item in result["top_k"] if item["event_group_id"]]
        self.assertEqual(len(event_ids), len(set(event_ids)))
        self.assertLess(len(result["top_k"]), len(result["ranked"]))

    def test_empty_db_returns_empty_source(self):
        db_path = _new_db()
        result = rag.retrieve_news("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        self.assertEqual(result["source"], "empty")
        self.assertEqual(result["top_k"], [])


class TestFallback(unittest.TestCase):
    def test_recency_fallback_when_no_embeddings_built(self):
        db_path = _new_db()
        dbmod.upsert_articles_normalized(pd.DataFrame([
            {"id": None, "headline": "NVDA news no embedding", "ticker": "NVDA",
             "pubdate": _today(1), "summary": "No embeddings built for this yet.", "primary_url": "u1"},
        ]), db_path=db_path)
        # build_article_embeddings intentionally NOT called

        result = rag.retrieve_news_with_fallback("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        self.assertEqual(result["source"], "recency_fallback")
        self.assertEqual(len(result["top_k"]), 1)

    def test_live_fallback_when_db_has_nothing(self):
        db_path = _new_db()

        def fake_live(ticker, top_k):
            return [{"headline": "Live headline", "summary": "Live summary", "pubdate": None, "source": "live"}]

        result = rag.retrieve_news_with_fallback(
            "NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder(), live_fallback_fn=fake_live,
        )
        self.assertEqual(result["source"], "live_fallback")
        self.assertEqual(result["top_k"][0]["headline"], "Live headline")

    def test_no_news_when_everything_empty(self):
        db_path = _new_db()
        result = rag.retrieve_news_with_fallback("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["top_k"], [])


class TestJaccardOverlap(unittest.TestCase):
    def test_identical_sets_overlap_one(self):
        items = [{"article_id": "a"}, {"article_id": "b"}]
        self.assertAlmostEqual(rag.jaccard_overlap(items, items), 1.0)

    def test_disjoint_sets_overlap_zero(self):
        a = [{"article_id": "a"}]
        b = [{"article_id": "b"}]
        self.assertAlmostEqual(rag.jaccard_overlap(a, b), 0.0)


if __name__ == "__main__":
    unittest.main()
