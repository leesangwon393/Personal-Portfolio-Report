"""Section 24 Retrieval tests:
- SAFE / NEUTRAL / AGGRESSIVE별 facet query가 올바르게 생성되는가
- Core Query는 모든 style에서 동일한가
- 각 query가 별도로 embedding되어 query마다 Top-N retrieval이 수행되는가
- 동일 article candidate가 merge 시 중복 제거되는가 (matched_queries로 병합 확인)
- ticker filtering이 정상 동작하는가
- semantic score(= 모든 query 중 최대 유사도)와 recency score가 정상 계산되는가
- final score가 semantic/recency 가중합과 일치하는가
- event dedup 이후 Top-K가 생성되는가
- fallback 정상 동작하는가
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


class TestBuildCoreQuery(unittest.TestCase):
    def test_core_query_same_for_every_style(self):
        # build_core_query doesn't take investor_style at all — this is the
        # point: core company info is not personalized.
        self.assertEqual(rag.build_core_query("NVDA"), rag.build_core_query("nvda"))
        self.assertIn("NVDA", rag.build_core_query("NVDA"))
        self.assertIn("earnings", rag.build_core_query("NVDA"))
        self.assertIn("revenue", rag.build_core_query("NVDA"))

    def test_build_retrieval_queries_core_identical_across_styles(self):
        safe = rag.build_retrieval_queries("NVDA", "SAFE")
        aggressive = rag.build_retrieval_queries("NVDA", "AGGRESSIVE")
        self.assertEqual(safe["core"], aggressive["core"])
        self.assertEqual(len(safe["core"]), 1)


class TestBuildStyleFacetQueries(unittest.TestCase):
    def test_each_style_has_multiple_facets(self):
        for style in ["SAFE", "NEUTRAL", "AGGRESSIVE"]:
            facets = rag.build_style_facet_queries("NVDA", style)
            self.assertGreaterEqual(len(facets), 2)
            for label, text in facets:
                self.assertTrue(label.startswith(style.lower()))
                self.assertIn("NVDA", text)

    def test_facet_queries_differ_across_styles(self):
        safe_texts = {text for _, text in rag.build_style_facet_queries("NVDA", "SAFE")}
        aggressive_texts = {text for _, text in rag.build_style_facet_queries("NVDA", "AGGRESSIVE")}
        self.assertTrue(safe_texts.isdisjoint(aggressive_texts))
        safe_all = " ".join(safe_texts)
        aggressive_all = " ".join(aggressive_texts)
        self.assertIn("downside risk", safe_all)
        self.assertIn("growth catalyst", aggressive_all)

    def test_facet_queries_are_not_one_long_query(self):
        # Section 3/21: style concerns must be split into several facet
        # queries, not concatenated into a single long string.
        facets = rag.build_style_facet_queries("NVDA", "SAFE")
        self.assertGreater(len(facets), 1)
        labels = [label for label, _ in facets]
        self.assertEqual(len(labels), len(set(labels)))  # distinct facets

    def test_build_retrieval_queries_shape(self):
        result = rag.build_retrieval_queries("NVDA", "AGGRESSIVE")
        self.assertEqual(set(result.keys()), {"core", "style_facets"})
        self.assertIsInstance(result["core"], list)
        self.assertIsInstance(result["style_facets"], list)
        self.assertGreaterEqual(len(result["style_facets"]), 2)

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

    def test_semantic_score_is_max_similarity_across_all_queries(self):
        # Section 13: semantic_score is the max cosine similarity across
        # EVERY query vector searched (core + all facets), not limited to
        # whichever query(s) the article happened to Top-N in.
        db_path = self._seed_nvda()
        result = rag.retrieve_news("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        model = FakeEmbedder()
        labeled = rag._labeled_queries("NVDA", "SAFE")
        query_vecs = [(label, rag.embed_query(text, model=model)) for label, text in labeled]

        candidates = {c["article_id"]: c for c in rag.fetch_ticker_candidates("NVDA", db_path=db_path, lookback_days=60)}
        for item in result["ranked"]:
            article_vec = rag._vec_from_blob(candidates[item["article_id"]]["embedding"])
            expected_max = max(rag.cosine_similarity(qvec, article_vec) for _, qvec in query_vecs)
            self.assertAlmostEqual(item["semantic_score"], expected_max, places=5)

    def test_core_query_never_changes_matched_queries_label_across_styles(self):
        # The "core" label appears in matched_queries independent of style
        # (it's the same query text every time), even though the *set* of
        # facet labels differs per style.
        db_path = self._seed_nvda()
        safe = rag.retrieve_news("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        aggressive = rag.retrieve_news("NVDA", "AGGRESSIVE", top_k=6, db_path=db_path, model=FakeEmbedder())
        self.assertEqual(safe["queries"]["core"], aggressive["queries"]["core"])
        self.assertNotEqual(safe["queries"]["style_facets"], aggressive["queries"]["style_facets"])

        safe_core_matched = {item["article_id"] for item in safe["ranked"] if "core" in item["matched_queries"]}
        aggressive_core_matched = {item["article_id"] for item in aggressive["ranked"] if "core" in item["matched_queries"]}
        self.assertEqual(safe_core_matched, aggressive_core_matched)

    def test_per_query_topn_respects_candidates_per_query(self):
        db_path = self._seed_nvda()
        result = rag.retrieve_news(
            "NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder(), candidates_per_query=2,
        )
        for label, ids in result["per_query_topn"].items():
            self.assertLessEqual(len(ids), 2)
        self.assertEqual(set(result["per_query_topn"].keys()), set(result["query_labels"]))

    def test_merge_deduplicates_article_matched_by_multiple_queries(self):
        # An article that lands in more than one query's Top-N must appear
        # exactly once in the merged/ranked pool, with all matching labels
        # recorded — never as duplicate rows.
        db_path = self._seed_nvda()
        result = rag.retrieve_news("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        article_ids = [item["article_id"] for item in result["ranked"]]
        self.assertEqual(len(article_ids), len(set(article_ids)))
        matched_more_than_one = [item for item in result["ranked"] if len(item["matched_queries"]) > 1]
        self.assertTrue(matched_more_than_one, "expected at least one article matched by >1 query")

    def test_candidate_pool_and_merge_counts_are_consistent(self):
        db_path = self._seed_nvda()
        result = rag.retrieve_news("NVDA", "SAFE", top_k=6, db_path=db_path, model=FakeEmbedder())
        self.assertEqual(result["candidate_pool_size"], 4)
        self.assertLessEqual(result["merged_candidate_count"], result["candidate_pool_size"])
        self.assertEqual(len(result["ranked"]), result["merged_candidate_count"] - result["exact_duplicate_count"])

    def test_weights_are_overridable_without_editing_retrieval_py(self):
        # SEMANTIC_WEIGHT/RECENCY_WEIGHT are meant to be tunable heuristics,
        # not hard-coded constants baked into the scoring logic —
        # retrieve_news() must accept overrides and use them in final_score.
        db_path = self._seed_nvda()
        result = rag.retrieve_news(
            "NVDA", "NEUTRAL", top_k=6, db_path=db_path, model=FakeEmbedder(),
            semantic_weight=0.3, recency_weight=0.7,
        )
        for item in result["ranked"]:
            expected = 0.3 * item["semantic_score"] + 0.7 * item["recency_score"]
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


class TestExactDuplicateRemoval(unittest.TestCase):
    def test_articles_sharing_content_hash_collapse_to_one_candidate(self):
        # Simulates a legacy DB where two different article_id rows ended
        # up with the same content_hash (ingestion normally prevents this;
        # _drop_exact_duplicates is the merge-time safety net for it).
        db_path = _new_db()
        conn_rows = [
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "NVDA",
             "pubdate": _today(1), "summary": "NVDA reported quarterly revenue beating estimates.",
             "primary_url": "u1"},
        ]
        dbmod.upsert_articles_normalized(pd.DataFrame(conn_rows), db_path=db_path)
        dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder())

        candidates = rag.fetch_ticker_candidates("NVDA", db_path=db_path, lookback_days=60)
        self.assertEqual(len(candidates), 1)
        by_id = {c["article_id"]: c for c in candidates}
        # Duplicate the single row under a second article_id with the same
        # content_hash/url, exactly the "slipped past ingestion" scenario.
        original = candidates[0]
        duped_id = original["article_id"] + "_dup"
        by_id[duped_id] = {**original, "article_id": duped_id}

        deduped_ids, dropped = rag._drop_exact_duplicates(by_id, [original["article_id"], duped_id])
        self.assertEqual(dropped, 1)
        self.assertEqual(deduped_ids, [original["article_id"]])


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
