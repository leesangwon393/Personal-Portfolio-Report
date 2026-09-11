"""Offline tests for style selection at the application boundaries."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app
import retrieval
from tests.fake_embedder import FakeEmbedder


class TestNewsIntegration(unittest.TestCase):
    def test_live_pool_is_ranked_differently_by_style_without_creating_db(self):
        articles = [
            {"headline": "NVDA " + " ".join(words), "summary": "", "pubdate": None}
            for facets in retrieval.STYLE_FACET_KEYWORDS.values()
            for words in facets.values()
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.db"
            fetch = Mock(return_value=articles)
            results = [retrieval.retrieve_news_with_fallback(
                "NVDA", style, top_k=1, db_path=path,
                model=FakeEmbedder(dim=4096), live_fallback_fn=fetch,
            ) for style in ("SAFE", "AGGRESSIVE")]
            self.assertFalse(path.exists())
            self.assertNotEqual(results[0]["top_k"][0]["headline"], results[1]["top_k"][0]["headline"])
            self.assertTrue(all(r["source"] == "live_semantic" for r in results))
            self.assertGreater(fetch.call_args.args[1], 1)

    def test_legacy_db_can_be_ranked_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "news.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE articles (id TEXT, ticker TEXT, headline TEXT, summary TEXT, pubdate TEXT)")
                connection.execute("INSERT INTO articles VALUES ('1','NVDA','Cash flow stability','Dividends',NULL)")
            result = retrieval.retrieve_news_with_fallback("NVDA", "SAFE", db_path=path, model=FakeEmbedder())
            self.assertEqual(result["source"], "db_semantic")
            self.assertEqual(result["top_k"][0]["article_id"], "1")

    def test_model_unavailable_is_explicit_recency_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            retrieval, "_get_embedding_model", side_effect=OSError("model unavailable")
        ), self.assertLogs("retrieval", level="ERROR"):
            result = retrieval.retrieve_news_with_fallback(
                "NVDA", "SAFE", top_k=1, db_path=Path(directory) / "absent.db",
                live_fallback_fn=lambda *_: [{"headline": "News"}],
            )
        self.assertEqual(result["source"], "live_fallback")
        self.assertIsNone(result["top_k"][0]["semantic_score"])

    def test_dashboard_passes_style_and_uses_selected_news(self):
        selection = {"top_k": [{"headline": "Selected news"}], "source": "rag", "queries": {}}
        with patch.object(app, "retrieve_news_with_fallback", return_value=selection) as retrieve, \
             patch.object(app, "load_financial_metrics", return_value=app.pd.DataFrame()), \
             patch.object(app, "fetch_live_financials", return_value={}):
            reports = app.build_reports({"NVDA": 1.0}, {"category": "AGGRESSIVE", "holdings": []})
        self.assertEqual(retrieve.call_args.args, ("NVDA", "AGGRESSIVE"))
        self.assertEqual(reports[0]["news"][0]["headline"], "Selected news")
        self.assertEqual(reports[0]["investor_style"], "AGGRESSIVE")

    def test_rss_cache_keeps_candidates_beyond_first_request_limit(self):
        response = Mock(content=("<rss><channel>" + "".join(
            f"<item><title>News {i}</title></item>" for i in range(8)
        ) + "</channel></rss>").encode())
        with patch.dict(app.NEWS_CACHE, {}, clear=True), patch.object(app.requests, "get", return_value=response) as get:
            self.assertEqual(len(app.fetch_live_news("NVDA", 3)), 3)
            self.assertEqual(len(app.fetch_live_news("NVDA", 30)), 8)
            get.assert_called_once()

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            retrieval.retrieve_news_with_fallback("NVDA", "SAFE", top_k=0)
