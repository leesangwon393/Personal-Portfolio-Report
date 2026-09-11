"""Section 24 DB tests:
- exact duplicate가 중복 저장되지 않는가
- 하나의 article에 여러 ticker 연결이 가능한가
- embedding이 article별 한 번만 생성되는가
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for p in (ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import db.db as dbmod  # noqa: E402
from fake_embedder import FakeEmbedder  # noqa: E402


def _mk_db() -> str:
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "news.db")
    dbmod.init_db(db_path=db_path)
    return db_path


class TestExactDuplicateIngestion(unittest.TestCase):
    def test_same_url_two_tickers_creates_one_article_two_links(self):
        db_path = _mk_db()
        df = pd.DataFrame([
            {"id": None, "headline": "Nvidia beats earnings", "ticker": "NVDA",
             "pubdate": "2026-08-20", "summary": "NVDA posted strong Q2 earnings.",
             "primary_url": "https://finance.yahoo.com/news/nvda-1"},
            {"id": None, "headline": "Nvidia beats earnings", "ticker": "MSFT",
             "pubdate": "2026-08-20", "summary": "NVDA posted strong Q2 earnings.",
             "primary_url": "https://finance.yahoo.com/news/nvda-1?utm=share"},
        ])
        stats = dbmod.upsert_articles_normalized(df, db_path=db_path)
        self.assertEqual(stats["new_articles"], 1)
        self.assertEqual(stats["duplicate_articles"], 1)

        conn = sqlite3.connect(db_path)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 1)
        tickers = {r[0] for r in conn.execute("SELECT ticker FROM article_tickers").fetchall()}
        conn.close()
        self.assertEqual(tickers, {"NVDA", "MSFT"})

    def test_same_content_hash_different_url_still_dedupes(self):
        db_path = _mk_db()
        df = pd.DataFrame([
            {"id": None, "headline": "Apple unveils new iPhone", "ticker": "AAPL",
             "pubdate": "2026-08-20", "summary": "Apple announced its newest iPhone model today.",
             "primary_url": "https://siteA.example.com/story1"},
            {"id": None, "headline": "Apple unveils new iPhone", "ticker": "AAPL",
             "pubdate": "2026-08-20", "summary": "Apple announced its newest iPhone model today.",
             "primary_url": "https://siteB.example.com/mirror-story1"},
        ])
        stats = dbmod.upsert_articles_normalized(df, db_path=db_path)
        self.assertEqual(stats["new_articles"], 1)
        self.assertEqual(stats["duplicate_articles"], 1)

    def test_different_articles_stay_separate(self):
        db_path = _mk_db()
        df = pd.DataFrame([
            {"id": None, "headline": "Nvidia beats earnings", "ticker": "NVDA",
             "pubdate": "2026-08-20", "summary": "NVDA posted strong Q2 earnings.",
             "primary_url": "https://finance.yahoo.com/news/nvda-1"},
            {"id": None, "headline": "Nvidia unveils new chip", "ticker": "NVDA",
             "pubdate": "2026-08-21", "summary": "NVDA announced a new AI chip.",
             "primary_url": "https://finance.yahoo.com/news/nvda-2"},
        ])
        stats = dbmod.upsert_articles_normalized(df, db_path=db_path)
        self.assertEqual(stats["new_articles"], 2)
        self.assertEqual(stats["duplicate_articles"], 0)


class TestArticleTickersManyToMany(unittest.TestCase):
    def test_one_article_multiple_tickers(self):
        db_path = _mk_db()
        df = pd.DataFrame([
            {"id": None, "headline": "Cloud giants report strong AI demand", "ticker": t,
             "pubdate": "2026-08-20", "summary": "Cloud giants report AI demand growth.",
             "primary_url": "https://finance.yahoo.com/news/cloud-1"}
            for t in ["MSFT", "AMZN", "GOOGL"]
        ])
        dbmod.upsert_articles_normalized(df, db_path=db_path)

        conn = sqlite3.connect(db_path)
        n_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        n_links = conn.execute("SELECT COUNT(*) FROM article_tickers").fetchone()[0]
        conn.close()
        self.assertEqual(n_articles, 1)
        self.assertEqual(n_links, 3)

    def test_relinking_same_ticker_twice_is_idempotent(self):
        db_path = _mk_db()
        row = {"id": None, "headline": "Repeat ingest", "ticker": "NVDA",
               "pubdate": "2026-08-20", "summary": "Same article ingested twice.",
               "primary_url": "https://finance.yahoo.com/news/repeat-1"}
        dbmod.upsert_articles_normalized(pd.DataFrame([row]), db_path=db_path)
        dbmod.upsert_articles_normalized(pd.DataFrame([row]), db_path=db_path)

        conn = sqlite3.connect(db_path)
        n_links = conn.execute("SELECT COUNT(*) FROM article_tickers").fetchone()[0]
        conn.close()
        self.assertEqual(n_links, 1)


class TestArticleEmbeddingsOncePerArticle(unittest.TestCase):
    def test_embedding_created_once_not_per_ticker(self):
        db_path = _mk_db()
        df = pd.DataFrame([
            {"id": None, "headline": "Shared AI news", "ticker": t,
             "pubdate": "2026-08-20", "summary": "Shared AI news across two tickers.",
             "primary_url": "https://finance.yahoo.com/news/shared-1"}
            for t in ["MSFT", "GOOGL"]
        ])
        dbmod.upsert_articles_normalized(df, db_path=db_path)
        n = dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder())
        self.assertEqual(n, 1)

        conn = sqlite3.connect(db_path)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM article_embeddings").fetchone()[0], 1)
        conn.close()

    def test_only_missing_skips_already_embedded(self):
        db_path = _mk_db()
        df = pd.DataFrame([
            {"id": None, "headline": "Article A", "ticker": "NVDA", "pubdate": "2026-08-20",
             "summary": "Summary A.", "primary_url": "u1"},
        ])
        dbmod.upsert_articles_normalized(df, db_path=db_path)
        first = dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder(), only_missing=True)
        second = dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder(), only_missing=True)
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)


if __name__ == "__main__":
    unittest.main()
