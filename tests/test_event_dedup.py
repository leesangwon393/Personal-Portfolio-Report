"""Section 24 Event Dedup tests:
- 유사한 기사들이 동일 event_group으로 묶이는가
- 서로 다른 이벤트가 잘못 합쳐지지 않는가
"""
from __future__ import annotations

import os
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


def _seed_db(rows) -> str:
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "news.db")
    dbmod.init_db(db_path=db_path)
    dbmod.upsert_articles_normalized(pd.DataFrame(rows), db_path=db_path)
    dbmod.build_article_embeddings(db_path=db_path, model=FakeEmbedder())
    return db_path


class TestEventDeduplication(unittest.TestCase):
    def test_similar_same_window_articles_share_event_group(self):
        db_path = _seed_db([
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "NVDA",
             "pubdate": "2026-08-28", "summary": "NVDA reported quarterly revenue beating estimates on strong demand.",
             "primary_url": "u1"},
            {"id": None, "headline": "Nvidia quarterly results top estimates", "ticker": "NVDA",
             "pubdate": "2026-08-29", "summary": "NVDA reported quarterly revenue beating estimates on strong demand.",
             "primary_url": "u2"},
        ])
        assignments = dbmod.assign_event_groups(db_path=db_path, similarity_threshold=0.6, date_window_days=2)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(len(set(assignments.values())), 1)

    def test_dissimilar_articles_not_merged(self):
        db_path = _seed_db([
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "NVDA",
             "pubdate": "2026-08-28", "summary": "NVDA reported quarterly revenue beating estimates on strong demand.",
             "primary_url": "u1"},
            {"id": None, "headline": "Nvidia faces new export regulation", "ticker": "NVDA",
             "pubdate": "2026-08-28", "summary": "Regulators proposed export controls on advanced chips.",
             "primary_url": "u3"},
        ])
        assignments = dbmod.assign_event_groups(db_path=db_path, similarity_threshold=0.9, date_window_days=2)
        self.assertEqual(assignments, {})

    def test_similar_but_outside_date_window_not_merged(self):
        db_path = _seed_db([
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "NVDA",
             "pubdate": "2026-08-01", "summary": "NVDA reported quarterly revenue beating estimates on strong demand.",
             "primary_url": "u1"},
            {"id": None, "headline": "Nvidia quarterly results top estimates", "ticker": "NVDA",
             "pubdate": "2026-08-29", "summary": "NVDA reported quarterly revenue beating estimates on strong demand.",
             "primary_url": "u2"},
        ])
        assignments = dbmod.assign_event_groups(db_path=db_path, similarity_threshold=0.6, date_window_days=2)
        self.assertEqual(assignments, {})

    def test_different_tickers_are_not_compared(self):
        # Two DIFFERENT tickers each with a single article: nothing to
        # group against within either ticker's own article set.
        db_path = _seed_db([
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "NVDA",
             "pubdate": "2026-08-28", "summary": "NVDA reported quarterly revenue beating estimates.",
             "primary_url": "u1"},
            {"id": None, "headline": "Nvidia beats earnings expectations", "ticker": "AAPL",
             "pubdate": "2026-08-28", "summary": "NVDA reported quarterly revenue beating estimates.",
             "primary_url": "u1"},
        ])
        assignments = dbmod.assign_event_groups(db_path=db_path, similarity_threshold=0.6, date_window_days=2)
        # u1 is the SAME article (same url) linked to both tickers, so it
        # only appears once overall and has nothing else to cluster with.
        self.assertEqual(assignments, {})


if __name__ == "__main__":
    unittest.main()
