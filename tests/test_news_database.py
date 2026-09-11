import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import news_paths
import retrieval
from scripts.prepare_news_db import prepare
from tests.fake_embedder import FakeEmbedder


class TestNewsDatabase(unittest.TestCase):
    def test_preparation_preserves_source_and_old_vectors_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            target = Path(directory) / "runtime.db"
            with sqlite3.connect(source) as conn:
                conn.execute("CREATE TABLE articles (id TEXT PRIMARY KEY, headline TEXT, ticker TEXT, pubdate TEXT, summary TEXT)")
                conn.execute("INSERT INTO articles VALUES ('1','Cash flow stability','NVDA','2025-11-13','Dividend coverage')")
                conn.execute("CREATE TABLE integrated_index (id TEXT PRIMARY KEY, ivect BLOB)")
                conn.execute("INSERT INTO integrated_index VALUES ('1', ?)", (b'original-vector',))
            original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            first = prepare(source, target, model=FakeEmbedder())
            second = prepare(source, target, model=FakeEmbedder())
            self.assertEqual(first["new_embeddings"], 1)
            self.assertEqual(second["new_embeddings"], 0)
            self.assertEqual(second["article_tickers"], 1)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_hash)
            with sqlite3.connect(target) as conn:
                self.assertEqual(conn.execute("SELECT ivect FROM integrated_index").fetchone()[0], b'original-vector')
                self.assertEqual(conn.execute("SELECT summary FROM articles").fetchone()[0], 'Dividend coverage')
                conn.execute("UPDATE articles SET pubdate='2000-01-01'")
            result = retrieval.retrieve_news_with_fallback("NVDA", "SAFE", db_path=target, model=FakeEmbedder())
            self.assertEqual(result["source"], "archive_semantic")
            self.assertEqual(result["top_k"][0]["article_id"], "1")

    def test_preparation_failure_does_not_replace_existing_database(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runtime.db"
            with sqlite3.connect(target) as conn:
                conn.execute("CREATE TABLE original (value TEXT)")
            before = target.read_bytes()
            with patch('db.db.migrate_legacy_articles', side_effect=RuntimeError('migration failed')):
                with self.assertRaises(RuntimeError):
                    prepare(Path(directory) / 'unused.db', target, model=FakeEmbedder())
            self.assertEqual(target.read_bytes(), before)

    def test_snapshot_cannot_be_used_as_writable_destination(self):
        with self.assertRaises(ValueError):
            prepare("same.db", "same.db")

    def test_readers_choose_prepared_database_when_available(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "runtime.db"
            with patch.object(news_paths, 'RUNTIME_DB_PATH', target):
                self.assertEqual(news_paths.get_news_db_path(), news_paths.SNAPSHOT_DB_PATH)
                target.touch()
                self.assertEqual(news_paths.get_news_db_path(), target)
