"""Unit tests for the pure metric functions in evaluation/retrieval_eval.py
(Section 20). These don't touch the DB or any embedding model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.retrieval_eval import (  # noqa: E402
    _average_score,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRetrievalMetrics(unittest.TestCase):
    def test_precision_at_k_partial_match(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "z"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 4), 2 / 4)

    def test_precision_at_k_empty_retrieved(self):
        self.assertEqual(precision_at_k([], {"a"}, 4), 0.0)

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c", "z"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 3), 2 / 3)

    def test_recall_at_k_no_relevant(self):
        self.assertEqual(recall_at_k(["a"], set(), 3), 0.0)

    def test_ndcg_perfect_ranking_is_one(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(ndcg_at_k(retrieved, relevant, 3), 1.0, places=6)

    def test_ndcg_worse_ranking_scores_lower(self):
        relevant = {"a", "b"}
        perfect = ndcg_at_k(["a", "b", "c"], relevant, 3)
        worse = ndcg_at_k(["c", "a", "b"], relevant, 3)
        self.assertGreater(perfect, worse)

    def test_ndcg_no_relevant_is_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], set(), 2), 0.0)


class TestAverageScore(unittest.TestCase):
    """average_semantic_score / average_recency_score diagnostics (Section 20)."""

    def test_average_of_present_values(self):
        items = [{"semantic_score": 0.4}, {"semantic_score": 0.6}]
        self.assertAlmostEqual(_average_score(items, "semantic_score"), 0.5)

    def test_none_values_are_skipped_not_zeroed(self):
        # recency/live fallback Top-K items carry semantic_score=None;
        # averaging them in as 0 would understate a case that fell back.
        items = [{"semantic_score": 0.8}, {"semantic_score": None}]
        self.assertAlmostEqual(_average_score(items, "semantic_score"), 0.8)

    def test_empty_items_is_zero(self):
        self.assertEqual(_average_score([], "semantic_score"), 0.0)


if __name__ == "__main__":
    unittest.main()
