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

from evaluation.retrieval_eval import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
