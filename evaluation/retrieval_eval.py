"""Retrieval evaluation for the style-aware RAG pipeline (Section 20/21).

Computes Precision@K / Recall@K / nDCG@K against a small hand-labeled
relevance file (sample_relevance_labels.json). Relevance can differ by
investor_style for the *same* ticker (and even the same article) — a
regulatory-risk article can be highly relevant for a SAFE-style case and
only marginally relevant for an AGGRESSIVE-style case, so each label row
is keyed on (ticker, investor_style), not just ticker.

Also reports a diagnostic (NOT a quality metric) Jaccard overlap between a
ticker's SAFE and AGGRESSIVE Top-K, per Section 21.

Usage:
    python3 evaluation/retrieval_eval.py --labels evaluation/sample_relevance_labels.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retrieval import TOP_K, jaccard_overlap, retrieve_news  # noqa: E402

DEFAULT_LABELS_PATH = Path(__file__).resolve().parent / "sample_relevance_labels.json"


def load_labels(path=None) -> list[dict]:
    path = Path(path or DEFAULT_LABELS_PATH)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Skip documentation/placeholder rows (see sample_relevance_labels.json's _comment entry).
    return [c for c in raw if c.get("ticker") and c["ticker"] != "__EXAMPLE__"]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for a in top if a in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for a in top if a in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, a in enumerate(top) if a in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _average_score(items: list[dict], key: str) -> float:
    """Diagnostic component-score average for this case's Top-K. None values
    (recency/live fallback items, which don't carry component scores) are
    skipped rather than treated as 0.
    """
    vals = [item[key] for item in items if item.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _query_hit_count(items: list[dict], predicate) -> int:
    """How many Top-K items have a matched_queries label satisfying
    `predicate` (e.g. label == "core", or label.startswith a style prefix).
    Fallback items carry matched_queries=None and never count.
    """
    count = 0
    for item in items:
        labels = item.get("matched_queries") or []
        if any(predicate(label) for label in labels):
            count += 1
    return count


def evaluate_case(case: dict, top_k: int = TOP_K, db_path=None) -> dict:
    result = retrieve_news(case["ticker"], case["investor_style"], top_k=top_k, db_path=db_path)
    items = result["top_k"]
    retrieved_ids = [item["article_id"] for item in items]
    relevant_ids = set(case.get("relevant_article_ids") or [])
    return {
        "ticker": result["ticker"],
        "investor_style": result["investor_style"],
        "core_query": result["queries"]["core"],
        "style_facet_queries": result["queries"]["style_facets"],
        "retrieved_ids": retrieved_ids,
        "precision@k": precision_at_k(retrieved_ids, relevant_ids, top_k),
        "recall@k": recall_at_k(retrieved_ids, relevant_ids, top_k),
        "ndcg@k": ndcg_at_k(retrieved_ids, relevant_ids, top_k),
        "average_semantic_score": _average_score(items, "semantic_score"),
        "average_recency_score": _average_score(items, "recency_score"),
        # Section 20 diagnostics: how many Top-K articles were pulled in by
        # the core query vs. by at least one style facet query, the merge
        # pool size before/after exact-dup collapse, and how many distinct
        # events survive into the ranked list (event diversity).
        "core_query_hit_count": _query_hit_count(items, lambda label: label == "core"),
        "style_facet_hit_count": _query_hit_count(items, lambda label: label != "core"),
        "merged_candidate_count": result.get("merged_candidate_count"),
        "exact_duplicate_count": result.get("exact_duplicate_count"),
        "event_diversity_count": len({
            item["event_group_id"] or f"__solo__{item['article_id']}" for item in result["ranked"]
        }),
    }


def diagnostic_style_overlap(ticker: str, top_k: int = TOP_K, db_path=None) -> dict:
    """Section 21: diagnostic only. Some overlap is expected and fine —
    important news (e.g. a major earnings miss) can matter to every style.
    """
    safe = retrieve_news(ticker, "SAFE", top_k=top_k, db_path=db_path)["top_k"]
    aggressive = retrieve_news(ticker, "AGGRESSIVE", top_k=top_k, db_path=db_path)["top_k"]
    return {
        "ticker": ticker.upper(),
        "safe_ids": [a["article_id"] for a in safe],
        "aggressive_ids": [a["article_id"] for a in aggressive],
        "jaccard_overlap": jaccard_overlap(safe, aggressive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    cases = load_labels(args.labels)
    if not cases:
        print("No labeled cases found (fill in evaluation/sample_relevance_labels.json first).")
        return

    results = [evaluate_case(c, top_k=args.top_k, db_path=args.db) for c in cases]

    avg = lambda key: sum(r[key] for r in results) / len(results)
    print(
        f"n={len(results)}  Precision@{args.top_k}={avg('precision@k'):.3f}  "
        f"Recall@{args.top_k}={avg('recall@k'):.3f}  nDCG@{args.top_k}={avg('ndcg@k'):.3f}"
    )
    for r in results:
        print(
            f"- {r['ticker']}/{r['investor_style']}: "
            f"P={r['precision@k']:.2f} R={r['recall@k']:.2f} nDCG={r['ndcg@k']:.2f} "
            f"avg_semantic={r['average_semantic_score']:.3f} avg_recency={r['average_recency_score']:.3f} "
            f"core_hits={r['core_query_hit_count']} facet_hits={r['style_facet_hit_count']} "
            f"merged={r['merged_candidate_count']} exact_dup={r['exact_duplicate_count']} "
            f"event_diversity={r['event_diversity_count']} "
            f"style_facet_queries={r['style_facet_queries']!r}"
        )

    tickers = sorted({c["ticker"] for c in cases})
    print("\n[diagnostic] SAFE vs AGGRESSIVE Top-K overlap (not a quality metric):")
    for t in tickers:
        diag = diagnostic_style_overlap(t, top_k=args.top_k, db_path=args.db)
        print(f"- {t}: jaccard={diag['jaccard_overlap']:.2f}")


if __name__ == "__main__":
    main()
