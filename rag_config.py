"""Central configuration for the style-aware RAG news pipeline.

`db/db.py` (ingestion, embeddings, event dedup) and `retrieval.py`
(style-aware retrieval, ranking) both import their tunable constants from
here instead of hard-coding magic numbers, so a single file controls the
whole pipeline's behavior. Nothing in this file is a trained parameter —
every value here is a threshold/weight a human can reason about and tune.
"""
from __future__ import annotations

import warnings

# --- Embedding (Section 6) ---
# Same MiniLM model used by the legacy integrated_index; not fine-tuned.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Event-level deduplication (Section 5) ---
# Two articles about the same ticker, published within EVENT_DATE_WINDOW_DAYS
# of each other, whose MiniLM embeddings have cosine similarity above
# EVENT_SIMILARITY_THRESHOLD are treated as the same underlying event and
# tagged with the same event_group_id.
EVENT_SIMILARITY_THRESHOLD = 0.90
EVENT_DATE_WINDOW_DAYS = 2

# --- Retrieval candidate window (Section 13) ---
# Only articles published within NEWS_LOOKBACK_DAYS are considered as
# retrieval candidates. If too few candidates are found, the window is
# widened using the multipliers below (60 -> 120 -> 240 days) before
# falling back to the recency-only fallback in retrieval.py.
NEWS_LOOKBACK_DAYS = 60
NEWS_LOOKBACK_EXPANSION = [1, 2, 4]
MIN_CANDIDATES_BEFORE_EXPAND = 3

# --- Multi-query retrieval (Section 8) ---
# Core Query and every Style Facet Query are embedded and searched
# *separately*, each contributing up to CANDIDATES_PER_QUERY articles to
# the merged candidate pool (Section 9) before reranking. A SAFE ticker
# with 1 core + 3 facet queries can surface up to 4 * CANDIDATES_PER_QUERY
# candidates pre-merge/dedup.
CANDIDATES_PER_QUERY = 5

# --- Reranking score (Section 13) ---
# final_score is a linear combination of two *independently computed*
# scores, not one blended query embedding:
#   - semantic_score: for a candidate that survived the multi-query merge,
#                      the MAX cosine similarity across every query vector
#                      (core + all style facets) that was searched for this
#                      (ticker, investor_style) — not a fixed-weight mix of
#                      "the" core score and "the" style score.
#   - recency_score:   exp decay by publish age, never embedded.
# These are initial heuristics, not a tuned/optimal split — change them
# here, not in retrieval.py, and they don't have to sum to 1 (a warning
# fires below if they don't, but retrieval still runs).
SEMANTIC_WEIGHT = 0.80
RECENCY_WEIGHT = 0.20
HALF_LIFE_DAYS = 14

_WEIGHT_SUM = SEMANTIC_WEIGHT + RECENCY_WEIGHT
if abs(_WEIGHT_SUM - 1.0) > 1e-6:
    warnings.warn(
        f"SEMANTIC_WEIGHT + RECENCY_WEIGHT = {_WEIGHT_SUM:.4f}, not 1.0 "
        "(rag_config.py). Retrieval still runs, but final_score will not "
        "be on the usual ~0-1 scale.",
        stacklevel=2,
    )

# --- Top-K retrieval (Section 15) ---
TOP_K = 6
