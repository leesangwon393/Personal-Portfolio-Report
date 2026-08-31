"""Central configuration for the style-aware RAG news pipeline.

`db/db.py` (ingestion, embeddings, event dedup) and `retrieval.py`
(style-aware retrieval, ranking) both import their tunable constants from
here instead of hard-coding magic numbers, so a single file controls the
whole pipeline's behavior. Nothing in this file is a trained parameter —
every value here is a threshold/weight a human can reason about and tune.
"""
from __future__ import annotations

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

# --- Ranking score (Section 12) ---
SEMANTIC_WEIGHT = 0.8
RECENCY_WEIGHT = 0.2
HALF_LIFE_DAYS = 14

# --- Top-K retrieval (Section 14) ---
TOP_K = 6
