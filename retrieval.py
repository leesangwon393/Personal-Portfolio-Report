"""Style-aware RAG retrieval over the news DB built by db/db.py.

This module replaces the old "ticker 최신 뉴스 10개" loader used by
model_inference.py with an actual retrieval pipeline:

    Ticker Candidate Filtering (Section 8, metadata filter — not embedded)
          -> Style-aware Query (Section 10/11)
          -> Semantic Similarity (Section 6, MiniLM cosine similarity)
          -> Recency Score (Section 9, ranking score — not embedded)
          -> Final Score (Section 12)
          -> Score Ranking
          -> Event Deduplication (Section 5/14 — one article per event_group)
          -> Top-K (Section 14)

Personalization happens twice: once here at retrieval time (the query
embedded for semantic search changes with investor_style — Section 10/11),
and again at generation time in model_inference.py's system/user prompt.

No FAISS/Chroma/Pinecone/vector DB product is used — candidates and their
MiniLM embeddings live in SQLite (db/news.db, article_embeddings table) and
similarity is a plain numpy cosine computation over the ticker-filtered
rows. Ticker filtering is a metadata filter over article_tickers, not an
embedded vector; recency is a separate ranking score, not part of any
embedding. Random Projection is not used anywhere in this module (see the
legacy section of db/db.py for where it still lives).
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from rag_config import (
    EMBEDDING_MODEL_NAME,
    HALF_LIFE_DAYS,
    MIN_CANDIDATES_BEFORE_EXPAND,
    NEWS_LOOKBACK_DAYS,
    NEWS_LOOKBACK_EXPANSION,
    RECENCY_WEIGHT,
    SEMANTIC_WEIGHT,
    TOP_K,
)

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT_DIR / "db" / "news.db"


# ──────────────────────────────────────────────────────────────
# Section 10: investor style normalization
# ──────────────────────────────────────────────────────────────
# The portfolio classifier (srisk_result/analyze_portfolio_risk.py, NOT
# touched by this change) already emits SAFE/NEUTRAL/AGGRESSIVE directly.
# model_inference.py's CLI historically also accepted "RISKY" as a synonym
# for AGGRESSIVE. This mapping keeps both — and any future aliases —
# resolving to the same three canonical styles the query templates below
# are keyed on.
INVESTOR_STYLE_ALIASES = {
    "SAFE": "SAFE",
    "CONSERVATIVE": "SAFE",
    "LOW": "SAFE",
    "LOW_RISK": "SAFE",
    "NEUTRAL": "NEUTRAL",
    "MODERATE": "NEUTRAL",
    "BALANCED": "NEUTRAL",
    "MEDIUM": "NEUTRAL",
    "AGGRESSIVE": "AGGRESSIVE",
    "RISKY": "AGGRESSIVE",
    "HIGH": "AGGRESSIVE",
    "HIGH_RISK": "AGGRESSIVE",
    "GROWTH": "AGGRESSIVE",
}


def normalize_investor_style(investor_style: str | None) -> str:
    key = (investor_style or "").strip().upper()
    return INVESTOR_STYLE_ALIASES.get(key, "NEUTRAL")


# ──────────────────────────────────────────────────────────────
# Section 11: style-aware query templates (config, easy to tune)
# ──────────────────────────────────────────────────────────────
# Each list gives *priority*, not exclusivity: a SAFE query does not exclude
# growth news and an AGGRESSIVE query does not exclude risk news — they just
# bias which article embeddings end up closest to the query embedding.
STYLE_QUERY_KEYWORDS: dict[str, list[str]] = {
    "SAFE": [
        "financial stability", "cash flow", "debt", "profitability",
        "margin deterioration", "earnings miss", "guidance cut",
        "regulatory risk", "customer concentration", "valuation risk",
        "downside risk",
    ],
    "NEUTRAL": [
        "earnings", "revenue", "profitability", "margins", "cash flow",
        "growth", "guidance", "valuation", "opportunities", "risks",
        "financial outlook",
    ],
    "AGGRESSIVE": [
        "revenue growth", "earnings surprise", "guidance raise",
        "growth catalyst", "new products", "market expansion", "AI demand",
        "new customers", "innovation", "upside opportunities",
    ],
}


def build_retrieval_query(ticker: str, investor_style: str) -> str:
    """Section 10/11: (ticker, investor_style) -> retrieval query string.

    This is the retrieval-side personalization: SAFE/NEUTRAL/AGGRESSIVE
    bias the *semantic search itself*, not just what the LLM is told to
    emphasize afterwards.
    """
    style = normalize_investor_style(investor_style)
    keywords = STYLE_QUERY_KEYWORDS.get(style, STYLE_QUERY_KEYWORDS["NEUTRAL"])
    return " ".join([ticker.upper(), *keywords])


# ──────────────────────────────────────────────────────────────
# Section 9: recency as a ranking score (never embedded)
# ──────────────────────────────────────────────────────────────
def calculate_recency_score(pubdate, *, half_life_days: float = HALF_LIFE_DAYS, now=None) -> float:
    """exp(-ln2 * days_old / half_life_days), clamped to today for future dates."""
    if pubdate is None:
        return 0.0
    try:
        dt = pd.to_datetime(pubdate, utc=True, errors="coerce")
    except Exception:
        return 0.0
    if dt is None or pd.isna(dt):
        return 0.0
    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.utcnow()
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    days_old = max((now_ts - dt).total_seconds() / 86400.0, 0.0)
    return float(math.exp(-math.log(2) * days_old / half_life_days))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ──────────────────────────────────────────────────────────────
# Embedding model (MiniLM, not fine-tuned — Section 6)
# ──────────────────────────────────────────────────────────────
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _embedding_model.max_seq_length = 512
    return _embedding_model


def embed_query(query: str, model=None) -> np.ndarray:
    model = model or _get_embedding_model()
    vec = model.encode([query], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype="float32")


def _vec_from_blob(blob) -> np.ndarray | None:
    if blob is None:
        return None
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    v = np.frombuffer(blob, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ──────────────────────────────────────────────────────────────
# Section 8/13: ticker metadata filter + candidate window
# ──────────────────────────────────────────────────────────────
def fetch_ticker_candidates(ticker: str, db_path=None, lookback_days: int = NEWS_LOOKBACK_DAYS) -> list[dict]:
    """Ticker filtering via article_tickers (metadata filter, not an embedded
    vector — Section 8). An article linked to several tickers is a
    candidate for each of them.
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT a.id AS article_id, a.headline, a.summary, a.pubdate,
                   a.source, a.url, a.event_group_id, e.embedding
            FROM article_tickers at
            JOIN articles a ON a.id = at.article_id
            JOIN article_embeddings e ON e.article_id = a.id
            WHERE at.ticker = ?
              AND (a.pubdate IS NULL OR a.pubdate >= datetime('now', ?))
            """,
            (ticker.upper(), f"-{int(lookback_days)} days"),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def fetch_recent_articles_fallback(ticker: str, top_k: int = TOP_K, db_path=None) -> list[dict]:
    """Section 19, fallback level 2: plain "latest N articles for ticker",
    recency-only, no embeddings required. This is what model_inference's
    old load_news() did before RAG existed — kept as the safety net for
    when article_embeddings hasn't been built yet for this ticker.
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT a.id AS article_id, a.headline, a.summary, a.pubdate,
                   a.source, a.url
            FROM article_tickers at
            JOIN articles a ON a.id = at.article_id
            WHERE at.ticker = ?
              AND a.summary IS NOT NULL AND LENGTH(TRIM(a.summary)) > 0
            ORDER BY COALESCE(a.pubdate, a.created_at) DESC
            LIMIT ?
            """,
            (ticker.upper(), top_k),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Section 12/14: full retrieval pipeline
# ──────────────────────────────────────────────────────────────
def retrieve_news(
    ticker: str,
    investor_style: str,
    top_k: int = TOP_K,
    db_path=None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    model=None,
    semantic_weight: float = SEMANTIC_WEIGHT,
    recency_weight: float = RECENCY_WEIGHT,
) -> dict:
    """Runs the full Section 14 pipeline and returns both the ranked
    (pre-dedup) candidates and the final top_k list, so debugging/eval code
    (Section 17/18/20) can inspect scores without re-running retrieval.
    """
    style = normalize_investor_style(investor_style)
    query = build_retrieval_query(ticker, style)

    windows = [max(1, int(lookback_days * m)) for m in NEWS_LOOKBACK_EXPANSION]
    candidates: list[dict] = []
    used_window = windows[0]
    for window in windows:
        candidates = fetch_ticker_candidates(ticker, db_path=db_path, lookback_days=window)
        used_window = window
        if len(candidates) >= max(top_k, MIN_CANDIDATES_BEFORE_EXPAND):
            break

    if not candidates:
        return {
            "ticker": ticker.upper(),
            "investor_style": style,
            "query": query,
            "lookback_days": used_window,
            "ranked": [],
            "top_k": [],
            "source": "empty",
        }

    query_vec = embed_query(query, model=model)
    ranked = []
    for c in candidates:
        article_vec = _vec_from_blob(c["embedding"])
        if article_vec is None:
            continue
        semantic_score = cosine_similarity(query_vec, article_vec)
        recency_score = calculate_recency_score(c["pubdate"])
        final_score = semantic_weight * semantic_score + recency_weight * recency_score
        ranked.append({
            "article_id": c["article_id"],
            "headline": c["headline"],
            "summary": c["summary"],
            "pubdate": c["pubdate"],
            "source": c["source"],
            "url": c["url"],
            "event_group_id": c["event_group_id"],
            "semantic_score": semantic_score,
            "recency_score": recency_score,
            "final_score": final_score,
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # Event Deduplication (Section 5/14/15): keep only the highest-scoring
    # article per event_group_id so Top-K isn't dominated by one event.
    # Articles with no event_group_id (didn't cluster with anything) are
    # always kept as their own "event".
    top_k_items = []
    seen_events = set()
    for item in ranked:
        event_key = item["event_group_id"] or f"__solo__{item['article_id']}"
        if event_key in seen_events:
            continue
        seen_events.add(event_key)
        top_k_items.append(item)
        if len(top_k_items) >= top_k:
            break

    return {
        "ticker": ticker.upper(),
        "investor_style": style,
        "query": query,
        "lookback_days": used_window,
        "ranked": ranked,
        "top_k": top_k_items,
        "source": "rag",
    }


def retrieve_news_with_fallback(
    ticker: str,
    investor_style: str,
    top_k: int = TOP_K,
    db_path=None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    model=None,
    live_fallback_fn=None,
) -> dict:
    """Section 19 fallback chain:

        1. Style-aware RAG Top-K
        2. DB recency fallback (fetch_recent_articles_fallback)
        3. live_fallback_fn(ticker, top_k) if provided (e.g. Yahoo RSS)
        4. empty top_k -> caller renders "관련 뉴스 없음"

    `live_fallback_fn` is injected by the caller (model_inference.py /
    app.py) instead of imported here, so this module stays independent of
    any live network fetch implementation.
    """
    result = retrieve_news(
        ticker, investor_style, top_k=top_k, db_path=db_path,
        lookback_days=lookback_days, model=model,
    )
    if result["top_k"]:
        return result

    fallback_articles = fetch_recent_articles_fallback(ticker, top_k=top_k, db_path=db_path)
    if fallback_articles:
        result["top_k"] = [
            {**a, "semantic_score": None, "recency_score": None,
             "final_score": None, "event_group_id": None}
            for a in fallback_articles
        ]
        result["source"] = "recency_fallback"
        return result

    if live_fallback_fn is not None:
        try:
            live_items = live_fallback_fn(ticker, top_k) or []
        except Exception:
            live_items = []
        if live_items:
            result["top_k"] = live_items
            result["source"] = "live_fallback"
            return result

    result["source"] = "none"
    return result


# ──────────────────────────────────────────────────────────────
# Section 21: diagnostic-only personalization check (not a quality metric)
# ──────────────────────────────────────────────────────────────
def jaccard_overlap(top_k_a: list[dict], top_k_b: list[dict]) -> float:
    ids_a = {item["article_id"] for item in top_k_a}
    ids_b = {item["article_id"] for item in top_k_b}
    union = ids_a | ids_b
    if not union:
        return 1.0
    return len(ids_a & ids_b) / len(union)


# ──────────────────────────────────────────────────────────────
# Section 18: debug CLI — inspect query + Top-K per style for one ticker
# ──────────────────────────────────────────────────────────────
def _format_debug(result: dict) -> str:
    lines = [
        f"ticker={result['ticker']} style={result['investor_style']} "
        f"source={result['source']} lookback_days={result['lookback_days']}",
        f"query={result['query']!r}",
    ]
    for i, item in enumerate(result["top_k"], 1):
        lines.append(
            f"  [{i}] final={item.get('final_score')} sem={item.get('semantic_score')} "
            f"rec={item.get('recency_score')} event={item.get('event_group_id')} "
            f"{str(item.get('pubdate'))[:10]} | {item.get('headline')}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Style-aware RAG retrieval debug CLI (Section 18)")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--style", default="NEUTRAL")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--db", default=None)
    parser.add_argument("--compare-styles", action="store_true",
                         help="Print SAFE/NEUTRAL/AGGRESSIVE Top-K side by side for the same ticker.")
    args = parser.parse_args()

    if args.compare_styles:
        for style in ["SAFE", "NEUTRAL", "AGGRESSIVE"]:
            result = retrieve_news(args.ticker, style, top_k=args.top_k, db_path=args.db)
            print(_format_debug(result))
            print()
    else:
        result = retrieve_news(args.ticker, args.style, top_k=args.top_k, db_path=args.db)
        print(_format_debug(result))


if __name__ == "__main__":
    main()
