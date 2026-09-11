"""Style-aware multi-query RAG retrieval over the news DB built by db/db.py.

This module replaces the old "ticker 최신 뉴스 10개" loader used by
model_inference.py with a multi-query retrieval pipeline:

    Ticker Candidate Filtering (metadata filter — not embedded)
          -> Core Query + several Style Facet Queries, each embedded and
             searched *separately* (Section 2-8)
          -> Per-query Top-N candidate retrieval + Candidate Merge
             (Section 8/9 — one row per article_id, matched_queries tracked)
          -> Exact Duplicate Removal (Section 11, defensive — ingestion
             already prevents this in the normal path)
          -> Semantic + Recency Reranking (Section 13)
          -> Event Deduplication (Section 12 — one article per event_group)
          -> Top-K

A single article is never scored against one long, blended "style-aware"
query string. Instead, a style's concerns are split into a handful of
narrow Facet Queries (e.g. SAFE = stability / downside_risk / external_risk)
plus one investor_style-independent Core Query, each embedded and searched
on its own. Candidates are merged into one pool keyed by article_id (so an
article matched by several queries is never duplicated), and only *after*
merging is a single semantic_score computed per article — the max cosine
similarity across every query vector that was searched for this (ticker,
investor_style). This keeps a company's core earnings/revenue/cash-flow
news from being crowded out just because a user's style keywords don't
happen to match it (Section 14/21: personalization must not become
information bias), and keeps each query's search intent legible instead of
diluted inside one long embedding.

Personalization happens twice: once here at retrieval time (which Facet
Queries get searched changes with investor_style — Section 4/5/6), and
again at generation time in model_inference.py's system/user prompt.

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
import hashlib
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from news_paths import get_news_db_path

from rag_config import (
    CANDIDATES_PER_QUERY,
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
DEFAULT_DB_PATH = get_news_db_path()


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
    "DEFENSIVE": "SAFE",
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
# Section 2: core query — same for every investor_style
# ──────────────────────────────────────────────────────────────
# What every investor needs to see about a company regardless of risk
# appetite: its core financial performance. This keyword list does not
# change with investor_style, which is the point — it's what keeps a
# company's headline earnings/revenue news showing up for SAFE and
# AGGRESSIVE users alike (Section 14/21).
CORE_QUERY_KEYWORDS: list[str] = [
    "earnings", "revenue", "profitability", "cash flow", "guidance",
    "financial performance",
]


def build_core_query(ticker: str) -> str:
    """Section 2: (ticker) -> core query string, identical across styles."""
    return " ".join([ticker.upper(), *CORE_QUERY_KEYWORDS])


# ──────────────────────────────────────────────────────────────
# Section 4/5/6: style facet query templates (config, easy to tune)
# ──────────────────────────────────────────────────────────────
# A style's concerns are split into 2-3 narrow facets instead of one long
# query string (Section 3/21) — each facet is one clear search intent, so
# an embedding isn't asked to represent "stability AND downside risk AND
# regulation" all at once. Facets give *priority*, not exclusivity: they
# don't exclude the opposite kind of news, they just bias which articles
# rank high for that particular facet's Top-N (Section 8). Core company
# news is already covered by build_core_query() above, so facets only need
# what a given style cares about *on top of* that shared baseline.
STYLE_FACET_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "SAFE": {
        "stability": [
            "financial stability", "debt", "cash flow stability",
            "liquidity", "balance sheet",
        ],
        "downside_risk": [
            "downside risk", "earnings miss", "margin deterioration",
            "guidance cut", "valuation risk",
        ],
        "external_risk": [
            "regulatory risk", "customer concentration", "competition risk",
            "supply risk", "geopolitical risk",
        ],
    },
    "NEUTRAL": {
        "growth": [
            "revenue growth", "earnings growth", "market growth",
            "business expansion",
        ],
        "profitability_valuation": [
            "profitability", "margin", "cash flow", "valuation",
            "financial efficiency",
        ],
        "risk_outlook": [
            "business risk", "guidance", "competition", "financial outlook",
            "uncertainty",
        ],
    },
    "AGGRESSIVE": {
        "growth": [
            "revenue growth", "earnings growth", "market expansion",
            "customer growth",
        ],
        "catalyst": [
            "new products", "innovation", "growth catalyst", "new customers",
            "partnership",
        ],
        "positive_momentum": [
            "earnings surprise", "guidance raise", "market share gain",
            "demand growth", "upside opportunity",
        ],
    },
}


def build_style_facet_queries(ticker: str, investor_style: str) -> list[tuple[str, str]]:
    """Section 4/5/6: (ticker, investor_style) -> [(facet_label, query), ...].

    facet_label (e.g. "safe_downside_risk") is only used internally for
    matched_queries/debug tracking (Section 9/16) — retrieval scoring
    itself doesn't care about the label, only the embedded query text.
    """
    style = normalize_investor_style(investor_style)
    facets = STYLE_FACET_KEYWORDS.get(style, STYLE_FACET_KEYWORDS["NEUTRAL"])
    return [
        (f"{style.lower()}_{name}", " ".join([ticker.upper(), *keywords]))
        for name, keywords in facets.items()
    ]


def _labeled_queries(ticker: str, investor_style: str) -> list[tuple[str, str]]:
    """Core Query (label "core") followed by that style's Facet Queries —
    the actual list of (label, query_text) pairs multi-query retrieval
    embeds and searches independently (Section 7/8).
    """
    style = normalize_investor_style(investor_style)
    return [("core", build_core_query(ticker)), *build_style_facet_queries(ticker, style)]


def build_retrieval_queries(ticker: str, investor_style: str) -> dict:
    """Section 7: (ticker, investor_style) -> {"core": [...], "style_facets": [...]}.

    Plain query-string lists (no labels) — the shape a caller/debug script
    wants to print or hand to an external retriever. retrieve_news() itself
    uses _labeled_queries() so it can still track which facet an article
    matched (matched_queries).
    """
    style = normalize_investor_style(investor_style)
    labeled = _labeled_queries(ticker, style)
    return {
        "core": [text for label, text in labeled if label == "core"],
        "style_facets": [text for label, text in labeled if label != "core"],
    }


# ──────────────────────────────────────────────────────────────
# Section 5: recency as a ranking score (never embedded)
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
    vector — Section 10). An article linked to several tickers is a
    candidate for each of them. Includes content_hash so the multi-query
    merge step (Section 11) can defensively re-check for exact duplicates.
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    if not Path(db_path).is_file():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT a.id AS article_id, a.headline, a.summary, a.pubdate,
                   a.source, a.url, a.event_group_id, a.content_hash, e.embedding
            FROM article_tickers at
            JOIN articles a ON a.id = at.article_id
            JOIN article_embeddings e ON e.article_id = a.id
            WHERE at.ticker = ?
              AND (? IS NULL OR a.pubdate IS NULL OR a.pubdate >= datetime('now', ?))
            """,
            (ticker.upper(), lookback_days,
             f"-{int(lookback_days)} days" if lookback_days is not None else None),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
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
    if not Path(db_path).is_file():
        return []
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
    except sqlite3.OperationalError:
        # Older databases store the ticker directly on articles.
        try:
            rows = conn.execute(
                "SELECT id AS article_id, headline, summary, pubdate FROM articles "
                "WHERE ticker = ? ORDER BY pubdate DESC LIMIT ?",
                (ticker.upper(), top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _drop_exact_duplicates(candidates_by_id: dict, article_ids: list[str]) -> tuple[list[str], int]:
    """Section 11: exact-duplicate safety net at merge time.

    Ingestion (`db.upsert_articles_normalized` / `_lookup_existing_article_id`)
    already prevents two different article_id rows for the same canonical
    URL or content_hash in the normal path, so this should usually drop 0.
    It exists as a defensive re-check for legacy rows (e.g. ingested before
    content_hash backfill via `db.py migrate-legacy`). Order is preserved;
    the first occurrence of a given URL/content_hash is kept.
    """
    seen_hashes: set[str] = set()
    seen_urls: set[str] = set()
    kept = []
    dropped = 0
    for aid in article_ids:
        c = candidates_by_id[aid]
        content_hash = c.get("content_hash")
        url = c.get("url")
        if (content_hash and content_hash in seen_hashes) or (url and url in seen_urls):
            dropped += 1
            continue
        if content_hash:
            seen_hashes.add(content_hash)
        if url:
            seen_urls.add(url)
        kept.append(aid)
    return kept, dropped


# ──────────────────────────────────────────────────────────────
# Section 8-15: full multi-query retrieval pipeline
# ──────────────────────────────────────────────────────────────
def retrieve_news(
    ticker: str,
    investor_style: str,
    top_k: int = TOP_K,
    db_path=None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    model=None,
    candidates_per_query: int = CANDIDATES_PER_QUERY,
    semantic_weight: float = SEMANTIC_WEIGHT,
    recency_weight: float = RECENCY_WEIGHT,
    candidate_articles: list[dict] | None = None,
) -> dict:
    """Runs the full multi-query retrieval pipeline and returns both the
    reranked (pre-event-dedup) candidates and the final top_k list, so
    debugging/eval code can inspect scores without re-running retrieval.

    Core Query and every Style Facet Query (Section 2-6) are embedded and
    searched *separately* against the same ticker-filtered candidate pool.
    Each query contributes its own Top-N (Section 8); those Top-N lists are
    merged into one article_id-keyed pool (Section 9) before a single
    semantic_score per article is computed as the max similarity across
    every query (Section 13) — never a fixed-weight mix of "the" core score
    and "the" style score.
    """
    if top_k < 1 or candidates_per_query < 1:
        raise ValueError("top_k and candidates_per_query must be positive")
    style = normalize_investor_style(investor_style)
    labeled_queries = _labeled_queries(ticker, style)
    query_labels = [label for label, _ in labeled_queries]
    queries_shape = {
        "core": [text for label, text in labeled_queries if label == "core"],
        "style_facets": [text for label, text in labeled_queries if label != "core"],
    }

    windows = [max(1, int(lookback_days * m)) for m in NEWS_LOOKBACK_EXPANSION]
    candidates: list[dict] = []
    used_window = windows[0]
    for window in windows:
        candidates = (candidate_articles if candidate_articles is not None else
                      fetch_ticker_candidates(ticker, db_path=db_path, lookback_days=window))
        used_window = window
        if candidate_articles is not None or len(candidates) >= max(top_k, MIN_CANDIDATES_BEFORE_EXPAND):
            break

    empty_result = {
        "ticker": ticker.upper(),
        "investor_style": style,
        "queries": queries_shape,
        "query_labels": query_labels,
        "lookback_days": used_window,
        "candidate_pool_size": len(candidates),
        "per_query_topn": {label: [] for label in query_labels},
        "merged_candidate_count": 0,
        "exact_duplicate_count": 0,
        "core_hit_count": 0,
        "ranked": [],
        "top_k": [],
        "source": "empty",
    }
    if not candidates:
        return empty_result

    # Ticker Candidate Filtering already happened in fetch_ticker_candidates
    # (Section 10). What's left here is per-query semantic search over that
    # same ticker-filtered pool — ticker is never re-embedded per query.
    candidates_by_id: dict[str, dict] = {}
    article_vecs: dict[str, np.ndarray] = {}
    for c in candidates:
        vec = _vec_from_blob(c["embedding"])
        if vec is None:
            continue
        candidates_by_id[c["article_id"]] = c
        article_vecs[c["article_id"]] = vec

    if not article_vecs:
        return empty_result

    query_vecs = [(label, embed_query(text, model=model)) for label, text in labeled_queries]

    # Similarity of every candidate to every query — cheap in-memory matrix,
    # since both the candidate pool and the query count are small. This is
    # what lets semantic_score (below) use "max across ALL queries" rather
    # than being limited to whichever query an article happened to Top-N in.
    sims: dict[str, dict[str, float]] = {
        aid: {label: cosine_similarity(qvec, vec) for label, qvec in query_vecs}
        for aid, vec in article_vecs.items()
    }

    # Per-query Top-N retrieval + Candidate Merge (Section 8/9): each query
    # independently ranks the full candidate pool and keeps its own
    # Top-N (CANDIDATES_PER_QUERY). matched_queries records which query
    # label(s) put a given article in its Top-N — merging is simply the
    # union of article_ids across all queries, deduplicated by using a
    # dict (an article matched by 2 queries never becomes 2 rows).
    matched_queries: dict[str, list[str]] = {aid: [] for aid in article_vecs}
    per_query_topn: dict[str, list[str]] = {}
    for label in query_labels:
        ranked_for_query = sorted(article_vecs, key=lambda aid: sims[aid][label], reverse=True)
        top_n = ranked_for_query[:candidates_per_query]
        per_query_topn[label] = top_n
        for aid in top_n:
            matched_queries[aid].append(label)

    merged_ids = [aid for aid in article_vecs if matched_queries[aid]]
    merged_candidate_count = len(merged_ids)

    # Exact Duplicate Removal (Section 11), applied to the merged pool.
    deduped_ids, exact_duplicate_count = _drop_exact_duplicates(candidates_by_id, merged_ids)

    # Semantic + Recency Reranking (Section 13).
    ranked = []
    for aid in deduped_ids:
        c = candidates_by_id[aid]
        semantic_score = max(sims[aid].values())
        recency_score = calculate_recency_score(c["pubdate"])
        final_score = semantic_weight * semantic_score + recency_weight * recency_score
        ranked.append({
            "article_id": aid,
            "headline": c["headline"],
            "summary": c["summary"],
            "pubdate": c["pubdate"],
            "source": c["source"],
            "url": c["url"],
            "event_group_id": c["event_group_id"],
            "matched_queries": matched_queries[aid],
            "semantic_score": semantic_score,
            "recency_score": recency_score,
            "final_score": final_score,
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # Event Deduplication (Section 12): keep only the highest-scoring
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

    # Section 14 diagnostic (not enforced as a hard rule): how many Top-K
    # slots are there specifically because of the core query. If this is
    # ever 0 for a ticker with real core-relevant news available, that's a
    # signal candidates_per_query/weights may be crowding core info out.
    core_hit_count = sum(1 for item in top_k_items if "core" in item["matched_queries"])

    return {
        "ticker": ticker.upper(),
        "investor_style": style,
        "queries": queries_shape,
        "query_labels": query_labels,
        "lookback_days": used_window,
        "candidate_pool_size": len(candidates),
        "per_query_topn": per_query_topn,
        "merged_candidate_count": merged_candidate_count,
        "exact_duplicate_count": exact_duplicate_count,
        "core_hit_count": core_hit_count,
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
        2. Embed and style-rank up to max(30, 5 * top_k) DB articles
        3. If DB is empty, fetch and style-rank a live candidate pool
        4. If embeddings are unavailable, return explicitly marked latest news
        5. empty top_k -> caller renders "관련 뉴스 없음"

    `live_fallback_fn` is injected by the caller (model_inference.py /
    app.py) instead of imported here, so this module stays independent of
    any live network fetch implementation.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    try:
        result = retrieve_news(
            ticker, investor_style, top_k=top_k, db_path=db_path,
            lookback_days=lookback_days, model=model,
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        logging.getLogger(__name__).exception("News embedding retrieval unavailable")
        result = retrieve_news(ticker, investor_style, top_k=top_k, candidate_articles=[])
    if result["top_k"]:
        return result

    # Historical snapshots may fall outside the live search window. Reuse
    # their stored text embeddings and label this as archive retrieval.
    archived = fetch_ticker_candidates(ticker, db_path=db_path, lookback_days=None)
    if archived:
        try:
            archive_result = retrieve_news(ticker, investor_style, top_k=top_k,
                                           model=model, candidate_articles=archived)
            if archive_result["top_k"]:
                archive_result["source"] = "archive_semantic"
                archive_result["lookback_days"] = None
                return archive_result
        except (ImportError, OSError, RuntimeError, ValueError):
            logging.getLogger(__name__).exception("Archived news retrieval unavailable")

    candidate_limit = max(30, top_k * 5)
    fallback_articles = fetch_recent_articles_fallback(ticker, top_k=candidate_limit, db_path=db_path)
    fallback_source = "recency_fallback"
    if not fallback_articles and live_fallback_fn is not None:
        try:
            fallback_articles = live_fallback_fn(ticker, candidate_limit) or []
            fallback_source = "live_fallback"
        except Exception:
            fallback_articles = []
    if fallback_articles:
        # Rank a broad candidate pool even when the DB has no stored embeddings.
        try:
            embedder = model or _get_embedding_model()
            texts = [f"{a.get('headline') or ''} {a.get('summary') or ''}" for a in fallback_articles]
            vectors = embedder.encode(texts, normalize_embeddings=True)
            candidates = []
            for article, content, vector in zip(fallback_articles, texts, vectors):
                digest = hashlib.sha256(content.encode()).hexdigest()
                candidates.append({
                    "headline": "", "summary": "", "pubdate": None,
                    "source": None, "url": None, "event_group_id": None,
                    **article, "article_id": str(article.get("article_id") or digest),
                    "content_hash": digest,
                    "embedding": np.asarray(vector, dtype=np.float32).tobytes(),
                })
            ranked = retrieve_news(ticker, investor_style, top_k=top_k,
                                   model=embedder, candidate_articles=candidates)
            if ranked["top_k"]:
                ranked["source"] = "live_semantic" if fallback_source == "live_fallback" else "db_semantic"
                return ranked
        except (ImportError, OSError, RuntimeError, ValueError):
            logging.getLogger(__name__).exception("News candidate ranking unavailable")
        result["top_k"] = [
            {**a, "matched_queries": None, "semantic_score": None, "recency_score": None,
             "final_score": None, "event_group_id": None}
            for a in fallback_articles[:top_k]
        ]
        result["source"] = fallback_source
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
# Section 16/18: debug CLI — inspect queries + candidates + Top-K per style
# ──────────────────────────────────────────────────────────────
def _format_debug(result: dict) -> str:
    lines = [
        f"ticker={result['ticker']} style={result['investor_style']} "
        f"source={result['source']} lookback_days={result['lookback_days']}",
        f"core_query={result['queries']['core']!r}",
        f"style_facet_queries={result['queries']['style_facets']!r}",
        f"candidate_pool_size={result.get('candidate_pool_size')} "
        f"merged_candidate_count={result.get('merged_candidate_count')} "
        f"exact_duplicate_count={result.get('exact_duplicate_count')} "
        f"core_hit_count={result.get('core_hit_count')}",
    ]
    for label, ids in (result.get("per_query_topn") or {}).items():
        lines.append(f"  per_query_topn[{label}]={ids}")
    for i, item in enumerate(result["top_k"], 1):
        lines.append(
            f"  [{i}] final={item.get('final_score')} semantic={item.get('semantic_score')} "
            f"rec={item.get('recency_score')} matched={item.get('matched_queries')} "
            f"event={item.get('event_group_id')} "
            f"{str(item.get('pubdate'))[:10]} | {item.get('headline')}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Style-aware multi-query RAG retrieval debug CLI (Section 16/18)")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--style", default="NEUTRAL")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--db", default=None)
    parser.add_argument("--lookback-days", type=int, default=NEWS_LOOKBACK_DAYS,
                        help="Candidate window; use 400 to inspect the bundled 2025 snapshot.")
    parser.add_argument("--compare-styles", action="store_true",
                         help="Print SAFE/NEUTRAL/AGGRESSIVE Top-K side by side for the same ticker.")
    args = parser.parse_args()

    if args.compare_styles:
        for style in ["SAFE", "NEUTRAL", "AGGRESSIVE"]:
            result = retrieve_news(args.ticker, style, top_k=args.top_k, db_path=args.db,
                                   lookback_days=args.lookback_days)
            print(_format_debug(result))
            print()
    else:
        result = retrieve_news(args.ticker, args.style, top_k=args.top_k, db_path=args.db,
                               lookback_days=args.lookback_days)
        print(_format_debug(result))


if __name__ == "__main__":
    main()
