"""Audit every stored ticker with real embeddings; no external news/API calls."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from news_paths import RUNTIME_DB_PATH
from retrieval import (
    build_retrieval_queries, retrieve_news_with_fallback, jaccard_overlap,
    _get_embedding_model, is_explicit_ticker_mismatch,
)


class CachedQueries:
    def __init__(self, queries):
        unique = sorted(set(queries))
        vectors = _get_embedding_model().encode(unique, batch_size=64, normalize_embeddings=True)
        self.vectors = dict(zip(unique, vectors))

    def encode(self, texts, normalize_embeddings=True):
        return np.asarray([self.vectors[text] for text in texts])


def audit():
    with sqlite3.connect(RUNTIME_DB_PATH.as_uri() + '?mode=ro', uri=True) as connection:
        tickers = [r[0] for r in connection.execute('SELECT DISTINCT ticker FROM article_tickers ORDER BY ticker')]
    styles = ('SAFE', 'NEUTRAL', 'AGGRESSIVE')
    queries = []
    for ticker in tickers:
        for style in styles:
            shaped = build_retrieval_queries(ticker, style)
            queries.extend(shaped['core'] + shaped['style_facets'])
    model = CachedQueries(queries)
    results = []
    for ticker in tickers:
        selected = {style: retrieve_news_with_fallback(ticker, style, top_k=3,
                    db_path=RUNTIME_DB_PATH, model=model) for style in styles}
        per_style = {}
        for style, selection in selected.items():
            articles = selection['top_k']
            assert len({a['article_id'] for a in articles}) == len(articles)
            assert not any(is_explicit_ticker_mismatch(ticker, a) for a in articles)
            per_style[style] = {
                'source': selection['source'], 'candidate_count': selection['candidate_pool_size'],
                'excluded_ticker_mismatch_ids': selection.get('excluded_ticker_mismatch_ids', []),
                'articles': [{key: a.get(key) for key in (
                    'article_id', 'headline', 'summary', 'pubdate', 'best_query', 'query_scores', 'final_score'
                )} for a in articles],
            }
        results.append({'ticker': ticker, 'styles': per_style,
                        'safe_aggressive_jaccard': jaccard_overlap(selected['SAFE']['top_k'], selected['AGGRESSIVE']['top_k'])})
    lists = [style for result in results for style in result['styles'].values()]
    slots = [a for style in lists for a in style['articles']]
    stats = {
        'tickers': len(tickers), 'style_queries': len(lists),
        'lists_with_three_articles': sum(len(r['articles']) == 3 for r in lists),
        'empty_lists': sum(not r['articles'] for r in lists),
        'tickers_with_different_safe_aggressive_sets': sum(r['safe_aggressive_jaccard'] < 1 for r in results),
        'tickers_with_identical_safe_aggressive_sets': sum(r['safe_aggressive_jaccard'] == 1 for r in results),
        'core_best_slots': sum(a['best_query'] == 'core' for a in slots),
        'selected_slots': len(slots),
        'excluded_ticker_article_pairs': len({(r['ticker'], aid) for r in results
            for s in r['styles'].values() for aid in s['excluded_ticker_mismatch_ids']}),
    }
    report = {'summary': stats, 'scope': 'Stored 2025 snapshot; structural checks across all tickers. No independent relevance ground truth.',
              'limitations': ['Ticker-mismatch safeguard covers nine explicitly configured tickers, not all companies.',
                             'Best-query scores explain ranking; they are not calibrated relevance probabilities.'],
              'results': results}
    output = ROOT / 'docs' / 'news_selection_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == '__main__':
    audit()
