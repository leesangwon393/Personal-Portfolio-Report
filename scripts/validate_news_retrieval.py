"""Validate source preservation and compare styles on the real news snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_paths import SNAPSHOT_DB_PATH, RUNTIME_DB_PATH
from retrieval import retrieve_news_with_fallback, jaccard_overlap


def validate() -> dict:
    with sqlite3.connect(SNAPSHOT_DB_PATH.as_uri() + '?mode=ro', uri=True) as original, \
         sqlite3.connect(RUNTIME_DB_PATH.as_uri() + '?mode=ro', uri=True) as working:
        articles = original.execute('SELECT id, headline, ticker, pubdate, summary FROM articles ORDER BY id').fetchall()
        preserved = working.execute('SELECT id, headline, ticker, pubdate, summary FROM articles ORDER BY id').fetchall()
        vectors = original.execute('SELECT id, ivect FROM integrated_index ORDER BY id').fetchall()
        preserved_vectors = working.execute('SELECT id, ivect FROM integrated_index ORDER BY id').fetchall()
        assert articles == preserved, 'Article content changed'
        assert vectors == preserved_vectors, 'Original vectors changed'
        stats = {
            'article_count': len(articles), 'original_index_count': len(vectors),
            'new_embedding_count': working.execute('SELECT count(*) FROM article_embeddings').fetchone()[0],
            'ticker_count': original.execute('SELECT count(DISTINCT ticker) FROM articles').fetchone()[0],
            'date_range': original.execute('SELECT min(pubdate), max(pubdate) FROM articles').fetchone(),
            'original_articles_unchanged': True, 'original_vectors_unchanged': True,
        }
    result = {'snapshot_sha256': hashlib.sha256(SNAPSHOT_DB_PATH.read_bytes()).hexdigest(),
              'database': stats, 'comparisons': [],
              'limitation': 'Different article selections demonstrate working personalization, not relevance or investment quality.'}
    for ticker in ('NVDA', 'AAPL', 'MSFT'):
        selections = {style: retrieve_news_with_fallback(ticker, style, top_k=3, db_path=RUNTIME_DB_PATH)
                      for style in ('SAFE', 'NEUTRAL', 'AGGRESSIVE')}
        assert all(len(r['top_k']) == 3 for r in selections.values())
        assert all(r['source'] in ('rag', 'archive_semantic') for r in selections.values())
        result['comparisons'].append({
            'ticker': ticker,
            'safe_aggressive_jaccard': jaccard_overlap(selections['SAFE']['top_k'], selections['AGGRESSIVE']['top_k']),
            'styles': {style: {'source': r['source'], 'articles': [
                {key: article.get(key) for key in ('article_id', 'headline', 'pubdate', 'matched_queries', 'final_score')}
                for article in r['top_k']]} for style, r in selections.items()},
        })
    import app
    with patch.object(app, 'get_latest_prices', return_value=({'NVDA': 100.0}, False)), \
         patch.object(app, 'fetch_live_financials', return_value={}), \
         patch.object(app, 'fetch_live_news', side_effect=AssertionError('Expected stored DB news')):
        response = app.app.test_client().post('/api/analyze', json={'portfolio': 'NVDA 1'})
    assert response.status_code == 200
    report = response.get_json()['reports'][0]
    assert report['news_selection']['source'] in ('rag', 'archive_semantic')
    assert len(report['news']) == 3
    assert '과거 뉴스' in report['news_impact']
    result['dashboard'] = {'status': response.status_code, 'news_count': len(report['news']),
                           'source': report['news_selection']['source'], 'historical_news_notice': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs' / 'news_retrieval_validation.json')
    args = parser.parse_args()
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'database': report['database'], 'style_overlap': {
        item['ticker']: item['safe_aggressive_jaccard'] for item in report['comparisons']
    }}, ensure_ascii=False))
