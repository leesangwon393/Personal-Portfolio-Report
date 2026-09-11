"""Prepare a writable semantic-search DB while preserving the bundled snapshot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_paths import RUNTIME_DB_PATH, SNAPSHOT_DB_PATH


def prepare(source=SNAPSHOT_DB_PATH, destination=RUNTIME_DB_PATH, model=None) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("The working DB must be separate from the source snapshot")
    input_path = destination if destination.exists() else source
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".sqlite", dir=destination.parent)
    os.close(descriptor)
    try:
        with sqlite3.connect(input_path.as_uri() + "?mode=ro", uri=True) as original, \
             sqlite3.connect(temporary) as working:
            original.backup(working)

        from db.db import migrate_legacy_articles, build_article_embeddings, assign_event_groups
        if model is None:
            from sentence_transformers import SentenceTransformer
            from rag_config import EMBEDDING_MODEL_NAME
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            model.max_seq_length = 512

        migrate_legacy_articles(db_path=temporary)
        built = build_article_embeddings(db_path=temporary, model=model)
        assign_event_groups(db_path=temporary)
        with sqlite3.connect(temporary) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Prepared DB failed its integrity check")
            counts = {table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                      for table in ("articles", "article_tickers", "article_embeddings", "integrated_index")}
        os.replace(temporary, destination)
        return {"destination": str(destination), "new_embeddings": built, **counts}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SNAPSHOT_DB_PATH)
    parser.add_argument("--destination", type=Path, default=RUNTIME_DB_PATH)
    args = parser.parse_args()
    print(prepare(args.source, args.destination))


if __name__ == "__main__":
    main()
