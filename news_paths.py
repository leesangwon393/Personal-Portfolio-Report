"""Shared locations for the bundled snapshot and writable news database."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SNAPSHOT_DB_PATH = ROOT_DIR / "db" / "news.db"
NEWS_BASE = Path(os.environ.get("NEWS_DB_BASE", str(ROOT_DIR / "local_data" / "news"))).expanduser()
RUNTIME_DB_PATH = NEWS_BASE / "db" / "news.db"


def get_news_db_path() -> Path:
    """Use the prepared database, or read the original snapshot until prepared."""
    return RUNTIME_DB_PATH if RUNTIME_DB_PATH.is_file() else SNAPSHOT_DB_PATH
