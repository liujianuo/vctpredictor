"""SQLite-backed local cache: raw HTML pages and parsed Match objects.

All cache-related code lives here. This module must NOT import
``requests`` or ``BeautifulSoup`` — it only knows how to store and
retrieve strings and ``Match`` objects, never how to fetch or parse.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from .models import Match

# Default cache location: <project root>/cache/vlr_cache.sqlite3
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "cache" / "vlr_cache.sqlite3"

_PAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    html TEXT NOT NULL,
    fetched_at TEXT NOT NULL
)
"""

_MATCHES_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    data TEXT NOT NULL,
    cached_at TEXT NOT NULL
)
"""

_DB_PATH_T = Union[str, Path, None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: _DB_PATH_T = None) -> sqlite3.Connection:
    """Open (creating if needed) the cache DB and ensure tables exist.

    ``db_path=None`` falls back to the module-level
    ``DEFAULT_DB_PATH``, resolved at call time so tests can
    monkeypatch it.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(_PAGES_SCHEMA)
    conn.execute(_MATCHES_SCHEMA)
    conn.commit()
    return conn


def get_cached_page(url: str, db_path: _DB_PATH_T = None) -> Optional[str]:
    """Return cached raw HTML for ``url``, or None on a cache miss."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT html FROM pages WHERE url = ?", (url,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def set_cached_page(url: str, html: str, db_path: _DB_PATH_T = None) -> None:
    """Store raw HTML for ``url``, overwriting any previous entry."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO pages (url, html, fetched_at) VALUES (?, ?, ?) "
            "ON CONFLICT(url) DO UPDATE SET "
            "html = excluded.html, fetched_at = excluded.fetched_at",
            (url, html, _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_match(match_id: str, db_path: _DB_PATH_T = None) -> Optional[Match]:
    """Return cached Match for ``match_id``, or None on a miss.

    A corrupted/unparseable cached row is treated as a miss (return
    None) so the caller re-fetches rather than crashing.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT data FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return Match.from_dict(json.loads(row[0]))
    except (ValueError, TypeError, KeyError):
        return None


def set_cached_match(match: Match, db_path: _DB_PATH_T = None) -> None:
    """Store a parsed Match, overwriting any previous entry for its id."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO matches (match_id, url, data, cached_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(match_id) DO UPDATE SET "
            "url = excluded.url, data = excluded.data, cached_at = excluded.cached_at",
            (match.match_id, match.url, json.dumps(match.to_dict()), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def is_stale(timestamp: Optional[Union[datetime, str]], ttl_seconds: Optional[int]) -> bool:
    """True if ``timestamp`` is older than ``ttl_seconds``.

    - ``ttl_seconds is None`` -> never stale (no expiry configured).
    - ``timestamp is None`` -> stale (no freshness information).
    - Naive datetimes are assumed to be UTC.
    """
    if ttl_seconds is None:
        return False
    if timestamp is None:
        return True
    if isinstance(timestamp, str):
        ts = datetime.fromisoformat(timestamp)
    elif isinstance(timestamp, datetime):
        ts = timestamp
    else:
        raise TypeError(
            f"timestamp must be datetime or ISO-8601 str, got {type(timestamp).__name__}"
        )
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_seconds = (datetime.now(timezone.utc) - ts).total_seconds()
    return age_seconds > ttl_seconds
