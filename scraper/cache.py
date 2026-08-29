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

from .models import IllegalScoreError, Match

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
    """Return the current UTC time as an ISO-8601 string.

    Returns:
        The current time (timezone-aware, UTC) formatted via
        ``datetime.isoformat()``.
    """
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: _DB_PATH_T = None) -> sqlite3.Connection:
    """Open (creating if needed) the cache DB and ensure tables exist.

    Creates the parent directory of ``db_path`` if it does not exist,
    opens (or creates) the SQLite database file, and issues
    ``CREATE TABLE IF NOT EXISTS`` for both the ``pages`` and
    ``matches`` tables before returning the connection.

    Args:
        db_path: Path to the SQLite database file. ``None`` (the
            default) falls back to the module-level
            ``DEFAULT_DB_PATH``, resolved at call time (not import
            time) so tests can monkeypatch it. Accepts
            ``str | Path | None``.

    Returns:
        An open ``sqlite3.Connection`` with the ``pages`` and
        ``matches`` tables guaranteed to exist. The caller is
        responsible for closing it.

    Raises:
        sqlite3.OperationalError: If the database file cannot be
            opened or the schema cannot be created (e.g. permissions
            or disk errors).
        OSError: If the parent directory cannot be created.
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
    """Look up the raw HTML previously cached for a URL.

    Opens its own connection (via :func:`get_connection`) and closes
    it before returning, so it is safe to call repeatedly without
    connection leaks.

    Args:
        url: The page URL to look up, exactly as it was cached (the
            ``pages`` table keys on it verbatim; no normalisation).
        db_path: Path to the SQLite database file, forwarded to
            :func:`get_connection`. ``None`` uses the default path.

    Returns:
        The cached HTML string for ``url``, or ``None`` if there is no
        cache entry for it (a cache miss).

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or
            queried.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT html FROM pages WHERE url = ?", (url,)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def set_cached_page(url: str, html: str, db_path: _DB_PATH_T = None) -> None:
    """Store raw HTML for a URL, overwriting any previous entry.

    Upserts into the ``pages`` table (``INSERT ... ON CONFLICT DO
    UPDATE``), so calling this twice for the same ``url`` replaces the
    old HTML and ``fetched_at`` timestamp rather than erroring or
    duplicating rows.

    Args:
        url: The page URL to cache under. Used verbatim as the primary
            key of the ``pages`` table.
        html: The raw HTML content to store.
        db_path: Path to the SQLite database file, forwarded to
            :func:`get_connection`. ``None`` uses the default path.

    Returns:
        None.

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or
            written to.
    """
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
    """Look up a previously cached, parsed match by its id.

    A corrupt cached row (unparseable JSON, JSON that fails to
    deserialize into a ``Match`` because of a missing key or wrong
    type, or a ``date`` string that is not valid ISO-8601) is treated
    as a miss (returns ``None``) so the caller re-fetches rather than
    crashing on a bad row. A row that deserializes but fails score
    validity is NOT a miss: it is a genuine data problem, so the
    resulting :class:`scraper.models.IllegalScoreError` propagates
    loudly instead of silently forcing a full re-fetch and re-parse
    on every call (which would re-raise the same error anyway).

    Args:
        match_id: The vlr.gg numeric match id to look up (see
            :func:`scraper.vlr.extract_match_id`).
        db_path: Path to the SQLite database file, forwarded to
            :func:`get_connection`. ``None`` uses the default path.

    Returns:
        The cached :class:`scraper.models.Match`, or ``None`` if there
        is no entry for ``match_id`` or the stored JSON is corrupt
        (fails to parse, or is structurally malformed).

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or
            queried.
        IllegalScoreError (a ``ValueError`` subclass): If the stored
            match deserializes to an illegal final map score
            (propagated from
            :meth:`scraper.models.MapResult.__post_init__` via
            :meth:`scraper.models.Match.from_dict`).
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
    except IllegalScoreError:
        # A row that deserializes but fails score validity is a
        # genuine data problem, not corruption: surface it loudly
        # rather than silently treating it as a miss (which would
        # force a full re-fetch and re-parse on every call).
        raise
    except (ValueError, TypeError, KeyError):
        # json.JSONDecodeError (a ValueError subclass) covers corrupt
        # JSON; ValueError also covers a ``date`` field that is not
        # valid ISO-8601 (from datetime.fromisoformat);
        # TypeError/KeyError cover structurally malformed rows. Any
        # of these means the row is corrupt, so it is treated as a
        # miss and the caller re-fetches. The IllegalScoreError above
        # is deliberately not caught here.
        return None


def set_cached_match(match: Match, db_path: _DB_PATH_T = None) -> None:
    """Store a parsed match, overwriting any previous entry for its id.

    Serializes ``match`` via :meth:`scraper.models.Match.to_dict` and
    upserts into the ``matches`` table (``INSERT ... ON CONFLICT DO
    UPDATE``), so calling this twice for the same ``match.match_id``
    replaces the old row rather than erroring or duplicating it.

    Args:
        match: The :class:`scraper.models.Match` to cache. Its
            ``match_id`` is used as the primary key and ``url`` is
            stored alongside the serialized data.
        db_path: Path to the SQLite database file, forwarded to
            :func:`get_connection`. ``None`` uses the default path.

    Returns:
        None.

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or
            written to.
    """
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


def list_cached_match_ids(db_path: _DB_PATH_T = None) -> list[str]:
    """Return every cached match id currently stored in the matches table.

    This is the bulk-read primitive the per-row accessors deliberately
    lack: the module's other functions are all single-row or
    single-URL lookups, but dataset materialisation (roadmap M8) needs
    to enumerate the whole cache to know *what* is there before
    deciding what to read. Ids are returned in SQLite's default rowid
    order, which for a table written exclusively via
    :func:`set_cached_match` (the only writer in this module) is
    insertion order — deterministic for a given cache, so re-running
    materialisation over an unchanged cache visits ids in the same
    sequence. No ordering guarantee is part of the contract (content
    is; callers must not depend on a specific order, only on the set
    of ids being complete).

    The ids are raw strings exactly as stored in the ``match_id``
    primary-key column — they are *not* parsed/validated here. Each
    caller is expected to pass them back into
    :func:`get_cached_match` one at a time, which applies the module's
    usual corrupt-vs-illegal error split per row; this function only
    lists, it does not deserialize anything.

    Args:
        db_path: Path to the SQLite database file, forwarded to
            :func:`get_connection`. ``None`` uses the default path.

    Returns:
        A list of every ``match_id`` string in the ``matches`` table
        (possibly empty).

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or
            queried.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT match_id FROM matches").fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def is_stale(timestamp: Optional[Union[datetime, str]], ttl_seconds: Optional[int]) -> bool:
    """Check whether a cached timestamp is older than a TTL.

    - ``ttl_seconds is None`` -> never stale (no expiry configured).
    - ``timestamp is None`` -> stale (no freshness information).
    - Naive datetimes are assumed to be UTC.

    Args:
        timestamp: When the cached data was fetched/stored. Accepts a
            ``datetime`` (naive datetimes are treated as UTC) or an
            ISO-8601 string, or ``None`` if unknown.
        ttl_seconds: The freshness window in seconds. ``None`` means
            no expiry is configured (data is never considered stale).

    Returns:
        ``True`` if ``timestamp`` is more than ``ttl_seconds`` in the
        past, or if ``timestamp`` is ``None`` while ``ttl_seconds`` is
        not; ``False`` otherwise (including whenever ``ttl_seconds``
        is ``None``).

    Raises:
        TypeError: If ``timestamp`` is neither ``None``, a
            ``datetime``, nor a ``str``.
        ValueError: If ``timestamp`` is a ``str`` that is not valid
            ISO-8601 (propagated from ``datetime.fromisoformat``).
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
