"""Tests for scraper.cache (SQLite page/match caching)."""

from datetime import datetime, timedelta, timezone

from scraper import cache
from scraper.models import MapResult, Match, Team


def make_match(match_id="1001", url="https://www.vlr.gg/1001/alpha-vs-beta"):
    return Match(
        match_id=match_id,
        url=url,
        event_name="Test Event",
        date=datetime(2026, 7, 15, 11, 0, 0),
        team1=Team(name="Alpha", team_id="1"),
        team2=Team(name="Beta", team_id="2"),
        team1_score=2,
        team2_score=1,
        best_of="Bo3",
        maps=[
            MapResult(map_name="Split", team1_score=13, team2_score=11, winner="Alpha", duration="41:10"),
            MapResult(map_name="Ascent", team1_score=10, team2_score=13, winner="Beta", duration="45:26"),
        ],
        status="completed",
    )


def test_get_connection_creates_db_and_tables(tmp_path):
    db = tmp_path / "nested" / "dir" / "test.sqlite3"
    conn = cache.get_connection(db)
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()
    assert {"pages", "matches"} <= tables
    assert db.exists()


def test_page_round_trip(tmp_path):
    db = tmp_path / "c.sqlite3"
    cache.set_cached_page("https://www.vlr.gg/x", "<html>hello</html>", db_path=db)
    assert cache.get_cached_page("https://www.vlr.gg/x", db_path=db) == "<html>hello</html>"


def test_page_miss_returns_none(tmp_path):
    db = tmp_path / "c.sqlite3"
    assert cache.get_cached_page("https://www.vlr.gg/nope", db_path=db) is None


def test_set_page_overwrites(tmp_path):
    db = tmp_path / "c.sqlite3"
    cache.set_cached_page("https://www.vlr.gg/x", "one", db_path=db)
    cache.set_cached_page("https://www.vlr.gg/x", "two", db_path=db)
    assert cache.get_cached_page("https://www.vlr.gg/x", db_path=db) == "two"


def test_match_round_trip(tmp_path):
    db = tmp_path / "c.sqlite3"
    match = make_match()
    cache.set_cached_match(match, db_path=db)
    got = cache.get_cached_match(match.match_id, db_path=db)
    assert got == match
    assert got.date == match.date
    assert got.maps == match.maps
    assert got.team1 == match.team1


def test_match_miss_returns_none(tmp_path):
    db = tmp_path / "c.sqlite3"
    assert cache.get_cached_match("does-not-exist", db_path=db) is None


def test_match_corrupt_row_treated_as_miss(tmp_path):
    db = tmp_path / "c.sqlite3"
    conn = cache.get_connection(db)
    try:
        conn.execute(
            "INSERT INTO matches (match_id, url, data, cached_at) VALUES (?, ?, ?, ?)",
            ("1", "https://x", "{not json", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    assert cache.get_cached_match("1", db_path=db) is None


def test_match_overwrite_updates(tmp_path):
    db = tmp_path / "c.sqlite3"
    cache.set_cached_match(make_match("1"), db_path=db)
    cache.set_cached_match(make_match("1", url="https://www.vlr.gg/1/new-url"), db_path=db)
    got = cache.get_cached_match("1", db_path=db)
    assert got.url == "https://www.vlr.gg/1/new-url"


def test_is_stale_no_ttl_never_stale():
    assert cache.is_stale(None, None) is False
    assert cache.is_stale(datetime.now(timezone.utc), None) is False


def test_is_stale_missing_timestamp_is_stale():
    assert cache.is_stale(None, 60) is True


def test_is_stale_fresh_within_ttl():
    assert cache.is_stale(datetime.now(timezone.utc), 60) is False
    assert cache.is_stale(datetime.now(timezone.utc).isoformat(), 60) is False


def test_is_stale_past_ttl():
    old = datetime.now(timezone.utc) - timedelta(seconds=120)
    assert cache.is_stale(old, 60) is True
    assert cache.is_stale(old.isoformat(), 60) is True


def test_is_stale_naive_timestamp_assumed_utc():
    # A naive timestamp in the past still counts as stale.
    naive = datetime.utcnow() - timedelta(seconds=120)
    assert cache.is_stale(naive, 60) is True
