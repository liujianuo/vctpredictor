"""Tests for scraper.vlr against saved HTML fixtures (no live network)."""

from datetime import datetime
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from scraper import cache, vlr
from scraper.models import Match, Team

FIXTURES = Path(__file__).parent / "fixtures"

MATCH_HTML = (FIXTURES / "match_page.html").read_text(encoding="utf-8")
UPCOMING_HTML = (FIXTURES / "match_page_upcoming.html").read_text(encoding="utf-8")
EVENT_HTML = (FIXTURES / "event_page.html").read_text(encoding="utf-8")

MATCH_URL = "https://www.vlr.gg/712803/fut-esports-vs-natus-vincere-vct-2026-emea-stage-2-w1"
UPCOMING_URL = "https://www.vlr.gg/731400/fut-esports-vs-karmine-corp-vct-2026-emea-stage-2-ubf"
EVENT_URL = "https://www.vlr.gg/event/matches/2976/vct-2026-emea-stage-2/?group=completed"


# --------------------------------------------------------------------------
# extract_match_id
# --------------------------------------------------------------------------


def test_extract_match_id_from_urls():
    assert vlr.extract_match_id("https://www.vlr.gg/712803/foo-vs-bar") == "712803"
    assert vlr.extract_match_id("/712803/foo-vs-bar") == "712803"
    assert vlr.extract_match_id("712803") == "712803"


def test_extract_match_id_rejects_non_match_url():
    with pytest.raises(vlr.VlrParseError):
        vlr.extract_match_id("https://www.vlr.gg/event/matches/2976/?group=completed")


# --------------------------------------------------------------------------
# parse_match
# --------------------------------------------------------------------------


INVALID_SCORE_HTML = """
<div class="vm-stats-game">
<div class="vm-stats-game-header">
<div class="team">
<div class="score">13</div>
</div>
<div class="map">
<div><span>Ascent</span></div>
</div>
<div class="team mod-right">
<div class="score mod-win">12</div>
</div>
</div>
</div>
"""


def test_parse_map_invalid_score_raises_vlr_parse_error():
    # 13-12 with a declared winner is an illegal final scoreline
    # (overtime, margin < 2). _parse_map must surface it as a
    # VlrParseError (via the score-validity wrapper), not a raw
    # ValueError.
    game_el = BeautifulSoup(INVALID_SCORE_HTML, "lxml").select_one(".vm-stats-game")
    with pytest.raises(vlr.VlrParseError) as excinfo:
        vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))
    message = str(excinfo.value)
    assert "Ascent" in message
    assert "13" in message
    assert "12" in message


def test_parse_match_completed():
    m = vlr.parse_match(MATCH_HTML, MATCH_URL)
    assert m.match_id == "712803"
    assert m.url == MATCH_URL
    assert m.event_name == "VCT 2026: EMEA Stage 2"
    assert m.date == datetime(2026, 7, 15, 11, 0, 0)
    assert m.team1.name == "FUT Esports"
    assert m.team1.team_id == "1184"
    assert m.team2.name == "Natus Vincere"
    assert m.team2.team_id == "4915"
    assert m.team1_score == 0
    assert m.team2_score == 2
    assert m.best_of == "Bo3"
    assert m.status == "completed"
    assert len(m.maps) == 2
    assert m.maps[0].map_name == "Split"
    assert m.maps[0].team1_score == 6
    assert m.maps[0].team2_score == 13
    assert m.maps[0].winner == "Natus Vincere"
    assert m.maps[0].duration == "59:20"
    assert m.maps[1].map_name == "Ascent"
    assert m.maps[1].team1_score == 9
    assert m.maps[1].team2_score == 13
    assert m.maps[1].winner == "Natus Vincere"


def test_parse_match_upcoming():
    m = vlr.parse_match(UPCOMING_HTML, UPCOMING_URL)
    assert m.status == "upcoming"
    assert m.team1_score is None
    assert m.team2_score is None
    assert m.maps == []
    assert m.team1.name == "FUT Esports"
    assert m.team2.name == "Karmine Corp"
    assert m.best_of == "Bo3"
    assert m.event_name == "VCT 2026: EMEA Stage 2"


def test_parse_match_rejects_non_match_page():
    with pytest.raises(vlr.VlrParseError):
        vlr.parse_match("<html><body>no match header here</body></html>", "https://www.vlr.gg/1/x")


def test_parse_match_extracts_match_id_from_url_not_html():
    m = vlr.parse_match(MATCH_HTML, "https://www.vlr.gg/999999/some-other-slug")
    assert m.match_id == "999999"


# --------------------------------------------------------------------------
# parse_event_match_links
# --------------------------------------------------------------------------


def test_parse_event_match_links():
    links = vlr.parse_event_match_links(EVENT_HTML)
    assert len(links) > 10
    assert all(link.startswith("/") for link in links)
    assert all(vlr.extract_match_id(link) for link in links)  # all are match URLs
    assert len(set(links)) == len(links)  # no duplicates


def test_parse_event_match_links_empty():
    assert vlr.parse_event_match_links("<html><body>nothing</body></html>") == []


# --------------------------------------------------------------------------
# fetch_page
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            import requests as _r

            raise _r.exceptions.HTTPError(f"{self.status_code} error")


def test_fetch_page_caches_and_force_refresh(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "DEFAULT_DB_PATH", tmp_path / "c.sqlite3")
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        assert kwargs["headers"]["User-Agent"]
        return _FakeResponse(f"html-{calls['n']}")

    monkeypatch.setattr(vlr.requests, "get", fake_get)
    assert vlr.fetch_page(MATCH_URL) == "html-1"
    assert vlr.fetch_page(MATCH_URL) == "html-1"  # served from cache
    assert calls["n"] == 1
    assert vlr.fetch_page(MATCH_URL, force_refresh=True) == "html-2"
    assert calls["n"] == 2


def test_fetch_page_raises_vlr_fetch_error_on_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "DEFAULT_DB_PATH", tmp_path / "c.sqlite3")
    monkeypatch.setattr(
        vlr.requests,
        "get",
        lambda url, **kwargs: _FakeResponse("oops", status_code=404),
    )
    with pytest.raises(vlr.VlrFetchError):
        vlr.fetch_page(MATCH_URL)


def test_fetch_page_raises_vlr_fetch_error_on_network_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "DEFAULT_DB_PATH", tmp_path / "c.sqlite3")

    def boom(url, **kwargs):
        import requests as _r

        raise _r.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(vlr.requests, "get", boom)
    with pytest.raises(vlr.VlrFetchError):
        vlr.fetch_page(MATCH_URL)


# --------------------------------------------------------------------------
# get_match / get_matches_from_event (cache-backed, requests mocked)
# --------------------------------------------------------------------------


def test_get_match_uses_cache_after_first_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "DEFAULT_DB_PATH", tmp_path / "c.sqlite3")
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _FakeResponse(MATCH_HTML)

    monkeypatch.setattr(vlr.requests, "get", fake_get)
    m1 = vlr.get_match(MATCH_URL)
    m2 = vlr.get_match(MATCH_URL)
    assert m1 == m2
    assert m1.match_id == "712803"
    assert calls["n"] == 1  # second call served entirely from cache


def test_get_matches_from_event(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "DEFAULT_DB_PATH", tmp_path / "c.sqlite3")
    monkeypatch.setattr(vlr, "POLITE_DELAY_SECONDS", 0)
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if "event/matches" in url:
            return _FakeResponse(EVENT_HTML)
        return _FakeResponse(MATCH_HTML)

    monkeypatch.setattr(vlr.requests, "get", fake_get)

    matches = vlr.get_matches_from_event(EVENT_URL)
    assert len(matches) > 0
    assert all(isinstance(m, Match) for m in matches)
    assert len({m.match_id for m in matches}) == len(matches)  # unique ids
    # 1 event page fetch + 1 fetch per match
    assert calls["n"] == len(matches) + 1

    # Second call: everything (event page + matches) comes from cache.
    calls["n"] = 0
    matches2 = vlr.get_matches_from_event(EVENT_URL)
    assert calls["n"] == 0
    assert matches2 == matches
