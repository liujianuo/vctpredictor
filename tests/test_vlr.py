"""Tests for scraper.vlr against saved HTML fixtures (no live network)."""

from datetime import datetime
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from scraper import cache, vlr
from scraper.models import Match, Team, VetoAction

FIXTURES = Path(__file__).parent / "fixtures"

MATCH_HTML = (FIXTURES / "match_page.html").read_text(encoding="utf-8")
UPCOMING_HTML = (FIXTURES / "match_page_upcoming.html").read_text(encoding="utf-8")
EVENT_HTML = (FIXTURES / "event_page.html").read_text(encoding="utf-8")
CLOSE_HTML = (FIXTURES / "match_page_close.html").read_text(encoding="utf-8")
SINGLE_OT_HTML = (FIXTURES / "match_page_single_ot.html").read_text(encoding="utf-8")

MATCH_URL = "https://www.vlr.gg/712803/fut-esports-vs-natus-vincere-vct-2026-emea-stage-2-w1"
UPCOMING_URL = "https://www.vlr.gg/731400/fut-esports-vs-karmine-corp-vct-2026-emea-stage-2-ubf"
EVENT_URL = "https://www.vlr.gg/event/matches/2976/vct-2026-emea-stage-2/?group=completed"
CLOSE_URL = (
    "https://www.vlr.gg/742478/gen-g-vs-nongshim-redforce-"
    "vct-2026-pacific-stage-2-ubsf"
)
SINGLE_OT_URL = (
    "https://www.vlr.gg/731773/team-liquid-brazil-vs-evil-geniuses-gc-"
    "game-changers-2026-brazil-finals-ubsf"
)


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


WINNER_MISMATCH_HTML = """
<div class="vm-stats-game">
<div class="vm-stats-game-header">
<div class="team">
<div class="score">2</div>
</div>
<div class="map">
<div><span>Ascent</span></div>
</div>
<div class="team mod-right">
<div class="score">13</div>
</div>
<div class="score mod-win"></div>
</div>
</div>
"""


def test_parse_map_winner_score_mismatch_raises_vlr_parse_error():
    # The .score.mod-win element sits outside any .team div, so the
    # ancestor lookup in _parse_map fails and the parser falls back to
    # team1's name. team2 outscored team1 (2-13), so that fallback
    # label contradicts the final score: it must fail loudly rather
    # than cache a mislabeled winner.
    game_el = BeautifulSoup(WINNER_MISMATCH_HTML, "lxml").select_one(".vm-stats-game")
    with pytest.raises(vlr.VlrParseError) as excinfo:
        vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))
    message = str(excinfo.value)
    assert "Ascent" in message
    assert "Team A" in message
    assert "Team B" in message


MISSING_SCORE_HTML = """
<div class="vm-stats-game">
<div class="vm-stats-game-header">
<div class="team">
<div class="score">-</div>
</div>
<div class="map">
<div><span>Ascent</span></div>
</div>
<div class="team mod-right">
<div class="score">13</div>
</div>
<div class="score mod-win"></div>
</div>
</div>
"""

MISSING_SCORE_WIN_INSIDE_HTML = """
<div class="vm-stats-game">
<div class="vm-stats-game-header">
<div class="team">
<div class="score mod-win">-</div>
</div>
<div class="map">
<div><span>Ascent</span></div>
</div>
<div class="team mod-right">
<div class="score">13</div>
</div>
</div>
</div>
"""


def test_parse_map_missing_score_drops_unverifiable_winner():
    # One score renders as non-numeric text ("-", e.g. a
    # forfeited/awarded map) while a .score.mod-win element is still
    # present. The win element sits outside any .team div here, so the
    # ancestor lookup fails and the fallback would label team1 as the
    # winner — but team2's 13 is the only parsed score, so that label
    # is unverifiable. The parser must drop the winner (None) rather
    # than cache a possibly-wrong label, and must not raise.
    game_el = BeautifulSoup(MISSING_SCORE_HTML, "lxml").select_one(".vm-stats-game")
    result = vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))
    assert result.map_name == "Ascent"
    assert result.team1_score is None
    assert result.team2_score == 13
    assert result.winner is None


def test_parse_map_missing_score_drops_winner_even_when_mod_win_is_positioned():
    # Same missing-score situation, but the .score.mod-win element is
    # correctly positioned inside team1's div (so the ancestor lookup
    # succeeds and names team1). team1's score still failed to parse,
    # so the winner cannot be verified against the final scores and
    # must still be dropped rather than trusted blindly.
    game_el = BeautifulSoup(MISSING_SCORE_WIN_INSIDE_HTML, "lxml").select_one(
        ".vm-stats-game"
    )
    result = vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))
    assert result.team1_score is None
    assert result.team2_score == 13
    assert result.winner is None


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
# _parse_veto_note
# --------------------------------------------------------------------------

# The exact (stripped) .match-header-note text from match_page.html:
# a standard Bo3 shape of ban, ban, pick, pick, ban, ban, decider.
VETO_NOTE = (
    "NAVI ban Haven; FUT ban Breeze; NAVI pick Split; FUT pick Ascent; "
    "NAVI ban Lotus; FUT ban Summit; Sunset remains"
)


def test_parse_veto_note_happy_path():
    # The full 7-segment Bo3 veto log parses into 7 VetoActions in
    # order: ban, ban, pick, pick, ban, ban, decider — each with the
    # right team token, action and map, and sequential step_index.
    actions = vlr._parse_veto_note(VETO_NOTE)
    assert len(actions) == 7
    assert [a.step_index for a in actions] == [0, 1, 2, 3, 4, 5, 6]
    assert [(a.team, a.action) for a in actions] == [
        ("NAVI", "ban"),
        ("FUT", "ban"),
        ("NAVI", "pick"),
        ("FUT", "pick"),
        ("NAVI", "ban"),
        ("FUT", "ban"),
        (None, "decider"),
    ]
    assert [a.map_name for a in actions] == [
        "Haven",
        "Breeze",
        "Split",
        "Ascent",
        "Lotus",
        "Summit",
        "Sunset",
    ]
    # Spot-check full equality on the first and last actions.
    assert actions[0] == VetoAction(step_index=0, team="NAVI", action="ban", map_name="Haven")
    assert actions[6] == VetoAction(step_index=6, team=None, action="decider", map_name="Sunset")


def test_parse_veto_note_unrecognized_segment_raises():
    # A segment matching neither the ban/pick nor the decider pattern
    # must fail loudly with the raw segment text in the message, not
    # be silently skipped.
    with pytest.raises(vlr.VlrParseError) as excinfo:
        vlr._parse_veto_note("NAVI ban Haven; NAVI swaps Haven")
    assert "NAVI swaps Haven" in str(excinfo.value)


def test_parse_veto_note_empty_or_whitespace_returns_empty():
    # An empty or all-whitespace note yields no actions rather than
    # raising.
    assert vlr._parse_veto_note("") == []
    assert vlr._parse_veto_note("   ") == []


def test_parse_veto_note_stray_trailing_semicolon_is_dropped():
    # A stray trailing ';' produces an empty final segment which is
    # dropped; the remaining segment still parses with step_index 0.
    actions = vlr._parse_veto_note("NAVI ban Haven;")
    assert actions == [VetoAction(step_index=0, team="NAVI", action="ban", map_name="Haven")]


# --------------------------------------------------------------------------
# parse_match — veto_actions end to end
# --------------------------------------------------------------------------


def test_parse_match_veto_actions_populated():
    # match_page.html carries a real .match-header-note; parse_match
    # must surface it as 7 structured actions on the Match.
    m = vlr.parse_match(MATCH_HTML, MATCH_URL)
    assert len(m.veto_actions) == 7
    assert m.veto_actions[0] == VetoAction(step_index=0, team="NAVI", action="ban", map_name="Haven")
    assert m.veto_actions[2] == VetoAction(step_index=2, team="NAVI", action="pick", map_name="Split")
    assert m.veto_actions[6] == VetoAction(step_index=6, team=None, action="decider", map_name="Sunset")


def test_parse_match_veto_actions_populated_second_fixture():
    # A second real Bo3 fixture (match_page_close.html, GEN vs NS)
    # also renders a note; the parser must populate it too.
    m = vlr.parse_match(CLOSE_HTML, CLOSE_URL)
    assert len(m.veto_actions) == 7
    assert m.veto_actions[0] == VetoAction(step_index=0, team="NS", action="ban", map_name="Lotus")
    assert m.veto_actions[5] == VetoAction(step_index=5, team="GEN", action="ban", map_name="Abyss")
    assert m.veto_actions[6] == VetoAction(step_index=6, team=None, action="decider", map_name="Sunset")


def test_parse_match_veto_actions_empty_when_no_note():
    # match_page_single_ot.html and match_page_upcoming.html have no
    # .match-header-note element; veto_actions must be an empty list,
    # not an error (same convention as unavailable maps/scores).
    m = vlr.parse_match(SINGLE_OT_HTML, SINGLE_OT_URL)
    assert m.veto_actions == []
    m = vlr.parse_match(UPCOMING_HTML, UPCOMING_URL)
    assert m.veto_actions == []


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
