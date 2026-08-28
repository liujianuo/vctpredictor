"""Player-map stats parser tests (roadmap M5, task 006).

Exercises the per-player, per-map stats parsing added in task 006
against real vlr.gg fixtures — the same files the score/veto suites
already use, since every completed fixture already contains complete
``.ovw-table`` stats data that the parser previously ignored (no new
fixtures were needed, per plan assumption 7).

Two baseline findings drive the shape of these tests:

1. **Multi-agent rows live in the "All Maps" aggregate block, not the
   per-map tables.** All 8 fixtures' per-map ``.ovw-table`` blocks
   have exactly one agent per player row; the multi-agent rows (an
   agent swap) appear only in the ``data-game-id="all"`` block, which
   ``parse_match`` filters out before per-map parsing (it has no
   ``.vm-stats-game-header``). Assumption 1's ``agents: list[str]``
   behavior is therefore locked in by calling
   ``_parse_player_stats_table`` directly on the real "All Maps"
   tables (7 of their 10 rows carry two agents).

2. **The upcoming fixture's "TBD" placeholder blocks do carry stats
   tables** (player names, empty stat cells, no agents). ``_parse_map``
   runs on them before ``parse_match`` discards the block by its
   "TBD" map name, so the new parsing must neither raise nor leak
   placeholder data into the cache.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from scraper import vlr
from scraper.models import MapResult, PlayerStats, Team

FIXTURES = Path(__file__).parent / "fixtures"

MATCH_HTML = (FIXTURES / "match_page.html").read_text(encoding="utf-8")
BO5_HTML = (FIXTURES / "match_page_bo5.html").read_text(encoding="utf-8")
UPCOMING_HTML = (FIXTURES / "match_page_upcoming.html").read_text(encoding="utf-8")

MATCH_URL = (
    "https://www.vlr.gg/712803/fut-esports-vs-natus-vincere-"
    "vct-2026-emea-stage-2-w1"
)
BO5_URL = (
    "https://www.vlr.gg/724645/jd-gaming-vs-tyloo-"
    "vct-2026-china-stage-2-gf"
)
UPCOMING_URL = (
    "https://www.vlr.gg/731400/fut-esports-vs-karmine-corp-"
    "vct-2026-emea-stage-2-ubf"
)


def test_parse_match_populates_every_player_stats_field():
    # Hand-transcribed from match_page.html's Split map (map 0). Table
    # 0 is FUT Esports in document order; every PlayerStats field is
    # asserted against the exact .side.mod-both values in the fixture
    # HTML. Percentages (kast 74%, hs_pct 27%) must come through as
    # plain floats without the % sign.
    m = vlr.parse_match(MATCH_HTML, MATCH_URL)
    split = m.maps[0]
    assert split.map_name == "Split"
    assert len(split.player_stats) == 10  # 5 players per team

    ps = split.player_stats[0]
    assert ps.player_name == "yetujey"
    assert ps.team_name == "FUT Esports"
    assert ps.rating == 1.08
    assert ps.acs == 230.0
    assert ps.kills == 14
    assert ps.deaths == 15
    assert ps.assists == 4
    assert ps.adr == 171.0
    assert ps.kast == 74.0
    assert ps.hs_pct == 27.0
    assert ps.first_kills == 1
    assert ps.first_deaths == 1
    assert ps.agents == ["viper"]

    # Last row (index 9) is table 1's last player, team2 (Natus
    # Vincere) — confirms both teams populate and the table DOM order
    # (team1 rows, then team2 rows) is preserved.
    ps = split.player_stats[9]
    assert ps.player_name == "Ruxic"
    assert ps.team_name == "Natus Vincere"
    assert ps.rating == 0.78
    assert ps.acs == 185.0
    assert ps.kills == 13
    assert ps.deaths == 15
    assert ps.assists == 8
    assert ps.adr == 126.0
    assert ps.kast == 68.0
    assert ps.hs_pct == 31.0
    assert ps.first_kills == 2
    assert ps.first_deaths == 2
    assert ps.agents == ["omen"]


def test_agent_picks_structure():
    # agent_picks maps each resolved team name to one agent per player
    # in table row order (first-listed agent only). Transcribed from
    # match_page.html's Split map tables.
    m = vlr.parse_match(MATCH_HTML, MATCH_URL)
    assert m.maps[0].agent_picks == {
        "FUT Esports": ["viper", "raze", "skye", "omen", "jett"],
        "Natus Vincere": ["viper", "sage", "waylay", "skye", "omen"],
    }


def test_parse_player_stats_table_multi_agent_rows():
    # Assumption 1's agents list: the "All Maps" aggregate block of
    # match_page.html is real HTML whose .ovw-tables carry multi-agent
    # rows (7 of 10 players swapped agents across the match). Per-map
    # tables in every fixture happen to be single-agent, so this is
    # where the len(agents) > 1 path is locked in — with real markup,
    # not a synthetic row. team_name is resolved positionally (the
    # plan's assumption 3), never from the .ovw-player-tag.
    soup = BeautifulSoup(MATCH_HTML, "lxml")
    all_block = soup.select_one('.vm-stats-game[data-game-id="all"]')
    assert all_block is not None
    tables = all_block.select(".ovw-table")
    assert len(tables) == 2
    fut = vlr._parse_player_stats_table(tables[0], "FUT Esports")
    navi = vlr._parse_player_stats_table(tables[1], "Natus Vincere")
    assert [p.agents for p in fut] == [
        ["viper", "omen"],
        ["skye", "sova"],
        ["raze", "chamber"],
        ["omen", "phoenix"],
        ["jett"],
    ]
    assert [p.agents for p in navi] == [
        ["viper", "vyse"],
        ["sage"],
        ["waylay", "yoru"],
        ["omen"],
        ["skye", "sova"],
    ]
    assert all(p.team_name == "FUT Esports" for p in fut)
    assert all(p.team_name == "Natus Vincere" for p in navi)
    # Swap order is preserved: each multi-agent row lists both agents.
    assert fut[0].agents == ["viper", "omen"]


def test_bo5_fixture_player_stats_populated():
    # A second real fixture (Bo5 grand final, JD Gaming vs TYLOO):
    # every one of the 4 played maps carries 10 player rows (5 per
    # team) with resolved team names and a two-team agent_picks.
    m = vlr.parse_match(BO5_HTML, BO5_URL)
    assert len(m.maps) == 4
    for map_result in m.maps:
        assert len(map_result.player_stats) == 10
        assert set(map_result.agent_picks) == {"JD Gaming", "TYLOO"}
        assert all(len(v) == 5 for v in map_result.agent_picks.values())
        for ps in map_result.player_stats:
            assert ps.player_name
            assert ps.team_name in ("JD Gaming", "TYLOO")
    # Spot-check one transcribed row: Lotus (map 0), table 0 row 0.
    lotus = m.maps[0]
    assert lotus.map_name == "Lotus"
    ps = lotus.player_stats[0]
    assert ps.player_name == "jkuro"
    assert ps.team_name == "JD Gaming"
    assert ps.rating == 1.44
    assert ps.acs == 247.0
    assert ps.kills == 20
    assert ps.deaths == 10
    assert ps.assists == 2
    assert ps.adr == 164.0
    assert ps.kast == 89.0
    assert ps.hs_pct == 46.0
    assert ps.first_kills == 2
    assert ps.first_deaths == 2
    assert ps.agents == ["raze"]


def test_upcoming_fixture_tbd_blocks_do_not_raise_or_leak():
    # The upcoming fixture's TBD placeholder blocks have .ovw-tables
    # with player names but empty stat cells and no agents; _parse_map
    # runs on them (they have a header) before parse_match discards
    # them by map name. The new stats parsing must neither raise nor
    # surface placeholder rows: the parsed Match has zero maps.
    m = vlr.parse_match(UPCOMING_HTML, UPCOMING_URL)
    assert m.maps == []


def test_parse_map_missing_stats_table_degrades_softly():
    # A real map block with no .ovw-table at all (e.g. a future
    # awarded/abandoned map that never rendered stats) must leave
    # player_stats == [] and agent_picks None — the same soft-missing
    # treatment _parse_map already gives duration/winner — not raise.
    html = """
    <div class="vm-stats-game">
      <div class="vm-stats-game-header">
        <div class="team"><div class="score">13</div></div>
        <div class="map"><div><span>Ascent</span></div></div>
        <div class="team mod-right"><div class="score">7</div></div>
      </div>
    </div>
    """
    game_el = BeautifulSoup(html, "lxml").select_one(".vm-stats-game")
    result = vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))
    assert result.player_stats == []
    assert result.agent_picks is None


def test_parse_map_single_stats_table_raises():
    # Exactly one .ovw-table (one team's stats missing) is a broken
    # render outside the expected 0-or-2 shapes: fail loudly rather
    # than silently dropping one team's stats.
    html = """
    <div class="vm-stats-game">
      <div class="vm-stats-game-header">
        <div class="team"><div class="score">13</div></div>
        <div class="map"><div><span>Ascent</span></div></div>
        <div class="team mod-right"><div class="score">7</div></div>
      </div>
      <div class="ovw-table">
        <div class="ovw-row mod-head"><div class="ovw-th"></div></div>
      </div>
    </div>
    """
    game_el = BeautifulSoup(html, "lxml").select_one(".vm-stats-game")
    with pytest.raises(vlr.VlrParseError):
        vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))


def test_parse_player_stats_table_row_without_name_raises():
    # A row inside a present table with no .ovw-player-name is a real
    # structural break (the table exists; its shape is wrong): fail
    # loudly, per the module's fail-loud convention, not silently drop
    # the row.
    html = """
    <div class="ovw-table">
      <div class="ovw-row mod-head"><div class="ovw-th">R</div></div>
      <div class="ovw-row">
        <div class="ovw-cell mod-player">
          <div class="ovw-player"><a href="/player/1/x"><div class="ovw-player-tag">TAG</div></a></div>
        </div>
      </div>
    </div>
    """
    table_el = BeautifulSoup(html, "lxml").select_one(".ovw-table")
    with pytest.raises(vlr.VlrParseError):
        vlr._parse_player_stats_table(table_el, "Team A")


def test_parse_float_percent_and_none():
    # _parse_float strips an optional trailing % (vlr.gg's KAST/HS%
    # columns) and returns None for empty/garbage like _parse_int.
    assert vlr._parse_float("74%") == 74.0
    assert vlr._parse_float("27%") == 27.0
    assert vlr._parse_float("171") == 171.0
    assert vlr._parse_float("1.08") == 1.08
    assert vlr._parse_float("") is None
    assert vlr._parse_float("-") is None
    assert vlr._parse_float("TBD") is None


def test_player_stats_dict_round_trip():
    # A fully-populated PlayerStats (including a multi-agent agents
    # list) must round-trip through to_dict/from_dict unchanged.
    ps = PlayerStats(
        player_name="yetujey",
        team_name="FUT Esports",
        rating=1.08,
        acs=230.0,
        kills=14,
        deaths=15,
        assists=4,
        adr=171.0,
        kast=74.0,
        hs_pct=27.0,
        first_kills=1,
        first_deaths=1,
        agents=["viper", "omen"],
    )
    assert PlayerStats.from_dict(ps.to_dict()) == ps


def test_player_stats_from_dict_defaults_agents_to_empty():
    # A stats dict written before this task (no agents key) must
    # deserialize with agents == [] rather than raising KeyError.
    d = {"player_name": "yetujey", "team_name": "FUT Esports", "rating": 1.08}
    ps = PlayerStats.from_dict(d)
    assert ps.agents == []
    assert ps.acs is None


def test_map_result_dict_round_trip_with_populated_player_stats():
    # A real parsed map with populated player_stats/agent_picks must
    # round-trip exactly — this is what the SQLite cache relies on
    # (test_get_matches_from_event compares cache-round-tripped matches
    # for full equality).
    m = vlr.parse_match(MATCH_HTML, MATCH_URL)
    for map_result in m.maps:
        assert map_result.player_stats  # populated
        assert map_result.agent_picks is not None
        assert MapResult.from_dict(map_result.to_dict()) == map_result


def test_map_result_from_dict_defaults_player_stats_to_empty():
    # A map dict written before this task (no player_stats key) must
    # deserialize with player_stats == [] rather than raising KeyError.
    d = {
        "map_name": "Split",
        "team1_score": 6,
        "team2_score": 13,
        "winner": "Natus Vincere",
        "duration": "59:20",
        "agent_picks": None,
    }
    mr = MapResult.from_dict(d)
    assert mr.player_stats == []
