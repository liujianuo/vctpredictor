"""Score-parser fixture suite (roadmap M2): real vlr.gg match pages.

Each test loads a real, fetched vlr.gg match page fixture and asserts
exact map scores, the winner's team name, and the derived OT flag
(``min(team1_score, team2_score) >= 12`` — the same rule
``MapResult.__post_init__`` applies internally, per plan assumption 1
this is a test-only computation, not a new persisted field).

The plan (tasks/003-score-fixture-suite) calls for five target
scorelines: 13-2 (regulation blowout), 13-11 (regulation close),
15-13 (single OT), 16-14 and 19-17 (multi-OT). Per plan assumption 4,
one real Bo3 (KIWOOM DRX vs Sharper Esports) contains *two* of the
targets — a 13-2 (Haven) and a 16-14 (Ascent) — so it serves both
tests from a single fixture file. Four fixture files cover all five
targets:

- match_page_blowout_multi_ot.html
    https://www.vlr.gg/730543/kiwoom-drx-vs-sharper-esports-vct-2026-pacific-stage-2-ubsf
    VCT 2026: Pacific Stage 2, Bo3, completed (KIWOOM DRX 2-1 Sharper Esports)
      maps[0] Haven   13-2  winner KIWOOM DRX     <- 13-2 target
      maps[1] Ascent  14-16 winner Sharper Esports <- 16-14 target
      maps[2] Sunset  13-9  winner KIWOOM DRX
- match_page_close.html
    https://www.vlr.gg/742478/gen-g-vs-nongshim-redforce-vct-2026-pacific-stage-2-ubsf
    VCT 2026: Pacific Stage 2, Bo3, completed (Gen.G 1-2 Nongshim RedForce)
      maps[0] Ascent 8-13   winner Nongshim RedForce
      maps[1] Summit 13-9   winner Gen.G
      maps[2] Sunset 11-13  winner Nongshim RedForce  <- 13-11 target
- match_page_single_ot.html
    https://www.vlr.gg/731773/team-liquid-brazil-vs-evil-geniuses-gc-game-changers-2026-brazil-finals-ubsf
    Game Changers 2026: Brazil Finals, Bo3, completed (TL Brazil 2-0 EG GC)
      maps[0] Breeze 13-4   winner Team Liquid Brazil
      maps[1] Haven  15-13  winner Team Liquid Brazil  <- 15-13 target
- match_page_multi_ot.html
    https://www.vlr.gg/712754/3-vs-sad-gc-game-changers-2026-north-america-stage-2-swiss-stage-r1
    Game Changers 2026: North America Stage 2, Bo3, completed (:3 1-2 SaD GC)
      maps[0] Breeze 13-15  winner SaD GC
      maps[1] Haven  13-10  winner :3
      maps[2] Split  17-19  winner SaD GC             <- 19-17 target

All fixtures were fetched from live vlr.gg pages (not synthesized), so
these tests double as a regression check that M1's score-validity
assertions accept genuine vlr.gg data: ``parse_match`` parses *every*
map on a page, so any illegal scoreline in any fixture would raise
``IllegalScoreError`` (wrapped as ``VlrParseError``) and fail the test.
"""

from pathlib import Path

import pytest

from scraper import vlr

FIXTURES = Path(__file__).parent / "fixtures"

BLOWOUT_MULTI_OT_HTML = (FIXTURES / "match_page_blowout_multi_ot.html").read_text(
    encoding="utf-8"
)
CLOSE_HTML = (FIXTURES / "match_page_close.html").read_text(encoding="utf-8")
SINGLE_OT_HTML = (FIXTURES / "match_page_single_ot.html").read_text(encoding="utf-8")
MULTI_OT_HTML = (FIXTURES / "match_page_multi_ot.html").read_text(encoding="utf-8")

BLOWOUT_MULTI_OT_URL = (
    "https://www.vlr.gg/730543/kiwoom-drx-vs-sharper-esports-"
    "vct-2026-pacific-stage-2-ubsf"
)
CLOSE_URL = (
    "https://www.vlr.gg/742478/gen-g-vs-nongshim-redforce-"
    "vct-2026-pacific-stage-2-ubsf"
)
SINGLE_OT_URL = (
    "https://www.vlr.gg/731773/team-liquid-brazil-vs-evil-geniuses-gc-"
    "game-changers-2026-brazil-finals-ubsf"
)
MULTI_OT_URL = (
    "https://www.vlr.gg/712754/3-vs-sad-gc-game-changers-2026-"
    "north-america-stage-2-swiss-stage-r1"
)


def _find_map_by_score(match, target_scores):
    """Locate the map in a parsed match whose final score matches ``target_scores``.

    Matches the target score as an *unordered* pair, e.g. ``(13, 2)``
    matches both a 13-2 and a 2-13 map, so the test then asserts the
    exact per-side values and winner explicitly. Locating by score
    rather than by the map's index within the page (which is recorded
    in the module docstring) keeps the tests robust to cosmetic
    fixture edits.

    Args:
        match: The parsed :class:`scraper.models.Match` produced from
            a fixture, exposing ``maps`` (list of ``MapResult``) and
            ``team1``/``team2`` (for the failure message).
        target_scores: A two-element iterable of ints, the final score
            as an unordered pair, e.g. ``(13, 2)``.

    Returns:
        The ``MapResult`` whose ``team1_score``/``team2_score`` equal
        ``target_scores`` as an unordered pair.

    Raises:
        pytest.fail.Exception: If no map matches (always when a map's
            score is unverifiable — e.g. a forfeit ``"-"`` parses to
            ``None`` and is skipped — or when the scoreline is simply
            absent from the fixture). The failure message lists the
            match-up and every found map scoreline to aid debugging.
    """
    for map_result in match.maps:
        if map_result.team1_score is None or map_result.team2_score is None:
            continue
        if {map_result.team1_score, map_result.team2_score} == set(target_scores):
            return map_result
    pytest.fail(
        f"no map with score {target_scores[0]}-{target_scores[1]} "
        f"in fixture for {match.team1.name} vs {match.team2.name}; "
        f"found: {[f'{m.team1_score}-{m.team2_score}' for m in match.maps]}"
    )


def _assert_scoreline(map_result, team1_score, team2_score, winner, is_overtime):
    """Assert one parsed map matches an expected scoreline exactly.

    Shared assertion for every target scoreline in this suite: checks
    the exact per-side scores, the winner's team name, and the derived
    OT flag ``min(score1, score2) >= 12`` (the same rule
    ``MapResult.__post_init__`` applies internally; per plan assumption
    1 this is a test-only computation, not a persisted field).

    Args:
        map_result: The parsed ``MapResult`` under test.
        team1_score: Expected ``team1_score`` of ``map_result``.
        team2_score: Expected ``team2_score`` of ``map_result``.
        winner: Expected ``winner`` team name of ``map_result``.
        is_overtime: Expected derived OT flag; ``True`` when both teams
            reached at least 12 rounds.

    Returns:
        Nothing; raises on any mismatch.

    Raises:
        AssertionError: If any of the four assertions fail.
    """
    assert map_result.team1_score == team1_score
    assert map_result.team2_score == team2_score
    assert map_result.winner == winner
    assert (min(team1_score, team2_score) >= 12) is is_overtime


def test_real_fixture_13_2_regulation_blowout():
    # Target 13-2: Haven, maps[0] of the shared blowout/multi-OT
    # fixture. KIWOOM DRX beat Sharper Esports 13-2 on the first map
    # of a Bo3 they went on to win 2-1. A regulation win is not OT.
    m = vlr.parse_match(BLOWOUT_MULTI_OT_HTML, BLOWOUT_MULTI_OT_URL)
    assert m.status == "completed"
    map_result = _find_map_by_score(m, (13, 2))
    assert map_result.map_name == "Haven"
    _assert_scoreline(map_result, team1_score=13, team2_score=2,
                      winner="KIWOOM DRX", is_overtime=False)


def test_real_fixture_13_11_regulation_close():
    # Target 13-11: Sunset, maps[2] of the close fixture. Nongshim
    # RedForce took the decider 13-11 to win the Bo3 2-1 over Gen.G.
    # 11 rounds is below the 12-round OT threshold, so not OT.
    m = vlr.parse_match(CLOSE_HTML, CLOSE_URL)
    assert m.status == "completed"
    map_result = _find_map_by_score(m, (13, 11))
    assert map_result.map_name == "Sunset"
    _assert_scoreline(map_result, team1_score=11, team2_score=13,
                      winner="Nongshim RedForce", is_overtime=False)


def test_real_fixture_15_13_single_overtime():
    # Target 15-13: Haven, maps[1] of the single-OT fixture. Team
    # Liquid Brazil won the map (and the 2-0 sweep) 15-13; both teams
    # reached 12+, so this is a single-overtime scoreline.
    m = vlr.parse_match(SINGLE_OT_HTML, SINGLE_OT_URL)
    assert m.status == "completed"
    map_result = _find_map_by_score(m, (15, 13))
    assert map_result.map_name == "Haven"
    _assert_scoreline(map_result, team1_score=15, team2_score=13,
                      winner="Team Liquid Brazil", is_overtime=True)


def test_real_fixture_16_14_multi_overtime():
    # Target 16-14: Ascent, maps[1] of the shared blowout/multi-OT
    # fixture. Sharper Esports won the map 16-14; both teams reached
    # 12+, so this is a multi-overtime scoreline.
    m = vlr.parse_match(BLOWOUT_MULTI_OT_HTML, BLOWOUT_MULTI_OT_URL)
    assert m.status == "completed"
    map_result = _find_map_by_score(m, (16, 14))
    assert map_result.map_name == "Ascent"
    _assert_scoreline(map_result, team1_score=14, team2_score=16,
                      winner="Sharper Esports", is_overtime=True)


def test_real_fixture_19_17_multi_overtime():
    # Target 19-17: Split, maps[2] of the multi-OT fixture. SaD GC won
    # the decider 19-17 to take the Bo3 2-1 over ":3"; both teams
    # reached 12+, so this is a deep multi-overtime scoreline.
    m = vlr.parse_match(MULTI_OT_HTML, MULTI_OT_URL)
    assert m.status == "completed"
    map_result = _find_map_by_score(m, (19, 17))
    assert map_result.map_name == "Split"
    _assert_scoreline(map_result, team1_score=17, team2_score=19,
                      winner="SaD GC", is_overtime=True)
