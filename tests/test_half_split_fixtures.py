"""Half-split fixture suite (roadmap M6): attack/defense half parsing.

M6 extracts each played map's per-team first/second half and
atk/def regulation round counts from the real vlr.gg header markup
(``.vm-stats-game-header .team`` spans). No new fixture files are
needed — every ``match_page*.html`` file used by tasks 003/005/006
already carries the half spans, and their expected values below are
hand-transcribed from those real headers (never guessed):

- match_page.html (FUT Esports vs Natus Vincere, VCT 2026: EMEA Stage 2)
    maps[0] Split 6-13. FUT's header spans are ``mod-ct=4 / mod-t=2``
    (CT first half), NAVI's ``mod-t=8 / mod-ct=5`` (T first half) —
    side order tracks which side each team started on, so half
    assignment comes from span position and the side from the class.
    Regulation-only map (no ``mod-ot`` span). Combined first half
    4+8=12; combined second half 2+5=7 is *truncated* — NAVI hit 13
    mid-half — which the invariant allows (<= 12).
- match_page_single_ot.html (TL Brazil vs EG GC, Game Changers 2026
    Brazil Finals)
    maps[1] Haven 15-13. Both teams reached 12 in regulation, so each
    header carries a third ``mod-ot`` span (TL ``mod-t=5 / mod-ct=7 /
    mod-ot=3``, EG ``mod-ct=7 / mod-t=5 / mod-ot=1``); both regulation
    halves sum to 12. The ``mod-ot`` values are read but NOT stored —
    vlr.gg's header exposes OT only as a combined per-team total, so
    the atk/def fields are regulation-only by design (plan
    assumption), and ``MapResult`` carries no OT-round field.
- match_page_upcoming.html (FUT Esports vs Karmine Corp, VCT 2026:
    EMEA Stage 2)
    The TBD placeholder blocks render half spans with a bare
    ``mod-`` class (no t/ct suffix) and text "0". These must parse to
    "no recognized side" — all-None — not crash and not count as real
    zeros (which would trip the first-half == 12 invariant with 0+0).

The plan (tasks/007-half-split-parser) requires an invariant check in
``MapResult.__post_init__``. Two deviations are implemented
deliberately. First: the plan's literal "combined second half == 12"
contradicts the plan's own transcribed Split values (2+5=7) and the
real fixtures — a second half truncates when a team reaches 13 rounds
mid-half. The enforced invariant is therefore "combined first half ==
12, combined second half <= 12" (see the regression test below).
Second (review finding 1): the plan's "independent of the
finished-map gate" was dropped — the invariant now runs only once the
map is known-finished (scores + winner all set), because a live
in-progress map's header renders partial round counts (e.g. a
mid-first-half 6-3) that legitimately violate it, and vlr.gg only
renders the winner element (``.score.mod-win``) once a map is
complete. The mismatch tests below therefore construct *finished*
maps with valid scorelines so the half-split data is the only
violation.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from scraper import vlr
from scraper.models import IllegalScoreError, MapResult, Team

FIXTURES = Path(__file__).parent / "fixtures"

MATCH_HTML = (FIXTURES / "match_page.html").read_text(encoding="utf-8")
SINGLE_OT_HTML = (FIXTURES / "match_page_single_ot.html").read_text(encoding="utf-8")
UPCOMING_HTML = (FIXTURES / "match_page_upcoming.html").read_text(encoding="utf-8")
BLOWOUT_MULTI_OT_HTML = (FIXTURES / "match_page_blowout_multi_ot.html").read_text(
    encoding="utf-8"
)
CLOSE_HTML = (FIXTURES / "match_page_close.html").read_text(encoding="utf-8")
MULTI_OT_HTML = (FIXTURES / "match_page_multi_ot.html").read_text(encoding="utf-8")
BO1_HTML = (FIXTURES / "match_page_bo1.html").read_text(encoding="utf-8")
BO5_HTML = (FIXTURES / "match_page_bo5.html").read_text(encoding="utf-8")

MATCH_URL = "https://www.vlr.gg/712803/fut-esports-vs-natus-vincere-vct-2026-emea-stage-2-w1"
SINGLE_OT_URL = (
    "https://www.vlr.gg/731773/team-liquid-brazil-vs-evil-geniuses-gc-"
    "game-changers-2026-brazil-finals-ubsf"
)
UPCOMING_URL = "https://www.vlr.gg/731400/fut-esports-vs-karmine-corp-vct-2026-emea-stage-2-ubf"
BLOWOUT_MULTI_OT_URL = (
    "https://www.vlr.gg/730543/kiwoom-drx-vs-sharper-esports-"
    "vct-2026-pacific-stage-2-ubsf"
)
CLOSE_URL = (
    "https://www.vlr.gg/742478/gen-g-vs-nongshim-redforce-"
    "vct-2026-pacific-stage-2-ubsf"
)
MULTI_OT_URL = (
    "https://www.vlr.gg/712754/3-vs-sad-gc-game-changers-2026-"
    "north-america-stage-2-swiss-stage-r1"
)
BO1_URL = (
    "https://www.vlr.gg/706131/mexico-vs-dominican-republic-"
    "esports-nations-cup-2026-north-america-qualifier-ubsf"
)
BO5_URL = (
    "https://www.vlr.gg/724645/jd-gaming-vs-tyloo-"
    "vct-2026-china-stage-2-gf"
)


def _assert_half_split(
    map_result,
    team1_first_half_rounds,
    team1_second_half_rounds,
    team1_atk_rounds,
    team1_def_rounds,
    team2_first_half_rounds,
    team2_second_half_rounds,
    team2_atk_rounds,
    team2_def_rounds,
):
    """Assert one parsed map's eight half-split fields match expectations exactly.

    Shared assertion for every real-fixture case in this suite: checks
    all eight of ``MapResult``'s half-split fields (the four per team)
    against hand-transcribed values from the fixture's
    ``.vm-stats-game-header`` spans. Also asserts the cross-team
    properties that hold on every real header — the combined
    first-half count is exactly 12 and the combined second-half count
    never exceeds 12 — plus the regulation-only property that
    ``atk + def == first + second`` per team — so a transcription
    error in the expected values fails loudly instead of passing
    silently.

    Args:
        map_result: The parsed ``MapResult`` under test.
        team1_first_half_rounds: Expected ``team1_first_half_rounds``.
        team1_second_half_rounds: Expected ``team1_second_half_rounds``.
        team1_atk_rounds: Expected ``team1_atk_rounds``.
        team1_def_rounds: Expected ``team1_def_rounds``.
        team2_first_half_rounds: Expected ``team2_first_half_rounds``.
        team2_second_half_rounds: Expected ``team2_second_half_rounds``.
        team2_atk_rounds: Expected ``team2_atk_rounds``.
        team2_def_rounds: Expected ``team2_def_rounds``.

    Returns:
        Nothing; raises on any mismatch.

    Raises:
        AssertionError: If any of the eight field assertions, the two
            cross-team half-sum assertions, or the two regulation-only
            atk/def assertions fail.
    """
    assert map_result.team1_first_half_rounds == team1_first_half_rounds
    assert map_result.team1_second_half_rounds == team1_second_half_rounds
    assert map_result.team1_atk_rounds == team1_atk_rounds
    assert map_result.team1_def_rounds == team1_def_rounds
    assert map_result.team2_first_half_rounds == team2_first_half_rounds
    assert map_result.team2_second_half_rounds == team2_second_half_rounds
    assert map_result.team2_atk_rounds == team2_atk_rounds
    assert map_result.team2_def_rounds == team2_def_rounds
    # Cross-team invariants every real header satisfies (see
    # MapResult.__post_init__): first half always exactly 12, second
    # half truncated (<= 12) but never more.
    assert team1_first_half_rounds + team2_first_half_rounds == 12
    assert team1_second_half_rounds + team2_second_half_rounds <= 12
    # Regulation-only atk/def totals: atk + def == first + second.
    assert team1_atk_rounds + team1_def_rounds == (
        team1_first_half_rounds + team1_second_half_rounds
    )
    assert team2_atk_rounds + team2_def_rounds == (
        team2_first_half_rounds + team2_second_half_rounds
    )


def test_regulation_map_split_half_splits():
    # match_page.html maps[0] Split (FUT Esports 6-13 Natus Vincere),
    # a regulation-only map (no mod-ot span). FUT's header spans are
    # mod-ct=4 / mod-t=2 — CT first half, so first_half=4 and the atk
    # count comes from the mod-t second-half span (2). NAVI's are
    # mod-t=8 / mod-ct=5 — T first half, so first_half=8. Combined
    # first half 4+8=12; combined second half 2+5=7 is truncated
    # (NAVI hit 13 mid-half), which the invariant allows (<= 12).
    m = vlr.parse_match(MATCH_HTML, MATCH_URL)
    assert m.status == "completed"
    split = m.maps[0]
    assert split.map_name == "Split"
    _assert_half_split(
        split,
        team1_first_half_rounds=4,
        team1_second_half_rounds=2,
        team1_atk_rounds=2,
        team1_def_rounds=4,
        team2_first_half_rounds=8,
        team2_second_half_rounds=5,
        team2_atk_rounds=8,
        team2_def_rounds=5,
    )


def test_overtime_map_haven_half_splits():
    # match_page_single_ot.html maps[1] Haven (Team Liquid Brazil
    # 15-13 EG GC). Both teams reached 12 in regulation, so both
    # regulation halves are complete (sum 12) and each header carries
    # a third mod-ot span (TL 3, EG 1). TL's spans are mod-t=5 /
    # mod-ct=7 / mod-ot=3, EG's mod-ct=7 / mod-t=5 / mod-ot=1.
    m = vlr.parse_match(SINGLE_OT_HTML, SINGLE_OT_URL)
    assert m.status == "completed"
    haven = next(x for x in m.maps if x.map_name == "Haven")
    _assert_half_split(
        haven,
        team1_first_half_rounds=5,
        team1_second_half_rounds=7,
        team1_atk_rounds=5,
        team1_def_rounds=7,
        team2_first_half_rounds=7,
        team2_second_half_rounds=5,
        team2_atk_rounds=5,
        team2_def_rounds=7,
    )
    # Scope boundary (plan assumption): the header's mod-ot spans are
    # read but not stored — MapResult carries no OT-round field, and
    # atk/def stay regulation-only (atk + def == first + second,
    # asserted by _assert_half_split above).
    assert not hasattr(haven, "team1_ot_rounds")
    assert not hasattr(haven, "team2_ot_rounds")


def test_upcoming_tbd_placeholder_half_spans_parse_to_none():
    # The upcoming fixture's TBD placeholder blocks render half spans
    # with a bare "mod-" class (no t/ct suffix) and text "0". These
    # must parse to "no recognized side" — all-None — rather than
    # crash or count as real zeros (which would trip the first-half
    # == 12 invariant with 0+0=0). parse_match discards TBD blocks by
    # name, so this exercises _parse_half_split directly on the
    # placeholder's team divs, plus a MapResult built from the
    # all-None values constructing cleanly.
    soup = BeautifulSoup(UPCOMING_HTML, "lxml")
    header = soup.select_one(".vm-stats-game .vm-stats-game-header")
    team_divs = header.select(".team")
    assert len(team_divs) == 2
    assert vlr._parse_half_split(team_divs[0]) == (None, None, None, None)
    assert vlr._parse_half_split(team_divs[1]) == (None, None, None, None)
    MapResult(
        map_name="TBD",
        team1_score=None,
        team2_score=None,
        winner=None,
        team1_first_half_rounds=None,
        team1_second_half_rounds=None,
        team1_atk_rounds=None,
        team1_def_rounds=None,
        team2_first_half_rounds=None,
        team2_second_half_rounds=None,
        team2_atk_rounds=None,
        team2_def_rounds=None,
    )


def test_first_half_sum_mismatch_raises_illegal_score_error():
    # Synthetic broken invariant — no legitimate vlr.gg page can
    # render it: a combined first half of 7+7=14 is impossible. The
    # map must be finished (scores + winner set) for the invariant to
    # run — it never fires on live/unfinished maps (review finding 1:
    # a live map's partial counts legitimately violate it). The 13-10
    # scoreline itself is valid, so the half-split data is the only
    # violation, and the half-split check runs before the score
    # checks, so the message names the first-half invariant.
    with pytest.raises(IllegalScoreError) as excinfo:
        MapResult(
            map_name="Ascent",
            team1_score=13,
            team2_score=10,
            winner="Team A",
            team1_first_half_rounds=7,
            team2_first_half_rounds=7,
        )
    message = str(excinfo.value)
    assert "Ascent" in message
    assert "combined first-half" in message


def test_second_half_over_12_raises_illegal_score_error():
    # A combined second half of 14 is impossible: a second half can be
    # truncated (fewer than 12 rounds, when a team reaches 13
    # mid-half) but never exceeds 12 rounds. Finished-map scoreline
    # (13-10) is valid, so the half-split data is the only violation.
    with pytest.raises(IllegalScoreError) as excinfo:
        MapResult(
            map_name="Split",
            team1_score=13,
            team2_score=10,
            winner="Team A",
            team1_second_half_rounds=8,
            team2_second_half_rounds=6,
        )
    assert "combined second-half" in str(excinfo.value)


def test_live_map_partial_half_data_does_not_raise():
    # Review finding 1 regression: a live match's in-progress map
    # renders partial round counts (mid-first-half 6-3, combined 9 !=
    # 12) with no winner element yet (vlr.gg only marks .score.mod-win
    # once a map is complete). The half-split invariant must not fire
    # on unfinished maps — this MapResult must construct cleanly with
    # its partial half data intact, not raise IllegalScoreError.
    result = MapResult(
        map_name="Ascent",
        team1_score=6,
        team2_score=3,
        winner=None,
        team1_first_half_rounds=6,
        team2_first_half_rounds=3,
        team1_atk_rounds=2,
        team1_def_rounds=4,
        team2_atk_rounds=1,
        team2_def_rounds=2,
    )
    assert result.team1_first_half_rounds == 6
    assert result.team2_first_half_rounds == 3


LIVE_MAP_HTML = """
<div class="vm-stats-game">
<div class="vm-stats-game-header">
<div class="team">
<div class="score">6</div>
<div>
<span class="mod-t">6</span>
</div>
</div>
<div class="map">
<div><span>Ascent</span></div>
</div>
<div class="team mod-right">
<div class="score">3</div>
<div>
<span class="mod-ct">3</span>
</div>
</div>
</div>
</div>
"""


def test_parse_map_live_in_progress_map_parses_partial_half_data():
    # End-to-end regression for both review findings: a live match's
    # in-progress map renders partial half spans (team1 mod-t=6,
    # team2 mod-ct=3 — combined first half 9, no second half) and no
    # .score.mod-win element. _parse_map must (finding 1) parse it
    # without raising — the half-split invariant only runs on
    # finished maps — and (finding 2) report None, not a fabricated
    # 0, for the side whose span never parsed (team1's def_rounds,
    # team2's atk_rounds).
    game_el = BeautifulSoup(LIVE_MAP_HTML, "lxml").select_one(".vm-stats-game")
    result = vlr._parse_map(game_el, Team(name="Team A"), Team(name="Team B"))
    assert result.map_name == "Ascent"
    assert result.team1_score == 6
    assert result.team2_score == 3
    assert result.winner is None
    assert result.team1_first_half_rounds == 6
    assert result.team1_second_half_rounds is None
    assert result.team1_atk_rounds == 6
    assert result.team1_def_rounds is None
    assert result.team2_first_half_rounds == 3
    assert result.team2_second_half_rounds is None
    assert result.team2_atk_rounds is None
    assert result.team2_def_rounds == 3


def test_parse_half_split_partial_side_reports_none_not_zero():
    # Review finding 2 regression at the unit level: when exactly one
    # of the mod-t/mod-ct spans is recognized (e.g. a live in-progress
    # half rendering only the currently active side), the side that
    # never parsed must report None — 0 is indistinguishable from
    # "genuinely won zero rounds on that side" even though the value
    # was never observed.
    team_div = BeautifulSoup(
        '<div class="team"><span class="mod-t">6</span></div>', "lxml"
    ).select_one(".team")
    assert vlr._parse_half_split(team_div) == (6, None, 6, None)


def test_truncated_second_half_does_not_raise():
    # Deviation from plan#2's literal "combined second half == 12":
    # the real Split fixture's second half sums to 7 (2+5) because
    # NAVI hit 13 rounds mid-half, so the enforced invariant is
    # "combined second half <= 12", not "== 12". A truncated second
    # half (a finished map whose game ended early) must construct
    # cleanly — this locks in the corrected invariant against the
    # plan's own transcribed fixture values.
    MapResult(
        map_name="Split",
        team1_score=6,
        team2_score=13,
        winner="Natus Vincere",
        team1_first_half_rounds=4,
        team1_second_half_rounds=2,
        team1_atk_rounds=2,
        team1_def_rounds=4,
        team2_first_half_rounds=8,
        team2_second_half_rounds=5,
        team2_atk_rounds=8,
        team2_def_rounds=5,
    )


def test_all_fixture_headers_satisfy_half_invariants():
    # Sweep every real completed-map header across all 8 fixtures
    # (mirrors the plan's all-real-headers verification): any parsed
    # map with both teams' half data must have a combined first half
    # of exactly 12 and a combined second half of at most 12
    # (truncation allowed). A violation here — parse_match raising
    # IllegalScoreError — means either the invariant is wrong or a
    # fixture regressed. 18 completed maps across the 7 non-upcoming
    # fixtures; the upcoming fixture's 3 TBD blocks parse to all-None
    # and are excluded by the None guard.
    cases = [
        (MATCH_HTML, MATCH_URL),
        (BLOWOUT_MULTI_OT_HTML, BLOWOUT_MULTI_OT_URL),
        (CLOSE_HTML, CLOSE_URL),
        (MULTI_OT_HTML, MULTI_OT_URL),
        (SINGLE_OT_HTML, SINGLE_OT_URL),
        (BO1_HTML, BO1_URL),
        (BO5_HTML, BO5_URL),
    ]
    checked = 0
    for html, url in cases:
        m = vlr.parse_match(html, url)
        for map_result in m.maps:
            if (
                map_result.team1_first_half_rounds is None
                or map_result.team2_first_half_rounds is None
            ):
                continue
            checked += 1
            assert (
                map_result.team1_first_half_rounds
                + map_result.team2_first_half_rounds
                == 12
            )
            assert (
                map_result.team1_second_half_rounds
                + map_result.team2_second_half_rounds
                <= 12
            )
    assert checked == 18
