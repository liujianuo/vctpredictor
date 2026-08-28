"""Tests for scraper.models — pure dataclass models, no scraping."""

import pytest

from scraper.models import IllegalScoreError, MapResult

# --------------------------------------------------------------------------
# MapResult score validity assertions
# --------------------------------------------------------------------------


def test_valid_regulation_score_does_not_raise():
    # 13-2 is a legal regulation scoreline (winner >= 13, no OT).
    MapResult(map_name="Ascent", team1_score=13, team2_score=2, winner="Team A")


def test_valid_single_overtime_score_does_not_raise():
    # 14-12 is a legal single-OT scoreline (winner >= 13, both >= 12,
    # margin of 2).
    MapResult(map_name="Split", team1_score=14, team2_score=12, winner="Team A")


def test_valid_multi_overtime_score_does_not_raise():
    # 16-14 is a legal multi-OT scoreline (winner >= 13, both >= 12,
    # margin of 2).
    MapResult(map_name="Sunset", team1_score=16, team2_score=14, winner="Team B")


def test_winner_score_below_13_raises():
    # 12-10 with a declared winner is illegal: a winner must reach 13.
    with pytest.raises(ValueError):
        MapResult(map_name="Ascent", team1_score=12, team2_score=10, winner="Team A")


def test_illegal_score_raises_illegal_score_error():
    # Illegal scorelines raise IllegalScoreError (a ValueError
    # subclass), so callers can distinguish score-validity failures
    # from unrelated ValueErrors (e.g. a corrupt cache row's bad date
    # field) while still matching ``except ValueError`` handlers.
    with pytest.raises(IllegalScoreError):
        MapResult(map_name="Ascent", team1_score=13, team2_score=12, winner="Team A")


def test_overtime_with_margin_below_2_raises():
    # 13-12 cannot be a final scoreline: both teams reached 12
    # (overtime), so the winner needs a margin of at least 2.
    with pytest.raises(ValueError):
        MapResult(map_name="Ascent", team1_score=13, team2_score=12, winner="Team A")


def test_forfeit_style_scoreline_raises():
    # A forfeited/defaulted map (winner declared, scores like 0-0 or
    # 2-0) has no legal winner_score >= 13, so it raises. This is
    # accepted per plan#1, which specified no carve-out for forfeits:
    # if vlr.gg ever renders one, it surfaces loudly rather than
    # caching a wrong label.
    with pytest.raises(ValueError):
        MapResult(map_name="Ascent", team1_score=0, team2_score=0, winner="Team A")
    with pytest.raises(ValueError):
        MapResult(map_name="Ascent", team1_score=2, team2_score=0, winner="Team A")


def test_error_message_includes_map_name_and_scores():
    with pytest.raises(ValueError) as excinfo:
        MapResult(map_name="Ascent", team1_score=13, team2_score=12, winner="Team A")
    message = str(excinfo.value)
    assert "Ascent" in message
    assert "13" in message
    assert "12" in message


def test_unfinished_map_with_scores_does_not_raise():
    # A live/unfinished map may show scores with no winner yet; that is
    # not a final label, so no validation applies.
    MapResult(map_name="Ascent", team1_score=6, team2_score=9, winner=None)


def test_upcoming_map_with_no_fields_does_not_raise():
    # The upcoming-map default (all three fields None) must construct
    # cleanly.
    MapResult(map_name="TBD", team1_score=None, team2_score=None, winner=None)


def test_partially_none_scores_do_not_raise():
    # A score may be unparseable (None) while the other is known; no
    # validation applies until all three fields are present.
    MapResult(map_name="Ascent", team1_score=None, team2_score=8, winner="Team A")
    MapResult(map_name="Ascent", team1_score=8, team2_score=None, winner="Team A")
