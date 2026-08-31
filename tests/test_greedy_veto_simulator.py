"""Tests for the rule-based greedy veto simulator (M25).

Covers the exact Bo1/Bo3/Bo5 output shapes with strict step-index turn
alternation, an independent per-step argmin/argmax cross-check against
direct ``features.map_win_rate.team_map_win_rate`` calls, determinism
(no set/dict iteration-order reliance), alphabetical tie-breaking,
map-pool size / unknown-format ``ValueError``s, the config-era
``map_pool=None`` resolution path (plus the no-era-covers ConfigError
case), caller-supplied map-name normalization, and a leakage test
proving a future map dated on/after the query date never reaches the
win-rate estimates (and flips the choice when legitimately included
later).
"""

import pandas as pd
import pytest

from features import map_win_rate
from models.greedy_veto_simulator import (
    ACTION_SEQUENCES,
    SimulatedVetoAction,
    simulate_veto,
)
from utils import config

# The as-of cutoff for the synthetic leagues: one hour after the last
# fixture match, so every fixture row is strictly before it.
QUERY_DATE = "2026-01-06T00:00:00"

# The 7-map pool all three ACTION_SEQUENCES walk, matching
# config.json's 2026-abyss era (Abyss, Ascent, Haven, Lotus, Split,
# Summit, Sunset) in ascending name order.
POOL = ("Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset")

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Wraps ``pd.DataFrame`` so every test fixture produces the same
    column order/dtypes regardless of which subset of columns a given
    fixture actually needs.

    Args:
        rows: A list of dicts, one per match; each must carry the keys
            in :data:`_MATCHES_COLS` (extra keys are ignored by the
            explicit ``columns=`` ordering).

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MATCHES_COLS`
        columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_MATCHES_COLS)


def _maps_df(rows):
    """Build a maps table with the fixed M8 column set.

    Mirrors :func:`_matches_df` for the maps side so every fixture
    shares one column order/dtype.

    Args:
        rows: A list of dicts, one per map; each must carry the keys in
            :data:`_MAPS_COLS`.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_MAPS_COLS` columns.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return pd.DataFrame(rows, columns=_MAPS_COLS)


def _build(match_rows, map_rows):
    """Build a ``(matches_df, maps_df)`` pair from parallel row lists.

    The shared constructor behind every synthetic league fixture so the
    column-order/dtype convention lives in one place. Each entry of
    ``match_rows``/``map_rows`` must carry the keys of
    :data:`_MATCHES_COLS`/:data:`_MAPS_COLS` respectively.

    Args:
        match_rows: A list of match dicts, one per completed match.
        map_rows: A list of map dicts, one per finished map.

    Returns:
        A ``(matches_df, maps_df)`` tuple built by :func:`_matches_df`
        / :func:`_maps_df`.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is
        if a fixture row is malformed).
    """
    return _matches_df(match_rows), _maps_df(map_rows)


def _add(match_rows, map_rows, mid, date, team1_id, team2_id, map_name, t1s, t2s):
    """Append one completed match and its finished map to the row lists.

    The single row-writing helper for the synthetic league fixtures.
    The map's ``winner`` is derived from the scores (never a
    display-name string), matching the existing test fixtures'
    convention; a null-score row is never produced because ``winner``
    is always set.

    Args:
        match_rows: The mutable match-row list to append to.
        map_rows: The mutable map-row list to append to.
        mid: The shared ``match_id`` for the new match and map.
        date: The match's ISO date string.
        team1_id: The match's team1 stable id.
        team2_id: The match's team2 stable id.
        map_name: The finished map's name.
        t1s: Rounds team1 won (the map's ``team1_score``).
        t2s: Rounds team2 won (the map's ``team2_score``).

    Returns:
        Nothing (appends in place).

    Raises:
        Nothing.
    """
    match_rows.append(
        {
            "match_id": mid,
            "date": date,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "status": "completed",
        }
    )
    map_rows.append(
        {
            "match_id": mid,
            "map_index": 0,
            "map_name": map_name,
            "team1_score": t1s,
            "team2_score": t2s,
            "winner": team1_id if t1s > t2s else team2_id,
        }
    )


def _stamp(i):
    """Return the ISO timestamp ``i`` hours after a fixed 2026-01-01 base.

    The shared clock for the synthetic leagues: every fixture's dates
    are one hour apart so chronological ordering is unambiguous and the
    as-of query date is strictly after everything.

    Args:
        i: The hour offset from the base.

    Returns:
        An ISO-8601 datetime string.

    Raises:
        Nothing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00")
    return (base + pd.Timedelta(hours=i)).isoformat()


def _league_tables():
    """Build the 16-map greedy-choice league.

    Team ``A`` is 0W-4L on Split (its weakest map) and 4W-0L on Haven
    (its strongest), with an overall 0.5 record; team ``B`` mirrors
    that with Sunset (0W-4L, weakest) and Ascent (4W-0L, strongest).
    Every opponent id is unique and plays only once, and all rows are
    dated before :data:`QUERY_DATE`. With the default shrinkage
    ``k = 10`` both teams' prior is 0.5, so:

    - ``A``: Split ``5/14 ~ 0.357`` (min), Haven ``9/14 ~ 0.643``
      (max), the other five maps ``0.5`` (no history);
    - ``B``: Sunset ``5/14 ~ 0.357`` (min), Ascent ``9/14 ~ 0.643``
      (max), the other five maps ``0.5`` (no history).

    The greedy walk is therefore fully deterministic for every format:
    the Bo3/Bo5 first ban/pick pairs are Split/Sunset and Haven/Ascent,
    and the late steps fall into mean ties resolved alphabetically.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 16 matches and 16 maps
        built by :func:`_build`.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    for i in range(4):
        _add(match_rows, map_rows, f"a_split_{i}", _stamp(i), "A", f"op{i}", "Split", 8, 13)
    for i in range(4):
        _add(match_rows, map_rows, f"a_haven_{i}", _stamp(4 + i), "A", f"op{10 + i}", "Haven", 13, 8)
    for i in range(4):
        _add(match_rows, map_rows, f"b_sunset_{i}", _stamp(8 + i), "B", f"op{20 + i}", "Sunset", 8, 13)
    for i in range(4):
        _add(match_rows, map_rows, f"b_ascent_{i}", _stamp(12 + i), "B", f"op{30 + i}", "Ascent", 13, 8)
    return _build(match_rows, map_rows)


def _leak_tables():
    """Build the leakage-check league with a future Ascent win for ``A``.

    Team ``A``'s strictly-before-:data:`QUERY_DATE` history is 2W-0L on
    Haven, 0W-2L on Ascent and 0W-1L on Split (overall 2W-3L, prior
    0.4). The final row — ``A`` beating an opponent on Ascent — is
    dated *exactly at* the no-leak query timestamp (``_stamp(6)``), so
    the strict ``<`` boundary excludes it. With ``k = 2`` the exclusion
    is consequential: without the future row ``A``'s Ascent mean is
    ``0.8/4 = 0.2`` (its argmin ban), while the Split mean is
    ``0.8/3 ~ 0.267``; if the future row leaks in (query moved to
    ``_stamp(7)``) the prior rises to 0.5 and Ascent's mean to 0.4
    while Split's stays ``1/3 ~ 0.333``, flipping ``A``'s first ban
    from Ascent to Split.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 6 matches and 6 maps built
        by :func:`_build`; the row with ``match_id == "l6"`` is the
        future Ascent win.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "l1", _stamp(0), "A", "op1", "Haven", 13, 8)
    _add(match_rows, map_rows, "l2", _stamp(1), "A", "op2", "Haven", 13, 8)
    _add(match_rows, map_rows, "l3", _stamp(2), "A", "op3", "Ascent", 8, 13)
    _add(match_rows, map_rows, "l4", _stamp(3), "A", "op4", "Ascent", 8, 13)
    _add(match_rows, map_rows, "l5", _stamp(4), "A", "op5", "Split", 8, 13)
    _add(match_rows, map_rows, "l6", _stamp(6), "A", "op6", "Ascent", 13, 8)
    return _build(match_rows, map_rows)


# --------------------------------------------------------------------------
# plan#5a: Bo3 shape end-to-end + per-step independent argmin/argmax
# --------------------------------------------------------------------------


def test_bo3_shape_and_independent_cross_check():
    # A full Bo3 run on the 16-map league must produce the exact
    # 7-step sequence (ban, ban, pick, pick, ban, ban, decider) with
    # strict step-index alternation (A, B, A, B, A, B, None). Each
    # chosen map is then independently re-derived: replay the walk and,
    # for every non-decider step, recompute the acting team's shrunk
    # mean for *every remaining candidate* with a direct
    # team_map_win_rate call and confirm the simulator chose the
    # argmin (ban) / argmax (pick) with the documented tie-break.
    matches_df, maps_df = _league_tables()
    out = simulate_veto(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert [(a.step_index, a.team, a.action, a.map_name) for a in out] == [
        (0, "A", "ban", "Split"),
        (1, "B", "ban", "Sunset"),
        (2, "A", "pick", "Haven"),
        (3, "B", "pick", "Ascent"),
        (4, "A", "ban", "Abyss"),
        (5, "B", "ban", "Lotus"),
        (6, None, "decider", "Summit"),
    ]
    assert out[0] == SimulatedVetoAction(0, "A", "ban", "Split")
    assert out[6] == SimulatedVetoAction(6, None, "decider", "Summit")
    # to_dict shape matches the real VetoAction.to_dict key names.
    assert out[0].to_dict() == {
        "step_index": 0,
        "team": "A",
        "action": "ban",
        "map_name": "Split",
    }
    remaining = set(POOL)
    for step_index, action in enumerate(ACTION_SEQUENCES["Bo3"]):
        if action == "decider":
            assert len(remaining) == 1
            expected = next(iter(remaining))
        else:
            acting = "A" if step_index % 2 == 0 else "B"
            scores = {
                name: map_win_rate.team_map_win_rate(
                    acting, name, QUERY_DATE, matches_df, maps_df,
                    map_win_rate.DEFAULT_K,
                ).mean
                for name in remaining
            }
            if action == "ban":
                expected = min(remaining, key=lambda n: (scores[n], n))
            else:
                expected = min(remaining, key=lambda n: (-scores[n], n))
        got = out[step_index]
        assert got.map_name == expected
        assert got.action == action
        if action == "decider":
            assert got.team is None
        else:
            assert got.team == ("A" if step_index % 2 == 0 else "B")
        remaining.remove(got.map_name)


# --------------------------------------------------------------------------
# plan#5b / 5c: Bo1 and Bo5 shapes
# --------------------------------------------------------------------------


def test_bo1_shape():
    # Bo1 = six alternating bans then a decider. On the 16-map league
    # A's bans are Split, Abyss, Ascent (its three weakest, the last
    # two alphabetical ties at 0.5) and B's are Sunset, Haven, Lotus;
    # Summit is left over.
    matches_df, maps_df = _league_tables()
    out = simulate_veto(
        "A", "B", "Bo1", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert [(a.step_index, a.team, a.action, a.map_name) for a in out] == [
        (0, "A", "ban", "Split"),
        (1, "B", "ban", "Sunset"),
        (2, "A", "ban", "Abyss"),
        (3, "B", "ban", "Haven"),
        (4, "A", "ban", "Ascent"),
        (5, "B", "ban", "Lotus"),
        (6, None, "decider", "Summit"),
    ]
    assert [a.action for a in out] == ["ban"] * 6 + ["decider"]


def test_bo5_shape():
    # Bo5 = ban, ban, then four alternating picks, then a decider. The
    # picks on the 16-map league are Haven/Ascent (the two strong maps)
    # then Abyss/Lotus (alphabetical tie-breaks at 0.5); Summit left.
    matches_df, maps_df = _league_tables()
    out = simulate_veto(
        "A", "B", "Bo5", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert [(a.step_index, a.team, a.action, a.map_name) for a in out] == [
        (0, "A", "ban", "Split"),
        (1, "B", "ban", "Sunset"),
        (2, "A", "pick", "Haven"),
        (3, "B", "pick", "Ascent"),
        (4, "A", "pick", "Abyss"),
        (5, "B", "pick", "Lotus"),
        (6, None, "decider", "Summit"),
    ]


# --------------------------------------------------------------------------
# plan#5d: determinism (no set/dict iteration-order reliance)
# --------------------------------------------------------------------------


def test_determinism_across_repeated_and_reordered_calls():
    # Identical arguments twice, plus the same pool passed in reverse
    # order: all three runs must produce byte-identical sequences. The
    # reverse-order run is the sharper check — the simulator walks a
    # `set` and builds per-team score dicts whose insertion order
    # changes with the pool order, so any hidden reliance on that
    # order would surface as a differing choice.
    matches_df, maps_df = _league_tables()
    first = simulate_veto(
        "A", "B", "Bo5", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    second = simulate_veto(
        "A", "B", "Bo5", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    reversed_pool = simulate_veto(
        "A", "B", "Bo5", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=tuple(reversed(POOL)),
    )
    assert first == second
    assert first == reversed_pool
    assert [a.to_dict() for a in first] == [a.to_dict() for a in second]


# --------------------------------------------------------------------------
# plan#5e: alphabetical tie-break (both a ban and a pick scenario)
# --------------------------------------------------------------------------


def test_alphabetical_tie_break_with_no_history():
    # With an empty league both teams are unseen, so every shrunk mean
    # falls back to the same 0.5 prior on all seven maps. Every step is
    # therefore a full tie and the documented secondary key (ascending
    # map name) must decide: the Bo3 walk becomes Abyss, Ascent, Haven,
    # Lotus, Split, Summit (ban/pick type per the sequence), decider
    # Sunset — covering both a ban tie (step 0) and a pick tie (step 2)
    # with the same alphabetically-first rule.
    matches_df, maps_df = _build([], [])
    probe = map_win_rate.team_map_win_rate(
        "A", "Haven", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K
    )
    assert probe.mean == 0.5  # the tie really is a tie before the test
    out = simulate_veto(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert [(a.step_index, a.team, a.action, a.map_name) for a in out] == [
        (0, "A", "ban", "Abyss"),
        (1, "B", "ban", "Ascent"),
        (2, "A", "pick", "Haven"),
        (3, "B", "pick", "Lotus"),
        (4, "A", "ban", "Split"),
        (5, "B", "ban", "Summit"),
        (6, None, "decider", "Sunset"),
    ]
    # A pick tie-break on the same league via the Bo5 first pick:
    # step 2 (A picks) must again be the alphabetically first map.
    bo5 = simulate_veto(
        "A", "B", "Bo5", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert (bo5[2].action, bo5[2].team, bo5[2].map_name) == ("pick", "A", "Haven")


# --------------------------------------------------------------------------
# plan#5f / 5g: fail-loud validation
# --------------------------------------------------------------------------


def test_map_pool_size_mismatch_raises_value_error():
    # A 5-map pool cannot feed a 7-step Bo3 sequence; the simulator
    # must refuse rather than silently generalise to a 5-map shape.
    matches_df, maps_df = _league_tables()
    with pytest.raises(ValueError, match="map_pool has 5 map"):
        simulate_veto(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            map_win_rate.DEFAULT_K,
            map_pool=("Abyss", "Ascent", "Haven", "Lotus", "Split"),
        )


def test_unknown_best_of_raises_value_error():
    # "Bo7" is not a key of ACTION_SEQUENCES; the error must name the
    # invalid value and the supported formats.
    matches_df, maps_df = _league_tables()
    with pytest.raises(ValueError, match="Bo7"):
        simulate_veto(
            "A", "B", "Bo7", QUERY_DATE, matches_df, maps_df,
            map_win_rate.DEFAULT_K,
            map_pool=POOL,
        )


def test_duplicate_pool_after_normalization_raises_value_error():
    # "breeze" and " Breeze " collapse to the same normalized name, so
    # the pool would desync the walk; fail loudly at validation time.
    matches_df, maps_df = _league_tables()
    with pytest.raises(ValueError, match="duplicate"):
        simulate_veto(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            map_win_rate.DEFAULT_K,
            map_pool=("breeze", " Breeze ", "Haven", "Lotus", "Split", "Summit", "Sunset"),
        )


# --------------------------------------------------------------------------
# plan#5h: leakage — a future map must not reach the win-rate estimates
# --------------------------------------------------------------------------


def test_future_map_does_not_leak_into_the_choice():
    # The league has a future Ascent win for A dated exactly at the
    # query timestamp. If it leaked, A's first ban would flip from
    # Ascent (0.2, the no-leak argmin) to Split (~0.333 after the prior
    # shift) — so the test is non-tautological: it proves the strict-<
    # cutoff really reaches team_map_win_rate/utils.asof, and that the
    # simulator's choice equals the result computed with the row
    # excluded.
    matches_df, maps_df = _leak_tables()
    no_leak = simulate_veto(
        "A", "B", "Bo3", _stamp(6), matches_df, maps_df, 2.0, map_pool=POOL
    )
    assert (no_leak[0].team, no_leak[0].action, no_leak[0].map_name) == (
        "A",
        "ban",
        "Ascent",
    )
    # The flip condition really holds: at the query cutoff Ascent's
    # mean is below Split's, so excluding the future row is the only
    # reason the choice is Ascent and not Split.
    ascent = map_win_rate.team_map_win_rate(
        "A", "Ascent", _stamp(6), matches_df, maps_df, 2.0
    ).mean
    split = map_win_rate.team_map_win_rate(
        "A", "Split", _stamp(6), matches_df, maps_df, 2.0
    ).mean
    assert ascent < split
    # Simulating on the tables with the future row physically removed
    # yields the identical full sequence.
    dropped_matches = matches_df[matches_df["match_id"] != "l6"]
    dropped_maps = maps_df[maps_df["match_id"] != "l6"]
    excluded = simulate_veto(
        "A", "B", "Bo3", _stamp(6), dropped_matches, dropped_maps, 2.0,
        map_pool=POOL,
    )
    assert excluded == no_leak
    # Once the query date moves past the future row (a legitimate part
    # of history), A's first ban flips to Split — proving the fixture
    # genuinely discriminates on the cutoff.
    with_leak = simulate_veto(
        "A", "B", "Bo3", _stamp(7), matches_df, maps_df, 2.0, map_pool=POOL
    )
    assert (with_leak[0].team, with_leak[0].action, with_leak[0].map_name) == (
        "A",
        "ban",
        "Split",
    )


# --------------------------------------------------------------------------
# plan#5i: caller-supplied map_pool normalization
# --------------------------------------------------------------------------


def test_caller_pool_names_are_normalized():
    # A caller-supplied pool with mixed case and stray whitespace must
    # normalize every entry (config.normalize_map_name) and produce the
    # identical sequence as the clean, already-normalized pool, with
    # all output map names in canonical title-case form.
    matches_df, maps_df = _league_tables()
    messy = (" abyss ", "ASCENT", "haven", "  lotus", "Split ", "SUMMIT", "Sunset ")
    messy_out = simulate_veto(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=messy,
    )
    clean_out = simulate_veto(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df, map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert messy_out == clean_out
    assert all(a.map_name in POOL for a in messy_out)


# --------------------------------------------------------------------------
# plan#4 (pool resolution): config-era default + no-era-covers ConfigError
# --------------------------------------------------------------------------


def test_default_map_pool_resolves_from_config_era():
    # map_pool=None resolves the pool from config.json's era covering
    # the query date's calendar date. 2026-08-23 falls inside the
    # 2026-abyss era (starts 2026-08-17), whose pool is exactly POOL —
    # so the default-path run must equal the explicit-pool run. The
    # league's January history is strictly before the August cutoff, so
    # the feature scores are unchanged.
    matches_df, maps_df = _league_tables()
    default_out = simulate_veto(
        "A", "B", "Bo3", "2026-08-23T12:15:00", matches_df, maps_df,
        map_win_rate.DEFAULT_K,
    )
    explicit_out = simulate_veto(
        "A", "B", "Bo3", "2026-08-23T12:15:00", matches_df, maps_df,
        map_win_rate.DEFAULT_K,
        map_pool=POOL,
    )
    assert default_out == explicit_out


def test_default_map_pool_raises_config_error_when_no_era_covers_date():
    # QUERY_DATE (2026-01-06) predates the first configured era
    # (2026-04-01), so the config-era resolution has no pool to answer
    # with and must fail loudly rather than guess today's pool.
    matches_df, maps_df = _league_tables()
    with pytest.raises(config.ConfigError):
        simulate_veto(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            map_win_rate.DEFAULT_K,
        )
