"""Tests for the batched design-matrix builder (models._shared, task 052).

``models._shared.build_feature_matrix`` is the batched sibling of
``build_feature_vector``: it must produce an ``(n, 13)`` matrix that is
element-for-element identical (per column, dtype-for-dtype) to looping
``build_feature_vector`` over the same row table — the correctness bar
task 052's decision 4 sets, with no tolerance. This module hosts the
synthetic parity test (fast, runs on every commit) and the real-v1
parity test (``@pytest.mark.slow``, checkpoint 11), which asserts the
same exactness against the session-scoped
``real_v1_train_design_matrix`` fixture's ground-truth ``X`` (built by
the per-row loop) column by column so a single-column mismatch is
immediately localised.
"""

import numpy as np
import pandas as pd
import pytest

from models import _shared
from tests._shared import _real_v1_available

# Column headers of the maps table the round-detail substrate reads.
_HALF_COLS = {
    "team1_first_half_rounds": 0.0,
    "team1_second_half_rounds": 0.0,
    "team2_first_half_rounds": 0.0,
    "team2_second_half_rounds": 0.0,
}


def _matches_df(rows):
    """Build a matches table with the full column set the 13 features read.

    Args:
        rows: A list of dicts, one per match, carrying the keys in the
            column set below (extra keys ignored by the explicit
            ``columns=`` ordering).

    Returns:
        A ``pandas.DataFrame`` with the full matches column set.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is).
    """
    return pd.DataFrame(
        rows,
        columns=[
            "match_id", "date", "team1_id", "team2_id", "team1_name",
            "team2_name", "event_name", "status",
        ],
    )


def _maps_df(rows):
    """Build a maps table with the full column set the 13 features read.

    Args:
        rows: A list of dicts, one per map, carrying the keys in the
            column set below.

    Returns:
        A ``pandas.DataFrame`` with the full maps column set.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is).
    """
    return pd.DataFrame(
        rows,
        columns=[
            "match_id", "map_index", "map_name", "team1_score", "team2_score",
            "winner", "team1_first_half_rounds", "team1_second_half_rounds",
            "team1_atk_rounds", "team1_def_rounds", "team2_first_half_rounds",
            "team2_second_half_rounds", "team2_atk_rounds", "team2_def_rounds",
        ],
    )


def _pms_df(rows):
    """Build a player_map_stats table with the columns the 13 features read.

    Args:
        rows: A list of dicts, one per player-map row.

    Returns:
        A ``pandas.DataFrame`` with the pms column set.

    Raises:
        Nothing (pandas ``ValueError`` on a missing key surfaces as-is).
    """
    return pd.DataFrame(
        rows,
        columns=[
            "match_id", "map_index", "player_name", "team_name", "rating",
            "acs", "first_kills", "first_deaths",
        ],
    )


def _regulation_map_row(match_id, map_index, map_name, t1_id, t2_id,
                        t1_name, t2_name, date, t1_score, t2_score, winner,
                        t1_atk, t1_def, t2_atk, t2_def):
    """Build one internally-valid regulation map dict (scores + round cols).

    Enforces ``round_detail``'s validation invariants by construction:
    per side ``atk + def == score`` and the opposing-side pairings each
    lie in ``[0, 12]`` and sum to the map total (asserted here so a
    mis-typed fixture never reaches the module).

    Args:
        match_id: The match id (also placed on the matches table).
        map_index: The 0-based map index within the match.
        map_name: The map name (title-cased).
        t1_id: The team1 side's id.
        t2_id: The team2 side's id.
        t1_name: The team1 side's display name.
        t2_name: The team2 side's display name.
        date: The match's ISO date string.
        t1_score: Rounds won by team1.
        t2_score: Rounds won by team2.
        winner: The finished-map winner marker (non-null).
        t1_atk: Team1's regulation attack-round wins.
        t1_def: Team1's regulation defence-round wins.
        t2_atk: Team2's regulation attack-round wins.
        t2_def: Team2's regulation defence-round wins.

    Returns:
        A dict with the full maps column set, ready for :func:`_maps_df`.

    Raises:
        ValueError: If the map's per-side sums or pairing bounds fail.
    """
    if t1_atk + t1_def != t1_score or t2_atk + t2_def != t2_score:
        raise ValueError("map per-side atk+def must equal the side's score")
    pairing1 = t1_atk + t2_def
    pairing2 = t1_def + t2_atk
    if not (0 <= pairing1 <= 12 and 0 <= pairing2 <= 12):
        raise ValueError("map opposing-side pairings must lie in [0, 12]")
    if pairing1 + pairing2 != t1_score + t2_score:
        raise ValueError("map pairings must partition the map's rounds")
    row = {
        "match_id": match_id,
        "map_index": map_index,
        "map_name": map_name,
        "team1_score": t1_score,
        "team2_score": t2_score,
        "winner": winner,
        "team1_atk_rounds": t1_atk,
        "team1_def_rounds": t1_def,
        "team2_atk_rounds": t2_atk,
        "team2_def_rounds": t2_def,
    }
    row.update(_HALF_COLS)
    return row


def _synthetic_league():
    """Build a four-match, three-team synthetic league plus a row table.

    Matches: ``m1`` A-B Haven 13-8; ``m2`` B-A Haven 13-11; ``m3`` A-C
    Bind 13-8; ``m4`` C-B Haven 13-8 — one finished regulation map per
    match, with per-side rosters (one player per side) carrying acs /
    rating and mirroring ``(first_kills, first_deaths)`` pairs so the
    per-map first-blood conservation invariant holds by construction.
    The row table holds one row per match at the match's own date (the
    assembler shape), plus an extra row at m1's date querying a map the
    pair never played (Bind) to exercise the zero-history map-prior
    edges through the whole matrix.

    Returns:
        A ``(rows_df, matches_df, maps_df, player_map_stats_df)``
        tuple; ``rows_df`` has ``team1_id, team2_id, map_name, date,
        match_id`` columns.

    Raises:
        Nothing.
    """
    d1, d2, d3, d4 = (
        "2026-01-01T10:00:00",
        "2026-01-03T10:00:00",
        "2026-01-05T10:00:00",
        "2026-01-07T10:00:00",
    )
    matches_rows = [
        {"match_id": "m1", "date": d1, "team1_id": "A", "team2_id": "B",
         "team1_name": "Alpha", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m2", "date": d2, "team1_id": "B", "team2_id": "A",
         "team1_name": "Beta", "team2_name": "Alpha",
         "event_name": "VCT 2026: EMEA Stage 1", "status": "completed"},
        {"match_id": "m3", "date": d3, "team1_id": "A", "team2_id": "C",
         "team1_name": "Alpha", "team2_name": "Gamma",
         "event_name": "VCT 2026: EMEA Stage 2", "status": "completed"},
        {"match_id": "m4", "date": d4, "team1_id": "C", "team2_id": "B",
         "team1_name": "Gamma", "team2_name": "Beta",
         "event_name": "VCT 2026: EMEA Stage 2", "status": "completed"},
    ]
    matches_df = _matches_df(matches_rows)

    maps_rows = [
        _regulation_map_row("m1", 0, "Haven", "A", "B", "Alpha", "Beta",
                            d1, 13, 8, "A", 7, 6, 3, 5),
        _regulation_map_row("m2", 0, "Haven", "B", "A", "Beta", "Alpha",
                            d2, 13, 11, "B", 7, 6, 6, 5),
        _regulation_map_row("m3", 0, "Bind", "A", "C", "Alpha", "Gamma",
                            d3, 13, 8, "A", 7, 6, 3, 5),
        _regulation_map_row("m4", 0, "Haven", "C", "B", "Gamma", "Beta",
                            d4, 13, 8, "C", 7, 6, 3, 5),
    ]
    maps_df = _maps_df(maps_rows)

    # m1: A wins Haven 13-8 (A margin +5); B mirrored.
    pms_rows = [
        {"match_id": "m1", "map_index": 0, "player_name": "pa1",
         "team_name": "Alpha", "rating": 1.10, "acs": 220.0,
         "first_kills": 11, "first_deaths": 10},
        {"match_id": "m1", "map_index": 0, "player_name": "pb1",
         "team_name": "Beta", "rating": 0.95, "acs": 180.0,
         "first_kills": 10, "first_deaths": 11},
        # m2: B wins Haven 13-11 (A margin -2); B mirrored.
        {"match_id": "m2", "map_index": 0, "player_name": "pb2",
         "team_name": "Beta", "rating": 1.05, "acs": 190.0,
         "first_kills": 12, "first_deaths": 9},
        {"match_id": "m2", "map_index": 0, "player_name": "pa2",
         "team_name": "Alpha", "rating": 1.08, "acs": 210.0,
         "first_kills": 9, "first_deaths": 12},
        # m3: A wins Bind 13-8 (A margin +5); C mirrored.
        {"match_id": "m3", "map_index": 0, "player_name": "pa3",
         "team_name": "Alpha", "rating": 1.15, "acs": 230.0,
         "first_kills": 8, "first_deaths": 7},
        {"match_id": "m3", "map_index": 0, "player_name": "pc1",
         "team_name": "Gamma", "rating": 0.90, "acs": 160.0,
         "first_kills": 7, "first_deaths": 8},
        # m4: C wins Haven 13-8 (C margin +5); B mirrored.
        {"match_id": "m4", "map_index": 0, "player_name": "pc2",
         "team_name": "Gamma", "rating": 0.98, "acs": 175.0,
         "first_kills": 10, "first_deaths": 6},
        {"match_id": "m4", "map_index": 0, "player_name": "pb3",
         "team_name": "Beta", "rating": 0.92, "acs": 170.0,
         "first_kills": 6, "first_deaths": 10},
    ]
    pms_df = _pms_df(pms_rows)

    rows_df = pd.DataFrame(
        [
            {"team1_id": "A", "team2_id": "B", "map_name": "Haven",
             "date": d1, "match_id": "m1"},
            # A never-played map for this pair (zero-history map priors).
            {"team1_id": "A", "team2_id": "B", "map_name": "Bind",
             "date": d1, "match_id": "m1"},
            {"team1_id": "B", "team2_id": "A", "map_name": "Haven",
             "date": d2, "match_id": "m2"},
            {"team1_id": "A", "team2_id": "C", "map_name": "Bind",
             "date": d3, "match_id": "m3"},
            {"team1_id": "C", "team2_id": "B", "map_name": "Haven",
             "date": d4, "match_id": "m4"},
        ]
    )
    return rows_df, matches_df, maps_df, pms_df


def test_build_feature_matrix_synthetic_bit_exact_parity():
    # build_feature_matrix must equal, element-for-element, a per-row
    # build_feature_vector loop over the same rows — per column and as
    # a whole matrix, with no tolerance (the correctness bar of task
    # 052's decision 4).
    rows_df, matches_df, maps_df, pms_df = _synthetic_league()
    expected = np.asarray(
        [
            _shared.build_feature_vector(
                row.team1_id,
                row.team2_id,
                row.map_name,
                row.date,
                matches_df,
                maps_df,
                pms_df,
            )
            for row in rows_df.itertuples(index=False)
        ],
        dtype=float,
    )
    got = _shared.build_feature_matrix(rows_df, matches_df, maps_df, pms_df)
    assert got.shape == expected.shape == (len(rows_df), len(_shared.FEATURE_NAMES))
    # Per-column assertions first (a single-column mismatch must be
    # immediately localised, not hidden inside one blanket comparison).
    for column_index, name in enumerate(_shared.FEATURE_NAMES):
        assert np.array_equal(
            got[:, column_index], expected[:, column_index], equal_nan=True
        ), f"column {name!r} (index {column_index}) differs from the single-row loop"
    assert np.array_equal(got, expected, equal_nan=True)
    assert got.dtype == np.float64


def test_build_feature_matrix_requires_match_id_on_rows():
    # rows_df must carry match_id (the batched contract — event stage
    # resolves by match_id, not by (team1, team2, date) re-resolution);
    # its absence is a KeyError naming the column.
    rows_df, matches_df, maps_df, pms_df = _synthetic_league()
    stripped = rows_df.drop(columns=["match_id"])
    with pytest.raises(KeyError, match="match_id"):
        _shared.build_feature_matrix(stripped, matches_df, maps_df, pms_df)


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_batched_matches_single_row_on_real_v1_train_split(
    real_v1_train_design_matrix,
):
    # The checkpoint-11 gate: on the real v1 train split (whose ground
    # truth X was built by looping build_feature_vector), the batched
    # matrix must match X element-for-element in every FEATURE_NAMES
    # column. One flagged risk (map_round_margin_variance's two-pass
    # np.var) is handled by the exact-order recipe in
    # features/closeness.batched_map_round_margin_variance, which feeds
    # np.var the literal same values in the literal same order the
    # single-row path assembles — verified empirically across all train
    # cutoffs and map names, so no tolerance fallback is needed for any
    # column.
    X, _y_ordinal, train_rows, matches_df, maps_df, player_map_stats_df = (
        real_v1_train_design_matrix
    )
    rows = train_rows[
        ["team1_id", "team2_id", "map_name", "date", "match_id"]
    ].copy()
    got = _shared.build_feature_matrix(rows, matches_df, maps_df, player_map_stats_df)
    assert got.shape == X.shape == (len(train_rows), len(_shared.FEATURE_NAMES))
    for column_index, name in enumerate(_shared.FEATURE_NAMES):
        assert np.array_equal(
            got[:, column_index], X[:, column_index]
        ), (
            f"batched column {name!r} (index {column_index}) differs from the "
            "per-row ground truth on real v1"
        )
