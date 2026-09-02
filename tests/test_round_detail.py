"""Tests for per-map round-detail derivation + validity assertions (M38.1).

Covers the two hand-computed derivation paths (regulation and OT maps),
the case-split reconciliation and opposing-side-pairing fail-loud
guards (each exercised through the public derivation path, which
re-validates on every call, plus the separately importable validation
function), the countable null-row exclusion contract, the required-
column contract, the long-frame output shape, and a skip-guarded
real-``data/v1`` smoke test pinning today's real numbers (2 excluded
maps, 29 OT maps, the naive-check 218/242 figure recorded in the module
docstring, and sane bounds on every derived field).
"""

from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from features import round_detail as rd


def _map_row(
    match_id,
    map_index,
    team1_score,
    team2_score,
    *,
    team1_atk_rounds,
    team1_def_rounds,
    team2_atk_rounds,
    team2_def_rounds,
    team1_first_half_rounds=0.0,
    team1_second_half_rounds=0.0,
    team2_first_half_rounds=0.0,
    team2_second_half_rounds=0.0,
) -> dict:
    """Build one maps-shaped fixture row dict.

    Maps each fixture value onto the module's own column-name constants
    so a fixture and the module share one spelling. The four half-split
    columns default to non-null ``0.0`` because the module's derivation
    never reads them — they participate only in null-row identification
    (the real-data finding recorded in the module docstring is that
    halves are never cross-checked against atk/def), so a fixture that
    omits them is still a fully usable non-null row. Pass ``None`` (or
    ``float("nan")``) for any round column to construct the null-row
    exclusion case.

    Args:
        match_id: The fixture match id (string or int).
        map_index: The 0-based map index within the match.
        team1_score: Rounds won by team1 on the finished map.
        team2_score: Rounds won by team2 on the finished map.
        team1_atk_rounds: Team1's regulation attack-round wins.
        team1_def_rounds: Team1's regulation defence-round wins.
        team2_atk_rounds: Team2's regulation attack-round wins.
        team2_def_rounds: Team2's regulation defence-round wins.
        team1_first_half_rounds: Team1's first-half round wins (default
            non-null ``0.0``; unused by derivation).
        team1_second_half_rounds: Team1's second-half round wins
            (default non-null ``0.0``; unused by derivation).
        team2_first_half_rounds: Team2's first-half round wins (default
            non-null ``0.0``; unused by derivation).
        team2_second_half_rounds: Team2's second-half round wins
            (default non-null ``0.0``; unused by derivation).

    Returns:
        A dict keyed by the maps-shaped column names (the module's
        :data:`rd.REQUIRED_COLUMNS` spellings).

    Raises:
        Nothing.
    """
    return {
        rd.MATCH_ID_COL: match_id,
        rd.MAP_INDEX_COL: map_index,
        rd.TEAM1_SCORE_COL: team1_score,
        rd.TEAM2_SCORE_COL: team2_score,
        rd.TEAM1_ATK_COL: team1_atk_rounds,
        rd.TEAM1_DEF_COL: team1_def_rounds,
        rd.TEAM2_ATK_COL: team2_atk_rounds,
        rd.TEAM2_DEF_COL: team2_def_rounds,
        rd.TEAM1_FIRST_HALF_COL: team1_first_half_rounds,
        rd.TEAM1_SECOND_HALF_COL: team1_second_half_rounds,
        rd.TEAM2_FIRST_HALF_COL: team2_first_half_rounds,
        rd.TEAM2_SECOND_HALF_COL: team2_second_half_rounds,
    }


def _maps_df(rows):
    """Build a maps-shaped fixture DataFrame from row dicts.

    Wraps ``pd.DataFrame`` so every fixture produces the fixed
    :data:`rd.REQUIRED_COLUMNS` column order regardless of how a given
    fixture's rows were written.

    Args:
        rows: A list of dicts from :func:`_map_row` (or plain dicts
            with the maps-shaped column keys), one per map row.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`rd.REQUIRED_COLUMNS`
        columns, in that order.

    Raises:
        Nothing (pandas ``ValueError`` on a malformed dict surfaces as
        is).
    """
    return pd.DataFrame(rows, columns=list(rd.REQUIRED_COLUMNS))


def _record(records, match_id, map_index, side):
    """Fetch one derived record row by its ``(match_id, map_index, side)`` key.

    The long-format ``records`` frame holds exactly one row per key
    (two rows per input map, one per side), so a lookup is a filtered
    row, not a list.

    Args:
        records: The derived ``records`` frame from
            :func:`rd.derive_map_round_details`.
        match_id: The match id to look up.
        map_index: The map index to look up.
        side: The side marker (``rd.TEAM1_SIDE`` or ``rd.TEAM2_SIDE``).

    Returns:
        The matching ``pandas.Series`` row.

    Raises:
        AssertionError: If the key does not match exactly one record
            row.
    """
    match = records[
        (records[rd.MATCH_ID_COL] == match_id)
        & (records[rd.MAP_INDEX_COL] == map_index)
        & (records["side"] == side)
    ]
    assert len(match) == 1, (
        f"expected exactly one record for ({match_id}, {map_index}, "
        f"{side}), found {len(match)}"
    )
    return match.iloc[0]


def test_regulation_map_derivation_hand_computed():
    # A two-map regulation fixture with hand-computed expectations: the
    # opposing-side pairings give atk/def rounds played, OT fields are
    # zero, and the signed margin is the full-map score difference.
    maps_df = _maps_df(
        [
            # 13-2 regulation: team1 attacks a full 12-round half (11
            # won attacking, 1 won by team2 defending), team2 attacks a
            # 3-round truncated half (1 won attacking, 2 won by team1
            # defending).
            _map_row(
                "m1", 0, 13, 2,
                team1_atk_rounds=11.0, team1_def_rounds=2.0,
                team2_atk_rounds=1.0, team2_def_rounds=1.0,
                team1_first_half_rounds=10.0, team1_second_half_rounds=3.0,
                team2_first_half_rounds=2.0, team2_second_half_rounds=0.0,
            ),
            # 13-11 regulation: both attack halves run the full 12
            # rounds (24 total), every pairing sum is 12.
            _map_row(
                "m1", 1, 13, 11,
                team1_atk_rounds=7.0, team1_def_rounds=6.0,
                team2_atk_rounds=6.0, team2_def_rounds=5.0,
            ),
        ]
    )

    result = rd.derive_map_round_details(maps_df)
    assert len(result.excluded) == 0

    row = _record(result.records, "m1", 0, rd.TEAM1_SIDE)
    assert {
        "match_id": row[rd.MATCH_ID_COL],
        "map_index": row[rd.MAP_INDEX_COL],
        "side": row["side"],
        "atk_rounds_won": row["atk_rounds_won"],
        "atk_rounds_played": row["atk_rounds_played"],
        "def_rounds_won": row["def_rounds_won"],
        "def_rounds_played": row["def_rounds_played"],
        "ot_rounds_won": row["ot_rounds_won"],
        "ot_rounds_played": row["ot_rounds_played"],
        "signed_margin": row["signed_margin"],
    } == {
        "match_id": "m1",
        "map_index": 0,
        "side": "team1",
        "atk_rounds_won": 11,
        "atk_rounds_played": 12,
        "def_rounds_won": 2,
        "def_rounds_played": 3,
        "ot_rounds_won": 0,
        "ot_rounds_played": 0,
        "signed_margin": 11,
    }
    row = _record(result.records, "m1", 0, rd.TEAM2_SIDE)
    assert (row["atk_rounds_won"], row["atk_rounds_played"]) == (1, 3)
    assert (row["def_rounds_won"], row["def_rounds_played"]) == (1, 12)
    assert (row["ot_rounds_won"], row["ot_rounds_played"]) == (0, 0)
    assert row["signed_margin"] == -11

    row = _record(result.records, "m1", 1, rd.TEAM1_SIDE)
    assert (row["atk_rounds_won"], row["atk_rounds_played"]) == (7, 12)
    assert (row["def_rounds_won"], row["def_rounds_played"]) == (6, 12)
    assert (row["ot_rounds_won"], row["ot_rounds_played"]) == (0, 0)
    assert row["signed_margin"] == 2
    row = _record(result.records, "m1", 1, rd.TEAM2_SIDE)
    assert (row["atk_rounds_won"], row["atk_rounds_played"]) == (6, 12)
    assert (row["def_rounds_won"], row["def_rounds_played"]) == (5, 12)
    assert row["signed_margin"] == -2


def test_ot_map_derivation_hand_computed():
    # OT maps (the real v1 644709 map-0 scoreline 14-12 and a 15-13
    # variant): regulation rounds stay capped at 12 won/12 played per
    # side, OT rounds are carried separately and shared (ot_rounds_played
    # identical on both sides), and signed_margin includes the OT
    # rounds.
    maps_df = _maps_df(
        [
            _map_row(
                "m2", 0, 14, 12,
                team1_atk_rounds=9.0, team1_def_rounds=3.0,
                team2_atk_rounds=9.0, team2_def_rounds=3.0,
                team1_first_half_rounds=3.0, team1_second_half_rounds=9.0,
                team2_first_half_rounds=9.0, team2_second_half_rounds=3.0,
            ),
            _map_row(
                "m2", 1, 15, 13,
                team1_atk_rounds=7.0, team1_def_rounds=5.0,
                team2_atk_rounds=7.0, team2_def_rounds=5.0,
            ),
        ]
    )

    result = rd.derive_map_round_details(maps_df)
    assert len(result.excluded) == 0

    row = _record(result.records, "m2", 0, rd.TEAM1_SIDE)
    assert (row["atk_rounds_won"], row["atk_rounds_played"]) == (9, 12)
    assert (row["def_rounds_won"], row["def_rounds_played"]) == (3, 12)
    assert (row["ot_rounds_won"], row["ot_rounds_played"]) == (2, 2)
    assert row["signed_margin"] == 2
    row = _record(result.records, "m2", 0, rd.TEAM2_SIDE)
    assert (row["ot_rounds_won"], row["ot_rounds_played"]) == (0, 2)
    assert row["signed_margin"] == -2

    row = _record(result.records, "m2", 1, rd.TEAM1_SIDE)
    assert (row["atk_rounds_won"], row["atk_rounds_played"]) == (7, 12)
    assert (row["def_rounds_won"], row["def_rounds_played"]) == (5, 12)
    assert (row["ot_rounds_won"], row["ot_rounds_played"]) == (3, 4)
    assert row["signed_margin"] == 2
    row = _record(result.records, "m2", 1, rd.TEAM2_SIDE)
    assert (row["atk_rounds_won"], row["atk_rounds_played"]) == (7, 12)
    assert (row["def_rounds_won"], row["def_rounds_played"]) == (5, 12)
    assert (row["ot_rounds_won"], row["ot_rounds_played"]) == (1, 4)
    assert row["signed_margin"] == -2


def test_records_shape_is_long_two_rows_per_map():
    # The output contract: a long frame, exactly two rows per input
    # map, columns in MapRoundDetail field order, side values from the
    # two literal markers, and numeric round fields as ints.
    maps_df = _maps_df(
        [
            _map_row(
                "m1", 0, 13, 9,
                team1_atk_rounds=10.0, team1_def_rounds=3.0,
                team2_atk_rounds=7.0, team2_def_rounds=2.0,
            ),
            _map_row(
                "m1", 1, 13, 4,
                team1_atk_rounds=8.0, team1_def_rounds=5.0,
                team2_atk_rounds=0.0, team2_def_rounds=4.0,
            ),
        ]
    )
    result = rd.derive_map_round_details(maps_df)
    assert list(result.records.columns) == [field.name for field in fields(rd.MapRoundDetail)]
    assert len(result.records) == 4
    keys = list(
        zip(
            result.records[rd.MATCH_ID_COL],
            result.records[rd.MAP_INDEX_COL],
        )
    )
    assert keys == [("m1", 0), ("m1", 0), ("m1", 1), ("m1", 1)]
    assert set(result.records["side"].unique()) == {rd.TEAM1_SIDE, rd.TEAM2_SIDE}
    numeric = [
        "atk_rounds_won",
        "atk_rounds_played",
        "def_rounds_won",
        "def_rounds_played",
        "ot_rounds_won",
        "ot_rounds_played",
        "signed_margin",
    ]
    for column in numeric:
        assert result.records[column].dtype == "int64"


def test_regulation_case_split_violation_raises_from_derivation():
    # A regulation row whose atk + def != score must fail loudly from
    # the public derivation path (validation always runs), listing the
    # offending (match_id, map_index) — the naive single-side check
    # hides exactly this.
    maps_df = _maps_df(
        [
            _map_row(
                "bad", 0, 13, 8,
                team1_atk_rounds=10.0, team1_def_rounds=4.0,
                team2_atk_rounds=3.0, team2_def_rounds=5.0,
            ),
            _map_row(
                "ok", 0, 13, 6,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=2.0, team2_def_rounds=4.0,
            ),
        ]
    )
    with pytest.raises(ValueError, match="case-split"):
        rd.derive_map_round_details(maps_df)


def test_ot_case_split_violation_raises():
    # An OT row whose per-side atk + def != 12 (the roadmap's
    # "atk == def == 12" shorthand) is a structural impossibility for a
    # finished OT map and must raise ValueError.
    maps_df = _maps_df(
        [
            _map_row(
                "m9", 0, 14, 12,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=9.0, team2_def_rounds=3.0,
            )
        ]
    )
    with pytest.raises(ValueError, match="case-split"):
        rd.derive_map_round_details(maps_df)


def test_regulation_pairing_violation_raises():
    # A side-mislabeling simulation that the per-side case split alone
    # cannot catch: atk + def == score on both sides, but team1's
    # attack half (team1_atk + team2_def) exceeds the 12-round ceiling
    # of a regulation half — the opposing-side pairing assertion must
    # trip.
    maps_df = _maps_df(
        [
            _map_row(
                "m4", 0, 13, 2,
                team1_atk_rounds=13.0, team1_def_rounds=0.0,
                team2_atk_rounds=0.0, team2_def_rounds=2.0,
            )
        ]
    )
    with pytest.raises(ValueError, match="pairing"):
        rd.derive_map_round_details(maps_df)


def test_ot_pairing_violation_raises():
    # An OT row whose per-side atk + def == 12 holds but whose
    # opposing-side pairings are not both exactly 12 (team1's and
    # team2's regulation attack halves would not each be a full 12
    # rounds) must raise the pairing ValueError.
    maps_df = _maps_df(
        [
            _map_row(
                "m5", 0, 14, 12,
                team1_atk_rounds=10.0, team1_def_rounds=2.0,
                team2_atk_rounds=2.0, team2_def_rounds=10.0,
            )
        ]
    )
    with pytest.raises(ValueError, match="pairing"):
        rd.derive_map_round_details(maps_df)


def test_null_row_excluded_visibly_not_imputed():
    # A one-null-row + one-valid-row fixture yields exactly one map's
    # worth of records (2) plus one excluded entry identifying the null
    # row by match_id/map_index — a visible, countable exclusion, never
    # an imputed value and never a silent drop.
    maps_df = _maps_df(
        [
            _map_row(
                "scrape-gap", 2, 6, 13,
                team1_atk_rounds=None, team1_def_rounds=None,
                team2_atk_rounds=None, team2_def_rounds=None,
                team1_first_half_rounds=None, team1_second_half_rounds=None,
                team2_first_half_rounds=None, team2_second_half_rounds=None,
            ),
            _map_row(
                "fine", 0, 13, 6,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=3.0, team2_def_rounds=3.0,
            ),
        ]
    )
    result = rd.derive_map_round_details(maps_df)
    assert len(result.excluded) == 1
    assert (result.excluded[0].match_id, result.excluded[0].map_index) == (
        "scrape-gap",
        2,
    )
    assert "null" in result.excluded[0].reason
    assert len(result.records) == 2
    assert set(result.records["side"]) == {rd.TEAM1_SIDE, rd.TEAM2_SIDE}
    assert set(result.records[rd.MATCH_ID_COL]) == {"fine"}


def test_validation_function_is_separately_importable():
    # validate_map_round_details is its own testable unit: it returns
    # the excluded rows without raising on a valid frame, reports null
    # rows without raising, and raises on a case-split violation, so a
    # test can exercise it directly without a full derivation call.
    valid = _maps_df(
        [
            _map_row(
                "m1", 0, 13, 6,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=3.0, team2_def_rounds=3.0,
            )
        ]
    )
    assert rd.validate_map_round_details(valid) == ()

    with_null = _maps_df(
        [
            _map_row(
                "n", 0, 6, 13,
                team1_atk_rounds=None, team1_def_rounds=None,
                team2_atk_rounds=None, team2_def_rounds=None,
            ),
            _map_row(
                "m1", 0, 13, 6,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=3.0, team2_def_rounds=3.0,
            ),
        ]
    )
    excluded = rd.validate_map_round_details(with_null)
    assert len(excluded) == 1
    assert (excluded[0].match_id, excluded[0].map_index) == ("n", 0)

    violating = _maps_df(
        [
            _map_row(
                "v", 0, 13, 6,
                team1_atk_rounds=9.0, team1_def_rounds=5.0,
                team2_atk_rounds=2.0, team2_def_rounds=4.0,
            )
        ]
    )
    with pytest.raises(ValueError, match="case-split"):
        rd.validate_map_round_details(violating)


def test_null_score_on_non_null_round_rows_raises():
    # A surviving row (round columns present) with a null score is
    # impossible for a finished map and must raise ValueError
    # defensively, before any score arithmetic.
    maps_df = _maps_df(
        [
            _map_row(
                "no-score", 0, float("nan"), 6,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=2.0, team2_def_rounds=4.0,
            )
        ]
    )
    with pytest.raises(ValueError, match="null/NaN"):
        rd.derive_map_round_details(maps_df)


def test_missing_required_column_raises_keyerror():
    # A maps-shaped frame missing one round column must raise KeyError
    # naming the gap (this module has no utils.asof dependency, so the
    # required-column check is its own).
    maps_df = _maps_df(
        [
            _map_row(
                "m1", 0, 13, 6,
                team1_atk_rounds=9.0, team1_def_rounds=4.0,
                team2_atk_rounds=2.0, team2_def_rounds=4.0,
            )
        ]
    ).drop(columns=[rd.TEAM2_ATK_COL])
    with pytest.raises(KeyError, match="team2_atk_rounds"):
        rd.derive_map_round_details(maps_df)


def test_empty_input_yields_empty_schema_frame():
    # An empty maps-shaped frame (no rows) derives to a zero-row
    # records frame carrying the full MapRoundDetail schema and an
    # empty excluded tuple — no special-case crash.
    result = rd.derive_map_round_details(_maps_df([]))
    assert result.excluded == ()
    assert result.records.empty
    assert list(result.records.columns) == [field.name for field in fields(rd.MapRoundDetail)]


@pytest.mark.skipif(
    not Path("data/v1/maps.parquet").exists(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_smoke_sane_numbers():
    # Real v1 scale, the numbers recorded in the module docstring: no
    # ValueError on the full 244-row table; exactly 2 excluded maps
    # (the match-712803 pair); exactly 29 distinct OT maps in the
    # derived records (matching the raw-table min(score) >= 12 count);
    # the naive single-side reconciliation reproduces the roadmap's
    # 218/242 on the raw table; and sane bounds on every derived field
    # once merged back onto the raw scores.
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    assert len(maps_df) == 244

    result = rd.derive_map_round_details(maps_df)
    assert len(result.excluded) == 2
    assert {(e.match_id, e.map_index) for e in result.excluded} == {
        ("712803", 0),
        ("712803", 1),
    }
    assert all("null" in e.reason for e in result.excluded)
    assert len(result.records) == 2 * (len(maps_df) - len(result.excluded))

    records = result.records
    ot_records = records[records["ot_rounds_played"] > 0]
    ot_map_count = ot_records[[rd.MATCH_ID_COL, rd.MAP_INDEX_COL]].drop_duplicates()
    raw_ot_count = int(
        (maps_df[[rd.TEAM1_SCORE_COL, rd.TEAM2_SCORE_COL]].min(axis=1) >= 12).sum()
    )
    assert len(ot_map_count) == raw_ot_count == 29

    # The module docstring's naive-figure anchor: the single-side check
    # over all 242 non-null rows passes on exactly 218 (team1 alone),
    # which is the roadmap's "218/242" — re-verified in code here.
    non_null = maps_df[maps_df[list(rd.ROUND_COLS)].notna().all(axis=1)]
    naive_team1 = int(
        (
            (non_null[rd.TEAM1_ATK_COL] + non_null[rd.TEAM1_DEF_COL])
            == non_null[rd.TEAM1_SCORE_COL]
        ).sum()
    )
    assert naive_team1 == 218

    # Every record reconciles with its raw scoreline.
    raw = maps_df[[rd.MATCH_ID_COL, rd.MAP_INDEX_COL, rd.TEAM1_SCORE_COL, rd.TEAM2_SCORE_COL]]
    merged = records.merge(raw, on=[rd.MATCH_ID_COL, rd.MAP_INDEX_COL], how="left", validate="many_to_one")
    assert merged[[rd.TEAM1_SCORE_COL, rd.TEAM2_SCORE_COL]].notna().all().all()

    is_team1 = merged["side"] == rd.TEAM1_SIDE
    own_score = merged[rd.TEAM1_SCORE_COL].where(is_team1, merged[rd.TEAM2_SCORE_COL])
    opp_score = merged[rd.TEAM2_SCORE_COL].where(is_team1, merged[rd.TEAM1_SCORE_COL])
    total = merged[rd.TEAM1_SCORE_COL] + merged[rd.TEAM2_SCORE_COL]

    assert (merged["atk_rounds_played"] >= 0).all()
    assert (merged["atk_rounds_played"] <= 12).all()
    assert (merged["def_rounds_played"] >= 0).all()
    assert (merged["def_rounds_played"] <= 12).all()
    assert (merged["signed_margin"].abs() == (own_score - opp_score).abs()).all()

    is_ot_row = merged["ot_rounds_played"] > 0
    reg_won = merged["atk_rounds_won"] + merged["def_rounds_won"]
    assert (merged.loc[~is_ot_row, "ot_rounds_won"] == 0).all()
    assert (merged.loc[~is_ot_row, "ot_rounds_played"] == 0).all()
    assert (reg_won[~is_ot_row] == own_score[~is_ot_row]).all()
    assert (reg_won[is_ot_row] == 12).all()
    assert (merged.loc[is_ot_row, "ot_rounds_played"] == total[is_ot_row] - 24).all()
    assert (merged.loc[is_ot_row, "ot_rounds_won"] == own_score[is_ot_row] - 12).all()
    assert (merged.loc[is_ot_row, "ot_rounds_won"] >= 0).all()
    assert (merged.loc[is_ot_row, "ot_rounds_won"] <= merged.loc[is_ot_row, "ot_rounds_played"]).all()

    # ot_rounds_played is shared per map (identical on both sides).
    per_map_ot_played = records.groupby([rd.MATCH_ID_COL, rd.MAP_INDEX_COL])["ot_rounds_played"].nunique()
    assert (per_map_ot_played == 1).all()

    # signed margin sign convention: team1-side records carry
    # team1_score - team2_score.
    assert (merged.loc[is_team1, "signed_margin"] == own_score[is_team1] - opp_score[is_team1]).all()
