"""Tests for labels (roadmap M9): four-way outcome labelling.

Follows the tests/test_materialize.py pattern: pure in-memory
DataFrame fixtures for the labelling logic, tmp_path for the two small
Parquet I/O tests, no real cache/data directories and no live
network.
"""

import logging

import pandas as pd
import pytest

import labels
import materialize


def _full_shape_maps_df(score_rows):
    """Build a maps table with M8's full column set from score rows.

    Args:
        score_rows: An iterable of ``(match_id, map_index,
            team1_score, team2_score)`` tuples to place as rows.

    Returns:
        A ``pandas.DataFrame`` with ``materialize.MAPS_COLUMNS`` in
        exactly that order, the four given values populated and every
        other column (``winner``, ``duration``, the eight half-split
        columns) left null — the shape ``materialize.build_maps_table``
        produces for a map whose header rendered no half data, which
        is what ``labels.load_maps_table`` reads in the end-to-end
        test. Only the four populated columns matter to
        ``labels.build_labels_table``.

    Raises:
        Nothing.
    """
    data = []
    for match_id, map_index, team1_score, team2_score in score_rows:
        row = dict.fromkeys(materialize.MAPS_COLUMNS)
        row.update(
            {
                "match_id": match_id,
                "map_index": map_index,
                "team1_score": team1_score,
                "team2_score": team2_score,
            }
        )
        data.append(row)
    return pd.DataFrame(data, columns=materialize.MAPS_COLUMNS)


# --------------------------------------------------------------------------
# compute_outcome
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("team1_score", "team2_score", "expected"),
    [
        # A wins in regulation: loser below 12, margin +7.
        (13, 6, ("A-regulation", 0, 7)),
        # Smallest legal OT win: both sides reached 12+, A wins by 2.
        (14, 12, ("A-OT", 1, 2)),
        # B wins in overtime, margin negative (team1 minus team2).
        (12, 14, ("B-OT", 2, -2)),
        # B wins in regulation.
        (6, 13, ("B-regulation", 3, -7)),
    ],
)
def test_compute_outcome_four_labels(team1_score, team2_score, expected):
    # One case per label: regulation is any win with the loser below
    # 12 rounds, overtime requires the loser to reach 12+, and the
    # signed margin always follows team1 minus team2.
    assert labels.compute_outcome(team1_score, team2_score) == expected


def test_compute_outcome_ot_criterion_is_loser_side_not_twelve_twelve():
    # A 17-15 map is overtime too: the criterion is the losing side
    # reaching 12+ rounds (min(scores) >= 12), not the specific
    # 12-12 scoreline.
    assert labels.compute_outcome(17, 15) == ("A-OT", 1, 2)


def test_compute_outcome_tie_raises():
    # A tied finished map has no winner and is impossible under the
    # score-validity invariant, so the function refuses to silently
    # mislabel one rather than guessing a side.
    with pytest.raises(ValueError, match="tie"):
        labels.compute_outcome(13, 13)


# --------------------------------------------------------------------------
# build_labels_table
# --------------------------------------------------------------------------


def test_build_labels_table_all_four_labels():
    # One row per map, LABELS_COLUMNS in fixed order, (match_id,
    # map_index) passed through untouched, and label/ordinal/margin
    # all following the scores.
    maps_df = pd.DataFrame(
        [
            {"match_id": "1", "map_index": 0, "team1_score": 13, "team2_score": 6},
            {"match_id": "1", "map_index": 1, "team1_score": 14, "team2_score": 12},
            {"match_id": "2", "map_index": 0, "team1_score": 12, "team2_score": 14},
            {"match_id": "2", "map_index": 1, "team1_score": 6, "team2_score": 13},
        ]
    )
    df, skipped = labels.build_labels_table(maps_df)
    assert skipped == 0
    assert list(df.columns) == list(labels.LABELS_COLUMNS)
    assert len(df) == 4
    assert list(df["match_id"]) == ["1", "1", "2", "2"]
    assert list(df["map_index"]) == [0, 1, 0, 1]
    assert list(df["outcome_label"]) == [
        "A-regulation",
        "A-OT",
        "B-OT",
        "B-regulation",
    ]
    assert list(df["outcome_ordinal"]) == [0, 1, 2, 3]
    assert list(df["round_margin"]) == [7, 2, -2, -7]


def test_build_labels_table_ignores_winner_column():
    # The label follows purely from the two scores; a stale or wrong
    # winner column must not influence it (design decision 2).
    maps_df = pd.DataFrame(
        [
            {
                "match_id": "1",
                "map_index": 0,
                "team1_score": 6,
                "team2_score": 13,
                "winner": "Alpha",
            }
        ]
    )
    df, skipped = labels.build_labels_table(maps_df)
    assert skipped == 0
    assert df.iloc[0]["outcome_label"] == "B-regulation"
    assert df.iloc[0]["outcome_ordinal"] == 3
    assert df.iloc[0]["round_margin"] == -7


def test_build_labels_table_skips_null_score_row():
    # A malformed row with a null score (which can in principle slip
    # past M8's winner-only finished check) is skipped and counted,
    # not raised and not coerced into compute_outcome.
    maps_df = pd.DataFrame(
        [
            {"match_id": "1", "map_index": 0, "team1_score": 13, "team2_score": 6},
            {"match_id": "1", "map_index": 1, "team1_score": None, "team2_score": 11},
            {"match_id": "2", "map_index": 0, "team1_score": 13, "team2_score": None},
        ]
    )
    df, skipped = labels.build_labels_table(maps_df)
    assert skipped == 2
    assert len(df) == 1
    assert list(df["outcome_label"]) == ["A-regulation"]


def test_build_labels_table_empty_input():
    # Zero-row input yields a zero-row, schema-correct output and a
    # zero skip count (an empty maps.parquet is M8's problem to flag,
    # not M9's).
    empty = pd.DataFrame(
        columns=["match_id", "map_index", "team1_score", "team2_score"]
    )
    df, skipped = labels.build_labels_table(empty)
    assert skipped == 0
    assert len(df) == 0
    assert list(df.columns) == list(labels.LABELS_COLUMNS)


def test_build_labels_table_empty_input_has_schema_dtypes():
    # Regression for review finding 1: the empty case must carry the
    # fixed LABELS_DTYPES schema, not the all-object/null columns a
    # bare ``pd.DataFrame([], columns=...)`` construction produces.
    # map_index/outcome_ordinal/round_margin stay int64 and the text
    # columns stay object, so empty and non-empty runs write the same
    # Parquet schema.
    empty = pd.DataFrame(
        columns=["match_id", "map_index", "team1_score", "team2_score"]
    )
    df, skipped = labels.build_labels_table(empty)
    assert skipped == 0
    assert len(df) == 0
    for column, dtype in labels.LABELS_DTYPES.items():
        assert df[column].dtype == dtype


def test_build_labels_table_tie_raises():
    # A tied non-null row is an invariant break: the vectorized
    # labelling path raises the same ValueError compute_outcome does
    # (message contains "tie") rather than silently guessing a side.
    maps_df = pd.DataFrame(
        [
            {
                "match_id": "1",
                "map_index": 0,
                "team1_score": 13,
                "team2_score": 13,
            }
        ]
    )
    with pytest.raises(ValueError, match="tie"):
        labels.build_labels_table(maps_df)


def test_build_labels_table_ordinal_matches_label():
    # OUTCOME_LABELS[ordinal] == label for every produced row: the
    # ordinal-to-string mapping is an invariant for the ordinal model,
    # not an implementation detail.
    maps_df = pd.DataFrame(
        [
            {"match_id": "1", "map_index": 0, "team1_score": 13, "team2_score": 6},
            {"match_id": "1", "map_index": 1, "team1_score": 14, "team2_score": 12},
            {"match_id": "1", "map_index": 2, "team1_score": 12, "team2_score": 14},
            {"match_id": "1", "map_index": 3, "team1_score": 6, "team2_score": 13},
        ]
    )
    df, _ = labels.build_labels_table(maps_df)
    for _, row in df.iterrows():
        assert labels.OUTCOME_LABELS[row["outcome_ordinal"]] == row["outcome_label"]


def test_build_labels_table_label_derives_from_outcome_labels(monkeypatch):
    # Regression for round-2 finding 4: the label string must be
    # derived from OUTCOME_LABELS via the computed ordinal, not from a
    # second hardcoded string list — changing OUTCOME_LABELS alone must
    # change the produced labels without any other edit.
    monkeypatch.setattr(
        labels,
        "OUTCOME_LABELS",
        ("X-regulation", "X-OT", "Y-OT", "Y-regulation"),
    )
    maps_df = pd.DataFrame(
        [
            {"match_id": "1", "map_index": 0, "team1_score": 13, "team2_score": 6},
            {"match_id": "1", "map_index": 1, "team1_score": 14, "team2_score": 12},
            {"match_id": "1", "map_index": 2, "team1_score": 12, "team2_score": 14},
            {"match_id": "1", "map_index": 3, "team1_score": 6, "team2_score": 13},
        ]
    )
    df, skipped = labels.build_labels_table(maps_df)
    assert skipped == 0
    assert list(df["outcome_label"]) == [
        "X-regulation",
        "X-OT",
        "Y-OT",
        "Y-regulation",
    ]


def test_build_labels_table_null_score_warning_scalar_with_duplicate_index(caplog):
    # Regression for round-2 finding 3: with a non-unique maps_df index
    # the null-score warning must report the one skipped row's scalar
    # values, not a multi-row Series repr (which the old .at lookup
    # produced for duplicate index labels).
    maps_df = pd.DataFrame(
        [
            {"match_id": "1", "map_index": 0, "team1_score": None, "team2_score": 11},
            {"match_id": "1", "map_index": 1, "team1_score": 13, "team2_score": 6},
        ],
        index=[0, 0],
    )
    with caplog.at_level(logging.WARNING):
        df, skipped = labels.build_labels_table(maps_df)
    assert skipped == 1
    assert len(df) == 1
    warning = caplog.text
    assert "match 1" in warning
    assert "map_index 0" in warning
    assert "nan-11" in warning
    assert "Name: match_id" not in warning
    assert "Name: map_index" not in warning


# --------------------------------------------------------------------------
# load_maps_table / write_labels_table
# --------------------------------------------------------------------------


def test_load_maps_table_roundtrip(tmp_path):
    # A maps.parquet written with M8's full column shape reads back
    # with the scores intact.
    maps_df = _full_shape_maps_df([("1", 0, 13, 6)])
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    maps_df.to_parquet(version_dir / "maps.parquet", index=False)
    loaded = labels.load_maps_table(tmp_path, "v1")
    assert len(loaded) == 1
    assert loaded.iloc[0]["match_id"] == "1"
    assert loaded.iloc[0]["team1_score"] == 13


def test_load_maps_table_missing_raises(tmp_path):
    # A missing maps.parquet (materialize.py never run for this
    # version) surfaces as FileNotFoundError — a clear "run
    # materialize.py first" signal, not a wrapped error.
    with pytest.raises(FileNotFoundError):
        labels.load_maps_table(tmp_path, "v1")


def test_write_labels_table_roundtrip(tmp_path):
    # A small labels table round-trips through tmp_path, including an
    # OT label and a negative margin.
    labels_df = pd.DataFrame(
        [
            {
                "match_id": "1",
                "map_index": 0,
                "outcome_label": "A-OT",
                "outcome_ordinal": 1,
                "round_margin": 2,
            }
        ],
        columns=labels.LABELS_COLUMNS,
    )
    labels.write_labels_table(labels_df, tmp_path, "v9")
    written = pd.read_parquet(tmp_path / "v9" / "labels.parquet")
    assert len(written) == 1
    assert written.iloc[0]["outcome_label"] == "A-OT"
    assert written.iloc[0]["round_margin"] == 2


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # Defaults: version v1, output dir "data" (the same conventions
    # materialize.py defaults to).
    args = labels.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"


def test_parse_args_overrides():
    # Each flag overrides its default and is passed through verbatim.
    args = labels.parse_args(["--version", "v2", "--output-dir", "/tmp/out"])
    assert args.version == "v2"
    assert args.output_dir == "/tmp/out"


# --------------------------------------------------------------------------
# main — end to end
# --------------------------------------------------------------------------


def test_main_end_to_end(tmp_path, caplog):
    # Seeding a full-shape maps.parquet covering three labels and
    # running main against it: labels.parquet appears beside it, the
    # row count matches, labels and ordinals and margins are correct,
    # the join keys pass through, and main returns 0.
    caplog.set_level(logging.INFO)
    maps_df = _full_shape_maps_df(
        [
            ("1", 0, 13, 6),
            ("1", 1, 14, 12),
            ("2", 0, 6, 13),
        ]
    )
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    maps_df.to_parquet(version_dir / "maps.parquet", index=False)

    rc = labels.main(["--output-dir", str(tmp_path), "--version", "v1"])
    assert rc == 0

    written = pd.read_parquet(version_dir / "labels.parquet")
    assert len(written) == 3
    assert list(written["match_id"]) == ["1", "1", "2"]
    assert list(written["map_index"]) == [0, 1, 0]
    assert list(written["outcome_label"]) == [
        "A-regulation",
        "A-OT",
        "B-regulation",
    ]
    assert list(written["outcome_ordinal"]) == [0, 1, 3]
    assert list(written["round_margin"]) == [7, 2, -7]
    assert "labelled 3 maps" in caplog.text


def test_main_against_empty_maps_table(tmp_path, caplog):
    # An empty maps.parquet (materialize.py materialised nothing) still
    # yields a schema-correct empty labels.parquet and a 0 return code:
    # M9 does not duplicate M8's empty-dataset signal.
    caplog.set_level(logging.INFO)
    empty = _full_shape_maps_df([])
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    empty.to_parquet(version_dir / "maps.parquet", index=False)

    rc = labels.main(["--output-dir", str(tmp_path), "--version", "v1"])
    assert rc == 0

    written = pd.read_parquet(version_dir / "labels.parquet")
    assert len(written) == 0
    assert list(written.columns) == list(labels.LABELS_COLUMNS)
    # Regression for review finding 1: even a zero-row labels.parquet
    # round-trips with the numeric columns as int64, not null/object.
    assert str(written["map_index"].dtype) == "int64"
    assert str(written["outcome_ordinal"].dtype) == "int64"
    assert str(written["round_margin"].dtype) == "int64"
    assert "labelled 0 maps" in caplog.text
