"""Tests for splits (roadmap M10): chronological split, walk-forward
folds, and out-of-fold prediction assembly.

Follows the tests/test_labels.py pattern: pure in-memory DataFrame
fixtures for the split/fold/assembly logic, tmp_path for the Parquet
I/O tests, no real cache/data directories and no live network.
"""

import logging

import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

from drivers import splits


def _dated_matches(n, start=None):
    """Build a minimal ``match_id`` + ``date`` matches table.

    Builds ``n`` chronologically ordered rows with distinct dates and
    zero-padded ids, so both the ``(date, match_id)`` sort used by
    ``splits.split_matches``/``splits.walk_forward_folds`` and any
    lexicographic fallback agree with insertion order — the tests can
    assert ``[m000, m001, ...]`` without re-deriving sort order.

    Args:
        n: The number of rows to build.
        start: The base timestamp for the first row's date; ``None``
            (the default) uses ``2026-01-01 00:00``. Each subsequent
            row is one hour later, so every date is valid, distinct,
            and monotonically increasing.

    Returns:
        A ``pandas.DataFrame`` with columns ``["match_id", "date"]``
        (the two columns ``splits.split_matches`` actually reads),
        ``n`` rows, ``match_id`` values ``"m000"`` through
        ``f"m{n-1:03d}"``, and ISO-formatted ``date`` strings.

    Raises:
        Nothing.
    """
    base = start if start is not None else Timestamp("2026-01-01")
    rows = [
        {
            "match_id": f"m{i:03d}",
            "date": (base + Timedelta(hours=i)).isoformat(),
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=["match_id", "date"])


# --------------------------------------------------------------------------
# split_matches
# --------------------------------------------------------------------------


def test_split_matches_partition_sizes_and_order():
    # n=40 with default test_frac 0.15: n_test = round(6.0) = 6, so the
    # earliest 34 matches are "train" and the last 6 are "test"; the
    # output is chronologically sorted and carries SPLITS_COLUMNS/
    # SPLITS_DTYPES exactly.
    result = splits.split_matches(_dated_matches(40))
    assert list(result.columns) == list(splits.SPLITS_COLUMNS)
    assert len(result) == 40
    assert list(result["match_id"]) == [f"m{i:03d}" for i in range(40)]
    assert list(result["split"]) == ["train"] * 34 + ["test"] * 6
    for column, dtype in splits.SPLITS_DTYPES.items():
        assert result[column].dtype == dtype


def test_split_matches_train_dates_before_test_dates():
    # Every train date must strictly precede every test date — the
    # chronological-split invariant, not just a size check.
    result = splits.split_matches(_dated_matches(40))
    train = result[result["split"] == "train"]
    test = result[result["split"] == "test"]
    assert train["date"].max() < test["date"].min()


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.1])
def test_split_matches_test_frac_out_of_range(bad_frac):
    # test_frac must be strictly inside (0, 1); the boundaries and
    # anything outside both raise.
    with pytest.raises(ValueError, match="test_frac"):
        splits.split_matches(_dated_matches(40), test_frac=bad_frac)


def test_split_matches_train_below_minimum_raises():
    # n=20 with default 0.15 -> 3 test / 17 train, which is below
    # MIN_TRAIN_MATCHES (20): an invariant-break, raised not skipped.
    with pytest.raises(ValueError, match="training"):
        splits.split_matches(_dated_matches(20))


def test_split_matches_sorts_shuffled_input():
    # A non-monotonic input row order must still sort chronologically
    # before the split is applied.
    shuffled = _dated_matches(40).sample(frac=1, random_state=0)
    result = splits.split_matches(shuffled)
    assert list(result["match_id"]) == [f"m{i:03d}" for i in range(40)]


def test_split_matches_duplicate_dates_broken_by_match_id():
    # Two rows sharing one date sort deterministically by match_id
    # (secondary key): "111" before "999". 23 distinct-date rows plus
    # the two shared-date rows = n=25, clearing the MIN_TRAIN_MATCHES
    # floor (21 train / 4 test).
    rows = [
        {"match_id": f"m{i:03d}", "date": f"2026-01-01T{i:02d}:00:00"}
        for i in range(23)
    ]
    rows.append({"match_id": "999", "date": "2026-01-02T00:00:00"})
    rows.append({"match_id": "111", "date": "2026-01-02T00:00:00"})
    result = splits.split_matches(pd.DataFrame(rows, columns=["match_id", "date"]))
    assert list(result["match_id"])[-2:] == ["111", "999"]


@pytest.mark.parametrize("columns", [["match_id"], ["date"], ["other"]])
def test_split_matches_missing_columns_raises(columns):
    # A frame lacking date_col or id_col surfaces as KeyError with the
    # missing column name preserved (same contract as labels.py).
    df = pd.DataFrame(
        [{"match_id": "m000", "date": "2026-01-01T00:00:00"}],
        columns=columns,
    )
    with pytest.raises(KeyError):
        splits.split_matches(df)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_split_matches_unparseable_date_raises():
    # An unparseable date propagates the pd.to_datetime error (a
    # ValueError subclass) rather than being silently mis-sorted.
    df = pd.DataFrame(
        [
            {"match_id": "m000", "date": "not-a-date"},
            {"match_id": "m001", "date": "2026-01-01T00:00:00"},
        ],
        columns=["match_id", "date"],
    )
    with pytest.raises(ValueError):
        splits.split_matches(df)


@pytest.mark.parametrize("null_value", [None, float("nan")])
def test_split_matches_null_date_raises(null_value):
    # A null date parses to NaT (not a pd.to_datetime error) and must be
    # rejected explicitly, otherwise NaT sorts to an arbitrary position
    # and breaks the train<test chronological invariant.
    df = pd.DataFrame(
        [
            {"match_id": "m000", "date": "2026-01-01T00:00:00"},
            {"match_id": "m001", "date": null_value},
        ],
        columns=["match_id", "date"],
    )
    with pytest.raises(ValueError, match="null"):
        splits.split_matches(df)


def test_split_matches_empty_raises():
    # A zero-row matches table cannot be split meaningfully; the empty
    # case is M8's problem to flag, so M10 raises instead of writing an
    # empty split.
    empty = pd.DataFrame(columns=["match_id", "date"])
    with pytest.raises(ValueError, match="empty"):
        splits.split_matches(empty)


# --------------------------------------------------------------------------
# walk_forward_folds
# --------------------------------------------------------------------------


def test_walk_forward_folds_worked_example():
    # n=40 with defaults: effective = min(5, 40 // 8 - 1) = 4 folds over
    # 5 blocks of 8. Each fold trains on the expanding union and
    # validates on the next block.
    folds = list(splits.walk_forward_folds(_dated_matches(40)))
    assert [f[0] for f in folds] == [1, 2, 3, 4]
    assert [len(f[1]) for f in folds] == [8, 16, 24, 32]
    assert [len(f[2]) for f in folds] == [8, 8, 8, 8]
    assert folds[0][1] == [f"m{i:03d}" for i in range(8)]
    assert folds[0][2] == [f"m{i:03d}" for i in range(8, 16)]
    assert folds[3][2] == [f"m{i:03d}" for i in range(32, 40)]


def test_walk_forward_folds_expanding_window():
    # Fold i's train set is exactly fold i-1's train set plus fold i-1's
    # validation block (the expanding-window property).
    folds = list(splits.walk_forward_folds(_dated_matches(40)))
    for i in range(1, len(folds)):
        prev, cur = folds[i - 1], folds[i]
        assert set(cur[1]) == set(prev[1]) | set(prev[2])


def test_walk_forward_folds_train_val_disjoint():
    # No fold ever validates on a match it trained on.
    for _, train_ids, val_ids in splits.walk_forward_folds(_dated_matches(40)):
        assert set(train_ids).isdisjoint(val_ids)


def test_walk_forward_folds_vals_cover_all_except_warmup():
    # The union of every fold's validation block covers every match
    # except the first/warm-up block, each exactly once.
    folds = list(splits.walk_forward_folds(_dated_matches(40)))
    all_ids = set(_dated_matches(40)["match_id"])
    covered = [mid for _, _, val_ids in folds for mid in val_ids]
    assert len(covered) == len(set(covered))
    warmup = set(folds[0][1])
    assert set(covered) == all_ids - warmup


def test_walk_forward_folds_too_small_raises():
    # n=8 with min_fold_block=8 -> effective = min(5, 1 - 1) = 0, too
    # small to form even one fold.
    with pytest.raises(ValueError, match="too small"):
        list(splits.walk_forward_folds(_dated_matches(8)))


def test_walk_forward_folds_sorts_input():
    # Non-monotonic input order sorts internally before folding, so the
    # fold contents are still chronological.
    shuffled = _dated_matches(40).sample(frac=1, random_state=1)
    folds = list(splits.walk_forward_folds(shuffled))
    assert folds[0][2] == [f"m{i:03d}" for i in range(8, 16)]


def test_walk_forward_folds_real_scale_83():
    # The concrete plan-derived example at data/v1's training size:
    # 83 matches -> exactly 5 folds over 6 blocks sized
    # [18, 13, 13, 13, 13, 13] (remainder folded into the 18-match
    # warm-up block).
    folds = list(splits.walk_forward_folds(_dated_matches(83)))
    assert len(folds) == 5
    assert [f[0] for f in folds] == [1, 2, 3, 4, 5]
    assert len(folds[0][1]) == 18
    assert [len(f[2]) for f in folds] == [13, 13, 13, 13, 13]


def test_walk_forward_folds_null_date_raises():
    # The same null-date rejection must apply to the fold generator's
    # internal sort, so a null-dated match cannot land in the wrong
    # walk-forward block and leak future data into an earlier fold's
    # training set (or vice versa).
    df = _dated_matches(40)
    df.loc[0, "date"] = None
    with pytest.raises(ValueError, match="null"):
        list(splits.walk_forward_folds(df))


# --------------------------------------------------------------------------
# assemble_out_of_fold_predictions
# --------------------------------------------------------------------------


def _fold_predictions_from_folds(folds, rows_per_match=1):
    """Build one well-formed predictions_df per fold from fold output.

    A test helper for the happy-path/cross-check cases: for each
    ``(fold_id, _, val_ids)`` tuple, it builds a ``predictions_df``
    with ``rows_per_match`` rows per validation match id (so
    ``rows_per_match=2`` simulates a one-row-per-map submission) and a
    simple ``pred`` column.

    Args:
        folds: The list of ``(fold_id, train_ids, val_ids)`` tuples as
            produced by ``splits.walk_forward_folds``.
        rows_per_match: How many rows to emit per distinct match id
            (1 = match-level, 2 = a two-map match).

    Returns:
        A list of ``(fold_id, predictions_df)`` tuples in ascending
        fold order, each predictions_df's ``match_id`` set exactly
        equal to its fold's validation ids.

    Raises:
        Nothing.
    """
    fold_predictions = []
    for fold_id, _, val_ids in folds:
        rows = []
        for mid in val_ids:
            for map_index in range(rows_per_match):
                rows.append(
                    {"match_id": mid, "map_index": map_index, "pred": 0.5}
                )
        fold_predictions.append((fold_id, pd.DataFrame(rows)))
    return fold_predictions


def _pred_df(ids):
    """Build a one-column ``match_id`` predictions DataFrame.

    A minimal predictions_df used by the structural-error tests, where
    only the match-id set matters, not any prediction/label columns.

    Args:
        ids: An iterable of match id values.

    Returns:
        A ``pandas.DataFrame`` with a single ``match_id`` column
        holding ``ids``.

    Raises:
        Nothing.
    """
    return pd.DataFrame({"match_id": list(ids)})


def test_assemble_out_of_fold_predictions_happy_path():
    # 4 folds x 8 matches -> 32 assembled rows, fold_id prepended in
    # ascending order, and coverage dict matching the recomputed folds.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    assembled, coverage = splits.assemble_out_of_fold_predictions(
        df, _fold_predictions_from_folds(folds)
    )
    assert len(assembled) == 32
    assert next(iter(assembled.columns)) == "fold_id"
    assert list(assembled["fold_id"]) == [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8
    assert coverage["train_matches"] == 40
    assert coverage["covered_matches"] == 32
    assert coverage["warmup_excluded_count"] == 8
    assert coverage["warmup_excluded_ids"] == folds[0][1]


def test_assemble_leak_raises():
    # A predictions_df that includes an id outside its fold's validation
    # set (here a warm-up/train id leaking in) is rejected as a leak.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    fold_predictions = []
    for fold_id, _, val_ids in folds:
        ids = list(val_ids)
        if fold_id == 2:
            ids.append(folds[0][1][0])
        fold_predictions.append((fold_id, pd.DataFrame({"match_id": ids})))
    with pytest.raises(ValueError, match="fold 2"):
        splits.assemble_out_of_fold_predictions(df, fold_predictions)


def test_assemble_incomplete_raises():
    # A predictions_df missing one of its fold's validation ids is
    # rejected as incomplete.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    fold_predictions = []
    for fold_id, _, val_ids in folds:
        ids = list(val_ids)
        if fold_id == 3:
            ids = ids[:-1]
        fold_predictions.append((fold_id, pd.DataFrame({"match_id": ids})))
    with pytest.raises(ValueError, match="fold 3"):
        splits.assemble_out_of_fold_predictions(df, fold_predictions)


def test_assemble_duplicate_fold_id_raises():
    # Submitting the same fold_id twice is a structural error caught
    # before any per-fold content check.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    fold_predictions = [
        (1, _pred_df(folds[0][2])),
        (1, _pred_df(folds[0][2])),
        (2, _pred_df(folds[1][2])),
        (3, _pred_df(folds[2][2])),
        (4, _pred_df(folds[3][2])),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        splits.assemble_out_of_fold_predictions(df, fold_predictions)


def test_assemble_missing_fold_raises():
    # Omitting an entire required fold_id is rejected.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    fold_predictions = [
        (1, _pred_df(folds[0][2])),
        (2, _pred_df(folds[1][2])),
        (4, _pred_df(folds[3][2])),
    ]
    with pytest.raises(ValueError, match="missing"):
        splits.assemble_out_of_fold_predictions(df, fold_predictions)


def test_assemble_invalid_fold_id_raises():
    # A fold_id that is not among the recomputed folds is rejected
    # explicitly (here 5 is used while fold 4 is the real last fold).
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    fold_predictions = [
        (1, _pred_df(folds[0][2])),
        (2, _pred_df(folds[1][2])),
        (3, _pred_df(folds[2][2])),
        (5, _pred_df(folds[3][2])),
    ]
    with pytest.raises(ValueError, match="fold_id 5"):
        splits.assemble_out_of_fold_predictions(df, fold_predictions)


def test_assemble_config_mismatch_hints_at_fold_parameters():
    # If the caller produced predictions with a different n_folds than
    # the one passed here, the recomputed fold set differs and the
    # error must name the likely cause (n_folds/min_fold_block
    # mismatch), not just report a bare missing-fold id.
    df = _dated_matches(40)
    folds_3 = list(splits.walk_forward_folds(df, n_folds=3))
    with pytest.raises(ValueError, match="n_folds"):
        splits.assemble_out_of_fold_predictions(
            df, _fold_predictions_from_folds(folds_3)
        )


def test_assemble_missing_id_col_raises():
    # A predictions_df lacking id_col surfaces as KeyError (the caller
    # forgot the join key), not a confusing set-membership error.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    fold_predictions = [
        (fold_id, pd.DataFrame({"not_match_id": list(val_ids)}))
        for fold_id, _, val_ids in folds
    ]
    with pytest.raises(KeyError):
        splits.assemble_out_of_fold_predictions(df, fold_predictions)


def test_assemble_map_level_granularity():
    # Two rows per match id (one per map) must still validate against
    # the match-level val_match_ids set: the check deduplicates by
    # match_id and covered_matches stays the distinct-match count.
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    assembled, coverage = splits.assemble_out_of_fold_predictions(
        df, _fold_predictions_from_folds(folds, rows_per_match=2)
    )
    assert len(assembled) == 64
    assert coverage["covered_matches"] == 32


def test_assemble_warmup_cross_check():
    # coverage["warmup_excluded_ids"] must equal the first block
    # walk_forward_folds itself produces on the same config (fold 1's
    # train ids, i.e. the warm-up block).
    df = _dated_matches(40)
    folds = list(splits.walk_forward_folds(df))
    _, coverage = splits.assemble_out_of_fold_predictions(
        df, _fold_predictions_from_folds(folds)
    )
    assert coverage["warmup_excluded_ids"] == folds[0][1]


# --------------------------------------------------------------------------
# join_split_to_maps
# --------------------------------------------------------------------------


def test_join_split_to_maps_adds_split_column():
    # The split column is merged in via match_id, other columns and the
    # original row order are untouched (including repeated match ids for
    # a Bo3).
    maps_df = pd.DataFrame(
        [
            {"match_id": "m000", "map_index": 0, "team1_score": 13},
            {"match_id": "m000", "map_index": 1, "team1_score": 10},
            {"match_id": "m001", "map_index": 0, "team1_score": 13},
        ]
    )
    splits_df = pd.DataFrame(
        [
            {"match_id": "m000", "date": "2026-01-01T00:00:00", "split": "train"},
            {"match_id": "m001", "date": "2026-01-01T01:00:00", "split": "test"},
        ],
        columns=splits.SPLITS_COLUMNS,
    )
    merged = splits.join_split_to_maps(maps_df, splits_df)
    assert list(merged.columns) == ["match_id", "map_index", "team1_score", "split"]
    assert list(merged["split"]) == ["train", "train", "test"]
    assert list(merged["team1_score"]) == [13, 10, 13]


def test_join_split_to_maps_missing_match_id_raises():
    # A map whose match_id is absent from splits_df means a stale/
    # mismatched dataset version; it is rejected rather than silently
    # producing a null split.
    maps_df = pd.DataFrame([{"match_id": "m000"}, {"match_id": "m999"}])
    splits_df = pd.DataFrame(
        [{"match_id": "m000", "date": "x", "split": "train"}],
        columns=splits.SPLITS_COLUMNS,
    )
    with pytest.raises(ValueError, match="absent"):
        splits.join_split_to_maps(maps_df, splits_df)


# --------------------------------------------------------------------------
# load_matches_table / write_splits_table
# --------------------------------------------------------------------------


def test_load_matches_table_roundtrip(tmp_path):
    # A matches.parquet written under tmp_path/v1 reads back intact.
    df = _dated_matches(5)
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    df.to_parquet(version_dir / "matches.parquet", index=False)
    loaded = splits.load_matches_table(tmp_path, "v1")
    assert len(loaded) == 5
    assert loaded.iloc[0]["match_id"] == "m000"


def test_load_matches_table_missing_raises(tmp_path):
    # A missing matches.parquet (materialize.py never run for this
    # version) surfaces as FileNotFoundError — a clear "run
    # materialize.py first" signal.
    with pytest.raises(FileNotFoundError):
        splits.load_matches_table(tmp_path, "v1")


def test_write_splits_table_roundtrip(tmp_path):
    # A computed split round-trips through tmp_path with both split
    # values present.
    splits_df = splits.split_matches(_dated_matches(30))
    splits.write_splits_table(splits_df, tmp_path, "v9")
    written = pd.read_parquet(tmp_path / "v9" / "splits.parquet")
    assert len(written) == 30
    assert set(written["split"]) == {"train", "test"}


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # Defaults: version v1, output dir "data", test_frac 0.15 — and no
    # calibration_frac (there is no static calibration slice).
    args = splits.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.test_frac == splits.DEFAULT_TEST_FRAC


def test_parse_args_overrides():
    # Each flag overrides its default and is passed through verbatim.
    args = splits.parse_args(
        ["--version", "v2", "--output-dir", "/tmp/out", "--test-frac", "0.2"]
    )
    assert args.version == "v2"
    assert args.output_dir == "/tmp/out"
    assert args.test_frac == 0.2


# --------------------------------------------------------------------------
# main — end to end
# --------------------------------------------------------------------------


def test_main_end_to_end(tmp_path, caplog):
    # Seeding a 40-match matches.parquet and running main: splits.parquet
    # appears beside it with only {"train", "test"} split values, the
    # expected 34/6 counts, and a summary log line carrying both counts.
    caplog.set_level(logging.INFO)
    df = _dated_matches(40)
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    df.to_parquet(version_dir / "matches.parquet", index=False)

    rc = splits.main(["--output-dir", str(tmp_path), "--version", "v1"])
    assert rc == 0

    written = pd.read_parquet(version_dir / "splits.parquet")
    assert set(written["split"]) == {"train", "test"}
    assert len(written[written["split"] == "train"]) == 34
    assert len(written[written["split"] == "test"]) == 6
    assert "34 train" in caplog.text
    assert "6 test" in caplog.text
