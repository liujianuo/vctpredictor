"""Tests for the shared training-set assembly module (roadmap M20/M24).

Covers the M24 addition to ``drivers/training_data.py``:
``assemble_out_of_fold_eta_rows`` on a small synthetic multi-fold
fixture (>= 2 effective walk-forward folds). The assertions encode the
leakage contract of decision C: the assembled OOF table's ``match_id``
set must be a strict subset of the train-split ids and disjoint from
the test-split ids, no warm-up-block match may appear, the coverage
dict's ``covered_matches`` must match the row-derived distinct-id
count, and every row's per-fold thresholds must be strictly increasing
(sanity on the per-fold refit). ``assemble_design_matrix``'s original
M20/M21 behaviour is already exercised by the training-driver tests in
``test_ordinal_logit.py`` / ``test_multinomial_logit.py``, so this file
only tests the new OOF-assembly path.
"""

import numpy as np
import pandas as pd
import pytest

from drivers import training_data
from evaluation import harness
from models import _shared
from utils import splits

_TEAM_NAMES = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
}


def _oof_fixture(n_train: int = 28, n_test: int = 5):
    """Build the synthetic multi-fold league tables for the OOF tests.

    ``n_train`` matches dated earliest (the walk-forward region) plus
    ``n_test`` matches dated latest (the ``"test"`` split), each a
    single finished map on Haven with a non-null winner string,
    cycling team pairs (team ids ``A``..``F`` with display names) and
    cycling outcome ordinals ``0..3``, and a two-player
    ``player_map_stats`` roster per side per match. The fixture is
    sized so the default walk-forward configuration
    (``n_folds=5, min_fold_block=8``) yields exactly 2 effective folds
    (``min(5, 28 // 8 - 1) == 2``) — the plan's ">= 2 effective folds"
    requirement.

    Args:
        n_train: Number of ``"train"`` matches (default 28).
        n_test: Number of ``"test"`` matches (default 5).

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df,
        player_map_stats_df)`` tuple of well-formed synthetic tables:
        matches carry ``match_id/date/team1_id/team2_id/team1_name/
        team2_name/event_name/status``, maps carry ``match_id/
        map_index/map_name/team1_score/team2_score/winner``, labels
        carry ``match_id/map_index/outcome_ordinal`` (cycling
        ``0..3``), splits come from
        :func:`utils.splits.split_matches` (so the earliest ``n_train``
        matches are ``"train"`` and the rest ``"test"``), and
        ``player_map_stats`` carries ``match_id/map_index/player_name/
        team_name/acs/rating``.

    Raises:
        Nothing (the fixture is static and well-formed).
    """
    match_rows = []
    map_rows = []
    label_rows = []
    pms_rows = []
    total = n_train + n_test
    for i in range(total):
        match_id = f"m{i:03d}"
        date = pd.Timestamp("2026-01-01T10:00:00") + pd.Timedelta(days=i)
        team1 = f"{chr(ord('A') + (i % 6))}"
        team2 = f"{chr(ord('A') + ((i + 1) % 6))}"
        match_rows.append(
            {
                "match_id": match_id,
                "date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "team1_id": team1,
                "team2_id": team2,
                "team1_name": _TEAM_NAMES[team1],
                "team2_name": _TEAM_NAMES[team2],
                "event_name": "VCT 2026: EMEA Stage 1",
                "status": "completed",
            }
        )
        # A close-but-finished 13-8 scoreline; win/loss is derived from
        # scores, never from the winner string, so any non-null winner
        # works as the completion signal.
        map_rows.append(
            {
                "match_id": match_id,
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": _TEAM_NAMES[team1],
            }
        )
        label_rows.append(
            {
                "match_id": match_id,
                "map_index": 0,
                "outcome_ordinal": i % 4,
            }
        )
        for side, team in ((1, team1), (2, team2)):
            for player_idx in range(2):
                pms_rows.append(
                    {
                        "match_id": match_id,
                        "map_index": 0,
                        "player_name": f"p{team}{i}_{player_idx}",
                        "team_name": _TEAM_NAMES[team],
                        "acs": 200.0 + i,
                        "rating": 1.1 + 0.01 * i,
                    }
                )

    matches_df = pd.DataFrame(match_rows)
    maps_df = pd.DataFrame(map_rows)
    labels_df = pd.DataFrame(label_rows)
    player_map_stats_df = pd.DataFrame(pms_rows)
    splits_df = splits.split_matches(matches_df)
    return (
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )


def test_out_of_fold_rows_are_train_only_and_leak_free():
    # Decision C's leakage contract: every assembled OOF row's match_id
    # must come from the train split (a strict subset — walk-forward
    # never validates the warm-up block) and must be disjoint from the
    # test split (touching the test split would contaminate M19's final
    # evaluation). The assembled table also carries the fold_id column
    # prepended by the assembler.
    (
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    ) = _oof_fixture()
    oof_df, _coverage = training_data.assemble_out_of_fold_eta_rows(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )
    train_ids = set(
        splits_df.loc[splits_df["split"] == "train", "match_id"]
    )
    test_ids = set(
        splits_df.loc[splits_df["split"] == "test", "match_id"]
    )
    oof_ids = set(oof_df["match_id"])
    assert oof_ids <= train_ids
    assert oof_ids < train_ids  # strict subset (warm-up block excluded)
    assert oof_ids.isdisjoint(test_ids)
    assert "fold_id" in oof_df.columns
    assert set(oof_df["fold_id"]) == {1, 2}


def test_out_of_fold_rows_exclude_warmup_block():
    # The walk-forward warm-up block (the earliest matches, which have
    # no strictly-prior training data) never receives an OOF prediction:
    # none of its match ids may appear in the assembled table, matching
    # the coverage dict's warmup_excluded_ids.
    (
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    ) = _oof_fixture()
    oof_df, coverage = training_data.assemble_out_of_fold_eta_rows(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )
    warmup_ids = set(coverage["warmup_excluded_ids"])
    assert len(warmup_ids) == coverage["warmup_excluded_count"]
    assert set(oof_df["match_id"]).isdisjoint(warmup_ids)
    # Sanity on the walk-forward geometry: 28 train matches, 2 folds,
    # remainder folded into the warm-up block -> 10 warm-up + 9 + 9.
    assert len(warmup_ids) == 10
    assert coverage["train_matches"] == 28


def test_coverage_covered_matches_matches_row_derived_count():
    # The coverage dict's covered_matches must agree with the
    # row-derived distinct match-id count of the assembled table (the
    # assembler's own accounting, cross-checked independently).
    (
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    ) = _oof_fixture()
    oof_df, coverage = training_data.assemble_out_of_fold_eta_rows(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )
    assert coverage["covered_matches"] == len(set(oof_df["match_id"]))
    # One map per match in this fixture, so rows == distinct matches.
    assert len(oof_df) == coverage["covered_matches"] == 18
    assert list(oof_df.columns) == [
        "fold_id",
        "match_id",
        "map_index",
        "eta",
        "theta1",
        "theta2",
        "theta3",
        "outcome_ordinal",
    ]


def test_each_row_thresholds_strictly_increasing():
    # Every OOF row carries its own fold model's thresholds; the
    # ordinal fit's softplus reparameterization must keep them strictly
    # increasing, so a sanity check per row locks the per-fold refit in.
    (
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    ) = _oof_fixture()
    oof_df, _ = training_data.assemble_out_of_fold_eta_rows(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )
    threshold_matrix = oof_df[["theta1", "theta2", "theta3"]].to_numpy()
    assert np.all(np.diff(threshold_matrix, axis=1) > 0.0)
    # Every eta must be finite (the dot product of a standardized row
    # with the fold model's coefficients).
    assert np.all(np.isfinite(oof_df["eta"].to_numpy()))


def test_out_of_fold_eta_rows_rejects_missing_train_split():
    # A splits table with no 'train' rows has no walk-forward region to
    # assemble from; it is a hard error, not an empty result.
    matches_df = pd.DataFrame(
        [
            {
                "match_id": "m000",
                "date": "2026-01-01T10:00:00",
                "team1_id": "A",
                "team2_id": "B",
                "team1_name": "Alpha",
                "team2_name": "Bravo",
                "event_name": "VCT 2026: EMEA Stage 1",
                "status": "completed",
            }
        ]
    )
    maps_df = pd.DataFrame(
        [
            {
                "match_id": "m000",
                "map_index": 0,
                "map_name": "Haven",
                "team1_score": 13,
                "team2_score": 8,
                "winner": "Alpha",
            }
        ]
    )
    labels_df = pd.DataFrame(
        [{"match_id": "m000", "map_index": 0, "outcome_ordinal": 0}]
    )
    splits_df = pd.DataFrame(
        [
            {"match_id": "m000", "date": "2026-01-01T10:00:00", "split": "test"},
            {"match_id": "m001", "date": "2026-01-02T10:00:00", "split": "test"},
        ]
    )
    empty_pms = pd.DataFrame(
        columns=["match_id", "map_index", "player_name", "team_name", "acs", "rating"]
    )
    with pytest.raises(ValueError, match="no 'train' rows"):
        training_data.assemble_out_of_fold_eta_rows(
            matches_df,
            maps_df,
            labels_df,
            splits_df,
            empty_pms,
        )


# --------------------------------------------------------------------------
# M36 (plan#2): assemble_bootstrap_design_matrix block-bootstrap resampler
# --------------------------------------------------------------------------

# The fixed draw of ``numpy.random.default_rng(7).choice(ids, 4,
# replace=True)`` over the four train match ids (verified against numpy
# when the fixture was built): m003 and m002 are each drawn twice, so
# the block-bootstrap property (each drawn match's *entire* row block
# appears together, repeats included) is directly exercised.
_BOOTSTRAP_SEED = 7
_BOOTSTRAP_DRAW = ("m003", "m002", "m002", "m003")


def _bootstrap_fixture():
    """Build the small synthetic league for the bootstrap-resampler tests.

    Six matches (m000..m005) dated one day apart, each with two
    finished maps (Haven then Bind, decisive 13-8 / 8-13 scorelines),
    cycling team pairs (ids ``A``..``F``) and cycling outcome ordinals
    (``mNNN`` map 0 -> ``(2*NNN) % 4``, map 1 -> ``(2*NNN + 1) % 4``, so
    the two maps of every match carry distinct ordinals and a drawn
    match's label pair is recognizable in the resampled ``y``), a
    two-player ``player_map_stats`` roster per side per match, and a
    hand-built splits table (``m000``..``m003`` train, ``m004``/
    ``m005`` test) via ``utils.splits.SPLITS_COLUMNS``/``SPLITS_DTYPES``
    (built by hand rather than :func:`utils.splits.split_matches` so the
    fixture can stay at 4 train matches — well below the 20-match floor
    that splitter enforces — keeping the feature-vector rebuild in each
    test call cheap).

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df,
        player_map_stats_df)`` tuple of well-formed synthetic tables,
        exactly the five tables
        :func:`drivers.training_data.assemble_bootstrap_design_matrix`
        consumes.

    Raises:
        Nothing (the fixture is static and well-formed).
    """
    _TEAM_NAMES = {
        "A": "Alpha",
        "B": "Bravo",
        "C": "Charlie",
        "D": "Delta",
        "E": "Echo",
        "F": "Foxtrot",
    }
    match_rows = []
    map_rows = []
    label_rows = []
    pms_rows = []
    for i in range(6):
        match_id = f"m{i:03d}"
        date = pd.Timestamp("2026-01-01T10:00:00") + pd.Timedelta(days=i)
        team1 = f"{chr(ord('A') + (i % 6))}"
        team2 = f"{chr(ord('A') + ((i + 1) % 6))}"
        match_rows.append(
            {
                "match_id": match_id,
                "date": date.strftime("%Y-%m-%dT%H:%M:%S"),
                "team1_id": team1,
                "team2_id": team2,
                "team1_name": _TEAM_NAMES[team1],
                "team2_name": _TEAM_NAMES[team2],
                "event_name": "VCT 2026: EMEA Stage 1",
                "status": "completed",
            }
        )
        for map_index, map_name, s1, s2 in ((0, "Haven", 13, 8), (1, "Bind", 8, 13)):
            map_rows.append(
                {
                    "match_id": match_id,
                    "map_index": map_index,
                    "map_name": map_name,
                    "team1_score": s1,
                    "team2_score": s2,
                    "winner": _TEAM_NAMES[team1] if s1 > s2 else _TEAM_NAMES[team2],
                }
            )
            label_rows.append(
                {
                    "match_id": match_id,
                    "map_index": map_index,
                    "outcome_ordinal": (i * 2 + map_index) % 4,
                }
            )
        for side, team in ((1, team1), (2, team2)):
            for player_idx in range(2):
                pms_rows.append(
                    {
                        "match_id": match_id,
                        "map_index": 0,
                        "player_name": f"p{team}{i}_{player_idx}",
                        "team_name": _TEAM_NAMES[team],
                        "acs": 200.0 + i,
                        "rating": 1.1 + 0.01 * i,
                    }
                )
    matches_df = pd.DataFrame(match_rows)
    maps_df = pd.DataFrame(map_rows)
    labels_df = pd.DataFrame(label_rows)
    player_map_stats_df = pd.DataFrame(pms_rows)
    split_rows = [
        {
            "match_id": f"m{i:03d}",
            "date": match_rows[i]["date"],
            "split": "train" if i < 4 else "test",
        }
        for i in range(6)
    ]
    splits_df = pd.DataFrame(
        split_rows, columns=splits.SPLITS_COLUMNS
    ).astype(splits.SPLITS_DTYPES)
    return (
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
    )


def _base_train_rows(fixture):
    """Return the base train-row table grouped by match, in appearance order.

    Re-derives the exact row table
    :func:`drivers.training_data.assemble_bootstrap_design_matrix`
    groups its resample over: ``evaluation.harness.build_held_out_maps``
    with ``split="train"``, grouped by ``match_id`` preserving
    first-appearance order. The returned dict lets the hand-computed
    test rebuild the resampled table from a known draw independently of
    the function under test.

    Args:
        fixture: The five-table tuple from :func:`_bootstrap_fixture`.

    Returns:
        A dict mapping each train ``match_id`` to its ``DataFrame`` row
        group (in the harness's row order), plus the ordered list of
        train match ids (first-appearance order) — returned as a
        ``(groups_by_match, match_ids)`` tuple.

    Raises:
        ValueError: If the train split is empty (propagated from
            :func:`evaluation.harness.build_held_out_maps`).
    """
    matches_df, maps_df, labels_df, splits_df, _pms_df = fixture
    rows = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="train"
    )
    groups = {
        match_id: group
        for match_id, group in rows.groupby("match_id", sort=False)
    }
    return groups, list(groups)


def test_bootstrap_matrix_same_seed_deterministic():
    # Same seed -> byte-identical resample -> byte-identical (X, y):
    # the determinism contract that makes a fixed --bootstrap-seed
    # reproduce byte-identical refit models.
    fixture = _bootstrap_fixture()
    X1, y1 = training_data.assemble_bootstrap_design_matrix(
        *fixture, np.random.default_rng(_BOOTSTRAP_SEED)
    )
    X2, y2 = training_data.assemble_bootstrap_design_matrix(
        *fixture, np.random.default_rng(_BOOTSTRAP_SEED)
    )
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    # The resample keeps the base train row count (4 match slots, 2
    # maps each) and the 11-feature width.
    assert X1.shape == (8, 11)
    assert y1.shape == (8,)


def test_bootstrap_matrix_different_seed_diverges():
    # Different seeds draw different match multisets, so the resampled
    # matrices must differ somewhere (the divergence contract: the
    # resample genuinely depends on the rng).
    fixture = _bootstrap_fixture()
    X7, _y7 = training_data.assemble_bootstrap_design_matrix(
        *fixture, np.random.default_rng(7)
    )
    X42, _y42 = training_data.assemble_bootstrap_design_matrix(
        *fixture, np.random.default_rng(42)
    )
    assert not np.array_equal(X7, X42)


def test_bootstrap_matrix_hand_computed_blocks():
    # The block-bootstrap-not-iid-row-bootstrap property, verified
    # against a fully hand-computed re-derivation: with seed 7 the draw
    # is (m003, m002, m002, m003) — each drawn match's ENTIRE row block
    # (both its maps) must appear together in draw order, repeats
    # included, never a match split across resample slots. The expected
    # (X, y) is rebuilt independently from the base tables by
    # concatenating the drawn matches' full row groups and recomputing
    # each row's feature vector from the ORIGINAL tables, and must equal
    # the function's output exactly.
    fixture = _bootstrap_fixture()
    matches_df, maps_df, labels_df, splits_df, pms_df = fixture

    # The draw itself is a cross-check: a fresh rng with the same seed
    # must reproduce the hardcoded draw.
    groups, match_ids = _base_train_rows(fixture)
    probe_rng = np.random.default_rng(_BOOTSTRAP_SEED)
    drawn = probe_rng.choice(
        np.asarray(match_ids), size=len(match_ids), replace=True
    )
    assert tuple(drawn) == _BOOTSTRAP_DRAW

    # Hand-built expected rows: concat each drawn match's full row
    # group in draw order, then re-derive every feature vector from the
    # original tables.
    expected_rows = pd.concat(
        [groups[match_id] for match_id in _BOOTSTRAP_DRAW], ignore_index=True
    )
    expected_X_rows = []
    expected_y = []
    for row in expected_rows.itertuples(index=False):
        expected_X_rows.append(
            _shared.build_feature_vector(
                row.team1_id,
                row.team2_id,
                row.map_name,
                row.date,
                matches_df,
                maps_df,
                pms_df,
            )
        )
        expected_y.append(int(row.outcome_ordinal))
    expected_X = np.asarray(expected_X_rows, dtype=float)
    expected_y = np.asarray(expected_y, dtype=int)

    X, y = training_data.assemble_bootstrap_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        pms_df,
        np.random.default_rng(_BOOTSTRAP_SEED),
    )
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)
    # The label sequence is exactly the drawn matches' label pairs in
    # draw order (m003 -> [2, 3], m002 -> [0, 1], repeated) — the
    # crisp block-contiguity statement.
    assert y.tolist() == [2, 3, 0, 1, 0, 1, 2, 3]


def test_bootstrap_matrix_labels_are_valid_ordinals():
    # Every resampled row's label must be a valid outcome ordinal and
    # the resample must preserve the base row count regardless of the
    # draw (all four matches present in the multiset).
    fixture = _bootstrap_fixture()
    _X, y = training_data.assemble_bootstrap_design_matrix(
        *fixture, np.random.default_rng(1234)
    )
    assert set(np.unique(y).tolist()) <= {0, 1, 2, 3}
    assert y.shape == (8,)
