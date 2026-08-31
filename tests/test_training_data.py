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
