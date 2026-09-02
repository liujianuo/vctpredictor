"""Tests for the stage-isolation evaluation (M34).

Covers the local ``"Bo<N>"`` parser (fourth copy) and the played-map
order derivation on a hand-built ``SampledVetoSequence`` fixture, the
Arm-A table builder (split restriction, join correctness, empty-split
``ValueError``), the Arm-B sampler (same-seed reproducibility /
different-seed divergence, degenerate zero-probability ``ValueError``,
truncation-to-``n_played`` on a swept-Bo3 fixture, ``n_played``
validation), a fully deterministic Arm-A-vs-Arm-B scoring run whose
per-position scores and resulting gap are hand-verified against
``utils.scoring``'s own primitives, the report builder's row-alignment
guard and gap arithmetic, the missing-identities guard, and a skip-
guarded real-``data/v1`` integration test wiring the fitted M20/M27/M28
artifacts over a small slice of the test split.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.stage_isolation import (
    _parse_best_of,
    _played_maps_in_order,
    build_actual_played_maps,
    build_stage_isolation_report,
    sample_predicted_map_identities,
    score_actual_played_maps,
    score_predicted_played_maps,
)
from models.ancestral_veto_sampler import (
    SampledVetoAction,
    SampledVetoSequence,
)
from utils import scoring

# The 7-map pool all three ACTION_SEQUENCES walk, matching
# config.json's 2026-abyss era in ascending name order (the same POOL
# constant tests/test_ancestral_veto_sampler.py uses).
POOL = ("Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset")

# The as-of cutoff for the synthetic stub runs.
QUERY_DATE = "2026-01-06T00:00:00"

# A small hand-built league for the scoring tests: m1 (Bo3, test,
# swept — only 2 of 3 maps played), m2 (Bo3, test, swept — 1 map),
# m3 (Bo3, train — held out of the evaluation).
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo3", "status": "completed"},
]

# Played maps per match, with decisive (never tied/null) scorelines so
# the tables stay consistent with the repo's labels convention.
_MAP_ROWS = [
    {"match_id": "m1", "map_index": 0, "map_name": "Haven",
     "team1_score": 13, "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Lotus",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 0, "map_name": "Split",
     "team1_score": 13, "team2_score": 9, "winner": "A"},
    {"match_id": "m3", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 5, "winner": "C"},
]

# True per-map ordinals (A-reg=0, A-OT=1, B-OT=2, B-reg=3).
_LABEL_ROWS = [
    {"match_id": "m1", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
    {"match_id": "m1", "map_index": 1, "outcome_label": "B-regulation",
     "outcome_ordinal": 3},
    {"match_id": "m2", "map_index": 0, "outcome_label": "A-OT",
     "outcome_ordinal": 1},
    {"match_id": "m3", "map_index": 0, "outcome_label": "A-regulation",
     "outcome_ordinal": 0},
]

_SPLIT_ROWS = [
    {"match_id": "m1", "split": "test"},
    {"match_id": "m2", "split": "test"},
    {"match_id": "m3", "split": "train"},
]


def _league_tables():
    """Build the synthetic matches/maps/labels/splits frames.

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df)`` tuple of
        ``pandas.DataFrame`` objects built from :data:`_MATCH_ROWS` /
        :data:`_MAP_ROWS` / :data:`_LABEL_ROWS` / :data:`_SPLIT_ROWS`.

    Raises:
        Nothing.
    """
    matches_df = pd.DataFrame(
        _MATCH_ROWS,
        columns=["match_id", "date", "team1_id", "team2_id", "best_of", "status"],
    )
    maps_df = pd.DataFrame(
        _MAP_ROWS,
        columns=["match_id", "map_index", "map_name", "team1_score", "team2_score", "winner"],
    )
    labels_df = pd.DataFrame(
        _LABEL_ROWS,
        columns=["match_id", "map_index", "outcome_label", "outcome_ordinal"],
    )
    splits_df = pd.DataFrame(_SPLIT_ROWS, columns=["match_id", "split"])
    return matches_df, maps_df, labels_df, splits_df


def _first_sorted_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """One-hot step predictor: always ban/pick the first sorted map.

    Returns ``[1.0, 0.0, ...]`` aligned to ``remaining_maps`` (which
    the sampler hands over already sorted), so every draw is
    deterministic regardless of the seed: each walk removes the
    alphabetically-first remaining map at every choosing step, and all
    ``n_samples`` walks are identical.

    Args:
        acting_team_id: The acting team's stable id (ignored).
        action: The step's action (ignored).
        remaining_maps: The sorted remaining-maps list the returned
            distribution aligns to.
        date: The as-of cutoff (ignored).
        matches_df: The materialised matches table (ignored).
        maps_df: The materialised maps table (ignored).

    Returns:
        A one-hot ``list`` of ``len(remaining_maps)`` probabilities
        with ``1.0`` on the first entry.

    Raises:
        Nothing.
    """
    probs = [0.0] * len(remaining_maps)
    probs[0] = 1.0
    return probs


def _uniform_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """Uniform step predictor over the remaining maps.

    Every remaining map gets ``1 / len(remaining_maps)``, so the
    sampler's draws are genuinely random: two seeds are expected to
    diverge (used by the determinism tests).

    Args:
        acting_team_id: The acting team's stable id (ignored).
        action: The step's action (ignored).
        remaining_maps: The sorted remaining-maps list the returned
            distribution aligns to.
        date: The as-of cutoff (ignored).
        matches_df: The materialised matches table (ignored).
        maps_df: The materialised maps table (ignored).

    Returns:
        A ``list`` of ``len(remaining_maps)`` equal ``float``
        probabilities summing to approximately 1.

    Raises:
        Nothing.
    """
    n = len(remaining_maps)
    return [1.0 / n] * n


# The fixed per-map four-way vectors the deterministic map stub returns
# (each sums to 1 in OUTCOME_LABELS order: A-reg, A-OT, B-OT, B-reg).
# All four categories carry strictly positive probability so log loss
# is always well defined.
_MAP_FOUR_WAY = {
    "Haven": (0.55, 0.15, 0.10, 0.20),
    "Lotus": (0.20, 0.15, 0.35, 0.30),
    "Split": (0.40, 0.25, 0.15, 0.20),
}


def _fixed_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Deterministic four-way map stub from :data:`_MAP_FOUR_WAY`.

    Returns the fixed 4-vector for ``map_name`` regardless of teams,
    date, or the tables — a fully deterministic Stage-2 model so both
    arms' aggregates are hand-computable.

    Args:
        team1_id: The queried team1's stable id (ignored).
        team2_id: The queried team2's stable id (ignored).
        map_name: The map to predict for; must be a key of
            :data:`_MAP_FOUR_WAY`.
        date: The as-of cutoff (ignored).
        matches_df: The materialised matches table (ignored).
        maps_df: The materialised maps table (ignored).

    Returns:
        The fixed 4-tuple from :data:`_MAP_FOUR_WAY` for ``map_name``.

    Raises:
        KeyError: If ``map_name`` is not a key of
            :data:`_MAP_FOUR_WAY`.
    """
    return _MAP_FOUR_WAY[map_name]


def _hand_built_sample(best_of, actions, sequence_probability=0.5):
    """Build a hand-made :class:`SampledVetoSequence` for unit tests.

    Lets the played-map-order / truncation tests construct sequences
    directly (including deliberately malformed ones) without going
    through the sampler.

    Args:
        best_of: The sequence's ``"Bo<N>"`` string.
        actions: An iterable of :class:`SampledVetoAction` objects.
        sequence_probability: The sequence's probability (``0.5`` by
            default).

    Returns:
        A :class:`SampledVetoSequence` with teams ``"A"``/``"B"`` and
        :data:`QUERY_DATE`.

    Raises:
        Nothing.
    """
    return SampledVetoSequence(
        team_a_id="A",
        team_b_id="B",
        best_of=best_of,
        date=QUERY_DATE,
        actions=tuple(actions),
        sequence_probability=sequence_probability,
    )


def _ban(step_index, map_name):
    """Build one ``ban`` :class:`SampledVetoAction`.

    The acting team alternates by step parity (even -> "A", odd ->
    "B"), matching the sampler's convention.

    Args:
        step_index: The step's 0-based position.
        map_name: The banned map's name.

    Returns:
        A :class:`SampledVetoAction` with ``action="ban"`` and a
        ``0.5`` probability.

    Raises:
        Nothing.
    """
    return SampledVetoAction(
        step_index=step_index,
        team="A" if step_index % 2 == 0 else "B",
        action="ban",
        map_name=map_name,
        probability=0.5,
    )


def _pick(step_index, map_name):
    """Build one ``pick`` :class:`SampledVetoAction`.

    Mirrors :func:`_ban` for the pick action.

    Args:
        step_index: The step's 0-based position.
        map_name: The picked map's name.

    Returns:
        A :class:`SampledVetoAction` with ``action="pick"`` and a
        ``0.5`` probability.

    Raises:
        Nothing.
    """
    return SampledVetoAction(
        step_index=step_index,
        team="A" if step_index % 2 == 0 else "B",
        action="pick",
        map_name=map_name,
        probability=0.5,
    )


def _decider(step_index, map_name):
    """Build one ``decider`` :class:`SampledVetoAction`.

    The forced last step: ``team=None`` and ``probability=1.0`` per the
    sampler's decider convention.

    Args:
        step_index: The step's 0-based position (always the last).
        map_name: The sole remaining map.

    Returns:
        A :class:`SampledVetoAction` with ``action="decider"``.

    Raises:
        Nothing.
    """
    return SampledVetoAction(
        step_index=step_index,
        team=None,
        action="decider",
        map_name=map_name,
        probability=1.0,
    )


# --------------------------------------------------------------------------
# plan#1: the local best-of parser (fourth copy)
# --------------------------------------------------------------------------


def test_parse_best_of_exact_mappings():
    # The three v1 values plus other positive-odd "Bo<N>" strings parse
    # to the plain ints; malformed strings raise ValueError and
    # non-strings raise TypeError (identical behaviour to the three
    # earlier copies).
    assert _parse_best_of("Bo1") == 1
    assert _parse_best_of("Bo3") == 3
    assert _parse_best_of("Bo5") == 5
    assert _parse_best_of("Bo7") == 7
    for bad in ("Bo2", "Bo4", "BestOf3", "bo3", "Bo", "BoX", "", "3", "Bo0"):
        with pytest.raises(ValueError):
            _parse_best_of(bad)
    for bad in (3, None):
        with pytest.raises(TypeError):
            _parse_best_of(bad)


# --------------------------------------------------------------------------
# plan#1: played-map order from a sampled sequence (local reimplementation)
# --------------------------------------------------------------------------


def test_played_maps_in_order_all_formats():
    # The maps actually played, in play order, are the pick/decider
    # actions in ascending step order; bans are never played.
    bo1 = _hand_built_sample(
        "Bo1",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _ban(2, "Haven"),
         _ban(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )
    assert _played_maps_in_order(bo1) == ("Sunset",)

    bo3 = _hand_built_sample(
        "Bo3",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _pick(2, "Haven"),
         _pick(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )
    assert _played_maps_in_order(bo3) == ("Haven", "Lotus", "Sunset")

    bo5 = _hand_built_sample(
        "Bo5",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _pick(2, "Haven"),
         _pick(3, "Lotus"), _pick(4, "Split"), _pick(5, "Summit"),
         _decider(6, "Sunset")],
    )
    assert _played_maps_in_order(bo5) == (
        "Haven", "Lotus", "Split", "Summit", "Sunset"
    )


def test_played_maps_in_order_raises_on_count_mismatch():
    # A sequence whose pick/decider count does not equal the parsed
    # best_of map count is an internal desync and must raise loudly.
    too_few = _hand_built_sample(
        "Bo3",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _ban(2, "Haven"),
         _ban(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )
    with pytest.raises(ValueError, match="played map"):
        _played_maps_in_order(too_few)
    wrong_label = _hand_built_sample(
        "Bo5",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _pick(2, "Haven"),
         _pick(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )
    with pytest.raises(ValueError, match="out of sync"):
        _played_maps_in_order(wrong_label)


# --------------------------------------------------------------------------
# plan#2: the Arm-A table builder
# --------------------------------------------------------------------------


def test_build_actual_played_maps_split_restriction_and_join():
    # Only test-split matches appear (m3 is train), only actually-played
    # maps (m1 has 2, m2 has 1), each row carries the match's best_of
    # plus the true outcome_ordinal, in maps_df order.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    actual = build_actual_played_maps(matches_df, maps_df, labels_df, splits_df)
    assert list(actual.columns) == [
        "match_id", "map_index", "date", "team1_id", "team2_id",
        "map_name", "best_of", "outcome_ordinal",
    ]
    assert len(actual) == 3
    assert actual["match_id"].tolist() == ["m1", "m1", "m2"]
    assert actual["map_index"].tolist() == [0, 1, 0]
    assert actual["map_name"].tolist() == ["Haven", "Lotus", "Split"]
    assert actual["best_of"].tolist() == ["Bo3", "Bo3", "Bo3"]
    assert actual["outcome_ordinal"].tolist() == [0, 3, 1]
    assert actual["team1_id"].tolist() == ["A", "A", "A"]
    assert actual["team2_id"].tolist() == ["B", "B", "B"]
    assert actual["date"].tolist() == [
        "2026-01-01T00:00:00", "2026-01-01T00:00:00", "2026-01-02T00:00:00"
    ]


def test_build_actual_played_maps_respects_split_argument():
    # An explicit split="train" yields only the train match's maps.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    actual = build_actual_played_maps(
        matches_df, maps_df, labels_df, splits_df, split="train"
    )
    assert actual["match_id"].tolist() == ["m3"]
    assert actual["map_name"].tolist() == ["Bind"]


def test_build_actual_played_maps_raises_on_empty_split():
    # A split value with no maps (all match ids present in splits_df
    # but none labelled with the requested value) yields an empty
    # restricted table and must raise loudly (the M19 precedent) —
    # distinct from the stale-dataset guard, which fires first when a
    # match id is absent from splits_df entirely.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    all_train = splits_df.replace({"test": "train"})
    with pytest.raises(ValueError, match="empty"):
        build_actual_played_maps(
            matches_df, maps_df, labels_df, all_train, split="test"
        )


# --------------------------------------------------------------------------
# plan#3: the Arm-B map-identity sampler
# --------------------------------------------------------------------------


def test_sampler_truncates_to_n_played_on_swept_bo3():
    # m1 is a swept Bo3: only n_played=2 positions actually happened,
    # so the sampler must return exactly positions 0 and 1 (the
    # un-played decider slot is dropped). With the one-hot predictor
    # every walk is identical (Haven, Lotus, Sunset), so every
    # position's pairs are the same map with weight 1.0 each.
    matches_df, maps_df, _, _ = _league_tables()
    identities = sample_predicted_map_identities(
        "m1", "A", "B", "Bo3", "2026-01-01T00:00:00",
        matches_df, maps_df,
        {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
        n_samples=6,
        rng=np.random.default_rng(7),
        n_played=2,
        map_pool=POOL,
    )
    assert set(identities) == {0, 1}
    for position in (0, 1):
        pairs = identities[position]
        assert len(pairs) == 6
        # All samples identical -> all weight 1/6, single map name.
        assert {name for name, _ in pairs} == {"Haven" if position == 0 else "Lotus"}
        assert sum(weight for _, weight in pairs) == pytest.approx(1.0)


def test_sampler_same_seed_reproducible_different_seeds_diverge():
    # Under the genuinely-random uniform step predictor, identical
    # (seed, inputs) must reproduce byte-identical identities across
    # two separate calls, while two different seeds are expected to
    # diverge.
    matches_df, maps_df, _, _ = _league_tables()
    kwargs = {
        "match_id": "m1",
        "team1_id": "A",
        "team2_id": "B",
        "best_of": "Bo3",
        "date": "2026-01-01T00:00:00",
        "matches_df": matches_df,
        "maps_df": maps_df,
        "predictor_fn_by_action": {
            "ban": _uniform_stub, "pick": _uniform_stub
        },
        "n_samples": 40,
        "n_played": 2,
        "map_pool": POOL,
    }
    run1 = sample_predicted_map_identities(
        rng=np.random.default_rng(11), **kwargs
    )
    run2 = sample_predicted_map_identities(
        rng=np.random.default_rng(11), **kwargs
    )
    run3 = sample_predicted_map_identities(
        rng=np.random.default_rng(12), **kwargs
    )
    assert run1 == run2
    assert run1 != run3


def test_sampler_raises_on_degenerate_zero_probability(monkeypatch):
    # A degenerate all-zero-probability sample set makes the weighted
    # aggregation undefined; the guard must raise naming the match.
    import evaluation.stage_isolation as si

    def zero_prob_samples(*args, **kwargs):
        """Stub sampler returning all-zero-probability sequences.

        Args:
            args / kwargs: The sampler's arguments (ignored).

        Returns:
            Three zero-probability :class:`SampledVetoSequence`
            objects.

        Raises:
            Nothing.
        """
        return [
            SampledVetoSequence("A", "B", "Bo3", QUERY_DATE, (), 0.0)
            for _ in range(3)
        ]

    monkeypatch.setattr(si, "sample_veto_sequences", zero_prob_samples)
    matches_df, maps_df, _, _ = _league_tables()
    with pytest.raises(ValueError, match="m1"):
        sample_predicted_map_identities(
            "m1", "A", "B", "Bo3", QUERY_DATE,
            matches_df, maps_df,
            {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
            n_samples=3,
            rng=np.random.default_rng(0),
            n_played=2,
            map_pool=POOL,
        )


def test_sampler_raises_when_played_count_below_n_played(monkeypatch):
    # A full veto sequence always has best_of played maps, so an
    # n_played exceeding a sample's played-map count is an internal
    # desync that must raise naming the match.
    import evaluation.stage_isolation as si

    too_short = _hand_built_sample(
        "Bo3",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _pick(2, "Haven"),
         _pick(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )

    def short_samples(*args, **kwargs):
        """Stub sampler returning one too-short Bo3 sequence.

        Args:
            args / kwargs: The sampler's arguments (ignored).

        Returns:
            A single Bo3 sequence with exactly 3 played maps.

        Raises:
            Nothing.
        """
        return [too_short]

    monkeypatch.setattr(si, "sample_veto_sequences", short_samples)
    matches_df, maps_df, _, _ = _league_tables()
    with pytest.raises(ValueError, match="m1"):
        sample_predicted_map_identities(
            "m1", "A", "B", "Bo3", QUERY_DATE,
            matches_df, maps_df,
            {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
            n_samples=1,
            rng=np.random.default_rng(0),
            n_played=4,
            map_pool=POOL,
        )


def test_sampler_rejects_non_positive_n_played():
    # n_played must be a positive integer (validated via
    # models._shared._validate_positive_int).
    matches_df, maps_df, _, _ = _league_tables()
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="n_played"):
            sample_predicted_map_identities(
                "m1", "A", "B", "Bo3", QUERY_DATE,
                matches_df, maps_df,
                {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
                n_samples=2,
                rng=np.random.default_rng(0),
                n_played=bad,
                map_pool=POOL,
            )


# --------------------------------------------------------------------------
# plan#4/#5: Arm A vs Arm B scoring with hand-verified numbers
# --------------------------------------------------------------------------


def test_arm_scoring_and_report_hand_computed():
    # Fully deterministic setup: one-hot predictor (every sampled walk
    # identical: bans Abyss/Ascent/Split/Summit, picks Haven/Lotus,
    # decider Sunset) and the fixed map model. Arm A scores the actual
    # maps (m1: Haven@0, Lotus@3; m2: Split@1). Arm B scores Stage 1's
    # predicted maps at the same positions (m1 pos0: Haven, pos1:
    # Lotus; m2 pos0: Haven — the decider Sunset slot of m1 is dropped
    # because n_played=2). The per-position metrics are hand-computed
    # from utils.scoring's primitives below, so both arms' aggregates
    # and the gap block are exactly verifiable.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    actual = build_actual_played_maps(matches_df, maps_df, labels_df, splits_df)

    predicted: dict = {}
    rng = np.random.default_rng(3)
    for match_id, group in actual.groupby("match_id", sort=False):
        row0 = group.iloc[0]
        identities = sample_predicted_map_identities(
            match_id, row0.team1_id, row0.team2_id, row0.best_of, row0.date,
            matches_df, maps_df,
            {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
            n_samples=4,
            rng=rng,
            n_played=len(group),
            map_pool=POOL,
        )
        for position, pairs in identities.items():
            predicted[(match_id, position)] = pairs

    scored_actual = score_actual_played_maps(
        _fixed_map_model_fn, actual, matches_df, maps_df
    )
    scored_predicted = score_predicted_played_maps(
        _fixed_map_model_fn, actual, predicted, matches_df, maps_df
    )
    # Identical shape/order in both arms (one row per held-out
    # position).
    assert scored_actual["match_id"].tolist() == ["m1", "m1", "m2"]
    assert scored_predicted["match_id"].tolist() == ["m1", "m1", "m2"]
    assert scored_actual["map_index"].tolist() == [0, 1, 0]
    assert scored_predicted["map_index"].tolist() == [0, 1, 0]

    # Hand-computed per-position scores from utils.scoring's primitives
    # on the known four-way vectors:
    haven = _MAP_FOUR_WAY["Haven"]
    lotus = _MAP_FOUR_WAY["Lotus"]
    split = _MAP_FOUR_WAY["Split"]
    # Arm A positions: Haven@0, Lotus@3, Split@1.
    actual_rps = [
        scoring.rps(haven, 0),
        scoring.rps(lotus, 3),
        scoring.rps(split, 1),
    ]
    actual_ll = [
        scoring.log_loss(haven, 0),
        scoring.log_loss(lotus, 3),
        scoring.log_loss(split, 1),
    ]
    actual_acc = [
        scoring.marginal_binary_accuracy(haven, 0),
        scoring.marginal_binary_accuracy(lotus, 3),
        scoring.marginal_binary_accuracy(split, 1),
    ]
    # Arm B positions: Haven@0, Lotus@3, Haven@1 (m2 pos0 queried with
    # Stage 1's predicted map Haven instead of the actual Split).
    predicted_rps = [
        scoring.rps(haven, 0),
        scoring.rps(lotus, 3),
        scoring.rps(haven, 1),
    ]
    predicted_ll = [
        scoring.log_loss(haven, 0),
        scoring.log_loss(lotus, 3),
        scoring.log_loss(haven, 1),
    ]
    predicted_acc = [
        scoring.marginal_binary_accuracy(haven, 0),
        scoring.marginal_binary_accuracy(lotus, 3),
        scoring.marginal_binary_accuracy(haven, 1),
    ]
    # Per-arm rows' score columns match the hand-computed per-position
    # values exactly (identity per row).
    assert scored_actual["rps"].tolist() == pytest.approx(actual_rps)
    assert scored_actual["log_loss"].tolist() == pytest.approx(actual_ll)
    assert scored_actual["marginal_correct"].tolist() == actual_acc
    assert scored_predicted["rps"].tolist() == pytest.approx(predicted_rps)
    assert scored_predicted["log_loss"].tolist() == pytest.approx(predicted_ll)
    assert scored_predicted["marginal_correct"].tolist() == predicted_acc

    report = build_stage_isolation_report(scored_actual, scored_predicted)
    expected_actual_block = {
        "n_eval": 3,
        "mean_rps": sum(actual_rps) / 3,
        "mean_log_loss": sum(actual_ll) / 3,
        "marginal_binary_accuracy": sum(actual_acc) / 3,
    }
    expected_predicted_block = {
        "n_eval": 3,
        "mean_rps": sum(predicted_rps) / 3,
        "mean_log_loss": sum(predicted_ll) / 3,
        "marginal_binary_accuracy": sum(predicted_acc) / 3,
    }
    assert report["actual_played_maps"] == pytest.approx(expected_actual_block)
    assert report["m29_predicted_maps"] == pytest.approx(expected_predicted_block)
    # Every gap is exactly m29_predicted_maps - actual_played_maps.
    assert report["gap"]["mean_rps_gap"] == pytest.approx(
        expected_predicted_block["mean_rps"] - expected_actual_block["mean_rps"]
    )
    assert report["gap"]["mean_log_loss_gap"] == pytest.approx(
        expected_predicted_block["mean_log_loss"]
        - expected_actual_block["mean_log_loss"]
    )
    assert report["gap"]["marginal_binary_accuracy_gap"] == pytest.approx(
        expected_predicted_block["marginal_binary_accuracy"]
        - expected_actual_block["marginal_binary_accuracy"]
    )
    # Spot-check the hand numbers: the Arm-B m2 position (Haven@1) is
    # the only row differing between arms, so the mean_rps gap must be
    # (rps(haven,1) - rps(split,1)) / 3.
    assert report["gap"]["mean_rps_gap"] == pytest.approx(
        (scoring.rps(haven, 1) - scoring.rps(split, 1)) / 3
    )
    assert report["gap"]["mean_log_loss_gap"] == pytest.approx(
        (scoring.log_loss(haven, 1) - scoring.log_loss(split, 1)) / 3
    )
    # Every value is a plain number: the report is json-serializable.
    json.dumps(report)


def test_report_raises_on_row_misalignment():
    # Two scored tables describing different positions (here: the same
    # rows reordered) must raise — a misaligned comparison would
    # silently pair different positions' scores.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    actual = build_actual_played_maps(matches_df, maps_df, labels_df, splits_df)
    scored_actual = score_actual_played_maps(
        _fixed_map_model_fn, actual, matches_df, maps_df
    )
    reordered = scored_actual.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="row-aligned"):
        build_stage_isolation_report(scored_actual, reordered)


def test_report_raises_on_row_count_mismatch():
    # Two scored tables of different lengths must raise.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    actual = build_actual_played_maps(matches_df, maps_df, labels_df, splits_df)
    scored_actual = score_actual_played_maps(
        _fixed_map_model_fn, actual, matches_df, maps_df
    )
    truncated = scored_actual.iloc[:-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="row counts"):
        build_stage_isolation_report(scored_actual, truncated)


def test_score_predicted_raises_on_missing_identities():
    # Every held-out position must have sampled identities; a missing
    # key is a sampling/scoring desync and must raise naming the
    # position.
    matches_df, maps_df, labels_df, splits_df = _league_tables()
    actual = build_actual_played_maps(matches_df, maps_df, labels_df, splits_df)
    with pytest.raises(ValueError, match="no predicted map identities"):
        score_predicted_played_maps(
            _fixed_map_model_fn, actual, {}, matches_df, maps_df
        )


# --------------------------------------------------------------------------
# plan#8: integration with real fitted models (skip-guarded)
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the real v1 tables and model artifacts exist.

    The skip guard for the real-data test: the materialised v1
    matches/maps/labels/splits tables plus the fitted ordinal-logit and
    ban/pick conditional-logit model artifacts must all be present
    (i.e. ``materialize.py`` / ``splits.py`` / ``labels.py`` and the
    model training drivers have been run).

    Returns:
        A bool: ``True`` iff every required ``data/v1`` file exists.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}").exists()
        for name in (
            "matches.parquet",
            "maps.parquet",
            "labels.parquet",
            "splits.parquet",
            "player_map_stats.parquet",
            "ordinal_logit_model.json",
            "conditional_logit_ban_model.json",
            "conditional_logit_pick_model.json",
        )
    )


def _real_v1_fitted_pipeline():
    """Load the real fitted Stage-1/Stage-2 models and tables from data/v1.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json`` and closes over
    ``player_map_stats.parquet`` via ``make_model_fn`` (the Stage-2
    ``map_model_fn``), and reconstructs the fitted M27/M28
    conditional-logit models from their artifacts via
    ``make_veto_step_predictor_fn`` (the ban/pick
    ``predictor_fn_by_action``). Also loads and returns the
    materialised v1 tables.

    Returns:
        A ``(map_model_fn, predictor_fn_by_action, matches_df, maps_df,
        labels_df, splits_df)`` tuple of the two pluggable callables
        and the four tables.

    Raises:
        FileNotFoundError: If an artifact is missing (the caller must
            skip-guard on :func:`_real_v1_available` first).
        ValueError / KeyError: Propagated from the ``from_dict`` /
            ``make_model_fn`` reconstruction calls.
    """
    from models.conditional_logit_ban import (
        from_dict as ban_from_dict,
    )
    from models.conditional_logit_ban import (
        make_veto_step_predictor_fn as make_ban_predictor_fn,
    )
    from models.conditional_logit_pick import (
        from_dict as pick_from_dict,
    )
    from models.conditional_logit_pick import (
        make_veto_step_predictor_fn as make_pick_predictor_fn,
    )
    from models.ordinal_logit import (
        from_dict as ordinal_from_dict,
    )
    from models.ordinal_logit import (
        make_model_fn as make_ordinal_model_fn,
    )

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")

    ordinal_model = ordinal_from_dict(
        json.loads(Path("data/v1/ordinal_logit_model.json").read_text(encoding="utf-8"))
    )
    map_model_fn = make_ordinal_model_fn(ordinal_model, player_map_stats_df)
    ban_model = ban_from_dict(
        json.loads(Path("data/v1/conditional_logit_ban_model.json").read_text(encoding="utf-8"))
    )
    pick_model = pick_from_dict(
        json.loads(Path("data/v1/conditional_logit_pick_model.json").read_text(encoding="utf-8"))
    )
    predictor_fn_by_action = {
        "ban": make_ban_predictor_fn(ban_model),
        "pick": make_pick_predictor_fn(pick_model),
    }
    return (
        map_model_fn,
        predictor_fn_by_action,
        matches_df,
        maps_df,
        labels_df,
        splits_df,
    )


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset / fitted model artifacts not present",
)
def test_real_v1_stage_isolation_end_to_end():
    # plan#8: wire the real fitted M20 ordinal model as map_model_fn
    # and the real fitted M27/M28 conditional-logit closures as the
    # ban/pick step predictors, run the full M34 pipeline on a small
    # slice of the real v1 test split (a couple of held-out matches,
    # keeping the smoke fast), and assert: both scored arms have the
    # identical row-aligned shape, every headline metric is finite and
    # in range, the gap arithmetic is arm-minus-arm, and the report is
    # json-serializable.
    (
        map_model_fn,
        predictor_fn_by_action,
        matches_df,
        maps_df,
        labels_df,
        splits_df,
    ) = _real_v1_fitted_pipeline()
    actual = build_actual_played_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    # A 4-position slice across the first held-out matches.
    slice_actual = actual.iloc[:4].reset_index(drop=True)

    scored_actual = score_actual_played_maps(
        map_model_fn, slice_actual, matches_df, maps_df
    )
    predicted: dict = {}
    rng = np.random.default_rng(2026)
    for match_id, group in slice_actual.groupby("match_id", sort=False):
        row0 = group.iloc[0]
        identities = sample_predicted_map_identities(
            match_id,
            row0.team1_id,
            row0.team2_id,
            row0.best_of,
            row0.date,
            matches_df,
            maps_df,
            predictor_fn_by_action,
            n_samples=2,
            rng=rng,
            n_played=len(group),
        )
        for position, pairs in identities.items():
            predicted[(match_id, position)] = pairs
    scored_predicted = score_predicted_played_maps(
        map_model_fn, slice_actual, predicted, matches_df, maps_df
    )

    assert scored_actual["match_id"].tolist() == scored_predicted["match_id"].tolist()
    assert scored_actual["map_index"].tolist() == scored_predicted["map_index"].tolist()

    report = build_stage_isolation_report(scored_actual, scored_predicted)
    for arm in ("actual_played_maps", "m29_predicted_maps"):
        block = report[arm]
        assert block["n_eval"] == len(slice_actual)
        assert 0.0 <= block["mean_rps"] <= 3.0
        assert math.isfinite(block["mean_rps"])
        assert block["mean_log_loss"] >= 0.0
        assert math.isfinite(block["mean_log_loss"])
        assert 0.0 <= block["marginal_binary_accuracy"] <= 1.0
    gap = report["gap"]
    assert gap["mean_rps_gap"] == pytest.approx(
        report["m29_predicted_maps"]["mean_rps"]
        - report["actual_played_maps"]["mean_rps"]
    )
    assert gap["mean_log_loss_gap"] == pytest.approx(
        report["m29_predicted_maps"]["mean_log_loss"]
        - report["actual_played_maps"]["mean_log_loss"]
    )
    assert gap["marginal_binary_accuracy_gap"] == pytest.approx(
        report["m29_predicted_maps"]["marginal_binary_accuracy"]
        - report["actual_played_maps"]["marginal_binary_accuracy"]
    )
    json.dumps(report)
