"""Tests for the veto-marginalised series prediction (M31).

Covers the local ``"Bo<N>"`` parser (third copy) and the played-map
order derivation on hand-built ``SampledVetoSequence`` fixtures for
Bo1/Bo3/Bo5 plus the count-mismatch guard, the four-way-to-binary
collapse's hand-computed values and wrong-length ``ValueError``, a
deterministic end-to-end run (one-hot step predictor + fixed per-map
four-way stub) against a hand-computed Bo3 aggregate, a Bo1 run
exercising the ban-only path, a stochastic two-sequence run with known
unequal ``sequence_probability`` values whose aggregate must match the
hand-computed probability-weighted expectation (distinguishing the
weighted form from the plain Monte Carlo mean), the
``make_series_model_fn`` closure shape and its direct acceptance by the
M33a harness, same-seed reproducibility / different-seed divergence,
the full ``ValueError`` matrix (wrong-length map-model output, zero
total sequence probability, malformed ``best_of``), and two skip-
guarded real-``data/v1`` tests wiring the fitted M20 ordinal model as
``map_model_fn`` and the fitted M27/M28 conditional-logit closures as
the step predictors: one direct integration run on a real held-out
match and one end-to-end smoke through the live M33a harness
(including a private multi-arm sanity check against the M32 flat
baseline arm — not M33b's headline report, which is out of scope).
"""

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.series_evaluation import (
    build_held_out_series,
    build_series_evaluation_report,
    build_series_multi_arm_report,
    flat_series_baseline_model,
    score_held_out_series,
)
from evaluation.veto_marginalized_series import (
    _collapse_to_binary_a_win,
    _parse_best_of,
    _played_maps_in_order,
    make_series_model_fn,
    predict_series_outcome_via_veto_marginalization,
)
from models.ancestral_veto_sampler import (
    SampledVetoAction,
    SampledVetoSequence,
)
from utils import scoring, series_paths

# The 7-map pool all three ACTION_SEQUENCES walk, matching
# config.json's 2026-abyss era in ascending name order (the same POOL
# constant tests/test_ancestral_veto_sampler.py uses).
POOL = ("Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset")

# The as-of cutoff for the synthetic stub runs: the predictors and the
# map model ignore the matches/maps tables entirely, so empty tables
# with the right columns are enough; a fixed date keeps calls
# reproducible.
QUERY_DATE = "2026-01-06T00:00:00"

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]

# The fixed per-map four-way vectors the deterministic map stub returns
# (each sums to 1 in OUTCOME_LABELS order: A-reg, A-OT, B-OT, B-reg).
# Collapsed A-win probabilities: Ascent 0.3, Haven 0.7, Lotus 0.3,
# Split 0.5, Sunset 0.5, Abyss 0.8, Summit 0.5.
_MAP_FOUR_WAY = {
    "Abyss": (0.4, 0.4, 0.1, 0.1),
    "Ascent": (0.1, 0.2, 0.3, 0.4),
    "Haven": (0.6, 0.1, 0.1, 0.2),
    "Lotus": (0.2, 0.1, 0.4, 0.3),
    "Split": (0.5, 0.0, 0.0, 0.5),
    "Summit": (0.25, 0.25, 0.25, 0.25),
    "Sunset": (0.5, 0.0, 0.0, 0.5),
}


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Wraps ``pd.DataFrame`` so every fixture produces the same column
    order/dtypes regardless of which subset of columns a given fixture
    actually needs.

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


def _empty_tables():
    """Return empty-but-well-typed matches/maps tables.

    The stub predictors and the stub map model ignore the tables
    entirely (they only need to accept the arguments), and the sampler
    is given an explicit ``map_pool``, so empty tables with the right
    columns are sufficient for every stub-driven test.

    Returns:
        A ``(matches_df, maps_df)`` tuple of empty frames with
        :data:`_MATCHES_COLS` / :data:`_MAPS_COLS`.

    Raises:
        Nothing.
    """
    return _matches_df([]), _maps_df([])


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


def _two_sequence_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """Two-sequence step predictor with known unequal probabilities.

    At the first ban step (the full 7-map pool is still in play) puts
    ``0.75`` on Haven and ``0.25`` on Abyss (zeros elsewhere), so the
    sampled sequences follow exactly one of two walks with known,
    unequal ``sequence_probability`` values (0.75 and 0.25). Every
    subsequent step is one-hot on the first sorted remaining map (via
    :func:`_first_sorted_stub`'s logic), keeping each walk fully
    deterministic once the first ban is drawn.

    Args:
        acting_team_id: The acting team's stable id (ignored).
        action: The step's action.
        remaining_maps: The sorted remaining-maps list the returned
            distribution aligns to.
        date: The as-of cutoff (ignored).
        matches_df: The materialised matches table (ignored).
        maps_df: The materialised maps table (ignored).

    Returns:
        A ``list`` of ``len(remaining_maps)`` probabilities: at the
        first ban step ``0.75``/``0.25`` on Haven/Abyss and ``0.0``
        elsewhere; one-hot on the first entry at every other step.

    Raises:
        Nothing.
    """
    remaining = sorted(remaining_maps)
    if action == "ban" and set(remaining) == set(POOL):
        return [
            0.75 if name == "Haven" else (0.25 if name == "Abyss" else 0.0)
            for name in remaining
        ]
    probs = [0.0] * len(remaining)
    probs[0] = 1.0
    return probs


def _fixed_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Deterministic four-way map stub from :data:`_MAP_FOUR_WAY`.

    Returns the fixed 4-vector for ``map_name`` regardless of teams,
    date, or the tables — a fully deterministic Stage-2 model so the
    pipeline's aggregates are hand-computable.

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
        KeyError: If ``map_name`` is not a key of :data:`_MAP_FOUR_WAY`.
    """
    return _MAP_FOUR_WAY[map_name]


def _wrong_length_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Four-way map stub returning a 3-vector (wrong length).

    Deliberately violates the :data:`MapOutcomeModelFn` contract so the
    pipeline's per-map length validation must reject it loudly.

    Args:
        team1_id / team2_id / map_name / date / matches_df / maps_df:
            The stub arguments; deliberately unused.

    Returns:
        A 3-element list — never the required 4 entries.

    Raises:
        Nothing.
    """
    return [0.25, 0.25, 0.5]


def _hand_built_sample(best_of, actions, sequence_probability=0.5):
    """Build a hand-made :class:`SampledVetoSequence` for unit tests.

    Lets the played-map-order tests construct sequences directly
    (including deliberately malformed ones) without going through the
    sampler.

    Args:
        best_of: The sequence's ``"Bo<N>"`` string.
        actions: An iterable of :class:`SampledVetoAction` objects.
        sequence_probability: The sequence's probability (``0.5`` by
            default; unused by the order derivation).

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
# plan#3: the local best-of parser (third copy)
# --------------------------------------------------------------------------


def test_parse_best_of_exact_mappings():
    # The three v1 values plus other positive-odd "Bo<N>" strings parse
    # to the plain ints utils.series_paths expects; malformed strings
    # raise ValueError and non-strings raise TypeError (identical
    # behaviour to the two earlier copies).
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
# plan#2: played-map order from a sampled sequence
# --------------------------------------------------------------------------


def test_played_maps_in_order_all_formats():
    # The maps actually played, in play order, are the pick/decider
    # actions in ascending step order; bans are never played. Bo1 has
    # only the decider, Bo3 two picks + decider, Bo5 four picks +
    # decider.
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
    assert _played_maps_in_order(bo5) == ("Haven", "Lotus", "Split", "Summit", "Sunset")


def test_played_maps_in_order_raises_on_count_mismatch():
    # A sequence whose pick/decider count does not equal the parsed
    # best_of map count is an internal desync and must raise loudly
    # rather than silently under/over-fill the probability vector.
    too_few = _hand_built_sample(
        "Bo3",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _ban(2, "Haven"),
         _ban(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )
    with pytest.raises(ValueError, match="played map"):
        _played_maps_in_order(too_few)
    # best_of string disagreeing with the action shape (Bo5 label on a
    # three-played-map walk).
    wrong_label = _hand_built_sample(
        "Bo5",
        [_ban(0, "Abyss"), _ban(1, "Ascent"), _pick(2, "Haven"),
         _pick(3, "Lotus"), _ban(4, "Split"), _ban(5, "Summit"),
         _decider(6, "Sunset")],
    )
    with pytest.raises(ValueError, match="out of sync"):
        _played_maps_in_order(wrong_label)


# --------------------------------------------------------------------------
# plan#4: the four-way -> binary collapse
# --------------------------------------------------------------------------


def test_collapse_to_binary_a_win_hand_computed():
    # P(A wins the map) = p_a_regulation + p_a_ot (indices 0 and 1 of
    # OUTCOME_LABELS order), hand-checked across the stub vectors.
    assert _collapse_to_binary_a_win((0.6, 0.1, 0.1, 0.2)) == pytest.approx(0.7)
    assert _collapse_to_binary_a_win((1.0, 0.0, 0.0, 0.0)) == pytest.approx(1.0)
    assert _collapse_to_binary_a_win((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)
    assert _collapse_to_binary_a_win((0.2, 0.2, 0.2, 0.2)) == pytest.approx(0.4)


def test_collapse_to_binary_a_win_rejects_wrong_length():
    # Only a 4-vector can be collapsed; any other length must raise
    # naming the actual length.
    for bad in ([0.25, 0.25, 0.25], [0.2, 0.2, 0.2, 0.2, 0.2], []):
        with pytest.raises(ValueError, match="probabilit"):
            _collapse_to_binary_a_win(bad)


# --------------------------------------------------------------------------
# plan#12: deterministic end-to-end (one-hot predictor + fixed map model)
# --------------------------------------------------------------------------


def test_deterministic_end_to_end_bo3_hand_computed():
    # The one-hot first-sorted predictor makes every walk identical
    # (bans Abyss, Ascent, Split, Summit; picks Haven, Lotus; decider
    # Sunset), so played maps are Haven (0.7), Lotus (0.3), Sunset
    # (0.5). Hand-computed Bo3 enumeration: (2,0)=0.21, (2,1)=0.29,
    # (1,2)=0.29, (0,2)=0.21. With all n_samples identical, the
    # weighted aggregate must equal that single-sample scoreline
    # exactly — a strong, cheap correctness check of the whole
    # pipeline.
    matches_df, maps_df = _empty_tables()
    prediction = predict_series_outcome_via_veto_marginalization(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        _fixed_map_model_fn,
        {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
        n_samples=10,
        rng=np.random.default_rng(42),
        map_pool=POOL,
    )
    assert prediction.best_of == 3
    assert prediction.outcome_order == ((2, 0), (2, 1), (1, 2), (0, 2))
    assert prediction.probabilities == pytest.approx(
        (0.21, 0.29, 0.29, 0.21), abs=1e-12
    )
    assert sum(prediction.probabilities) == pytest.approx(1.0)
    assert len(prediction.samples) == 10
    for sample in prediction.samples:
        assert sample.played_maps == ("Haven", "Lotus", "Sunset")
        assert sample.weight == pytest.approx(0.1)
        assert sample.per_map_win_prob == pytest.approx((0.7, 0.3, 0.5))
        expected_four_way = (
            _MAP_FOUR_WAY["Haven"],
            _MAP_FOUR_WAY["Lotus"],
            _MAP_FOUR_WAY["Sunset"],
        )
        assert len(sample.per_map_four_way) == 3
        for actual, expected in zip(sample.per_map_four_way, expected_four_way):
            assert actual == pytest.approx(expected)
        assert sample.scoreline_probabilities == pytest.approx(
            (0.21, 0.29, 0.29, 0.21), abs=1e-12
        )
    # The prediction satisfies the SeriesModelFn convention directly.
    assert prediction.as_tuple() == prediction.probabilities


def test_deterministic_end_to_end_bo1_ban_only():
    # Bo1 walks need only the "ban" key (six bans, then the forced
    # decider). The one-hot predictor bans Abyss..Summit in sorted
    # order, so the sole played map is Sunset (p_a = 0.5); the Bo1
    # scoreline vocabulary is ((1,0), (0,1)) and the aggregate must be
    # [0.5, 0.5].
    matches_df, maps_df = _empty_tables()
    prediction = predict_series_outcome_via_veto_marginalization(
        "A", "B", "Bo1", QUERY_DATE, matches_df, maps_df,
        _fixed_map_model_fn,
        {"ban": _first_sorted_stub},
        n_samples=7,
        rng=np.random.default_rng(1),
        map_pool=POOL,
    )
    assert prediction.best_of == 1
    assert prediction.outcome_order == ((1, 0), (0, 1))
    assert prediction.probabilities == pytest.approx((0.5, 0.5), abs=1e-12)
    for sample in prediction.samples:
        assert sample.played_maps == ("Sunset",)
        assert sample.per_map_four_way[0] == pytest.approx(_MAP_FOUR_WAY["Sunset"])


# --------------------------------------------------------------------------
# plan#12: stochastic two-sequence weighted average
# --------------------------------------------------------------------------


def test_stochastic_two_sequence_weighted_average():
    # The first ban draws Haven (0.75) or Abyss (0.25); thereafter the
    # walk is one-hot deterministic. So two sequences with known,
    # unequal sequence_probability values (0.75 and 0.25) are sampled:
    #   X (Haven banned): played Ascent(0.3), Lotus(0.3), Sunset(0.5)
    #     -> S_X = [0.09, 0.21, 0.21, 0.49];
    #   Y (Abyss banned): played Haven(0.7), Lotus(0.3), Sunset(0.5)
    #     -> S_Y = [0.21, 0.29, 0.29, 0.21].
    # The realized weighted aggregate is (0.75*k_X*S_X +
    # 0.25*k_Y*S_Y)/(0.75*k_X + 0.25*k_Y), which concentrates on the
    # theoretical expectation E[p*S]/E[p] with pi_X=0.75, pi_Y=0.25:
    #   E[p] = 0.75^2 + 0.25^2 = 0.625
    #   (2,0): 0.75*0.75*0.09 + 0.25*0.25*0.21 = 0.06375 / 0.625 = 0.102
    #   (2,1): 0.75*0.75*0.21 + 0.25*0.25*0.29 = 0.13625 / 0.625 = 0.218
    #   (1,2): 0.218 (symmetric)
    #   (0,2): 0.75*0.75*0.49 + 0.25*0.25*0.21 = 0.28875 / 0.625 = 0.462
    # This is *not* the plain Monte Carlo mean (0.75*S_X + 0.25*S_Y =
    # [0.12, 0.23, 0.23, 0.42]) — the weighted form is what the
    # roadmap specifies, and the tolerance (abs=0.01, ~10x the MC
    # noise at n=4000) separates the two.
    matches_df, maps_df = _empty_tables()
    n_samples = 4000
    prediction = predict_series_outcome_via_veto_marginalization(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        _fixed_map_model_fn,
        {"ban": _two_sequence_stub, "pick": _first_sorted_stub},
        n_samples=n_samples,
        rng=np.random.default_rng(2026),
        map_pool=POOL,
    )
    # Both sequences must be drawn, with the expected per-sequence
    # probabilities.
    sequence_keys = {s.played_maps for s in prediction.samples}
    assert ("Ascent", "Lotus", "Sunset") in sequence_keys
    assert ("Haven", "Lotus", "Sunset") in sequence_keys
    for sample in prediction.samples:
        if sample.played_maps == ("Ascent", "Lotus", "Sunset"):
            assert sample.sequence.sequence_probability == pytest.approx(0.75)
        else:
            assert sample.sequence.sequence_probability == pytest.approx(0.25)
    assert sum(s.weight for s in prediction.samples) == pytest.approx(1.0)
    assert prediction.probabilities == pytest.approx(
        (0.102, 0.218, 0.218, 0.462), abs=0.01
    )


# --------------------------------------------------------------------------
# plan#12: determinism (same seed identical, different seeds diverge)
# --------------------------------------------------------------------------


def test_same_seed_reproducible_different_seeds_diverge():
    # Under the genuinely-random uniform step predictor, identical
    # (seed, inputs) must reproduce byte-identical aggregates across
    # two separate calls, while two different seeds are expected to
    # diverge.
    matches_df, maps_df = _empty_tables()
    kwargs = {
        "team1_id": "A",
        "team2_id": "B",
        "best_of": "Bo3",
        "date": QUERY_DATE,
        "matches_df": matches_df,
        "maps_df": maps_df,
        "map_model_fn": _fixed_map_model_fn,
        "predictor_fn_by_action": {"ban": _uniform_stub, "pick": _uniform_stub},
        "n_samples": 30,
        "map_pool": POOL,
    }
    run1 = predict_series_outcome_via_veto_marginalization(
        rng=np.random.default_rng(7), **kwargs
    )
    run2 = predict_series_outcome_via_veto_marginalization(
        rng=np.random.default_rng(7), **kwargs
    )
    run3 = predict_series_outcome_via_veto_marginalization(
        rng=np.random.default_rng(8), **kwargs
    )
    assert run1.probabilities == run2.probabilities
    assert run1.samples == run2.samples
    assert run1.probabilities != run3.probabilities


# --------------------------------------------------------------------------
# plan#12: the SeriesModelFn adapter factory
# --------------------------------------------------------------------------


def test_make_series_model_fn_shape_and_harness_acceptance():
    # The returned closure must be exactly the SeriesModelFn shape (6
    # positional args in the fixed order, output length best_of_int +
    # 1) and be directly accepted by
    # evaluation.series_evaluation.score_held_out_series against a
    # small synthetic held-out table (mirrors M33a's own stub-based
    # tests before real data).
    model_fn = make_series_model_fn(
        _fixed_map_model_fn,
        {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
        n_samples=5,
        rng=np.random.default_rng(3),
        map_pool=POOL,
    )
    params = list(inspect.signature(model_fn).parameters)
    assert params == [
        "team1_id", "team2_id", "best_of", "date", "matches_df", "maps_df"
    ]

    # A small hand-built league: m1 (Bo3, test, A 2-1), m2 (Bo5, test,
    # A 3-2), m3 (Bo3, train).
    match_rows = [
        {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
         "team2_id": "B", "best_of": "Bo3", "status": "completed"},
        {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
         "team2_id": "B", "best_of": "Bo5", "status": "completed"},
        {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
         "team2_id": "D", "best_of": "Bo3", "status": "completed"},
    ]
    map_rows = [
        {"match_id": "m1", "map_index": 0, "map_name": "Bind",
         "team1_score": 13, "team2_score": 8, "winner": "A"},
        {"match_id": "m1", "map_index": 1, "map_name": "Haven",
         "team1_score": 13, "team2_score": 11, "winner": "A"},
        {"match_id": "m1", "map_index": 2, "map_name": "Split",
         "team1_score": 8, "team2_score": 13, "winner": "B"},
        {"match_id": "m2", "map_index": 0, "map_name": "Bind",
         "team1_score": 13, "team2_score": 8, "winner": "A"},
        {"match_id": "m2", "map_index": 1, "map_name": "Haven",
         "team1_score": 13, "team2_score": 9, "winner": "A"},
        {"match_id": "m2", "map_index": 2, "map_name": "Split",
         "team1_score": 13, "team2_score": 11, "winner": "A"},
        {"match_id": "m2", "map_index": 3, "map_name": "Ascent",
         "team1_score": 7, "team2_score": 13, "winner": "B"},
        {"match_id": "m2", "map_index": 4, "map_name": "Icebox",
         "team1_score": 9, "team2_score": 13, "winner": "B"},
        {"match_id": "m3", "map_index": 0, "map_name": "Bind",
         "team1_score": 13, "team2_score": 5, "winner": "C"},
        {"match_id": "m3", "map_index": 1, "map_name": "Haven",
         "team1_score": 13, "team2_score": 6, "winner": "C"},
    ]
    split_rows = [
        {"match_id": "m1", "split": "test"},
        {"match_id": "m2", "split": "test"},
        {"match_id": "m3", "split": "train"},
    ]
    matches_df = pd.DataFrame(
        match_rows, columns=["match_id", "date", "team1_id", "team2_id", "best_of", "status"]
    )
    maps_df = pd.DataFrame(
        map_rows,
        columns=["match_id", "map_index", "map_name", "team1_score", "team2_score", "winner"],
    )
    splits_df = pd.DataFrame(split_rows, columns=["match_id", "split"])

    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    scored = score_held_out_series(model_fn, held_out, matches_df, maps_df)
    assert len(scored) == 2  # m1 and m2 only (m3 is train)
    for row in scored.itertuples(index=False):
        probs = row.probabilities
        assert len(probs) == row.best_of_int + 1
        assert sum(probs) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert math.isfinite(row.rps)
    report = build_series_evaluation_report(scored)
    assert report["n_eval_total"] == 2
    json.dumps(report)


# --------------------------------------------------------------------------
# plan#10: the ValueError matrix
# --------------------------------------------------------------------------


def test_wrong_length_map_model_output_raises():
    # A map model returning anything other than a 4-vector must raise
    # naming the offending map and sample, before any collapse.
    matches_df, maps_df = _empty_tables()
    with pytest.raises(ValueError, match="Haven"):
        predict_series_outcome_via_veto_marginalization(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            _wrong_length_map_model_fn,
            {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
            n_samples=2,
            rng=np.random.default_rng(0),
            map_pool=POOL,
        )


def test_malformed_best_of_raises():
    # The local parser rejects a malformed "Bo<N>" string before any
    # sampling happens.
    matches_df, maps_df = _empty_tables()
    with pytest.raises(ValueError, match="Bo<N>"):
        predict_series_outcome_via_veto_marginalization(
            "A", "B", "BestOf3", QUERY_DATE, matches_df, maps_df,
            _fixed_map_model_fn,
            {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
            n_samples=1,
            rng=np.random.default_rng(0),
            map_pool=POOL,
        )


def test_zero_total_sequence_probability_raises(monkeypatch):
    # A degenerate all-zero-probability sample set makes the weighted
    # average undefined; the guard must raise naming the condition.
    # The sampler can never produce such a set with valid predictors
    # (every drawn probability is strictly positive), so the sampler
    # is monkeypatched here to return one, exercising the guard
    # directly.
    import evaluation.veto_marginalized_series as vms

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

    monkeypatch.setattr(vms, "sample_veto_sequences", zero_prob_samples)
    matches_df, maps_df = _empty_tables()
    with pytest.raises(ValueError, match="exactly 0.0"):
        vms.predict_series_outcome_via_veto_marginalization(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            _fixed_map_model_fn,
            {"ban": _first_sorted_stub, "pick": _first_sorted_stub},
            n_samples=3,
            rng=np.random.default_rng(0),
            map_pool=POOL,
        )


# --------------------------------------------------------------------------
# plan#13: integration with real fitted models (skip-guarded)
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the real v1 tables and model artifacts exist.

    The skip guard for the real-data tests: the materialised v1
    matches/maps/splits/player_map_stats tables plus the fitted
    ordinal-logit and ban/pick conditional-logit model artifacts must
    all be present (i.e. ``materialize.py`` / ``splits.py`` and the
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
            "splits.parquet",
            "player_map_stats.parquet",
            "ordinal_logit_model.json",
            "conditional_logit_ban_model.json",
            "conditional_logit_pick_model.json",
        )
    )


def _real_v1_fitted_pipeline():
    """Load the real fitted Stage-1/Stage-2 models from data/v1.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json`` and closes over
    ``player_map_stats.parquet`` via ``make_model_fn`` (the
    ``map_model_fn``), and reconstructs the fitted M27/M28
    conditional-logit models from their artifacts via
    ``make_veto_step_predictor_fn`` (the ban/pick ``predictor_fn_by_action``).
    Also loads and returns the materialised v1 tables.

    Returns:
        A ``(map_model_fn, predictor_fn_by_action, matches_df, maps_df,
        splits_df)`` tuple of the two pluggable callables and the three
        tables.

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
    return map_model_fn, predictor_fn_by_action, matches_df, maps_df, splits_df


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset / fitted model artifacts not present",
)
def test_real_fitted_models_integration():
    # plan#13: wire the real fitted M20 ordinal model as map_model_fn
    # and the real fitted M27/M28 conditional-logit closures as the
    # step predictors, run the full pipeline on one real held-out Bo3
    # match, and assert: the aggregate forms a valid simplex (accepted
    # by utils.scoring's own validation via rps), the per-sample
    # detail lengths line up (played_maps / per_map_four_way /
    # per_map_win_prob all length best_of, each four-way vector 4
    # long), and the weights sum to 1.0 within tolerance.
    (map_model_fn, predictor_fn_by_action, matches_df, maps_df, splits_df) = (
        _real_v1_fitted_pipeline()
    )
    held_out = build_held_out_series(matches_df, maps_df, splits_df)
    row = held_out.itertuples(index=False).__next__()
    prediction = predict_series_outcome_via_veto_marginalization(
        row.team1_id, row.team2_id, row.best_of, row.date,
        matches_df, maps_df, map_model_fn, predictor_fn_by_action,
        n_samples=6,
        rng=np.random.default_rng(99),
    )
    probs = prediction.probabilities
    assert prediction.best_of == row.best_of_int
    assert prediction.outcome_order == series_paths.series_outcome_order(row.best_of_int)
    assert len(probs) == row.best_of_int + 1
    assert sum(probs) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # utils.scoring's own simplex validation accepts the aggregate.
    scoring.rps(probs, row.outcome_index)
    assert sum(s.weight for s in prediction.samples) == pytest.approx(1.0)
    for sample in prediction.samples:
        assert len(sample.played_maps) == row.best_of_int
        assert len(sample.per_map_four_way) == row.best_of_int
        assert len(sample.per_map_win_prob) == row.best_of_int
        assert all(len(v) == 4 for v in sample.per_map_four_way)
        assert len(sample.scoreline_probabilities) == row.best_of_int + 1


# --------------------------------------------------------------------------
# plan#14: end-to-end smoke through the live M33a harness (skip-guarded)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset / fitted model artifacts not present",
)
def test_real_v1_harness_end_to_end():
    # plan#14: build make_series_model_fn(...) with the real fitted
    # models and run it through build_held_out_series +
    # score_held_out_series on the data/v1 test split; the resulting
    # report must be finite and well-formed. Optionally (and here,
    # additionally) build the multi-arm report against the M32 flat
    # baseline arm as a private sanity check that the two arms are
    # comparable end-to-end — this is *not* M33b's headline report (no
    # artifact is written), only a test-time proof that M31's model
    # function is consumable by the harness M33b will later use.
    (map_model_fn, predictor_fn_by_action, matches_df, maps_df, splits_df) = (
        _real_v1_fitted_pipeline()
    )
    # Score a 5-match slice of the real test split (the full 15-row
    # split would add ~a minute of fitted-model feature builds; the
    # slice exercises the identical harness path with the real models
    # and keeps the smoke fast).
    held_out = build_held_out_series(matches_df, maps_df, splits_df).head(5)
    model_fn = make_series_model_fn(
        map_model_fn,
        predictor_fn_by_action,
        n_samples=2,
        rng=np.random.default_rng(2026),
    )
    scored = score_held_out_series(model_fn, held_out, matches_df, maps_df)
    assert len(scored) == len(held_out)
    for row in scored.itertuples(index=False):
        probs = row.probabilities
        assert len(probs) == row.best_of_int + 1
        assert sum(probs) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert math.isfinite(row.rps)
    report = build_series_evaluation_report(scored)
    assert report["n_eval_total"] == len(scored)
    assert "Bo3" in report
    bo3 = report["Bo3"]
    assert math.isfinite(bo3["mean_rps"])
    assert 0.0 <= bo3["mean_rps"] <= 3.0
    assert math.isfinite(bo3["mean_log_loss"])
    assert bo3["mean_log_loss"] >= 0.0
    assert 0.0 <= bo3["marginal_binary_accuracy"] <= 1.0
    json.dumps(report)

    # Private sanity: the M31 arm and the M32 flat baseline arm score
    # the identical held-out rows, so the multi-arm report builds.
    baseline_scored = score_held_out_series(
        flat_series_baseline_model, held_out, matches_df, maps_df
    )
    multi = build_series_multi_arm_report(
        {"m31_veto_marginalized": scored, "m32_flat": baseline_scored},
        baseline_arm="m32_flat",
    )
    assert "m31_veto_marginalized" in multi
    assert "deltas_vs_m32_flat" in multi
    json.dumps(multi)
