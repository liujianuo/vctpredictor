"""Tests for the P2 derived display quantities (``presentation/derived.py``).

Pure synthetic tests: every input is a directly-constructed dataclass
(:class:`drivers.predict.PerMapPrediction` /
:class:`drivers.predict.SeriesPrediction` /
:class:`drivers.predict.RankedVetoPrediction` /
:class:`drivers.predict.PredictionResult`), never a ``Predictor`` and
never real data, so no ``slow`` marker is needed — P2 has no
real-data surface. Each of the eight §4.4 rows is exercised with
hand-computed expected values, including the length-validation
``ValueError`` guards, the favourite-flip exactly-0.5 boundary rule,
the D4 structural (distinct-object, equal-field) containment check for
``greedy_rank``, and a parity test pinning ``p_a_wins_series`` to the
CLI ``main()`` formula in ``drivers/predict.py``.
"""

from dataclasses import dataclass

import pytest

from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    RankedVetoPrediction,
    SeriesPrediction,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from presentation import derived
from utils import series_paths


def _make_series(probabilities, outcome_order, best_of):
    """Build a synthetic :class:`SeriesPrediction` from raw vectors.

    Args:
        probabilities: The ``best_of + 1`` scoreline probabilities.
        outcome_order: The ``best_of + 1`` terminal ``(a_wins,
            b_wins)`` pairs, parallel to ``probabilities``.
        best_of: The parsed map count (``1``/``3``/``5``).

    Returns:
        A frozen :class:`SeriesPrediction` wrapping the two vectors
        (converted to tuples) and the map count.

    Raises:
        Nothing.
    """
    return SeriesPrediction(
        probabilities=tuple(probabilities),
        outcome_order=tuple(outcome_order),
        best_of=best_of,
    )


def _make_per_map(probabilities, *, interval_low=None, interval_high=None):
    """Build a synthetic :class:`PerMapPrediction` from a 4-vector.

    Args:
        probabilities: The four map-outcome probabilities in
            ``OUTCOME_LABELS`` order (A-regulation, A-OT, B-OT,
            B-regulation).
        interval_low: The four lower band endpoints, or ``None``
            meaning "not computed".
        interval_high: The four upper band endpoints, or ``None``
            alongside ``interval_low``.

    Returns:
        A frozen :class:`PerMapPrediction` named ``"Ascent"`` with
        ``n_games_backing`` 0 and the given probabilities/intervals.

    Raises:
        Nothing.
    """
    return PerMapPrediction(
        map_name="Ascent",
        probabilities=tuple(probabilities),
        interval_low=interval_low,
        interval_high=interval_high,
        n_games_backing=0,
    )


def _make_ranked_veto(predicted_veto, *, veto_probability=0.1):
    """Build a synthetic :class:`RankedVetoPrediction` entry.

    Wraps the given action sequence in a minimal
    :class:`PredictionResult` (empty ``per_map``, a dummy Bo1
    :class:`SeriesPrediction`, ``veto_sensitivity`` ``None``) so tests
    can exercise :func:`presentation.derived.coverage_mass` and
    :func:`presentation.derived.greedy_rank` without a ``Predictor``.

    Args:
        predicted_veto: The entry's veto action sequence (any iterable
            of four-field action records, e.g.
            :class:`SimulatedVetoAction`).
        veto_probability: The exact joint probability assigned to the
            entry.

    Returns:
        A frozen :class:`RankedVetoPrediction` whose ``result``
        carries the given sequence as its ``predicted_veto``.

    Raises:
        Nothing.
    """
    result = PredictionResult(
        predicted_veto=tuple(predicted_veto),
        per_map=(),
        series=_make_series((0.5, 0.5), ((1, 0), (0, 1)), 1),
        veto_sensitivity=None,
    )
    return RankedVetoPrediction(
        veto_probability=veto_probability,
        result=result,
    )


def _veto_sequence(map_name):
    """Build a synthetic 2-step veto sequence (ban + forced decider).

    A small shared builder for the ``greedy_rank`` tests: an A-team
    ban of ``map_name`` at step 0 followed by a team-less ``"Sunset"``
    decider at step 6, so a sequence exercises both a real team id and
    the ``team=None`` decider case. Every call constructs fresh
    :class:`SimulatedVetoAction` objects, so two calls with the same
    ``map_name`` are distinct objects with equal field values — the
    situation D4's structural comparison must match.

    Args:
        map_name: The map side A bans at step 0.

    Returns:
        A 2-tuple of frozen :class:`SimulatedVetoAction` instances
        ``(ban map_name at step 0, decider Sunset at step 6)``.

    Raises:
        Nothing.
    """
    return (
        SimulatedVetoAction(0, "A", "ban", map_name),
        SimulatedVetoAction(6, None, "decider", "Sunset"),
    )


@dataclass(frozen=True)
class _SyntheticRankedAction:
    """A richer action record than :class:`SimulatedVetoAction`.

    A frozen dataclass carrying the four fields
    :func:`presentation.derived.greedy_rank` projects onto plus an
    extra sampler-only ``probability`` field. Two of these are never
    ``==``-equal to a :class:`SimulatedVetoAction` with the same four
    values (different class and an extra field), so a listing entry
    built from them exercises the projection rather than plain
    equality.

    Attributes:
        step_index: The 0-based position of this action in the veto
            sequence.
        team: The acting team's stable id, or ``None`` for a decider
            action.
        action: One of ``"ban"``, ``"pick"`` or ``"decider"``.
        map_name: The chosen map's normalized name.
        probability: The sampler-only joint probability the projection
            deliberately drops.
    """

    step_index: int
    team: str | None
    action: str
    map_name: str
    probability: float


def test_p_a_wins_series_hand_computed_vectors():
    # Bo1/Bo3/Bo5 hand-computed sums: only scorelines where a_wins >
    # b_wins contribute. Bo3 (0.3, 0.25, 0.2, 0.25) -> 0.55 (first two);
    # Bo5 (0.1, 0.15, 0.2, 0.2, 0.2, 0.15) -> 0.45 (first three).
    bo1 = _make_series((0.6, 0.4), ((1, 0), (0, 1)), 1)
    assert derived.p_a_wins_series(bo1) == pytest.approx(0.6)

    bo3 = _make_series(
        (0.3, 0.25, 0.2, 0.25),
        ((2, 0), (2, 1), (1, 2), (0, 2)),
        3,
    )
    assert derived.p_a_wins_series(bo3) == pytest.approx(0.55)

    bo5 = _make_series(
        (0.1, 0.15, 0.2, 0.2, 0.2, 0.15),
        ((3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3)),
        5,
    )
    assert derived.p_a_wins_series(bo5) == pytest.approx(0.45)


def test_p_a_wins_series_length_mismatch_raises():
    # probabilities and outcome_order of different lengths must raise
    # ValueError rather than silently truncating via zip.
    series = _make_series((0.5, 0.3, 0.2), ((1, 0), (0, 1)), 1)
    with pytest.raises(ValueError):
        derived.p_a_wins_series(series)


def test_p_a_wins_series_matches_main_formula():
    # Parity test: the exported value must equal drivers/predict.py
    # main()'s inline formula (a plain sum over the same zip), so the
    # artifact number and the CLI log line cannot diverge.
    series = _make_series(
        (0.4, 0.2, 0.15, 0.25),
        ((2, 0), (2, 1), (1, 2), (0, 2)),
        3,
    )
    expected = sum(
        probability
        for probability, (a_wins, b_wins) in zip(
            series.probabilities, series.outcome_order
        )
        if a_wins > b_wins
    )
    assert derived.p_a_wins_series(series) == expected


def test_scoreline_labels_bo1_bo3_bo5():
    # Real outcome_order values (via series_outcome_order): labels are
    # exactly parallel and A-first, including the B-favoured "0-2".
    for best_of, expected in (
        (1, ("1-0", "0-1")),
        (3, ("2-0", "2-1", "1-2", "0-2")),
        (5, ("3-0", "3-1", "3-2", "2-3", "1-3", "0-3")),
    ):
        order = series_paths.series_outcome_order(best_of)
        labels = derived.scoreline_labels(order)
        assert labels == expected
        assert len(labels) == len(order)


def test_p_a_wins_map_and_p_overtime_collapse():
    # Hand-computed 4-vector: A-reg 0.5 + A-OT 0.1 = 0.6; A-OT 0.1 +
    # B-OT 0.1 = 0.2.
    per_map = _make_per_map((0.5, 0.1, 0.1, 0.3))
    assert derived.p_a_wins_map(per_map) == pytest.approx(0.6)
    assert derived.p_overtime(per_map) == pytest.approx(0.2)


def test_p_a_wins_map_wrong_length_raises():
    # A non-length-4 vector must raise ValueError (the dataclass
    # annotation is not enforced at runtime).
    with pytest.raises(ValueError):
        derived.p_a_wins_map(_make_per_map((0.5, 0.5, 0.0)))


def test_p_overtime_wrong_length_raises():
    # Same guard as p_a_wins_map: a non-length-4 vector raises.
    with pytest.raises(ValueError):
        derived.p_overtime(_make_per_map((0.5, 0.5, 0.0)))


def test_coverage_mass_multi_entry_and_empty():
    # Sum of veto_probability over the listing; empty listing -> 0.0.
    entries = (
        _make_ranked_veto(_veto_sequence("Haven"), veto_probability=0.3),
        _make_ranked_veto(_veto_sequence("Split"), veto_probability=0.2),
    )
    assert derived.coverage_mass(entries) == pytest.approx(0.5)
    empty_mass = derived.coverage_mass(())
    assert empty_mass == 0.0
    assert isinstance(empty_mass, float)


def test_favorite_flips_directions_and_boundaries():
    # Strictly opposite sides of 0.5 flip; same side does not; exactly
    # 0.5 on either side never flips.
    assert derived.favorite_flips(0.4, 0.6) is True
    assert derived.favorite_flips(0.6, 0.4) is True
    assert derived.favorite_flips(0.4, 0.4) is False
    assert derived.favorite_flips(0.6, 0.6) is False
    assert derived.favorite_flips(0.5, 0.6) is False
    assert derived.favorite_flips(0.6, 0.5) is False
    assert derived.favorite_flips(0.5, 0.4) is False
    assert derived.favorite_flips(0.5, 0.5) is False


def test_greedy_rank_at_rank_1():
    # The greedy sequence matches the first entry -> rank 1.
    greedy = _veto_sequence("Haven")
    top = (
        _make_ranked_veto(greedy),
        _make_ranked_veto(_veto_sequence("Split")),
    )
    assert derived.greedy_rank(greedy, top) == 1


def test_greedy_rank_at_rank_3():
    # The greedy sequence matches the third entry -> rank 3.
    greedy = _veto_sequence("Haven")
    top = (
        _make_ranked_veto(_veto_sequence("Split")),
        _make_ranked_veto(_veto_sequence("Bind")),
        _make_ranked_veto(greedy),
    )
    assert derived.greedy_rank(greedy, top) == 3


def test_greedy_rank_matches_distinct_equal_objects():
    # D4 correctness: the greedy sequence and the ranked entry's
    # sequence are built as *distinct* objects with equal field values
    # (including a team=None decider step), so the structural
    # comparison must match without relying on object identity.
    greedy = _veto_sequence("Haven")
    entry_actions = _veto_sequence("Haven")
    assert greedy[0] is not entry_actions[0]
    assert greedy[1] is not entry_actions[1]
    top = (_make_ranked_veto(entry_actions),)
    assert derived.greedy_rank(greedy, top) == 1


def test_greedy_rank_projection_ignores_extra_fields_and_type():
    # D4 projection: the ranked-entry side uses a richer action type
    # (extra probability field, different class) with the same four
    # projected values; plain == would fail (different class + extra
    # field) but the four-field projection matches -> rank 1.
    greedy = _veto_sequence("Haven")
    entry_actions = (
        _SyntheticRankedAction(0, "A", "ban", "Haven", 0.25),
        _SyntheticRankedAction(6, None, "decider", "Sunset", 0.25),
    )
    top = (_make_ranked_veto(entry_actions),)
    assert derived.greedy_rank(greedy, top) == 1


def test_greedy_rank_absent_and_empty():
    # No match -> None; empty listing -> None (never a fabricated 0).
    greedy = _veto_sequence("Haven")
    top = (_make_ranked_veto(_veto_sequence("Split")),)
    assert derived.greedy_rank(greedy, top) is None
    assert derived.greedy_rank(greedy, ()) is None


def test_intervals_present_cases():
    # All-null -> False; one interval_low -> True; one interval_high
    # only -> True; empty iterable -> False.
    all_null = (_make_per_map((0.5, 0.1, 0.1, 0.3)),)
    assert derived.intervals_present(all_null) is False

    low = (
        _make_per_map(
            (0.5, 0.1, 0.1, 0.3), interval_low=(0.3, 0.0, 0.0, 0.2)
        ),
    )
    assert derived.intervals_present(low) is True

    high = (
        _make_per_map(
            (0.5, 0.1, 0.1, 0.3), interval_high=(0.7, 0.3, 0.3, 0.5)
        ),
    )
    assert derived.intervals_present(high) is True

    assert derived.intervals_present(()) is False
