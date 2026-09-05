"""Tests for P3 artifact reshaping + team-name resolution.

Pure synthetic tests: every input is either a directly-constructed
dataclass (:class:`drivers.predict.PredictionResult`,
:class:`drivers.predict.RankedVetoPrediction`,
:class:`drivers.predict.SeriesPrediction`,
:class:`drivers.predict.PerMapPrediction`,
:class:`drivers.predict.VetoSensitivity` and the veto-simulator action
records) or a small synthetic ``pandas.DataFrame`` — never a
``Predictor`` and never real data, so no ``slow`` marker and no
``_real_v1_available()`` guard are needed. The reshaped output is
exercised end-to-end: the seven P3-owned fixture keys, the D4 decider
null, the D5 rank-by-position, the D6/D7 guards, the D8 JSON-native
promise, and a full wire-contract round-trip that stubs the P4/P5
deferred keys locally (this file deliberately does not import
``tests/test_contract.py``'s private builders).
"""

import json

import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError

from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    RankedVetoPrediction,
    SeriesPrediction,
    VetoSensitivity,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from presentation import contract, derived, reshape
from utils import series_paths


def _make_series(probabilities, best_of, *, outcome_order=None):
    """Build a synthetic :class:`SeriesPrediction`.

    Args:
        probabilities: The ``best_of + 1`` scoreline probabilities.
        best_of: The parsed map count (``1``/``3``/``5``), used to
            pick the canonical ``outcome_order`` when ``outcome_order``
            is not given.
        outcome_order: An optional explicit ``(a_wins, b_wins)`` pair
            sequence overriding the canonical order (used by the D6
            mismatch test).

    Returns:
        A frozen :class:`SeriesPrediction` wrapping the probability
        vector (as a tuple), the canonical or explicit outcome order
        (as a tuple), and ``best_of``.

    Raises:
        Nothing.
    """
    if outcome_order is None:
        outcome_order = series_paths.series_outcome_order(best_of)
    return SeriesPrediction(
        probabilities=tuple(probabilities),
        outcome_order=tuple(outcome_order),
        best_of=best_of,
    )


def _make_per_map(probabilities, *, map_name="Ascent", interval_low=None,
                  interval_high=None, n_games_backing=0):
    """Build a synthetic :class:`PerMapPrediction`.

    Args:
        probabilities: The four map-outcome probabilities in
            ``OUTCOME_LABELS`` order (A-regulation, A-OT, B-OT,
            B-regulation).
        map_name: The played map's normalized name.
        interval_low: The four lower band endpoints, or ``None``
            meaning "not computed".
        interval_high: The four upper band endpoints, or ``None``
            alongside ``interval_low``.
        n_games_backing: The weaker side's as-of game count.

    Returns:
        A frozen :class:`PerMapPrediction` carrying the given fields.

    Raises:
        Nothing.
    """
    return PerMapPrediction(
        map_name=map_name,
        probabilities=tuple(probabilities),
        interval_low=interval_low,
        interval_high=interval_high,
        n_games_backing=n_games_backing,
    )


def _make_actions(first_map="Ascent"):
    """Build a synthetic 7-step veto sequence ending in a team-less decider.

    Args:
        first_map: The map side A bans at step 0 (varying this yields
            distinct sequences for the greedy-rank and ordering tests).

    Returns:
        A 7-tuple of :class:`SimulatedVetoAction` records in the Bo3
        shape ``ban, ban, pick, pick, ban, ban, decider``: team ids
        alternate ``"A"``/``"B"`` by step-index parity, and the final
        decider step carries ``team=None``.

    Raises:
        Nothing.
    """
    sequence = ("ban", "ban", "pick", "pick", "ban", "ban", "decider")
    maps = [first_map, "Haven", "Lotus", "Split", "Summit", "Abyss", "Sunset"]
    actions = []
    for index, action in enumerate(sequence):
        if action == "decider":
            team = None
        else:
            team = "A" if index % 2 == 0 else "B"
        actions.append(SimulatedVetoAction(index, team, action, maps[index]))
    return tuple(actions)


def _make_veto_sensitivity():
    """Build a synthetic :class:`VetoSensitivity` record.

    Returns:
        A frozen :class:`VetoSensitivity` with length-4 vectors and a
        scalar ``mean_band_width`` of ``0.2``.

    Raises:
        Nothing.
    """
    return VetoSensitivity(
        unweighted_band_low=(0.3, 0.2, 0.1, 0.0),
        unweighted_band_high=(0.5, 0.4, 0.3, 0.2),
        band_widths=(0.2, 0.2, 0.2, 0.2),
        mean_band_width=0.2,
        weighted_mean=(0.4, 0.3, 0.2, 0.1),
        weighted_variance=(0.01, 0.01, 0.01, 0.01),
    )


def _make_ranked_entry(predicted_veto, series, *, veto_probability=0.4,
                       per_map=()):
    """Build a synthetic :class:`RankedVetoPrediction` entry.

    Wraps the given sequence in a minimal inner
    :class:`PredictionResult` (``veto_sensitivity`` ``None``, an empty
    nested ``top_vetos``) so a listing entry is value-complete without
    a ``Predictor``.

    Args:
        predicted_veto: The entry's veto action sequence.
        series: The entry's exact-M30 :class:`SeriesPrediction`.
        veto_probability: The exact joint probability assigned to the
            entry.
        per_map: The entry's played-map records (defaults to empty).

    Returns:
        A frozen :class:`RankedVetoPrediction` whose ``result``
        carries the given sequence, series and per-map records.

    Raises:
        Nothing.
    """
    inner = PredictionResult(
        predicted_veto=tuple(predicted_veto),
        per_map=tuple(per_map),
        series=series,
        veto_sensitivity=None,
    )
    return RankedVetoPrediction(
        veto_probability=veto_probability, result=inner
    )


# Sentinel so a caller can pass ``veto_sensitivity=None`` explicitly to
# exercise the D7 guard, distinct from "use the default real record".
_UNSET = object()


def _make_result(*, predicted_veto=None, per_map=(), series=None,
                 veto_sensitivity=_UNSET, top_vetos=()):
    """Build a synthetic top-level :class:`PredictionResult`.

    Args:
        predicted_veto: The greedy veto sequence (defaults to
            :func:`_make_actions`).
        per_map: The greedy veto's played-map records (defaults to
            empty).
        series: The top-level M31 :class:`SeriesPrediction` (defaults
            to a Bo3 four-way distribution).
        veto_sensitivity: The top-level spread summary (defaults to a
            real :func:`_make_veto_sensitivity` record; pass ``None``
            explicitly to exercise the D7 guard).
        top_vetos: The ranked listing entries (defaults to empty).

    Returns:
        A frozen :class:`PredictionResult` carrying the given fields.

    Raises:
        Nothing.
    """
    if predicted_veto is None:
        predicted_veto = _make_actions()
    if series is None:
        series = _make_series((0.4, 0.3, 0.2, 0.1), 3)
    if veto_sensitivity is _UNSET:
        veto_sensitivity = _make_veto_sensitivity()
    return PredictionResult(
        predicted_veto=tuple(predicted_veto),
        per_map=tuple(per_map),
        series=series,
        veto_sensitivity=veto_sensitivity,
        top_vetos=tuple(top_vetos),
    )


def _team_names():
    """Build the synthetic ``team_id → display name`` mapping used by tests.

    Returns:
        The dict ``{"A": "Team A", "B": "Team B"}``.

    Raises:
        Nothing.
    """
    return {"A": "Team A", "B": "Team B"}


def _iter_nodes(value):
    """Yield every node in a nested value, depth-first.

    Args:
        value: Any value (dict, list, tuple, or scalar).

    Yields:
        Each node reachable from ``value``, starting with ``value``
        itself and then descending into dict values and list/tuple
        items.

    Raises:
        Nothing.
    """
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_nodes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_nodes(child)


def _assemble_artifact(core, best_of, best_of_int):
    """Wrap a reshaped fixture core in a complete, valid ``Artifact`` dict.

    Fills the P4/P5-deferred ``Fixture`` keys (and every
    ``Artifact``-level key) with locally-defined stub values so the
    seven-key reshaped core can be validated against the full wire
    schema — deliberately without importing
    ``tests/test_contract.py``'s private builders.

    Args:
        core: The partial dict returned by
            :func:`presentation.reshape.reshape_fixture_core` (the
            seven P3-owned ``Fixture`` keys).
        best_of: One of ``"Bo1"``, ``"Bo3"`` or ``"Bo5"``.
        best_of_int: The matching map count (``1``/``3``/``5``).

    Returns:
        A complete ``Artifact`` dict with a single fixture: the seven
        core keys plus the deferred ``match_id``/``event``/
        ``scheduled_at``/``best_of``/``best_of_int``/
        ``bo5_unvalidated``/``map_leverage``/``narrative`` stubs and
        the top-level ``generated_at``/``model_version``/
        ``dataset_version``/``knobs``/``intervals_present``/
        ``metrics`` keys.

    Raises:
        Nothing.
    """
    fixture = dict(core)
    fixture.update(
        {
            "match_id": "1",
            "event": "Example Event",
            "scheduled_at": "2026-08-28T12:00:00",
            "best_of": best_of,
            "best_of_int": best_of_int,
            "bo5_unvalidated": False,
            "map_leverage": [],
            "narrative": "",
        }
    )
    return {
        "generated_at": "2026-08-28T12:00:00",
        "model_version": "abc1234",
        "dataset_version": "v1",
        "knobs": {"n_samples": 30, "seed": 5, "ci_level": 0.9, "top_n": 10},
        "intervals_present": False,
        "fixtures": [fixture],
        "metrics": {},
    }


def test_build_team_name_map_scans_both_column_pairs():
    # Both team1 and team2 id/name column pairs contribute to the map.
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "team1_id": ["397", "399"],
            "team1_name": ["Team A", "Team C"],
            "team2_id": ["398", "400"],
            "team2_name": ["Team B", "Team D"],
        }
    )
    assert reshape.build_team_name_map(df) == {
        "397": "Team A",
        "398": "Team B",
        "399": "Team C",
        "400": "Team D",
    }


def test_build_team_name_map_skips_nan_id_and_name():
    # A NaN id row and a NaN name row are both skipped (no "nan" value
    # is ever mapped).
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "team1_id": ["397", float("nan"), "399"],
            "team1_name": ["Team A", "Team X", float("nan")],
            "team2_id": ["398", "400", "401"],
            "team2_name": ["Team B", "Team D", "Team E"],
        }
    )
    assert reshape.build_team_name_map(df) == {
        "397": "Team A",
        "398": "Team B",
        "400": "Team D",
        "401": "Team E",
    }


def test_build_team_name_map_rebrand_resolves_to_most_recent_date():
    # One id appears with two names; the most recent date wins.
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-06-01"],
            "team1_id": ["397", "397"],
            "team1_name": ["Old Name", "New Name"],
            "team2_id": ["398", "399"],
            "team2_name": ["Team B", "Team C"],
        }
    )
    assert reshape.build_team_name_map(df)["397"] == "New Name"


def test_build_team_name_map_empty_frame_gives_empty_map():
    # An empty frame yields an empty mapping (no column read before any
    # row).
    df = pd.DataFrame(
        columns=["team1_id", "team1_name", "team2_id", "team2_name"]
    )
    assert reshape.build_team_name_map(df) == {}


def test_build_team_name_map_missing_column_raises_keyerror():
    # A non-empty frame missing a required id column raises KeyError.
    df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team1_id": ["397"],
            "team1_name": ["Team A"],
            "team2_name": ["Team B"],
        }
    )
    with pytest.raises(KeyError):
        reshape.build_team_name_map(df)


def test_resolve_fixture_teams_emit_id_and_name():
    # team_a / team_b are {id, name} dicts resolved through the map.
    result = _make_result()
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert core["team_a"] == {"id": "A", "name": "Team A"}
    assert core["team_b"] == {"id": "B", "name": "Team B"}


def test_unknown_fixture_team_raises_unknown_team_error():
    # An unknown fixture-level id raises UnknownTeamError naming the id.
    result = _make_result()
    with pytest.raises(reshape.UnknownTeamError, match="team_id 'A'"):
        reshape.reshape_fixture_core(result, "A", "B", {"B": "Team B"})


def test_unknown_team_inside_ranked_entry_raises():
    # An id unknown only inside a ranked entry's sequence raises the
    # same error (the fixture teams themselves resolve fine).
    entry_actions = (
        SimulatedVetoAction(0, "C", "ban", "Ascent"),
        SimulatedVetoAction(6, None, "decider", "Sunset"),
    )
    entry = _make_ranked_entry(
        entry_actions, _make_series((0.2, 0.3, 0.3, 0.2), 3)
    )
    result = _make_result(top_vetos=(entry,))
    with pytest.raises(reshape.UnknownTeamError, match="'C'"):
        reshape.reshape_fixture_core(result, "A", "B", _team_names())


def test_decider_step_emits_null_team_and_name():
    # A team=None decider emits team: None / team_name: None; passing it
    # through an empty name map proves it is not handed to the resolver.
    decider_only = (SimulatedVetoAction(6, None, "decider", "Sunset"),)
    reshaped = reshape._reshape_veto_actions(decider_only, {})
    assert reshaped[0]["team"] is None
    assert reshaped[0]["team_name"] is None
    assert reshaped[0]["action"] == "decider"
    assert reshaped[0]["map_name"] == "Sunset"
    # A non-decider step still resolves to id + name.
    resolved = reshape._reshape_veto_actions(
        (SimulatedVetoAction(0, "A", "ban", "Ascent"),), _team_names()
    )
    assert resolved[0] == {
        "step_index": 0,
        "team": "A",
        "team_name": "Team A",
        "action": "ban",
        "map_name": "Ascent",
    }


def test_ranked_entries_omit_structurally_constant_fields():
    # No top_vetos or veto_sensitivity key appears anywhere under a
    # ranked entry (checked recursively over every nested dict).
    entries = (
        _make_ranked_entry(
            _make_actions("Ascent"),
            _make_series((0.2, 0.3, 0.3, 0.2), 3),
            veto_probability=0.5,
        ),
        _make_ranked_entry(
            _make_actions("Bind"),
            _make_series((0.1, 0.2, 0.4, 0.3), 3),
            veto_probability=0.3,
        ),
    )
    result = _make_result(top_vetos=entries)
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    for entry_wire in core["top_vetos"]:
        for node in _iter_nodes(entry_wire):
            if isinstance(node, dict):
                assert "top_vetos" not in node
                assert "veto_sensitivity" not in node


def test_outcome_order_hoisted_exactly_once():
    # With a listing of >=3 entries, "outcome_order" appears exactly
    # once in the serialized fixture core (the hoisted copy).
    entries = (
        _make_ranked_entry(
            _make_actions("Ascent"),
            _make_series((0.2, 0.3, 0.3, 0.2), 3),
            veto_probability=0.5,
        ),
        _make_ranked_entry(
            _make_actions("Bind"),
            _make_series((0.1, 0.2, 0.4, 0.3), 3),
            veto_probability=0.3,
        ),
        _make_ranked_entry(
            _make_actions("Haven"),
            _make_series((0.4, 0.3, 0.2, 0.1), 3),
            veto_probability=0.2,
        ),
    )
    result = _make_result(top_vetos=entries)
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert json.dumps(core).count("outcome_order") == 1


def test_rank_assigned_by_position_without_resorting():
    # Ranks are 1..N in the order the caller supplied; the deliberately
    # non-descending veto_probability values prove no re-sorting.
    entries = (
        _make_ranked_entry(
            _make_actions("Ascent"),
            _make_series((0.2, 0.3, 0.3, 0.2), 3),
            veto_probability=0.1,
        ),
        _make_ranked_entry(
            _make_actions("Bind"),
            _make_series((0.1, 0.2, 0.4, 0.3), 3),
            veto_probability=0.9,
        ),
        _make_ranked_entry(
            _make_actions("Haven"),
            _make_series((0.4, 0.3, 0.2, 0.1), 3),
            veto_probability=0.5,
        ),
    )
    result = _make_result(top_vetos=entries)
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert [entry["rank"] for entry in core["top_vetos"]] == [1, 2, 3]
    assert [entry["veto_probability"] for entry in core["top_vetos"]] == [
        0.1,
        0.9,
        0.5,
    ]


def test_per_map_null_intervals_stay_none():
    # Null intervals stay None (not a fabricated zero vector).
    pm = _make_per_map((0.5, 0.1, 0.1, 0.3))
    reshaped = reshape._reshape_per_map(pm)
    assert reshaped["interval_low"] is None
    assert reshaped["interval_high"] is None
    assert reshaped["probabilities"] == [0.5, 0.1, 0.1, 0.3]


def test_per_map_populated_intervals_become_lists():
    # Populated intervals become 4-element lists.
    pm = _make_per_map(
        (0.5, 0.1, 0.1, 0.3),
        interval_low=(0.3, 0.0, 0.0, 0.2),
        interval_high=(0.7, 0.3, 0.3, 0.5),
    )
    reshaped = reshape._reshape_per_map(pm)
    assert reshaped["interval_low"] == [0.3, 0.0, 0.0, 0.2]
    assert reshaped["interval_high"] == [0.7, 0.3, 0.3, 0.5]


def test_per_map_derived_collapses_match_p2():
    # p_a_wins_map / p_overtime equal derived's functions exactly.
    pm = _make_per_map((0.5, 0.1, 0.1, 0.3))
    reshaped = reshape._reshape_per_map(pm)
    assert reshaped["p_a_wins_map"] == derived.p_a_wins_map(pm)
    assert reshaped["p_overtime"] == derived.p_overtime(pm)


def test_favorite_flips_computed_against_overall():
    # One entry flips the favourite (its p_a_wins_series is on the
    # opposite side of 0.5 from the overall), one does not.
    overall = _make_series((0.6, 0.2, 0.1, 0.1), 3)
    flipping = _make_ranked_entry(
        _make_actions("Ascent"),
        _make_series((0.1, 0.1, 0.2, 0.6), 3),
    )
    non_flipping = _make_ranked_entry(
        _make_actions("Bind"),
        _make_series((0.5, 0.3, 0.1, 0.1), 3),
    )
    result = _make_result(series=overall, top_vetos=(flipping, non_flipping))
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert core["top_vetos"][0]["favorite_flips"] is True
    assert core["top_vetos"][1]["favorite_flips"] is False


def test_coverage_mass_equals_p2():
    # coverage_mass equals derived.coverage_mass over the same listing.
    entries = (
        _make_ranked_entry(
            _make_actions("Ascent"),
            _make_series((0.2, 0.3, 0.3, 0.2), 3),
            veto_probability=0.5,
        ),
        _make_ranked_entry(
            _make_actions("Bind"),
            _make_series((0.1, 0.2, 0.4, 0.3), 3),
            veto_probability=0.3,
        ),
    )
    result = _make_result(top_vetos=entries)
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert core["coverage_mass"] == derived.coverage_mass(entries)
    assert core["coverage_mass"] == pytest.approx(0.8)


def test_greedy_rank_populated_and_none():
    # greedy_rank is 1-based when the greedy sequence is in the listing
    # and None when it is not.
    in_listing = _make_ranked_entry(
        _make_actions("Ascent"), _make_series((0.2, 0.3, 0.3, 0.2), 3)
    )
    result = _make_result(top_vetos=(in_listing,))
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert core["overall"]["greedy_rank"] == 1

    result2 = _make_result(
        predicted_veto=_make_actions("Bind"),
        top_vetos=(in_listing,),
    )
    core2 = reshape.reshape_fixture_core(result2, "A", "B", _team_names())
    assert core2["overall"]["greedy_rank"] is None


def test_null_top_level_veto_sensitivity_raises():
    # D7: a null top-level veto_sensitivity raises ValueError (the
    # schema requires a real object there).
    result = _make_result(veto_sensitivity=None)
    with pytest.raises(ValueError, match="veto_sensitivity"):
        reshape.reshape_fixture_core(result, "A", "B", _team_names())


def test_ranked_outcome_order_mismatch_raises_naming_rank():
    # D6: a ranked entry whose outcome_order differs from the hoisted
    # copy raises ValueError naming the rank.
    bad = _make_series(
        (0.2, 0.3, 0.3, 0.2),
        3,
        outcome_order=((2, 0), (2, 1), (1, 2), (1, 1)),
    )
    entry = _make_ranked_entry(_make_actions("Ascent"), bad)
    result = _make_result(top_vetos=(entry,))
    with pytest.raises(ValueError, match="rank 1"):
        reshape.reshape_fixture_core(result, "A", "B", _team_names())


def test_output_is_json_native_with_no_tuples():
    # json.dumps succeeds and a recursive walk finds no tuple anywhere
    # in the emitted core (D8).
    entries = (
        _make_ranked_entry(
            _make_actions("Ascent"),
            _make_series((0.2, 0.3, 0.3, 0.2), 3),
            veto_probability=0.5,
        ),
        _make_ranked_entry(
            _make_actions("Bind"),
            _make_series((0.1, 0.2, 0.4, 0.3), 3),
            veto_probability=0.3,
        ),
    )
    result = _make_result(top_vetos=entries)
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert json.dumps(core)  # must not raise
    assert not any(
        isinstance(node, tuple) for node in _iter_nodes(core)
    )


def test_reshaped_core_validates_as_full_artifact():
    # Fill the deferred keys with stubs, wrap in a minimal Artifact,
    # and the full wire contract validates.
    entries = (
        _make_ranked_entry(
            _make_actions("Ascent"),
            _make_series((0.2, 0.3, 0.3, 0.2), 3),
            veto_probability=0.5,
        ),
        _make_ranked_entry(
            _make_actions("Bind"),
            _make_series((0.1, 0.2, 0.4, 0.3), 3),
            veto_probability=0.3,
        ),
    )
    result = _make_result(top_vetos=entries)
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    artifact = _assemble_artifact(core, "Bo3", 3)
    assert contract.validate_artifact(artifact) is None


def test_reshaped_ranked_entry_reinserting_omitted_fields_rejected():
    # Re-inserting "top_vetos" (or separately "veto_sensitivity") into a
    # ranked entry makes validation raise, proving additionalProperties
    # catches a reshaping regression.
    entry = _make_ranked_entry(
        _make_actions("Ascent"), _make_series((0.2, 0.3, 0.3, 0.2), 3)
    )
    result = _make_result(top_vetos=(entry,))
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())

    artifact = _assemble_artifact(core, "Bo3", 3)
    artifact["fixtures"][0]["top_vetos"][0]["top_vetos"] = []
    with pytest.raises(ValidationError, match="Additional properties"):
        contract.validate_artifact(artifact)

    artifact = _assemble_artifact(core, "Bo3", 3)
    artifact["fixtures"][0]["top_vetos"][0]["veto_sensitivity"] = None
    with pytest.raises(ValidationError, match="Additional properties"):
        contract.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("best_of", "best_of_int"),
    [("Bo1", 1), ("Bo3", 3), ("Bo5", 5)],
)
def test_round_trip_validates_across_bo_formats(best_of, best_of_int):
    # The hoisted outcome_order length and the overall + ranked-entry
    # series_probabilities lengths all agree with best_of_int + 1 in
    # every bo format, and the assembled artifact validates.
    count = best_of_int + 1
    series = _make_series([0.1] * count, best_of_int)
    entry = _make_ranked_entry(
        _make_actions("Ascent"), _make_series([0.1] * count, best_of_int)
    )
    result = _make_result(series=series, top_vetos=(entry,))
    core = reshape.reshape_fixture_core(result, "A", "B", _team_names())
    assert len(core["outcome_order"]) == count
    assert len(core["scoreline_labels"]) == count
    assert len(core["overall"]["series_probabilities"]) == count
    assert len(core["top_vetos"][0]["series_probabilities"]) == count
    artifact = _assemble_artifact(core, best_of, best_of_int)
    assert contract.validate_artifact(artifact) is None
