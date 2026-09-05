"""Tests for the presentation wire data contract (roadmap P1).

Pure in-memory tests: no real cache/data directories and no network.
Synthetic minimal and maximal artifact dicts are validated against the
checked-in ``presentation/contract.schema.json``, deliberately-invalid
variants assert each contract rule is enforced, and a drift-guard test
cross-checks the Python-side TypedDict field sets against the schema's
declared properties so the two representations cannot silently diverge.
"""

import pytest
from jsonschema.exceptions import ValidationError

from presentation import contract


def _per_map(map_name="Ascent", with_intervals=False):
    """Build a valid ``PerMap`` wire dict.

    Args:
        map_name: The played map's normalized name.
        with_intervals: When True, populate ``interval_low`` /
            ``interval_high`` with length-4 number arrays; when False,
            leave both ``None`` ("not computed").

    Returns:
        A ``PerMap`` dict with the four fixed-order probabilities
        ``[0.5, 0.1, 0.1, 0.3]`` (A-regulation, A-OT, B-OT,
        B-regulation), ``n_games_backing`` 0, and ``p_a_wins_map`` /
        ``p_overtime`` derived consistently from those probabilities.

    Raises:
        Nothing.
    """
    entry = {
        "map_name": map_name,
        "probabilities": [0.5, 0.1, 0.1, 0.3],
        "interval_low": None,
        "interval_high": None,
        "n_games_backing": 0,
        "p_a_wins_map": 0.6,
        "p_overtime": 0.2,
    }
    if with_intervals:
        entry["interval_low"] = [0.3, 0.0, 0.0, 0.2]
        entry["interval_high"] = [0.7, 0.3, 0.3, 0.5]
    return entry


def _veto_action(
    step_index=0,
    action="ban",
    team="397",
    team_name="Team A",
    map_name="Ascent",
):
    """Build a valid ``VetoAction`` wire dict.

    Args:
        step_index: The 0-based position in the veto sequence.
        action: One of ``"ban"``, ``"pick"`` or ``"decider"``.
        team: The acting team's stable ``team_id``.
        team_name: The acting team's display name.
        map_name: The chosen map's normalized name.

    Returns:
        A ``VetoAction`` dict with the given values.

    Raises:
        Nothing.
    """
    return {
        "step_index": step_index,
        "team": team,
        "team_name": team_name,
        "action": action,
        "map_name": map_name,
    }


def _seven_actions():
    """Build the canonical 7-step Bo3 veto sequence.

    Returns:
        A list of seven ``VetoAction`` dicts in step order — the Bo3
        shape ``ban, ban, pick, pick, ban, ban, decider`` from
        ``models.greedy_veto_simulator.ACTION_SEQUENCES`` — with team
        ids alternating by step-index parity and the decider last.

    Raises:
        Nothing.
    """
    sequence = ("ban", "ban", "pick", "pick", "ban", "ban", "decider")
    maps = ("Ascent", "Haven", "Lotus", "Split", "Summit", "Abyss", "Sunset")
    return [
        _veto_action(
            step_index=index,
            action=action,
            team="397" if index % 2 == 0 else "398",
            team_name="Team A" if index % 2 == 0 else "Team B",
            map_name=maps[index],
        )
        for index, action in enumerate(sequence)
    ]


def _overall(greedy_rank=None, with_intervals=False, with_greedy_veto=False):
    """Build a valid ``OverallResult`` wire dict.

    Args:
        greedy_rank: The ``greedy_rank`` value (a number, or ``None``).
        with_intervals: Whether the per-map entries carry populated
            (non-null) interval vectors.
        with_greedy_veto: Whether ``greedy_veto`` carries the full
            7-step sequence (True) or an empty list (False).

    Returns:
        An ``OverallResult`` dict with a populated ``veto_sensitivity``
        block, one ``per_map`` entry, a length-4
        ``series_probabilities`` vector, and the requested
        ``greedy_veto`` / ``greedy_rank``.

    Raises:
        Nothing.
    """
    return {
        "series_probabilities": [0.4, 0.3, 0.2, 0.1],
        "p_a_wins_series": 0.7,
        "veto_sensitivity": {
            "unweighted_band_low": [0.3, 0.2, 0.1, 0.0],
            "unweighted_band_high": [0.5, 0.4, 0.3, 0.2],
            "band_widths": [0.2, 0.2, 0.2, 0.2],
            "mean_band_width": 0.2,
            "weighted_mean": [0.4, 0.3, 0.2, 0.1],
            "weighted_variance": [0.01, 0.01, 0.01, 0.01],
        },
        "per_map": [_per_map("Ascent", with_intervals)],
        "greedy_veto": _seven_actions() if with_greedy_veto else [],
        "greedy_rank": greedy_rank,
    }


def _ranked_veto(rank=1, with_intervals=True):
    """Build a valid ``RankedVeto`` wire dict.

    Args:
        rank: The 1-based rank in the top-N listing.
        with_intervals: Whether the per-map entries carry populated
            intervals.

    Returns:
        A ``RankedVeto`` dict with the full 7-action sequence, two
        ``per_map`` entries, an exact-M30 ``series_probabilities``
        vector, and — critically — no ``veto_sensitivity`` and no
        ``top_vetos`` key (the structurally-omitted fields).

    Raises:
        Nothing.
    """
    return {
        "rank": rank,
        "veto_probability": 0.4,
        "actions": _seven_actions(),
        "per_map": [
            _per_map("Ascent", with_intervals),
            _per_map("Haven", with_intervals),
        ],
        "series_probabilities": [0.2, 0.3, 0.3, 0.2],
        "p_a_wins_series": 0.5,
        "favorite_flips": False,
    }


def _leverage_entry():
    """Build a valid ``MapLeverage`` wire dict.

    Returns:
        A ``MapLeverage`` dict for one map.

    Raises:
        Nothing.
    """
    return {
        "map_name": "Haven",
        "p_played": 0.5,
        "p_a_given_played": 0.6,
        "p_a_given_not": 0.4,
        "swing": 0.2,
        "flips_favorite": False,
        "n_vetos_backing": 5,
    }


def _fixture(
    match_id="1",
    greedy_rank=None,
    with_intervals=False,
    with_greedy_veto=False,
    top_vetos=(),
    map_leverage=(),
    coverage_mass=0.0,
    narrative="",
):
    """Build a valid ``Fixture`` wire dict.

    Args:
        match_id: The fixture's match id string.
        greedy_rank: The ``overall.greedy_rank`` value (a number, or
            ``None``).
        with_intervals: Whether the per-map entries carry populated
            (non-null) interval vectors.
        with_greedy_veto: Whether ``overall.greedy_veto`` carries the
            full 7-step sequence (True) or an empty list (False).
        top_vetos: The ``top_vetos`` entries (defaults to empty).
        map_leverage: The ``map_leverage`` entries (defaults to empty).
        coverage_mass: The ``coverage_mass`` value.
        narrative: The ``narrative`` string.

    Returns:
        A ``Fixture`` dict with a Bo3 ``outcome_order`` /
        ``scoreline_labels`` pair, fixed team_a/team_b, and the
        requested overall/top_vetos/map_leverage/coverage_mass/
        narrative.

    Raises:
        Nothing.
    """
    return {
        "match_id": match_id,
        "event": "Example Event",
        "scheduled_at": "2026-08-28T12:00:00",
        "best_of": "Bo3",
        "best_of_int": 3,
        "bo5_unvalidated": False,
        "team_a": {"id": "397", "name": "Team A"},
        "team_b": {"id": "398", "name": "Team B"},
        "outcome_order": [[2, 0], [2, 1], [1, 2], [0, 2]],
        "scoreline_labels": ["2-0", "2-1", "1-2", "0-2"],
        "overall": _overall(
            greedy_rank=greedy_rank,
            with_intervals=with_intervals,
            with_greedy_veto=with_greedy_veto,
        ),
        "top_vetos": list(top_vetos),
        "coverage_mass": coverage_mass,
        "map_leverage": list(map_leverage),
        "narrative": narrative,
    }


def _minimal_artifact():
    """Build a minimal-but-valid artifact dict.

    Returns:
        An ``Artifact`` dict with one fixture: ``intervals_present``
        False, null intervals throughout, empty ``top_vetos`` /
        ``map_leverage`` / ``greedy_veto``, an empty ``narrative``,
        ``greedy_rank`` null, and an empty stub ``metrics`` object.

    Raises:
        Nothing.
    """
    return {
        "generated_at": "2026-08-28T12:00:00",
        "model_version": "abc1234",
        "dataset_version": "v1",
        "knobs": {"n_samples": 30, "seed": 5, "ci_level": 0.9, "top_n": 10},
        "intervals_present": False,
        "fixtures": [_fixture()],
        "metrics": {},
    }


def _maximal_artifact():
    """Build a maximal-but-valid artifact dict.

    Returns:
        An ``Artifact`` dict with two fixtures and every optional wire
        element populated: ``intervals_present`` True, populated
        interval vectors, a non-empty ``top_vetos`` (full 7-action
        sequence), a non-empty ``map_leverage``, the 7-step
        ``greedy_veto``, and ``greedy_rank`` as a number in the first
        fixture and ``None`` in the second.

    Raises:
        Nothing.
    """
    return {
        "generated_at": "2026-08-28T12:00:00",
        "model_version": "abc1234",
        "dataset_version": "v1",
        "knobs": {"n_samples": 30, "seed": 5, "ci_level": 0.9, "top_n": 10},
        "intervals_present": True,
        "fixtures": [
            _fixture(
                match_id="1",
                greedy_rank=3,
                with_intervals=True,
                with_greedy_veto=True,
                top_vetos=[_ranked_veto(rank=1, with_intervals=True)],
                map_leverage=[_leverage_entry()],
                coverage_mass=0.4,
                narrative="Team A is favoured unless the veto reaches Haven.",
            ),
            _fixture(
                match_id="2",
                greedy_rank=None,
                with_intervals=True,
                with_greedy_veto=True,
                top_vetos=[_ranked_veto(rank=1, with_intervals=True)],
                map_leverage=[_leverage_entry()],
                coverage_mass=0.4,
                narrative="",
            ),
        ],
        "metrics": {"log_loss": 0.8, "rps": 0.21},
    }


def _artifact_for_best_of(best_of, best_of_int):
    """Build a valid artifact for an arbitrary bo format with one top veto.

    Args:
        best_of: One of ``"Bo1"``, ``"Bo3"`` or ``"Bo5"`` — the
            fixture's bo label.
        best_of_int: The matching map count (``1``, ``3`` or ``5``).

    Returns:
        An ``Artifact`` dict whose single fixture uses the given bo
        format, with ``outcome_order``, ``scoreline_labels``, the
        overall ``series_probabilities`` and the single ranked veto's
        ``series_probabilities`` all at the correct ``best_of_int + 1``
        length (so the artifact is valid for that bo format).

    Raises:
        Nothing.
    """
    scorelines = [[best_of_int - b, b] for b in range(best_of_int + 1)]
    artifact = _minimal_artifact()
    fixture = artifact["fixtures"][0]
    fixture["best_of"] = best_of
    fixture["best_of_int"] = best_of_int
    fixture["outcome_order"] = scorelines
    fixture["scoreline_labels"] = [f"{a}-{b}" for a, b in scorelines]
    fixture["overall"]["series_probabilities"] = [0.1] * (best_of_int + 1)
    fixture["top_vetos"] = [_ranked_veto(rank=1)]
    fixture["top_vetos"][0]["series_probabilities"] = [0.1] * (best_of_int + 1)
    return artifact


def test_minimal_artifact_validates():
    # The minimal artifact (null intervals, empty lists, empty
    # narrative, stub metrics) is valid, and validate_artifact returns
    # None on success (its documented contract).
    assert contract.validate_artifact(_minimal_artifact()) is None


def test_maximal_artifact_validates():
    # The maximal artifact (populated intervals, top_vetos with the
    # full 7-action sequence, map_leverage, greedy_rank as a number and
    # as null across two fixtures) is valid.
    assert contract.validate_artifact(_maximal_artifact()) is None


def test_ranked_veto_with_veto_sensitivity_rejected():
    # A RankedVeto carrying a veto_sensitivity key must be rejected:
    # structurally-constant fields are omitted, not shipped null, and
    # additionalProperties:false enforces the omission.
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["veto_sensitivity"] = None
    with pytest.raises(ValidationError, match="Additional properties"):
        contract.validate_artifact(artifact)


def test_ranked_veto_with_top_vetos_rejected():
    # Same for a nested top_vetos key on a RankedVeto (G1's recorded
    # consequence: the inner list is always empty and stripped, so it
    # must be absent — not [], not null).
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["top_vetos"] = []
    with pytest.raises(ValidationError, match="Additional properties"):
        contract.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("probabilities", [0.5, 0.1, 0.1]),
        ("probabilities", [0.5, 0.1, 0.1, 0.2, 0.1]),
        ("interval_low", [0.3, 0.1]),
        ("interval_high", [0.3, 0.1, 0.1, 0.1, 0.2]),
    ],
)
def test_length_4_vector_wrong_length_rejected(field, bad):
    # Per-map probability/interval vectors are exactly length 4: both
    # under-length and over-length vectors are rejected.
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["per_map"][0][field] = bad
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)


def test_probability_shipped_as_string_rejected():
    # Floats are shipped as numbers, never formatted strings: a "30%"
    # string in a probability vector is rejected (every probability
    # field is typed "number" in the schema).
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["per_map"][0]["probabilities"][0] = (
        "30%"
    )
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)


def test_missing_required_top_level_field_rejected():
    # A missing required top-level field (metrics) is rejected — the
    # artifact is complete on day one per decision D, so every top-level
    # field is non-optional.
    artifact = _minimal_artifact()
    del artifact["metrics"]
    with pytest.raises(ValidationError, match="'metrics' is a required property"):
        contract.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("best_of", "Bo7"),
        ("best_of_int", 7),
        ("action", "skip"),
    ],
)
def test_enum_out_of_set_rejected(field, bad):
    # best_of / best_of_int / action are closed enums; a value outside
    # the allowed set is rejected.
    artifact = _maximal_artifact()
    if field == "action":
        artifact["fixtures"][0]["top_vetos"][0]["actions"][0]["action"] = bad
    else:
        artifact["fixtures"][0][field] = bad
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("best_of", "best_of_int", "expected"),
    [("Bo1", 1, 2), ("Bo3", 3, 4), ("Bo5", 5, 6)],
)
def test_series_probabilities_correct_length_validates(
    best_of, best_of_int, expected
):
    # Both the overall and the ranked-veto series_probabilities must be
    # exactly best_of_int + 1 long; a correct-length artifact validates
    # for all three bo formats (the if/then mapping is right, not just
    # for Bo3).
    artifact = _artifact_for_best_of(best_of, best_of_int)
    assert contract.validate_artifact(artifact) is None


@pytest.mark.parametrize(
    ("best_of", "best_of_int", "expected"),
    [("Bo1", 1, 2), ("Bo3", 3, 4), ("Bo5", 5, 6)],
)
@pytest.mark.parametrize("location", ["overall", "top_vetos"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_series_probabilities_wrong_length_rejected(
    best_of, best_of_int, expected, location, delta
):
    # A series_probabilities vector one entry too short or too long is
    # rejected, in both the overall result and the ranked-veto entry,
    # for all three bo formats.
    artifact = _artifact_for_best_of(best_of, best_of_int)
    bad = [0.1] * (expected + delta)
    if location == "overall":
        artifact["fixtures"][0]["overall"]["series_probabilities"] = bad
    else:
        artifact["fixtures"][0]["top_vetos"][0]["series_probabilities"] = bad
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("best_of", "best_of_int", "expected"),
    [("Bo1", 1, 2), ("Bo3", 3, 4), ("Bo5", 5, 6)],
)
def test_outcome_order_correct_length_validates(
    best_of, best_of_int, expected
):
    # outcome_order must be exactly best_of_int + 1 long; a
    # correct-length outcome_order validates for all three bo formats
    # (the if/then mapping for outcome_order is right, not just Bo3).
    artifact = _artifact_for_best_of(best_of, best_of_int)
    assert len(artifact["fixtures"][0]["outcome_order"]) == expected
    assert contract.validate_artifact(artifact) is None


@pytest.mark.parametrize(
    ("best_of", "best_of_int", "expected"),
    [("Bo1", 1, 2), ("Bo3", 3, 4), ("Bo5", 5, 6)],
)
@pytest.mark.parametrize("delta", [-1, 1])
def test_outcome_order_wrong_length_rejected(
    best_of, best_of_int, expected, delta
):
    # outcome_order is pinned to best_of_int + 1 in the schema itself
    # (not just via the scoreline_labels parity check): a vector one
    # entry short or long is rejected for all three bo formats even
    # when scoreline_labels is kept parallel to it, so the
    # positional-index chain is closed at the hoisted end.
    artifact = _artifact_for_best_of(best_of, best_of_int)
    fixture = artifact["fixtures"][0]
    bad = [[0, 0] for _ in range(expected + delta)]
    fixture["outcome_order"] = bad
    fixture["scoreline_labels"] = [str(i) for i in range(expected + delta)]
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)


def test_scoreline_labels_outcome_order_length_mismatch_rejected():
    # scoreline_labels must parallel outcome_order; JSON Schema cannot
    # express cross-field length parity, so validate_artifact checks it
    # in Python after schema validation. A mismatch (one label dropped)
    # is rejected with a clear error naming the field.
    artifact = _minimal_artifact()
    artifact["fixtures"][0]["scoreline_labels"] = ["2-0", "2-1", "1-2"]
    with pytest.raises(ValidationError, match="scoreline_labels"):
        contract.validate_artifact(artifact)


def test_wire_types_match_schema_field_sets():
    # Drift guard: each Python-side TypedDict's field set and required
    # keys must match its JSON-Schema definition's declared properties
    # and required list, so the two representations cannot silently
    # drift apart (mechanical cross-check, not manual review).
    schema = contract.load_schema()
    definitions = schema["definitions"]
    for wire_type, definition_name in contract.WIRE_TYPE_DEFINITIONS.items():
        node = definitions[definition_name]
        declared = set(node.get("properties", {}))
        required = set(node.get("required", []))
        assert set(wire_type.__annotations__) == declared, (
            f"{wire_type.__name__} field set {sorted(wire_type.__annotations__)} "
            f"drifts from schema {definition_name} properties {sorted(declared)}"
        )
        assert set(wire_type.__required_keys__) == required, (
            f"{wire_type.__name__} required keys "
            f"{sorted(wire_type.__required_keys__)} drift from schema "
            f"{definition_name} required {sorted(required)}"
        )


def test_veto_action_null_team_validates():
    # D4: a decider step may carry team: null / team_name: null (the
    # keys stay required); the widened ["string", "null"] type accepts
    # it, so a structurally teamless decider no longer fails.
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["actions"][6] = _veto_action(
        step_index=6,
        action="decider",
        team=None,
        team_name=None,
        map_name="Sunset",
    )
    assert contract.validate_artifact(artifact) is None


def test_veto_action_missing_team_still_rejected():
    # D4 widened the value type, not the key: a veto action missing the
    # "team" key is still rejected (team stays in required).
    artifact = _maximal_artifact()
    del artifact["fixtures"][0]["top_vetos"][0]["actions"][0]["team"]
    with pytest.raises(ValidationError, match="'team' is a required property"):
        contract.validate_artifact(artifact)


@pytest.mark.parametrize("action", ["ban", "pick"])
def test_veto_action_non_decider_null_team_rejected(action):
    # D4 null is legal only for the decider step: a ban or pick with
    # team: null / team_name: null is rejected by the VetoAction-level
    # if/then/else (an acting step must name its acting team).
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["actions"][0] = _veto_action(
        step_index=0,
        action=action,
        team=None,
        team_name=None,
        map_name="Ascent",
    )
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)


def test_veto_action_non_string_non_null_team_rejected():
    # A non-string, non-null team (an int) is rejected — the widened
    # type is ["string", "null"], not a free-for-all.
    artifact = _maximal_artifact()
    artifact["fixtures"][0]["top_vetos"][0]["actions"][0]["team"] = 5
    with pytest.raises(ValidationError):
        contract.validate_artifact(artifact)
