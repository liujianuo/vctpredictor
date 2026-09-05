"""Tests for P4 map-leverage attribution (``presentation/leverage.py``).

Pure synthetic tests: every input is a directly-constructed dataclass
(:class:`drivers.predict.RankedVetoPrediction`,
:class:`drivers.predict.PredictionResult`,
:class:`drivers.predict.SeriesPrediction`,
:class:`drivers.predict.PerMapPrediction` and the veto-simulator action
records) — never a ``Predictor`` and never real data, so no ``slow``
marker and no ``_real_v1_available()`` guard are needed. The §13 four
cases are exercised with hand-computed expectations, plus the D9
product ranking (including a case that fails under a raw-``|swing|``
sort), the tie-break, the probability-weighted arithmetic, the D8
degenerate-skip guards, the constraint-(d) interval-independence, and a
full wire-contract round-trip through
:func:`presentation.contract.validate_artifact` with the P3/P5-owned
keys stubbed locally (this file deliberately does not import
``tests/test_reshape.py``'s private builders).
"""

import json
import math

import pytest

from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    RankedVetoPrediction,
    SeriesPrediction,
    VetoSensitivity,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from presentation import contract, leverage, reshape


def _make_actions(played_maps, banned_maps):
    """Build a veto sequence marking ``played_maps`` as pick/decider steps.

    Emits one :class:`SimulatedVetoAction` per entry: the ``banned_maps``
    become ``"ban"`` steps first (team ids alternating ``"A"``/``"B"`` by
    step-index parity), then every ``played_maps`` entry except the last
    becomes a ``"pick"`` step, and the last becomes the team-less
    ``"decider"`` step (``team=None``). This is the minimal shape the
    leverage module reads — it only inspects each action's ``action``
    and ``map_name``, so the step kinds are chosen to exercise both
    halves of D5's played rule.

    Args:
        played_maps: The map names to mark as played (a non-empty
            sequence whose last entry becomes the decider).
        banned_maps: The map names to mark as banned.

    Returns:
        A tuple of :class:`SimulatedVetoAction` records, bans first
        then picks then the decider, with contiguous 0-based
        ``step_index`` values.

    Raises:
        Nothing — an empty ``played_maps`` simply yields only ban
            steps, and the decider is appended only when at least one
            played map is supplied.
    """
    actions = []
    index = 0
    for map_name in banned_maps:
        team = "A" if index % 2 == 0 else "B"
        actions.append(SimulatedVetoAction(index, team, "ban", map_name))
        index += 1
    for map_name in played_maps[:-1]:
        team = "A" if index % 2 == 0 else "B"
        actions.append(SimulatedVetoAction(index, team, "pick", map_name))
        index += 1
    if played_maps:
        actions.append(
            SimulatedVetoAction(index, None, "decider", played_maps[-1])
        )
    return tuple(actions)


def _make_entry(
    played_maps,
    banned_maps,
    p_a_wins,
    *,
    veto_probability=0.1,
    per_map=(),
):
    """Build one synthetic ranked-veto entry with a chosen conditional.

    Wraps the :func:`_make_actions` sequence in a minimal inner
    :class:`PredictionResult` (``veto_sensitivity`` ``None``, the given
    ``per_map`` records) whose Bo1 :class:`SeriesPrediction` carries
    ``p_a_wins_series`` exactly equal to ``p_a_wins``, so a test can set
    each entry's conditional series-win probability directly.

    Args:
        played_maps: The entry's played map names (see
            :func:`_make_actions`).
        banned_maps: The entry's banned map names.
        p_a_wins: The entry's conditional series-win probability
            (Bo1, so it is exactly ``probabilities[0]``).
        veto_probability: The exact joint probability assigned to the
            entry.
        per_map: The entry's played-map records (defaults to empty).

    Returns:
        A :class:`RankedVetoPrediction` whose ``result`` carries the
        built sequence, the Bo1 series, and the per-map records.

    Raises:
        Nothing.
    """
    series = SeriesPrediction(
        probabilities=(p_a_wins, 1.0 - p_a_wins),
        outcome_order=((1, 0), (0, 1)),
        best_of=1,
    )
    inner = PredictionResult(
        predicted_veto=_make_actions(played_maps, banned_maps),
        per_map=tuple(per_map),
        series=series,
        veto_sensitivity=None,
    )
    return RankedVetoPrediction(
        veto_probability=veto_probability, result=inner
    )


def _make_top_result(entries):
    """Build a synthetic top-level :class:`PredictionResult` for a round-trip.

    Produces a value-complete top-level result whose ``top_vetos`` are
    the given entries, whose greedy ``predicted_veto`` is the first
    entry's sequence (so the greedy rank resolves to 1), whose Bo1
    ``series`` matches the entries' outcome order, and whose
    ``veto_sensitivity`` is a real (non-``None``) record.

    Args:
        entries: The ranked-veto listing entries.

    Returns:
        A :class:`PredictionResult` wrapping the entries and the
        matching top-level fields.

    Raises:
        IndexError: If ``entries`` is empty (the greedy sequence is
            taken from the first entry).
    """
    return PredictionResult(
        predicted_veto=entries[0].result.predicted_veto,
        per_map=(),
        series=SeriesPrediction(
            probabilities=(0.5, 0.5),
            outcome_order=((1, 0), (0, 1)),
            best_of=1,
        ),
        veto_sensitivity=VetoSensitivity(
            unweighted_band_low=(0.3, 0.2),
            unweighted_band_high=(0.5, 0.4),
            band_widths=(0.2, 0.2),
            mean_band_width=0.2,
            weighted_mean=(0.4, 0.3),
            weighted_variance=(0.01, 0.01),
        ),
        top_vetos=tuple(entries),
    )


def _assemble_artifact(core, best_of, best_of_int, map_leverage):
    """Wrap a reshaped fixture core plus a real map_leverage list in an
    Artifact.

    Fills the P5-deferred ``Fixture`` keys (and every ``Artifact``-level
    key) with locally-defined stub values so the seven reshaped core
    keys plus the real ``map_leverage`` list can be validated against
    the full wire schema — deliberately without importing
    ``tests/test_reshape.py``'s private builders.

    Args:
        core: The partial dict returned by
            :func:`presentation.reshape.reshape_fixture_core`.
        best_of: One of ``"Bo1"``, ``"Bo3"`` or ``"Bo5"``.
        best_of_int: The matching map count (``1``/``3``/``5``).
        map_leverage: The real computed map-leverage list.

    Returns:
        A complete ``Artifact`` dict with a single fixture whose
        ``map_leverage`` is the supplied list.

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
            "map_leverage": map_leverage,
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


def _by_name(records):
    """Index a map-leverage output list by ``map_name``.

    Args:
        records: The list returned by
            :func:`presentation.leverage.compute_map_leverage`.

    Returns:
        A ``dict`` mapping each row's ``map_name`` to its row dict.

    Raises:
        Nothing.
    """
    return {row["map_name"]: row for row in records}


def test_flat_case_zero_swing_still_emitted():
    # §13 case 1: every entry carries the same p_a_wins_series, so every
    # emitted map has swing 0.0 and flips_favorite False; zero swing is
    # not degeneracy (only played-in-all/none is), so all maps appear.
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.6,
        ),
        _make_entry(
            ["Lotus", "Split", "Sunset"],
            ["Ascent", "Bind", "Haven", "Abyss"],
            0.6,
        ),
        _make_entry(
            ["Abyss", "Ascent", "Lotus"],
            ["Bind", "Haven", "Split", "Sunset"],
            0.6,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    assert len(output) == 7
    for row in output:
        assert row["swing"] == 0.0
        assert row["flips_favorite"] is False


def test_flip_case_flips_favorite_and_swing():
    # §13 case 2: "Ascent" is played only in the >0.5 entries and banned
    # in the <0.5 entries, so its two branch means sit on opposite sides
    # of 0.5: p_a_given_played 0.85 vs p_a_given_not 0.15 -> swing 0.7,
    # flips_favorite True.
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.9,
        ),
        _make_entry(
            ["Ascent", "Lotus", "Split"],
            ["Bind", "Haven", "Sunset", "Abyss"],
            0.8,
        ),
        _make_entry(
            ["Bind", "Haven", "Lotus"],
            ["Ascent", "Split", "Sunset", "Abyss"],
            0.2,
        ),
        _make_entry(
            ["Split", "Sunset", "Abyss"],
            ["Ascent", "Bind", "Haven", "Lotus"],
            0.1,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    ascent = _by_name(output)["Ascent"]
    assert ascent["flips_favorite"] is True
    assert ascent["p_a_given_played"] == pytest.approx(0.85)
    assert ascent["p_a_given_not"] == pytest.approx(0.15)
    assert ascent["swing"] == pytest.approx(0.7)
    assert ascent["swing"] > 0


def test_same_side_map_does_not_flip():
    # A map whose two branch means sit on the same side of 0.5 (played
    # 0.75, not-played 0.6) does not flip, even though its swing is
    # non-zero.
    entries = (
        _make_entry(
            ["Bind", "Haven", "Lotus"],
            ["Ascent", "Split", "Sunset", "Abyss"],
            0.8,
        ),
        _make_entry(
            ["Bind", "Split", "Sunset"],
            ["Ascent", "Haven", "Lotus", "Abyss"],
            0.7,
        ),
        _make_entry(
            ["Ascent", "Haven", "Lotus"],
            ["Bind", "Split", "Sunset", "Abyss"],
            0.6,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    bind = _by_name(output)["Bind"]
    assert bind["flips_favorite"] is False
    assert bind["p_a_given_played"] == pytest.approx(0.75)
    assert bind["p_a_given_not"] == pytest.approx(0.6)
    assert bind["swing"] == pytest.approx(0.15)


def test_exactly_half_never_flips():
    # A2 boundary: a map whose played-branch mean is exactly 0.5 never
    # flips, regardless of the not-played branch.
    entries = (
        _make_entry(["Haven"], ["Ascent", "Bind"], 0.5),
        _make_entry(["Ascent"], ["Haven", "Bind"], 0.2),
    )
    output = leverage.compute_map_leverage(entries)
    haven = _by_name(output)["Haven"]
    assert haven["p_a_given_played"] == pytest.approx(0.5)
    assert haven["flips_favorite"] is False


def test_always_played_and_never_played_maps_emit_nothing():
    # §13 case 3: "Ascent" is played in every entry and "Abyss" is
    # banned in every entry; both are degenerate (no contrast) and must
    # be absent, not emitted as spurious zero-swing rows. Surviving maps
    # are unaffected.
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.8,
        ),
        _make_entry(
            ["Ascent", "Lotus", "Split"],
            ["Bind", "Haven", "Sunset", "Abyss"],
            0.6,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    names = {row["map_name"] for row in output}
    assert "Ascent" not in names
    assert "Abyss" not in names
    assert "Bind" in names
    assert "Haven" in names


def test_below_minimum_backing_map_still_emitted():
    # §13 case 4: a map played in exactly 1 of 10 entries is emitted
    # with n_vetos_backing == 1 and a small p_played (D10: P4 does not
    # filter; the minimum-backing threshold is a later narrative rule).
    entries = []
    for index in range(10):
        if index == 0:
            played = ["Ascent", "Bind", "Haven"]
            banned = ["Lotus", "Split", "Sunset", "Abyss"]
        else:
            played = ["Lotus", "Split", "Sunset"]
            banned = ["Ascent", "Bind", "Haven", "Abyss"]
        entries.append(
            _make_entry(played, banned, 0.6, veto_probability=0.1)
        )
    output = leverage.compute_map_leverage(entries)
    bind = _by_name(output)["Bind"]
    assert bind["n_vetos_backing"] == 1
    assert bind["p_played"] == pytest.approx(0.1)


def test_ranking_by_abs_swing_times_p_played():
    # D9: "Big" has a larger |swing| but a tiny p_played, while
    # "Moderate" has a smaller |swing| and a much larger p_played; the
    # product ordering must rank "Moderate" first (a raw-|swing| sort
    # would rank "Big" first).
    entries = (
        _make_entry(
            ["Big", "Ascent", "Bind"],
            ["Moderate", "Haven", "Lotus", "Split"],
            0.95,
            veto_probability=0.05,
        ),
        _make_entry(
            ["Big", "Haven", "Lotus"],
            ["Moderate", "Ascent", "Bind", "Split"],
            0.95,
            veto_probability=0.05,
        ),
        _make_entry(
            ["Moderate", "Ascent", "Haven"],
            ["Big", "Bind", "Lotus", "Split"],
            0.7,
            veto_probability=0.45,
        ),
        _make_entry(
            ["Lotus", "Split", "Bind"],
            ["Big", "Moderate", "Ascent", "Haven"],
            0.5,
            veto_probability=0.45,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    big = _by_name(output)["Big"]
    moderate = _by_name(output)["Moderate"]
    assert abs(big["swing"]) > abs(moderate["swing"])
    names = [row["map_name"] for row in output]
    assert names.index("Moderate") < names.index("Big")


def test_tie_break_map_name_ascending():
    # D9/A3: every emitted map ties on |swing| * p_played (0.6 * 0.5),
    # so the ascending map_name tie-break fully determines the order.
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.8,
        ),
        _make_entry(
            ["Lotus", "Split", "Sunset"],
            ["Ascent", "Bind", "Haven", "Abyss"],
            0.2,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    assert [row["map_name"] for row in output] == [
        "Ascent",
        "Bind",
        "Haven",
        "Lotus",
        "Split",
        "Sunset",
    ]


def test_p_played_is_mass_weighted():
    # Unequal veto_probability weights: a map played in the weight-0.9
    # entry has p_played 0.9, not the plain 1/2 entry count.
    entries = (
        _make_entry(["Ascent"], ["Bind"], 0.7, veto_probability=0.9),
        _make_entry(["Bind"], ["Ascent"], 0.3, veto_probability=0.1),
    )
    output = leverage.compute_map_leverage(entries)
    assert _by_name(output)["Ascent"]["p_played"] == pytest.approx(0.9)


def test_conditional_means_are_probability_weighted():
    # Two entries play "Ascent" with unequal weights and different
    # p_a_wins: the weighted mean (0.9*0.9 + 0.1*0.3) / (0.9 + 0.1) =
    # 0.84 differs from the plain average 0.6, pinning the weighting.
    entries = (
        _make_entry(["Ascent"], ["Bind"], 0.9, veto_probability=0.9),
        _make_entry(["Ascent"], ["Bind"], 0.3, veto_probability=0.1),
        _make_entry(["Bind"], ["Ascent"], 0.5, veto_probability=0.5),
    )
    output = leverage.compute_map_leverage(entries)
    ascent = _by_name(output)["Ascent"]
    assert ascent["p_a_given_played"] == pytest.approx(0.84)
    assert ascent["p_a_given_played"] != pytest.approx(0.6)


def test_swing_is_exact_difference():
    # swing is exactly p_a_given_played - p_a_given_not (no re-rounding).
    entries = (
        _make_entry(["Ascent"], ["Bind"], 0.8, veto_probability=0.5),
        _make_entry(["Bind"], ["Ascent"], 0.2, veto_probability=0.5),
    )
    output = leverage.compute_map_leverage(entries)
    ascent = _by_name(output)["Ascent"]
    assert ascent["swing"] == (
        ascent["p_a_given_played"] - ascent["p_a_given_not"]
    )
    assert ascent["swing"] == pytest.approx(0.6)


def test_empty_listing_returns_empty():
    # An empty listing returns [] (no entries, no mass, no maps).
    assert leverage.compute_map_leverage(()) == []
    assert leverage.compute_map_leverage([]) == []


def test_all_zero_probability_returns_empty_without_error():
    # A listing whose entries all carry veto_probability 0.0 has total
    # mass 0.0 -> [] with no ZeroDivisionError (A4).
    entries = (
        _make_entry(["Ascent"], ["Bind"], 0.6, veto_probability=0.0),
        _make_entry(["Bind"], ["Ascent"], 0.4, veto_probability=0.0),
    )
    assert leverage.compute_map_leverage(entries) == []


def test_single_entry_returns_empty():
    # A single entry plays every map it plays "in all" and bans every
    # other map "in none", so no map has contrast and the result is [].
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.6,
        ),
    )
    assert leverage.compute_map_leverage(entries) == []


def test_zero_probability_entry_does_not_produce_nan():
    # "Lone" is played only in the zero-weight entry, so its played_mass
    # is 0.0 while its n_vetos_backing is 1 — the D8 condition-3 guard
    # must skip it before dividing, not emit NaN. Every other emitted
    # float must be finite.
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lone", "Lotus", "Split", "Abyss"],
            0.9,
            veto_probability=0.5,
        ),
        _make_entry(
            ["Lone", "Lotus", "Split"],
            ["Ascent", "Bind", "Haven", "Abyss"],
            0.1,
            veto_probability=0.0,
        ),
        _make_entry(
            ["Lotus", "Split", "Abyss"],
            ["Lone", "Ascent", "Bind", "Haven"],
            0.4,
            veto_probability=0.5,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    assert "Lone" not in _by_name(output)
    for row in output:
        for key in ("p_played", "p_a_given_played", "p_a_given_not", "swing"):
            assert math.isfinite(row[key])


def test_output_independent_of_epistemic_intervals():
    # Constraint (d): a listing whose per-map records carry populated
    # bootstrap intervals and an otherwise-identical listing with null
    # intervals produce equal output — epistemic width never leaks into
    # structural swing.
    populated_pm = PerMapPrediction(
        map_name="Ascent",
        probabilities=(0.5, 0.1, 0.1, 0.3),
        interval_low=(0.3, 0.0, 0.0, 0.2),
        interval_high=(0.7, 0.3, 0.3, 0.5),
        n_games_backing=12,
    )
    null_pm = PerMapPrediction(
        map_name="Ascent",
        probabilities=(0.5, 0.1, 0.1, 0.3),
        interval_low=None,
        interval_high=None,
        n_games_backing=0,
    )
    common = [
        (["Ascent", "Bind", "Haven"],
         ["Lotus", "Split", "Sunset", "Abyss"], 0.8),
        (["Lotus", "Split", "Sunset"],
         ["Ascent", "Bind", "Haven", "Abyss"], 0.2),
    ]
    populated = tuple(
        _make_entry(played, banned, p_a, per_map=(populated_pm,))
        for played, banned, p_a in common
    )
    null_intervals = tuple(
        _make_entry(played, banned, p_a, per_map=(null_pm,))
        for played, banned, p_a in common
    )
    assert leverage.compute_map_leverage(populated) == (
        leverage.compute_map_leverage(null_intervals)
    )


def test_wire_shape_and_json_native():
    # Every emitted entry has exactly the seven MapLeverage keys with
    # the right JSON-native value types, and json.dumps round-trips
    # without a custom encoder (no tuples anywhere).
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.8,
            veto_probability=0.6,
        ),
        _make_entry(
            ["Lotus", "Split", "Sunset"],
            ["Ascent", "Bind", "Haven", "Abyss"],
            0.2,
            veto_probability=0.4,
        ),
    )
    output = leverage.compute_map_leverage(entries)
    expected_keys = {
        "map_name",
        "p_played",
        "p_a_given_played",
        "p_a_given_not",
        "swing",
        "flips_favorite",
        "n_vetos_backing",
    }
    for row in output:
        assert set(row) == expected_keys
        assert isinstance(row["map_name"], str)
        assert isinstance(row["p_played"], float)
        assert isinstance(row["p_a_given_played"], float)
        assert isinstance(row["p_a_given_not"], float)
        assert isinstance(row["swing"], float)
        assert isinstance(row["flips_favorite"], bool)
        assert isinstance(row["n_vetos_backing"], int)
    assert json.dumps(output)


def test_full_artifact_round_trip_validates_real_map_leverage():
    # The real map_leverage list is validated by the checked-in schema,
    # not just by isinstance: reshape the seven P3 keys from a synthetic
    # result, stub the P5 keys, and validate the whole artifact.
    entries = (
        _make_entry(
            ["Ascent", "Bind", "Haven"],
            ["Lotus", "Split", "Sunset", "Abyss"],
            0.8,
            veto_probability=0.6,
        ),
        _make_entry(
            ["Lotus", "Split", "Sunset"],
            ["Ascent", "Bind", "Haven", "Abyss"],
            0.2,
            veto_probability=0.4,
        ),
    )
    map_leverage = leverage.compute_map_leverage(entries)
    result = _make_top_result(entries)
    core = reshape.reshape_fixture_core(
        result, "A", "B", {"A": "Team A", "B": "Team B"}
    )
    artifact = _assemble_artifact(core, "Bo1", 1, map_leverage)
    assert contract.validate_artifact(artifact) is None
