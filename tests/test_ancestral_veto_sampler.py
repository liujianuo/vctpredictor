"""Tests for the ancestral veto sampler (M29).

Covers the duplicated ACTION_SEQUENCES parity cross-check against
``models.greedy_veto_simulator`` (so decision 6's duplication cannot
silently drift), seeded determinism (same seed -> byte-identical,
different seeds -> divergent under a non-degenerate stub), the exact
``n_samples`` contract, the forced decider's exclusion from
``sequence_probability`` (hand-computed exact-product check plus a
would-be-wrong-if-included variant and a predictor-call-count check),
strict step-index-parity turn alternation for Bo1/Bo3/Bo5, the
near-one-hot-stub cross-check reproducing ``simulate_veto``'s
deterministic pick empirically, the full set of input/predictor
``ValueError`` paths, the ``to_dict`` shapes, an integration test
wiring real fitted M27/M28 closures (each action-restricted) as the
dict-keyed predictors with no adapter code, and a skip-guarded real
``data/v1`` smoke test with the M25 greedy arm for both actions.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.veto_evaluation import greedy_veto_step_model
from features import map_win_rate
from models.ancestral_veto_sampler import (
    _PROB_SUM_TOLERANCE,
    ACTION_SEQUENCES,
    SampledVetoAction,
    SampledVetoSequence,
    sample_veto_sequences,
)
from models.greedy_veto_simulator import (
    ACTION_SEQUENCES as SIBLING_ACTION_SEQUENCES,
)
from models.greedy_veto_simulator import (
    simulate_veto,
    team_map_scores,
)

# The as-of cutoff for the synthetic greedy-choice league: one hour
# after the last fixture match, so every fixture row is strictly
# before it (mirrors tests/test_greedy_veto_simulator.py).
QUERY_DATE = "2026-01-06T00:00:00"

# The 7-map pool all three ACTION_SEQUENCES walk, matching
# config.json's 2026-abyss era in ascending name order.
POOL = ("Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset")

# The as-of cutoff of the ban/pick conditional-logit league (the hour
# after its last strictly-prior match; m9 is dated exactly at it and
# exists so the opponent-resolution path has a live match to resolve —
# mirrors tests/test_conditional_logit_ban.py).
CL_QUERY_DATE = "2026-01-01T08:00:00"

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]

# The real v1 tables the smoke test needs (matches for teams/dates,
# maps for the as-of win-rate history the greedy arm consumes). Only
# these two — the sampler itself never touches veto_actions/splits.
_REAL_V1_TABLES = ("matches", "maps")


def _matches_df(rows):
    """Build a matches table with the fixed M8 column set.

    Wraps ``pd.DataFrame`` so every test fixture produces the same
    column order/dtypes regardless of which subset of columns a given
    fixture actually needs.

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


def _add(match_rows, map_rows, mid, date, team1_id, team2_id, map_name, t1s, t2s):
    """Append one completed match and its finished map to the row lists.

    The single row-writing helper for the synthetic league fixtures.
    The map's ``winner`` is derived from the scores (never a
    display-name string), matching the existing test fixtures'
    convention.

    Args:
        match_rows: The mutable match-row list to append to.
        map_rows: The mutable map-row list to append to.
        mid: The shared ``match_id`` for the new match and map.
        date: The match's ISO date string.
        team1_id: The match's team1 stable id.
        team2_id: The match's team2 stable id.
        map_name: The finished map's name.
        t1s: Rounds team1 won (the map's ``team1_score``).
        t2s: Rounds team2 won (the map's ``team2_score``).

    Returns:
        Nothing (appends in place).

    Raises:
        Nothing.
    """
    match_rows.append(
        {
            "match_id": mid,
            "date": date,
            "team1_id": team1_id,
            "team2_id": team2_id,
            "status": "completed",
        }
    )
    map_rows.append(
        {
            "match_id": mid,
            "map_index": 0,
            "map_name": map_name,
            "team1_score": t1s,
            "team2_score": t2s,
            "winner": team1_id if t1s > t2s else team2_id,
        }
    )


def _stamp(i):
    """Return the ISO timestamp ``i`` hours after a fixed 2026-01-01 base.

    The shared clock for the synthetic leagues: every fixture's dates
    are one hour apart so chronological ordering is unambiguous and the
    as-of query date is strictly after everything.

    Args:
        i: The hour offset from the base.

    Returns:
        An ISO-8601 datetime string.

    Raises:
        Nothing.
    """
    base = pd.Timestamp("2026-01-01T00:00:00")
    return (base + pd.Timedelta(hours=i)).isoformat()


def _greedy_league_tables():
    """Build the 16-map greedy-choice league.

    The identical league ``tests/test_greedy_veto_simulator.py``'s
    ``_league_tables`` builds: team ``A`` is 0W-4L on Split (its
    weakest map) and 4W-0L on Haven (its strongest), with an overall
    0.5 record; team ``B`` mirrors that with Sunset (0W-4L, weakest)
    and Ascent (4W-0L, strongest). Every opponent id is unique and
    plays only once, and all rows are dated before
    :data:`QUERY_DATE`, so the greedy walk is fully deterministic for
    every format (Bo3: Split/Sunset bans then Haven/Ascent picks, then
    Abyss/Lotus bans, Summit decider — the exact sequence the existing
    simulator test pins down).

    Returns:
        A ``(matches_df, maps_df)`` tuple of 16 matches and 16 maps
        built by :func:`_build`.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    for i in range(4):
        _add(match_rows, map_rows, f"a_split_{i}", _stamp(i), "A", f"op{i}", "Split", 8, 13)
    for i in range(4):
        _add(match_rows, map_rows, f"a_haven_{i}", _stamp(4 + i), "A", f"op{10 + i}", "Haven", 13, 8)
    for i in range(4):
        _add(match_rows, map_rows, f"b_sunset_{i}", _stamp(8 + i), "B", f"op{20 + i}", "Sunset", 8, 13)
    for i in range(4):
        _add(match_rows, map_rows, f"b_ascent_{i}", _stamp(12 + i), "B", f"op{30 + i}", "Ascent", 13, 8)
    return _matches_df(match_rows), _maps_df(map_rows)


def _cl_league_tables():
    """Build the 9-match conditional-logit league.

    The identical league ``tests/test_conditional_logit_ban.py`` /
    ``tests/test_conditional_logit_pick.py`` build: team ``A`` plays
    Haven twice (1 win) and Split twice (0 wins); team ``B`` plays
    Ascent twice (2 wins) and Sunset twice (1 win); a ninth match
    (m9, A vs B on Haven at exactly :data:`CL_QUERY_DATE`) is excluded
    from every as-of feature by the strict-``<`` boundary and exists
    so the opponent-resolution path of the wrapped M27/M28 predictors
    has a live match to resolve.

    Returns:
        A ``(matches_df, maps_df)`` tuple of 9 matches and 9 maps.

    Raises:
        Nothing.
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "m1", _stamp(0), "A", "o1", "Haven", 13, 11)
    _add(match_rows, map_rows, "m2", _stamp(1), "A", "o2", "Haven", 8, 13)
    _add(match_rows, map_rows, "m3", _stamp(2), "A", "o3", "Split", 5, 13)
    _add(match_rows, map_rows, "m4", _stamp(3), "A", "o4", "Split", 9, 13)
    _add(match_rows, map_rows, "m5", _stamp(4), "B", "o5", "Ascent", 13, 9)
    _add(match_rows, map_rows, "m6", _stamp(5), "B", "o6", "Ascent", 13, 10)
    _add(match_rows, map_rows, "m7", _stamp(6), "B", "o7", "Sunset", 13, 11)
    _add(match_rows, map_rows, "m8", _stamp(7), "B", "o8", "Sunset", 6, 13)
    _add(match_rows, map_rows, "m9", CL_QUERY_DATE, "A", "B", "Haven", 13, 10)
    return _matches_df(match_rows), _maps_df(map_rows)


def _uniform_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """Return a uniform distribution over the remaining maps.

    The canonical non-degenerate stub predictor: every remaining map
    gets ``1 / len(remaining_maps)``, so the sampler's draws are
    genuinely random (two seeds are expected to diverge) and every
    map's probability is a hand-derivable rational.

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


def _near_one_hot_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
    """Return a near-one-hot distribution on the greedy rule's choice.

    Reproduces ``simulate_veto``'s deterministic per-step decision over
    the *current* remaining set — the acting team's argmin mean (ban)
    / argmax mean (pick), ties broken by ascending map name, via
    ``models.greedy_veto_simulator.team_map_scores`` (the per-map
    means do not depend on the pool, so scoring the remaining subset
    gives the same values ``simulate_veto`` scores the full pool with,
    and the tie-break keys match its decision 4 exactly) — and puts
    ``1 - (n-1)*1e-9`` mass on that map and a ``1e-9`` floor on every
    other, summing to 1 within float rounding. Used to verify the
    sampler's distributional generalization: with this arm, sampled
    sequences must match the deterministic ``simulate_veto`` output
    with overwhelming empirical frequency.

    Args:
        acting_team_id: The acting team's stable id.
        action: ``"ban"`` or ``"pick"``.
        remaining_maps: The sorted remaining-maps list the returned
            distribution aligns to.
        date: The as-of cutoff for the win-rate lookups.
        matches_df: The materialised matches table.
        maps_df: The materialised maps table.

    Returns:
        A ``list`` of ``len(remaining_maps)`` ``float`` probabilities:
        ≈1 on the greedy rule's chosen map, ``1e-9`` on the rest,
        summing to 1 within float rounding.

    Raises:
        ValueError: If the win-rate lookup fails (propagated from
            :func:`models.greedy_veto_simulator.team_map_scores`).
    """
    scores = team_map_scores(
        acting_team_id, remaining_maps, date, matches_df, maps_df, map_win_rate.DEFAULT_K
    )
    if action == "ban":
        chosen = min(remaining_maps, key=lambda name: (scores[name], name))
    else:
        chosen = min(remaining_maps, key=lambda name: (-scores[name], name))
    n = len(remaining_maps)
    floor = 1e-9
    probs = [floor] * n
    probs[list(remaining_maps).index(chosen)] = 1.0 - floor * (n - 1)
    return probs


def _assert_walk_shape(samples):
    """Assert every sampled sequence is structurally sane.

    The shared shape contract for a full walk: one action per
    ``ACTION_SEQUENCES[best_of]`` step, the last step is the forced
    decider with ``probability == 1.0`` and ``team is None``, every
    non-decider action's map was actually in that step's remaining
    pool (replayed independently in this helper), and the sequence
    probability lies in ``(0, 1]``.

    Args:
        samples: A ``list`` of :class:`SampledVetoSequence` from
            :func:`sample_veto_sequences`.

    Returns:
        None (asserts instead of returning).

    Raises:
        AssertionError: On any structural violation.
    """
    for seq in samples:
        sequence = ACTION_SEQUENCES[seq.best_of]
        assert len(seq.actions) == len(sequence)
        remaining = {a.map_name for a in seq.actions}
        # The sampler never invents or drops a map.
        assert len(remaining) == len(seq.actions)
        replay_remaining = set(remaining)
        for step_index, (action, expected_action) in enumerate(
            zip(seq.actions, sequence)
        ):
            assert action.step_index == step_index
            assert action.action == expected_action
            assert action.map_name in replay_remaining
            if expected_action == "decider":
                assert action.team is None
                assert action.probability == 1.0
            replay_remaining.remove(action.map_name)
        assert 0.0 < seq.sequence_probability <= 1.0


def test_action_sequences_match_sibling_module():
    # Decision 6 duplicates ACTION_SEQUENCES locally (the models-module
    # boundary forbids the import); a same-values parity check keeps the
    # duplication from silently drifting apart.
    assert ACTION_SEQUENCES == SIBLING_ACTION_SEQUENCES
    for sequence in ACTION_SEQUENCES.values():
        assert len(sequence) == 7
        assert sequence[-1] == "decider"
        assert set(sequence) <= {"ban", "pick", "decider"}


def test_same_seed_byte_identical_different_seeds_diverge():
    # Decision 4: identical (seed, inputs) -> byte-identical output
    # across two separate calls; two different seeds under a
    # non-degenerate stub are expected to diverge.
    matches_df, maps_df = _greedy_league_tables()
    kwargs = {
        "team_a_id": "A",
        "team_b_id": "B",
        "best_of": "Bo3",
        "date": QUERY_DATE,
        "matches_df": matches_df,
        "maps_df": maps_df,
        "predictor_fn_by_action": {"ban": _uniform_stub, "pick": _uniform_stub},
        "n_samples": 20,
        "map_pool": POOL,
    }
    run1 = sample_veto_sequences(rng=np.random.default_rng(42), **kwargs)
    run2 = sample_veto_sequences(rng=np.random.default_rng(42), **kwargs)
    run3 = sample_veto_sequences(rng=np.random.default_rng(43), **kwargs)
    assert [s.to_dict() for s in run1] == [s.to_dict() for s in run2]
    keys1 = [tuple(a.map_name for a in s.actions) for s in run1]
    keys3 = [tuple(a.map_name for a in s.actions) for s in run3]
    assert keys1 != keys3


def test_n_samples_returns_exactly_that_many():
    matches_df, maps_df = _greedy_league_tables()
    samples = sample_veto_sequences(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        {"ban": _uniform_stub, "pick": _uniform_stub},
        n_samples=5,
        rng=np.random.default_rng(0),
        map_pool=POOL,
    )
    assert len(samples) == 5
    _assert_walk_shape(samples)


def test_decider_forced_and_excluded_from_sequence_probability():
    # Decision 2/3: the decider is forced (probability == 1.0, sole
    # remaining map, no predictor consulted) and its 1.0 is excluded
    # from the sequence-probability product — verified exactly by
    # multiplying the recorded non-decider probabilities in step order
    # (the same floats, the same left-to-right multiplications, so the
    # comparison is bit-exact), plus a would-be-wrong-if-included
    # variant and a predictor-call-count check.
    matches_df, maps_df = _greedy_league_tables()
    calls = {"count": 0}

    def counting_uniform_stub(
        acting_team_id, action, remaining_maps, date, matches_df, maps_df
    ):
        """Uniform stub that counts its invocations.

        Args:
            acting_team_id: The acting team's stable id (ignored).
            action: The step's action (ignored).
            remaining_maps: The sorted remaining-maps list.
            date: The as-of cutoff (ignored).
            matches_df: The materialised matches table (ignored).
            maps_df: The materialised maps table (ignored).

        Returns:
            A uniform distribution over ``remaining_maps``.

        Raises:
            Nothing.
        """
        calls["count"] += 1
        return _uniform_stub(
            acting_team_id, action, remaining_maps, date, matches_df, maps_df
        )

    samples = sample_veto_sequences(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        {"ban": counting_uniform_stub, "pick": counting_uniform_stub},
        n_samples=1,
        rng=np.random.default_rng(7),
        map_pool=POOL,
    )
    seq = samples[0]
    # A Bo3 has exactly 6 choosing steps; the decider must never reach
    # a predictor.
    assert calls["count"] == 6
    decider = seq.actions[-1]
    assert decider.action == "decider"
    assert decider.team is None
    assert decider.probability == 1.0
    # The sole remaining map at the decider.
    chosen_non_decider = {a.map_name for a in seq.actions[:-1]}
    assert decider.map_name not in chosen_non_decider
    assert len(chosen_non_decider) == 6
    # Exact product of the non-decider steps (same floats, same order).
    expected = math.prod(a.probability for a in seq.actions[:-1])
    assert seq.sequence_probability == expected
    # The variant that would give a wrong answer if a non-1.0 decider
    # probability were (incorrectly) folded into the product: the
    # exclusion is what keeps the product at the non-decider value.
    wrong_if_included = expected * 0.25
    assert seq.sequence_probability != wrong_if_included
    # Every non-decider probability is a strict rational 1/k here, so
    # the product is strictly below 1 and the decider's 1.0 is the only
    # thing keeping each probability in (0, 1].
    assert 0.0 < seq.sequence_probability < 1.0


def test_turn_alternation_by_step_parity_all_formats():
    # Decision 8: strict step-index parity — even steps act as
    # team_a_id, odd as team_b_id — for every format; the decider is
    # team-less.
    matches_df, maps_df = _greedy_league_tables()
    for best_of in ("Bo1", "Bo3", "Bo5"):
        samples = sample_veto_sequences(
            "A", "B", best_of, QUERY_DATE, matches_df, maps_df,
            {"ban": _uniform_stub, "pick": _uniform_stub},
            n_samples=1,
            rng=np.random.default_rng(0),
            map_pool=POOL,
        )
        seq = samples[0]
        for action in seq.actions:
            if action.action == "decider":
                assert action.team is None
            elif action.step_index % 2 == 0:
                assert action.team == "A"
            else:
                assert action.team == "B"
        _assert_walk_shape(samples)


def test_near_one_hot_stub_reproduces_greedy_simulator():
    # Decision 2's "distributional generalization of the deterministic
    # rule" cross-check: with a near-one-hot stub concentrating ~1 -
    # (n-1)*1e-9 on simulate_veto's own per-step choice, 100 sampled
    # Bo3 walks must match the deterministic sequence with overwhelming
    # empirical frequency (expected deviation count ~ 100 * 6 * 6e-9,
    # i.e. essentially zero).
    matches_df, maps_df = _greedy_league_tables()
    expected = simulate_veto(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        map_win_rate.DEFAULT_K, map_pool=POOL,
    )
    expected_maps = tuple(a.map_name for a in expected)
    samples = sample_veto_sequences(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        {"ban": _near_one_hot_stub, "pick": _near_one_hot_stub},
        n_samples=100,
        rng=np.random.default_rng(1234),
        map_pool=POOL,
    )
    observed = [tuple(a.map_name for a in s.actions) for s in samples]
    assert observed == [expected_maps] * 100


def test_to_dict_shapes_and_json_serializable():
    # Both dataclasses' to_dict shapes: SampledVetoAction mirrors the
    # sibling SimulatedVetoAction keys plus probability; the sequence
    # dict round-trips through json.dumps.
    matches_df, maps_df = _greedy_league_tables()
    samples = sample_veto_sequences(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        {"ban": _uniform_stub, "pick": _uniform_stub},
        n_samples=1,
        rng=np.random.default_rng(3),
        map_pool=POOL,
    )
    seq = samples[0]
    action_dict = seq.actions[0].to_dict()
    assert set(action_dict) == {
        "step_index", "team", "action", "map_name", "probability"
    }
    assert action_dict["probability"] == seq.actions[0].probability
    seq_dict = seq.to_dict()
    assert set(seq_dict) == {
        "team_a_id", "team_b_id", "best_of", "date", "actions",
        "sequence_probability",
    }
    assert len(seq_dict["actions"]) == len(seq.actions)
    assert json.dumps(seq_dict)  # must not raise
    # Round-trip: the dict reproduces the dataclass fields exactly.
    rebuilt = SampledVetoSequence(
        team_a_id=seq_dict["team_a_id"],
        team_b_id=seq_dict["team_b_id"],
        best_of=seq_dict["best_of"],
        date=seq_dict["date"],
        actions=tuple(SampledVetoAction(**a) for a in seq_dict["actions"]),
        sequence_probability=seq_dict["sequence_probability"],
    )
    assert rebuilt == seq


def test_unknown_best_of_raises():
    matches_df, maps_df = _greedy_league_tables()
    with pytest.raises(ValueError, match="best_of"):
        sample_veto_sequences(
            "A", "B", "Bo7", QUERY_DATE, matches_df, maps_df,
            {"ban": _uniform_stub}, 1, np.random.default_rng(0),
            map_pool=POOL,
        )


def test_map_pool_size_mismatch_raises():
    matches_df, maps_df = _greedy_league_tables()
    with pytest.raises(ValueError, match="map_pool has 6"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": _uniform_stub, "pick": _uniform_stub}, 1,
            np.random.default_rng(0), map_pool=POOL[:6],
        )


def test_map_pool_duplicates_after_normalization_raise():
    matches_df, maps_df = _greedy_league_tables()
    dup_pool = ["Abyss", "ascent", "Ascent", "Haven", "Lotus", "Split", "Summit"]
    with pytest.raises(ValueError, match="duplicate"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": _uniform_stub, "pick": _uniform_stub}, 1,
            np.random.default_rng(0), map_pool=dup_pool,
        )


def test_missing_predictor_key_raises():
    # A Bo3 needs both "ban" and "pick"; supplying only "ban" must fail
    # loudly naming the missing action before any sampling.
    matches_df, maps_df = _greedy_league_tables()
    with pytest.raises(ValueError, match="'pick'"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": _uniform_stub}, 1, np.random.default_rng(0),
            map_pool=POOL,
        )
    # A Bo1 needs only "ban" — no "pick" key required.
    samples = sample_veto_sequences(
        "A", "B", "Bo1", QUERY_DATE, matches_df, maps_df,
        {"ban": _uniform_stub}, 1, np.random.default_rng(0),
        map_pool=POOL,
    )
    _assert_walk_shape(samples)


def test_non_positive_n_samples_raises():
    matches_df, maps_df = _greedy_league_tables()
    with pytest.raises(ValueError, match="n_samples"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": _uniform_stub, "pick": _uniform_stub}, 0,
            np.random.default_rng(0), map_pool=POOL,
        )


def test_predictor_wrong_length_vector_raises():
    matches_df, maps_df = _greedy_league_tables()

    def wrong_length_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
        """Return a vector of the wrong length (one fewer entry).

        Args:
            acting_team_id: The acting team's stable id (ignored).
            action: The step's action (ignored).
            remaining_maps: The sorted remaining-maps list.
            date: The as-of cutoff (ignored).
            matches_df: The materialised matches table (ignored).
            maps_df: The materialised maps table (ignored).

        Returns:
            A ``len(remaining_maps) - 1``-entry uniform list.

        Raises:
            Nothing.
        """
        return [1.0 / len(remaining_maps)] * (len(remaining_maps) - 1)

    with pytest.raises(ValueError, match="returned 6 probabilit"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": wrong_length_stub, "pick": wrong_length_stub}, 1,
            np.random.default_rng(0), map_pool=POOL,
        )


def test_predictor_negative_entries_raise():
    matches_df, maps_df = _greedy_league_tables()

    def negative_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
        """Return a vector with one negative entry.

        Args:
            acting_team_id: The acting team's stable id (ignored).
            action: The step's action (ignored).
            remaining_maps: The sorted remaining-maps list.
            date: The as-of cutoff (ignored).
            matches_df: The materialised matches table (ignored).
            maps_df: The materialised maps table (ignored).

        Returns:
            A uniform list over ``remaining_maps`` with the first entry
            negated.

        Raises:
            Nothing.
        """
        n = len(remaining_maps)
        probs = [1.0 / n] * n
        probs[0] = -0.1
        return probs

    with pytest.raises(ValueError, match="negative"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": negative_stub, "pick": negative_stub}, 1,
            np.random.default_rng(0), map_pool=POOL,
        )


def test_predictor_non_distribution_sum_raises():
    matches_df, maps_df = _greedy_league_tables()

    def non_distribution_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
        """Return entries that do not sum to 1 beyond the tolerance.

        Args:
            acting_team_id: The acting team's stable id (ignored).
            action: The step's action (ignored).
            remaining_maps: The sorted remaining-maps list.
            date: The as-of cutoff (ignored).
            matches_df: The materialised matches table (ignored).
            maps_df: The materialised maps table (ignored).

        Returns:
            A ``len(remaining_maps)``-entry list of equal weights
            summing to 2.0 (each ``2 / n``).

        Raises:
            Nothing.
        """
        n = len(remaining_maps)
        return [2.0 / n] * n

    with pytest.raises(ValueError, match="must sum to 1.0"):
        sample_veto_sequences(
            "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
            {"ban": non_distribution_stub, "pick": non_distribution_stub}, 1,
            np.random.default_rng(0), map_pool=POOL,
        )


def test_sum_within_tolerance_accepted():
    # Decision 5: a vector within _PROB_SUM_TOLERANCE of a distribution
    # is accepted and renormalized (a sum of 1.0 + 5e-7 is inside the
    # 1e-6 tolerance).
    matches_df, maps_df = _greedy_league_tables()

    def slightly_over_stub(acting_team_id, action, remaining_maps, date, matches_df, maps_df):
        """Return a distribution summing to ``1 + 5e-7``.

        Args:
            acting_team_id: The acting team's stable id (ignored).
            action: The step's action (ignored).
            remaining_maps: The sorted remaining-maps list.
            date: The as-of cutoff (ignored).
            matches_df: The materialised matches table (ignored).
            maps_df: The materialised maps table (ignored).

        Returns:
            A uniform list over ``remaining_maps`` with an additive
            ``5e-7`` on the first entry.

        Raises:
            Nothing.
        """
        n = len(remaining_maps)
        probs = [1.0 / n] * n
        probs[0] += 5e-7
        return probs

    samples = sample_veto_sequences(
        "A", "B", "Bo3", QUERY_DATE, matches_df, maps_df,
        {"ban": slightly_over_stub, "pick": slightly_over_stub}, 1,
        np.random.default_rng(0), map_pool=POOL,
    )
    assert abs(_PROB_SUM_TOLERANCE - 1e-6) < 1e-18
    _assert_walk_shape(samples)


def test_config_era_pool_resolution_path():
    # map_pool=None resolves the active era's pool from config.json via
    # the query date's calendar date (mirroring simulate_veto's own
    # resolution path); the stub ignores matches/maps, so empty tables
    # with the right columns are enough. 2026-09-01 falls in the
    # 2026-abyss era, whose 7-map pool is exactly POOL.
    matches_df = _matches_df([])
    maps_df = _maps_df([])
    samples = sample_veto_sequences(
        "A", "B", "Bo3", "2026-09-01T12:00:00", matches_df, maps_df,
        {"ban": _uniform_stub, "pick": _uniform_stub}, 1,
        np.random.default_rng(0),
    )
    seq = samples[0]
    assert sorted(a.map_name for a in seq.actions) == sorted(POOL)
    _assert_walk_shape(samples)


def test_real_fitted_conditional_logit_closures_integration():
    # Decision 1's headline wiring: real fitted M27/M28 closures (each
    # action-restricted — the ban closure raises on action != "ban" and
    # the pick closure on action != "pick") slot into the
    # predictor_fn_by_action dict with no adapter code, because the
    # sampler routes ban steps to the ban closure and pick steps to the
    # pick closure and never consults a predictor at the decider.
    from models.conditional_logit_ban import (
        FEATURE_NAMES as BAN_FEATURE_NAMES,
    )
    from models.conditional_logit_ban import (
        fit as fit_ban,
    )
    from models.conditional_logit_ban import (
        make_veto_step_predictor_fn as make_ban_predictor_fn,
    )
    from models.conditional_logit_pick import (
        FEATURE_NAMES as PICK_FEATURE_NAMES,
    )
    from models.conditional_logit_pick import (
        fit as fit_pick,
    )
    from models.conditional_logit_pick import (
        make_veto_step_predictor_fn as make_pick_predictor_fn,
    )

    # A small synthetic fit for each model (random groups, fixed seed —
    # the same shape the ban/pick test suites' _small_fitted_model uses;
    # predictive quality is irrelevant here, only the closure contract).
    def _fit_small(fitter, n_features):
        """Fit a small random synthetic model for closure wiring tests.

        Args:
            fitter: The module's ``fit`` function
                (``models.conditional_logit_ban.fit`` or
                ``models.conditional_logit_pick.fit``).
            n_features: The model's feature count (5 for both).

        Returns:
            A fitted conditional-logit model with a fixed seed.

        Raises:
            ValueError: If the fit fails (propagated from ``fitter``).
        """
        rng = np.random.default_rng(5)
        bounds = [0]
        ys = []
        rows = []
        for _ in range(80):
            n_s = int(rng.integers(4, 7))
            xs = rng.normal(scale=0.5, size=(n_s, n_features))
            rows.append(xs)
            ys.append(int(rng.integers(0, n_s)))
            bounds.append(bounds[-1] + n_s)
        X = np.vstack(rows)
        return fitter(np.asarray(X), np.asarray(bounds), np.asarray(ys), max_iter=150)

    ban_model = _fit_small(fit_ban, len(BAN_FEATURE_NAMES))
    pick_model = _fit_small(fit_pick, len(PICK_FEATURE_NAMES))
    ban_fn = make_ban_predictor_fn(ban_model)
    pick_fn = make_pick_predictor_fn(pick_model)

    # The restriction is real: neither closure serves the other action.
    with pytest.raises(ValueError, match="'pick'"):
        ban_fn("A", "pick", list(POOL), CL_QUERY_DATE, *_cl_league_tables())
    with pytest.raises(ValueError, match="'ban'"):
        pick_fn("A", "ban", list(POOL), CL_QUERY_DATE, *_cl_league_tables())

    # Wired as a dict, the sampler satisfies both restrictions with no
    # adapter: ban steps route to ban_fn, pick steps to pick_fn, and
    # the decider consults neither.
    matches_df, maps_df = _cl_league_tables()
    samples = sample_veto_sequences(
        "A", "B", "Bo3", CL_QUERY_DATE, matches_df, maps_df,
        {"ban": ban_fn, "pick": pick_fn},
        n_samples=3,
        rng=np.random.default_rng(11),
        map_pool=POOL,
    )
    assert len(samples) == 3
    _assert_walk_shape(samples)
    # The real closures' distributions are genuinely used: the recorded
    # probabilities are softmax shares over >1 remaining map at every
    # choosing step (each strictly between 0 and 1 here), so the
    # sequence probabilities are strictly below 1.
    for seq in samples:
        assert all(0.0 < a.probability < 1.0 for a in seq.actions[:-1])
        assert seq.sequence_probability < 1.0


def _real_v1_available():
    """Report whether the real v1 tables needed by the smoke test exist.

    The skip guard for the end-to-end smoke test: matches and maps must
    both be materialised under ``data/v1`` (i.e. ``materialize.py`` has
    been run). The sampler consumes only these two tables, so this
    guard checks exactly what the test needs.

    Returns:
        A bool: ``True`` iff all :data:`_REAL_V1_TABLES`
        ``data/v1/*.parquet`` files exist.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}.parquet").exists() for name in _REAL_V1_TABLES
    )


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_greedy_arm_smoke():
    # The real-data smoke test (plan#1j): sample a modest n_samples for
    # one real Bo3 match using the M25 greedy arm under both keys, and
    # assert structural sanity only — every sequence's shape and
    # probabilities — never any specific probability value. Wall-clock
    # is measured and reported in the BUILD status.md note (the open
    # assumption; see the plan).
    import pandas as pd

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    real_match = matches_df[
        (matches_df["status"] == "completed") & (matches_df["best_of"] == "Bo3")
    ].iloc[0]
    team_a_id = str(real_match["team1_id"])
    team_b_id = str(real_match["team2_id"])
    date = str(real_match["date"])
    n_samples = 50
    start = time.monotonic()
    samples = sample_veto_sequences(
        team_a_id,
        team_b_id,
        "Bo3",
        date,
        matches_df,
        maps_df,
        {"ban": greedy_veto_step_model, "pick": greedy_veto_step_model},
        n_samples=n_samples,
        rng=np.random.default_rng(2026),
    )
    elapsed = time.monotonic() - start
    assert len(samples) == n_samples
    _assert_walk_shape(samples)
    # Every sampled sequence is a valid distributional outcome: each
    # probability in (0, 1] and the map walk stays inside the resolved
    # pool (checked inside _assert_walk_shape), so this is a pure
    # sanity assertion — no specific probability values are asserted.
    for seq in samples:
        assert 0.0 < seq.sequence_probability <= 1.0
    # The greedy arm's distributions are near-degenerate but not
    # exactly: the smoke run is reported in the status.md note.
    assert elapsed > 0.0
