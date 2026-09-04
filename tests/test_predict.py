"""Tests for the M39 predict() public API (drivers/predict.py).

Covers the documented result dataclasses and their ``to_dict`` JSON
serialization; the :func:`drivers.predict.make_predictor` factory
contract (returns the 5-argument ``predict`` closure; missing
table/model artifact raises ``FileNotFoundError`` unchanged; the
temperature staleness guard fires through the registry factory);
a synthetic end-to-end ``predict`` run with the table loaders, the
Stage-2 registry factory, the Stage-1 artifact loader, the greedy
simulator, the map-win-rate estimator and the M31 veto-marginalized
entry point monkeypatched to fast deterministic stubs whose outputs are
hand-computable (so the ``PerMapPrediction`` shapes, the
``n_games_backing`` values, the ``SeriesPrediction`` aggregate and the
weighted-mean cross-check against the M31 aggregate are asserted
exactly), including the D6 per-call fresh-RNG idempotence wiring; the
D4 bootstrap-interval path (per-map epistemic bands landed from
hand-known replicate 4-vectors); the D5 veto-sensitivity path
(``VetoSensitivity`` fields match hand-computed percentile bands /
weighted moments over the same M31 sample detail); propagated error
paths (invalid ``best_of`` / wrong-size ``map_pool`` raise
``ValueError`` from the real greedy simulator); a ``main()`` CLI smoke
test with the predictor stubbed (JSON printed to stdout, exit 0); and
a ``skipif``-guarded real-v1 integration smoke test asserting finite,
well-formed output. No real fitted artifacts are required by the
non-smoke tests.
"""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drivers import predict as pred
from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    SeriesPrediction,
    VetoSensitivity,
)
from evaluation.veto_marginalized_series import (
    SeriesVetoSample,
    VetoMarginalizedSeriesPrediction,
)
from models import _shared
from models.ancestral_veto_sampler import (
    SampledVetoAction,
    SampledVetoSequence,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from models.ordinal_logit import OrdinalLogitModel
from utils import series_paths

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_SAMPLES = pred.DEFAULT_N_SAMPLES
DEFAULT_SEED = pred.DEFAULT_SEED
DEFAULT_CI_LEVEL = pred.DEFAULT_CI_LEVEL

# The 7-map pool the stub greedy veto / M31 stubs veto over. The map
# names double as the stub map-model fn's key (each played map gets the
# same hand-known 4-vector) and as the stub backing rule's key (see
# _stub_team_map_win_rate).
_STUB_POOL = (
    "Haven",
    "Split",
    "Bind",
    "Ascent",
    "Lotus",
    "Icebox",
    "Sunset",
)

# The stub Bo3 greedy veto: a full 7-action sequence (bans, then picks,
# then the forced decider) whose pick/decider maps — Bind, Ascent,
# Sunset in step order — are the played maps the predict() per-map
# section must iterate in play order.
_STUB_VETO_ACTIONS = (
    SimulatedVetoAction(0, "A", "ban", "Haven"),
    SimulatedVetoAction(1, "B", "ban", "Split"),
    SimulatedVetoAction(2, "A", "pick", "Bind"),
    SimulatedVetoAction(3, "B", "pick", "Ascent"),
    SimulatedVetoAction(4, "A", "ban", "Lotus"),
    SimulatedVetoAction(5, "B", "ban", "Icebox"),
    SimulatedVetoAction(6, None, "decider", "Sunset"),
)

# A tiny hand-built league. The columns mirror the real tables'
# materialised shapes (matches/maps carry the columns the as-of layer
# reads); team ids/dates are arbitrary — every stub model ignores the
# tables, and the error-path tests raise before any table access.
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo3", "status": "completed"},
]

_MAP_ROWS = [
    {"match_id": "m1", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 11, "winner": "A"},
    {"match_id": "m1", "map_index": 2, "map_name": "Split",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 0, "map_name": "Ascent",
     "team1_score": 13, "team2_score": 6, "winner": "C"},
    {"match_id": "m2", "map_index": 1, "map_name": "Sunset",
     "team1_score": 13, "team2_score": 7, "winner": "C"},
    {"match_id": "m2", "map_index": 2, "map_name": "Lotus",
     "team1_score": 5, "team2_score": 13, "winner": "D"},
]


def _league_tables():
    """Build the synthetic matches/maps/player_map_stats frames.

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple of
        ``pandas.DataFrame`` objects: the two tables from the module-
        level row constants, and an empty ``player_map_stats`` frame
        (the stub models never read it).

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
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    return matches_df, maps_df, player_map_stats_df


def _install_table_stubs(monkeypatch):
    """Patch make_predictor's three table loaders to the synthetic league.

    Replaces ``load_matches_table`` / ``load_maps_table`` /
    ``load_player_map_stats_table`` on the ``drivers.evaluate`` module
    (the namespace ``drivers.predict`` reads them through) with stubs
    returning the hand-built league frames, so a ``make_predictor``
    call needs no real parquet files on disk. Used by the tests that
    exercise the *real* Stage-2 registry factory (the missing-model-
    artifact and staleness-guard tests), where the registry and the
    artifact files must stay real while only the table I/O is stubbed.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        Nothing.

    Raises:
        Nothing.
    """
    matches_df, maps_df, player_map_stats_df = _league_tables()
    monkeypatch.setattr(
        pred.evaluate,
        "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        pred.evaluate,
        "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        pred.evaluate,
        "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )


def _install_table_and_model_stubs(monkeypatch):
    """Patch the tables plus the two fitted-model sources into stubs.

    Composes :func:`_install_table_stubs` with stubs for the Stage-2
    registry factory (returns :func:`_stub_map_model_fn`, the
    hand-known temperature-scaled 4-vector closure) and the Stage-1
    artifact loader (returns ``(None, None)`` — the
    ``make_veto_step_predictor_fn`` closures over ``None`` are never
    invoked on the synthetic paths, since the M31 entry point and/or
    the greedy simulator are stubbed wherever a full ``predict`` run
    happens). Lets the error-path tests build a working
    ``make_predictor`` while keeping the *greedy simulator real* so
    its propagated ``ValueError``s (invalid ``best_of``, wrong-size
    ``map_pool``) are exercised end to end.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        Nothing.

    Raises:
        Nothing.
    """
    _install_table_stubs(monkeypatch)
    monkeypatch.setitem(
        pred.evaluate.MODEL_REGISTRY,
        pred._TEMPERATURE_MAP_MODEL_KEY,
        lambda output_dir, version: _stub_map_model_fn,
    )
    monkeypatch.setattr(
        pred, "_load_veto_models", lambda output_dir, version: (None, None)
    )


def _synthetic_ordinal_model() -> OrdinalLogitModel:
    """Build a minimal but genuine frozen OrdinalLogitModel replicate.

    Constructs a real :class:`models.ordinal_logit.OrdinalLogitModel`
    with zero coefficients, the canonical 13-feature names, trivial
    thresholds and an identity standardizer — structurally valid (the
    ``from_dict`` shape checks pass) and never actually scored: the
    D4 interval path's ``make_model_fn`` is monkeypatched to route
    each model to a hand-known 4-vector by object identity, so the
    model's coefficients are irrelevant.

    Returns:
        An ``OrdinalLogitModel`` with ``feature_names`` equal to
        ``models._shared.FEATURE_NAMES`` (13 entries) and matching
        zero-coefficient arrays.

    Raises:
        Nothing.
    """
    n_features = len(_shared.FEATURE_NAMES)
    return OrdinalLogitModel(
        coefficients=np.zeros(n_features),
        thresholds=np.array([-1.0, 0.0, 1.0]),
        standardizer_means=np.zeros(n_features),
        standardizer_stds=np.ones(n_features),
        feature_names=tuple(_shared.FEATURE_NAMES),
        converged=True,
        n_iter=10,
        final_loss=1.0,
        n_train=10,
        l2_lambda=1.0,
    )


def _make_model_fn_with_tilts(tilts_by_id):
    """Build a make_model_fn stub routing each model to a tilted vector.

    Returns a replacement for ``models.ordinal_logit.make_model_fn``
    whose returned closure ignores the tables and returns the
    hand-known 4-vector ``(0.25 + d, 0.25 - d, 0.25, 0.25)`` where
    ``d`` is the tilt recorded for the model's ``id()`` — a valid
    simplex for every ``d`` — so the D4 interval path's per-replicate
    rows (and therefore the landed percentile bands) are
    hand-computable from the tilts.

    Args:
        tilts_by_id: A ``{id(model): tilt}`` dict mapping each
            bootstrap replicate model to its tilt.

    Returns:
        The ``make_model_fn(model, player_map_stats_df) -> closure``
        stub described above.

    Raises:
        Nothing.
    """

    def stub_make_model_fn(model, player_map_stats_df):
        d = tilts_by_id[id(model)]

        def stub_map_fn(
            team1_id, team2_id, map_name, date, matches_df, maps_df
        ):
            return (0.25 + d, 0.25 - d, 0.25, 0.25)

        return stub_map_fn

    return stub_make_model_fn


def _expected_interval_bands(tilts, ci_level):
    """Re-derive the expected per-category bands from the tilt rule.

    Independently recomputes, via numpy, the per-column percentile
    bands the real
    :func:`evaluation.bootstrap_intervals.replicate_matrix_intervals`
    computes over the tilt-rule replicate vectors ``(0.25 + d,
    0.25 - d, 0.25, 0.25)`` for the given tilts — the test's
    cross-check that the driver routed the right rows into the right
    helper.

    Args:
        tilts: The per-model tilts (one entry per replicate).
        ci_level: The band level in ``(0, 1)``.

    Returns:
        A ``(lo, hi)`` tuple of per-category band endpoints, each a
        tuple of 4 floats.

    Raises:
        Nothing.
    """
    matrix = np.asarray(
        [[0.25 + d, 0.25 - d, 0.25, 0.25] for d in tilts], dtype=float
    )
    lo = np.percentile(matrix, (1.0 - ci_level) / 2.0 * 100.0, axis=0)
    hi = np.percentile(matrix, (1.0 + ci_level) / 2.0 * 100.0, axis=0)
    return tuple(float(x) for x in lo), tuple(float(x) for x in hi)


def _expected_unweighted_bands(n_samples, ci_level):
    """Re-derive the stub M31 rows' unweighted percentile bands.

    Independently recomputes, via numpy, the per-column percentile
    bands :func:`_stub_prediction`'s hand-known rows ``[0.50 - 0.05*i,
    0.30 + 0.05*i, 0.10, 0.10]`` (for ``i`` in ``range(n_samples)``)
    produce at ``ci_level`` — the cross-check for the D5
    ``veto_sensitivity.unweighted_band_*`` fields.

    Args:
        n_samples: How many stub rows (one per sample).
        ci_level: The band level in ``(0, 1)``.

    Returns:
        A ``(lo, hi)`` tuple of per-category band endpoints, each a
        tuple of 4 floats.

    Raises:
        Nothing.
    """
    matrix = np.asarray(
        [
            [0.50 - 0.05 * i, 0.30 + 0.05 * i, 0.10, 0.10]
            for i in range(n_samples)
        ],
        dtype=float,
    )
    lo = np.percentile(matrix, (1.0 - ci_level) / 2.0 * 100.0, axis=0)
    hi = np.percentile(matrix, (1.0 + ci_level) / 2.0 * 100.0, axis=0)
    return tuple(float(x) for x in lo), tuple(float(x) for x in hi)


def _expected_weighted_moments(n_samples):
    """Re-derive the stub M31 rows' weighted mean and variance.

    Independently recomputes, via numpy's weighted ``average``, the
    per-category weighted first/second moments over
    :func:`_stub_prediction`'s hand-known rows with the cycling
    weights ``[0.5, 0.3, 0.2]`` — the cross-check for the D5
    ``veto_sensitivity.weighted_mean`` / ``weighted_variance`` fields.

    Args:
        n_samples: How many stub rows (one per sample).

    Returns:
        A ``(means, variances)`` tuple of per-category moment tuples,
        each of 4 floats.

    Raises:
        Nothing.
    """
    matrix = np.asarray(
        [
            [0.50 - 0.05 * i, 0.30 + 0.05 * i, 0.10, 0.10]
            for i in range(n_samples)
        ],
        dtype=float,
    )
    weights = np.asarray(
        [[0.5, 0.3, 0.2][i % 3] for i in range(n_samples)], dtype=float
    )
    means = np.average(matrix, axis=0, weights=weights)
    variances = np.average(
        (matrix - means) ** 2, axis=0, weights=weights
    )
    return (
        tuple(float(x) for x in means),
        tuple(float(x) for x in variances),
    )


def _ordinal_artifact_dict(thresholds=None):
    """Build a valid serialized ordinal-logit artifact dict.

    Produces the plain dict :func:`models.ordinal_logit.from_dict`
    accepts: zero 13-vector coefficients aligned with
    :data:`models._shared.FEATURE_NAMES`, an identity standardizer,
    the default thresholds ``(-1.0, 0.0, 1.0)`` (overridable, for the
    staleness-guard test) and plain diagnostic scalars — so tests of
    the *real* Stage-2 registry factory can write a genuine
    ``ordinal_logit_model.json`` into a temp dir without real v1
    artifacts.

    Args:
        thresholds: The 3 thresholds to serialize; ``None`` (default)
            uses ``(-1.0, 0.0, 1.0)``.

    Returns:
        The artifact dict.

    Raises:
        Nothing.
    """
    n_features = len(_shared.FEATURE_NAMES)
    return {
        "feature_names": list(_shared.FEATURE_NAMES),
        "coefficients": [0.0] * n_features,
        "thresholds": (
            [-1.0, 0.0, 1.0] if thresholds is None else list(thresholds)
        ),
        "standardizer_means": [0.0] * n_features,
        "standardizer_stds": [1.0] * n_features,
        "converged": True,
        "n_iter": 10,
        "final_loss": 1.0,
        "n_train": 10,
        "l2_lambda": 1.0,
    }


def _temperature_artifact_dict(thresholds=None):
    """Build a valid serialized temperature-scaling artifact dict.

    Produces the plain dict :func:`models.temperature_scaling.from_dict`
    accepts: a positive temperature, the default thresholds
    ``(-1.0, 0.0, 1.0)`` (overridable — the staleness guard compares
    this stored copy against the loaded base model's thresholds) and
    the remaining scalar/dict metadata fields — so tests of the *real*
    Stage-2 registry factory can write a genuine
    ``temperature_scaling_model.json`` into a temp dir.

    Args:
        thresholds: The 3 stored thresholds; ``None`` (default) uses
            ``(-1.0, 0.0, 1.0)`` (matching
            :func:`_ordinal_artifact_dict`'s default, so the guard
            passes).

    Returns:
        The artifact dict.

    Raises:
        Nothing.
    """
    return {
        "temperature": 1.5,
        "thresholds": (
            [-1.0, 0.0, 1.0] if thresholds is None else list(thresholds)
        ),
        "n_calibration": 10,
        "oof_coverage": {},
        "t_grid_min": 0.05,
        "t_grid_max": 20.0,
        "calibration_nll_at_t1": 1.0,
        "calibration_nll_at_t_star": 0.9,
    }


def _stub_sequence(
    best_of: str, team1_id: str, team2_id: str, date: str, index: int
) -> SampledVetoSequence:
    """Build a minimal but real SampledVetoSequence for the stubs.

    Constructs a genuine :class:`models.ancestral_veto_sampler
    .SampledVetoSequence` carrying one synthetic pick action (enough
    for the frozen-dataclass contract — the predict() sensitivity path
    never reads the raw sequence fields, only the ``SeriesVetoSample``
    reporting fields, but constructing the real type keeps the stub's
    shape honest).

    Args:
        best_of: The ``"Bo<N>"`` series-length string.
        team1_id: Side A's stable id.
        team2_id: Side B's stable id.
        date: The as-of date string.
        index: A per-sample index, only to make the pick's map name
            unique across samples.

    Returns:
        A ``SampledVetoSequence`` with ``sequence_probability`` equal
        to ``1.0`` (a trivial non-degenerate walk).

    Raises:
        Nothing.
    """
    return SampledVetoSequence(
        team_a_id=team1_id,
        team_b_id=team2_id,
        best_of=best_of,
        date=date,
        actions=(
            SampledVetoAction(
                step_index=0,
                team=team1_id,
                action="pick",
                map_name=f"stub{index}",
                probability=1.0,
            ),
        ),
        sequence_probability=1.0,
    )


def _stub_prediction(
    team1_id: str,
    team2_id: str,
    best_of: str,
    date: str,
    n_samples: int,
) -> VetoMarginalizedSeriesPrediction:
    """Build a hand-computable M31 prediction for the stub runs.

    Produces the aggregated probabilities and the per-sample scoreline
    detail the predict() wiring reads, with values chosen so the
    resulting spread is hand-computable: for a Bo3 with ``n_samples``
    samples, sample ``i``'s scoreline vector is ``[0.50 - 0.05*i,
    0.30 + 0.05*i, 0.10, 0.10]`` (a valid simplex for every ``i``)
    with weights cycling ``[0.5, 0.3, 0.2]``, so the aggregated
    probabilities (the weighted mean — exactly the M31 aggregation
    definition) are hand-known and the sensitivity bands are the
    percentiles over the hand-known rows.

    Args:
        team1_id: Side A's stable id (embedded in the sample
            sequences).
        team2_id: Side B's stable id.
        best_of: The ``"Bo<N>"`` series-length string.
        date: The as-of date string (embedded in the sample sequences).
        n_samples: How many samples to produce.

    Returns:
        A ``VetoMarginalizedSeriesPrediction`` with ``n_samples``
        ``SeriesVetoSample`` records carrying the hand-computable
        vectors and weights above, whose ``probabilities`` are the
        weighted mean of the rows and whose ``outcome_order`` is the
        canonical ``utils.series_paths.series_outcome_order``
        vocabulary.

    Raises:
        ValueError: If ``best_of`` is not a supported ``"Bo<N>"``
            string (from the ``int`` slice — only Bo3 rows are defined
            here; callers must pass ``"Bo3"``).
    """
    best_of_int = int(best_of[2:])
    rows = [
        [0.50 - 0.05 * i, 0.30 + 0.05 * i, 0.10, 0.10]
        for i in range(n_samples)
    ]
    weights = [[0.5, 0.3, 0.2][i % 3] for i in range(n_samples)]
    total = sum(weights)
    aggregated = [
        sum(w * row[j] for w, row in zip(weights, rows)) / total
        for j in range(best_of_int + 1)
    ]
    samples = tuple(
        SeriesVetoSample(
            sequence=_stub_sequence(best_of, team1_id, team2_id, date, i),
            weight=float(weights[i]),
            played_maps=tuple(f"map{j}" for j in range(best_of_int)),
            per_map_four_way=((0.25, 0.25, 0.25, 0.25),) * best_of_int,
            per_map_win_prob=(0.5,) * best_of_int,
            scoreline_probabilities=tuple(rows[i]),
        )
        for i in range(n_samples)
    )
    return VetoMarginalizedSeriesPrediction(
        probabilities=tuple(aggregated),
        best_of=best_of_int,
        outcome_order=series_paths.series_outcome_order(best_of_int),
        samples=samples,
    )


def _stub_simulate_veto(
    team_a_id,
    team_b_id,
    best_of,
    date,
    matches_df,
    maps_df,
    k,
    map_pool=None,
):
    """Stub the greedy simulator with the fixed hand-known veto sequence.

    Ignores every input and returns the module-level
    :data:`_STUB_VETO_ACTIONS` tuple (a full Bo3 veto whose
    pick/decider maps are Bind, Ascent, Sunset), asserting the driver
    passed the shared :data:`features.map_win_rate.DEFAULT_K` (the D7
    wiring check) — exactly like the real
    :func:`models.greedy_veto_simulator.simulate_veto` contract.

    Args:
        team_a_id / team_b_id / best_of / date / matches_df / maps_df:
            Unused by the stub (signature parity).
        k: The shrinkage strength; asserted equal to
            ``map_win_rate.DEFAULT_K``.
        map_pool: Unused by the stub.

    Returns:
        The :data:`_STUB_VETO_ACTIONS` tuple, wrapped in a ``list``
        like the real simulator's return type.

    Raises:
        AssertionError: If ``k`` is not the shared default.
    """
    assert k == 10.0
    return list(_STUB_VETO_ACTIONS)


def _stub_team_map_win_rate(
    team_id, map_name, date, matches_df, maps_df, k
):
    """Stub features.map_win_rate.team_map_win_rate with hand-known games.

    Returns a ``SimpleNamespace`` carrying a deterministic ``games``
    count per (team, map): team ``"A"`` has 10 games everywhere except
    3 on ``"Sunset"``; team ``"B"`` has 4 everywhere except 7 on
    ``"Bind"``. Every other team has 0 games. Also asserts the driver
    passed ``map_win_rate.DEFAULT_K`` (the wiring check that
    ``predict`` uses the shared default), mirroring the real
    estimator's contract.

    Args:
        team_id: The queried team id.
        map_name: The queried map.
        date / matches_df / maps_df: Unused by the stub.
        k: The shrinkage strength; asserted equal to
            ``map_win_rate.DEFAULT_K``.

    Returns:
        A ``SimpleNamespace(games=int)`` carrying the deterministic
        count.

    Raises:
        AssertionError: If ``k`` is not the shared default.
    """
    assert k == 10.0
    if team_id == "A":
        games = 3 if map_name == "Sunset" else 10
    elif team_id == "B":
        games = 7 if map_name == "Bind" else 4
    else:
        games = 0
    return SimpleNamespace(games=games)


def _stub_map_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """Stub the Stage-2 four-way map model with a hand-known 4-vector.

    Returns the fixed vector ``(0.6, 0.1, 0.1, 0.2)`` (in
    :data:`models._shared.OUTCOME_LABELS` order, summing to exactly
    1.0) for every played map, so the per-map point probabilities are
    hand-assertable.

    Args:
        team1_id / team2_id / map_name / date / matches_df / maps_df:
            Unused by the stub (signature parity).

    Returns:
        The 4-tuple ``(0.6, 0.1, 0.1, 0.2)``.

    Raises:
        Nothing.
    """
    return (0.6, 0.1, 0.1, 0.2)


@pytest.fixture
def stub_predictor_wiring(monkeypatch):
    """Monkeypatch make_predictor's loaders/models into deterministic stubs.

    Routes the three input tables to the synthetic league, replaces the
    Stage-2 registry factory (returns :func:`_stub_map_model_fn`), the
    Stage-1 artifact loader (returns ``(None, None)`` — the
    ``make_veto_step_predictor_fn`` closures over ``None`` are never
    called, since the M31 entry point is stubbed), the map-win-rate
    estimator, the greedy simulator and the M31 veto-marginalized entry
    point with the hand-computable stubs above, and installs call state
    (exposed on the returned dict) so tests can verify the driver
    called the M31 entry point exactly once per ``predict`` call. The
    D6 per-call fresh-RNG mechanism (identical calls reproduce
    identical output) is asserted by the calling tests via a
    ``numpy.random.default_rng`` call-counting wrapper rather than by
    object identities, which CPython's allocator can reuse. The real
    spread/interval helpers in ``evaluation.veto_conditional_variance``
    / ``evaluation.bootstrap_intervals`` run untouched. All patches are
    reverted by monkeypatch at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        A dict of the call state: ``predict_calls`` (int).

    Raises:
        Nothing.
    """
    call_state = {"predict_calls": 0}

    def stub_m31_entry(
        team1_id,
        team2_id,
        best_of,
        date,
        matches_df,
        maps_df,
        map_model_fn,
        predictor_fn_by_action,
        n_samples,
        rng,
        map_pool=None,
    ):
        call_state["predict_calls"] += 1
        assert callable(map_model_fn)
        assert set(predictor_fn_by_action) == {"ban", "pick"}
        assert isinstance(rng, np.random.Generator)
        assert n_samples > 0
        assert tuple(map_pool) == tuple(_STUB_POOL) or map_pool is None
        return _stub_prediction(
            team1_id, team2_id, best_of, date, n_samples
        )

    _install_table_and_model_stubs(monkeypatch)
    monkeypatch.setattr(
        pred.map_win_rate, "team_map_win_rate", _stub_team_map_win_rate
    )
    monkeypatch.setattr(
        pred.greedy_veto_simulator, "simulate_veto", _stub_simulate_veto
    )
    monkeypatch.setattr(
        pred.veto_marginalized_series,
        "predict_series_outcome_via_veto_marginalization",
        stub_m31_entry,
    )
    return call_state


# --------------------------------------------------------------------------
# plan#9a: result-dataclass to_dict() JSON serializability
# --------------------------------------------------------------------------


def test_result_dataclasses_to_dict_json_serializable():
    # Every result dataclass (constructed directly with hand-known
    # values, including a real SimulatedVetoAction in the
    # PredictionResult) serializes to plain JSON types that
    # json.dumps round-trips: the veto actions go through
    # SimulatedVetoAction.to_dict, per_map/series/veto_sensitivity nest
    # their own to_dict outputs, and interval_* is None when no
    # bootstrap models backed it.
    per_map = PerMapPrediction(
        map_name="Bind",
        probabilities=(0.6, 0.1, 0.1, 0.2),
        interval_low=None,
        interval_high=None,
        n_games_backing=7,
    )
    per_map_with_interval = PerMapPrediction(
        map_name="Ascent",
        probabilities=(0.5, 0.1, 0.1, 0.3),
        interval_low=(0.4, 0.05, 0.05, 0.2),
        interval_high=(0.6, 0.15, 0.15, 0.4),
        n_games_backing=3,
    )
    series = SeriesPrediction(
        probabilities=(0.5, 0.3, 0.1, 0.1),
        outcome_order=((2, 0), (2, 1), (1, 2), (0, 2)),
        best_of=3,
    )
    sensitivity = VetoSensitivity(
        unweighted_band_low=(0.4, 0.3, 0.1, 0.1),
        unweighted_band_high=(0.6, 0.4, 0.1, 0.1),
        band_widths=(0.2, 0.1, 0.0, 0.0),
        mean_band_width=0.075,
        weighted_mean=(0.5, 0.3, 0.1, 0.1),
        weighted_variance=(0.01, 0.01, 0.0, 0.0),
    )
    result = PredictionResult(
        predicted_veto=_STUB_VETO_ACTIONS,
        per_map=(per_map, per_map_with_interval),
        series=series,
        veto_sensitivity=sensitivity,
    )

    per_map_dict = per_map.to_dict()
    assert per_map_dict == {
        "map_name": "Bind",
        "probabilities": [0.6, 0.1, 0.1, 0.2],
        "interval_low": None,
        "interval_high": None,
        "n_games_backing": 7,
    }
    assert per_map_with_interval.to_dict()["interval_low"] == [
        0.4, 0.05, 0.05, 0.2
    ]
    assert series.to_dict() == {
        "probabilities": [0.5, 0.3, 0.1, 0.1],
        "outcome_order": [[2, 0], [2, 1], [1, 2], [0, 2]],
        "best_of": 3,
    }
    assert sensitivity.to_dict()["weighted_mean"] == [0.5, 0.3, 0.1, 0.1]

    result_dict = result.to_dict()
    assert set(result_dict) == {
        "predicted_veto", "per_map", "series", "veto_sensitivity"
    }
    # The nested per-map to_dict round-trips through JSON.
    assert json.loads(json.dumps(result_dict)) == result_dict
    assert result_dict["predicted_veto"][0] == {
        "step_index": 0,
        "team": "A",
        "action": "ban",
        "map_name": "Haven",
    }
    assert result_dict["predicted_veto"][-1] == {
        "step_index": 6,
        "team": None,
        "action": "decider",
        "map_name": "Sunset",
    }
    assert len(result_dict["per_map"]) == 2
    assert result_dict["series"]["best_of"] == 3
    assert result_dict["veto_sensitivity"]["mean_band_width"] == 0.075


# --------------------------------------------------------------------------
# plan#9b: make_predictor factory contract
# --------------------------------------------------------------------------


def test_make_predictor_returns_5_arg_callable(tmp_path, stub_predictor_wiring):
    # make_predictor returns a callable whose signature is exactly the
    # documented 5-arg public API — (team_a, team_b, best_of, map_pool,
    # as_of_date) — closing over the (stubbed) tables/models.
    predictor = pred.make_predictor("data", "v1", n_samples=3)
    assert callable(predictor)
    parameters = list(inspect.signature(predictor).parameters)
    assert parameters == ["team_a", "team_b", "best_of", "map_pool", "as_of_date"]


def test_make_predictor_defaults_match_documented(tmp_path, stub_predictor_wiring):
    # The factory keyword defaults are the documented D7 constants —
    # referenced through the module constants so a stale hardcode can
    # never drift silently.
    signature = inspect.signature(pred.make_predictor)
    assert signature.parameters["n_samples"].default == DEFAULT_N_SAMPLES
    assert signature.parameters["seed"].default == DEFAULT_SEED
    assert signature.parameters["ci_level"].default == DEFAULT_CI_LEVEL
    assert signature.parameters["bootstrap_models"].default is None


def test_make_predictor_rejects_bad_knobs(tmp_path, stub_predictor_wiring):
    # n_samples < 1 and an out-of-(0, 1) ci_level are hard errors at
    # factory time, before any table/artifact load.
    with pytest.raises(ValueError, match="n_samples"):
        pred.make_predictor("data", "v1", n_samples=0)
    with pytest.raises(ValueError, match="ci_level"):
        pred.make_predictor("data", "v1", ci_level=1.5)


def test_make_predictor_missing_table_raises_file_not_found(tmp_path):
    # No tables exist under the empty tmp dir (no stubs): the first
    # table load must raise FileNotFoundError unchanged — the "run the
    # prerequisite first" signal, never a silent fallback.
    with pytest.raises(FileNotFoundError):
        pred.make_predictor(tmp_path, "v1")


def test_make_predictor_missing_model_artifact_raises_file_not_found(
    tmp_path, monkeypatch
):
    # Tables exist (stubbed loaders) but no fitted model artifacts are
    # under tmp_path/v1: the real Stage-2 registry factory reads
    # ordinal_logit_model.json and raises FileNotFoundError unchanged —
    # proving the model-artifact prerequisite flows through
    # make_predictor's registry wiring.
    matches_df, maps_df, player_map_stats_df = _league_tables()
    monkeypatch.setattr(
        pred.evaluate,
        "load_matches_table",
        lambda output_dir, version: matches_df,
    )
    monkeypatch.setattr(
        pred.evaluate,
        "load_maps_table",
        lambda output_dir, version: maps_df,
    )
    monkeypatch.setattr(
        pred.evaluate,
        "load_player_map_stats_table",
        lambda output_dir, version: player_map_stats_df,
    )
    with pytest.raises(FileNotFoundError):
        pred.make_predictor(tmp_path, "v1")


# --------------------------------------------------------------------------
# plan#9c: synthetic predict() shapes and wiring
# --------------------------------------------------------------------------


def test_synthetic_predict_shapes_and_wiring(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # A full synthetic predict() run against the stubbed league: the
    # result carries the full 7-action predicted veto; three per-map
    # entries in play order (Bind, Ascent, Sunset) each with the
    # hand-known temperature-scaled 4-vector (summing to 1), None
    # intervals (no bootstrap models), and the hand-known
    # n_games_backing min-games values (7/4/3); and a SeriesPrediction
    # whose probabilities equal the stub M31 aggregate exactly
    # (0.465/0.335/0.1/0.1), whose best_of/outcome_order match the
    # canonical Bo3 vocabulary. The M31 entry point was called exactly
    # once. A second identical call reproduces identical output (D6)
    # and reconstructs a fresh per-call rng (the default_rng call
    # count is 2) — the per-call fresh-RNG mechanism.
    call_state = stub_predictor_wiring
    rng_constructs = {"count": 0}
    real_default_rng = pred.np.random.default_rng

    def counting_default_rng(seed):
        rng_constructs["count"] += 1
        return real_default_rng(seed)

    monkeypatch.setattr(
        pred.np.random, "default_rng", counting_default_rng
    )
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")

    # (a) The full deterministic veto sequence, in step order.
    assert result.predicted_veto == _STUB_VETO_ACTIONS
    assert [a.action for a in result.predicted_veto] == [
        "ban", "ban", "pick", "pick", "ban", "ban", "decider"
    ]

    # (b) Per-map entries in play order: the picks ascending then the
    # decider (Bind, Ascent, Sunset).
    assert [entry.map_name for entry in result.per_map] == [
        "Bind", "Ascent", "Sunset"
    ]
    for entry in result.per_map:
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)
        assert sum(entry.probabilities) == pytest.approx(1.0)
        assert entry.interval_low is None
        assert entry.interval_high is None
    assert [entry.n_games_backing for entry in result.per_map] == [7, 4, 3]

    # (c) The series aggregate equals the stub M31 aggregate (the
    # weighted mean 0.465/0.335/0.1/0.1), with the canonical Bo3
    # vocabulary.
    assert result.series.best_of == 3
    assert result.series.outcome_order == series_paths.series_outcome_order(3)
    assert result.series.probabilities == pytest.approx(
        [0.465, 0.335, 0.1, 0.1]
    )
    # The veto_sensitivity weighted mean over the same sample detail
    # must equal the M31 weighted-average aggregate exactly (the
    # cross-validation the M37 driver's test makes).
    assert result.veto_sensitivity.weighted_mean == pytest.approx(
        result.series.probabilities
    )
    assert len(result.veto_sensitivity.unweighted_band_low) == 4
    assert call_state["predict_calls"] == 1

    # JSON serializability of the full result.
    round_tripped = json.loads(json.dumps(result.to_dict()))
    assert round_tripped["per_map"][0]["map_name"] == "Bind"
    assert round_tripped["per_map"][0]["interval_low"] is None
    assert round_tripped["series"]["probabilities"] == pytest.approx(
        [0.465, 0.335, 0.1, 0.1]
    )

    # (D6) A second identical call reproduces identical output and
    # reconstructs a fresh rng (two default_rng constructions for two
    # calls — object identities are NOT compared, since CPython's
    # allocator can reuse a freed object's address).
    result2 = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")
    assert call_state["predict_calls"] == 2
    assert rng_constructs["count"] == 2
    assert result2.to_dict() == result.to_dict()


# --------------------------------------------------------------------------
# plan#10a: D4 bootstrap-interval path (per-map epistemic bands)
# --------------------------------------------------------------------------


def test_per_map_interval_bands_from_bootstrap_models(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # D4: with three synthetic OrdinalLogitModel replicates (and
    # make_model_fn monkeypatched to route each model to the hand-known
    # tilted vector (0.25+d, 0.25-d, 0.25, 0.25)), every played map's
    # interval_low/interval_high must be the per-category percentile
    # bands the real replicate_matrix_intervals computes over those
    # vectors — landed by predict(), not None. The point probabilities
    # stay the temperature-scaled stub vector (0.6, 0.1, 0.1, 0.2)
    # while the bands center on the raw-ordinal 0.25: the documented
    # D3/D4 asymmetry.
    tilts = [0.0, 0.02, 0.04]
    models = [_synthetic_ordinal_model() for _ in tilts]
    tilts_by_id = {id(model): d for model, d in zip(models, tilts)}
    monkeypatch.setattr(
        pred.ordinal_logit,
        "make_model_fn",
        _make_model_fn_with_tilts(tilts_by_id),
    )
    lo, hi = _expected_interval_bands(tilts, ci_level=0.9)

    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=models,
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")

    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)
        assert entry.interval_low == pytest.approx(lo)
        assert entry.interval_high == pytest.approx(hi)
        for low, high in zip(entry.interval_low, entry.interval_high):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_per_map_intervals_none_without_bootstrap_models(
    tmp_path, stub_predictor_wiring
):
    # D4 negative path: without bootstrap_models the interval fields are
    # None on every per-map entry (no epistemic interval) while the
    # point probabilities and backing are still populated — the CLI
    # path (bootstrap models are a caller/library concern).
    predictor = pred.make_predictor("data", "v1", n_samples=3)
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")
    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.interval_low is None
        assert entry.interval_high is None
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)


# --------------------------------------------------------------------------
# plan#10b: D5 veto_sensitivity path (hand-computed spread)
# --------------------------------------------------------------------------


def test_veto_sensitivity_matches_hand_computed_spread(
    tmp_path, stub_predictor_wiring
):
    # D5: the veto_sensitivity must be the M37 summary of the SAME M31
    # per-sample detail that produced series.probabilities. With
    # n_samples=3 and ci_level=0.9 the stub rows are
    # [0.50-0.05i, 0.30+0.05i, 0.10, 0.10] with weights cycling
    # [0.5, 0.3, 0.2]; the unweighted bands / band widths / mean width
    # and the weighted mean/variance must match an independent numpy
    # re-derivation of those same rows, and the weighted mean must
    # equal the series aggregate (the M31 aggregation definition).
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")

    lo, hi = _expected_unweighted_bands(n_samples=3, ci_level=0.9)
    weighted_means, weighted_variances = _expected_weighted_moments(
        n_samples=3
    )
    widths = tuple(hi_i - lo_i for lo_i, hi_i in zip(lo, hi))

    sensitivity = result.veto_sensitivity
    assert sensitivity.unweighted_band_low == pytest.approx(lo)
    assert sensitivity.unweighted_band_high == pytest.approx(hi)
    assert sensitivity.band_widths == pytest.approx(widths)
    assert sensitivity.mean_band_width == pytest.approx(
        sum(widths) / len(widths)
    )
    assert sensitivity.weighted_mean == pytest.approx(weighted_means)
    # The weighted mean over the samples equals the M31 weighted
    # average aggregate (the cross-validation the M37 driver makes).
    assert sensitivity.weighted_mean == pytest.approx(
        result.series.probabilities
    )
    assert sensitivity.weighted_variance == pytest.approx(weighted_variances)


def test_veto_sensitivity_all_samples_identical_zero_width(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # The "resolves the moment the veto happens" boundary case: when
    # the M31 sample set is degenerate (every sample carries the
    # identical scoreline vector), every band collapses to a point and
    # mean_band_width is exactly 0.0. The stub predict call's cycling
    # rows are not degenerate, so instead we patch the M31 stub to a
    # one-sample Bo3 prediction with a single row — the minimal
    # deterministic case.
    def single_sample_stub(
        team1_id,
        team2_id,
        best_of,
        date,
        matches_df,
        maps_df,
        map_model_fn,
        predictor_fn_by_action,
        n_samples,
        rng,
        map_pool=None,
    ):
        return _stub_prediction(team1_id, team2_id, best_of, date, 1)

    monkeypatch.setattr(
        pred.veto_marginalized_series,
        "predict_series_outcome_via_veto_marginalization",
        single_sample_stub,
    )
    predictor = pred.make_predictor(
        "data", "v1", n_samples=1, seed=2026, ci_level=0.9
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")

    assert result.veto_sensitivity.mean_band_width == 0.0
    assert all(w == 0.0 for w in result.veto_sensitivity.band_widths)
    assert result.veto_sensitivity.weighted_mean == pytest.approx(
        result.series.probabilities
    )


# --------------------------------------------------------------------------
# plan#11a: propagated error paths (real greedy simulator)
# --------------------------------------------------------------------------


def test_predict_invalid_best_of_propagates_value_error(
    tmp_path, monkeypatch
):
    # best_of outside the supported Bo1/Bo3/Bo5 keys propagates the
    # greedy simulator's ValueError unchanged ("not a supported veto
    # format") — the simulator runs REAL here (only the tables/models
    # are stubbed), so the propagation is exercised end to end.
    _install_table_and_model_stubs(monkeypatch)
    predictor = pred.make_predictor("data", "v1", n_samples=2)
    with pytest.raises(ValueError, match="not a supported veto format"):
        predictor("A", "B", "Bo7", _STUB_POOL, "2026-01-01T00:00:00")


def test_predict_wrong_size_map_pool_propagates_value_error(
    tmp_path, monkeypatch
):
    # A non-7 map_pool propagates the greedy simulator's ValueError
    # unchanged ("a Bo3 veto needs 7") — the D8 fail-loud clause, not
    # re-validated by predict itself.
    _install_table_and_model_stubs(monkeypatch)
    predictor = pred.make_predictor("data", "v1", n_samples=2)
    with pytest.raises(ValueError, match="needs 7"):
        predictor(
            "A", "B", "Bo3", ("Bind", "Haven", "Split"), "2026-01-01T00:00:00"
        )


# --------------------------------------------------------------------------
# plan#11b: staleness guard fires through the registry factory
# --------------------------------------------------------------------------


def test_temperature_staleness_guard_raises_value_error(tmp_path, monkeypatch):
    # The M24 staleness guard fires through make_predictor's registry
    # wiring: with real base/temperature artifacts on disk (tables
    # stubbed) whose stored temperature thresholds do NOT match the
    # loaded base model's thresholds, the registry factory raises
    # ValueError ("re-run train_temperature_scaling.py") at
    # make_predictor time — never a silent application of a stale T.
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    # Tampered: stored thresholds [0.0, 1.0, 2.0] vs the base's
    # [-1.0, 0.0, 1.0].
    (version_dir / "temperature_scaling_model.json").write_text(
        json.dumps(_temperature_artifact_dict(thresholds=[0.0, 1.0, 2.0])),
        encoding="utf-8",
    )
    _install_table_stubs(monkeypatch)
    with pytest.raises(ValueError, match="re-run train_temperature_scaling"):
        pred.make_predictor(tmp_path, "v1")


def test_temperature_staleness_guard_passes_on_matching_thresholds(
    tmp_path, monkeypatch
):
    # Positive control for the guard: with matching thresholds the
    # registry factory loads successfully (and make_predictor then
    # fails only on the missing Stage-1 ban/pick artifacts — proving
    # the guard itself did not raise).
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    (version_dir / "temperature_scaling_model.json").write_text(
        json.dumps(_temperature_artifact_dict()), encoding="utf-8"
    )
    _install_table_stubs(monkeypatch)
    with pytest.raises(FileNotFoundError, match="conditional_logit_ban"):
        pred.make_predictor(tmp_path, "v1")


# --------------------------------------------------------------------------
# plan#11c: thin CLI — parse_args defaults/overrides and main() JSON print
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: required query args are required (argparse SystemExit
    # when absent); the optional knobs carry the documented defaults.
    with pytest.raises(SystemExit):
        pred.parse_args([])
    args = pred.parse_args(
        ["--team-a", "A", "--team-b", "B", "--best-of", "Bo3",
         "--as-of-date", "2026-01-01T00:00:00"]
    )
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.map_pool is None
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED
    assert args.ci_level == DEFAULT_CI_LEVEL


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int counts/seeds and
    # non-float --ci-level are rejected by argparse (SystemExit), and a
    # --best-of outside Bo1/Bo3/Bo5 is rejected by choices=.
    args = pred.parse_args(
        ["--version", "v2", "--output-dir", "out",
         "--team-a", "A", "--team-b", "B", "--best-of", "Bo5",
         "--map-pool", "Bind,Haven,Split,Ascent,Lotus,Icebox,Sunset",
         "--as-of-date", "2026-01-01T00:00:00",
         "--n-samples", "2", "--seed", "7", "--ci-level", "0.8"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.team_a == "A"
    assert args.best_of == "Bo5"
    assert args.map_pool == "Bind,Haven,Split,Ascent,Lotus,Icebox,Sunset"
    assert args.n_samples == 2
    assert args.seed == 7
    assert args.ci_level == 0.8
    with pytest.raises(SystemExit):
        pred.parse_args(["--team-a", "A", "--team-b", "B",
                         "--best-of", "Bo7", "--as-of-date", "d"])
    with pytest.raises(SystemExit):
        pred.parse_args(["--team-a", "A", "--team-b", "B",
                         "--best-of", "Bo3", "--as-of-date", "d",
                         "--n-samples", "many"])


def test_main_rejects_malformed_map_pool():
    # A --map-pool with an empty comma-separated part (leading/trailing/
    # double comma) is a hard ValueError before any table/artifact load.
    with pytest.raises(ValueError, match="empty"):
        pred.main(
            ["--team-a", "A", "--team-b", "B", "--best-of", "Bo3",
             "--as-of-date", "2026-01-01T00:00:00",
             "--map-pool", "Bind,,Haven,Split,Ascent,Lotus,Icebox,Sunset"]
        )


def test_main_prints_json_result(capsys, monkeypatch):
    # main() builds the predictor, calls predict once and prints the
    # JSON-serialized result (indent=2, sorted keys) to stdout with a
    # one-line log summary; exit 0. The predictor factory is stubbed to
    # a canned result so no tables/artifacts are needed.
    def stub_make_predictor(
        output_dir,
        version,
        *,
        n_samples=pred.DEFAULT_N_SAMPLES,
        seed=pred.DEFAULT_SEED,
        ci_level=pred.DEFAULT_CI_LEVEL,
        bootstrap_models=None,
    ):
        def stub_predict(team_a, team_b, best_of, map_pool, as_of_date):
            return PredictionResult(
                predicted_veto=_STUB_VETO_ACTIONS,
                per_map=(
                    PerMapPrediction(
                        map_name="Bind",
                        probabilities=(0.6, 0.1, 0.1, 0.2),
                        interval_low=None,
                        interval_high=None,
                        n_games_backing=7,
                    ),
                ),
                series=SeriesPrediction(
                    probabilities=(0.5, 0.3, 0.1, 0.1),
                    outcome_order=series_paths.series_outcome_order(3),
                    best_of=3,
                ),
                veto_sensitivity=VetoSensitivity(
                    unweighted_band_low=(0.4, 0.3, 0.1, 0.1),
                    unweighted_band_high=(0.6, 0.4, 0.1, 0.1),
                    band_widths=(0.2, 0.1, 0.0, 0.0),
                    mean_band_width=0.075,
                    weighted_mean=(0.5, 0.3, 0.1, 0.1),
                    weighted_variance=(0.01, 0.01, 0.0, 0.0),
                ),
            )

        return stub_predict

    monkeypatch.setattr(pred, "make_predictor", stub_make_predictor)
    rc = pred.main(
        ["--team-a", "A", "--team-b", "B", "--best-of", "Bo3",
         "--as-of-date", "2026-01-01T00:00:00", "--n-samples", "3",
         "--seed", "7", "--ci-level", "0.9"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["per_map"][0]["map_name"] == "Bind"
    assert printed["per_map"][0]["interval_low"] is None
    assert printed["series"]["best_of"] == 3
    assert printed["veto_sensitivity"]["mean_band_width"] == 0.075
    assert printed["predicted_veto"][0]["action"] == "ban"


# --------------------------------------------------------------------------
# plan#11d: real-v1 integration smoke test (slow + skip-guarded)
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the real v1 tables and model artifacts exist.

    The skip guard for the real-data smoke test: the materialised v1
    matches/maps/player_map_stats tables plus the fitted ordinal,
    temperature-scaling and ban/pick conditional-logit model artifacts
    must all be present (i.e. ``materialize.py`` and the four training
    drivers have been run).

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
            "player_map_stats.parquet",
            "ordinal_logit_model.json",
            "temperature_scaling_model.json",
            "conditional_logit_ban_model.json",
            "conditional_logit_pick_model.json",
        )
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(),
    reason="real v1 tables/artifacts not present",
)
def test_real_v1_predict_smoke():
    # A tiny real-v1 run (n_samples=2, no bootstrap) against the real
    # fitted models and tables for one mid-season Bo3 match (so both
    # teams have as-of history): the result carries the full 7-action
    # veto, three per-map entries with finite [0, 1] probabilities
    # summing to ~1 and None intervals (no bootstrap models from the
    # CLI-style call), non-negative n_games_backing, a Bo3
    # SeriesPrediction whose probabilities form a valid simplex with
    # the canonical outcome_order, a finite veto_sensitivity summary,
    # and a fully json.dumps-serializable to_dict.
    matches_df = pred.evaluate.load_matches_table(Path("data"), "v1")
    row = matches_df[
        (matches_df["best_of"] == "Bo3")
        & (matches_df["date"] >= "2026-07-01")
    ].iloc[0]
    predictor = pred.make_predictor(
        "data", "v1", n_samples=2, seed=2026, ci_level=0.9
    )
    result = predictor(
        str(row["team1_id"]),
        str(row["team2_id"]),
        "Bo3",
        None,
        row["date"],
    )

    assert len(result.predicted_veto) == 7
    assert [a.action for a in result.predicted_veto] == [
        "ban", "ban", "pick", "pick", "ban", "ban", "decider"
    ]
    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert len(entry.probabilities) == 4
        assert sum(entry.probabilities) == pytest.approx(1.0)
        assert all(np.isfinite(p) for p in entry.probabilities)
        assert all(0.0 <= p <= 1.0 for p in entry.probabilities)
        assert entry.interval_low is None
        assert entry.interval_high is None
        assert entry.n_games_backing >= 0

    assert result.series.best_of == 3
    assert result.series.outcome_order == series_paths.series_outcome_order(3)
    assert len(result.series.probabilities) == 4
    assert sum(result.series.probabilities) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in result.series.probabilities)

    sensitivity = result.veto_sensitivity
    assert len(sensitivity.unweighted_band_low) == 4
    assert len(sensitivity.unweighted_band_high) == 4
    for low, high in zip(
        sensitivity.unweighted_band_low, sensitivity.unweighted_band_high
    ):
        assert low <= high
        assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert sensitivity.mean_band_width >= 0.0

    # The full result round-trips through json.dumps.
    json.dumps(result.to_dict())
