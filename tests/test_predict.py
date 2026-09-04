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
hand-known replicate 4-vectors); the D10 M39.3 auto-load path
(``bootstrap_models=None`` auto-loading a persisted
``ordinal_bootstrap_replicates.json`` artifact into real per-map
bands, its soft-missing no-interval case, its ``np.allclose``
staleness guard against the base ordinal artifact, the ``()`` escape
hatch and the caller-supplied override); the D5 veto-sensitivity path
(``VetoSensitivity`` fields match hand-computed percentile bands /
weighted moments over the same M31 sample detail); propagated error
paths (invalid ``best_of`` / wrong-size ``map_pool`` raise
``ValueError`` from the real greedy simulator); a ``main()`` CLI smoke
test with the predictor stubbed (JSON printed to stdout, exit 0); the
M39.1 persistent layer (``Predictor`` wraps :func:`make_predictor`
loading exactly once per construction — ``--stream`` CLI mode
answering a JSONL query stream from stdin with one persistent
``Predictor``, its ``parse_args`` mutual-exclusion validation, and the
stream-mode error propagation); M39.4 (the top-``top_n`` enumeration
folded into ``predict()`` — every ``PredictionResult`` widens with a
``top_vetos`` field whose ranked entries carry ``veto_sensitivity
None`` (G6) and inherit the D10 auto-load intervals (G5); the
keyword-only ``top_n`` knob on ``predict``/``Predictor.predict`` with
its ``top_n < 1`` fail-fast; the ``--top-n`` CLI flag threading
through both one-shot and ``--stream`` modes; and the shared
``_build_ranked_veto_entries`` helper keeping
``make_top_vetos_fn``'s listing equal to ``predict(...).top_vetos``
for the same query — the task-055 contract preserved); and
a ``skipif``-guarded real-v1 integration smoke test asserting finite,
well-formed output. No real fitted artifacts are required by the
non-smoke tests.
"""

import inspect
import io
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
    Predictor,
    RankedVetoPrediction,
    SeriesPrediction,
    VetoSensitivity,
)
from evaluation.veto_marginalized_series import (
    SeriesVetoSample,
    VetoMarginalizedSeriesPrediction,
)
from models import _shared, ordinal_logit
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
    artifact loader (returns ``(None, None)``). Callers that only
    construct the factory (error-path tests, whose real greedy
    simulator raises before any enumeration) need nothing more; callers
    that run a **full** ``predict`` must additionally stub the two
    ``make_veto_step_predictor_fn`` factories — since M39.4/G2
    ``predict``'s fourth step exhaustively enumerates over the wired
    ``predictor_fn_by_action`` dict, and the real factories' closures
    over the ``None`` loader models would crash when invoked (the
    ``stub_predictor_wiring`` fixture and
    :func:`_install_top_vetos_wiring` both add the deterministic
    :func:`_top_two_step_stub` arms). Lets the error-path tests build
    a working ``make_predictor`` while keeping the *greedy simulator
    real* so its propagated ``ValueError``s (invalid ``best_of``,
    wrong-size ``map_pool``) are exercised end to end.

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
    Stage-1 artifact loader (returns ``(None, None)``), the two
    ``make_veto_step_predictor_fn`` factories (return
    :func:`_top_two_step_stub` — since M39.4/G2 predict()'s fourth
    step exhaustively enumerates over the wired
    ``predictor_fn_by_action`` dict, so the factories must produce
    callable arms rather than crash on the ``None`` loader models),
    the map-win-rate
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
    # M39.4 (G2): predict()'s new fourth step exhaustively enumerates
    # veto sequences over the make_predictor-wired
    # predictor_fn_by_action dict, so the two make_veto_step_predictor_fn
    # factories must be stubbed too — the table-stub loader returns
    # (None, None) models, which the real factories' closures would
    # crash on when the enumeration invoked them. The same
    # deterministic _top_two_step_stub arm the top_vetos wiring uses
    # keeps every predict() call's top_vetos listing hand-computable.
    monkeypatch.setattr(
        pred.conditional_logit_ban,
        "make_veto_step_predictor_fn",
        lambda model: _top_two_step_stub,
    )
    monkeypatch.setattr(
        pred.conditional_logit_pick,
        "make_veto_step_predictor_fn",
        lambda model: _top_two_step_stub,
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
        "predicted_veto", "per_map", "series", "veto_sensitivity",
        "top_vetos",
    }
    # The nested per-map to_dict round-trips through JSON.
    assert json.loads(json.dumps(result_dict)) == result_dict
    # M39.4 (G1): top_vetos defaults to () and serializes as [] when
    # the result carries no ranking.
    assert result_dict["top_vetos"] == []
    assert json.loads(json.dumps(result_dict))["top_vetos"] == []
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

    # The M39.2 path (F5): a PredictionResult with a None
    # veto_sensitivity (a single fixed veto has no Monte Carlo spread)
    # serializes veto_sensitivity as null, not as a fabricated
    # zero-width band — and the whole dict still round-trips through
    # json.dumps.
    no_spread_result = PredictionResult(
        predicted_veto=_STUB_VETO_ACTIONS,
        per_map=(per_map,),
        series=series,
        veto_sensitivity=None,
    )
    no_spread_dict = no_spread_result.to_dict()
    assert no_spread_dict["veto_sensitivity"] is None
    assert no_spread_dict["top_vetos"] == []
    assert json.loads(json.dumps(no_spread_dict))["veto_sensitivity"] is None


def test_predict_result_serializes_non_empty_top_vetos():
    # M39.4 (G1): a PredictionResult carrying a filled top_vetos tuple
    # serializes each RankedVetoPrediction under the "top_vetos" key,
    # and each inner ranked result's own "top_vetos" is [] (G1's
    # recorded consequence — a fixed-veto conditional result carries
    # no nested ranking — the double-serialization check).
    per_map = PerMapPrediction(
        map_name="Bind",
        probabilities=(0.6, 0.1, 0.1, 0.2),
        interval_low=None,
        interval_high=None,
        n_games_backing=7,
    )
    inner = PredictionResult(
        predicted_veto=_STUB_VETO_ACTIONS,
        per_map=(per_map,),
        series=SeriesPrediction(
            probabilities=(0.5, 0.3, 0.1, 0.1),
            outcome_order=series_paths.series_outcome_order(3),
            best_of=3,
        ),
        veto_sensitivity=None,
    )
    assert inner.top_vetos == ()
    entry = RankedVetoPrediction(veto_probability=0.25, result=inner)
    outer = PredictionResult(
        predicted_veto=_STUB_VETO_ACTIONS,
        per_map=(per_map,),
        series=inner.series,
        veto_sensitivity=None,
        top_vetos=(entry,),
    )
    outer_dict = outer.to_dict()
    assert isinstance(outer_dict["top_vetos"], list)
    assert len(outer_dict["top_vetos"]) == 1
    assert outer_dict["top_vetos"][0]["veto_probability"] == 0.25
    # The double serialization: the inner ranked result's own
    # "top_vetos" key is present and empty.
    assert outer_dict["top_vetos"][0]["result"]["top_vetos"] == []
    assert json.loads(json.dumps(outer_dict)) == outer_dict


# --------------------------------------------------------------------------
# plan#6: M39.2 RankedVetoPrediction wrapper (F4) and the F5 None widening
# --------------------------------------------------------------------------


def test_ranked_veto_prediction_to_dict_json_serializable():
    # F4: RankedVetoPrediction nests its inner PredictionResult's
    # to_dict under "result" beside the flat "veto_probability" float;
    # the composed dict round-trips through json.dumps.
    per_map = PerMapPrediction(
        map_name="Bind",
        probabilities=(0.6, 0.1, 0.1, 0.2),
        interval_low=None,
        interval_high=None,
        n_games_backing=7,
    )
    result = PredictionResult(
        predicted_veto=_STUB_VETO_ACTIONS,
        per_map=(per_map,),
        series=SeriesPrediction(
            probabilities=(0.5, 0.3, 0.1, 0.1),
            outcome_order=series_paths.series_outcome_order(3),
            best_of=3,
        ),
        veto_sensitivity=None,
    )
    entry = RankedVetoPrediction(veto_probability=0.25, result=result)
    entry_dict = entry.to_dict()
    assert set(entry_dict) == {"veto_probability", "result"}
    assert entry_dict["veto_probability"] == 0.25
    assert entry_dict["result"]["predicted_veto"][0] == {
        "step_index": 0,
        "team": "A",
        "action": "ban",
        "map_name": "Haven",
    }
    assert entry_dict["result"]["veto_sensitivity"] is None
    round_tripped = json.loads(json.dumps(entry_dict))
    assert round_tripped["veto_probability"] == 0.25
    assert round_tripped["result"]["per_map"][0]["map_name"] == "Bind"


# --------------------------------------------------------------------------
# plan#9: M39.2 make_top_vetos_fn / top_vetos (exact top-veto listing)
# --------------------------------------------------------------------------


def _top_two_step_stub(
    acting_team_id, action, remaining_maps, date, matches_df, maps_df
):
    """Return a step distribution concentrated on the two first maps.

    The deterministic non-uniform stub predictor for the top_vetos
    wiring tests: the alphabetically-first remaining map receives
    ``0.6``, the second ``0.4`` and every other map ``0.0`` (summing
    to exactly 1 for every remaining-set size the enumeration consults;
    the decider never reaches a predictor). Under this arm the
    always-first veto walk is the unique ``0.6**6`` top-ranked sequence
    and every other feasible first/second pattern has a smaller,
    hand-computable product, so the ranking and the per-veto result
    fields are all hand-assertable.

    Args:
        acting_team_id: The acting team's stable id (ignored).
        action: The step's action (ignored).
        remaining_maps: The sorted remaining-maps list the returned
            distribution aligns to.
        date: The as-of cutoff (ignored).
        matches_df: The materialised matches table (ignored).
        maps_df: The materialised maps table (ignored).

    Returns:
        A ``list`` of ``len(remaining_maps)`` probabilities: ``0.6``
        on the first entry, ``0.4`` on the second, ``0.0`` on the
        rest.

    Raises:
        Nothing.
    """
    n = len(remaining_maps)
    probs = [0.0] * n
    probs[0] = 0.6
    probs[1] = 0.4
    return probs


def _install_top_vetos_wiring(monkeypatch):
    """Patch make_top_vetos_fn's loaders/model sources into stubs.

    Mirrors :func:`_install_table_and_model_stubs` for the M39.2
    factory: routes the three input tables to the synthetic league,
    the Stage-2 registry factory to
    :func:`_stub_map_model_fn` (the hand-known temperature-scaled
    4-vector), the Stage-1 artifact loader to ``(None, None)``, the two
    ``make_veto_step_predictor_fn`` factories to closures returning
    :func:`_top_two_step_stub` (so the exact enumeration inside
    ``top_vetos`` consumes a deterministic, table-ignoring arm under
    both action keys — the real factories are never needed), and the
    map-win-rate estimator to :func:`_stub_team_map_win_rate` (so the
    real :func:`_n_games_backing_for_map` backing queries resolve
    deterministically). No real parquet files or fitted artifacts are
    needed; the enumeration runs for real over the stubbed predictors
    (fast — the 5,040-sequence walk costs ~120 memoised stub calls).

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
    monkeypatch.setattr(
        pred.conditional_logit_ban,
        "make_veto_step_predictor_fn",
        lambda model: _top_two_step_stub,
    )
    monkeypatch.setattr(
        pred.conditional_logit_pick,
        "make_veto_step_predictor_fn",
        lambda model: _top_two_step_stub,
    )
    monkeypatch.setattr(
        pred.map_win_rate, "team_map_win_rate", _stub_team_map_win_rate
    )


@pytest.fixture
def stub_top_vetos_wiring(monkeypatch):
    """Monkeypatch make_top_vetos_fn's loaders/models into deterministic stubs.

    Thin fixture wrapper over :func:`_install_top_vetos_wiring` so the
    top_vetos tests can request the deterministic synthetic wiring by
    name (mirroring the ``stub_predictor_wiring`` fixture's role for
    the make_predictor tests). All patches are reverted by monkeypatch
    at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        Nothing.

    Raises:
        Nothing.
    """
    _install_top_vetos_wiring(monkeypatch)


def test_make_top_vetos_fn_missing_table_raises_file_not_found(tmp_path):
    # F3 load contract: with no tables under the empty tmp dir (no
    # stubs) the first table load raises FileNotFoundError unchanged —
    # the standard "run the prerequisite first" signal, mirroring
    # test_make_predictor_missing_table_raises_file_not_found.
    with pytest.raises(FileNotFoundError):
        pred.make_top_vetos_fn(tmp_path, "v1")


def test_make_top_vetos_fn_returns_6_arg_closure_with_documented_defaults(
    tmp_path, stub_top_vetos_wiring
):
    # F3/F6: make_top_vetos_fn returns a callable whose signature is
    # exactly the documented 6-arg public API — (team_a, team_b,
    # best_of, map_pool, as_of_date, n) — with n defaulting to
    # DEFAULT_TOP_N and no n_samples/seed knobs (no M31 sampling on
    # this path, F6); the factory keywords default to the documented
    # constants.
    top_vetos = pred.make_top_vetos_fn("data", "v1")
    assert callable(top_vetos)
    parameters = list(inspect.signature(top_vetos).parameters)
    assert parameters == [
        "team_a", "team_b", "best_of", "map_pool", "as_of_date", "n"
    ]
    assert (
        inspect.signature(top_vetos).parameters["n"].default
        == pred.DEFAULT_TOP_N
    )
    assert pred.DEFAULT_TOP_N == 10
    factory = inspect.signature(pred.make_top_vetos_fn)
    assert factory.parameters["ci_level"].default == DEFAULT_CI_LEVEL
    assert factory.parameters["bootstrap_models"].default is None
    assert "n_samples" not in factory.parameters
    assert "seed" not in factory.parameters


def test_top_vetos_synthetic_shapes_wiring_and_ranking(
    tmp_path, stub_top_vetos_wiring
):
    # A full synthetic top_vetos run against the stubbed league with
    # the top-two stub arm: the returned tuple has length min(n, 5040)
    # and is sorted descending by veto_probability; every entry's
    # result has veto_sensitivity None (F5) and a series that sums to 1
    # and equals an independently-computed
    # series_probabilities_in_order call over that specific veto's
    # collapsed per-map win probabilities; every per-map entry's
    # probabilities/n_games_backing match direct calls; and the first
    # (unique 0.6**6) entry's full predicted_veto matches the specific
    # enumerated sequence's actions with the probability field dropped.
    matches_df, maps_df, _ = _league_tables()
    as_of_date = "2026-01-01T00:00:00"
    top_vetos = pred.make_top_vetos_fn("data", "v1", ci_level=0.9)
    entries = top_vetos("A", "B", "Bo3", _STUB_POOL, as_of_date, n=10)

    assert len(entries) == 10
    probabilities = [e.veto_probability for e in entries]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(e.result.veto_sensitivity is None for e in entries)

    # The unique top walk: always alphabetically-first over the sorted
    # pool leaves Sunset as the decider.
    assert entries[0].veto_probability == pytest.approx(0.6**6)
    assert [a.map_name for a in entries[0].result.predicted_veto] == [
        "Ascent", "Bind", "Haven", "Icebox", "Lotus", "Split", "Sunset"
    ]
    assert [a.action for a in entries[0].result.predicted_veto] == [
        "ban", "ban", "pick", "pick", "ban", "ban", "decider"
    ]

    # Every entry: per-map fields and the exact-M30 series.
    for entry in entries:
        played_maps = [
            a.map_name
            for a in entry.result.predicted_veto
            if a.action in ("pick", "decider")
        ]
        assert [pm.map_name for pm in entry.result.per_map] == played_maps
        for pm in entry.result.per_map:
            # The stub map model is called directly for this map.
            assert pm.probabilities == _stub_map_model_fn(
                "A", "B", pm.map_name, as_of_date, matches_df, maps_df
            )
            assert sum(pm.probabilities) == pytest.approx(1.0)
            assert pm.interval_low is None
            assert pm.interval_high is None
            assert pm.n_games_backing == pred._n_games_backing_for_map(
                "A", "B", pm.map_name, as_of_date, matches_df, maps_df
            )
        # The exact-M30 series over this veto's collapsed per-map win
        # probabilities ((0.6, 0.1, 0.1, 0.2) -> A-win 0.7 per played
        # map under the constant stub map model).
        expected_series = series_paths.series_probabilities_in_order(
            [0.7] * len(played_maps), 3
        )
        assert entry.result.series.probabilities == pytest.approx(
            expected_series
        )
        assert sum(entry.result.series.probabilities) == pytest.approx(1.0)
        assert entry.result.series.outcome_order == series_paths.series_outcome_order(3)
        assert entry.result.series.best_of == 3

    # Cross-check the ranking against an independent enumeration +
    # stable descending sort: entry i's probability and full action
    # tuple must match ranked sequence i with probability dropped.
    enumerated = pred.ancestral_veto_sampler.enumerate_veto_sequences(
        "A", "B", "Bo3", as_of_date, matches_df, maps_df,
        {"ban": _top_two_step_stub, "pick": _top_two_step_stub},
        map_pool=_STUB_POOL,
    )
    assert len(enumerated) == 5040
    ranked = sorted(
        enumerated,
        key=lambda seq: seq.sequence_probability,
        reverse=True,
    )
    for entry, seq in zip(entries, ranked[:10]):
        assert entry.veto_probability == seq.sequence_probability
        actual = tuple(
            (a.step_index, a.team, a.action, a.map_name)
            for a in entry.result.predicted_veto
        )
        expected = tuple(
            (a.step_index, a.team, a.action, a.map_name)
            for a in seq.actions
        )
        assert actual == expected

    # The whole listing round-trips through json.dumps.
    json.dumps([e.to_dict() for e in entries])


def test_top_vetos_n_edge_cases(tmp_path, stub_top_vetos_wiring):
    # F7: n=1 returns exactly the single best entry; n larger than the
    # 5,040-sequence total silently returns all 5,040 (documented, not
    # an error).
    top_vetos = pred.make_top_vetos_fn("data", "v1")
    one = top_vetos("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00", n=1)
    assert len(one) == 1
    assert one[0].veto_probability == pytest.approx(0.6**6)
    all_sequences = top_vetos(
        "A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00", n=5041
    )
    assert len(all_sequences) == 5040


def test_top_vetos_rejects_n_lt_1_before_any_enumeration(
    tmp_path, monkeypatch
):
    # F7's fail-fast clause: n=0 and n=-1 raise ValueError before any
    # enumeration work happens — asserted via a call-counting stub on
    # the enumerate entry point, which is never invoked for the n<1
    # cases (the factory itself is otherwise fully stubbed, so a
    # positive-n call would reach it).
    _install_top_vetos_wiring(monkeypatch)
    calls = {"count": 0}

    def counting_enumerate(*args, **kwargs):
        """Count invocations and fail if reached (n<1 must never call it).

        Args:
            *args: Positional arguments (unused).
            **kwargs: Keyword arguments (unused).

        Returns:
            Nothing (raises instead).

        Raises:
            AssertionError: Always — reaching the enumerator means the
                n<1 validation failed to run first.
        """
        calls["count"] += 1
        raise AssertionError(
            "enumerate_veto_sequences must not be called for n < 1"
        )

    monkeypatch.setattr(
        pred.ancestral_veto_sampler,
        "enumerate_veto_sequences",
        counting_enumerate,
    )
    top_vetos = pred.make_top_vetos_fn("data", "v1")
    for bad_n in (0, -1):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            top_vetos(
                "A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00", n=bad_n
            )
    assert calls["count"] == 0


def test_top_vetos_intervals_from_bootstrap_models(tmp_path, monkeypatch):
    # D4 parity on the top_vetos path: with replicate models supplied
    # to make_top_vetos_fn, every per-map interval_low/interval_high of
    # the returned entries is the per-category percentile band over the
    # hand-known tilted replicate vectors (landed by the duplicated
    # interval body inside top_vetos), while the point probabilities
    # stay the temperature-scaled stub vector.
    tilts = [0.0, 0.02, 0.04]
    models = [_synthetic_ordinal_model() for _ in tilts]
    tilts_by_id = {id(model): d for model, d in zip(models, tilts)}
    monkeypatch.setattr(
        pred.ordinal_logit,
        "make_model_fn",
        _make_model_fn_with_tilts(tilts_by_id),
    )
    lo, hi = _expected_interval_bands(tilts, ci_level=0.9)
    _install_top_vetos_wiring(monkeypatch)
    top_vetos = pred.make_top_vetos_fn(
        "data", "v1", ci_level=0.9, bootstrap_models=models
    )
    entries = top_vetos("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00", n=1)
    assert len(entries) == 1
    assert len(entries[0].result.per_map) == 3
    for pm in entries[0].result.per_map:
        assert pm.probabilities == (0.6, 0.1, 0.1, 0.2)
        assert pm.interval_low == pytest.approx(lo)
        assert pm.interval_high == pytest.approx(hi)
        for low, high in zip(pm.interval_low, pm.interval_high):
            assert low <= high


def test_top_vetos_intervals_none_without_bootstrap_models(
    tmp_path, stub_top_vetos_wiring
):
    # D4 negative path on top_vetos: without bootstrap_models every
    # per-map interval field is None while the point probabilities and
    # backing are still populated.
    top_vetos = pred.make_top_vetos_fn("data", "v1")
    entries = top_vetos("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00", n=3)
    assert len(entries) == 3
    for entry in entries:
        for pm in entry.result.per_map:
            assert pm.interval_low is None
            assert pm.interval_high is None
            assert pm.probabilities == (0.6, 0.1, 0.1, 0.2)


# --------------------------------------------------------------------------
# plan#9b: make_predictor factory contract
# --------------------------------------------------------------------------


def test_make_predictor_returns_5_arg_callable(tmp_path, stub_predictor_wiring):
    # make_predictor returns a callable whose signature is exactly the
    # documented public API — the five positional args (team_a, team_b,
    # best_of, map_pool, as_of_date) plus, since M39.4 (G3), the
    # keyword-only top_n defaulting to DEFAULT_TOP_N — closing over the
    # (stubbed) tables/models. The explicit () escape hatch (D10)
    # keeps the factory off the real-data auto-load path so the test
    # stays hermetic regardless of whether
    # data/v1/ordinal_bootstrap_replicates.json exists.
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, bootstrap_models=()
    )
    assert callable(predictor)
    parameters = inspect.signature(predictor).parameters
    assert list(parameters) == [
        "team_a", "team_b", "best_of", "map_pool", "as_of_date",
        "top_n",
    ]
    top_n_param = parameters["top_n"]
    assert top_n_param.kind == inspect.Parameter.KEYWORD_ONLY
    assert top_n_param.default == pred.DEFAULT_TOP_N
    assert pred.DEFAULT_TOP_N == 10


def test_make_predictor_defaults_match_documented(tmp_path, stub_predictor_wiring):
    # The factory keyword defaults are the documented D7 constants —
    # referenced through the module constants so a stale hardcode can
    # never drift silently.
    signature = inspect.signature(pred.make_predictor)
    assert signature.parameters["n_samples"].default == DEFAULT_N_SAMPLES
    assert signature.parameters["seed"].default == DEFAULT_SEED
    assert signature.parameters["ci_level"].default == DEFAULT_CI_LEVEL
    # M39.3 (D10): the bootstrap_models signature default stays None —
    # only its runtime meaning changed (None now auto-loads the
    # persisted replicates artifact; () is the no-interval escape
    # hatch).
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
    # intervals (bootstrap_models=() — D10's escape hatch, so the run
    # stays off the real-data auto-load path), and the hand-known
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
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=(),
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
    # D4 negative path via the D10 escape hatch: an explicit empty
    # bootstrap_models=() — never conflated with the None default,
    # which auto-loads the persisted artifact — means no interval on
    # every per-map entry (no epistemic interval) while the point
    # probabilities and backing are still populated, regardless of
    # whether a real artifact sits at data/v1. The None-with-no-
    # artifact soft-missing case is covered separately against an
    # empty tmp_path.
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, bootstrap_models=()
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")
    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.interval_low is None
        assert entry.interval_high is None
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)


# --------------------------------------------------------------------------
# plan#10c: D10 M39.3 persisted-replicate auto-load path
# --------------------------------------------------------------------------


def _write_replicate_artifact(version_dir, models, base_thresholds):
    """Write a genuine ordinal_bootstrap_replicates.json fixture artifact.

    Builds the exact artifact dict the producer driver
    (``drivers/train_bootstrap_replicates.py``) writes — ``"config"``
    (``n_bootstrap_map``/``bootstrap_seed``), ``"replicates"`` (one
    ``ordinal_logit.to_dict`` entry per model with the derived
    ``"coefficient_report"`` key stripped) and
    ``"base_ordinal_thresholds"`` (the provenance copy the auto-load
    staleness guard compares) — and writes it under ``version_dir``
    with the repo's ``json.dumps(..., indent=2, sort_keys=True) + "\n"``
    formatting.

    Args:
        version_dir: The ``<output_dir>/<version>`` directory to write
            into (created if absent).
        models: The replicate models to serialize (each an
            ``OrdinalLogitModel``).
        base_thresholds: The 3 base-model thresholds the artifact
            claims as its provenance copy; tests perturb this away
            from the on-disk base artifact to trip the staleness
            guard.

    Returns:
        The written artifact path (a ``Path``).

    Raises:
        Nothing.
    """
    version_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "config": {
            "n_bootstrap_map": len(models),
            "bootstrap_seed": 2026,
        },
        "replicates": [
            {
                key: value
                for key, value in ordinal_logit.to_dict(model).items()
                if key != "coefficient_report"
            }
            for model in models
        ],
        "base_ordinal_thresholds": [float(t) for t in base_thresholds],
    }
    path = version_dir / "ordinal_bootstrap_replicates.json"
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _make_model_fn_routing_on_coefficient_zero(
    model, player_map_stats_df
):
    """Build a make_model_fn stub routing each model's tilt via coefficient 0.

    The D10 auto-load counterpart of
    :func:`_make_model_fn_with_tilts`: returns a closure that ignores
    the tables and returns the hand-known 4-vector
    ``(0.25 + d, 0.25 - d, 0.25, 0.25)`` where ``d =
    model.coefficients[0]`` — a tilt that survives the
    ``to_dict``/``from_dict`` round trip, unlike an object identity, so
    the auto-load path's *deserialized* replicates (created inside
    ``make_predictor``, invisible to the test) land hand-computable
    per-replicate rows.

    Args:
        model: The replicate ``OrdinalLogitModel`` whose
            ``coefficients[0]`` sizes the tilt.
        player_map_stats_df: Unused by the stub (signature parity).

    Returns:
        The 6-argument ``stub_map_fn`` closure returning the tilted
        4-vector.

    Raises:
        Nothing.
    """
    d = float(model.coefficients[0])

    def stub_map_fn(
        team1_id, team2_id, map_name, date, matches_df, maps_df
    ):
        return (0.25 + d, 0.25 - d, 0.25, 0.25)

    return stub_map_fn


def test_make_predictor_auto_loads_persisted_replicates(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # M39.3 (D10): with a genuine ordinal_bootstrap_replicates.json on
    # disk (built via ordinal_logit.to_dict over synthetic models whose
    # coefficient[0] encodes a tilt) and bootstrap_models not passed
    # (the None default), make_predictor auto-loads the persisted
    # replicates: every per-map interval_low/high of a predict() call
    # is the per-category percentile band over the tilt-routed
    # replicate vectors, landed via the auto-load path instead of an
    # explicit argument (D4 parity with
    # test_per_map_interval_bands_from_bootstrap_models, except the
    # models came from disk, so the routing stub reads each
    # deserialized model's coefficient[0]). The on-disk base artifact
    # carries matching thresholds so the staleness guard passes.
    tilts = [0.0, 0.02, 0.04]
    models = [_synthetic_ordinal_model() for _ in tilts]
    for model, d in zip(models, tilts):
        model.coefficients[0] = d
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    _write_replicate_artifact(version_dir, models, [-1.0, 0.0, 1.0])
    monkeypatch.setattr(
        pred.ordinal_logit,
        "make_model_fn",
        _make_model_fn_routing_on_coefficient_zero,
    )
    lo, hi = _expected_interval_bands(tilts, ci_level=0.9)

    predictor = pred.make_predictor(
        tmp_path, "v1", n_samples=3, seed=2026, ci_level=0.9
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


def test_predictor_threads_auto_load_default(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # Item 9 verification (Predictor needs no code change): Predictor's
    # own bootstrap_models=None default is forwarded to make_predictor
    # unchanged, so a Predictor constructed over a directory holding a
    # valid persisted-replicates artifact auto-loads it (D10) — its
    # .predict call lands real non-None per-map intervals through the
    # E1 single-load wrapper, proving the CLI modes (one-shot builds
    # make_predictor, --stream builds Predictor, neither passes
    # bootstrap_models) pick up the auto-load for free.
    tilts = [0.0, 0.02, 0.04]
    models = [_synthetic_ordinal_model() for _ in tilts]
    for model, d in zip(models, tilts):
        model.coefficients[0] = d
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    _write_replicate_artifact(version_dir, models, [-1.0, 0.0, 1.0])
    monkeypatch.setattr(
        pred.ordinal_logit,
        "make_model_fn",
        _make_model_fn_routing_on_coefficient_zero,
    )
    lo, hi = _expected_interval_bands(tilts, ci_level=0.9)

    predictor = Predictor(
        tmp_path, "v1", n_samples=3, seed=2026, ci_level=0.9
    )
    result = predictor.predict(
        "A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00"
    )

    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)
        assert entry.interval_low == pytest.approx(lo)
        assert entry.interval_high == pytest.approx(hi)


def test_make_predictor_auto_load_soft_missing_no_intervals(
    tmp_path, stub_predictor_wiring
):
    # M39.3 soft missing-artifact case: with no
    # ordinal_bootstrap_replicates.json under the output dir, the None
    # default behaves exactly as before the milestone — None closed
    # over, every per-map interval field None — and crucially raises
    # no FileNotFoundError (the three tables / four required artifacts
    # keep their hard failure; this input is the roadmap's one soft
    # exception). Mirrors
    # test_per_map_intervals_none_without_bootstrap_models but through
    # the None default against an empty tmp_path rather than the ()
    # escape hatch.
    predictor = pred.make_predictor(tmp_path, "v1", n_samples=3)
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")
    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.interval_low is None
        assert entry.interval_high is None
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)


def test_make_predictor_replicate_staleness_guard_raises_value_error(
    tmp_path, monkeypatch
):
    # D10's staleness guard fires on the auto-load path: with a
    # genuine base ordinal artifact on disk whose thresholds are
    # [-1.0, 0.0, 1.0] and a replicate artifact whose
    # base_ordinal_thresholds provenance copy is deliberately perturbed
    # to [0.0, 1.0, 2.0], make_predictor raises ValueError ("re-run
    # train_bootstrap_replicates.py") rather than silently applying
    # replicates fit against a different base model — mirroring the
    # decision-E temperature guard's wording style and firing only on
    # the auto-load path (bootstrap_models not passed).
    _install_table_and_model_stubs(monkeypatch)
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    models = [_synthetic_ordinal_model()]
    _write_replicate_artifact(version_dir, models, [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="re-run train_bootstrap_replicates"):
        pred.make_predictor(tmp_path, "v1")


def test_make_predictor_empty_tuple_escape_hatch_with_artifact_present(
    tmp_path, stub_predictor_wiring
):
    # D10's () escape hatch: an explicit empty bootstrap_models=()
    # suppresses intervals even when a valid, matching artifact is
    # present on disk — () is never conflated with the None default,
    # which would auto-load that artifact (the on-disk base artifact
    # is written with matching thresholds so the pair is fully valid;
    # the () call must skip both the artifact read and the guard).
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    models = [_synthetic_ordinal_model(), _synthetic_ordinal_model()]
    _write_replicate_artifact(version_dir, models, [-1.0, 0.0, 1.0])

    predictor = pred.make_predictor(
        tmp_path, "v1", n_samples=3, bootstrap_models=()
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")
    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.interval_low is None
        assert entry.interval_high is None
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)


def test_make_predictor_explicit_models_override_auto_load(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # D10: an explicit non-empty bootstrap_models sequence still
    # overrides — auto-load is skipped whenever the caller passes
    # anything other than None, even with a mismatched artifact on
    # disk whose provenance copy would trip the staleness guard if it
    # were read. The passed models' tilts (id-routed, as in the D4
    # explicit-path test) land the bands; the predict() call succeeds
    # (no raise from the never-read artifact).
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    # Mismatched provenance: reading this artifact would raise, so its
    # being ignored is exactly what the override guarantees.
    _write_replicate_artifact(
        version_dir, [_synthetic_ordinal_model()], [0.0, 1.0, 2.0]
    )
    tilts = [0.0, 0.02, 0.04]
    explicit_models = [_synthetic_ordinal_model() for _ in tilts]
    tilts_by_id = {
        id(model): d for model, d in zip(explicit_models, tilts)
    }
    monkeypatch.setattr(
        pred.ordinal_logit,
        "make_model_fn",
        _make_model_fn_with_tilts(tilts_by_id),
    )
    lo, hi = _expected_interval_bands(tilts, ci_level=0.9)

    predictor = pred.make_predictor(
        tmp_path,
        "v1",
        n_samples=3,
        seed=2026,
        ci_level=0.9,
        bootstrap_models=explicit_models,
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")

    assert len(result.per_map) == 3
    for entry in result.per_map:
        assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)
        assert entry.interval_low == pytest.approx(lo)
        assert entry.interval_high == pytest.approx(hi)


# --------------------------------------------------------------------------
# plan#4-7/9: M39.4 — top_n folded into predict() (G1-G6)
# --------------------------------------------------------------------------


def test_predict_top_vetos_default_ranking_and_shapes(
    tmp_path, stub_predictor_wiring
):
    # M39.4 (G1/G4/G6): a predict() call with the default top_n
    # returns exactly min(DEFAULT_TOP_N, 5040) top_vetos entries
    # sorted by descending veto_probability (stable; ties keep
    # enumeration order — same rule as top_vetos's own ranking test);
    # every entry's result has veto_sensitivity None (G6) and an
    # exact-M30 series over that specific veto's played maps; the
    # top-level veto_sensitivity stays a real M31 VetoSensitivity; the
    # greedy predicted_veto is computed independently (F7) and is NOT
    # assumed to be the top-ranked entry; and the whole result
    # round-trips through to_dict (each inner ranked result's own
    # "top_vetos" is [] — G1's double-serialization consequence).
    matches_df, maps_df, _ = _league_tables()
    as_of_date = "2026-01-01T00:00:00"
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=(),
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, as_of_date)

    assert pred.DEFAULT_TOP_N == 10
    assert len(result.top_vetos) == 10
    probabilities = [e.veto_probability for e in result.top_vetos]
    assert probabilities == sorted(probabilities, reverse=True)

    # G6: the top-level veto_sensitivity is a real M31 summary while
    # every inner ranked result's is None.
    assert result.veto_sensitivity is not None
    assert all(
        e.result.veto_sensitivity is None for e in result.top_vetos
    )

    # The unique top walk under the top-two stub arm: always-first
    # over the sorted pool leaves Sunset as the decider.
    assert result.top_vetos[0].veto_probability == pytest.approx(0.6**6)
    assert [
        a.map_name for a in result.top_vetos[0].result.predicted_veto
    ] == ["Ascent", "Bind", "Haven", "Icebox", "Lotus", "Split",
          "Sunset"]
    # The stub greedy veto's played maps (Bind/Ascent/Sunset) differ
    # from the top-ranked enumerated veto's — the two are computed
    # independently.
    assert [pm.map_name for pm in result.per_map] == [
        "Bind", "Ascent", "Sunset"
    ]
    assert [
        pm.map_name for pm in result.top_vetos[0].result.per_map
    ] == ["Haven", "Icebox", "Sunset"]

    # Every entry: per-map fields + the exact-M30 series (the constant
    # stub map vector (0.6, 0.1, 0.1, 0.2) collapses to an A-win
    # probability of 0.7 per played map).
    for entry in result.top_vetos:
        played_maps = [
            a.map_name
            for a in entry.result.predicted_veto
            if a.action in ("pick", "decider")
        ]
        assert [pm.map_name for pm in entry.result.per_map] == played_maps
        for pm in entry.result.per_map:
            assert pm.probabilities == (0.6, 0.1, 0.1, 0.2)
            assert pm.interval_low is None
            assert pm.interval_high is None
            assert pm.n_games_backing == pred._n_games_backing_for_map(
                "A", "B", pm.map_name, as_of_date, matches_df, maps_df
            )
        expected_series = series_paths.series_probabilities_in_order(
            [0.7] * len(played_maps), 3
        )
        assert entry.result.series.probabilities == pytest.approx(
            expected_series
        )
        assert (
            entry.result.series.outcome_order
            == series_paths.series_outcome_order(3)
        )
        assert entry.result.series.best_of == 3
        # G1: a fixed-veto conditional result carries no nested
        # ranking.
        assert entry.result.top_vetos == ()

    # Cross-check the ranking against an independent enumeration +
    # stable descending sort (mirroring the top_vetos ranking test).
    enumerated = pred.ancestral_veto_sampler.enumerate_veto_sequences(
        "A", "B", "Bo3", as_of_date, matches_df, maps_df,
        {"ban": _top_two_step_stub, "pick": _top_two_step_stub},
        map_pool=_STUB_POOL,
    )
    ranked = sorted(
        enumerated,
        key=lambda seq: seq.sequence_probability,
        reverse=True,
    )
    for entry, seq in zip(result.top_vetos, ranked[:10]):
        assert entry.veto_probability == seq.sequence_probability

    # The whole result round-trips through JSON; the top_vetos list
    # carries the double-serialized inner results.
    result_dict = result.to_dict()
    assert len(result_dict["top_vetos"]) == 10
    assert all(
        inner["result"]["top_vetos"] == []
        for inner in result_dict["top_vetos"]
    )
    json.loads(json.dumps(result_dict))


def test_predict_top_n_edge_cases(tmp_path, stub_predictor_wiring):
    # M39.4 (G3): top_n=1 returns exactly the single best entry and
    # top_n larger than the 5,040-sequence total silently returns all
    # 5,040 without error (mirroring test_top_vetos_n_edge_cases) —
    # the enumeration always runs in full; top_n only slices the
    # returned ranking.
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, bootstrap_models=()
    )
    as_of_date = "2026-01-01T00:00:00"
    one = predictor("A", "B", "Bo3", _STUB_POOL, as_of_date, top_n=1)
    assert len(one.top_vetos) == 1
    assert one.top_vetos[0].veto_probability == pytest.approx(0.6**6)
    # The greedy/M31 fields are still present beside the ranked entry.
    assert one.veto_sensitivity is not None
    all_sequences = predictor(
        "A", "B", "Bo3", _STUB_POOL, as_of_date, top_n=5041
    )
    assert len(all_sequences.top_vetos) == 5040


def test_predict_rejects_top_n_lt_1_before_any_enumeration(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # M39.4 (G3/A1): top_n=0 and top_n=-1 raise ValueError before any
    # enumeration work — asserted via a call-counting stub on the
    # enumerate entry point, which is never invoked (the fail-fast
    # sits at the top of predict's body, before even the greedy/M31
    # work; no top_n=0-means-skip convention exists, so 0 is simply
    # invalid).
    calls = {"enumerate": 0}

    def counting_enumerate(*args, **kwargs):
        """Count invocations and fail if reached (top_n<1 must never call it).

        Args:
            *args: Positional arguments (unused).
            **kwargs: Keyword arguments (unused).

        Returns:
            Nothing (raises instead).

        Raises:
            AssertionError: Always — reaching the enumerator means the
                top_n validation failed to run first.
        """
        calls["enumerate"] += 1
        raise AssertionError(
            "enumerate_veto_sequences must not be called for top_n < 1"
        )

    monkeypatch.setattr(
        pred.ancestral_veto_sampler,
        "enumerate_veto_sequences",
        counting_enumerate,
    )
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, bootstrap_models=()
    )
    for bad_top_n in (0, -1):
        with pytest.raises(
            ValueError, match="top_n must be a positive integer"
        ):
            predictor(
                "A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00",
                top_n=bad_top_n,
            )
    assert calls["enumerate"] == 0


def test_predict_top_vetos_intervals_via_auto_load(
    tmp_path, stub_predictor_wiring, monkeypatch
):
    # M39.4 (G5): with a genuine ordinal_bootstrap_replicates.json on
    # disk (coefficient[0]-encoded tilts) and bootstrap_models not
    # passed, the D10 auto-load reaches EVERY ranked
    # RankedVetoPrediction.result.per_map entry — each per-map
    # interval_low/high is the same per-category percentile band over
    # the auto-loaded replicate vectors that the greedy top-level
    # per_map entries carry (every map shares the stub vector, so all
    # intervals equal lo/hi).
    tilts = [0.0, 0.02, 0.04]
    models = [_synthetic_ordinal_model() for _ in tilts]
    for model, d in zip(models, tilts):
        model.coefficients[0] = d
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    (version_dir / "ordinal_logit_model.json").write_text(
        json.dumps(_ordinal_artifact_dict()), encoding="utf-8"
    )
    _write_replicate_artifact(version_dir, models, [-1.0, 0.0, 1.0])
    monkeypatch.setattr(
        pred.ordinal_logit,
        "make_model_fn",
        _make_model_fn_routing_on_coefficient_zero,
    )
    lo, hi = _expected_interval_bands(tilts, ci_level=0.9)

    predictor = pred.make_predictor(
        tmp_path, "v1", n_samples=3, seed=2026, ci_level=0.9
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00")

    # The greedy per-map intervals land the auto-loaded bands.
    for entry in result.per_map:
        assert entry.interval_low == pytest.approx(lo)
        assert entry.interval_high == pytest.approx(hi)
    # Every ranked entry's per-map intervals inherit the same D10
    # auto-load (G5).
    assert len(result.top_vetos) == 10
    for ranked in result.top_vetos:
        for pm in ranked.result.per_map:
            assert pm.interval_low == pytest.approx(lo)
            assert pm.interval_high == pytest.approx(hi)
            assert pm.probabilities == (0.6, 0.1, 0.1, 0.2)


def test_predict_top_vetos_matches_standalone_top_vetos(
    tmp_path, stub_predictor_wiring
):
    # Task-055 contract preserved (roadmap Tests bullet): for the same
    # query, ci_level and bootstrap_models, make_top_vetos_fn's
    # top_vetos(n=10) listing equals predict(...).top_vetos entry for
    # entry by veto_probability and the relevant PredictionResult
    # fields (predicted_veto, per_map, series; inner top_vetos is ()
    # on both sides) — the shared _build_ranked_veto_entries helper
    # (G2) guarantees the two paths build identical entries.
    as_of_date = "2026-01-01T00:00:00"
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=(),
    )
    result = predictor("A", "B", "Bo3", _STUB_POOL, as_of_date)
    top_vetos = pred.make_top_vetos_fn("data", "v1", ci_level=0.9)
    listing = top_vetos("A", "B", "Bo3", _STUB_POOL, as_of_date, n=10)

    assert len(listing) == len(result.top_vetos) == 10
    for standalone, folded in zip(listing, result.top_vetos):
        assert standalone.veto_probability == folded.veto_probability
        assert (
            standalone.result.predicted_veto
            == folded.result.predicted_veto
        )
        assert [
            pm.to_dict() for pm in standalone.result.per_map
        ] == [pm.to_dict() for pm in folded.result.per_map]
        assert (
            standalone.result.series.to_dict()
            == folded.result.series.to_dict()
        )
        assert standalone.result.veto_sensitivity is None
        assert standalone.result.top_vetos == ()
        assert folded.result.top_vetos == ()


def test_predictor_predict_forwards_top_n(tmp_path, stub_predictor_wiring):
    # M39.4 (A5): Predictor.predict accepts the keyword-only top_n and
    # forwards it to the wrapped closure — the widened result's
    # top_vetos shape matches calling make_predictor(...) directly
    # with the same top_n (bitwise-equal to_dict), and omitting top_n
    # uses DEFAULT_TOP_N per call (no top_n at construction time).
    as_of_date = "2026-01-01T00:00:00"
    predictor = Predictor(
        tmp_path, "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=(),
    )
    direct = pred.make_predictor(
        tmp_path, "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=(),
    )
    small = predictor.predict(
        "A", "B", "Bo3", _STUB_POOL, as_of_date, top_n=3
    )
    assert len(small.top_vetos) == 3
    direct_small = direct(
        "A", "B", "Bo3", _STUB_POOL, as_of_date, top_n=3
    )
    assert small.to_dict() == direct_small.to_dict()
    default = predictor.predict("A", "B", "Bo3", _STUB_POOL, as_of_date)
    assert len(default.top_vetos) == pred.DEFAULT_TOP_N == 10


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
    # bootstrap_models=() (D10's escape hatch) keeps the run off the
    # real-data auto-load path.
    predictor = pred.make_predictor(
        "data", "v1", n_samples=3, seed=2026, ci_level=0.9,
        bootstrap_models=(),
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
        "data", "v1", n_samples=1, seed=2026, ci_level=0.9,
        bootstrap_models=(),
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
    predictor = pred.make_predictor(
        "data", "v1", n_samples=2, bootstrap_models=()
    )
    with pytest.raises(ValueError, match="not a supported veto format"):
        predictor("A", "B", "Bo7", _STUB_POOL, "2026-01-01T00:00:00")


def test_predict_wrong_size_map_pool_propagates_value_error(
    tmp_path, monkeypatch
):
    # A non-7 map_pool propagates the greedy simulator's ValueError
    # unchanged ("a Bo3 veto needs 7") — the D8 fail-loud clause, not
    # re-validated by predict itself.
    _install_table_and_model_stubs(monkeypatch)
    predictor = pred.make_predictor(
        "data", "v1", n_samples=2, bootstrap_models=()
    )
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
    # M39.4 (G7): --top-n defaults to DEFAULT_TOP_N.
    assert args.top_n == pred.DEFAULT_TOP_N
    assert pred.DEFAULT_TOP_N == 10


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int counts/seeds and
    # non-float --ci-level are rejected by argparse (SystemExit), and a
    # --best-of outside Bo1/Bo3/Bo5 is rejected by choices=.
    args = pred.parse_args(
        ["--version", "v2", "--output-dir", "out",
         "--team-a", "A", "--team-b", "B", "--best-of", "Bo5",
         "--map-pool", "Bind,Haven,Split,Ascent,Lotus,Icebox,Sunset",
         "--as-of-date", "2026-01-01T00:00:00",
         "--n-samples", "2", "--seed", "7", "--ci-level", "0.8",
         "--top-n", "3"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.team_a == "A"
    assert args.best_of == "Bo5"
    assert args.map_pool == "Bind,Haven,Split,Ascent,Lotus,Icebox,Sunset"
    assert args.n_samples == 2
    assert args.seed == 7
    assert args.ci_level == 0.8
    assert args.top_n == 3
    # Non-int --top-n is rejected by argparse (SystemExit).
    with pytest.raises(SystemExit):
        pred.parse_args(["--team-a", "A", "--team-b", "B",
                         "--best-of", "Bo3", "--as-of-date", "d",
                         "--top-n", "many"])
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
    # main() builds the predictor, calls predict once (with
    # top_n=args.top_n — the M39.4/G7 session-level knob — asserted
    # inside the stub closure) and prints the
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
        def stub_predict(
            team_a, team_b, best_of, map_pool, as_of_date, *, top_n=10
        ):
            assert top_n == 3
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
         "--seed", "7", "--ci-level", "0.9", "--top-n", "3"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["per_map"][0]["map_name"] == "Bind"
    assert printed["per_map"][0]["interval_low"] is None
    assert printed["series"]["best_of"] == 3
    assert printed["veto_sensitivity"]["mean_band_width"] == 0.075
    assert printed["predicted_veto"][0]["action"] == "ban"
    # M39.4 (G7): the one-shot output carries the additive top_vetos
    # key automatically via to_dict — [] here because the canned stub
    # result carries no ranking.
    assert printed["top_vetos"] == []


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
    # summing to ~1 and None intervals, non-negative n_games_backing,
    # a Bo3 SeriesPrediction whose probabilities form a valid simplex
    # with the canonical outcome_order, a finite veto_sensitivity
    # summary, and a fully json.dumps-serializable to_dict. The
    # explicit bootstrap_models=() (D10's escape hatch) keeps the
    # intervals None deterministically — since M39.3 the bare None
    # default would auto-load data/v1/ordinal_bootstrap_replicates.json
    # when present (that auto-load is exercised end to end by
    # test_train_bootstrap_replicates.py's real-v1 smoke instead).
    matches_df = pred.evaluate.load_matches_table(Path("data"), "v1")
    row = matches_df[
        (matches_df["best_of"] == "Bo3")
        & (matches_df["date"] >= "2026-07-01")
    ].iloc[0]
    predictor = pred.make_predictor(
        "data", "v1", n_samples=2, seed=2026, ci_level=0.9,
        bootstrap_models=(),
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


# --------------------------------------------------------------------------
# plan#6: Predictor wraps make_predictor (fast, monkeypatched factory)
# --------------------------------------------------------------------------


def _canned_prediction_result(n_games_backing: int) -> PredictionResult:
    """Build one canned PredictionResult for the stream-mode stubs.

    Constructs a full :class:`PredictionResult` — a single Bind
    per-map entry whose ``n_games_backing`` is the passed value (the
    per-line discriminator the stream-mode tests vary by the queried
    ``team_a``), plus the canned veto/series/sensitivity records —
    mirroring the construction pattern of
    ``test_main_prints_json_result``'s inline stub so each printed
    stream line is a distinguishably different, fully
    ``json.dumps``-serializable object while the whole canned result
    stays hand-assertable.

    Args:
        n_games_backing: The per-map entry's ``n_games_backing``
            value.

    Returns:
        A fresh ``PredictionResult`` whose single ``per_map`` entry
        carries ``n_games_backing`` equal to the argument.

    Raises:
        Nothing.
    """
    return PredictionResult(
        predicted_veto=_STUB_VETO_ACTIONS,
        per_map=(
            PerMapPrediction(
                map_name="Bind",
                probabilities=(0.6, 0.1, 0.1, 0.2),
                interval_low=None,
                interval_high=None,
                n_games_backing=n_games_backing,
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


def test_predictor_wraps_make_predictor_single_load(tmp_path, monkeypatch):
    # E1: Predictor.__init__ invokes make_predictor exactly once,
    # forwarding output_dir/version and every keyword unchanged (no
    # top_n at construction — M39.4/G3, A5); each
    # .predict call delegates to the returned closure with the exact
    # arguments passed plus the forwarded keyword-only top_n
    # (defaulting to DEFAULT_TOP_N per call) and returns its result
    # unmodified. The factory
    # is never re-invoked per call.
    factory_calls = {"count": 0}
    forwarded = {}
    closure_calls = []
    returned = []

    def stub_make_predictor(
        output_dir,
        version,
        *,
        n_samples=pred.DEFAULT_N_SAMPLES,
        seed=pred.DEFAULT_SEED,
        ci_level=pred.DEFAULT_CI_LEVEL,
        bootstrap_models=None,
    ):
        factory_calls["count"] += 1
        forwarded.update(
            output_dir=output_dir,
            version=version,
            n_samples=n_samples,
            seed=seed,
            ci_level=ci_level,
            bootstrap_models=bootstrap_models,
        )

        def stub_predict(
            team_a, team_b, best_of, map_pool, as_of_date, *, top_n=10
        ):
            closure_calls.append(
                (team_a, team_b, best_of, map_pool, as_of_date, top_n)
            )
            result = _canned_prediction_result(
                n_games_backing=len(closure_calls)
            )
            returned.append(result)
            return result

        return stub_predict

    monkeypatch.setattr(pred, "make_predictor", stub_make_predictor)
    predictor = Predictor(tmp_path, "v2", n_samples=5, seed=9, ci_level=0.8)
    assert factory_calls["count"] == 1
    assert forwarded == {
        "output_dir": tmp_path,
        "version": "v2",
        "n_samples": 5,
        "seed": 9,
        "ci_level": 0.8,
        "bootstrap_models": None,
    }

    first = predictor.predict(
        "A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00"
    )
    second = predictor.predict("C", "D", "Bo1", None, "2026-01-02T00:00:00")
    third = predictor.predict(
        "E", "F", "Bo5", _STUB_POOL, "2026-01-03T00:00:00"
    )

    assert factory_calls["count"] == 1
    assert closure_calls == [
        ("A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00", 10),
        ("C", "D", "Bo1", None, "2026-01-02T00:00:00", 10),
        ("E", "F", "Bo5", _STUB_POOL, "2026-01-03T00:00:00", 10),
    ]
    # Each .predict return is the stub closure's own result object,
    # unmodified.
    assert first is returned[0]
    assert second is returned[1]
    assert third is returned[2]
    assert first.to_dict()["per_map"][0]["n_games_backing"] == 1
    assert second.to_dict()["per_map"][0]["n_games_backing"] == 2


# --------------------------------------------------------------------------
# plan#7: Predictor real wiring loads once (synthetic league)
# --------------------------------------------------------------------------


def test_predictor_real_wiring_loads_once(tmp_path, monkeypatch):
    # E1 end to end through the REAL make_predictor wiring (only the
    # I/O and model sources are stubbed, each wrapped with a call
    # counter): constructing one Predictor invokes each table loader,
    # the Stage-2 registry factory and _load_veto_models exactly once;
    # two .predict calls with different team_a/team_b/as_of_date values
    # re-run the per-call M31 work (counter reaches 2) without
    # re-loading anything. Both results are the hand-computable
    # synthetic values: per-map probabilities (0.6, 0.1, 0.1, 0.2),
    # backing [7, 4, 3] for the A/B call and [0, 0, 0] for the C/D
    # call (neither side has stub history), series aggregate
    # 0.465/0.335/0.1/0.1.
    matches_df, maps_df, player_map_stats_df = _league_tables()
    counters = {
        "matches": 0,
        "maps": 0,
        "stats": 0,
        "registry": 0,
        "veto_models": 0,
        "m31": 0,
    }

    def counting_matches(output_dir, version):
        counters["matches"] += 1
        return matches_df

    def counting_maps(output_dir, version):
        counters["maps"] += 1
        return maps_df

    def counting_stats(output_dir, version):
        counters["stats"] += 1
        return player_map_stats_df

    def counting_registry(output_dir, version):
        counters["registry"] += 1
        return _stub_map_model_fn

    def counting_veto_models(output_dir, version):
        counters["veto_models"] += 1
        return (None, None)

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
        counters["m31"] += 1
        assert callable(map_model_fn)
        assert set(predictor_fn_by_action) == {"ban", "pick"}
        assert isinstance(rng, np.random.Generator)
        assert n_samples > 0
        assert map_pool is None or tuple(map_pool) == tuple(_STUB_POOL)
        return _stub_prediction(team1_id, team2_id, best_of, date, n_samples)

    monkeypatch.setattr(
        pred.evaluate, "load_matches_table", counting_matches
    )
    monkeypatch.setattr(pred.evaluate, "load_maps_table", counting_maps)
    monkeypatch.setattr(
        pred.evaluate, "load_player_map_stats_table", counting_stats
    )
    monkeypatch.setitem(
        pred.evaluate.MODEL_REGISTRY,
        pred._TEMPERATURE_MAP_MODEL_KEY,
        counting_registry,
    )
    monkeypatch.setattr(pred, "_load_veto_models", counting_veto_models)
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
    # M39.4 (G2): predict()'s fourth step enumerates over the wired
    # predictor_fn_by_action dict, so the make_veto_step_predictor_fn
    # factories must produce the deterministic stub arm (the counting
    # veto_models loader returns (None, None), which the real
    # factories' closures would crash on when invoked).
    monkeypatch.setattr(
        pred.conditional_logit_ban,
        "make_veto_step_predictor_fn",
        lambda model: _top_two_step_stub,
    )
    monkeypatch.setattr(
        pred.conditional_logit_pick,
        "make_veto_step_predictor_fn",
        lambda model: _top_two_step_stub,
    )

    predictor = Predictor(tmp_path, "v1", n_samples=3, seed=2026, ci_level=0.9)
    assert counters == {
        "matches": 1,
        "maps": 1,
        "stats": 1,
        "registry": 1,
        "veto_models": 1,
        "m31": 0,
    }

    result_a = predictor.predict(
        "A", "B", "Bo3", _STUB_POOL, "2026-01-01T00:00:00"
    )
    # The None-pool call keeps an explicit map_pool=None (the M31 stub
    # tolerates it), but its as-of date must sit inside a configured
    # config.json era: since M39.4 the predict() enumeration resolves a
    # None pool against config (as the real greedy simulator always
    # did), and the synthetic 2026-01-02 predates the first era
    # (2026-04-01). 2026-04-15 is inside era 2026-s1-bind.
    result_c = predictor.predict(
        "C", "D", "Bo3", None, "2026-04-15T00:00:00"
    )

    # The loads still happened exactly once; only the per-call M31
    # work advanced across the two calls.
    assert counters == {
        "matches": 1,
        "maps": 1,
        "stats": 1,
        "registry": 1,
        "veto_models": 1,
        "m31": 2,
    }

    # Hand-computable result shapes for both calls.
    for result in (result_a, result_c):
        assert [a.action for a in result.predicted_veto] == [
            "ban", "ban", "pick", "pick", "ban", "ban", "decider"
        ]
        assert [entry.map_name for entry in result.per_map] == [
            "Bind", "Ascent", "Sunset"
        ]
        for entry in result.per_map:
            assert entry.probabilities == (0.6, 0.1, 0.1, 0.2)
            assert entry.interval_low is None
            assert entry.interval_high is None
        assert result.series.best_of == 3
        assert result.series.probabilities == pytest.approx(
            [0.465, 0.335, 0.1, 0.1]
        )
    # A/B backing is the hand-known min-games (7/4/3); C/D have no
    # stub history, so every backing is 0.
    assert [e.n_games_backing for e in result_a.per_map] == [7, 4, 3]
    assert [e.n_games_backing for e in result_c.per_map] == [0, 0, 0]


# --------------------------------------------------------------------------
# plan#8: parse_args --stream flag and E3 mutual-exclusion validation
# --------------------------------------------------------------------------


def test_parse_args_stream_flag_alone():
    # E2/E3: parse_args(["--stream"]) succeeds with no query flags
    # (the manual post-parse checks pass with all four at None) and
    # the stream attribute is True.
    args = pred.parse_args(["--stream"])
    assert args.stream is True
    assert args.team_a is None
    assert args.team_b is None
    assert args.best_of is None
    assert args.as_of_date is None
    assert args.map_pool is None
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED
    assert args.ci_level == DEFAULT_CI_LEVEL


@pytest.mark.parametrize(
    "query_flags",
    [
        ["--team-a", "A"],
        ["--team-b", "B"],
        ["--best-of", "Bo3"],
        ["--as-of-date", "2026-01-01T00:00:00"],
        ["--map-pool", "Bind,Haven,Split,Ascent,Lotus,Icebox,Sunset"],
    ],
)
def test_parse_args_stream_rejects_any_query_flag(query_flags):
    # E3: --stream is mutually exclusive with each of the five query
    # flags — any one combined with --stream is a SystemExit from the
    # manual post-parse check.
    with pytest.raises(SystemExit):
        pred.parse_args(["--stream"] + query_flags)


@pytest.mark.parametrize(
    "flag", ["--team-a", "--team-b", "--best-of", "--as-of-date"]
)
def test_parse_args_requires_all_four_query_flags_without_stream(flag):
    # E3 re-confirmation (new assertion, not a modification of
    # test_parse_args_defaults): without --stream, omitting any one of
    # the four query flags is a SystemExit from the manual post-parse
    # required-arg check.
    full = [
        "--team-a", "A",
        "--team-b", "B",
        "--best-of", "Bo3",
        "--as-of-date", "2026-01-01T00:00:00",
    ]
    index = full.index(flag)
    del full[index:index + 2]
    with pytest.raises(SystemExit):
        pred.parse_args(full)


# --------------------------------------------------------------------------
# plan#9: main() stream mode end-to-end (stubbed)
# --------------------------------------------------------------------------


def test_main_stream_mode_end_to_end(capsys, monkeypatch):
    # E4-E6: main(["--stream"]) builds the Predictor (via the stubbed
    # make_predictor — called exactly once), answers a JSONL stdin
    # stream of one blank line + two valid queries (one with an
    # explicit map_pool array, one without), and prints exactly two
    # compact one-line JSON results whose per_map n_games_backing
    # reflects each query's team_a. The map_pool-bearing query reaches
    # the stub closure as a tuple; the other reaches it as None.
    factory_calls = {"count": 0}
    call_log = []

    def stub_make_predictor(
        output_dir,
        version,
        *,
        n_samples=pred.DEFAULT_N_SAMPLES,
        seed=pred.DEFAULT_SEED,
        ci_level=pred.DEFAULT_CI_LEVEL,
        bootstrap_models=None,
    ):
        factory_calls["count"] += 1
        assert n_samples == 3
        assert seed == 7
        assert ci_level == 0.9

        def stub_predict(
            team_a, team_b, best_of, map_pool, as_of_date, *, top_n=10
        ):
            call_log.append(
                (team_a, team_b, best_of, map_pool, as_of_date, top_n)
            )
            # Vary the per-map backing by the queried team_a so each
            # printed line is distinguishable.
            return _canned_prediction_result(n_games_backing=int(team_a))

        return stub_predict

    monkeypatch.setattr(pred, "make_predictor", stub_make_predictor)
    pool = ["Bind", "Haven", "Split", "Ascent", "Lotus", "Icebox",
            "Sunset"]
    stdin_text = "\n".join([
        "",
        json.dumps({
            "team_a": "10", "team_b": "B", "best_of": "Bo3",
            "as_of_date": "2026-01-01T00:00:00", "map_pool": pool,
        }),
        json.dumps({
            "team_a": "20", "team_b": "D", "best_of": "Bo3",
            "as_of_date": "2026-01-02T00:00:00",
        }),
        "",
    ])
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))

    rc = pred.main(
        ["--stream", "--n-samples", "3", "--seed", "7", "--ci-level",
         "0.9", "--top-n", "5"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    output_lines = [
        line for line in captured.out.splitlines() if line.strip()
    ]
    # One blank stdin line produced no output; exactly two results.
    assert len(output_lines) == 2
    parsed_lines = [json.loads(line) for line in output_lines]
    assert [line["per_map"][0]["n_games_backing"] for line in parsed_lines] == [
        10, 20
    ]
    # M39.4 (G7): the session-level --top-n reaches every per-query
    # Predictor.predict call (no per-query top_n in the JSON schema),
    # and every printed line carries the additive top_vetos key.
    assert [line["top_vetos"] for line in parsed_lines] == [[], []]
    # One predictor build for the whole stream; the queries reached the
    # closure with the map_pool as tuple / None respectively and the
    # session top_n=5 on both.
    assert factory_calls["count"] == 1
    assert call_log[0][3] == tuple(pool)
    assert call_log[1][3] is None
    assert call_log[0][0] == "10"
    assert call_log[1][0] == "20"
    assert [call[5] for call in call_log] == [5, 5]


# --------------------------------------------------------------------------
# plan#10: stream-mode error propagation (stubbed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdin_text, expected_exception",
    [
        # A malformed JSON line aborts the stream with the concrete
        # json.JSONDecodeError uncaught.
        ("{not valid json\n", json.JSONDecodeError),
        # A query object missing the required "team_a" key aborts the
        # stream with KeyError uncaught.
        (
            json.dumps({
                "team_b": "B", "best_of": "Bo3",
                "as_of_date": "2026-01-01T00:00:00",
            })
            + "\n",
            KeyError,
        ),
    ],
)
def test_stream_mode_errors_propagate_uncaught(
    monkeypatch, stdin_text, expected_exception
):
    # E5: neither a malformed JSON line nor a query missing a required
    # key is swallowed — the exception propagates uncaught out of
    # main() (after the single Predictor build), terminating the
    # stream.
    factory_calls = {"count": 0}

    def stub_make_predictor(
        output_dir,
        version,
        *,
        n_samples=pred.DEFAULT_N_SAMPLES,
        seed=pred.DEFAULT_SEED,
        ci_level=pred.DEFAULT_CI_LEVEL,
        bootstrap_models=None,
    ):
        factory_calls["count"] += 1

        def stub_predict(
            team_a, team_b, best_of, map_pool, as_of_date, *, top_n=10
        ):
            return _canned_prediction_result(n_games_backing=1)

        return stub_predict

    monkeypatch.setattr(pred, "make_predictor", stub_make_predictor)
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
    with pytest.raises(expected_exception):
        pred.main(["--stream"])
    assert factory_calls["count"] == 1



# --------------------------------------------------------------------------
# plan#10: real-v1 top_vetos smoke test (slow + skip-guarded)
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(),
    reason="real v1 tables/artifacts not present",
)
def test_real_v1_top_vetos_smoke():
    # A tiny real-v1 run (n=3, no bootstrap) against the real fitted
    # ban/pick models and tables for one mid-season Bo3 match: the
    # exact 5,040-sequence enumeration stays tractable (the F2
    # memoisation caps real per-step predictor calls at 120), the
    # returned three entries are finite, sorted descending by
    # veto_probability, every result carries veto_sensitivity None and
    # a Bo3 series summing to 1, and the whole listing is
    # json.dumps-serializable. Wall-clock is measured (the real check
    # that F2's memoisation works — a non-memoised walk would take
    # minutes here) and reported in the BUILD status.md note.
    import time

    matches_df = pred.evaluate.load_matches_table(Path("data"), "v1")
    row = matches_df[
        (matches_df["best_of"] == "Bo3")
        & (matches_df["date"] >= "2026-07-01")
    ].iloc[0]
    top_vetos = pred.make_top_vetos_fn("data", "v1", ci_level=0.9)
    start = time.monotonic()
    entries = top_vetos(
        str(row["team1_id"]),
        str(row["team2_id"]),
        "Bo3",
        None,
        row["date"],
        n=3,
    )
    elapsed = time.monotonic() - start

    assert len(entries) == 3
    probabilities = [e.veto_probability for e in entries]
    assert probabilities == sorted(probabilities, reverse=True)
    for entry in entries:
        assert 0.0 <= entry.veto_probability <= 1.0
        result = entry.result
        assert len(result.predicted_veto) == 7
        assert [a.action for a in result.predicted_veto] == [
            "ban", "ban", "pick", "pick", "ban", "ban", "decider"
        ]
        assert len(result.per_map) == 3
        for pm in result.per_map:
            assert len(pm.probabilities) == 4
            assert sum(pm.probabilities) == pytest.approx(1.0)
            assert all(np.isfinite(p) for p in pm.probabilities)
            assert pm.interval_low is None
            assert pm.interval_high is None
            assert pm.n_games_backing >= 0
        assert result.veto_sensitivity is None
        assert result.series.best_of == 3
        assert result.series.outcome_order == series_paths.series_outcome_order(3)
        assert len(result.series.probabilities) == 4
        assert sum(result.series.probabilities) == pytest.approx(1.0)
    # The full listing round-trips through json.dumps.
    json.dumps([e.to_dict() for e in entries])
    assert elapsed > 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(),
    reason="real v1 tables/artifacts not present",
)
def test_real_v1_predict_top_vetos_smoke():
    # A tiny real-v1 run of the M39.4 fold (n_samples=2, no bootstrap,
    # default top_n=10) against the real fitted models and tables for
    # one mid-season Bo3 match: predict() completes (the exact
    # 5,040-sequence enumeration stays tractable via F2's memoisation
    # — ~120 real predictor calls — plus ten per-veto result
    # constructions) and returns a non-empty top_vetos tuple of ten
    # entries sorted descending by veto_probability, each with
    # veto_sensitivity None (G6) and None intervals (the () escape
    # hatch), while the top-level veto_sensitivity stays a real
    # non-None M31 summary, and the whole widened result is
    # json.dumps-serializable. Wall-clock is measured (the added
    # per-call enumeration cost) and reported in the BUILD status.md
    # note.
    import time

    matches_df = pred.evaluate.load_matches_table(Path("data"), "v1")
    row = matches_df[
        (matches_df["best_of"] == "Bo3")
        & (matches_df["date"] >= "2026-07-01")
    ].iloc[0]
    predictor = pred.make_predictor(
        "data", "v1", n_samples=2, seed=2026, ci_level=0.9,
        bootstrap_models=(),
    )
    start = time.monotonic()
    result = predictor(
        str(row["team1_id"]),
        str(row["team2_id"]),
        "Bo3",
        None,
        row["date"],
    )
    elapsed = time.monotonic() - start

    assert len(result.top_vetos) == 10
    probabilities = [e.veto_probability for e in result.top_vetos]
    assert probabilities == sorted(probabilities, reverse=True)
    assert result.veto_sensitivity is not None
    for entry in result.top_vetos:
        assert 0.0 <= entry.veto_probability <= 1.0
        ranked_result = entry.result
        assert ranked_result.veto_sensitivity is None
        assert len(ranked_result.predicted_veto) == 7
        assert [a.action for a in ranked_result.predicted_veto] == [
            "ban", "ban", "pick", "pick", "ban", "ban", "decider"
        ]
        assert len(ranked_result.per_map) == 3
        for pm in ranked_result.per_map:
            assert len(pm.probabilities) == 4
            assert sum(pm.probabilities) == pytest.approx(1.0)
            assert all(np.isfinite(p) for p in pm.probabilities)
            assert pm.interval_low is None
            assert pm.interval_high is None
            assert pm.n_games_backing >= 0
        assert ranked_result.series.best_of == 3
        assert len(ranked_result.series.probabilities) == 4
        assert sum(ranked_result.series.probabilities) == pytest.approx(1.0)
        assert ranked_result.top_vetos == ()
    # The full widened result round-trips through json.dumps.
    json.dumps(result.to_dict())
    assert elapsed > 0.0
