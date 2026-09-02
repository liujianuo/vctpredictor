"""Tests for the M37 veto-conditional variance CLI driver
(drivers/evaluate_veto_conditional_variance.py).

Covers the CLI/IO glue only — the pure spread math is already tested in
tests/test_veto_conditional_variance.py and the M31 pipeline in
tests/test_veto_marginalized_series.py. The tests here are:
``parse_args`` defaults and flag overrides; a synthetic end-to-end
``main()`` run with the table/artifact loaders and the M31
``predict_series_outcome_via_veto_marginalization`` entry point
monkeypatched to deterministic stubs whose per-sample scoreline detail
is hand-computable — including one fully-deterministic-veto stub case
(the Bo5 series: every sampled veto sequence produces the identical
scoreline distribution, so ``mean_band_width == 0`` per series — the
"resolves the moment the veto happens" boundary case) and one genuinely-
stochastic stub case (the Bo3 series: a nonzero spread whose bands and
weighted moments are cross-checked against independently re-derived
numpy percentiles / hand arithmetic) — plus the single-rng sequential-
consumption wiring (the M36-contrast mechanism: the same Generator
object must be passed to every series call, never reconstructed); the
missing-prerequisite-artifact ``FileNotFoundError`` contract; invalid
flag values; and a ``skipif``-guarded real-v1 integration smoke test
asserting finite, ``[0, 1]``-bounded, ``lo <= hi`` bands, a
``json.dumps``-serializable report, and the expected Bo5-absence guard
in the aggregate. No real fitted artifacts are required by the non-smoke
tests.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drivers import evaluate_veto_conditional_variance as ecv
from evaluation.veto_marginalized_series import (
    SeriesVetoSample,
    VetoMarginalizedSeriesPrediction,
)
from models.ancestral_veto_sampler import (
    SampledVetoAction,
    SampledVetoSequence,
)
from utils import series_paths

# The default knobs the parse_args defaults must match (referenced
# through the module constants so this test never hardcodes a stale
# value).
DEFAULT_N_SAMPLES = ecv.DEFAULT_N_SAMPLES
DEFAULT_SEED = ecv.DEFAULT_SEED
DEFAULT_CI_LEVEL = ecv.DEFAULT_CI_LEVEL

# A tiny hand-built league (the evaluate_series league): m1/m2 Bo3 test
# matches, m3 a Bo5 test match, m4 a Bo3 train match. Team ids/dates are
# arbitrary — every stub model ignores the tables.
_MATCH_ROWS = [
    {"match_id": "m1", "date": "2026-01-01T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m2", "date": "2026-01-02T00:00:00", "team1_id": "A",
     "team2_id": "B", "best_of": "Bo3", "status": "completed"},
    {"match_id": "m3", "date": "2026-01-03T00:00:00", "team1_id": "C",
     "team2_id": "D", "best_of": "Bo5", "status": "completed"},
    {"match_id": "m4", "date": "2026-01-04T00:00:00", "team1_id": "E",
     "team2_id": "F", "best_of": "Bo3", "status": "completed"},
]

_MAP_ROWS = [
    {"match_id": "m1", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 8, "winner": "A"},
    {"match_id": "m1", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 11, "winner": "A"},
    {"match_id": "m1", "map_index": 2, "map_name": "Split",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 0, "map_name": "Bind",
     "team1_score": 8, "team2_score": 13, "winner": "B"},
    {"match_id": "m2", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 9, "winner": "A"},
    {"match_id": "m2", "map_index": 2, "map_name": "Split",
     "team1_score": 13, "team2_score": 10, "winner": "A"},
    {"match_id": "m3", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 5, "winner": "C"},
    {"match_id": "m3", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "C"},
    {"match_id": "m3", "map_index": 2, "map_name": "Split",
     "team1_score": 8, "team2_score": 13, "winner": "D"},
    {"match_id": "m3", "map_index": 3, "map_name": "Ascent",
     "team1_score": 9, "team2_score": 13, "winner": "D"},
    {"match_id": "m3", "map_index": 4, "map_name": "Icebox",
     "team1_score": 13, "team2_score": 11, "winner": "C"},
    {"match_id": "m4", "map_index": 0, "map_name": "Bind",
     "team1_score": 13, "team2_score": 6, "winner": "E"},
    {"match_id": "m4", "map_index": 1, "map_name": "Haven",
     "team1_score": 13, "team2_score": 7, "winner": "E"},
    {"match_id": "m4", "map_index": 2, "map_name": "Split",
     "team1_score": 5, "team2_score": 13, "winner": "F"},
]

_SPLIT_ROWS = [
    {"match_id": "m1", "split": "test"},
    {"match_id": "m2", "split": "test"},
    {"match_id": "m3", "split": "test"},
    {"match_id": "m4", "split": "train"},
]


def _league_tables():
    """Build the synthetic matches/maps/splits frames for the stub run.

    Returns:
        A ``(matches_df, maps_df, splits_df)`` tuple of
        ``pandas.DataFrame`` objects built from the module-level row
        constants.

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
    splits_df = pd.DataFrame(_SPLIT_ROWS, columns=["match_id", "split"])
    return matches_df, maps_df, splits_df


def _stub_sequence(best_of: str, team1_id: str, team2_id: str, date: str, index: int):
    """Build a minimal but real SampledVetoSequence dataclass for the stub.

    Constructs a genuine :class:`models.ancestral_veto_sampler
    .SampledVetoSequence` carrying one synthetic pick action (enough for
    the frozen-dataclass contract — the driver never reads the raw
    sequence fields, only the ``SeriesVetoSample`` reporting fields, but
    constructing the real types keeps the stub's shape honest).

    Args:
        best_of: The ``"Bo<N>"`` series-length string.
        team1_id: Side A's stable id.
        team2_id: Side B's stable id.
        date: The as-of date string.
        index: A per-sample index, only to make the pick's map name
            unique across samples.

    Returns:
        A ``SampledVetoSequence`` with ``sequence_probability`` equal to
        ``1.0`` (a trivial non-degenerate walk).

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
    """Build a hand-computable M31 prediction for the stub driver run.

    Produces the per-sample scoreline detail the driver reads, with
    values chosen so the resulting spread is hand-computable:

    - **Bo3 (the genuinely-stochastic veto case):** sample ``i``'s
      scoreline vector is ``[0.50 - 0.05*i, 0.30 + 0.05*i, 0.10, 0.10]``
      (a valid simplex for every ``i``) with weights cycling
      ``[0.5, 0.3, 0.2]``, so the per-category bands at ``ci_level =
      0.9`` are ``(0.405, 0.495)`` / ``(0.305, 0.395)`` / ``(0.1, 0.1)``
      / ``(0.1, 0.1)``, the weighted means are ``(0.465, 0.335, 0.10,
      0.10)`` (the same convex combination the M31 weighted average
      defines — the test asserts the driver's ``weighted_mean`` equals
      the recorded ``point_estimate`` exactly, cross-validating the
      wiring against M31's own aggregation), and the weighted variances
      are ``(0.001525, 0.001525, 0.0, 0.0)``.
    - **Bo5 (the fully-deterministic-veto case):** every sample's
      vector is the same ``[0.3, 0.3, 0.2, 0.1, 0.1, 0.0]`` with uniform
      weights, so every band collapses to a point and ``mean_band_width
      == 0.0`` per series — the "resolves the moment the veto happens"
      boundary case where there is effectively no veto-sequence
      ambiguity.

    The aggregated ``probabilities`` are the weighted mean of the sample
    rows, and the ``outcome_order`` is the canonical
    ``utils.series_paths.series_outcome_order`` vocabulary — both
    exactly as the real M31 function produces.

    Args:
        team1_id: Side A's stable id (embedded in the sample sequences).
        team2_id: Side B's stable id.
        best_of: The ``"Bo<N>"`` series-length string.
        date: The as-of date string (embedded in the sample sequences).
        n_samples: How many samples to produce (the driver's
            ``--n-samples``).

    Returns:
        A ``VetoMarginalizedSeriesPrediction`` with ``n_samples``
        ``SeriesVetoSample`` records carrying the hand-computable
        vectors and weights above.

    Raises:
        Nothing.
    """
    best_of_int = int(best_of[2:])
    if best_of == "Bo5":
        rows = [[0.3, 0.3, 0.2, 0.1, 0.1, 0.0]] * n_samples
        weights = [1.0 / n_samples] * n_samples
    else:
        rows = [
            [0.50 - 0.05 * i, 0.30 + 0.05 * i, 0.10, 0.10]
            for i in range(n_samples)
        ]
        weights = [[0.5, 0.3, 0.2][i % 3] for i in range(n_samples)]
    # The weighted average, exactly as M31's aggregation defines it.
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


def _stub_predict(
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
    call_state=None,
):
    """Stub replacement for the M31 veto-marginalized entry point.

    Asserts the driver wired the two pluggable callables through
    (``map_model_fn`` callable, ``predictor_fn_by_action`` carrying
    exactly ``ban`` and ``pick``) and — the M36-contrast wiring the
    plan's assumption 9 asks REVIEW to verify — that the *same*
    ``numpy.random.Generator`` object is passed to every call: the
    stub records ``id(rng)`` per invocation and consumes one draw from
    it (``rng.random()``), so if the driver ever reconstructed or
    reset the rng per series the recorded ids would differ and the
    end-to-end test's same-identity assertion would fail. The returned
    prediction is the hand-computable :func:`_stub_prediction` for the
    series' ``best_of``.

    Args:
        team1_id / team2_id / best_of / date / matches_df / maps_df:
            The M31 pipeline arguments; only ``best_of`` (and the team
            ids/dates, embedded into the sample sequences) are read.
        map_model_fn: Asserted callable (the wired Stage-2 model).
        predictor_fn_by_action: Asserted to carry exactly the two
            Stage-1 keys.
        n_samples: The driver's ``--n-samples``, passed through.
        rng: The driver's one sequential rng; asserted a Generator and
            recorded by identity, then consumed.
        map_pool: Unused; accepted for signature parity.
        call_state: An optional dict whose ``"predict_calls"`` /
            ``"rng_ids"`` keys record the call count and the rng
            identities.

    Returns:
        A hand-computable ``VetoMarginalizedSeriesPrediction`` (see
        :func:`_stub_prediction`).

    Raises:
        AssertionError: If any wiring assertion fails.
    """
    if call_state is not None:
        call_state["predict_calls"] += 1
        call_state["rng_ids"].append(id(rng))
    assert callable(map_model_fn)
    assert set(predictor_fn_by_action) == {"ban", "pick"}
    assert isinstance(rng, np.random.Generator)
    assert n_samples > 0
    rng.random()  # consume the shared rng (sequential advancement)
    return _stub_prediction(team1_id, team2_id, best_of, date, n_samples)


@pytest.fixture
def stub_everything(monkeypatch):
    """Monkeypatch the driver's loaders and the M31 entry point.

    Routes the input tables to the synthetic league, replaces the
    artifact loader and ``predict_series_outcome_via_veto_marginalization``
    with deterministic stubs, and installs call state (exposed on the
    returned dict) so the end-to-end test can verify the driver called
    the M31 entry point exactly once per held-out series and passed the
    *same* rng object to every call. The real pure harness function
    (``build_held_out_series``) and the real spread helpers run
    untouched. All patches are reverted by monkeypatch at test teardown.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        A dict of the call state: ``predict_calls`` (int) and
        ``rng_ids`` (list of the per-call rng object ids).

    Raises:
        Nothing.
    """
    matches_df, maps_df, splits_df = _league_tables()
    player_map_stats_df = pd.DataFrame(
        {"player_id": [], "map_name": [], "team_id": [], "n_wins": []}
    )
    call_state = {"predict_calls": 0, "rng_ids": []}

    def stub_predict(
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
        return _stub_predict(
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
            map_pool=map_pool,
            call_state=call_state,
        )

    monkeypatch.setattr(ecv.evaluate, "load_matches_table",
                        lambda output_dir, version: matches_df)
    monkeypatch.setattr(ecv.evaluate, "load_maps_table",
                        lambda output_dir, version: maps_df)
    monkeypatch.setattr(ecv.evaluate, "load_splits_table",
                        lambda output_dir, version: splits_df)
    monkeypatch.setattr(ecv.evaluate, "load_player_map_stats_table",
                        lambda output_dir, version: player_map_stats_df)
    monkeypatch.setattr(ecv, "_load_fitted_models",
                        lambda output_dir, version: (None, None, None))
    monkeypatch.setattr(
        ecv.veto_marginalized_series,
        "predict_series_outcome_via_veto_marginalization",
        stub_predict,
    )
    return call_state


# --------------------------------------------------------------------------
# plan#6a: parse_args defaults and flag overrides
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # No flags: the documented defaults for all five knobs.
    args = ecv.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.n_samples == DEFAULT_N_SAMPLES
    assert args.seed == DEFAULT_SEED
    assert args.ci_level == DEFAULT_CI_LEVEL


def test_parse_args_flag_overrides():
    # Every flag overrides its default; non-int --n-samples/--seed and
    # non-float --ci-level are rejected by argparse (SystemExit).
    args = ecv.parse_args(
        ["--version", "v2", "--output-dir", "out",
         "--n-samples", "40", "--seed", "7", "--ci-level", "0.8"]
    )
    assert args.version == "v2"
    assert args.output_dir == "out"
    assert args.n_samples == 40
    assert args.seed == 7
    assert args.ci_level == 0.8
    with pytest.raises(SystemExit):
        ecv.parse_args(["--n-samples", "many"])
    with pytest.raises(SystemExit):
        ecv.parse_args(["--ci-level", "wide"])


# --------------------------------------------------------------------------
# plan#6b: synthetic end-to-end main() with hand-computable stubs
# --------------------------------------------------------------------------


def test_main_end_to_end_synthetic_report(tmp_path, stub_everything):
    # A full main() run against the synthetic league with every loader
    # and the M31 entry point stubbed: the artifact is written with the
    # interval-definition caveat, the config block, and 3 per-series
    # entries (m1/m2 Bo3, m3 Bo5). The Bo3 entries' bands / weighted
    # moments match the hand-computed values (numpy-percentile
    # re-derivation for the bands; hand arithmetic for the moments), the
    # Bo5 entry has mean_band_width exactly 0.0 (the fully-deterministic-
    # veto boundary case), the aggregate groups by best_of with the
    # guarded mean-of-mean_band_width headline, the M31 entry point was
    # called exactly once per series, and — the M36-contrast wiring —
    # the SAME rng object was passed to every call (sequential
    # consumption, never reconstructed per series).
    call_state = stub_everything
    rc = ecv.main(
        ["--output-dir", str(tmp_path),
         "--n-samples", "3", "--seed", "2026", "--ci-level", "0.9"]
    )
    assert rc == 0

    artifact_path = tmp_path / "v1" / "veto_conditional_variance_report.json"
    assert artifact_path.exists()
    text = artifact_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    report = json.loads(text)

    assert set(report) == {
        "interval_definition", "config", "per_series", "aggregate_by_best_of"
    }
    assert "joint simplex" in report["interval_definition"]
    assert "M36" in report["interval_definition"]
    assert report["config"] == {
        "n_samples": 3,
        "seed": 2026,
        "ci_level": 0.9,
    }

    # Wiring: one M31 call per held-out series (m1/m2/m3), all with the
    # same rng object (never reconstructed per series — the M37 plan's
    # assumption-9 contrast with M36's fixed-reset rng).
    assert call_state["predict_calls"] == 3
    assert len(call_state["rng_ids"]) == 3
    assert len(set(call_state["rng_ids"])) == 1

    per_series = report["per_series"]
    assert len(per_series) == 3
    by_match = {entry["match_id"]: entry for entry in per_series}

    # The Bo3 stochastic case (m1/m2): hand-computed bands, widths,
    # mean_band_width, and weighted moments.
    for match_id in ("m1", "m2"):
        entry = by_match[match_id]
        assert entry["best_of"] == "Bo3"
        assert entry["best_of_int"] == 3
        assert entry["outcome_order"] == [[2, 0], [2, 1], [1, 2], [0, 2]]
        assert entry["point_estimate"] == pytest.approx(
            [0.465, 0.335, 0.10, 0.10]
        )
        assert entry["unweighted_band_low"] == pytest.approx(
            [0.405, 0.305, 0.1, 0.1]
        )
        assert entry["unweighted_band_high"] == pytest.approx(
            [0.495, 0.395, 0.1, 0.1]
        )
        assert entry["band_widths"] == pytest.approx([0.09, 0.09, 0.0, 0.0])
        assert entry["mean_band_width"] == pytest.approx(0.045)
        assert entry["weighted_mean"] == pytest.approx(
            [0.465, 0.335, 0.10, 0.10]
        )
        assert entry["weighted_variance"] == pytest.approx(
            [0.001525, 0.001525, 0.0, 0.0]
        )
        # The weighted_mean must equal the point_estimate exactly (M31's
        # aggregate IS the weighted average of the same per-sample
        # vectors with the same weights — cross-validates the wiring).
        assert entry["weighted_mean"] == pytest.approx(
            entry["point_estimate"]
        )
        for low, high in zip(
            entry["unweighted_band_low"], entry["unweighted_band_high"]
        ):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0

    # The Bo5 fully-deterministic-veto case (m3): every sampled veto
    # sequence produces the identical scoreline distribution, so every
    # band collapses to a point and mean_band_width is exactly 0.0.
    bo5 = by_match["m3"]
    assert bo5["best_of"] == "Bo5"
    assert bo5["best_of_int"] == 5
    assert bo5["point_estimate"] == pytest.approx(
        [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]
    )
    assert bo5["unweighted_band_low"] == pytest.approx(
        [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]
    )
    assert bo5["unweighted_band_high"] == pytest.approx(
        [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]
    )
    assert bo5["mean_band_width"] == 0.0
    assert bo5["weighted_variance"] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # The per-best_of aggregate: Bo3 has 2 series with mean
    # mean_band_width 0.045; Bo5 has 1 series with mean 0.0.
    assert report["aggregate_by_best_of"] == {
        "Bo3": {"n_series": 2, "mean_mean_band_width": 0.045},
        "Bo5": {"n_series": 1, "mean_mean_band_width": 0.0},
    }

    # The full report round-trips through json.dumps (every value is a
    # plain str/int/float/list/dict).
    json.dumps(report)


# --------------------------------------------------------------------------
# plan#6c: missing prerequisite artifact and invalid flag values
# --------------------------------------------------------------------------


def test_main_missing_prerequisite_artifact_raises_file_not_found(tmp_path):
    # No fitted artifacts exist under the empty tmp output dir: the
    # first artifact load must raise FileNotFoundError unchanged (the
    # "run the training driver first" signal), never a silent fallback
    # or a wrapped exception.
    with pytest.raises(FileNotFoundError):
        ecv.main(["--output-dir", str(tmp_path)])


def test_main_rejects_invalid_flag_values(tmp_path):
    # A non-positive --n-samples and an out-of-(0, 1) --ci-level are
    # hard errors before any work starts (even with everything stubbed
    # away the validation must fire first).
    with pytest.raises(ValueError, match="positive"):
        ecv.main(["--output-dir", str(tmp_path), "--n-samples", "0"])
    with pytest.raises(ValueError, match="--ci-level"):
        ecv.main(["--output-dir", str(tmp_path), "--ci-level", "1.5"])


# --------------------------------------------------------------------------
# plan#6: real-v1 integration smoke test (skip-guarded)
# --------------------------------------------------------------------------


def _real_v1_available():
    """Report whether the real v1 tables and model artifacts exist.

    The skip guard for the real-data smoke test: the materialised v1
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


@pytest.mark.skipif(
    not _real_v1_available(), reason="real v1 tables/artifacts not present"
)
def test_real_v1_smoke_finite_well_ordered_bands():
    # A tiny real-v1 run (n_samples=2) against the real fitted models:
    # the artifact must be written, every per-series band must be
    # ordered lo <= hi with simplex-adjacent [0, 1] bounds (a finite,
    # well-ordered band is guaranteed by construction only in that weak
    # sense), the weighted moments must be finite and non-negative, the
    # aggregate must be guarded for the real v1 report's legitimate
    # Bo5-absence (Bo3 present, Bo5 absent — the plan's assumption-10
    # guard), and every value must be plain json.dumps-serializable.
    rc = ecv.main(
        ["--n-samples", "2", "--seed", "2026", "--ci-level", "0.9"]
    )
    assert rc == 0

    artifact_path = Path("data/v1/veto_conditional_variance_report.json")
    assert artifact_path.exists()
    report = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert report["config"]["n_samples"] == 2
    assert report["config"]["seed"] == 2026
    assert report["config"]["ci_level"] == 0.9

    # The real v1 test split is 100% Bo3: the aggregate has a Bo3
    # group and no Bo5 group (guarded, never a bare key index).
    assert "Bo3" in report["aggregate_by_best_of"]
    assert "Bo5" not in report["aggregate_by_best_of"]
    assert report["aggregate_by_best_of"]["Bo3"]["n_series"] == 15
    assert report["aggregate_by_best_of"]["Bo3"]["mean_mean_band_width"] >= 0.0

    assert len(report["per_series"]) == 15
    for entry in report["per_series"]:
        k = entry["best_of_int"] + 1
        assert len(entry["unweighted_band_low"]) == k
        assert len(entry["unweighted_band_high"]) == k
        assert len(entry["point_estimate"]) == k
        assert len(entry["weighted_mean"]) == k
        assert len(entry["weighted_variance"]) == k
        for low, high in zip(
            entry["unweighted_band_low"], entry["unweighted_band_high"]
        ):
            assert low <= high
            assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
        assert entry["mean_band_width"] >= 0.0
        assert all(v >= 0.0 for v in entry["weighted_variance"])
        assert entry["weighted_mean"] == pytest.approx(
            entry["point_estimate"]
        )
        assert len(entry["outcome_order"]) == k
        assert all(len(scoreline) == 2 for scoreline in entry["outcome_order"])

    # The full report round-trips through json.dumps.
    json.dumps(report)
