"""Tests for the P5 export driver and the decision-I provenance sidecar.

Covers, in plan order: the export CLI's ``parse_args`` defaults and
overrides; fixture reading (D2/D3 validation, D9's past-drop, the
file-level failure modes); ``build_fixture``'s fifteen-key assembly and
the D5 ``bo5_unvalidated`` flag; a full schema round-trip through
``contract.validate_artifact`` and ``json``; an end-to-end ``main()``
run with a stubbed ``Predictor`` and stubbed ``load_matches_table``;
the D12 ``intervals_present`` OR; the D13/D14 failure policy
(per-fixture ``ValueError``/``KeyError``/``ConfigError`` excluded,
``TypeError`` propagated, threshold abort writing nothing, all-past
list writing ``fixtures: []``, constructor failure fatal, bad fraction
rejected before I/O); the D15 atomic write; the decision-I sidecar
reader/writer/CLI; and one ``slow`` + skip-guarded real-v1 smoke test.

**Deliberate choice (item 24).** Every non-``slow`` test stubs both the
``Predictor`` and ``evaluate.load_matches_table``, so no test needs real
artifacts: the per-fixture loop is driven by synthetic
``PredictionResult`` records built with the same builder pattern as
``tests/test_reshape.py`` (copied here — the repo does not share
builders across test modules today), and the name map is built from a
small synthetic matches table. Only the final smoke test touches real
``data/v1``, and it copies the required files into ``tmp_path/v1`` so
it never mutates committed state (A7).
"""

import json
import logging
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError

from drivers import export_predictions as ep
from drivers import model_provenance, predict
from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    RankedVetoPrediction,
    SeriesPrediction,
    VetoSensitivity,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from presentation import contract
from tests._shared import _real_v1_available as _real_v1_tables_available
from utils import series_paths

# --------------------------------------------------------------------------
# Synthetic builders (copied from tests/test_reshape.py — see module
# docstring for why they are duplicated rather than imported).
# --------------------------------------------------------------------------


def _make_series(probabilities, best_of, *, outcome_order=None):
    """Build a synthetic :class:`SeriesPrediction`.

    Args:
        probabilities: The ``best_of + 1`` scoreline probabilities.
        best_of: The parsed map count (``1``/``3``/``5``), used to
            pick the canonical ``outcome_order`` when ``outcome_order``
            is not given.
        outcome_order: An optional explicit ``(a_wins, b_wins)`` pair
            sequence overriding the canonical order.

    Returns:
        A frozen :class:`SeriesPrediction` wrapping the probability
        vector (as a tuple), the canonical or explicit outcome order
        (as a tuple), and ``best_of``.

    Raises:
        ValueError: If ``best_of`` is not a valid odd map count when
            ``outcome_order`` is not supplied — propagated unchanged
            from :func:`utils.series_paths.series_outcome_order`.
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
            ``OUTCOME_LABELS`` order.
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
        first_map: The map side A bans at step 0.

    Returns:
        A 7-tuple of :class:`SimulatedVetoAction` records in the Bo3
        shape ``ban, ban, pick, pick, ban, ban, decider`` with team
        ids alternating ``"A"``/``"B"`` by step parity and ``team=None``
        on the decider.

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

    Args:
        predicted_veto: The entry's veto action sequence.
        series: The entry's exact-M30 :class:`SeriesPrediction`.
        veto_probability: The exact joint probability assigned to the
            entry.
        per_map: The entry's played-map records (defaults to empty).

    Returns:
        A frozen :class:`RankedVetoPrediction` whose ``result`` carries
        the given sequence, series and per-map records.

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


# --------------------------------------------------------------------------
# Local helpers (this file only).
# --------------------------------------------------------------------------


def _write_json(path, obj):
    """Write ``obj`` to ``path`` as the repo's JSON serialization.

    Args:
        path: The destination file path.
        obj: The JSON-serializable object to write.

    Returns:
        None.

    Raises:
        TypeError: If ``obj`` is not JSON-serializable (propagated from
            ``json.dumps``).
        OSError: If the file cannot be written.
    """
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _make_spec(match_id="1", *, best_of="Bo3", best_of_int=3,
               team_a_id="A", team_b_id="B", scheduled_at=None):
    """Build a synthetic :class:`FixtureSpec`.

    Args:
        match_id: The fixture's match id.
        best_of: The series length string.
        best_of_int: The derived map count (must agree with
            ``best_of``).
        team_a_id: Team A's id.
        team_b_id: Team B's id.
        scheduled_at: The scheduled-at ISO string (defaults to a
            near-future value after the tests' usual as-of dates).

    Returns:
        A frozen :class:`FixtureSpec` with the given fields.

    Raises:
        Nothing.
    """
    if scheduled_at is None:
        scheduled_at = "2026-09-13T10:00:00"
    return ep.FixtureSpec(
        match_id=match_id,
        event="Example Event",
        scheduled_at=scheduled_at,
        best_of=best_of,
        best_of_int=best_of_int,
        team_a_id=team_a_id,
        team_b_id=team_b_id,
    )


def _stub_matches_df():
    """Build the synthetic matches table backing the name-map stubs.

    Returns:
        A ``pandas.DataFrame`` with the four id/name columns (plus a
        ``date`` column) mapping ``"A"`` → ``"Team A"`` and ``"B"`` →
        ``"Team B"``.

    Raises:
        Nothing.
    """
    return pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team1_id": ["A"],
            "team1_name": ["Team A"],
            "team2_id": ["B"],
            "team2_name": ["Team B"],
        }
    )


def _load_matches_table_stub(matches_df, output_dir, version):
    """A ``load_matches_table`` stand-in returning a fixed DataFrame.

    Args:
        matches_df: The DataFrame to return on every call.
        output_dir: Ignored (matches the real loader's signature).
        version: Ignored (matches the real loader's signature).

    Returns:
        ``matches_df`` unchanged.

    Raises:
        Nothing.
    """
    return matches_df


class _StubPredictor:
    """A recording stand-in for ``drivers.predict.Predictor``.

    When called (as the monkeypatched ``drivers.predict.Predictor``) it
    records the construction keyword arguments and returns itself; its
    ``predict`` method records each call and returns the next canned
    result from ``results`` (cycled), or raises ``error`` when set — so
    tests can drive the export loop without real artifacts.

    Attributes:
        construction_kwargs: A list of the keyword dicts passed to each
            construction call (one entry per ``__call__`` invocation).
        calls: A list of 6-tuples recording every ``predict`` call's
            ``(team_a, team_b, best_of, map_pool, as_of_date, top_n)``.
    """

    def __init__(self, results=(), error=None):
        """Initialize the stub with its canned behaviour.

        Args:
            results: An iterable of ``PredictionResult`` objects to
                return from successive ``predict`` calls, cycled.
            error: An optional exception instance to raise from every
                ``predict`` call when set (overrides ``results``).

        Returns:
            Nothing.

        Raises:
            Nothing.
        """
        self.results = list(results)
        self.error = error
        self.construction_kwargs = []
        self.calls = []
        self._index = 0

    def __call__(self, output_dir, version, *, n_samples, seed, ci_level,
                 **kwargs):
        """Record one construction and return this instance.

        Args:
            output_dir: The ``--output-dir`` value (a ``Path``).
            version: The ``--version`` value.
            n_samples: The ``--n-samples`` knob.
            seed: The ``--seed`` knob.
            ci_level: The ``--ci-level`` knob.
            **kwargs: Any additional construction keyword arguments
                (e.g. ``bootstrap_models``), recorded verbatim so tests
                can assert they were *not* passed by the export (D10).

        Returns:
            ``self``, so the constructed "predictor" is this stub.

        Raises:
            Nothing.
        """
        self.construction_kwargs.append(
            {
                "output_dir": output_dir,
                "version": version,
                "n_samples": n_samples,
                "seed": seed,
                "ci_level": ci_level,
                **kwargs,
            }
        )
        return self

    def predict(self, team_a, team_b, best_of, map_pool, as_of_date, *,
                top_n):
        """Record one call and return the next canned result.

        Args:
            team_a: The queried team A id.
            team_b: The queried team B id.
            best_of: The ``"Bo<N>"`` string.
            map_pool: The map pool (the export always passes ``None``).
            as_of_date: The shared as-of ISO string.
            top_n: The ``--top-n`` knob.

        Returns:
            The next canned ``PredictionResult`` from ``results``,
            cycled.

        Raises:
            The configured ``error`` instance, when ``error`` is set.
            RuntimeError: If ``error`` is ``None`` and ``results`` is
                empty (a test-setup mistake, never swallowed by the
                export's ``PER_FIXTURE_ERRORS``).
        """
        self.calls.append(
            (team_a, team_b, best_of, map_pool, as_of_date, top_n)
        )
        if self.error is not None:
            raise self.error
        if not self.results:
            raise RuntimeError(
                "_StubPredictor.predict called with no canned results "
                "and no error"
            )
        result = self.results[self._index % len(self.results)]
        self._index += 1
        return result


class _RaisingCtor:
    """A ``Predictor`` stand-in whose construction always raises ValueError.

    Used to test that a construction-time failure (the D10/temperature
    staleness guard's shape) propagates out of ``main()`` and writes
    nothing, per D13's "fatal, before any fixture" policy.
    """

    def __init__(self, output_dir, version, *, n_samples, seed, ci_level,
                 bootstrap_models=None):
        """Raise a construction-time ValueError.

        Args:
            output_dir: Ignored.
            version: Ignored.
            n_samples: Ignored.
            seed: Ignored.
            ci_level: Ignored.
            bootstrap_models: Ignored.

        Returns:
            Nothing (always raises).

        Raises:
            ValueError: Always, with a staleness-guard-shaped message.
        """
        raise ValueError("staleness guard tripped (test stub)")


def _make_test_artifact():
    """Build a minimal, schema-valid artifact dict from synthetic stubs.

    Returns:
        A complete ``contract.Artifact``-shaped dict with one Bo3
        fixture, stub knobs, ``intervals_present=False`` and empty
        ``metrics``.

    Raises:
        Nothing (the synthetic inputs are well-formed).
    """
    fixture = ep.build_fixture(
        _make_result(), _make_spec("1"), _team_names()
    )
    return ep.build_artifact(
        [fixture],
        generated_at="2026-09-13T10:00:00",
        model_version="abc1234",
        dataset_version="v1",
        knobs={"n_samples": 30, "seed": 5, "ci_level": 0.9, "top_n": 10},
        intervals_present=False,
    )


def _raising_validate_artifact(artifact):
    """A ``validate_artifact`` stand-in that always raises ValidationError.

    Used to exercise D15's "validate before any byte touches disk"
    invariant: when this is monkeypatched over
    ``presentation.contract.validate_artifact``, ``write_artifact``
    must leave a pre-existing artifact byte-identical and create no
    temp file.

    Args:
        artifact: The artifact dict (ignored).

    Returns:
        Nothing (always raises).

    Raises:
        jsonschema.exceptions.ValidationError: Always, with "boom".
    """
    raise ValidationError("boom")


def _raising_replace(src, dst):
    """An ``os.replace`` stand-in that always raises OSError.

    Used to exercise D15's ``finally`` cleanup: when this is
    monkeypatched over ``os.replace``, ``write_artifact`` must remove
    the already-created temp file and leave a pre-existing artifact
    untouched.

    Args:
        src: The source path (ignored).
        dst: The destination path (ignored).

    Returns:
        Nothing (always raises).

    Raises:
        OSError: Always, with "replace failed".
    """
    raise OSError("replace failed")


def _real_v1_smoke_available():
    """Report whether the real v1 tables and the four fitted models exist.

    The Convention-B-style skip guard for the smoke test: the parquet
    half is delegated to ``tests._shared._real_v1_available`` (the
    five bare table names), and this file additionally requires the
    four fitted ``*_model.json`` artifacts the export consumes.

    Returns:
        A bool: ``True`` iff all five ``data/v1/*.parquet`` tables and
        ``ordinal_logit_model.json`` / ``temperature_scaling_model.json``
        / ``conditional_logit_ban_model.json`` /
        ``conditional_logit_pick_model.json`` exist.

    Raises:
        Nothing.
    """
    return _real_v1_tables_available() and all(
        Path(f"data/v1/{name}.json").exists()
        for name in (
            "ordinal_logit_model",
            "temperature_scaling_model",
            "conditional_logit_ban_model",
            "conditional_logit_pick_model",
        )
    )


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # All nine defaults come from predict.DEFAULT_* / the module
    # constants — never re-typed literals.
    args = ep.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.fixtures == ep.DEFAULT_FIXTURES_PATH
    assert args.as_of_date is None
    assert args.n_samples == predict.DEFAULT_N_SAMPLES
    assert args.seed == predict.DEFAULT_SEED
    assert args.ci_level == predict.DEFAULT_CI_LEVEL
    assert args.top_n == predict.DEFAULT_TOP_N
    assert args.max_failure_fraction == ep.DEFAULT_MAX_FAILURE_FRACTION


def test_parse_args_full_override():
    # Every flag overridable in one run.
    args = ep.parse_args(
        [
            "--version", "v2",
            "--output-dir", "/tmp/out",
            "--fixtures", "custom.json",
            "--as-of-date", "2026-09-13T10:00:00",
            "--n-samples", "5",
            "--seed", "99",
            "--ci-level", "0.8",
            "--top-n", "3",
            "--max-failure-fraction", "0.25",
        ]
    )
    assert args.version == "v2"
    assert args.output_dir == "/tmp/out"
    assert args.fixtures == "custom.json"
    assert args.as_of_date == "2026-09-13T10:00:00"
    assert args.n_samples == 5
    assert args.seed == 99
    assert args.ci_level == pytest.approx(0.8)
    assert args.top_n == 3
    assert args.max_failure_fraction == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Fixture reading (D2/D3/D9 + file-level failures)
# --------------------------------------------------------------------------


def test_read_fixtures_valid_two_records(tmp_path):
    # A valid two-record file parses to two FixtureSpecs with best_of_int
    # mapped through BEST_OF_MAP_COUNT.
    path = tmp_path / "fixtures.json"
    _write_json(
        path,
        {
            "fixtures": [
                {
                    "match_id": "1",
                    "event": "E1",
                    "scheduled_at": "2026-09-13T10:00:00",
                    "best_of": "Bo3",
                    "team_a_id": "397",
                    "team_b_id": "6392",
                },
                {
                    "match_id": "2",
                    "event": "E2",
                    "scheduled_at": "2026-09-14T10:00:00",
                    "best_of": "Bo5",
                    "team_a_id": "474",
                    "team_b_id": "2593",
                },
            ]
        },
    )
    as_of = datetime.fromisoformat("2026-08-01T00:00:00")
    specs, stats = ep.read_fixtures(path, as_of)
    assert [spec.match_id for spec in specs] == ["1", "2"]
    assert specs[0].best_of_int == 3
    assert specs[1].best_of_int == 5
    assert specs[0].best_of == "Bo3"
    assert specs[1].best_of == "Bo5"
    assert stats.n_records == 2
    assert stats.n_dropped_past == 0
    assert stats.n_invalid == 0
    assert stats.invalid_ids == ()


@pytest.mark.parametrize(
    ("bad_record", "expected_invalid_id"),
    [
        # missing required key (match_id absent)
        (
            {
                "event": "E",
                "scheduled_at": "2026-09-13T10:00:00",
                "best_of": "Bo3",
                "team_a_id": "397",
                "team_b_id": "6392",
            },
            "<unreadable>",
        ),
        # blank value (team_a_id is whitespace only)
        (
            {
                "match_id": "1",
                "event": "E",
                "scheduled_at": "2026-09-13T10:00:00",
                "best_of": "Bo3",
                "team_a_id": "   ",
                "team_b_id": "6392",
            },
            "1",
        ),
        # unknown best_of
        (
            {
                "match_id": "1",
                "event": "E",
                "scheduled_at": "2026-09-13T10:00:00",
                "best_of": "Bo7",
                "team_a_id": "397",
                "team_b_id": "6392",
            },
            "1",
        ),
        # unparseable scheduled_at
        (
            {
                "match_id": "1",
                "event": "E",
                "scheduled_at": "not-a-date",
                "best_of": "Bo3",
                "team_a_id": "397",
                "team_b_id": "6392",
            },
            "1",
        ),
        # timezone-aware scheduled_at
        (
            {
                "match_id": "1",
                "event": "E",
                "scheduled_at": "2026-09-13T10:00:00+00:00",
                "best_of": "Bo3",
                "team_a_id": "397",
                "team_b_id": "6392",
            },
            "1",
        ),
    ],
)
def test_read_fixtures_one_invalid_record_excluded(
    tmp_path, bad_record, expected_invalid_id
):
    # Each bad record becomes one counted invalid record (not a raise)
    # while the sibling survives.
    good = {
        "match_id": "2",
        "event": "E2",
        "scheduled_at": "2026-09-14T10:00:00",
        "best_of": "Bo5",
        "team_a_id": "474",
        "team_b_id": "2593",
    }
    path = tmp_path / "fixtures.json"
    _write_json(path, {"fixtures": [bad_record, good]})
    as_of = datetime.fromisoformat("2026-08-01T00:00:00")
    specs, stats = ep.read_fixtures(path, as_of)
    assert [spec.match_id for spec in specs] == ["2"]
    assert stats.n_records == 2
    assert stats.n_invalid == 1
    assert stats.n_dropped_past == 0
    assert stats.invalid_ids == (expected_invalid_id,)


def test_read_fixtures_ignores_unknown_extra_keys(tmp_path):
    # D2: unknown extra keys are ignored, not rejected.
    record = {
        "match_id": "1",
        "event": "E",
        "scheduled_at": "2026-09-13T10:00:00",
        "best_of": "Bo3",
        "team_a_id": "397",
        "team_b_id": "6392",
        "event_id": "x",
        "stage": "y",
        "vlr_url": "https://example.com/event/1",
    }
    path = tmp_path / "fixtures.json"
    _write_json(path, {"fixtures": [record]})
    specs, stats = ep.read_fixtures(
        path, datetime.fromisoformat("2026-08-01T00:00:00")
    )
    assert len(specs) == 1
    assert stats.n_invalid == 0
    assert specs[0].match_id == "1"
    assert not hasattr(specs[0], "event_id")


def test_read_fixtures_drops_past_and_keeps_exact_as_of(tmp_path):
    # D9: scheduled_at < as_of is dropped (strictly); exactly at as_of
    # is kept.
    as_of = datetime.fromisoformat("2026-09-13T10:00:00")
    path = tmp_path / "fixtures.json"
    _write_json(
        path,
        {
            "fixtures": [
                {
                    "match_id": "past",
                    "event": "E",
                    "scheduled_at": "2026-09-12T10:00:00",
                    "best_of": "Bo3",
                    "team_a_id": "397",
                    "team_b_id": "6392",
                },
                {
                    "match_id": "exact",
                    "event": "E",
                    "scheduled_at": "2026-09-13T10:00:00",
                    "best_of": "Bo3",
                    "team_a_id": "397",
                    "team_b_id": "6392",
                },
            ]
        },
    )
    specs, stats = ep.read_fixtures(path, as_of)
    assert [spec.match_id for spec in specs] == ["exact"]
    assert stats.n_dropped_past == 1
    assert stats.n_invalid == 0


def test_read_fixtures_missing_file_raises(tmp_path):
    # FileNotFoundError propagates unchanged (the "create the file
    # first" signal), never wrapped.
    with pytest.raises(FileNotFoundError):
        ep.read_fixtures(
            tmp_path / "nope.json",
            datetime.fromisoformat("2026-08-01T00:00:00"),
        )


def test_read_fixtures_malformed_json_raises(tmp_path):
    path = tmp_path / "fixtures.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ep.FixtureFileError):
        ep.read_fixtures(path, datetime.fromisoformat("2026-08-01T00:00:00"))


def test_read_fixtures_top_level_list_raises(tmp_path):
    path = tmp_path / "fixtures.json"
    _write_json(path, [{"match_id": "1"}])
    with pytest.raises(ep.FixtureFileError):
        ep.read_fixtures(path, datetime.fromisoformat("2026-08-01T00:00:00"))


def test_read_fixtures_missing_fixtures_key_raises(tmp_path):
    path = tmp_path / "fixtures.json"
    _write_json(path, {"other": []})
    with pytest.raises(ep.FixtureFileError, match="fixtures"):
        ep.read_fixtures(path, datetime.fromisoformat("2026-08-01T00:00:00"))


def test_read_fixtures_fixtures_not_a_list_raises(tmp_path):
    path = tmp_path / "fixtures.json"
    _write_json(path, {"fixtures": {"match_id": "1"}})
    with pytest.raises(ep.FixtureFileError, match="list"):
        ep.read_fixtures(path, datetime.fromisoformat("2026-08-01T00:00:00"))


# --------------------------------------------------------------------------
# Assembly (D4/D5)
# --------------------------------------------------------------------------

_FIXTURE_KEYS = {
    "match_id",
    "event",
    "scheduled_at",
    "best_of",
    "best_of_int",
    "bo5_unvalidated",
    "team_a",
    "team_b",
    "outcome_order",
    "scoreline_labels",
    "overall",
    "top_vetos",
    "coverage_mass",
    "map_leverage",
    "narrative",
}


def test_build_fixture_exact_fifteen_keys():
    # Exactly the fifteen schema-required keys; identity fields verbatim
    # from the spec; narrative ""; map_leverage [] for an empty listing.
    result = _make_result()
    spec = _make_spec("1")
    fixture = ep.build_fixture(result, spec, _team_names())
    assert set(fixture) == _FIXTURE_KEYS
    assert fixture["narrative"] == ""
    assert fixture["bo5_unvalidated"] is False
    assert fixture["best_of_int"] == 3
    assert fixture["best_of_int"] == len(fixture["outcome_order"]) - 1
    assert fixture["match_id"] == "1"
    assert fixture["event"] == "Example Event"
    assert fixture["scheduled_at"] == "2026-09-13T10:00:00"
    assert fixture["best_of"] == "Bo3"
    assert fixture["map_leverage"] == []


@pytest.mark.parametrize(
    ("best_of", "best_of_int", "flag"),
    [("Bo1", 1, False), ("Bo3", 3, False), ("Bo5", 5, True)],
)
def test_build_fixture_bo5_flag(best_of, best_of_int, flag):
    # D5: bo5_unvalidated is best_of_int == 5, never a corpus count.
    series = _make_series([0.1] * (best_of_int + 1), best_of_int)
    result = _make_result(series=series)
    spec = _make_spec("1", best_of=best_of, best_of_int=best_of_int)
    fixture = ep.build_fixture(result, spec, _team_names())
    assert fixture["bo5_unvalidated"] is flag
    assert fixture["best_of_int"] == best_of_int
    assert len(fixture["outcome_order"]) == best_of_int + 1


# --------------------------------------------------------------------------
# Schema round-trip
# --------------------------------------------------------------------------


def test_full_artifact_validates_and_round_trips_json():
    # A fully assembled artifact (two fixtures, from stubs) validates
    # and survives json.dumps -> json.loads unchanged (JSON-native).
    entry = _make_ranked_entry(
        _make_actions("Ascent"), _make_series((0.2, 0.3, 0.3, 0.2), 3)
    )
    result = _make_result(top_vetos=(entry,))
    fixture1 = ep.build_fixture(result, _make_spec("1"), _team_names())
    fixture2 = ep.build_fixture(result, _make_spec("2"), _team_names())
    knobs = {"n_samples": 30, "seed": 5, "ci_level": 0.9, "top_n": 10}
    artifact = ep.build_artifact(
        [fixture1, fixture2],
        generated_at="2026-09-13T10:00:00",
        model_version="abc1234",
        dataset_version="v1",
        knobs=knobs,
        intervals_present=False,
    )
    assert contract.validate_artifact(artifact) is None
    assert json.loads(json.dumps(artifact)) == artifact
    assert artifact["metrics"] == {}
    assert artifact["knobs"] == knobs


# --------------------------------------------------------------------------
# End-to-end main() with stubs
# --------------------------------------------------------------------------


def _run_export_main(
    tmp_path, monkeypatch, records, argv, stub, matches_df=None
):
    """Run :func:`ep.main` with stubbed predictor/table loading.

    Writes ``records`` to ``tmp_path/fixtures.json``, stubs
    ``evaluate.load_matches_table`` to return ``matches_df`` (or the
    default synthetic frame) and ``predict.Predictor`` to ``stub``, then
    invokes ``ep.main`` with ``--output-dir tmp_path --version v1
    --fixtures <the written file>`` plus the caller's extra ``argv``.

    Args:
        tmp_path: The pytest tmp dir.
        monkeypatch: The pytest monkeypatch fixture.
        records: The fixtures list to write.
        argv: Extra CLI arguments (after the shared three).
        stub: The ``_StubPredictor`` instance to stand in for
            ``drivers.predict.Predictor``.
        matches_df: An optional custom matches DataFrame; defaults to
            :func:`_stub_matches_df`.

    Returns:
        The ``stub`` instance (so the caller can assert recorded
        construction/predict calls).

    Raises:
        Anything :func:`ep.main` raises (the caller asserts on it).
    """
    fixtures_path = tmp_path / "fixtures.json"
    _write_json(fixtures_path, {"fixtures": records})
    if matches_df is None:
        matches_df = _stub_matches_df()
    monkeypatch.setattr(
        ep.evaluate,
        "load_matches_table",
        partial(_load_matches_table_stub, matches_df),
    )
    monkeypatch.setattr(ep.predict, "Predictor", stub)
    ep.main(
        [
            "--output-dir", str(tmp_path),
            "--version", "v1",
            "--fixtures", str(fixtures_path),
        ]
        + argv
    )
    return stub


def _good_record(match_id="1"):
    """Return one valid, near-future fixture record.

    Args:
        match_id: The match id for the record.

    Returns:
        A six-key dict with teams A/B and a Bo3 series.

    Raises:
        Nothing.
    """
    return {
        "match_id": match_id,
        "event": "Example Event",
        "scheduled_at": "2026-09-13T10:00:00",
        "best_of": "Bo3",
        "team_a_id": "A",
        "team_b_id": "B",
    }


def test_main_end_to_end_writes_and_records_calls(tmp_path, monkeypatch):
    # One Predictor construction (with bootstrap_models never passed —
    # D10), one predict per kept fixture with map_pool=None / the
    # shared as-of / the --top-n value; artifact validates and carries
    # generated_at == --as-of-date.
    stub = _StubPredictor(results=[_make_result(), _make_result()])
    _run_export_main(
        tmp_path,
        monkeypatch,
        [_good_record("1"), _good_record("2")],
        [
            "--as-of-date", "2026-09-01T00:00:00",
            "--n-samples", "5",
            "--seed", "99",
            "--ci-level", "0.8",
            "--top-n", "7",
        ],
        stub,
    )
    assert len(stub.construction_kwargs) == 1
    ctor = stub.construction_kwargs[0]
    assert ctor["version"] == "v1"
    assert ctor["n_samples"] == 5
    assert ctor["seed"] == 99
    assert ctor["ci_level"] == pytest.approx(0.8)
    assert "bootstrap_models" not in ctor
    assert len(stub.calls) == 2
    for team_a, team_b, best_of, map_pool, as_of_date, top_n in stub.calls:
        assert (team_a, team_b, best_of) == ("A", "B", "Bo3")
        assert map_pool is None
        assert as_of_date == "2026-09-01T00:00:00"
        assert top_n == 7

    artifact_path = tmp_path / "v1" / "predictions.json"
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    contract.validate_artifact(artifact)
    assert artifact["generated_at"] == "2026-09-01T00:00:00"
    assert artifact["dataset_version"] == "v1"
    assert len(artifact["fixtures"]) == 2
    assert artifact["knobs"] == {
        "n_samples": 5,
        "seed": 99,
        "ci_level": 0.8,
        "top_n": 7,
    }


# --------------------------------------------------------------------------
# intervals_present (D12)
# --------------------------------------------------------------------------


def test_intervals_present_true_when_any_non_null():
    # A non-null interval_low on any fixture makes the OR True.
    pm = _make_per_map(
        (0.5, 0.1, 0.1, 0.3),
        interval_low=(0.3, 0.0, 0.0, 0.2),
        interval_high=(0.7, 0.3, 0.3, 0.5),
    )
    stub = _StubPredictor(results=[_make_result(per_map=(pm,))])
    _fixtures, failures, present = ep.export_fixtures(
        stub,
        [_make_spec("1")],
        _team_names(),
        as_of_iso="2026-09-01T00:00:00",
        top_n=10,
    )
    assert failures == 0
    assert present is True


def test_intervals_present_false_when_all_null():
    # Every stub interval null -> False.
    pm = _make_per_map((0.5, 0.1, 0.1, 0.3))
    stub = _StubPredictor(results=[_make_result(per_map=(pm,))])
    _fixtures, failures, present = ep.export_fixtures(
        stub,
        [_make_spec("1")],
        _team_names(),
        as_of_iso="2026-09-01T00:00:00",
        top_n=10,
    )
    assert failures == 0
    assert present is False


def test_intervals_present_false_for_empty_list():
    # Zero exported fixtures -> False.
    stub = _StubPredictor(results=[_make_result()])
    fixtures, failures, present = ep.export_fixtures(
        stub,
        [],
        _team_names(),
        as_of_iso="2026-09-01T00:00:00",
        top_n=10,
    )
    assert fixtures == []
    assert failures == 0
    assert present is False


# --------------------------------------------------------------------------
# Failure policy (D13/D14)
# --------------------------------------------------------------------------


def test_export_fixtures_excludes_unknown_team_and_logs(caplog):
    # D7: a team missing from the name map is excluded with an ERROR
    # log naming the id, the sibling is exported.
    stub = _StubPredictor(results=[_make_result()])
    specs = [
        _make_spec("1"),
        _make_spec("2", team_b_id="MISSING"),
    ]
    with caplog.at_level(logging.ERROR):
        fixtures, failures, _present = ep.export_fixtures(
            stub,
            specs,
            _team_names(),
            as_of_iso="2026-09-01T00:00:00",
            top_n=10,
        )
    assert [fixture["match_id"] for fixture in fixtures] == ["1"]
    assert failures == 1
    assert len(stub.calls) == 1
    assert any("MISSING" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "error_type",
    [ValueError, KeyError, ep.ConfigError],
    ids=["ValueError", "KeyError", "ConfigError"],
)
def test_export_fixtures_predict_per_fixture_error_excluded(error_type):
    # Each PER_FIXTURE_ERRORS member out of predict is excluded and
    # counted (D13).
    stub = _StubPredictor(
        results=[_make_result()], error=error_type("boom")
    )
    fixtures, failures, _present = ep.export_fixtures(
        stub,
        [_make_spec("1")],
        _team_names(),
        as_of_iso="2026-09-01T00:00:00",
        top_n=10,
    )
    assert fixtures == []
    assert failures == 1


def test_export_fixtures_typeerror_propagates():
    # TypeError is never swallowed (D13).
    stub = _StubPredictor(results=[_make_result()], error=TypeError("boom"))
    with pytest.raises(TypeError, match="boom"):
        ep.export_fixtures(
            stub,
            [_make_spec("1")],
            _team_names(),
            as_of_iso="2026-09-01T00:00:00",
            top_n=10,
        )


def test_main_abort_writes_nothing_and_keeps_existing(tmp_path, monkeypatch):
    # D14: with --max-failure-fraction 0.0 and one of two fixtures
    # failing, ExportAbortedError is raised and nothing is written; a
    # pre-existing artifact's bytes stay unchanged.
    stub = _StubPredictor(results=[_make_result(), _make_result()])
    records = [_good_record("1"), _good_record("2")]
    # Make the second record fail pre-flight: team_b_id not in the name
    # map.
    records[1]["team_b_id"] = "MISSING"
    argv = [
        "--as-of-date", "2026-09-01T00:00:00",
        "--max-failure-fraction", "0.0",
    ]
    with pytest.raises(ep.ExportAbortedError):
        _run_export_main(tmp_path, monkeypatch, records, argv, stub)
    artifact_path = tmp_path / "v1" / "predictions.json"
    # The first run raised inside _run_export_main, so nothing was
    # written.
    assert not artifact_path.exists()

    # Pre-existing artifact is preserved on a second aborted run.
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"previous bytes\n")
    with pytest.raises(ep.ExportAbortedError):
        _run_export_main(tmp_path, monkeypatch, records, argv, stub)
    assert artifact_path.read_bytes() == b"previous bytes\n"
    assert list((tmp_path / "v1").glob("*.tmp")) == []


def test_main_invalid_records_count_toward_failure_fraction(
    tmp_path, monkeypatch
):
    # D14: invalid records (D2/D3 violations) count toward the failure
    # fraction. One valid + two invalid records = 3 attempted, 2
    # failed, 2/3 > 0.5, so the run aborts and writes nothing.
    stub = _StubPredictor(results=[_make_result()])
    records = [
        _good_record("1"),
        # Missing team_a_id -> invalid (D2).
        {
            "match_id": "2",
            "event": "E",
            "scheduled_at": "2026-09-13T10:00:00",
            "best_of": "Bo3",
            "team_b_id": "B",
        },
        # Unknown best_of -> invalid (D3).
        {
            "match_id": "3",
            "event": "E",
            "scheduled_at": "2026-09-13T10:00:00",
            "best_of": "Bo7",
            "team_a_id": "A",
            "team_b_id": "B",
        },
    ]
    argv = [
        "--as-of-date", "2026-09-01T00:00:00",
        "--max-failure-fraction", "0.5",
    ]
    with pytest.raises(ep.ExportAbortedError):
        _run_export_main(tmp_path, monkeypatch, records, argv, stub)
    assert not (tmp_path / "v1" / "predictions.json").exists()


def test_main_all_past_fixtures_writes_empty_artifact(tmp_path, monkeypatch):
    # D14/D9: an all-past list writes a valid artifacts with fixtures
    # == [] and does not abort.
    stub = _StubPredictor(results=[_make_result()])
    record = _good_record("past")
    record["scheduled_at"] = "2026-09-01T10:00:00"
    _run_export_main(
        tmp_path,
        monkeypatch,
        [record],
        ["--as-of-date", "2026-09-13T00:00:00"],
        stub,
    )
    assert len(stub.calls) == 0
    artifact_path = tmp_path / "v1" / "predictions.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    contract.validate_artifact(artifact)
    assert artifact["fixtures"] == []


def test_main_constructor_valueerror_propagates(tmp_path, monkeypatch):
    # D13's fatal path: a Predictor constructor ValueError propagates
    # and writes nothing.
    monkeypatch.setattr(ep.evaluate, "load_matches_table",
                        partial(_load_matches_table_stub, _stub_matches_df()))
    monkeypatch.setattr(ep.predict, "Predictor", _RaisingCtor)
    fixtures_path = tmp_path / "fixtures.json"
    _write_json(fixtures_path, {"fixtures": [_good_record("1")]})
    with pytest.raises(ValueError, match="staleness"):
        ep.main(
            [
                "--output-dir", str(tmp_path),
                "--version", "v1",
                "--fixtures", str(fixtures_path),
                "--as-of-date", "2026-09-01T00:00:00",
            ]
        )
    assert not (tmp_path / "v1" / "predictions.json").exists()


def test_main_invalid_failure_fraction_raises_before_io(tmp_path):
    # D14: --max-failure-fraction outside [0, 1] raises before any I/O
    # (no fixtures file needed).
    with pytest.raises(ValueError, match="max-failure-fraction"):
        ep.main(
            [
                "--output-dir", str(tmp_path),
                "--version", "v1",
                "--max-failure-fraction", "1.5",
            ]
        )


def test_main_rejects_tz_aware_as_of_date(tmp_path):
    # A5/D8: a timezone-aware --as-of-date raises before any I/O.
    with pytest.raises(ValueError, match="timezone-aware"):
        ep.main(
            [
                "--output-dir", str(tmp_path),
                "--version", "v1",
                "--as-of-date", "2026-09-13T10:00:00+00:00",
            ]
        )


# --------------------------------------------------------------------------
# Atomic write (D15)
# --------------------------------------------------------------------------


def test_write_artifact_leaves_no_temp_files(tmp_path):
    # After a successful run no *.tmp file remains in the output dir.
    path = tmp_path / "v1" / "predictions.json"
    ep.write_artifact(_make_test_artifact(), path)
    assert path.exists()
    assert list((tmp_path / "v1").glob("*.tmp")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_artifact_validate_raise_keeps_existing(tmp_path, monkeypatch):
    # When validate_artifact raises, the pre-existing artifact is
    # byte-identical and no temp file remains (D15's invariant).
    path = tmp_path / "v1" / "predictions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"previous bytes\n")

    monkeypatch.setattr(
        ep.contract, "validate_artifact", _raising_validate_artifact
    )
    with pytest.raises(ValidationError, match="boom"):
        ep.write_artifact(_make_test_artifact(), path)
    assert path.read_bytes() == b"previous bytes\n"
    assert list((tmp_path / "v1").glob("*.tmp")) == []


def test_write_artifact_replace_raise_cleans_temp(tmp_path, monkeypatch):
    # When the final os.replace raises, the finally removes the temp
    # file and the pre-existing artifact is untouched.
    path = tmp_path / "v1" / "predictions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"previous bytes\n")

    monkeypatch.setattr(ep.os, "replace", _raising_replace)
    with pytest.raises(OSError, match="replace failed"):
        ep.write_artifact(_make_test_artifact(), path)
    assert path.read_bytes() == b"previous bytes\n"
    assert list((tmp_path / "v1").glob("*.tmp")) == []


# --------------------------------------------------------------------------
# Provenance (D17/D18)
# --------------------------------------------------------------------------


def test_write_and_read_model_provenance_round_trip(tmp_path):
    # write_model_provenance writes exactly the three keys and
    # read_model_version returns the sha.
    path = model_provenance.write_model_provenance(
        tmp_path,
        "v1",
        git_sha="abc1234",
        trained_at="2026-09-13T10:00:00",
        drivers=["train_ordinal_logit.py", "train_temperature_scaling.py"],
    )
    assert path == tmp_path / "v1" / "model_provenance.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"git_sha", "trained_at", "drivers"}
    assert data["git_sha"] == "abc1234"
    assert data["trained_at"] == "2026-09-13T10:00:00"
    assert data["drivers"] == [
        "train_ordinal_logit.py",
        "train_temperature_scaling.py",
    ]
    assert model_provenance.read_model_version(tmp_path, "v1") == "abc1234"


def test_write_model_provenance_rejects_bad_arguments(tmp_path):
    # git_sha must be non-empty; drivers a non-empty sequence of
    # non-empty strings.
    with pytest.raises(ValueError, match="git_sha"):
        model_provenance.write_model_provenance(
            tmp_path, "v1", git_sha="", trained_at="2026-09-13T10:00:00",
            drivers=["a.py"],
        )
    with pytest.raises(ValueError, match="drivers"):
        model_provenance.write_model_provenance(
            tmp_path, "v1", git_sha="abc", trained_at="2026-09-13T10:00:00",
            drivers=[],
        )
    with pytest.raises(ValueError, match="driver"):
        model_provenance.write_model_provenance(
            tmp_path, "v1", git_sha="abc", trained_at="2026-09-13T10:00:00",
            drivers=["a.py", 5],
        )


def test_read_model_version_missing_returns_unstamped(tmp_path, caplog):
    # A missing sidecar soft-paths to "unstamped" and logs at WARNING.
    with caplog.at_level(logging.WARNING):
        version = model_provenance.read_model_version(tmp_path, "v1")
    assert version == model_provenance.UNSTAMPED_MODEL_VERSION
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert any("python -m drivers.model_provenance" in record.message
               for record in caplog.records)


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"trained_at": "x", "drivers": ["a.py"]}),
        json.dumps({"git_sha": "   ", "trained_at": "x", "drivers": ["a.py"]}),
        json.dumps({"git_sha": 123, "trained_at": "x", "drivers": ["a.py"]}),
    ],
)
def test_read_model_version_malformed_sidecar_soft(tmp_path, caplog, content):
    # Every unusable sidecar takes the same soft path without raising.
    path = tmp_path / "v1" / "model_provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert model_provenance.read_model_version(tmp_path, "v1") == (
            model_provenance.UNSTAMPED_MODEL_VERSION
        )
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_export_main_model_version_unstamped_and_sha(tmp_path, monkeypatch):
    # main() puts "unstamped" into model_version when no sidecar exists,
    # and the sha when one does.
    stub = _StubPredictor(results=[_make_result()])
    records = [_good_record("1")]
    argv = ["--as-of-date", "2026-09-01T00:00:00"]

    _run_export_main(tmp_path, monkeypatch, records, argv, stub)
    artifact_path = tmp_path / "v1" / "predictions.json"
    unstamped = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert (
        unstamped["model_version"] == model_provenance.UNSTAMPED_MODEL_VERSION
    )

    model_provenance.write_model_provenance(
        tmp_path, "v1", git_sha="deadbeef",
        trained_at="2026-09-13T10:00:00", drivers=["a.py"],
    )
    stub2 = _StubPredictor(results=[_make_result()])
    _run_export_main(tmp_path, monkeypatch, records, argv, stub2)
    stamped = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert stamped["model_version"] == "deadbeef"


def test_model_provenance_parse_args_overrides():
    # The stamping CLI honours --git-sha/--trained-at/--drivers.
    args = model_provenance.parse_args(
        [
            "--version", "v2",
            "--output-dir", "/tmp/x",
            "--git-sha", "abc",
            "--trained-at", "2026-09-13T10:00:00",
            "--drivers", "a.py,b.py",
        ]
    )
    assert args.version == "v2"
    assert args.output_dir == "/tmp/x"
    assert args.git_sha == "abc"
    assert args.trained_at == "2026-09-13T10:00:00"
    assert args.drivers == "a.py,b.py"


def test_model_provenance_main_honours_overrides(tmp_path):
    # The stamping CLI writes the overridden three values.
    rc = model_provenance.main(
        [
            "--output-dir", str(tmp_path),
            "--version", "v1",
            "--git-sha", "deadbeef",
            "--trained-at", "2026-09-13T10:00:00",
            "--drivers", "a.py,b.py",
        ]
    )
    assert rc == 0
    data = json.loads(
        (tmp_path / "v1" / "model_provenance.json").read_text(encoding="utf-8")
    )
    assert data["git_sha"] == "deadbeef"
    assert data["trained_at"] == "2026-09-13T10:00:00"
    assert data["drivers"] == ["a.py", "b.py"]


def test_model_provenance_main_rejects_tz_aware_trained_at(tmp_path):
    # A timezone-aware --trained-at raises before writing.
    with pytest.raises(ValueError, match="timezone-aware"):
        model_provenance.main(
            [
                "--output-dir", str(tmp_path),
                "--version", "v1",
                "--git-sha", "abc",
                "--trained-at", "2026-09-13T10:00:00+00:00",
            ]
        )
    assert not (tmp_path / "v1" / "model_provenance.json").exists()


# --------------------------------------------------------------------------
# Real-v1 smoke test (slow + skip-guarded)
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_smoke_available(),
    reason="materialised v1 tables + the four fitted *_model.json "
    "artifacts not present (run materialize.py and the training "
    "drivers first)",
)
def test_real_v1_export_smoke(tmp_path):
    # The only test binding the schema to the shipped dataclasses: copy
    # the three tables, the four model artifacts and (when present) the
    # replicates + sidecar into tmp_path/v1, then run the real export
    # with reduced knobs. The team pair is DELIBERATELY repeated across
    # the two records: predict() resolves event_stage by an exact
    # (team_a, team_b, as_of) matches-row lookup (finding 1), so only
    # fixtures matching the as-of row can be predicted — a second,
    # different pair would fail and abort the run. The assertions below
    # therefore pin that the two exported records differ only in their
    # identity keys (match_id/event/scheduled_at).
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    for table in ("matches", "maps", "player_map_stats"):
        shutil.copy2(
            Path(f"data/v1/{table}.parquet"), v1_dir / f"{table}.parquet"
        )
    for model in (
        "ordinal_logit_model",
        "temperature_scaling_model",
        "conditional_logit_ban_model",
        "conditional_logit_pick_model",
    ):
        shutil.copy2(Path(f"data/v1/{model}.json"), v1_dir / f"{model}.json")
    for optional in ("ordinal_bootstrap_replicates.json",
                     "model_provenance.json"):
        source = Path(f"data/v1/{optional}")
        if source.exists():
            shutil.copy2(source, v1_dir / optional)

    matches = pd.read_parquet(v1_dir / "matches.parquet")
    completed = matches[matches["status"] == "completed"].sort_values("date")
    row = completed.iloc[-1]
    team_a = str(row["team1_id"])
    team_b = str(row["team2_id"])
    as_of_date = str(row["date"])
    records = [
        {
            "match_id": "999901",
            "event": "Smoke Event 1",
            "scheduled_at": "2026-09-13T10:00:00",
            "best_of": "Bo3",
            "team_a_id": team_a,
            "team_b_id": team_b,
        },
        {
            "match_id": "999902",
            "event": "Smoke Event 2",
            "scheduled_at": "2026-09-14T10:00:00",
            "best_of": "Bo3",
            "team_a_id": team_a,
            "team_b_id": team_b,
        },
    ]
    fixtures_path = tmp_path / "fixtures.json"
    _write_json(fixtures_path, {"fixtures": records})

    rc = ep.main(
        [
            "--output-dir", str(tmp_path),
            "--version", "v1",
            "--fixtures", str(fixtures_path),
            "--as-of-date", as_of_date,
            "--top-n", "1",
            "--n-samples", "2",
        ]
    )
    assert rc == 0
    artifact_path = tmp_path / "v1" / "predictions.json"
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    contract.validate_artifact(artifact)
    assert json.loads(json.dumps(artifact)) == artifact
    fixtures = artifact["fixtures"]
    assert len(fixtures) == 2
    first, second = fixtures
    assert first["match_id"] == "999901"
    assert second["match_id"] == "999902"
    assert first["event"] == "Smoke Event 1"
    assert second["event"] == "Smoke Event 2"
    assert first["scheduled_at"] == "2026-09-13T10:00:00"
    assert second["scheduled_at"] == "2026-09-14T10:00:00"
    # The two records are otherwise identical — the same team pair was
    # predicted twice (see the comment above; finding 1).
    identity_keys = {"match_id", "event", "scheduled_at"}
    assert {k: v for k, v in first.items() if k not in identity_keys} == {
        k: v for k, v in second.items() if k not in identity_keys
    }
    assert artifact["model_version"]
