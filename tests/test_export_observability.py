"""Tests for the P6 export observability diagnostics.

Owns exactly P6's three diagnostics: :func:`count_null_interval_maps`
(D4's null-interval tally), :func:`coverage_diagnostic` (D5's §8
reconciliation gap), and the driver's timing log lines (D3). The
null-interval verdict and the extended summary line are exercised
through :func:`drivers.export_predictions.main` with a stubbed
``Predictor``.

**D11 rule.** No test asserts a wall-clock duration, a threshold, or an
ordering of durations. Timing tests assert only that the log lines
exist, are emitted the right number of times, and carry parseable
non-negative numbers. The measured run is a one-off BUILD activity, not
a test.

The synthetic builders below are written locally (this file does not
import private helpers from ``tests/test_export_predictions.py``, per
the repo's no-cross-test-module-imports convention).
"""

import json
import logging
import re
from functools import partial

import pandas as pd
import pytest

from drivers import export_predictions as ep
from drivers.predict import (
    PerMapPrediction,
    PredictionResult,
    RankedVetoPrediction,
    SeriesPrediction,
    VetoSensitivity,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from utils import series_paths

# --------------------------------------------------------------------------
# Dict builders for the two pure diagnostic functions.
# --------------------------------------------------------------------------


def _per_map(interval):
    """Build one synthetic ``PerMap`` dict for the null-interval counter.

    Only the two interval keys the counter reads are populated — the
    other wire fields are irrelevant to that function and are omitted.

    Args:
        interval: When truthy, both ``interval_low`` and
            ``interval_high`` are populated 4-tuples; when falsy, both
            are ``None`` (the §4.3 soft-missing case).

    Returns:
        A dict with exactly the ``interval_low`` and ``interval_high``
        keys, both populated or both ``None``.

    Raises:
        Nothing.
    """
    if interval:
        return {
            "interval_low": (0.3, 0.0, 0.0, 0.2),
            "interval_high": (0.7, 0.3, 0.3, 0.5),
        }
    return {"interval_low": None, "interval_high": None}


def _fixture(entries, overall, *, match_id="1"):
    """Build a synthetic ``Fixture`` dict for :func:`coverage_diagnostic`.

    Populates only the keys the coverage diagnostic reads — ``top_vetos``
    (each entry carrying ``veto_probability`` and ``p_a_wins_series``),
    ``overall.p_a_wins_series``, ``match_id`` — plus ``coverage_mass``
    (the summed veto probability, which D5 says equals the diagnostic's
    recomputed ``mass`` by construction).

    Args:
        entries: The ``top_vetos`` listing as a list of dicts, each
            with numeric ``veto_probability`` and ``p_a_wins_series``
            keys.
        overall: The fixture's overall ``p_a_wins_series`` float.
        match_id: The fixture's match id (keyword-only, default
            ``"1"``).

    Returns:
        A ``Fixture``-shaped dict carrying the fields above (its
        ``overall`` also has an empty ``per_map`` list).

    Raises:
        TypeError: If an entry's ``veto_probability`` is not numeric —
            propagated unchanged from ``sum``.
    """
    return {
        "match_id": match_id,
        "top_vetos": list(entries),
        "overall": {"p_a_wins_series": overall, "per_map": []},
        "coverage_mass": sum(entry["veto_probability"] for entry in entries),
    }


# --------------------------------------------------------------------------
# Dataclass builders for the main()-level tests (mirrors
# tests/test_export_predictions.py, copied rather than imported).
# --------------------------------------------------------------------------


def _make_series(probabilities, best_of):
    """Build a synthetic :class:`SeriesPrediction`.

    Args:
        probabilities: The ``best_of + 1`` scoreline probabilities.
        best_of: The parsed map count (``1``/``3``/``5``), used to pick
            the canonical ``outcome_order``.

    Returns:
        A frozen :class:`SeriesPrediction` wrapping the probability
        tuple, the canonical outcome order, and ``best_of``.

    Raises:
        ValueError: If ``best_of`` is not a valid odd map count —
            propagated unchanged from
            :func:`utils.series_paths.series_outcome_order`.
    """
    return SeriesPrediction(
        probabilities=tuple(probabilities),
        outcome_order=tuple(series_paths.series_outcome_order(best_of)),
        best_of=best_of,
    )


def _make_per_map_pred(probabilities, *, interval_low=None,
                       interval_high=None):
    """Build a synthetic :class:`PerMapPrediction`.

    Args:
        probabilities: The four map-outcome probabilities.
        interval_low: The four lower band endpoints, or ``None``.
        interval_high: The four upper band endpoints, or ``None``
            alongside ``interval_low``.

    Returns:
        A frozen :class:`PerMapPrediction` carrying the given fields.

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


def _make_actions(first_map="Ascent"):
    """Build a synthetic 7-step Bo3 veto sequence.

    Args:
        first_map: The map side A bans at step 0.

    Returns:
        A 7-tuple of :class:`SimulatedVetoAction` records in the Bo3
        shape ``ban, ban, pick, pick, ban, ban, decider`` with team ids
        ``"A"``/``"B"`` by step parity and ``team=None`` on the decider.

    Raises:
        Nothing.
    """
    sequence = ("ban", "ban", "pick", "pick", "ban", "ban", "decider")
    maps = [first_map, "Haven", "Lotus", "Split", "Summit", "Abyss", "Sunset"]
    actions = []
    for index, action in enumerate(sequence):
        team = None if action == "decider" else (
            "A" if index % 2 == 0 else "B"
        )
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


def _make_ranked_entry(veto_probability):
    """Build a synthetic :class:`RankedVetoPrediction` entry.

    Args:
        veto_probability: The exact joint probability assigned to the
            entry.

    Returns:
        A frozen :class:`RankedVetoPrediction` whose ``result`` carries
        a default Bo3 series and an empty ``per_map``.

    Raises:
        Nothing.
    """
    inner = PredictionResult(
        predicted_veto=_make_actions(),
        per_map=(),
        series=_make_series((0.2, 0.3, 0.3, 0.2), 3),
        veto_sensitivity=None,
    )
    return RankedVetoPrediction(
        veto_probability=veto_probability, result=inner
    )


def _make_result(*, per_map=(), top_vetos=()):
    """Build a synthetic top-level :class:`PredictionResult`.

    Args:
        per_map: The greedy veto's played-map records (defaults to
            empty).
        top_vetos: The ranked listing entries (defaults to empty).

    Returns:
        A frozen :class:`PredictionResult` with a default Bo3 series,
        a real :class:`VetoSensitivity`, and the given ``per_map`` /
        ``top_vetos``.

    Raises:
        Nothing.
    """
    return PredictionResult(
        predicted_veto=_make_actions(),
        per_map=tuple(per_map),
        series=_make_series((0.4, 0.3, 0.2, 0.1), 3),
        veto_sensitivity=_make_veto_sensitivity(),
        top_vetos=tuple(top_vetos),
    )


def _stub_matches_df():
    """Build the synthetic matches table backing the name-map stub.

    Returns:
        A ``pandas.DataFrame`` mapping ``"A"`` → ``"Team A"`` and
        ``"B"`` → ``"Team B"``.

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
    """A stand-in for ``drivers.predict.Predictor`` used by ``main()``.

    When called (as the monkeypatched ``drivers.predict.Predictor``) it
    returns itself; its ``predict`` method returns the next canned
    result from ``results``, cycled.

    Attributes:
        results: The canned ``PredictionResult`` list to cycle through.
    """

    def __init__(self, results=()):
        """Initialize the stub with its canned results.

        Args:
            results: An iterable of ``PredictionResult`` objects to
                return from successive ``predict`` calls, cycled.

        Returns:
            Nothing.

        Raises:
            Nothing.
        """
        self.results = list(results)
        self._index = 0

    def __call__(self, output_dir, version, *, n_samples, seed, ci_level,
                 **kwargs):
        """Record nothing and return this instance.

        Args:
            output_dir: The ``--output-dir`` value (a ``Path``).
            version: The ``--version`` value.
            n_samples: The ``--n-samples`` knob.
            seed: The ``--seed`` knob.
            ci_level: The ``--ci-level`` knob.
            **kwargs: Any additional construction keywords, accepted
                and ignored.

        Returns:
            ``self``.

        Raises:
            Nothing.
        """
        return self

    def predict(self, team_a, team_b, best_of, map_pool, as_of_date, *,
                top_n):
        """Return the next canned result.

        Args:
            team_a: The queried team A id.
            team_b: The queried team B id.
            best_of: The ``"Bo<N>"`` string.
            map_pool: The map pool (always ``None`` from the export).
            as_of_date: The shared as-of ISO string.
            top_n: The ``--top-n`` knob.

        Returns:
            The next canned ``PredictionResult``, cycled.

        Raises:
            RuntimeError: If ``results`` is empty (a test-setup
                mistake, never swallowed by the export's
                ``PER_FIXTURE_ERRORS``).
        """
        if not self.results:
            raise RuntimeError(
                "_StubPredictor.predict called with no canned results"
            )
        result = self.results[self._index % len(self.results)]
        self._index += 1
        return result


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


def _run_export_main(tmp_path, monkeypatch, records, stub, argv=None):
    """Run :func:`ep.main` with a stubbed predictor and table loader.

    Writes ``records`` to ``tmp_path/fixtures.json``, stubs
    ``evaluate.load_matches_table`` to return the synthetic frame and
    ``predict.Predictor`` to ``stub``, then invokes ``ep.main`` with
    ``--output-dir tmp_path --version v1 --fixtures <the written file>``
    plus the caller's extra ``argv``.

    Args:
        tmp_path: The pytest tmp dir.
        monkeypatch: The pytest monkeypatch fixture.
        records: The fixtures list to write.
        stub: The :class:`_StubPredictor` instance to stand in for
            ``drivers.predict.Predictor``.
        argv: Extra CLI arguments (after the shared three); ``None``
            means none.

    Returns:
        The ``stub`` instance.

    Raises:
        Anything :func:`ep.main` raises (the caller asserts on it).
    """
    fixtures_path = tmp_path / "fixtures.json"
    _write_json(fixtures_path, {"fixtures": records})
    matches_df = _stub_matches_df()
    monkeypatch.setattr(
        ep.evaluate,
        "load_matches_table",
        partial(_load_matches_table_stub, matches_df),
    )
    monkeypatch.setattr(ep.predict, "Predictor", stub)
    extra = argv if argv is not None else []
    ep.main(
        [
            "--output-dir", str(tmp_path),
            "--version", "v1",
            "--fixtures", str(fixtures_path),
        ]
        + extra
    )
    return stub


def _key_paths_with(obj, names):
    """Return the dotted paths of every dict key named in ``names``.

    Recursively walks a JSON-native object (dict/list/leaf) and returns
    the dot-joined path string of each dict key whose name equals one
    of the given names, so a test can assert a diagnostic key never
    reached the written artifact.

    Args:
        obj: The JSON-native object to walk.
        names: The iterable of forbidden key names to look for.

    Returns:
        A list of dotted path strings (e.g. ``"fixtures.0.gap"``),
        one per occurrence, in walk order.

    Raises:
        Nothing.
    """
    forbidden = set(names)
    found: list[str] = []

    def walk(node, path):
        """Recurse one node, appending forbidden-key paths to ``found``.

        Args:
            node: The current JSON-native node.
            path: The dotted path string to this node (``""`` at the
                root).

        Returns:
            None (mutates the enclosing ``found`` list).

        Raises:
            Nothing.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in forbidden:
                    found.append(here)
                walk(value, here)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}.{index}")

    walk(obj, "")
    return found


# --------------------------------------------------------------------------
# count_null_interval_maps (D4)
# --------------------------------------------------------------------------


def test_count_null_interval_maps_all_null():
    # Every entry both-null -> the full count is returned.
    assert ep.count_null_interval_maps([_per_map(False), _per_map(False)]) == 2


def test_count_null_interval_maps_all_populated():
    # Every entry both-populated -> 0.
    assert ep.count_null_interval_maps([_per_map(True), _per_map(True)]) == 0


def test_count_null_interval_maps_mixed():
    # One null, one populated -> exactly the null count.
    assert ep.count_null_interval_maps(
        [_per_map(False), _per_map(True), _per_map(False)]
    ) == 2


def test_count_null_interval_maps_empty():
    # Empty sequence -> 0.
    assert ep.count_null_interval_maps([]) == 0


def test_count_null_interval_maps_partial_band_not_counted():
    # D4 is strictly both-null: a low-set/high-None entry is not a null
    # interval, so it is not counted.
    mixed = [
        _per_map(False),
        {"interval_low": (0.3, 0.0, 0.0, 0.2), "interval_high": None},
    ]
    assert ep.count_null_interval_maps(mixed) == 1


# --------------------------------------------------------------------------
# coverage_diagnostic (D5)
# --------------------------------------------------------------------------


def test_coverage_diagnostic_two_entries_weighted_and_gap():
    # Hand-computed: mass=0.6; weighted=(0.4*0.7+0.2*0.3)/0.6; gap vs 0.5.
    fixture = _fixture(
        [
            {"veto_probability": 0.4, "p_a_wins_series": 0.7},
            {"veto_probability": 0.2, "p_a_wins_series": 0.3},
        ],
        0.5,
    )
    diag = ep.coverage_diagnostic(fixture)
    assert diag.mass == pytest.approx(0.6)
    assert diag.weighted == pytest.approx(
        (0.4 * 0.7 + 0.2 * 0.3) / 0.6
    )
    assert diag.gap == pytest.approx(diag.weighted - 0.5)


def test_coverage_diagnostic_partial_coverage_divisor_is_mass():
    # Probabilities summing well under 1: the divisor is the summed
    # mass, not 1.0.
    fixture = _fixture(
        [
            {"veto_probability": 0.1, "p_a_wins_series": 0.8},
            {"veto_probability": 0.2, "p_a_wins_series": 0.4},
        ],
        0.6,
    )
    diag = ep.coverage_diagnostic(fixture)
    assert diag.mass == pytest.approx(0.3)
    assert diag.weighted == pytest.approx(
        (0.1 * 0.8 + 0.2 * 0.4) / 0.3
    )


def test_coverage_diagnostic_zero_mass_returns_none():
    # Empty listing and an all-zero-probability listing both yield None
    # with no exception (never ZeroDivisionError, never a 0.0 gap).
    assert ep.coverage_diagnostic(_fixture([], 0.5)) is None
    assert (
        ep.coverage_diagnostic(
            _fixture([{"veto_probability": 0.0, "p_a_wins_series": 0.7}], 0.5)
        )
        is None
    )


def test_coverage_diagnostic_single_entry_gap():
    # One entry -> weighted is that entry, gap is entry - overall.
    fixture = _fixture(
        [{"veto_probability": 0.4, "p_a_wins_series": 0.7}], 0.55
    )
    diag = ep.coverage_diagnostic(fixture)
    assert diag.weighted == pytest.approx(0.7)
    assert diag.gap == pytest.approx(0.7 - 0.55)


def test_coverage_diagnostic_mass_matches_coverage_mass_key():
    # D5: the recomputed mass equals fixture["coverage_mass"] by
    # construction (asserted, not read from the key).
    fixture = _fixture(
        [
            {"veto_probability": 0.3, "p_a_wins_series": 0.6},
            {"veto_probability": 0.25, "p_a_wins_series": 0.5},
        ],
        0.5,
    )
    diag = ep.coverage_diagnostic(fixture)
    assert diag.mass == pytest.approx(fixture["coverage_mass"])


# --------------------------------------------------------------------------
# Timing log lines (D3/D11)
# --------------------------------------------------------------------------


def test_timing_lines_present_and_parseable(tmp_path, monkeypatch, caplog):
    # One construction line, one per-fixture line per success, one
    # aggregate line; every number parses as a float >= 0.0 (D11).
    stub = _StubPredictor(results=[_make_result(), _make_result()])
    with caplog.at_level(logging.INFO):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1"), _good_record("2")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    constructed = [
        r.message for r in caplog.records
        if "predictor constructed in" in r.message
    ]
    assert len(constructed) == 1
    constructed_match = re.search(r"([0-9]+\.[0-9]+)s", constructed[0])
    assert float(constructed_match.group(1)) >= 0.0

    per_fixture = [
        r.message for r in caplog.records
        if " predicted in " in r.message and " assembled in " in r.message
    ]
    assert len(per_fixture) == 2
    for message in per_fixture:
        match = re.search(
            r"predicted in ([0-9]+\.[0-9]+)s, assembled in "
            r"([0-9]+\.[0-9]+)s",
            message,
        )
        assert float(match.group(1)) >= 0.0
        assert float(match.group(2)) >= 0.0

    aggregate = [
        r.message for r in caplog.records
        if " fixture(s) in " in r.message and " total (mean " in r.message
    ]
    assert len(aggregate) == 1
    match = re.search(
        r"in ([0-9]+\.[0-9]+)s total \(mean ([0-9]+\.[0-9]+)s/fixture\), "
        r"assembled in ([0-9]+\.[0-9]+)s total",
        aggregate[0],
    )
    for group in match.groups():
        assert float(group) >= 0.0


# --------------------------------------------------------------------------
# Null-interval verdict (D4)
# --------------------------------------------------------------------------


def test_null_interval_warning_replicate_free(tmp_path, monkeypatch, caplog):
    # Null overall intervals -> a WARNING naming
    # train_bootstrap_replicates.py with the right null/total counts.
    result = _make_result(per_map=(_make_per_map_pred((0.5, 0.1, 0.1, 0.3)),))
    stub = _StubPredictor(results=[result])
    with caplog.at_level(logging.WARNING):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "null intervals" in r.message
    ]
    assert len(warnings) == 1
    assert "train_bootstrap_replicates.py" in warnings[0].message
    assert "1 of 1 exported per_map entries carry null intervals" in (
        warnings[0].message
    )


def test_null_interval_all_carry_intervals_info(tmp_path, monkeypatch, caplog):
    # Populated intervals -> no WARNING and the "all ... carry
    # intervals" INFO.
    pm = _make_per_map_pred(
        (0.5, 0.1, 0.1, 0.3),
        interval_low=(0.3, 0.0, 0.0, 0.2),
        interval_high=(0.7, 0.3, 0.3, 0.5),
    )
    stub = _StubPredictor(results=[_make_result(per_map=(pm,))])
    with caplog.at_level(logging.INFO):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    assert not any("null intervals" in r.message for r in caplog.records)
    infos = [
        r.message for r in caplog.records
        if "carry intervals" in r.message
    ]
    assert infos == ["all 1 exported per_map entries carry intervals"]


def test_null_interval_mixed_count(tmp_path, monkeypatch, caplog):
    # One fixture with intervals, one without -> the count (1 of 2) is
    # asserted, not just the warning's presence.
    pm_null = _make_per_map_pred((0.5, 0.1, 0.1, 0.3))
    pm_full = _make_per_map_pred(
        (0.5, 0.1, 0.1, 0.3),
        interval_low=(0.3, 0.0, 0.0, 0.2),
        interval_high=(0.7, 0.3, 0.3, 0.5),
    )
    stub = _StubPredictor(
        results=[
            _make_result(per_map=(pm_null,)),
            _make_result(per_map=(pm_full,)),
        ]
    )
    with caplog.at_level(logging.WARNING):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1"), _good_record("2")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "null intervals" in r.message
    ]
    assert len(warnings) == 1
    assert "1 of 2 exported per_map entries carry null intervals" in (
        warnings[0].message
    )


# --------------------------------------------------------------------------
# Coverage diagnostic logging (D5)
# --------------------------------------------------------------------------


def test_coverage_diagnostic_logged_and_never_exported(
    tmp_path, monkeypatch, caplog
):
    # A non-zero-mass fixture logs one INFO line naming its match_id;
    # the written artifact carries no gap/weighted/coverage_diagnostic
    # key anywhere (D5: diagnostic only, never in the artifact).
    result = _make_result(top_vetos=(_make_ranked_entry(0.4),))
    stub = _StubPredictor(results=[result])
    with caplog.at_level(logging.INFO):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    lines = [
        r.message for r in caplog.records
        if r.message.startswith("coverage diagnostic 1: mass=")
    ]
    assert len(lines) == 1
    artifact = json.loads(
        (tmp_path / "v1" / "predictions.json").read_text(encoding="utf-8")
    )
    assert _key_paths_with(
        artifact, {"gap", "weighted", "coverage_diagnostic"}
    ) == []


def test_coverage_diagnostic_zero_mass_logs_skipped(
    tmp_path, monkeypatch, caplog
):
    # An empty top_vetos listing (zero coverage mass) logs the "no
    # coverage mass" INFO instead of a numeric diagnostic.
    stub = _StubPredictor(results=[_make_result()])
    with caplog.at_level(logging.INFO):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    lines = [
        r.message for r in caplog.records
        if r.message == "coverage diagnostic 1: no coverage mass, skipped"
    ]
    assert len(lines) == 1


# --------------------------------------------------------------------------
# Summary line (D7)
# --------------------------------------------------------------------------


def test_summary_line_appends_fields_without_reordering(
    tmp_path, monkeypatch, caplog
):
    # The appended D7 fields are present and the pre-existing fields
    # still precede them in their original order.
    stub = _StubPredictor(results=[_make_result()])
    with caplog.at_level(logging.INFO):
        _run_export_main(
            tmp_path,
            monkeypatch,
            [_good_record("1")],
            stub,
            argv=["--as-of-date", "2026-09-01T00:00:00"],
        )
    summaries = [
        r.message for r in caplog.records if r.message.startswith("exported ")
    ]
    assert len(summaries) == 1
    message = summaries[0]
    existing = ["dropped_past=", "failed=", "intervals_present=",
                "model_version=", "dataset_version="]
    appended = ["null_interval_maps=", "predict_seconds=", "elapsed_seconds="]
    for field in existing + appended:
        assert field in message
    positions = [message.index(field) for field in existing + appended]
    assert positions == sorted(positions)
