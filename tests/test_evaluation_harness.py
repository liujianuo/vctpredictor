"""Tests for the map-outcome evaluation harness (M19).

Covers the held-out-set assembly (:func:`build_held_out_maps`'s
join/filter correctness and its empty-split ``ValueError`), per-map
scoring (:func:`score_held_out_maps`'s per-row scores cross-checked
against directly calling ``utils.scoring``, plus the wrong-length
vector rejection), the report builder
(:func:`build_evaluation_report`'s calibration arithmetic with a
hand-computable fixture), and a skip-guarded end-to-end run of the M18
baseline adapter over the real ``data/v1`` ``test`` split that locks in
the floor numbers later models must beat.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from evaluation.harness import (
    HELD_OUT_COLUMNS,
    OUTCOME_LABELS,
    PREDICTION_COLUMNS,
    SCORED_COLUMNS,
    build_evaluation_report,
    build_held_out_maps,
    four_way_baseline_model,
    score_held_out_maps,
)
from utils import scoring, splits

_MATCHES_COLS = ["match_id", "date", "team1_id", "team2_id", "status"]
_MAPS_COLS = [
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
]
_LABELS_COLS = [
    "match_id",
    "map_index",
    "outcome_label",
    "outcome_ordinal",
    "round_margin",
]
_SPLITS_COLS = ["match_id", "date", "split"]


def _label(t1s, t2s):
    """Derive the (label, ordinal, margin) tuple for one scoreline.

    The tiny from-scratch transcription of ``drivers.labels``'s
    labelling rule used to build fixture labels tables: A = team1
    (``t1s``), B = team2 (``t2s``), overtime iff the loser reached 12+
    rounds, ordinal 0..3 in OUTCOME_LABELS order.

    Args:
        t1s: Rounds team1 won on the finished map.
        t2s: Rounds team2 won on the finished map.

    Returns:
        A ``(outcome_label, outcome_ordinal, round_margin)`` tuple in
        :data:`OUTCOME_LABELS` terms.

    Raises:
        ValueError: If the scoreline is tied (no winner to label),
            propagated from the fixture builder's caller.
    """
    if t1s == t2s:
        raise ValueError(f"tied fixture scoreline {t1s}-{t2s} has no label")
    overtime = min(t1s, t2s) >= 12
    margin = t1s - t2s
    if t1s > t2s:
        return ("A-OT", 1, margin) if overtime else ("A-regulation", 0, margin)
    return ("B-OT", 2, margin) if overtime else ("B-regulation", 3, margin)


def _add(match_rows, map_rows, mid, date, team1_id, team2_id, map_name, t1s, t2s):
    """Append one completed match and its finished map to the row lists.

    The single row-writing helper for the synthetic league fixtures,
    mirroring ``test_four_way_baseline.py``'s convention: the map's
    ``winner`` is derived from the scores (never a display-name
    string).

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
        ValueError: If the scoreline is tied (propagated from
            :func:`_label` via the caller's labels build).
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


def _labels_df(map_rows):
    """Build a labels table from the fixture map rows.

    One row per map in ``map_rows`` in input order, labelled by
    :func:`_label` from the two scores. Every fixture map is finished
    (both scores present), so no row is skipped.

    Args:
        map_rows: A list of map dicts (each with ``match_id``,
            ``map_index``, ``team1_score``, ``team2_score``).

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_LABELS_COLS`
        columns, one row per map.

    Raises:
        ValueError: If any map's scoreline is tied (propagated from
            :func:`_label`).
    """
    rows = []
    for row in map_rows:
        label, ordinal, margin = _label(row["team1_score"], row["team2_score"])
        rows.append(
            {
                "match_id": row["match_id"],
                "map_index": row["map_index"],
                "outcome_label": label,
                "outcome_ordinal": ordinal,
                "round_margin": margin,
            }
        )
    return pd.DataFrame(rows, columns=_LABELS_COLS)


def _splits_df(match_rows, test_ids):
    """Build a splits table assigning ``test_ids`` to the test split.

    Every other match in ``match_rows`` is assigned ``"train"``, so the
    fixture's split has exactly the two values ``utils.splits`` defines.

    Args:
        match_rows: A list of match dicts (each with ``match_id`` and
            ``date``).
        test_ids: The set of ``match_id`` values to label ``"test"``.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_SPLITS_COLS`
        columns, one row per match in ``match_rows`` order.

    Raises:
        Nothing.
    """
    rows = [
        {
            "match_id": row["match_id"],
            "date": row["date"],
            "split": "test" if row["match_id"] in test_ids else "train",
        }
        for row in match_rows
    ]
    return pd.DataFrame(rows, columns=_SPLITS_COLS)


def _harness_tables():
    """Build the 4-match, 4-map held-out assembly league.

    Four matches, one finished map each, with an explicit split:
    ``m1``/``m2`` are ``train`` and ``m3``/``m4`` are ``test``. The
    scorelines are chosen so the four outcome ordinals each appear
    exactly once across the whole league (and the two test maps are
    ``m3`` 14-12 -> A-OT ordinal 1 and ``m4`` 12-14 -> B-OT ordinal 2).

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df)`` tuple of
        4-match, 4-map, 4-label, 4-split-row frames built with the
        fixed column conventions above.

    Raises:
        ValueError: If any fixture scoreline is tied (propagated from
            :func:`_label`).
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "m1", "2026-05-01T10:00:00", "A", "X", "Haven", 13, 8)
    _add(match_rows, map_rows, "m2", "2026-05-02T10:00:00", "B", "Y", "Haven", 8, 13)
    _add(match_rows, map_rows, "m3", "2026-05-03T10:00:00", "C", "Z", "Bind", 14, 12)
    _add(match_rows, map_rows, "m4", "2026-05-04T10:00:00", "D", "W", "Bind", 12, 14)
    matches_df = pd.DataFrame(match_rows, columns=_MATCHES_COLS)
    maps_df = pd.DataFrame(map_rows, columns=_MAPS_COLS)
    labels_df = _labels_df(map_rows)
    splits_df = _splits_df(match_rows, test_ids={"m3", "m4"})
    return matches_df, maps_df, labels_df, splits_df


def _calibration_tables():
    """Build the 4-map all-test league for report arithmetic.

    Four matches, one finished map each, all in the ``test`` split.
    Scorelines 13-8, 13-8, 14-12, 12-14 give true ordinals
    ``[0, 0, 1, 2]`` (A-regulation, A-regulation, A-OT, B-OT). Paired
    with the fixed stub vector ``(0.5, 0.1, 0.1, 0.3)`` used by
    :func:`_calibration_stub`, every report number is independently
    hand-computable:

    - predicted means ``(0.5, 0.1, 0.1, 0.3)``,
    - observed frequencies ``(0.5, 0.25, 0.25, 0.0)``,
    - gaps ``(0.0, 0.15, 0.15, 0.3)`` -> most miscalibrated is
      ``B-regulation`` (unique max gap),
    - OT rate: predicted ``0.1 + 0.1 = 0.2``, observed
      ``2/4 = 0.5``, gap ``0.3``.

    Returns:
        A ``(matches_df, maps_df, labels_df, splits_df)`` tuple of
        four one-map matches, every one in the test split.

    Raises:
        ValueError: If any fixture scoreline is tied (propagated from
            :func:`_label`).
    """
    match_rows = []
    map_rows = []
    _add(match_rows, map_rows, "c1", "2026-05-01T10:00:00", "A", "X", "Haven", 13, 8)
    _add(match_rows, map_rows, "c2", "2026-05-02T10:00:00", "B", "Y", "Haven", 13, 8)
    _add(match_rows, map_rows, "c3", "2026-05-03T10:00:00", "C", "Z", "Bind", 14, 12)
    _add(match_rows, map_rows, "c4", "2026-05-04T10:00:00", "D", "W", "Bind", 12, 14)
    matches_df = pd.DataFrame(match_rows, columns=_MATCHES_COLS)
    maps_df = pd.DataFrame(map_rows, columns=_MAPS_COLS)
    labels_df = _labels_df(map_rows)
    splits_df = _splits_df(match_rows, test_ids={"c1", "c2", "c3", "c4"})
    return matches_df, maps_df, labels_df, splits_df


def _stub_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """A deterministic per-team stub ModelFn for score-correctness tests.

    Returns a fixed 4-vector chosen by ``team1_id`` so different
    held-out maps get different, known predictions: ``C`` ->
    ``(0.5, 0.2, 0.1, 0.2)`` and ``D`` -> ``(0.1, 0.1, 0.3, 0.5)``.
    All other arguments are ignored — the stub asserts nothing about
    the history tables it is handed.

    Args:
        team1_id: The queried team1's id; selects the returned vector.
        team2_id: Ignored (kept for the ModelFn signature).
        map_name: Ignored (kept for the ModelFn signature).
        date: Ignored (kept for the ModelFn signature).
        matches_df: Ignored (kept for the ModelFn signature).
        maps_df: Ignored (kept for the ModelFn signature).

    Returns:
        The 4-tuple of probabilities for ``team1_id``.

    Raises:
        KeyError: If ``team1_id`` is not ``C`` or ``D`` (a fixture
            bug, not a data condition).
    """
    return {"C": (0.5, 0.2, 0.1, 0.2), "D": (0.1, 0.1, 0.3, 0.5)}[team1_id]


def _calibration_stub(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """A constant-vector stub ModelFn for the report-arithmetic fixture.

    Returns the fixed vector ``(0.5, 0.1, 0.1, 0.3)`` for every map
    regardless of its arguments — the vector :func:`_calibration_tables`
    was designed around (see its docstring's hand computation).

    Args:
        team1_id: Ignored (kept for the ModelFn signature).
        team2_id: Ignored (kept for the ModelFn signature).
        map_name: Ignored (kept for the ModelFn signature).
        date: Ignored (kept for the ModelFn signature).
        matches_df: Ignored (kept for the ModelFn signature).
        maps_df: Ignored (kept for the ModelFn signature).

    Returns:
        The 4-tuple ``(0.5, 0.1, 0.1, 0.3)``.

    Raises:
        Nothing.
    """
    return (0.5, 0.1, 0.1, 0.3)


def _real_v1_available():
    """Report whether the materialised v1 tables exist on disk.

    The skip guard for the real-data test, matching the convention in
    ``test_four_way_baseline.py``: all four Parquet files must exist
    (i.e. ``materialize.py``, ``labels.py`` and ``splits.py`` have
    been run).

    Returns:
        A bool: ``True`` iff ``data/v1/{matches,maps,labels,splits}.parquet``
        all exist.

    Raises:
        Nothing.
    """
    return all(
        Path(f"data/v1/{name}.parquet").exists()
        for name in ("matches", "maps", "labels", "splits")
    )


# --------------------------------------------------------------------------
# plan#9a: build_held_out_maps join/filter correctness
# --------------------------------------------------------------------------


def test_build_held_out_maps_filters_to_split_and_joins_identifiers():
    # The test split of the 4-match league is exactly m3/m4, and each
    # held-out row must carry the matches-side team ids/date and the
    # labels-side ordinal joined in from the other two tables.
    matches_df, maps_df, labels_df, splits_df = _harness_tables()
    held_out = build_held_out_maps(matches_df, maps_df, labels_df, splits_df)
    assert list(held_out.columns) == list(HELD_OUT_COLUMNS)
    assert len(held_out) == 2
    assert set(held_out["match_id"]) == {"m3", "m4"}
    m3 = held_out[held_out["match_id"] == "m3"].iloc[0]
    assert m3["team1_id"] == "C"
    assert m3["team2_id"] == "Z"
    assert m3["date"] == "2026-05-03T10:00:00"
    assert m3["map_name"] == "Bind"
    assert m3["outcome_ordinal"] == 1  # 14-12 -> A-OT
    m4 = held_out[held_out["match_id"] == "m4"].iloc[0]
    assert m4["team1_id"] == "D"
    assert m4["team2_id"] == "W"
    assert m4["outcome_ordinal"] == 2  # 12-14 -> B-OT
    # The train side is the complementary two maps, ordinals 0 and 3.
    train = build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="train"
    )
    assert set(train["match_id"]) == {"m1", "m2"}
    assert set(train["outcome_ordinal"]) == {0, 3}


def test_build_held_out_maps_drops_unlabelled_test_maps():
    # Removing m4's label row exercises the documented inner-join
    # behavior: a test-split map whose label is absent (the
    # skipped-null-score case labels.py permits) is silently excluded
    # from the held-out set rather than erroring.
    matches_df, maps_df, labels_df, splits_df = _harness_tables()
    labels_df = labels_df[labels_df["match_id"] != "m4"]
    held_out = build_held_out_maps(matches_df, maps_df, labels_df, splits_df)
    assert list(held_out["match_id"]) == ["m3"]
    assert list(held_out.columns) == list(HELD_OUT_COLUMNS)


def test_build_held_out_maps_raises_on_empty_split():
    # No match assigned to "test" makes the split-restricted result
    # empty: the harness must fail loudly rather than return an empty
    # held-out table (which would silently mean "nothing evaluated").
    matches_df, maps_df, labels_df, splits_df = _harness_tables()
    splits_df = splits_df.assign(split="train")
    with pytest.raises(ValueError, match="no held-out maps"):
        build_held_out_maps(matches_df, maps_df, labels_df, splits_df)


# --------------------------------------------------------------------------
# plan#9b: score_held_out_maps per-row score correctness
# --------------------------------------------------------------------------


def test_score_held_out_maps_cross_checks_scoring_functions():
    # With the per-team stub, each held-out row's recorded probabilities
    # and per-map scores must exactly equal what directly calling
    # utils.scoring's own rps/log_loss/marginal_binary_accuracy on the
    # same (probs, ordinal) inputs returns — the harness must not
    # reimplement or distort the metric math.
    matches_df, maps_df, labels_df, splits_df = _harness_tables()
    held_out = build_held_out_maps(matches_df, maps_df, labels_df, splits_df)
    scored = score_held_out_maps(_stub_model_fn, held_out, matches_df, maps_df)
    assert list(scored.columns) == list(SCORED_COLUMNS)
    assert len(scored) == 2
    m3 = scored[scored["match_id"] == "m3"].iloc[0]
    m3_probs = (0.5, 0.2, 0.1, 0.2)
    assert list(m3[list(PREDICTION_COLUMNS)]) == pytest.approx(list(m3_probs))
    assert m3["outcome_ordinal"] == 1
    assert m3["rps"] == pytest.approx(scoring.rps(m3_probs, 1))
    assert m3["log_loss"] == pytest.approx(scoring.log_loss(m3_probs, 1))
    assert m3["marginal_correct"] == scoring.marginal_binary_accuracy(
        m3_probs, 1
    )
    m4 = scored[scored["match_id"] == "m4"].iloc[0]
    m4_probs = (0.1, 0.1, 0.3, 0.5)
    assert list(m4[list(PREDICTION_COLUMNS)]) == pytest.approx(list(m4_probs))
    assert m4["outcome_ordinal"] == 2
    assert m4["rps"] == pytest.approx(scoring.rps(m4_probs, 2))
    assert m4["log_loss"] == pytest.approx(scoring.log_loss(m4_probs, 2))
    assert m4["marginal_correct"] == scoring.marginal_binary_accuracy(
        m4_probs, 2
    )


def _short_model_fn(team1_id, team2_id, map_name, date, matches_df, maps_df):
    """A deliberately wrong ModelFn returning only two probabilities.

    Exercises the harness's length validation: the returned sequence is
    not the required 4-vector, so ``score_held_out_maps`` must reject
    it with a per-map error naming the offending map rather than
    silently mis-scoring.

    Args:
        team1_id: Ignored (kept for the ModelFn signature).
        team2_id: Ignored (kept for the ModelFn signature).
        map_name: Ignored (kept for the ModelFn signature).
        date: Ignored (kept for the ModelFn signature).
        matches_df: Ignored (kept for the ModelFn signature).
        maps_df: Ignored (kept for the ModelFn signature).

    Returns:
        The 2-tuple ``(0.5, 0.5)`` — deliberately the wrong length.

    Raises:
        Nothing.
    """
    return (0.5, 0.5)


def test_score_held_out_maps_rejects_wrong_length_vector():
    # A model returning a 2-vector (or any length other than 4) must be
    # a hard ValueError naming the offending map, not a silent
    # mis-scoring or a confusing downstream error.
    matches_df, maps_df, labels_df, splits_df = _harness_tables()
    held_out = build_held_out_maps(matches_df, maps_df, labels_df, splits_df)
    with pytest.raises(ValueError, match="expected exactly 4"):
        score_held_out_maps(_short_model_fn, held_out, matches_df, maps_df)


# --------------------------------------------------------------------------
# plan#9c: build_evaluation_report calibration arithmetic
# --------------------------------------------------------------------------


def test_build_evaluation_report_calibration_arithmetic():
    # All four maps in the test split with true ordinals [0, 0, 1, 2]
    # and the fixed stub vector (0.5, 0.1, 0.1, 0.3) make every report
    # number independently hand-computable (see _calibration_tables):
    # predicted means (0.5, 0.1, 0.1, 0.3), observed frequencies
    # (0.5, 0.25, 0.25, 0.0), gaps (0.0, 0.15, 0.15, 0.3) -> most
    # miscalibrated "B-regulation"; OT predicted 0.2 vs observed 0.5.
    matches_df, maps_df, labels_df, splits_df = _calibration_tables()
    held_out = build_held_out_maps(matches_df, maps_df, labels_df, splits_df)
    scored = score_held_out_maps(
        _calibration_stub, held_out, matches_df, maps_df
    )
    report = build_evaluation_report(scored)

    assert report["n_eval"] == 4
    # Headline metrics must be traceable to the shared scoring batch
    # functions over the prediction columns, not a second mean.
    prob_rows = scored[list(PREDICTION_COLUMNS)].to_numpy()
    ordinals = scored["outcome_ordinal"].to_numpy()
    assert report["mean_rps"] == pytest.approx(scoring.mean_rps(prob_rows, ordinals))
    assert report["mean_log_loss"] == pytest.approx(
        scoring.mean_log_loss(prob_rows, ordinals)
    )
    assert report["marginal_binary_accuracy"] == pytest.approx(
        scoring.mean_marginal_binary_accuracy(prob_rows, ordinals)
    )
    # Calibration table arithmetic.
    cal = {entry["category"]: entry for entry in report["calibration"]}
    assert list(cal) == list(OUTCOME_LABELS)
    assert cal["A-regulation"]["predicted_mean_prob"] == pytest.approx(0.5)
    assert cal["A-regulation"]["observed_frequency"] == pytest.approx(0.5)
    assert cal["A-regulation"]["gap"] == pytest.approx(0.0)
    assert cal["A-OT"]["predicted_mean_prob"] == pytest.approx(0.1)
    assert cal["A-OT"]["observed_frequency"] == pytest.approx(0.25)
    assert cal["A-OT"]["gap"] == pytest.approx(0.15)
    assert cal["B-OT"]["predicted_mean_prob"] == pytest.approx(0.1)
    assert cal["B-OT"]["observed_frequency"] == pytest.approx(0.25)
    assert cal["B-OT"]["gap"] == pytest.approx(0.15)
    assert cal["B-regulation"]["predicted_mean_prob"] == pytest.approx(0.3)
    assert cal["B-regulation"]["observed_frequency"] == pytest.approx(0.0)
    assert cal["B-regulation"]["gap"] == pytest.approx(0.3)
    assert report["most_miscalibrated_category"] == "B-regulation"
    ot = report["predicted_vs_observed_ot_rate"]
    assert ot["predicted"] == pytest.approx(0.2)
    assert ot["observed"] == pytest.approx(0.5)
    assert ot["gap"] == pytest.approx(0.3)
    # The whole report must be JSON-serializable (the CLI writes it
    # with json.dumps, so a numpy scalar would break the artifact).
    assert json.dumps(report)


# --------------------------------------------------------------------------
# plan#9d: real v1 end-to-end through the M18 baseline adapter
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_v1_harness_end_to_end():
    # Run the whole harness with the M18 baseline adapter over the real
    # v1 test split: the held-out count must match an independent
    # recomputation, the scored table must have the fixed schema, and
    # the report must satisfy plausible bounds (RPS in [0, 3], log loss
    # finite and positive, accuracy in [0, 1], 4-category calibration
    # table in OUTCOME_LABELS order). This is the test that produces
    # and locks in the M18 floor numbers later models must beat.
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")

    held_out = build_held_out_maps(matches_df, maps_df, labels_df, splits_df)
    assert list(held_out.columns) == list(HELD_OUT_COLUMNS)
    # Independently recompute the expected test-split map count the same
    # way the harness does: split -> test filter -> label inner join.
    test_maps = splits.join_split_to_maps(maps_df, splits_df)
    test_maps = test_maps[test_maps["split"] == "test"]
    labelled = test_maps.merge(
        labels_df[["match_id", "map_index"]],
        on=["match_id", "map_index"],
        how="inner",
    )
    expected_n = len(labelled)
    assert expected_n > 0
    assert len(held_out) == expected_n

    scored = score_held_out_maps(
        four_way_baseline_model, held_out, matches_df, maps_df
    )
    assert list(scored.columns) == list(SCORED_COLUMNS)
    assert len(scored) == expected_n

    report = build_evaluation_report(scored)
    assert report["n_eval"] == expected_n
    assert 0.0 <= report["mean_rps"] <= 3.0
    assert math.isfinite(report["mean_log_loss"])
    assert report["mean_log_loss"] > 0.0
    assert 0.0 <= report["marginal_binary_accuracy"] <= 1.0
    assert [entry["category"] for entry in report["calibration"]] == list(
        OUTCOME_LABELS
    )
    assert report["most_miscalibrated_category"] in OUTCOME_LABELS
    ot = report["predicted_vs_observed_ot_rate"]
    assert 0.0 <= ot["predicted"] <= 1.0
    assert 0.0 <= ot["observed"] <= 1.0
    # Every predicted vector in the scored table is a valid simplex
    # (finite, non-negative, sums to 1), i.e. the whole held-out run
    # produced scorable predictions with no NaN anywhere.
    pred_matrix = scored[list(PREDICTION_COLUMNS)].to_numpy()
    assert math.isfinite(pred_matrix.sum())
    assert (pred_matrix >= 0.0).all()
    assert all(
        abs(pred_matrix[i].sum() - 1.0) < 1e-9 for i in range(len(scored))
    )
