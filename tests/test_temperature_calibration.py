"""Tests for the temperature-calibration comparison module (M24).

Covers ``build_calibration_comparison_report`` on synthetic scored
DataFrames: a case engineered so ``T`` visibly changes ``mean_log_loss``
but leaves ``binary_side_agreement_rate == 1.0`` (decisions_invariant
True), a second case engineered so it does *not* (decisions_invariant
False — both branches reachable), the row-alignment ``ValueError``
(different row counts / different match ids / same ids in different
order), the JSON-serializability of the full report, and a real v1
end-to-end run via ``drivers/evaluate_temperature_calibration.py`` that
asserts **internal consistency only** (deltas are finite floats,
``decisions_invariant`` is a bool, agreement rates are in ``[0, 1]``)
and prints the actually-measured values — decision B forbids asserting
a predetermined direction for the deltas or the invariant flag.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from evaluation import temperature_calibration as tc
from tests._shared import _real_v1_available as _real_v1_tables_available
from utils import scoring

# The exact column order of a score_held_out_maps table (kept in sync
# with evaluation.harness.SCORED_COLUMNS so the fixtures are honest
# "already-scored" tables).
_SCORED_COLUMNS = (
    "match_id",
    "map_index",
    "date",
    "team1_id",
    "team2_id",
    "map_name",
    "outcome_ordinal",
    "p_a_regulation",
    "p_a_ot",
    "p_b_ot",
    "p_b_regulation",
    "rps",
    "log_loss",
    "marginal_correct",
)


def _real_v1_available():
    """Report whether the real v1 tables and both model artifacts exist.

    The skip guard for the end-to-end comparison test: the parquet
    half is delegated to ``tests._shared._real_v1_available`` (the
    single shared home of the bare-table-name convention — all five
    ``data/v1/*.parquet`` files), and this module additionally requires
    both fitted artifacts (``ordinal_logit_model.json`` and
    ``temperature_scaling_model.json``, i.e. ``train_ordinal_logit.py``
    and ``train_temperature_scaling.py`` have both been run) because
    the end-to-end comparison loads them.

    Returns:
        A bool: ``True`` iff all five ``data/v1/*.parquet`` files and
        both ``data/v1/*.json`` artifacts exist.

    Raises:
        Nothing.
    """
    return _real_v1_tables_available() and Path(
        "data/v1/ordinal_logit_model.json"
    ).exists() and Path(
        "data/v1/temperature_scaling_model.json"
    ).exists()


def _scored_frame(rows):
    """Build a score_held_out_maps-shaped DataFrame from per-map rows.

    Constructs an honest "already-scored" table with exactly
    :data:`_SCORED_COLUMNS`: the identifying columns, the true ordinal,
    the four prediction columns in ``OUTCOME_LABELS`` order, and the
    three per-row scores computed through ``utils.scoring``'s own
    per-observation functions (the same computation
    ``evaluation.harness.score_held_out_maps`` performs), so the
    fixture is a genuine scored table, not a stub.

    Args:
        rows: An iterable of ``(match_id, map_index, outcome_ordinal,
            probs)`` tuples, ``probs`` a 4-sequence summing to 1 in
            ``OUTCOME_LABELS`` order.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`_SCORED_COLUMNS`,
        one row per input in the input order (the order
        ``build_held_out_maps`` would have produced, i.e. the order the
        comparison report requires both tables to share).

    Raises:
        ValueError: If a ``probs`` vector fails ``utils.scoring``'s
            validation or a true ordinal is out of range (propagated
            from the per-row scoring calls).
    """
    records: list[dict] = []
    for i, (match_id, map_index, ordinal, probs) in enumerate(rows):
        probs = tuple(float(p) for p in probs)
        records.append(
            {
                "match_id": match_id,
                "map_index": map_index,
                "date": f"2024-01-{i + 1:02d}",
                "team1_id": "A",
                "team2_id": "B",
                "map_name": f"map-{i}",
                "outcome_ordinal": ordinal,
                "p_a_regulation": probs[0],
                "p_a_ot": probs[1],
                "p_b_ot": probs[2],
                "p_b_regulation": probs[3],
                "rps": scoring.rps(probs, ordinal),
                "log_loss": scoring.log_loss(probs, ordinal),
                "marginal_correct": scoring.marginal_binary_accuracy(
                    probs, ordinal
                ),
            }
        )
    return pd.DataFrame(records, columns=list(_SCORED_COLUMNS))


# --------------------------------------------------------------------------
# plan#12: build_calibration_comparison_report on synthetic fixtures
# --------------------------------------------------------------------------


# The four rows of the decisions-invariant fixture: on every row the
# calibrated probabilities are strictly more confident on the same
# argmax category (and the same side), so the two decision agreement
# rates are 1.0 while every log loss visibly drops.
_INVARIANT_ROWS = [
    ("m1", 0, 1, (0.10, 0.60, 0.20, 0.10), (0.05, 0.85, 0.05, 0.05)),
    ("m1", 1, 0, (0.55, 0.15, 0.15, 0.15), (0.70, 0.10, 0.10, 0.10)),
    ("m2", 0, 2, (0.10, 0.20, 0.60, 0.10), (0.05, 0.05, 0.85, 0.05)),
    ("m2", 1, 3, (0.15, 0.15, 0.15, 0.55), (0.10, 0.10, 0.10, 0.70)),
]


def test_report_decisions_invariant_branch_with_visible_log_loss_change():
    # The decisions_invariant == True branch: temperature scaling is
    # engineered to sharpen every row's confidence on the *same*
    # argmax category and the *same* side, so both agreement rates are
    # exactly 1.0 while mean_log_loss visibly drops (the delta is
    # hand-computed below). The calibrated table is built from the same
    # (match_id, map_index) rows, only with different prediction
    # columns — exactly what scoring one model twice produces.
    uncal = _scored_frame(
        [(m, i, y, p) for m, i, y, p, _ in _INVARIANT_ROWS]
    )
    cal = _scored_frame(
        [(m, i, y, p) for m, i, y, _, p in _INVARIANT_ROWS]
    )
    report = tc.build_calibration_comparison_report(uncal, cal)

    assert report["n_eval"] == 4
    assert report["binary_side_agreement_rate"] == pytest.approx(1.0)
    assert report["argmax_category_agreement_rate"] == pytest.approx(1.0)
    assert report["decisions_invariant"] is True

    # Hand-computed log losses: uncalibrated rows put 0.60/0.55/0.60/
    # 0.55 on their true categories; calibrated put 0.85/0.70/0.85/
    # 0.70. The delta must match the closed form exactly.
    uncal_ll = -(math.log(0.6) + math.log(0.55)) / 2
    cal_ll = -(math.log(0.85) + math.log(0.70)) / 2
    assert report["uncalibrated"]["mean_log_loss"] == pytest.approx(uncal_ll)
    assert report["calibrated"]["mean_log_loss"] == pytest.approx(cal_ll)
    assert report["mean_log_loss_delta"] == pytest.approx(cal_ll - uncal_ll)
    assert report["mean_log_loss_delta"] < -0.25  # visibly improved

    # Hand-computed RPS (per-row values below), summed over the four
    # rows; both accuracies are 1.0 (every row's side call is correct
    # and unchanged), so the accuracy delta is exactly 0.
    assert report["uncalibrated"]["mean_rps"] == pytest.approx(
        (0.11 + 0.315 + 0.11 + 0.315) / 4
    )
    assert report["calibrated"]["mean_rps"] == pytest.approx(
        (0.015 + 0.14 + 0.015 + 0.14) / 4
    )
    assert report["mean_rps_delta"] == pytest.approx(
        (0.015 + 0.14 + 0.015 + 0.14 - 0.11 - 0.315 - 0.11 - 0.315) / 4
    )
    assert report["uncalibrated"]["marginal_binary_accuracy"] == pytest.approx(1.0)
    assert report["calibrated"]["marginal_binary_accuracy"] == pytest.approx(1.0)
    assert report["marginal_binary_accuracy_delta"] == pytest.approx(0.0)


# The first row of the decision-flipping fixture: the calibrated model
# moves from argmax 1 / side A to argmax 2 / side B — a genuine
# decision flip on that map. The other three rows keep their decisions.
_FLIPPING_ROWS = [
    ("m1", 0, 1, (0.10, 0.45, 0.35, 0.10), (0.05, 0.10, 0.80, 0.05)),
    ("m1", 1, 0, (0.55, 0.15, 0.15, 0.15), (0.70, 0.10, 0.10, 0.10)),
    ("m2", 0, 2, (0.10, 0.20, 0.60, 0.10), (0.05, 0.05, 0.85, 0.05)),
    ("m2", 1, 3, (0.15, 0.15, 0.15, 0.55), (0.10, 0.10, 0.10, 0.70)),
]


def test_report_decisions_not_invariant_branch():
    # The decisions_invariant == False branch: on row m1/0 the
    # calibrated model flips both the binary side call (A -> B) and the
    # 4-way argmax category (1 -> 2), so both agreement rates are 3/4
    # and decisions_invariant is False — proving the not-invariant
    # branch of decision B is reachable, not just asserted. The log
    # loss also *worsens* here (0.45 -> 0.10 on the true category);
    # that direction is deliberately not part of any assertion.
    uncal = _scored_frame(
        [(m, i, y, p) for m, i, y, p, _ in _FLIPPING_ROWS]
    )
    cal = _scored_frame(
        [(m, i, y, p) for m, i, y, _, p in _FLIPPING_ROWS]
    )
    report = tc.build_calibration_comparison_report(uncal, cal)

    assert report["n_eval"] == 4
    assert report["binary_side_agreement_rate"] == pytest.approx(0.75)
    assert report["argmax_category_agreement_rate"] == pytest.approx(0.75)
    assert report["decisions_invariant"] is False

    # Hand-computed: uncalibrated true-category probabilities are
    # 0.45/0.55/0.60/0.55; calibrated are 0.10/0.70/0.85/0.70.
    uncal_ll = (
        -(math.log(0.45) + math.log(0.55) + math.log(0.6) + math.log(0.55)) / 4
    )
    cal_ll = (
        -(math.log(0.10) + math.log(0.70) + math.log(0.85) + math.log(0.70)) / 4
    )
    assert report["mean_log_loss_delta"] == pytest.approx(cal_ll - uncal_ll)
    assert report["mean_log_loss_delta"] > 0.1  # visibly worse, allowed


def test_report_rejects_row_count_mismatch():
    # Two scored tables describing different numbers of held-out rows
    # cannot be compared; the pair is a hard error.
    uncal = _scored_frame(
        [(m, i, y, p) for m, i, y, p, _ in _INVARIANT_ROWS]
    )
    cal = _scored_frame(
        [(m, i, y, p) for m, i, y, _, p in _INVARIANT_ROWS[:-1]]
    )
    with pytest.raises(ValueError, match="row counts"):
        tc.build_calibration_comparison_report(uncal, cal)


def test_report_rejects_different_match_ids():
    # The same number of rows but a different map at some position is a
    # misalignment, not a valid comparison.
    uncal = _scored_frame(
        [(m, i, y, p) for m, i, y, p, _ in _INVARIANT_ROWS]
    )
    cal = _scored_frame(
        [
            ("m9", i, y, p) if (m, i) == ("m1", 0) else (m, i, y, p)
            for m, i, y, _, p in _INVARIANT_ROWS
        ]
    )
    with pytest.raises(ValueError, match="not row-aligned"):
        tc.build_calibration_comparison_report(uncal, cal)


def test_report_rejects_same_ids_in_different_order():
    # The same ids in a different order are still a misalignment: the
    # report compares positionally, so row 2 of one table would be
    # paired with row 3 of the other.
    uncal = _scored_frame(
        [(m, i, y, p) for m, i, y, p, _ in _INVARIANT_ROWS]
    )
    cal = _scored_frame(
        list(reversed([(m, i, y, p) for m, i, y, _, p in _INVARIANT_ROWS]))
    )
    with pytest.raises(ValueError, match="not row-aligned"):
        tc.build_calibration_comparison_report(uncal, cal)


def test_report_json_serializable_and_field_types():
    # The full report dict must be directly json.dumps-able (the CLI
    # driver writes it with json.dumps), with the documented key set
    # and the documented value types: int n_eval, three metric floats
    # per model, three float deltas, two float rates in [0, 1], one
    # bool decisions_invariant.
    uncal = _scored_frame(
        [(m, i, y, p) for m, i, y, p, _ in _INVARIANT_ROWS]
    )
    cal = _scored_frame(
        [(m, i, y, p) for m, i, y, _, p in _INVARIANT_ROWS]
    )
    report = tc.build_calibration_comparison_report(uncal, cal)
    serialized = json.dumps(report, sort_keys=True)
    restored = json.loads(serialized)
    assert restored == report

    assert set(report.keys()) == {
        "n_eval",
        "uncalibrated",
        "calibrated",
        "mean_rps_delta",
        "mean_log_loss_delta",
        "marginal_binary_accuracy_delta",
        "binary_side_agreement_rate",
        "argmax_category_agreement_rate",
        "decisions_invariant",
    }
    assert isinstance(report["n_eval"], int)
    for model_key in ("uncalibrated", "calibrated"):
        assert set(report[model_key].keys()) == {
            "mean_rps",
            "mean_log_loss",
            "marginal_binary_accuracy",
        }
        for value in report[model_key].values():
            assert math.isfinite(value)
    for delta in (
        "mean_rps_delta",
        "mean_log_loss_delta",
        "marginal_binary_accuracy_delta",
    ):
        assert math.isfinite(report[delta])
    for rate in ("binary_side_agreement_rate", "argmax_category_agreement_rate"):
        assert 0.0 <= report[rate] <= 1.0
    assert isinstance(report["decisions_invariant"], bool)


# --------------------------------------------------------------------------
# plan#14: real v1 end-to-end via the comparison CLI
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + both model artifacts not present "
    "(run materialize.py, train_ordinal_logit.py and "
    "train_temperature_scaling.py first)",
)
def test_real_v1_temperature_calibration_end_to_end():
    # The M24 finding on real data/v1: run the comparison CLI (which
    # scores the 35-map held-out test split with both the uncalibrated
    # ordinal registry factory and the ordinal_logit_temperature
    # factory, builds the comparison report, and writes
    # data/v1/temperature_calibration_report.json), reload the
    # artifact, and assert **internal consistency only** — decision B
    # explicitly forbids assuming the outcome ahead of time: deltas are
    # finite floats, decisions_invariant is a bool, agreement rates are
    # in [0, 1], n_eval is the known 35-map v1 test-split size, and the
    # report equals an independent in-process recomputation through the
    # registry factories (so a wiring bug in the CLI — e.g. swapping
    # the two tables — would be caught). The measured values are
    # printed for the record.
    import pandas as pd

    from drivers import evaluate
    from drivers.evaluate_temperature_calibration import main as cli_main
    from evaluation import harness

    rc = cli_main(["--version", "v1"])
    assert rc == 0

    artifact_path = Path("data/v1/temperature_calibration_report.json")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["n_eval"] == 35
    for model_key in ("uncalibrated", "calibrated"):
        for value in artifact[model_key].values():
            assert math.isfinite(value)
    for delta in (
        "mean_rps_delta",
        "mean_log_loss_delta",
        "marginal_binary_accuracy_delta",
    ):
        assert math.isfinite(artifact[delta])
    for rate in ("binary_side_agreement_rate", "argmax_category_agreement_rate"):
        assert 0.0 <= artifact[rate] <= 1.0
    assert isinstance(artifact["decisions_invariant"], bool)
    # Internal consistency of the deltas with the per-model metrics.
    for metric in ("mean_rps", "mean_log_loss", "marginal_binary_accuracy"):
        assert artifact[f"{metric}_delta"] == pytest.approx(
            artifact["calibrated"][metric] - artifact["uncalibrated"][metric]
        )

    # Independent in-process recomputation through the registry
    # factories (not through the CLI's own scored tables), so the CLI's
    # wiring is double-checked.
    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    held_out = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    uncalibrated_fn = evaluate.MODEL_REGISTRY["ordinal_logit"]("data", "v1")
    calibrated_fn = evaluate.MODEL_REGISTRY["ordinal_logit_temperature"](
        "data", "v1"
    )
    scored_uncalibrated = harness.score_held_out_maps(
        uncalibrated_fn, held_out, matches_df, maps_df
    )
    scored_calibrated = harness.score_held_out_maps(
        calibrated_fn, held_out, matches_df, maps_df
    )
    report = tc.build_calibration_comparison_report(
        scored_uncalibrated, scored_calibrated
    )
    assert report == artifact

    # The fitted T (for the summary line only — read from the
    # calibration artifact the training driver wrote).
    from models import temperature_scaling

    temp_model = temperature_scaling.from_dict(
        json.loads(
            Path("data/v1/temperature_scaling_model.json").read_text(
                encoding="utf-8"
            )
        )
    )
    print(
        "M24 temperature calibration on real v1 test split (n_eval=35): "
        f"temperature={temp_model.temperature!r} "
        f"mean_rps_delta={artifact['mean_rps_delta']!r} "
        f"mean_log_loss_delta={artifact['mean_log_loss_delta']!r} "
        f"marginal_binary_accuracy_delta="
        f"{artifact['marginal_binary_accuracy_delta']!r} "
        f"binary_side_agreement_rate="
        f"{artifact['binary_side_agreement_rate']!r} "
        f"argmax_category_agreement_rate="
        f"{artifact['argmax_category_agreement_rate']!r} "
        f"decisions_invariant={artifact['decisions_invariant']!r}"
    )
