"""Tests for the granularity-ablation comparison module (M22).

Covers ``marginalize_ordinal_probs``'s arithmetic on a hand-built
``(n, 4)`` fixture (the two summed columns checked against hand
computation, and rows still summing to ~1), ``build_ablation_report``'s
metric values against hand-computed ``mean_log_loss``/accuracy on tiny
fixtures with known answers, the verdict boundary in both directions
(one fixture that trips ``accuracy_gap >= 0.10`` alone, one that trips
``log_loss_gap >= 0.05`` alone, and one that trips neither), the two
gap signs (a fixture where the binary model is obviously better must
produce positive gaps, and vice versa), JSON-serializability of the
full report dict, the verbatim ``verdict_rule`` text, and the input
validation errors.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluation import granularity_ablation as ga
from tests._shared import _real_v1_available as _real_v1_tables_available


def _real_v1_available():
    """Report whether the real v1 tables and both model artifacts exist.

    The skip guard for the end-to-end ablation test: the parquet half
    is delegated to ``tests._shared._real_v1_available`` (the single
    shared home of the bare-table-name convention — all five
    ``data/v1/*.parquet`` files), and this module additionally requires
    both fitted model artifacts (``ordinal_logit_model.json`` and
    ``binary_logit_model.json``, i.e. both training drivers have been
    run) because the end-to-end ablation loads them.

    Returns:
        A bool: ``True`` iff all five ``data/v1/*.parquet`` files and
        both ``data/v1/*_logit_model.json`` artifacts exist.

    Raises:
        Nothing.
    """
    return _real_v1_tables_available() and all(
        Path(f"data/v1/{name}_logit_model.json").exists()
        for name in ("ordinal", "binary")
    )


def test_marginalize_ordinal_probs_sums_hand_built_fixture():
    # The two output columns must equal the hand-computed sums
    # (A-side = ordinals 0 + 1, B-side = ordinals 2 + 3) and every
    # output row must still sum to ~1 (no renormalization is performed,
    # but the inputs already sum to 1).
    rows_4way = np.array(
        [
            [0.2, 0.3, 0.1, 0.4],
            [0.5, 0.2, 0.2, 0.1],
            [0.0, 0.25, 0.35, 0.4],
        ]
    )
    binary = ga.marginalize_ordinal_probs(rows_4way)
    assert binary.shape == (3, 2)
    assert binary[:, 0] == pytest.approx([0.5, 0.7, 0.25])
    assert binary[:, 1] == pytest.approx([0.5, 0.3, 0.75])
    assert binary.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])


def test_marginalize_ordinal_probs_rejects_wrong_shape():
    # An (n, 3) or (n, 5) input would silently misalign the summed
    # categories; a wrong column count is a hard error.
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        ga.marginalize_ordinal_probs(np.ones((3, 3)))
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        ga.marginalize_ordinal_probs(np.ones((3, 5)))
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        ga.marginalize_ordinal_probs(np.ones(4))


def test_build_ablation_report_hand_computed_metrics():
    # A tiny fixture with a known answer: the binary model is nearly
    # perfect (p = 0.99 on the true category, so log loss is
    # -log(0.99) per row and accuracy is 1.0); the ordinal model is
    # uniform (log loss log 2 per row, accuracy 3/4 because the tie in
    # the argmax-collapsed side resolves toward A and the single
    # B-truth row is therefore wrong). All metric values are
    # hand-computed; exact 0.0/1.0 probabilities are deliberately
    # avoided because utils.scoring.log_loss hard-errors on a zero
    # probability on the true category (real model output is clipped
    # into [eps, 1-eps], so this never arises in production).
    truth = np.array([0, 1, 0, 0])
    binary_probs = np.array(
        [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.01]]
    )
    ordinal_probs = np.full((4, 2), 0.5)
    report = ga.build_ablation_report(
        binary_probs, ordinal_probs, truth, n_train_binary_model=209,
        n_train_ordinal_model=209,
    )
    assert report["n_eval"] == 4
    assert report["binary_logit"]["mean_log_loss"] == pytest.approx(
        -math.log(0.99)
    )
    assert report["binary_logit"]["accuracy"] == pytest.approx(1.0)
    assert report["binary_logit"]["n_train"] == 209
    assert report["ordinal_marginalized"]["mean_log_loss"] == pytest.approx(
        math.log(2.0)
    )
    assert report["ordinal_marginalized"]["accuracy"] == pytest.approx(0.75)
    assert report["ordinal_marginalized"]["n_train"] == 209
    assert report["accuracy_gap"] == pytest.approx(0.25)
    assert report["log_loss_gap"] == pytest.approx(
        math.log(2.0) + math.log(0.99)
    )
    assert report["granularity_verdict"] == "costs_accuracy"


def test_verdict_accuracy_gap_boundary_trips_alone():
    # The accuracy-gap-only fixture: the binary model and the ordinal
    # model are equally (un)confident on every row except that the
    # ordinal flips all five B-truth rows to side A (p_a = 0.51), so
    # accuracy_gap = 0.5 (>= 0.10, trips) while the per-row log-loss
    # gap on a flipped row is only ~0.04, keeping log_loss_gap
    # (0.020) below 0.05. Verdict must be "costs_accuracy".
    truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    binary_probs = np.array(
        [[0.51, 0.49] if not t else [0.49, 0.51] for t in truth]
    )
    ordinal_probs = np.full((10, 2), [0.51, 0.49])
    report = ga.build_ablation_report(
        binary_probs, ordinal_probs, truth, n_train_binary_model=209,
        n_train_ordinal_model=209,
    )
    assert report["binary_logit"]["accuracy"] == pytest.approx(1.0)
    assert report["ordinal_marginalized"]["accuracy"] == pytest.approx(0.5)
    assert report["accuracy_gap"] == pytest.approx(0.5)
    assert report["log_loss_gap"] == pytest.approx(
        0.5 * math.log(0.51 / 0.49), abs=1e-6
    )
    assert report["log_loss_gap"] < 0.05
    assert report["granularity_verdict"] == "costs_accuracy"


def test_verdict_log_loss_gap_boundary_trips_alone():
    # The log-loss-gap-only fixture: both models are always on the
    # correct side (accuracy_gap = 0, below 0.10) but the ordinal model
    # is less confident everywhere (p = 0.8 vs 0.9), so log_loss_gap =
    # log(0.9/0.8) ~ 0.118 (>= 0.05, trips). Verdict must be
    # "costs_accuracy".
    truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    binary_probs = np.array(
        [[0.9, 0.1] if not t else [0.1, 0.9] for t in truth]
    )
    ordinal_probs = np.array(
        [[0.8, 0.2] if not t else [0.2, 0.8] for t in truth]
    )
    report = ga.build_ablation_report(
        binary_probs, ordinal_probs, truth, n_train_binary_model=209,
        n_train_ordinal_model=209,
    )
    assert report["accuracy_gap"] == pytest.approx(0.0)
    assert report["binary_logit"]["accuracy"] == pytest.approx(1.0)
    assert report["ordinal_marginalized"]["accuracy"] == pytest.approx(1.0)
    assert report["log_loss_gap"] == pytest.approx(math.log(0.9 / 0.8))
    assert report["log_loss_gap"] >= 0.05
    assert report["granularity_verdict"] == "costs_accuracy"


def test_verdict_viable_when_neither_boundary_trips():
    # The neither-boundary fixture: the binary model is only slightly
    # more confident than the ordinal (p = 0.55 vs 0.53) and both are
    # always on the correct side, so accuracy_gap = 0 (< 0.10) and
    # log_loss_gap = log(0.55/0.53) ~ 0.037 (< 0.05). Verdict must be
    # "viable".
    truth = np.array([0, 0, 0, 0, 0])
    binary_probs = np.full((5, 2), [0.55, 0.45])
    ordinal_probs = np.full((5, 2), [0.53, 0.47])
    report = ga.build_ablation_report(
        binary_probs, ordinal_probs, truth, n_train_binary_model=209,
        n_train_ordinal_model=209,
    )
    assert report["accuracy_gap"] == pytest.approx(0.0)
    assert report["log_loss_gap"] == pytest.approx(math.log(0.55 / 0.53))
    assert report["log_loss_gap"] < 0.05
    assert report["granularity_verdict"] == "viable"


def test_gap_signs_positive_when_binary_better_and_vice_versa():
    # When the binary model is obviously better (0.99-confident vs
    # uniform), both gaps must be positive (binary more accurate AND
    # lower log loss); when it is obviously worse (uniform vs
    # 0.99-confident), both must be negative. The 0.99 (not exact 1.0)
    # confidence avoids utils.scoring's hard error on a zero
    # probability on the true category.
    truth = np.array([0, 1, 0, 0, 1, 0])
    confident = np.array(
        [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.99, 0.01]]
    )
    uniform = np.full((6, 2), 0.5)
    report_binary_better = ga.build_ablation_report(
        confident, uniform, truth, n_train_binary_model=10,
        n_train_ordinal_model=20,
    )
    assert report_binary_better["accuracy_gap"] > 0.0
    assert report_binary_better["log_loss_gap"] > 0.0
    assert report_binary_better["binary_logit"]["n_train"] == 10
    assert report_binary_better["ordinal_marginalized"]["n_train"] == 20

    report_binary_worse = ga.build_ablation_report(
        uniform, confident, truth, n_train_binary_model=10,
        n_train_ordinal_model=20,
    )
    assert report_binary_worse["accuracy_gap"] < 0.0
    assert report_binary_worse["log_loss_gap"] < 0.0
    assert report_binary_worse["granularity_verdict"] == "viable"


def test_report_json_serializable_and_verdict_rule_verbatim():
    # The full report dict must be directly json.dumps-able (the CLI
    # driver writes it with json.dumps), and the embedded verdict_rule
    # must match the module constant verbatim so the artifact is
    # self-documenting.
    truth = np.array([0, 1, 0, 0])
    binary_probs = np.array(
        [[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.01]]
    )
    ordinal_probs = np.full((4, 2), 0.5)
    report = ga.build_ablation_report(
        binary_probs, ordinal_probs, truth, n_train_binary_model=209,
        n_train_ordinal_model=209,
    )
    serialized = json.dumps(report)
    restored = json.loads(serialized)
    assert restored == report
    assert report["verdict_rule"] == ga._VERDICT_RULE
    assert ga._ACCURACY_GAP_THRESHOLD == 0.10
    assert ga._LOG_LOSS_GAP_THRESHOLD == 0.05
    assert report["granularity_verdict"] in ("costs_accuracy", "viable")


def test_build_ablation_report_rejects_mismatched_shapes():
    # Row-count mismatches between the two prediction arrays and the
    # label vector would silently pair the wrong rows; a wrong column
    # count or a non-1-D label vector is a hard error.
    truth = np.array([0, 1, 0])
    binary_probs = np.ones((3, 2)) * 0.5
    ordinal_probs = np.ones((4, 2)) * 0.5
    with pytest.raises(ValueError, match="row counts differ"):
        ga.build_ablation_report(
            binary_probs, ordinal_probs, truth, n_train_binary_model=1,
            n_train_ordinal_model=1,
        )
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        ga.build_ablation_report(
            np.ones((3, 3)), np.ones((3, 2)), truth, n_train_binary_model=1,
            n_train_ordinal_model=1,
        )
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        ga.build_ablation_report(
            np.ones((3, 2)), np.ones((3, 3)), truth, n_train_binary_model=1,
            n_train_ordinal_model=1,
        )
    with pytest.raises(ValueError, match="1-D label"):
        ga.build_ablation_report(
            binary_probs, np.ones((3, 2)), truth.reshape(3, 1),
            n_train_binary_model=1, n_train_ordinal_model=1,
        )


def test_build_ablation_report_rejects_invalid_labels():
    # A true label outside {0, 1} is not a valid category index for a
    # 2-category problem and must fail loudly (propagated from
    # utils.scoring's validation).
    truth = np.array([0, 1, 2])
    binary_probs = np.ones((3, 2)) * 0.5
    with pytest.raises(ValueError, match="in \\[0, 2\\)"):
        ga.build_ablation_report(
            binary_probs, binary_probs, truth, n_train_binary_model=1,
            n_train_ordinal_model=1,
        )


# --------------------------------------------------------------------------
# plan#8: real v1 end-to-end via the ablation CLI
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_v1_available(),
    reason="materialised v1 dataset + trained model artifacts not present "
    "(run materialize.py and both training drivers first)",
)
def test_real_v1_ablation_end_to_end():
    # The M22 finding on real data/v1: run the ablation CLI (which
    # assembles the shared 35-map test matrix once, scores both models
    # directly, marginalizes the ordinal four-way output, and writes
    # data/v1/granularity_ablation_report.json), reload the artifact,
    # assert n_eval is the known 35-map v1 test-split size, and
    # independently recompute the report in-process (not just read the
    # CLI's artifact) so the numbers are double-checked. Prints and
    # records the actual accuracy_gap / log_loss_gap /
    # granularity_verdict — the roadmap's M22 viability-gate finding.
    import pandas as pd

    from drivers import ablate_granularity, training_data
    from models import binary_logit, ordinal_logit

    rc = ablate_granularity.main(["--version", "v1"])
    assert rc == 0

    artifact = json.loads(
        Path("data/v1/granularity_ablation_report.json").read_text(encoding="utf-8")
    )
    assert artifact["n_eval"] == 35
    assert artifact["granularity_verdict"] in ("costs_accuracy", "viable")
    assert artifact["verdict_rule"] == ga._VERDICT_RULE
    for model_key in ("binary_logit", "ordinal_marginalized"):
        assert math.isfinite(artifact[model_key]["mean_log_loss"])
        assert 0.0 <= artifact[model_key]["accuracy"] <= 1.0

    matches_df = pd.read_parquet("data/v1/matches.parquet")
    maps_df = pd.read_parquet("data/v1/maps.parquet")
    labels_df = pd.read_parquet("data/v1/labels.parquet")
    splits_df = pd.read_parquet("data/v1/splits.parquet")
    player_map_stats_df = pd.read_parquet("data/v1/player_map_stats.parquet")
    X_test, y_test_ordinal = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="test",
    )
    ordinal_model = ordinal_logit.from_dict(
        json.loads(
            Path("data/v1/ordinal_logit_model.json").read_text(encoding="utf-8")
        )
    )
    binary_model = binary_logit.from_dict(
        json.loads(
            Path("data/v1/binary_logit_model.json").read_text(encoding="utf-8")
        )
    )
    ordinal_probs_4way = np.asarray(
        [
            ordinal_logit.predict_proba(X_test[i], ordinal_model)
            for i in range(X_test.shape[0])
        ],
        dtype=float,
    )
    ordinal_marginal = ga.marginalize_ordinal_probs(ordinal_probs_4way)
    binary_probs = np.asarray(
        [
            binary_logit.predict_proba(X_test[i], binary_model)
            for i in range(X_test.shape[0])
        ],
        dtype=float,
    )
    truth = (y_test_ordinal >= 2).astype(int)
    report = ga.build_ablation_report(
        binary_probs,
        ordinal_marginal,
        truth,
        n_train_binary_model=binary_model.n_train,
        n_train_ordinal_model=ordinal_model.n_train,
    )
    for key in (
        "granularity_verdict",
        "accuracy_gap",
        "log_loss_gap",
        "n_eval",
    ):
        assert report[key] == artifact[key]
    assert report["binary_logit"] == artifact["binary_logit"]
    assert report["ordinal_marginalized"] == artifact["ordinal_marginalized"]

    print(
        "M22 granularity ablation on real v1 test split (n_eval=35): "
        f"binary_logit acc={report['binary_logit']['accuracy']!r} "
        f"ll={report['binary_logit']['mean_log_loss']!r} "
        f"ordinal_marginalized acc={report['ordinal_marginalized']['accuracy']!r} "
        f"ll={report['ordinal_marginalized']['mean_log_loss']!r} "
        f"accuracy_gap={report['accuracy_gap']!r} "
        f"log_loss_gap={report['log_loss_gap']!r} "
        f"granularity_verdict={report['granularity_verdict']!r}"
    )
