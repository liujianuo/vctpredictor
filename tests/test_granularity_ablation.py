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

import numpy as np
import pytest

from evaluation import granularity_ablation as ga


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
