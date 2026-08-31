"""Calibration-comparison report for the M24 temperature-scaled ordinal model.

Builds the comparison report that answers the M24 question on the M19
held-out test split: given the *same* held-out rows scored twice — once
by the uncalibrated M20 ordinal model, once by the same model with the
fitted temperature ``T`` applied — what changed? The report pairs the
two already-scored tables (both produced by
:func:`evaluation.harness.score_held_out_maps` on the identical
held-out rows) and reports, per metric, the calibrated-minus-
uncalibrated delta, plus two agreement rates measuring whether the
*decisions* (binary side call, 4-way argmax category) survived the
scaling unchanged.

**Decision B (recorded verbatim from the M24 plan, do not re-derive) —
invariance is *measured*, not assumed.** The task text suggests
"temperature scaling is monotone, so accuracy/RPS-argmax are
invariant." This is **not generally true for this model** and must not
be asserted as fact: ``eta / T`` is monotone in ``eta`` for fixed
``T > 0``, but the marginal A/B decision boundary is
``theta_2 + eta/T > 0``, i.e. ``eta > -theta_2 * T`` — a boundary that
itself *moves* with ``T`` (unless ``theta_2 == 0``, which it is not:
the M20 artifact's ``thresholds[1]`` is ``-0.0671``). So the binary
side call, the 4-way argmax category, and therefore
``marginal_binary_accuracy``/RPS-argmax *can* change with ``T``, in
principle. Log loss and calibration are what temperature scaling is
*designed* to move; whether accuracy/argmax happen to stay fixed on the
actual v1 test split is an empirical question this module measures and
states (as the report's ``decisions_invariant`` field), not an
assumption.

Scope / conventions (recorded here, do not re-derive later):

- **Pure and dependency-light.** This module does no file I/O, has no
  CLI / ``argparse`` entry point, and never touches ``drivers/``. It
  takes two already-scored DataFrames in and returns a dict out —
  matching ``evaluation.harness`` / ``evaluation.granularity_ablation``'s
  own convention. **It deliberately does NOT import
  ``evaluation.harness``**: there is no shared-module exception at the
  ``evaluation/`` rung — the module-boundary test forbids ANY sibling
  ``evaluation/`` import unconditionally (the same rule
  ``evaluation/granularity_ablation.py`` documents and obeys). The
  three headline metrics are instead computed through the *same shared
  ``utils.scoring`` batch functions* :func:`evaluation.harness.build_evaluation_report`
  calls (:func:`utils.scoring.mean_rps` / ``mean_log_loss`` /
  ``mean_marginal_binary_accuracy`` on the prediction columns and the
  true ordinals), so the numbers this report carries are bit-identical
  to the harness report's headline numbers — the "no second ad hoc mean
  implementation" precedent, satisfied by reusing the metric
  implementations rather than a sibling module import. The driver
  (:mod:`drivers.evaluate_temperature_calibration`) owns all table
  loading, model loading and scoring; this module receives only the
  finished scored tables.
- **Row-alignment contract.** The two scored tables must describe the
  *same* held-out rows *in the same order* (both are the output of
  :func:`evaluation.harness.score_held_out_maps` called on the one
  held-out table from :func:`evaluation.harness.build_held_out_maps`,
  which preserves row order). The function validates this by comparing
  the ``(match_id, map_index)`` pairs positionally and raises
  ``ValueError`` on any mismatch — a misaligned comparison would
  silently pair two different maps' predictions and corrupt every
  delta.
- **Agreement-rate definitions (decided here, do not re-derive).**
  ``binary_side_agreement_rate`` is the fraction of rows whose marginal
  side call ``(P0 + P1) > 0.5`` (side A iff the summed probability of
  the two A-side categories strictly exceeds 0.5) is *identical*
  between the two models; ``argmax_category_agreement_rate`` is the
  fraction of rows whose ``argmax`` over the four prediction columns
  (ties resolved by numpy's first-maximum convention, identically for
  both tables) is identical. ``decisions_invariant`` is the conjunction
  — ``True`` exactly when both rates are ``1.0`` — documenting the
  *measured* outcome on this run, not an assumed one.
- **Delta signs.** All three deltas are ``calibrated - uncalibrated``,
  so a *negative* ``mean_log_loss_delta`` / ``mean_rps_delta`` means
  temperature scaling improved (lowered) that metric, and a *positive*
  ``marginal_binary_accuracy_delta`` means it improved accuracy. The
  report asserts no direction — decision B forbids assuming the
  outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import scoring

# The four predicted-probability columns of a
# ``score_held_out_maps``-shaped table, in OUTCOME_LABELS order —
# kept in sync with, deliberately *not* imported from,
# ``evaluation.harness.PREDICTION_COLUMNS`` (no sibling-evaluation
# import; see the module docstring).
_PREDICTION_COLUMNS = (
    "p_a_regulation",
    "p_a_ot",
    "p_b_ot",
    "p_b_regulation",
)

# The identifying columns used for the row-alignment validation.
_ID_COLUMNS = ("match_id", "map_index")


def build_calibration_comparison_report(
    scored_uncalibrated_df: pd.DataFrame,
    scored_calibrated_df: pd.DataFrame,
) -> dict:
    """Compare the uncalibrated and temperature-scaled scored tables.

    Takes the two ``evaluation.harness.score_held_out_maps``-shaped
    tables produced by scoring the *identical* held-out rows with the
    uncalibrated M20 ordinal model and with its temperature-scaled
    variant, validates they are row-aligned (same ``(match_id,
    map_index)`` pairs in the same order), and returns the comparison
    report: the three headline metrics (``mean_rps``, ``mean_log_loss``,
    ``marginal_binary_accuracy``) for each model computed through the
    shared ``utils.scoring`` batch functions (the exact computation
    path :func:`evaluation.harness.build_evaluation_report` uses, so no
    ad hoc mean implementation exists), the calibrated-minus-
    uncalibrated delta of each, the two decision-agreement rates
    (:data:`_PREDICTION_COLUMNS` side call ``(P0 + P1) > 0.5`` and
    4-way ``argmax`` — see the module docstring for the definitions),
    and the ``decisions_invariant`` bool documenting the *measured*
    outcome (decision B: measured, not assumed).

    Args:
        scored_uncalibrated_df: The scored table from
            :func:`evaluation.harness.score_held_out_maps` for the
            uncalibrated ordinal model (needs :data:`_PREDICTION_COLUMNS`,
            ``outcome_ordinal``, ``match_id`` and ``map_index``; the
            per-row ``rps``/``log_loss``/``marginal_correct`` columns
            are not read).
        scored_calibrated_df: The same held-out rows scored with the
            temperature-scaled variant, same column requirements.

    Returns:
        A dict with keys ``n_eval`` (int), ``uncalibrated`` and
        ``calibrated`` (each ``{"mean_rps": float, "mean_log_loss":
        float, "marginal_binary_accuracy": float}``),
        ``mean_rps_delta`` / ``mean_log_loss_delta`` /
        ``marginal_binary_accuracy_delta`` (each ``calibrated -
        uncalibrated``, so negative log-loss/RPS deltas mean the
        calibrated model improved), ``binary_side_agreement_rate``
        (float in ``[0, 1]``), ``argmax_category_agreement_rate``
        (float in ``[0, 1]``) and ``decisions_invariant`` (bool =
        both agreement rates exactly ``1.0``). Every value is a plain
        str/int/float/bool so the dict is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If the two tables have different row counts or
            differ in any ``(match_id, map_index)`` pair at the same
            position (the row-alignment contract — a misalignment would
            silently pair two different maps' predictions); or if either
            table is empty / a prediction row fails the metric
            validation (propagated from the ``utils.scoring`` batch
            functions, e.g. ``log_loss`` on a zero-probability true
            category).
        KeyError: If either table lacks a prediction column,
            ``outcome_ordinal``, ``match_id`` or ``map_index``
            (propagated from pandas column indexing).
    """
    if len(scored_uncalibrated_df) != len(scored_calibrated_df):
        raise ValueError(
            f"scored tables have different row counts: uncalibrated "
            f"{len(scored_uncalibrated_df)} vs calibrated "
            f"{len(scored_calibrated_df)}; they must describe the same "
            "held-out rows"
        )
    uncal_ids = scored_uncalibrated_df[list(_ID_COLUMNS)].to_numpy()
    cal_ids = scored_calibrated_df[list(_ID_COLUMNS)].to_numpy()
    mismatch_mask = uncal_ids != cal_ids
    if mismatch_mask.any():
        idx = int(np.argmax(mismatch_mask.any(axis=1)))
        raise ValueError(
            "scored tables are not row-aligned: the held-out rows "
            f"differ at position {idx} "
            f"(uncalibrated {tuple(uncal_ids[idx])!r} vs calibrated "
            f"{tuple(cal_ids[idx])!r}); score both models on the "
            "identical build_held_out_maps table"
        )

    def _metrics(scored_df: pd.DataFrame) -> dict:
        """Compute the three headline metrics for one scored table.

        Reuses the exact computation path of
        :func:`evaluation.harness.build_evaluation_report`: the
        prediction columns and true ordinals are handed to the shared
        ``utils.scoring`` batch functions, so there is no second, ad
        hoc mean implementation to drift from the harness numbers.

        Args:
            scored_df: One ``score_held_out_maps``-shaped table.

        Returns:
            A dict with ``mean_rps`` / ``mean_log_loss`` /
            ``marginal_binary_accuracy`` floats.

        Raises:
            ValueError / KeyError: Propagated from the ``utils.scoring``
                batch functions / pandas column indexing exactly as
                documented on :func:`build_calibration_comparison_report`.
        """
        prob_rows = scored_df[list(_PREDICTION_COLUMNS)].to_numpy()
        true_indices = scored_df["outcome_ordinal"].to_numpy()
        return {
            "mean_rps": scoring.mean_rps(prob_rows, true_indices),
            "mean_log_loss": scoring.mean_log_loss(prob_rows, true_indices),
            "marginal_binary_accuracy": scoring.mean_marginal_binary_accuracy(
                prob_rows, true_indices
            ),
        }

    uncal_metrics = _metrics(scored_uncalibrated_df)
    cal_metrics = _metrics(scored_calibrated_df)

    # Decision-agreement rates: the marginal side call (P0 + P1 > 0.5)
    # and the 4-way argmax category, compared row by row between the
    # two tables. Both use numpy's own comparison/argmax conventions
    # identically for the two tables.
    uncal_probs = scored_uncalibrated_df[list(_PREDICTION_COLUMNS)].to_numpy()
    cal_probs = scored_calibrated_df[list(_PREDICTION_COLUMNS)].to_numpy()
    uncal_side_a = uncal_probs[:, 0] + uncal_probs[:, 1] > 0.5
    cal_side_a = cal_probs[:, 0] + cal_probs[:, 1] > 0.5
    binary_side_agreement_rate = float(
        (uncal_side_a == cal_side_a).mean()
    )
    argmax_category_agreement_rate = float(
        (np.argmax(uncal_probs, axis=1) == np.argmax(cal_probs, axis=1)).mean()
    )
    decisions_invariant = (
        binary_side_agreement_rate == 1.0
        and argmax_category_agreement_rate == 1.0
    )

    return {
        "n_eval": len(scored_uncalibrated_df),
        "uncalibrated": uncal_metrics,
        "calibrated": cal_metrics,
        "mean_rps_delta": cal_metrics["mean_rps"] - uncal_metrics["mean_rps"],
        "mean_log_loss_delta": (
            cal_metrics["mean_log_loss"] - uncal_metrics["mean_log_loss"]
        ),
        "marginal_binary_accuracy_delta": (
            cal_metrics["marginal_binary_accuracy"]
            - uncal_metrics["marginal_binary_accuracy"]
        ),
        "binary_side_agreement_rate": binary_side_agreement_rate,
        "argmax_category_agreement_rate": argmax_category_agreement_rate,
        "decisions_invariant": decisions_invariant,
    }
