"""Command-line granularity ablation — the M22 viability gate (roadmap M22).

Thin command-line wrapper around :mod:`evaluation.granularity_ablation`,
which owns the pure comparison logic (marginalization, binary metrics,
verdict). This module adds only the CLI/IO glue: argument parsing
(:func:`parse_args`), loading the five input tables (reusing the four
``load_*_table`` helpers from :mod:`drivers.evaluate` plus that
module's :func:`drivers.evaluate.load_player_map_stats_table`),
assembling the *test*-split design matrix via
:func:`drivers.training_data.assemble_design_matrix` (the identical
11-feature matrix both M20's ordinal logit and M22's binary logit were
trained on, restricted to ``split="test"`` — the same 35-map v1
held-out set the M19 harness scores), loading the two fitted model
artifacts (``ordinal_logit_model.json`` and
``binary_logit_model.json``), scoring every test row with both models'
:func:`predict_proba` directly, marginalizing the ordinal model's
four-way output down to the binary target, and writing the comparison
report artifact ``data/<version>/granularity_ablation_report.json``.

**Prerequisite:** both training drivers must already have been run for
the requested version (``drivers/train_ordinal_logit.py`` for the
M20 arm, ``drivers/train_binary_logit.py`` for the M22 arm). This
script does not orchestrate training itself (matching the
one-script-one-job convention): if either model artifact is missing,
the ``FileNotFoundError`` propagates unchanged as a clear "run the
training drivers first" signal.

**Design notes (see the plan's Design Decision C for the full
reasoning):**

- This driver deliberately does **not** go through
  ``drivers/evaluate.py``'s ``MODEL_REGISTRY``: the binary model has no
  honest four-way output (its natural prediction is a 2-vector
  ``(p_a, p_b)``), and the registry's ``harness.ModelFn`` contract is
  fixed at a 4-vector in ``OUTCOME_LABELS`` order. Forcing it through
  would mean inventing a fake OT/regulation split with no basis.
- ``predict_proba`` is called **directly on the raw feature rows
  already assembled in step 2** — not via ``make_model_fn``'s
  6-argument closure, since ``X_test`` is already built and re-deriving
  it per row through the closure would silently redo the exact same
  feature computation a second time for no benefit.
- The true labels are derived as ``true_binary_labels =
  (y_test_ordinal >= 2).astype(int)`` — the **category-index
  convention ``0`` = "A wins", ``1`` = "B wins"** that
  :func:`evaluation.granularity_ablation.build_ablation_report` /
  ``utils.scoring`` require (index 0 is side A for ``k = 2``). This is
  the complement of the model-target conversion ``(y_ordinal <= 1)``
  used in the training driver (there ``1`` = "A wins"), and using the
  training-side formula here would invert every per-row log-loss
  reading (``mean_log_loss`` reads ``probs[true_index]``).

Artifact written per run (scoped by dataset version):

- ``data/<version>/granularity_ablation_report.json`` — the
  :func:`evaluation.granularity_ablation.build_ablation_report` dict,
  written with ``json.dumps(..., indent=2, sort_keys=True)`` plus a
  trailing newline (the same serialization convention as every other
  artifact in this repo).

Exit codes:

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from drivers import evaluate, training_data
from evaluation import granularity_ablation
from models import binary_logit, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the ablate_granularity.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with two attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``) and ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``). Together
        they locate the five input tables, the two model artifacts
        (``ordinal_logit_model.json`` / ``binary_logit_model.json``)
        and the output artifact
        ``<output_dir>/<version>/granularity_ablation_report.json``.
        There are deliberately no other flags: the comparison uses the
        documented defaults throughout, matching the other drivers'
        no-flags precedent.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M22 granularity ablation: score the fitted binary "
            "logit and the marginalized ordinal logit on the M10 test "
            "split, compare binary log loss / accuracy, and write "
            "granularity_ablation_report.json."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="input/output subdirectory name under --output-dir (default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "parent directory the version subdirectory lives under "
            "(default: data)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the granularity ablation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The five input tables are loaded for the requested version
    (matches/maps/labels/splits via the ``drivers.evaluate`` helpers,
    plus ``player_map_stats`` via
    :func:`drivers.evaluate.load_player_map_stats_table`), the test
    design matrix is assembled *once*
    (:func:`drivers.training_data.assemble_design_matrix` with
    ``split="test"`` — the identical shared 11-feature matrix both
    models score against; it is not assembled a second time), both
    fitted model artifacts are loaded via ``json.load`` + the two
    ``from_dict`` functions, every test row is scored by both models'
    :func:`predict_proba` (the ordinal model's 4-vector, the binary
    model's 2-vector), the four-vectors are marginalized down to the
    binary target (:func:`evaluation.granularity_ablation.marginalize_ordinal_probs`),
    the true binary labels are derived in ``utils.scoring``'s
    category-index convention (``0`` = "A wins", ``1`` = "B wins" —
    see the module docstring's design notes), the full report is built
    (:func:`evaluation.granularity_ablation.build_ablation_report`),
    the artifact is written as
    ``<output_dir>/<version>/granularity_ablation_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary (``granularity_verdict``,
    ``accuracy_gap``, ``log_loss_gap``, both models'
    ``accuracy``/``mean_log_loss``) is logged.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the five input tables, or either
            fitted model artifact (``ordinal_logit_model.json`` /
            ``binary_logit_model.json``), does not exist for the
            requested version — the model artifacts propagate as-is
            from ``json.load`` as a clear "run the training drivers
            first" signal; the ablation never retrains anything.
        ValueError: If the test split is empty, a label is invalid, a
            feature computation fails, or the report validation fails
            (propagated from :func:`drivers.training_data.assemble_design_matrix`
            / the two ``predict_proba`` calls /
            :func:`evaluation.granularity_ablation.build_ablation_report`).
        KeyError: If any input table or model artifact lacks a required
            key/column (propagated from the feature modules / harness /
            the two ``from_dict`` functions).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The shared test design matrix, assembled once: both models consume
    # the identical 11-feature vector, so there is exactly one join and
    # one feature computation per test row (do not call
    # assemble_design_matrix a second time).
    X_test, y_test_ordinal = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="test",
    )

    ordinal_path = output_dir / args.version / "ordinal_logit_model.json"
    with open(ordinal_path, encoding="utf-8") as handle:
        ordinal_model = ordinal_logit.from_dict(json.load(handle))
    binary_path = output_dir / args.version / "binary_logit_model.json"
    with open(binary_path, encoding="utf-8") as handle:
        binary_model = binary_logit.from_dict(json.load(handle))

    # Score each test row with both models directly on the already-
    # assembled raw feature rows (not through make_model_fn's closure,
    # which would re-derive the exact same feature vector a second
    # time).
    ordinal_rows_4way: list[np.ndarray] = []
    binary_rows_2way: list[np.ndarray] = []
    for i in range(X_test.shape[0]):
        ordinal_rows_4way.append(
            np.asarray(ordinal_logit.predict_proba(X_test[i], ordinal_model))
        )
        binary_rows_2way.append(
            np.asarray(binary_logit.predict_proba(X_test[i], binary_model))
        )
    ordinal_probs_4way = np.asarray(ordinal_rows_4way, dtype=float)
    ordinal_marginal_probs = granularity_ablation.marginalize_ordinal_probs(
        ordinal_probs_4way
    )
    binary_probs = np.asarray(binary_rows_2way, dtype=float)

    # The true labels in utils.scoring's category-index convention:
    # 0 = "A wins" (ordinals 0/1), 1 = "B wins" (ordinals 2/3). This is
    # the complement of the training driver's model-target conversion
    # (y_ordinal <= 1) — see the module docstring's design notes.
    true_binary_labels = (y_test_ordinal >= 2).astype(int)

    report = granularity_ablation.build_ablation_report(
        binary_probs,
        ordinal_marginal_probs,
        true_binary_labels,
        n_train_binary_model=binary_model.n_train,
        n_train_ordinal_model=ordinal_model.n_train,
    )

    artifact_path = (
        output_dir / args.version / "granularity_ablation_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "granularity ablation on %d held-out maps (%s/%s): "
        "verdict=%s accuracy_gap=%.4f log_loss_gap=%.4f | "
        "binary_logit acc=%.4f ll=%.4f | ordinal_marginalized acc=%.4f ll=%.4f",
        report["n_eval"],
        output_dir,
        args.version,
        report["granularity_verdict"],
        report["accuracy_gap"],
        report["log_loss_gap"],
        report["binary_logit"]["accuracy"],
        report["binary_logit"]["mean_log_loss"],
        report["ordinal_marginalized"]["accuracy"],
        report["ordinal_marginalized"]["mean_log_loss"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
