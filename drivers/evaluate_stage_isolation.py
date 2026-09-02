"""Command-line stage-isolation evaluation — the M34 headline report driver.

Thin command-line wrapper around :mod:`evaluation.stage_isolation`
(the M34 stage-isolation core), mirroring the
``drivers/evaluate_series.py`` shape exactly: ``parse_args`` -> load
tables/artifacts -> build the Arm-A table -> sample + score Arm B ->
build the two-arm report -> write the JSON artifact -> log a one-line
summary. This module adds only the CLI/IO glue and the model wiring —
all comparison logic is delegated to
:func:`evaluation.stage_isolation.build_stage_isolation_report`.

The two arms are scored over the *identical* held-out ``split="test"``
map-position table (built once via
:func:`evaluation.stage_isolation.build_actual_played_maps` — the
row-alignment contract
:func:`evaluation.stage_isolation.build_stage_isolation_report`
validates):

- **``actual_played_maps`` (Arm A)** — Stage 2 (the fitted M20
  ordinal-logit four-way map model,
  ``models.ordinal_logit.make_model_fn`` on the fitted
  ``ordinal_logit_model.json`` + ``player_map_stats.parquet``) scored
  on each held-out map position's *actual* ``map_name`` against its
  true ``outcome_ordinal``.
- **``m29_predicted_maps`` (Arm B)** — the same Stage 2 scored on the
  maps M29's ancestral veto sampler (driven by the fitted M27/M28
  conditional-logit ban/pick predictors, ``make_veto_step_predictor_fn``
  on their artifacts — the identical Stage-1 wiring
  ``drivers/evaluate_series.py`` uses) would have predicted at those
  same positions, truncated to the ``n_played`` positions that
  actually happened and weighted by each sample's normalized
  ``sequence_probability``.

Only the map identity queried differs between the arms — Stage 2's
model and inputs, and the true outcomes, are identical — so the gap
(Arm B minus Arm A per metric) is the cost of Stage 1 error
compounding into Stage 2, isolated from any error in Stage 2 itself.

**The ``n_samples`` default is a measured wall-clock choice, not a
guess.** This task samples per **map position** (not per whole series
like M33b), so the per-sample cost differs from M33b's measured
~1.5-1.8 s/sequence. BUILD timed a real run on the v1 fitted models
(see the ``tasks/037`` BUILD status note for the measured numbers) and
chose :data:`DEFAULT_N_SAMPLES` so a full real-v1 run (35 held-out
map positions across the 15 Bo3 test-split matches) finishes in a
reasonable single CLI invocation (~3.5 minutes).

**Prerequisite artifacts.** This driver depends on
``drivers/train_ordinal_logit.py``, ``drivers/train_conditional_logit_ban.py``
and ``drivers/train_conditional_logit_pick.py`` having already been
run for the requested version (mirroring ``drivers/evaluate_series.py``).
A missing artifact raises ``FileNotFoundError`` unchanged — no silent
fallback, the same "run the training driver first" doctrine every
existing ``evaluate_*.py`` driver follows.

Artifact written per run (scoped by dataset version):

- ``data/<version>/stage_isolation_report.json`` — the
  :func:`evaluation.stage_isolation.build_stage_isolation_report`
  dict (keys ``"actual_played_maps"``, ``"m29_predicted_maps"``,
  ``"gap"``) plus the provenance keys ``"n_samples"`` and ``"seed"``
  (so the report is self-describing and reproducible without
  cross-referencing the invocation command), written with
  ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
  newline (the repo-wide convention).

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

from drivers import evaluate
from evaluation import stage_isolation
from models import conditional_logit_ban, conditional_logit_pick, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The default number of M29 veto walks sampled per held-out match by
# the Arm-B sampler, and the default RNG seed. Both are documented
# choices, not constants-of-convenience: DEFAULT_N_SAMPLES is the
# measured wall-clock compromise described in the module docstring (a
# real v1 run at this setting lands around 3.5 minutes for the full
# 35-position Bo3 test split), and DEFAULT_SEED matches the repo's
# "current year" seed convention already used by tasks/035's real-data
# smoke test and drivers/evaluate_series.py.
DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 2026


def _load_fitted_models(
    output_dir: Path, version: str
) -> tuple[object, object, object]:
    """Load the three fitted model artifacts this driver needs.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json``, the fitted M27 conditional-logit ban
    model from ``conditional_logit_ban_model.json``, and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the exact artifact-loading
    logic ``drivers/evaluate_series.py::_load_fitted_models`` already
    provides. ``drivers/`` modules may import ``evaluation/`` and
    ``models/`` freely (the module-boundary rule that forced
    ``evaluation/stage_isolation.py`` to stay import-free does not
    apply here), so this driver reuses the same shape as a local copy:
    the triplicated ``json.load`` + ``from_dict`` boilerplate lives in
    exactly this one helper. The three ``from_dict`` calls are
    deliberately independent of each other and of the four input
    tables, so a missing artifact fails fast with the standard "run
    the training driver first" signal.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ordinal_model, ban_model, pick_model)`` tuple of the three
        deserialized fitted models, in the order the driver wires them
        (the ordinal model feeds ``map_model_fn``; the ban/pick models
        feed the ``predictor_fn_by_action`` dict).

    Raises:
        FileNotFoundError: If any of the three model artifacts does not
            exist for the requested version (i.e. the corresponding
            training driver has not been run for it) — propagated
            unchanged from ``open()`` as a clear "run the training
            driver first" signal, never wrapped or silently skipped.
        KeyError: If an artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        ValueError: If an artifact's shapes are inconsistent (propagated
            from the ``from_dict`` calls).
    """
    ordinal_model = ordinal_logit.from_dict(
        json.loads(
            (
                output_dir / version / "ordinal_logit_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    ban_model = conditional_logit_ban.from_dict(
        json.loads(
            (
                output_dir
                / version
                / "conditional_logit_ban_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    pick_model = conditional_logit_pick.from_dict(
        json.loads(
            (
                output_dir
                / version
                / "conditional_logit_pick_model.json"
            ).read_text(encoding="utf-8")
        )
    )
    return ordinal_model, ban_model, pick_model


def _summarize(report: dict) -> str:
    """Format the two arms' headline numbers and the gap for the summary line.

    Composes the one-line summary segment: both arms'
    ``mean_rps`` / ``mean_log_loss`` / ``marginal_binary_accuracy``
    plus the three ``gap`` fields, all read from the
    :func:`evaluation.stage_isolation.build_stage_isolation_report`
    dict (whose ``"gap"`` values are always present — a two-arm stage
    isolation report never omits a metric, unlike the series report's
    per-``best_of`` groups).

    Args:
        report: The ``build_stage_isolation_report`` dict (with the
            ``"actual_played_maps"`` / ``"m29_predicted_maps"`` arm
            blocks and the ``"gap"`` block).

    Returns:
        A single-line string like ``"actual_played_maps
        mean_rps=0.670438 mean_log_loss=1.342904
        marginal_binary_accuracy=0.6000 | m29_predicted_maps
        mean_rps=... | gap mean_rps=+0.012345 mean_log_loss=...
        marginal_binary_accuracy=..."`` — the two arm blocks joined by
        ``" | "`` with the gap segment appended.

    Raises:
        KeyError: If ``report`` lacks any of the arm/gap keys
            (propagated from dict indexing — the report builder always
            emits all three blocks).
    """
    actual = report["actual_played_maps"]
    predicted = report["m29_predicted_maps"]
    gap = report["gap"]
    return (
        f"actual_played_maps mean_rps={actual['mean_rps']:.6f} "
        f"mean_log_loss={actual['mean_log_loss']:.6f} "
        f"marginal_binary_accuracy="
        f"{actual['marginal_binary_accuracy']:.4f}"
        f" | m29_predicted_maps mean_rps={predicted['mean_rps']:.6f} "
        f"mean_log_loss={predicted['mean_log_loss']:.6f} "
        f"marginal_binary_accuracy="
        f"{predicted['marginal_binary_accuracy']:.4f}"
        f" | gap mean_rps={gap['mean_rps_gap']:+.6f} "
        f"mean_log_loss={gap['mean_log_loss_gap']:+.6f} "
        f"marginal_binary_accuracy="
        f"{gap['marginal_binary_accuracy_gap']:+.4f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_stage_isolation.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with four attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), and the
        two reproducibility knobs: ``n_samples`` (``int``, the M29
        walks sampled per held-out match by the Arm-B sampler, default
        :data:`DEFAULT_N_SAMPLES` — the measured wall-clock choice
        documented in the module docstring) and ``seed`` (``int``, the
        ``numpy.random.default_rng`` seed for the whole run, default
        :data:`DEFAULT_SEED`). Together they locate the five input
        tables, the three fitted model artifacts, and the output
        artifact ``<output_dir>/<version>/stage_isolation_report.json``.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-samples``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M34 stage-isolation evaluation: score the fitted "
            "M20 ordinal-logit four-way map model on the actually-"
            "played held-out test-split maps (Arm A) vs on the maps "
            "the M29 ancestral sampler predicts at the same positions "
            "(Arm B, driven by the fitted M27/M28 conditional-logit "
            "ban/pick predictors, weighted by sequence probability), "
            "compare mean RPS / mean log loss / marginal binary "
            "accuracy, and write stage_isolation_report.json with the "
            "gap = Arm B - Arm A."
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
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per held-out match by the "
            "Arm-B sampler (default: "
            f"{DEFAULT_N_SAMPLES} — a measured wall-clock compromise "
            "on real v1 fitted models; raise it for a smoother "
            "aggregate, lower it for a faster run)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for numpy.random.default_rng, giving the whole run "
            f"byte-identical reproducibility (default: {DEFAULT_SEED})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage-isolation evaluation end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The three fitted model artifacts are loaded first
    (:func:`_load_fitted_models` — a missing artifact raises
    ``FileNotFoundError`` as the "run the training drivers first"
    signal), the five input tables are loaded for the requested version
    (matches/maps/splits/player_map_stats via the ``drivers.evaluate``
    helpers, plus the labels table), the pluggable callables are wired
    (``map_model_fn`` = :func:`models.ordinal_logit.make_model_fn`
    over the fitted ordinal model + ``player_map_stats.parquet`;
    ``predictor_fn_by_action`` = the ban/pick conditional-logit
    ``make_veto_step_predictor_fn`` closures — the identical wiring
    ``drivers/evaluate_series.py`` uses), a single
    ``numpy.random.default_rng(args.seed)`` is constructed and consumed
    sequentially by the Arm-B sampler across every held-out match in
    the Arm-A table's first-appearance order (so a fixed seed
    reproduces the whole run byte-identically), the Arm-A table is
    built once
    (:func:`evaluation.stage_isolation.build_actual_played_maps`), Arm
    A is scored
    (:func:`evaluation.stage_isolation.score_actual_played_maps`), Arm
    B's per-match predicted map identities are sampled
    (:func:`evaluation.stage_isolation.sample_predicted_map_identities`,
    truncating each sample to the match's ``n_played`` positions) and
    scored
    (:func:`evaluation.stage_isolation.score_predicted_played_maps`),
    the two-arm report is built
    (:func:`evaluation.stage_isolation.build_stage_isolation_report`),
    the ``n_samples`` / ``seed`` provenance keys are merged in, the
    artifact is written as
    ``<output_dir>/<version>/stage_isolation_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: both arms'
    ``mean_rps`` / ``mean_log_loss`` / ``marginal_binary_accuracy``
    plus the gap (Arm B minus Arm A).

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the three fitted model artifacts
            or any of the five input tables does not exist for the
            requested version (i.e. the training drivers /
            ``materialize.py`` / ``splits.py`` / ``labels.py`` have not
            been run for it) — propagated unchanged as a clear "run the
            prerequisite first" signal.
        ValueError: If the test split's actually-played map set is
            empty (from
            :func:`evaluation.stage_isolation.build_actual_played_maps`);
            if the Arm-B sampler rejects any input, including a
            degenerate all-zero-probability sample set or a sample
            with fewer than ``n_played`` maps (propagated from
            :func:`evaluation.stage_isolation.sample_predicted_map_identities`);
            if a scored vector has the wrong length or fails the metric
            validation (from
            :func:`evaluation.stage_isolation.score_actual_played_maps`
            / :func:`evaluation.stage_isolation.score_predicted_played_maps`,
            including ``log_loss``'s hard error on a zero-probability
            true category); or if the two scored tables are not
            row-aligned (from
            :func:`evaluation.stage_isolation.build_stage_isolation_report`).
        KeyError: If any input table or model artifact lacks a required
            column/key (propagated from the pure functions /
            ``from_dict`` calls).
        OSError / TypeError: If the artifact cannot be written
            (propagated from ``json.dumps`` / ``Path.write_text``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    ordinal_model, ban_model, pick_model = _load_fitted_models(
        output_dir, args.version
    )

    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The pluggable callables, wired exactly as
    # drivers/evaluate_series.py does: the fitted M20 ordinal-logit
    # closure as the Stage-2 four-way map model (used by *both* arms),
    # and the fitted M27/M28 conditional-logit closures as the Stage-1
    # ban/pick veto-step predictors (used by Arm B's sampler).
    map_model_fn = ordinal_logit.make_model_fn(
        ordinal_model, player_map_stats_df
    )
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }
    rng = np.random.default_rng(args.seed)

    # The one held-out map-position row table both arms are scored on,
    # in the one order — the row-alignment contract of the report.
    actual_df = stage_isolation.build_actual_played_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )

    scored_actual = stage_isolation.score_actual_played_maps(
        map_model_fn, actual_df, matches_df, maps_df
    )

    # Arm B: one sampler call per held-out match (in first-appearance
    # order, so the single rng is consumed deterministically), then one
    # effective score per position blended from the weighted samples.
    predicted_by_position: dict[tuple, list[tuple[str, float]]] = {}
    for match_id, group in actual_df.groupby("match_id", sort=False):
        row0 = group.iloc[0]
        identities = stage_isolation.sample_predicted_map_identities(
            match_id,
            row0.team1_id,
            row0.team2_id,
            row0.best_of,
            row0.date,
            matches_df,
            maps_df,
            predictor_fn_by_action,
            args.n_samples,
            rng,
            n_played=len(group),
        )
        for position, pairs in identities.items():
            predicted_by_position[(match_id, position)] = pairs
    scored_predicted = stage_isolation.score_predicted_played_maps(
        map_model_fn, actual_df, predicted_by_position, matches_df, maps_df
    )

    report = stage_isolation.build_stage_isolation_report(
        scored_actual, scored_predicted
    )
    # Provenance: record the resolved knobs so the report is
    # self-describing and reproducible without cross-referencing the
    # invocation command.
    report["n_samples"] = args.n_samples
    report["seed"] = args.seed

    artifact_path = (
        output_dir / args.version / "stage_isolation_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "stage isolation on %d held-out map positions (%s/%s, "
        "n_samples=%d seed=%d): %s",
        report["actual_played_maps"]["n_eval"],
        output_dir,
        args.version,
        args.n_samples,
        args.seed,
        _summarize(report),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
