"""Per-category reliability-diagrams driver — the M38 report (roadmap M38).

Thin command-line wrapper that builds BOTH arms of M38's per-category
reliability diagrams over the real fitted v1 pipeline and writes one
combined artifact:

- **Map arm** — the four-way map output: scores the fitted M20
  ordinal-logit Stage-2 map model (via :mod:`drivers.evaluate`'s
  ``MODEL_REGISTRY["ordinal_logit"]`` factory, the exact pattern
  ``drivers/evaluate_temperature_calibration.py`` already established
  for "score the real M20 model on the M19 held-out map set") over the
  M19 held-out ``split="test"`` map set
  (:func:`evaluation.harness.build_held_out_maps` +
  :func:`evaluation.harness.score_held_out_maps`), extracts the
  ``(n_eval, 4)`` prediction matrix and the true ``outcome_ordinal``
  vector from the scored table, and builds one binned reliability
  diagram per :data:`evaluation.harness.OUTCOME_LABELS` category.
- **Series arm** — the series scorelines: scores the real M31
  veto-marginalized pipeline (the M31 series model built from the real
  fitted artifacts exactly as ``drivers/evaluate_series.py`` wires it —
  copied ``_load_fitted_models`` helper, ``map_model_fn``,
  ``predictor_fn_by_action`` and
  :func:`evaluation.veto_marginalized_series.make_series_model_fn`, per
  the repo's convention that each ``evaluate_*.py`` driver duplicates
  this small private wiring; a shared loading spot is a flagged future
  refactor, out of scope) over the M33a held-out
  ``split="test"`` series set via
  :func:`evaluation.series_evaluation.score_held_out_series` — one
  veto-marginalization pass at ``n_samples=10`` (the same cost class
  as ``evaluate_series.py``'s own single pass, NOT M37's larger
  variance-measurement ``n_samples``), then **groups the scored table
  per ``best_of``** (K varies across groups, so each group is reported
  separately — mirrors
  :func:`evaluation.series_evaluation.build_series_evaluation_report`'s
  own per-``best_of`` grouping) and builds one binned reliability
  diagram per scoreline category (labels ``f"{a}-{b}"`` from
  ``utils.series_paths.series_outcome_order`` — the plan's assumption
  D, an ad hoc labeling scheme local to this artifact).

All binned-calibration math lives in the new pure module
:mod:`evaluation.reliability_diagrams` (a downward-only ``evaluation/``
module with no orchestration and no sibling import, per the
module-boundary DAG); this driver owns only the "load fitted artifacts
+ build held-out rows + score both arms + group + build reports +
write one artifact" loop, because only ``drivers/`` may depend
laterally on the sibling ``evaluation.harness`` /
``evaluation.series_evaluation`` / ``evaluation.veto_marginalized_series``
modules.

Scope decisions (recorded here, do not re-derive later; all are the
M38 plan's decisions, restated for REVIEW):

1. **One combined driver, one combined artifact** (decision 6): the
   roadmap phrases M38 as a single milestone covering "both the
   four-way map output and the series scorelines", so this one script
   builds both arms and writes one
   ``data/<version>/reliability_diagrams_report.json`` with top-level
   keys ``"map"`` and ``"series"`` — not two separate driver scripts.
2. **Per-category = one-vs-rest binary calibration, binned per
   category, generic over K** (decisions 1-2): for category ``c`` the
   pure module treats column ``c`` of the prediction matrix against
   the ``true_index == c`` indicator as one binary calibration problem
   and builds a quantile-binned (equal-count) reliability diagram;
   the same function scores a ``K=4`` map row and a ``K=6`` series
   row identically. The map arm uses ``n_bins_map`` default 5 (35
   held-out maps -> ~7/bin); the series arm uses ``n_bins_series``
   default 3 (15 held-out Bo3 series -> ~5/bin) — both chosen for
   v1's small held-out N (decision 10), both CLI-overridable.
3. **Sparse-group handling lives at driver level, not in the pure
   module** (decision 9): if a ``best_of`` group's row count is
   smaller than ``n_bins_series`` the driver **omits that group from
   the report** and logs a warning naming the group and its count,
   rather than calling :func:`evaluation.reliability_diagrams
   .build_reliability_report` and letting it raise on ``n_bins >
   n_eval`` — mirroring ``evaluation.series_evaluation``'s "empty Bo5
   group is omitted, not an error" precedent (the real v1 test split
   has 0 Bo5 series today). The pure module itself still raises if
   called directly with ``n_bins > n_eval``; only the driver adds the
   skip-and-warn behavior on top.
4. **The ``--n-samples`` default (10) is ``evaluate_series.py``'s
   measured single-scoring-pass default** (context item above): ~1.5-
   1.8 s per sampled veto sequence per series on real v1 fitted
   models, so the full 15-series Bo3 test-split run lands around 4-5
   minutes — deliberately NOT M37's ``n_samples=30`` (sized for
   variance measurement, a different question). ``--seed`` default
   2026 reproduces the whole run byte-identically (one
   ``numpy.random.default_rng`` consumed sequentially across the
   series, the repo's standard wiring).

**Artifact shape** (decision 11):
``{"map": <build_reliability_report dict>, "series": {"Bo3": {...},
"Bo5": {...}, "n_eval_total": int}, "n_bins_map": int, "n_bins_series":
int, "n_samples": int, "seed": int}`` — each arm/group block is the
pure module's report dict (``n_eval`` / ``n_bins`` / per-category
``expected_calibration_error`` + per-bin ``mean_predicted_prob`` /
``observed_frequency`` / ``gap`` pairs, the numeric data a reliability
plot would be drawn from — plan assumption C: no rendered charts
anywhere). ``series["n_eval_total"]`` is the total number of scored
held-out series across all groups (before any sparse-group omission),
mirroring ``build_series_evaluation_report``'s own ``n_eval_total``
convention. Written to ``data/<version>/reliability_diagrams_report.json``
via ``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
newline (the repo-wide convention). ``data/<version>/`` is gitignored
(repo convention, unchanged).

**Prerequisite artifacts.** This driver depends on the four input
tables (``materialize.py`` / ``splits.py`` / ``labels.py``) and on
``train_ordinal_logit.py``, ``train_conditional_logit_ban.py`` and
``train_conditional_logit_pick.py`` having already been run for the
requested version. A missing artifact raises ``FileNotFoundError``
unchanged — no silent fallback, the same "run the training driver
first" doctrine every existing ``evaluate_*.py`` driver follows.

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
from evaluation import (
    harness,
    reliability_diagrams,
    series_evaluation,
    veto_marginalized_series,
)
from models import conditional_logit_ban, conditional_logit_pick, ordinal_logit
from utils import series_paths
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The default bin counts and reproducibility knobs. DEFAULT_N_BINS_MAP
# (5) and DEFAULT_N_BINS_SERIES (3) are the M38 plan's decision-10
# choices for v1's small held-out N (35 maps -> ~7/bin; 15 held-out
# Bo3 series -> ~5/bin), both CLI-overridable. DEFAULT_N_SAMPLES (10)
# and DEFAULT_SEED (2026) are evaluate_series.py's measured defaults
# for a single scoring pass over the 15-series Bo3 test split (see the
# module docstring's scope decision 4 — deliberately not M37's larger
# variance-measurement n_samples).
DEFAULT_N_BINS_MAP = 5
DEFAULT_N_BINS_SERIES = 3
DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 2026


def _load_fitted_models(
    output_dir: Path, version: str
) -> tuple[object, object, object]:
    """Load the three fitted model artifacts the series arm needs.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json``, the fitted M27 conditional-logit ban
    model from ``conditional_logit_ban_model.json``, and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the exact artifact-loading
    pattern ``drivers/evaluate_series.py::_load_fitted_models`` already
    provides, copied (not imported) into this driver per the repo's
    convention that each ``evaluate_*.py`` driver independently
    duplicates this helper (the same choice
    ``drivers/evaluate_bootstrap_intervals.py`` /
    ``drivers/evaluate_veto_conditional_variance.py`` made and flagged —
    flagged here too: a shared loading spot would be a refactor
    opportunity, out of scope for this task). All three artifacts are
    used unchanged (fixed fitted models — no bootstrapping anywhere in
    M38, which measures calibration/reliability, a different component
    from M36's epistemic or M37's structural spread). The three
    ``from_dict`` calls are deliberately independent of each other and
    of the four input tables, so a missing artifact fails fast with the
    standard "run the training driver first" signal.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ordinal_model, ban_model, pick_model)`` tuple of the three
        deserialized fitted models, in the order the series arm wires
        them (the ordinal model feeds ``map_model_fn``; the ban/pick
        models feed the ``predictor_fn_by_action`` dict).

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


def _worst_calibrated_category(reliability_block: dict) -> str:
    """Return the worst-calibrated category label of a reliability block.

    Picks the category with the largest ``expected_calibration_error``
    from a :func:`evaluation.reliability_diagrams
    .build_reliability_report` dict — the headline "which category is
    the most miscalibrated" number the CLI summary (and the BUILD note)
    record per arm/group. Ties resolve to the earliest category in the
    block's ``categories`` order (matching
    ``evaluation.harness.build_evaluation_report``'s own
    most-miscalibrated-category tie rule).

    Args:
        reliability_block: A report dict with a ``categories`` list of
            per-category dicts, each carrying ``category`` (str) and
            ``expected_calibration_error`` (float) — the exact shape
            :func:`evaluation.reliability_diagrams.build_reliability_report`
            returns.

    Returns:
        The ``category`` string of the category with the largest
            ``expected_calibration_error`` (ties -> earliest in
            ``categories`` order).

    Raises:
        ValueError: If ``categories`` is empty (a block with no
            categories has no worst category to name).
        KeyError: If a category dict lacks ``category`` or
            ``expected_calibration_error`` (propagated from dict
            indexing).
    """
    if not reliability_block["categories"]:
        raise ValueError(
            "cannot name the worst-calibrated category of a reliability "
            "block with no categories"
        )
    return max(
        reliability_block["categories"],
        key=lambda category: category["expected_calibration_error"],
    )["category"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_reliability_diagrams.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with six attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), and the
        four M38 knobs: ``n_bins_map`` (``int``, the quantile bin
        count for the map arm, default :data:`DEFAULT_N_BINS_MAP`),
        ``n_bins_series`` (``int``, the quantile bin count per
        ``best_of`` group of the series arm, default
        :data:`DEFAULT_N_BINS_SERIES`), ``n_samples`` (``int``, the
        M29 veto walks sampled per held-out series by the M31 arm,
        default :data:`DEFAULT_N_SAMPLES`), and ``seed`` (``int``,
        the ``numpy.random.default_rng`` seed for the whole run,
        default :data:`DEFAULT_SEED`). Together they locate the five
        input tables, the three fitted model artifacts, and the output
        artifact
        ``<output_dir>/<version>/reliability_diagrams_report.json``.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-bins-map``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M38 per-category reliability diagrams: score the "
            "fitted ordinal-logit map model over the M19 held-out map "
            "set and the real M31 veto-marginalized series pipeline "
            "over the M33a held-out series set, and build one binned "
            "one-vs-rest reliability diagram per outcome category "
            "(four-way labels for the map arm, f'{a}-{b}' scoreline "
            "labels per best_of group for the series arm) with "
            "count-weighted expected calibration error per category; "
            "write reliability_diagrams_report.json."
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
        "--n-bins-map",
        type=int,
        default=DEFAULT_N_BINS_MAP,
        help=(
            "quantile (equal-count) bin count for the map arm's "
            f"per-category reliability diagrams (default: "
            f"{DEFAULT_N_BINS_MAP} — 35 held-out v1 maps -> ~7 rows per "
            "bin; must not exceed the map n_eval or the pure module "
            "raises)"
        ),
    )
    parser.add_argument(
        "--n-bins-series",
        type=int,
        default=DEFAULT_N_BINS_SERIES,
        help=(
            "quantile (equal-count) bin count for each best_of group of "
            "the series arm's per-category reliability diagrams "
            f"(default: {DEFAULT_N_BINS_SERIES} — 15 held-out v1 Bo3 "
            "series -> ~5 rows per bin; a best_of group with fewer rows "
            "than this is omitted from the report with a warning, not "
            "an error)"
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per held-out series by the M31 "
            f"arm (default: {DEFAULT_N_SAMPLES} — a measured wall-clock "
            "compromise: ~1.5-1.8s per sampled sequence on real v1 "
            "fitted models, so the full 15-series v1 test split takes "
            "~4-5 minutes; raise it for a smoother aggregate, lower it "
            "for a faster run)"
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
    """Run the M38 per-category reliability report end to end.

    Logging is configured first so the summary line is visible from the
    CLI, then the flag values are validated (positive ``n_samples``,
    ``n_bins_map`` and ``n_bins_series``). The five input tables are
    loaded for the requested version (matches/maps/labels/splits/
    player_map_stats via the ``drivers.evaluate`` helpers), and the two
    arms are built and scored:

    **Map arm** (decision 7): the held-out test map set is built once
    (:func:`evaluation.harness.build_held_out_maps` with
    ``split="test"``), the fitted M20 map model is obtained by reusing
    :mod:`drivers.evaluate`'s ``MODEL_REGISTRY["ordinal_logit"]``
    factory (which owns the artifact loading and the
    ``player_map_stats`` closure), the map set is scored via
    :func:`evaluation.harness.score_held_out_maps`, and the report is
    built from the scored table's prediction columns
    (:data:`evaluation.harness.PREDICTION_COLUMNS` -> ``prob_rows``)
    and true ordinals (``outcome_ordinal`` -> ``true_indices``) via
    :func:`evaluation.reliability_diagrams.build_reliability_report`
    with :data:`evaluation.harness.OUTCOME_LABELS` and
    ``n_bins=args.n_bins_map``.

    **Series arm** (decisions 8-9): the three fitted artifacts are
    loaded (:func:`_load_fitted_models`), the M31 arm's pluggable
    callables are wired exactly as ``drivers/evaluate_series.py`` does
    (``map_model_fn`` = :func:`models.ordinal_logit.make_model_fn`
    over the fitted ordinal model + ``player_map_stats.parquet``;
    ``predictor_fn_by_action`` = the ban/pick conditional-logit
    ``make_veto_step_predictor_fn`` closures), a single
    ``numpy.random.default_rng(args.seed)`` is constructed and passed
    into :func:`evaluation.veto_marginalized_series.make_series_model_fn`
    with ``n_samples=args.n_samples``, the held-out test-split series
    table is built once
    (:func:`evaluation.series_evaluation.build_held_out_series`), the
    series set is scored via
    :func:`evaluation.series_evaluation.score_held_out_series`, and the
    scored table is **grouped per ``best_of``**: for each group with at
    least ``n_bins_series`` rows, ``prob_rows =
    np.array(list(subset["probabilities"]))``, ``true_indices =
    subset["outcome_index"].to_numpy()``, ``category_labels = the
    f"{a}-{b}" scoreline strings of
    utils.series_paths.series_outcome_order(best_of_int)``, and the
    pure report builder is called with ``n_bins=args.n_bins_series``;
    a group with fewer rows than ``n_bins_series`` is **omitted from
    the report** with a logged warning naming the group and its count
    (decision 9 — never a hard error, and never a silent skip). The
    combined artifact (decision 11: the ``map`` block, the ``series``
    block with its per-``best_of`` reports plus the ``n_eval_total``
    count, and the four resolved knobs) is written as
    ``<output_dir>/<version>/reliability_diagrams_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: the map arm's held-out
    map count and worst-calibrated category, and per included series
    group its series count and worst-calibrated category.

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
            ``materialize.py`` / ``labels.py`` / ``splits.py`` have not
            been run for it) — propagated unchanged as a clear "run the
            prerequisite first" signal.
        ValueError: If ``n_samples``, ``n_bins_map`` or
            ``n_bins_series`` is not positive; if the test split's
            held-out map or series set is empty or malformed (from
            :func:`evaluation.harness.build_held_out_maps` /
            :func:`evaluation.series_evaluation.build_held_out_series`);
            if a prediction fails the harness/metric validation (from
            the two ``score_*`` functions); if ``n_bins_map`` exceeds
            the map arm's ``n_eval`` (from
            :func:`evaluation.reliability_diagrams.build_reliability_report`
            — the map arm has no skip-and-warn, unlike the series arm);
            or if the M31 pipeline rejects any input (propagated from
            :mod:`evaluation.veto_marginalized_series`).
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

    if args.n_samples < 1:
        raise ValueError(
            f"--n-samples must be a positive integer, got {args.n_samples}"
        )
    if args.n_bins_map < 1:
        raise ValueError(
            f"--n-bins-map must be a positive integer, got "
            f"{args.n_bins_map}"
        )
    if args.n_bins_series < 1:
        raise ValueError(
            f"--n-bins-series must be a positive integer, got "
            f"{args.n_bins_series}"
        )

    output_dir = Path(args.output_dir)
    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # --- Map arm (decision 7): the fitted M20 ordinal-logit model over
    # the M19 held-out test map set, via the registry factory.
    held_out_maps = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    map_model_fn = evaluate.MODEL_REGISTRY["ordinal_logit"](
        output_dir, args.version
    )
    scored_maps = harness.score_held_out_maps(
        map_model_fn, held_out_maps, matches_df, maps_df
    )
    map_report = reliability_diagrams.build_reliability_report(
        scored_maps[list(harness.PREDICTION_COLUMNS)].to_numpy(),
        scored_maps["outcome_ordinal"].to_numpy(),
        harness.OUTCOME_LABELS,
        n_bins=args.n_bins_map,
    )

    # --- Series arm (decision 8): the real M31 veto-marginalized
    # pipeline over the M33a held-out test series set, one scoring
    # pass at n_samples=args.n_samples.
    ordinal_model, ban_model, pick_model = _load_fitted_models(
        output_dir, args.version
    )
    series_map_model_fn = ordinal_logit.make_model_fn(
        ordinal_model, player_map_stats_df
    )
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }
    rng = np.random.default_rng(args.seed)
    veto_model_fn = veto_marginalized_series.make_series_model_fn(
        series_map_model_fn,
        predictor_fn_by_action,
        n_samples=args.n_samples,
        rng=rng,
    )
    held_out_series = series_evaluation.build_held_out_series(
        matches_df, maps_df, splits_df, split="test"
    )
    scored_series = series_evaluation.score_held_out_series(
        veto_model_fn, held_out_series, matches_df, maps_df
    )

    # Group the scored series table per best_of (K varies across
    # groups, so each group must be reported separately, mirroring
    # build_series_evaluation_report's own per-best_of grouping).
    series_report: dict = {}
    for best_of in sorted(scored_series["best_of"].unique()):
        subset = scored_series[scored_series["best_of"] == best_of]
        if len(subset) < args.n_bins_series:
            # Decision 9: a group too small to fill n_bins_series
            # quantile bins is omitted with a warning, never a hard
            # error (the real v1 report legitimately has no Bo5 group)
            # and never a silent skip.
            logger.warning(
                "best_of group %s has %d held-out series, fewer than "
                "--n-bins-series=%d: omitting it from the reliability "
                "diagrams report (a quantile bin would be empty)",
                best_of,
                len(subset),
                args.n_bins_series,
            )
            continue
        best_of_int = int(subset["best_of_int"].iloc[0])
        prob_rows = np.array(list(subset["probabilities"]))
        true_indices = subset["outcome_index"].to_numpy()
        category_labels = [
            f"{a_wins}-{b_wins}"
            for a_wins, b_wins in series_paths.series_outcome_order(
                best_of_int
            )
        ]
        series_report[str(best_of)] = (
            reliability_diagrams.build_reliability_report(
                prob_rows,
                true_indices,
                category_labels,
                n_bins=args.n_bins_series,
            )
        )
    series_report["n_eval_total"] = len(scored_series)

    report = {
        "map": map_report,
        "series": series_report,
        "n_bins_map": args.n_bins_map,
        "n_bins_series": args.n_bins_series,
        "n_samples": args.n_samples,
        "seed": args.seed,
    }

    artifact_path = (
        output_dir / args.version / "reliability_diagrams_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # One-line summary: the worst-calibrated category per arm, with the
    # series arm broken out per included best_of group.
    map_worst = _worst_calibrated_category(map_report)
    map_worst_ece = next(
        category["expected_calibration_error"]
        for category in map_report["categories"]
        if category["category"] == map_worst
    )
    series_summary = "; ".join(
        (
            f"{best_of}: {block['n_eval']} series, worst category "
            f"{_worst_calibrated_category(block)} "
            f"(ece={max(c['expected_calibration_error'] for c in block['categories']):.6f})"
        )
        for best_of, block in sorted(series_report.items())
        if best_of != "n_eval_total"
    )
    logger.info(
        "per-category reliability diagrams on %d held-out maps and %d "
        "held-out series (%s/%s, n_bins_map=%d n_bins_series=%d "
        "n_samples=%d seed=%d): map worst category %s "
        "(ece=%.6f)%s%s",
        map_report["n_eval"],
        series_report["n_eval_total"],
        output_dir,
        args.version,
        args.n_bins_map,
        args.n_bins_series,
        args.n_samples,
        args.seed,
        map_worst,
        map_worst_ece,
        "; " if series_summary else "",
        series_summary,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
