"""Bootstrap prediction-interval driver — the M36 epistemic report (roadmap M36).

The orchestration half of M36 (bootstrap prediction intervals,
epistemic): resamples the M10 ``"train"`` split with replacement at the
*match* level (a block bootstrap — the M36 plan assumption 2), refits
the Stage-2 map model (``models.ordinal_logit``) per replicate, and
propagates the spread of the replicate predictions into per-map
(4-way) and per-series (K-way scoreline) percentile intervals, reported
alongside ``n_games_backing`` per map. The pure interval-math helpers
live in :mod:`evaluation.bootstrap_intervals` (a downward-only
``evaluation/`` module with no orchestration, per the module-boundary
DAG); this driver owns the "resample + refit + run the M31 pipeline B
times" loop because only ``drivers/`` may depend upward on
``drivers.training_data`` and laterally on the sibling
``evaluation.veto_marginalized_series`` /
``evaluation.series_evaluation`` modules.

Phase 6 framing (restated here, do not re-derive later): M36
(epistemic — parameter uncertainty from a finite training sample), M37
(structural — spread across sampled veto sequences, resolves the moment
the veto happens) and M38 (calibration/reliability) are **three
distinct components, reported separately, never collapsed into one
number**. This driver builds ONLY the epistemic component. It must not
sample multiple veto sequences *for the purpose of measuring their
spread* (that is M37's job) and must not touch calibration/reliability
diagrams (M38). Where the pipeline needs *a* veto sample to propagate
Stage-2 spread into a series number, the veto-sampling randomness is
held **fixed** across bootstrap replicates: every series-level model
closure is constructed with a **freshly reconstructed**
``numpy.random.default_rng(veto_seed)`` — never advanced across
replicates — so the reported series spread is attributable to Stage-2
parameter uncertainty only, not to veto-sequence variance. **This
fixed-veto-rng mechanism is easy to misread as a bug** (advancing the
veto rng across replicates would silently reintroduce structural veto
variance into what is supposed to be a pure epistemic measurement); it
is deliberate and documented.

Scope decisions (recorded here, do not re-derive later; all are the M36
plan's assumptions, restated for REVIEW):

1. **Only the Stage-2 map model is bootstrapped.** The M27/M28
   conditional-logit ban/pick predictors (Stage 1) are loaded as fixed,
   already-fitted artifacts and reused unchanged across every bootstrap
   replicate, exactly as ``drivers/evaluate_series.py`` loads them.
   Bootstrapping the veto predictors too would mix epistemic parameter
   uncertainty from two different stages and would overlap with M37's
   explicit remit (veto-conditional variance).
2. **Resample unit is the match, not the map row.** Each draw
   resamples whole matches with replacement, carrying every map of a
   resampled match into the resampled training set together (see
   :func:`drivers.training_data.assemble_bootstrap_design_matrix`).
3. **Two separate replicate counts, sized from measured wall-clock**
   (this is the task's dominant cost risk; the numbers below were
   measured on real v1 fitted models by the BUILD phase, mirroring the
   "probe timed, then choose default from measurement" precedent of
   tasks 036/037/038):
   - Per-map replicates: each replicate rebuilds the full train
     design matrix (209 v1 train maps, ~0.12 s of feature-vector
     computation per row), refits the ordinal logit (~0.35 s), and
     scores the ~35 v1 held-out test maps (~0.12 s each) — measured
     ~30 s per replicate on real v1 data, so :data:`DEFAULT_N_BOOTSTRAP_MAP`
     (12) lands the per-map section around 6 minutes.
   - Per-series replicates: each replicate runs the *entire* M31
     veto-marginalized pipeline over all 15 held-out v1 test series;
     measured ~1.1 s per sampled veto sequence per series on real v1
     fitted models (i.e. ~17 s per series pass at
     ``n_samples=1``), so the series section cost is dominated by
     ``n_bootstrap_series + 1`` (the +1 is the nominal point
     estimate) series passes at :data:`DEFAULT_VETO_N_SAMPLES` (2):
     :data:`DEFAULT_N_BOOTSTRAP_SERIES` (12) + 1 nominal → 13 × ~33 s
     ≈ 7 minutes. The two sections together land around 13 minutes,
     within the plan's ~15-20 minute budget; both counts are CLI
     flags, not constants.
   - The feature-vector recomputation per replicate is a **known
     future-milestone optimization, not a defect**: caching/reusing
     feature vectors across replicates would require reworking the M31
     pipeline's opaque ``map_model_fn`` to expose a feature cache, a
     nontrivial architecture change out of scope for an M-sized task
     (M36 plan assumption 8).
4. **Independent per-category percentile bands, not a joint simplex
   region.** For each of the 4 (per-map) or K (per-series scoreline)
   categories, the interval is the ``[ci_level low-percentile,
   ci_level high-percentile]`` of that category's probability across
   the replicate models, computed independently per category via
   :func:`evaluation.bootstrap_intervals.replicate_matrix_intervals`.
   The K bands are marginal bands and do not jointly form a calibrated
   region on the simplex — documented in the module, the helper module,
   and the artifact (``interval_definition`` key) so it is not misread
   as a joint credible region. Default ``ci_level = 0.90`` (5th/95th
   percentiles), chosen over 0.95 because a 95% percentile estimate is
   noisier at the replicate counts this task's wall-clock budget can
   afford; exposed as ``--ci-level``.
5. **Non-converged replicate fits are kept, not dropped.** A
   non-converged ``models.ordinal_logit.fit`` ("hit ``max_iter`` /
   line-search failure") is still a valid, if suboptimal, model, and
   dropping some replicates would silently bias the percentile bands
   toward whichever resamples happened to converge. The artifact
   reports ``n_bootstrap_map_converged`` /
   ``n_bootstrap_series_converged`` as transparency diagnostics, not
   filters.
6. **Two independent, separately-flagged seeds.** ``--bootstrap-seed``
   seeds one ``numpy.random.default_rng`` consumed *sequentially*:
   first the ``n_bootstrap_map`` match-resampling draws for the per-map
   section, then a **fresh independent** draw of ``n_bootstrap_series``
   more for the series section (documented choice: the series replicate
   models are *not* the first ``n_bootstrap_series`` of the map
   replicate models — either is defensible per the plan, and the fresh
   draw keeps the two sections' replicate sets independent). The same
   seed reproduces byte-identical resamples and therefore
   byte-identical refit models. ``--veto-seed`` is a *separate* flag
   reconstructed as a fresh ``numpy.random.default_rng(veto_seed)``
   identically for every single series-level model closure (nominal and
   every replicate), never advanced across them — the mechanism that
   isolates the epistemic component (see the Phase 6 note above).
7. **No hyperparameter flags.** Each bootstrap refit uses
   ``models.ordinal_logit.fit``'s existing defaults, matching
   ``drivers/train_ordinal_logit.py``'s "no hyperparameter flags"
   precedent; tuning is out of scope. The nominal Stage-2 model is fit
   once over the full ``"train"`` split exactly like
   ``drivers/train_ordinal_logit.py`` does — the point estimate every
   interval is centered informationally around (reported in the
   artifact even though the intervals themselves come from the
   replicates, not from the nominal model).
8. **Held-out scope: the test split, both stages.** Per-map intervals
   are computed over ``evaluation.harness.build_held_out_maps(...,
   split="test")`` (the ~35 v1 test maps M34/M35 already evaluate
   over); series intervals over
   ``evaluation.series_evaluation.build_held_out_series(...,
   split="test")`` (the 15 v1 Bo3 test series M33b already evaluates
   over). Both reuse the identical fitted-model wiring
   ``drivers/evaluate_series.py`` established (``_load_fitted_models``
   — copied into this driver per the repo's convention that each
   ``evaluate_*.py`` driver independently duplicates this helper, the
   same choice ``drivers/evaluate_compounding_diagnostics.py`` made
   and flagged). This task does **not** implement a general
   "predict interval for an arbitrary future match" entry point — that
   is M39's job — but the pure helpers in
   :mod:`evaluation.bootstrap_intervals` are written generically enough
   that M39 can reuse them directly against a list of already-fitted
   replicate models.

**Prerequisite artifacts.** This driver depends on
``drivers/train_ordinal_logit.py``, ``drivers/train_conditional_logit_ban.py``
and ``drivers/train_conditional_logit_pick.py`` having already been
run for the requested version (three artifacts, mirroring
``drivers/evaluate_series.py``'s prerequisite pattern). A missing
artifact raises ``FileNotFoundError`` unchanged — no silent fallback,
the same "run the training driver first" doctrine every existing
``evaluate_*.py`` driver follows. The nominal Stage-2 refit and every
bootstrap refit happen at runtime inside this driver (they are not
artifacts).

**The ``--veto-n-samples`` default is a measured wall-clock choice, not
a guess.** The nominal and every series replicate use the *same*
``veto_n_samples`` (a single knob): ~1.1 s per sampled sequence per
series on real v1 fitted models, so :data:`DEFAULT_VETO_N_SAMPLES` (2)
lands each series pass (15 series) around 33 s. Because the veto rng is
held fixed across replicates, the *same* sampled veto sequences feed
every replicate, so the smaller ``n_samples`` adds the same structural
sampling noise to every replicate's aggregate (a shared offset that does
not widen the measured epistemic spread) — the fidelity of each
replicate's *point* estimate is lower than ``evaluate_series.py``'s
default of 10, and the reported nominal series vectors here therefore
use ``veto_n_samples``, not evaluate_series' ``DEFAULT_N_SAMPLES``;
recorded explicitly so the two drivers' nominal series numbers are not
compared as if they were the same knob.

Artifact written per run (scoped by dataset version):

- ``data/<version>/bootstrap_intervals_report.json`` — a dict with
  keys ``"interval_definition"`` (a plain-string restatement of the
  marginal-bands-not-joint-region caveat, decision 4), ``"config"``
  (``n_bootstrap_map``, ``n_bootstrap_series``,
  ``n_bootstrap_map_converged``, ``n_bootstrap_series_converged``,
  ``bootstrap_seed``, ``veto_seed``, ``veto_n_samples``, ``ci_level``),
  ``"per_map"`` (a list, one entry per held-out test map: the
  identifying columns, ``outcome_ordinal``, ``nominal`` (the 4-vector),
  ``interval_low`` / ``interval_high`` (the two 4-vectors of band
  endpoints), and ``n_games_backing`` (int, the weaker side's as-of
  per-map game count)) and ``"series"`` (a list, one entry per held-out
  test series: the identifying columns, ``outcome_index``, ``nominal``
  / ``interval_low`` / ``interval_high`` (K-vectors),
  ``played_maps`` (the series' actual played map names in play order)
  and ``n_games_backing`` (an int per played map, list-aligned with
  ``played_maps``)). Written with ``json.dumps(..., indent=2,
  sort_keys=True)`` plus a trailing newline (the repo-wide
  convention).

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
import pandas as pd

from drivers import evaluate, training_data
from evaluation import (
    bootstrap_intervals,
    harness,
    series_evaluation,
    veto_marginalized_series,
)
from features import map_win_rate
from models import conditional_logit_ban, conditional_logit_pick, ordinal_logit
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The two replicate counts and the two seeds, chosen from measured
# wall-clock on real v1 fitted models (documented in the module
# docstring, decision 3/6): ~30 s per per-map replicate (train-matrix
# rebuild + refit + scoring the ~35 test maps) and ~1.1 s per sampled
# veto sequence per series (so ~33 s per series pass at
# DEFAULT_VETO_N_SAMPLES = 2 over the 15 v1 test series). The two
# defaults together land the full real-v1 run around 13 minutes, within
# the plan's ~15-20 minute budget; both counts are CLI flags, not
# constants-of-convenience, and both seeds follow the repo's "current
# year" seed convention already used by drivers/evaluate_series.py.
DEFAULT_N_BOOTSTRAP_MAP = 12
DEFAULT_N_BOOTSTRAP_SERIES = 12
DEFAULT_BOOTSTRAP_SEED = 2026
DEFAULT_VETO_SEED = 2026
DEFAULT_VETO_N_SAMPLES = 2
DEFAULT_CI_LEVEL = 0.90


def _load_fitted_models(
    output_dir: Path, version: str
) -> tuple[object, object, object]:
    """Load the three fitted model artifacts this driver needs.

    Reconstructs the fitted M20 ordinal-logit model from
    ``ordinal_logit_model.json``, the fitted M27 conditional-logit ban
    model from ``conditional_logit_ban_model.json``, and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the exact artifact-loading
    pattern ``drivers/evaluate_series.py::_load_fitted_models`` already
    provides, copied (not imported) into this driver per the repo's
    convention that each ``evaluate_*.py`` driver independently
    duplicates this helper (the same choice
    ``drivers/evaluate_compounding_diagnostics.py`` made and flagged —
    flagged here too: a shared loading spot would be a refactor
    opportunity, out of scope for this task). The three ``from_dict``
    calls are deliberately independent of each other and of the five
    input tables, so a missing artifact fails fast with the standard
    "run the training driver first" signal. Note the ordinal artifact is
    *not* the model used for the intervals: the driver refits the
    nominal Stage-2 model at runtime and bootstraps further replicates;
    this load only provides the fixed Stage-1 ban/pick artifacts plus
    the (unused for scoring here) nominal artifact for consistency with
    ``evaluate_series.py``'s wiring.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ordinal_model, ban_model, pick_model)`` tuple of the three
        deserialized fitted models, in the order the driver wires them
        (the ban/pick models feed the ``predictor_fn_by_action`` dict;
        the ordinal model is loaded for wiring parity but the driver
        refits its own nominal/bootstrap Stage-2 models at runtime).

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


def _map_four_way_vector(
    map_model_fn,
    row,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> list[float]:
    """Predict one held-out map row with a 6-argument map model closure.

    Calls the generic model-interface closure (the shape
    ``models.ordinal_logit.make_model_fn`` produces) once for a held-out
    map row, extracting the four arguments from the row's
    ``team1_id``/``team2_id``/``map_name``/``date`` fields and passing
    the full tables through unchanged (the as-of leakage boundary is the
    model's own responsibility, inherited from ``utils.asof``'s strict
    ``<`` cutoff). The returned vector is converted to a plain ``list``
    so it is JSON-serializable and matrix-stackable.

    Args:
        map_model_fn: Any callable satisfying the generic model
            interface ``(team1_id, team2_id, map_name, date,
            matches_df, maps_df) -> Sequence[float]``.
        row: A held-out map row (``itertuples``-style) carrying
            ``team1_id``, ``team2_id``, ``map_name`` and ``date``.
        matches_df: The full materialised ``matches`` table, passed
            through to ``map_model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``map_model_fn`` unchanged.

    Returns:
        The model's 4-vector as a ``list`` of ``float`` in
        ``models._shared.OUTCOME_LABELS`` order.

    Raises:
        ValueError / KeyError: Propagated from ``map_model_fn`` (see
            :func:`models.ordinal_logit.make_model_fn`'s docstring).
    """
    return list(
        map_model_fn(
            row.team1_id,
            row.team2_id,
            row.map_name,
            row.date,
            matches_df,
            maps_df,
        )
    )


def _series_scoreline_vectors(
    held_out_series: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_model_fn,
    predictor_fn_by_action: dict,
    veto_n_samples: int,
    veto_seed: int,
) -> list[list[float]]:
    """Score every held-out series with one Stage-2 model via the M31 pipeline.

    Runs the full M31 veto-marginalized pipeline once over the held-out
    series table: constructs
    :func:`evaluation.veto_marginalized_series.make_series_model_fn`
    with the given Stage-2 ``map_model_fn``, the fixed Stage-1
    ``predictor_fn_by_action`` dict, ``veto_n_samples``, and a **freshly
    reconstructed** ``numpy.random.default_rng(veto_seed)`` — the
    fixed-veto-rng mechanism (module docstring decision 6): every call
    to this helper for every model reconstructs the *same* rng, so every
    model's veto draws are the same draws and only the Stage-2
    coefficients differ between the returned vectors. Scores the
    held-out series via
    :func:`evaluation.series_evaluation.score_held_out_series` and
    returns each series' ``best_of + 1``-length scoreline probability
    vector in ``utils.series_paths.series_outcome_order`` order.

    Args:
        held_out_series: The held-out series table from
            :func:`evaluation.series_evaluation.build_held_out_series`.
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.
        map_model_fn: The Stage-2 four-way per-map model closure for the
            model being scored (nominal or one bootstrap replicate).
        predictor_fn_by_action: The fixed Stage-1 ``"ban"``/``"pick"``
            conditional-logit predictor closures (identical for every
            model — decision 1).
        veto_n_samples: The M29 veto sequences sampled per series by the
            M31 pipeline (the driver's ``--veto-n-samples``).
        veto_seed: The seed reconstructed identically for every model's
            M31 rng (the driver's ``--veto-seed``); never advanced
            across models.

    Returns:
        A list of ``best_of + 1``-length ``float`` scoreline
        probability vectors, one per held-out series in table order.

    Raises:
        ValueError / KeyError / TypeError: Propagated from
            :func:`evaluation.veto_marginalized_series.make_series_model_fn`
            / :func:`evaluation.series_evaluation.score_held_out_series`
            (see those docstrings — e.g. a degenerate all-zero-
            probability sample set, a wrong-length scored vector, a
            malformed ``best_of``).
    """
    # Freshly reconstructed rng per model — never advanced across
    # replicates (the fixed-veto-rng mechanism, decision 6). This is
    # deliberately identical for every model; do not "fix" it to reuse a
    # single rng object.
    rng = np.random.default_rng(veto_seed)
    series_model_fn = veto_marginalized_series.make_series_model_fn(
        map_model_fn,
        predictor_fn_by_action,
        n_samples=veto_n_samples,
        rng=rng,
    )
    scored = series_evaluation.score_held_out_series(
        series_model_fn, held_out_series, matches_df, maps_df
    )
    return [list(vector) for vector in scored["probabilities"]]


def _games_backing_for_map(
    team1_id: str,
    team2_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> int:
    """Compute one map's ``n_games_backing`` via the shared feature estimator.

    Queries :func:`features.map_win_rate.team_map_win_rate` for both
    teams (with :data:`features.map_win_rate.DEFAULT_K` — the same as-of
    cutoff Stage 2's own ``map_win_rate_diff`` feature uses; ``k`` only
    affects the shrunk ``mean``/``variance``, never the ``games``
    count, so the specific ``k`` does not change the result) and returns
    :func:`evaluation.bootstrap_intervals.n_games_backing` —
    ``min(games_team1, games_team2)``, the weaker side's as-of, map-
    specific sample size backing this prediction (M36 plan assumption
    9: ``min``, not ``sum``, so a data-rich side never overstates
    confidence against a brand-new opponent).

    Args:
        team1_id: The queried team1's stable id.
        team2_id: The queried team2's stable id.
        map_name: The map to query backing for.
        date: The as-of cutoff (the map's own match date; strict ``<``).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.

    Returns:
        ``min(team1_games, team2_games)`` as an ``int``; ``0`` when
        either side has no as-of games on that map.

    Raises:
        ValueError: If the query date is null/unparseable/timezone-aware
            or an as-of map has a tied/null score (propagated from
            :func:`features.map_win_rate.team_map_win_rate`).
        KeyError: If either table lacks a required column (propagated
            from the same call).
        TypeError: If the query date is list-like (propagated from the
            same call).
    """
    games_a = map_win_rate.team_map_win_rate(
        team1_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).games
    games_b = map_win_rate.team_map_win_rate(
        team2_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).games
    return bootstrap_intervals.n_games_backing(games_a, games_b)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the evaluate_bootstrap_intervals.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with nine attributes: ``version``
        (``str``, the input/output subdirectory name, default
        ``"v1"``), ``output_dir`` (``str``, the parent directory the
        version subdirectory lives under, default ``"data"``), the two
        replicate counts ``n_bootstrap_map`` (``int``, default
        :data:`DEFAULT_N_BOOTSTRAP_MAP`) and ``n_bootstrap_series``
        (``int``, default :data:`DEFAULT_N_BOOTSTRAP_SERIES`), the two
        independent seeds ``bootstrap_seed`` (``int``, default
        :data:`DEFAULT_BOOTSTRAP_SEED`) and ``veto_seed`` (``int``,
        default :data:`DEFAULT_VETO_SEED`), the M31 sampling knob
        ``veto_n_samples`` (``int``, default
        :data:`DEFAULT_VETO_N_SAMPLES`), and the interval level
        ``ci_level`` (``float``, default :data:`DEFAULT_CI_LEVEL`).
        Together they locate the five input tables, the three fitted
        model artifacts, and the output artifact
        ``<output_dir>/<version>/bootstrap_intervals_report.json``.

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag or a non-int ``--n-bootstrap-map``).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M36 bootstrap prediction-interval report: resample "
            "the train split at the match level (block bootstrap), "
            "refit the Stage-2 ordinal-logit map model per replicate, "
            "and propagate the replicate spread into per-map (4-way) and "
            "per-series (K-way scoreline) percentile intervals via the "
            "M31 veto-marginalized pipeline (veto rng held fixed across "
            "replicates — epistemic only, no structural variance); "
            "write bootstrap_intervals_report.json."
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
        "--n-bootstrap-map",
        type=int,
        default=DEFAULT_N_BOOTSTRAP_MAP,
        help=(
            "per-map bootstrap replicate count (default: "
            f"{DEFAULT_N_BOOTSTRAP_MAP} — a measured wall-clock choice "
            "on real v1 data: ~30s per replicate, so the per-map section "
            "lands around 6 minutes; raise for smoother percentile "
            "bands, lower for a faster run)"
        ),
    )
    parser.add_argument(
        "--n-bootstrap-series",
        type=int,
        default=DEFAULT_N_BOOTSTRAP_SERIES,
        help=(
            "per-series bootstrap replicate count (default: "
            f"{DEFAULT_N_BOOTSTRAP_SERIES} — a measured wall-clock "
            "choice: each replicate runs the full M31 pipeline over the "
            "15 v1 test series at --veto-n-samples, ~33s per pass at "
            "the default, so the series section lands around 7 minutes; "
            "raise for smoother percentile bands, lower for a faster "
            "run)"
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=(
            "seed for numpy.random.default_rng, driving the match-"
            "resampling draws (consumed sequentially: the per-map "
            "replicates first, then a fresh independent draw for the "
            f"series replicates; default: {DEFAULT_BOOTSTRAP_SEED})"
        ),
    )
    parser.add_argument(
        "--veto-seed",
        type=int,
        default=DEFAULT_VETO_SEED,
        help=(
            "seed reconstructed identically for every series-level M31 "
            "rng (never advanced across replicates), isolating the "
            f"epistemic component (default: {DEFAULT_VETO_SEED})"
        ),
    )
    parser.add_argument(
        "--veto-n-samples",
        type=int,
        default=DEFAULT_VETO_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per series by the M31 pipeline "
            "for the nominal and every replicate series prediction "
            f"(default: {DEFAULT_VETO_N_SAMPLES} — a measured wall-clock "
            "choice: ~1.1s per sampled sequence per series on real v1 "
            "fitted models, and since the veto rng is fixed across "
            "replicates the same draws feed every replicate)"
        ),
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help=(
            "interval level in (0, 1): each per-category band spans the "
            f"middle ci_level fraction of the replicate distribution "
            f"(default: {DEFAULT_CI_LEVEL} — 5th/95th percentiles)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bootstrap prediction-interval report end to end.

    Logging is configured first so the summary line is visible from the
    CLI. The three fitted model artifacts are loaded first
    (:func:`_load_fitted_models` — a missing artifact raises
    ``FileNotFoundError`` as the "run the training drivers first"
    signal), the five input tables are loaded for the requested version
    (matches/maps/labels/splits/player_map_stats via the
    ``drivers.evaluate`` helpers), the flag values are validated
    (positive replicate counts, ``ci_level`` in ``(0, 1)``), and then:

    1. **Nominal Stage-2 model.** ``models.ordinal_logit.fit`` over the
       full ``"train"`` split via
       :func:`drivers.training_data.assemble_design_matrix` — the point
       estimate every interval is centered informationally around
       (reported in the artifact even though the intervals are computed
       from the replicates).
    2. **Per-map section.** One
       :func:`drivers.training_data.assemble_bootstrap_design_matrix`
       draw + ``ordinal_logit.fit`` per ``--n-bootstrap-map`` replicate
       (all draws from the single ``--bootstrap-seed`` rng), each
       replicate's model scored on the held-out test-split maps via
       :func:`models.ordinal_logit.make_model_fn`, the per-category
       bands computed via
       :func:`evaluation.bootstrap_intervals.replicate_matrix_intervals`,
       and each map's ``n_games_backing`` computed via
       :func:`_games_backing_for_map` (M36 plan assumption 9).
    3. **Per-series section.** A **fresh independent** draw of
       ``--n-bootstrap-series`` more replicate models from the same
       bootstrap rng (decision 6), then the nominal model and every
       series replicate's Stage-2 closure scored through the full M31
       pipeline via :func:`_series_scoreline_vectors` (each with a
       freshly reconstructed ``--veto-seed`` rng — the fixed-veto-rng
       mechanism), the per-category K-way bands computed per series,
       and each series' actual played maps' ``n_games_backing`` values
       attached (list-aligned with the series' played-map list).

    The combined report (``interval_definition`` caveat string, the
    ``config`` block with the replicate counts / convergence counts /
    seeds / knobs, the ``per_map`` list and the ``series`` list) is
    written as
    ``<output_dir>/<version>/bootstrap_intervals_report.json``
    (``json.dumps(..., indent=2, sort_keys=True)`` plus a trailing
    newline), and a one-line summary is logged: the per-map and
    per-series map/series counts, the replicate counts with their
    converged counts, and the mean per-category interval width across
    maps and across series (the headline finding).

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
        ValueError: If ``n_bootstrap_map`` / ``n_bootstrap_series`` are
            not positive or ``ci_level`` is not in ``(0, 1)``; if the
            test split's held-out map/series set is empty (from the
            harness); if a replicate fit's label vector is empty or a
            feature computation fails (from
            :func:`drivers.training_data.assemble_bootstrap_design_matrix`
            / ``models.ordinal_logit.fit``); if the M31 pipeline rejects
            any input (propagated from
            :mod:`evaluation.veto_marginalized_series`); or if the
            interval helpers reject malformed replicate collections
            (from :mod:`evaluation.bootstrap_intervals`).
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

    if args.n_bootstrap_map < 1:
        raise ValueError(
            f"--n-bootstrap-map must be a positive integer, got "
            f"{args.n_bootstrap_map}"
        )
    if args.n_bootstrap_series < 1:
        raise ValueError(
            f"--n-bootstrap-series must be a positive integer, got "
            f"{args.n_bootstrap_series}"
        )
    if not (0.0 < args.ci_level < 1.0):
        raise ValueError(
            f"--ci-level must be strictly between 0 and 1, got "
            f"{args.ci_level}"
        )

    output_dir = Path(args.output_dir)
    _ordinal_model, ban_model, pick_model = _load_fitted_models(
        output_dir, args.version
    )

    matches_df = evaluate.load_matches_table(output_dir, args.version)
    maps_df = evaluate.load_maps_table(output_dir, args.version)
    labels_df = evaluate.load_labels_table(output_dir, args.version)
    splits_df = evaluate.load_splits_table(output_dir, args.version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, args.version
    )

    # The fixed Stage-1 predictors (decision 1): loaded once, reused
    # unchanged by the nominal and every bootstrap replicate.
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }

    # Step 1: the nominal Stage-2 model over the full train split.
    X_nominal, y_nominal = training_data.assemble_design_matrix(
        matches_df,
        maps_df,
        labels_df,
        splits_df,
        player_map_stats_df,
        split="train",
    )
    nominal_model = ordinal_logit.fit(X_nominal, y_nominal)
    nominal_map_fn = ordinal_logit.make_model_fn(
        nominal_model, player_map_stats_df
    )

    # The one bootstrap rng, consumed sequentially (decision 6): first
    # the per-map replicates' draws, then a fresh independent draw for
    # the series replicates.
    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)

    # Step 2: the per-map section.
    held_out_maps = harness.build_held_out_maps(
        matches_df, maps_df, labels_df, splits_df, split="test"
    )
    map_rows = list(held_out_maps.itertuples(index=False))
    nominal_map_vectors = [
        _map_four_way_vector(nominal_map_fn, row, matches_df, maps_df)
        for row in map_rows
    ]

    map_replicate_models: list = []
    for _ in range(args.n_bootstrap_map):
        X_boot, y_boot = training_data.assemble_bootstrap_design_matrix(
            matches_df,
            maps_df,
            labels_df,
            splits_df,
            player_map_stats_df,
            bootstrap_rng,
        )
        map_replicate_models.append(ordinal_logit.fit(X_boot, y_boot))
    n_map_converged = sum(
        int(model.converged) for model in map_replicate_models
    )

    # (n_replicates, n_maps, 4) replicate probability tensor.
    replicate_map_tensor = np.zeros(
        (args.n_bootstrap_map, len(map_rows), 4), dtype=float
    )
    for r, model in enumerate(map_replicate_models):
        map_fn = ordinal_logit.make_model_fn(model, player_map_stats_df)
        for c, row in enumerate(map_rows):
            replicate_map_tensor[r, c, :] = _map_four_way_vector(
                map_fn, row, matches_df, maps_df
            )

    per_map: list[dict] = []
    for c, row in enumerate(map_rows):
        intervals = bootstrap_intervals.replicate_matrix_intervals(
            replicate_map_tensor[:, c, :], ci_level=args.ci_level
        )
        per_map.append(
            {
                "match_id": row.match_id,
                "map_index": int(row.map_index),
                "date": row.date,
                "team1_id": row.team1_id,
                "team2_id": row.team2_id,
                "map_name": row.map_name,
                "outcome_ordinal": int(row.outcome_ordinal),
                "nominal": nominal_map_vectors[c],
                "interval_low": [lo for lo, _hi in intervals],
                "interval_high": [hi for _lo, hi in intervals],
                "n_games_backing": _games_backing_for_map(
                    row.team1_id,
                    row.team2_id,
                    row.map_name,
                    row.date,
                    matches_df,
                    maps_df,
                ),
            }
        )

    # Step 3: the per-series section.
    held_out_series = series_evaluation.build_held_out_series(
        matches_df, maps_df, splits_df, split="test"
    )
    series_rows = list(held_out_series.itertuples(index=False))

    # A fresh independent draw of series replicate models (decision 6),
    # continuing the sequential consumption of the bootstrap rng.
    series_replicate_models: list = []
    for _ in range(args.n_bootstrap_series):
        X_boot, y_boot = training_data.assemble_bootstrap_design_matrix(
            matches_df,
            maps_df,
            labels_df,
            splits_df,
            player_map_stats_df,
            bootstrap_rng,
        )
        series_replicate_models.append(ordinal_logit.fit(X_boot, y_boot))
    n_series_converged = sum(
        int(model.converged) for model in series_replicate_models
    )

    # Nominal + every series replicate, each with a freshly
    # reconstructed veto rng (the fixed-veto-rng mechanism).
    nominal_series_vectors = _series_scoreline_vectors(
        held_out_series,
        matches_df,
        maps_df,
        nominal_map_fn,
        predictor_fn_by_action,
        args.veto_n_samples,
        args.veto_seed,
    )
    series_replicate_vectors: list[list[list[float]]] = []
    for model in series_replicate_models:
        map_fn = ordinal_logit.make_model_fn(model, player_map_stats_df)
        series_replicate_vectors.append(
            _series_scoreline_vectors(
                held_out_series,
                matches_df,
                maps_df,
                map_fn,
                predictor_fn_by_action,
                args.veto_n_samples,
                args.veto_seed,
            )
        )

    # The actual played maps per held-out series, in play order, for the
    # per-map n_games_backing list (M36 plan assumption 9).
    maps_by_match: dict = {
        match_id: group.sort_values("map_index")
        for match_id, group in maps_df.groupby("match_id", sort=True)
    }

    series: list[dict] = []
    for s, row in enumerate(series_rows):
        intervals = bootstrap_intervals.replicate_matrix_intervals(
            np.asarray(
                [vectors[s] for vectors in series_replicate_vectors],
                dtype=float,
            ),
            ci_level=args.ci_level,
        )
        played = maps_by_match.get(row.match_id)
        played_maps = [m.map_name for m in played.itertuples(index=False)]
        backing = [
            _games_backing_for_map(
                row.team1_id,
                row.team2_id,
                map_name,
                row.date,
                matches_df,
                maps_df,
            )
            for map_name in played_maps
        ]
        series.append(
            {
                "match_id": row.match_id,
                "date": row.date,
                "team1_id": row.team1_id,
                "team2_id": row.team2_id,
                "best_of": row.best_of,
                "best_of_int": int(row.best_of_int),
                "a_wins": int(row.a_wins),
                "b_wins": int(row.b_wins),
                "outcome_index": int(row.outcome_index),
                "nominal": nominal_series_vectors[s],
                "interval_low": [lo for lo, _hi in intervals],
                "interval_high": [hi for _lo, hi in intervals],
                "played_maps": played_maps,
                "n_games_backing": backing,
            }
        )

    report = {
        "interval_definition": (
            "independent per-category percentile bands "
            "(marginal, NOT a joint simplex credible region): each "
            "category's band is the [low_percentile, high_percentile] "
            "of that category's probability across the bootstrap "
            "replicates, computed independently per category"
        ),
        "config": {
            "n_bootstrap_map": args.n_bootstrap_map,
            "n_bootstrap_series": args.n_bootstrap_series,
            "n_bootstrap_map_converged": n_map_converged,
            "n_bootstrap_series_converged": n_series_converged,
            "bootstrap_seed": args.bootstrap_seed,
            "veto_seed": args.veto_seed,
            "veto_n_samples": args.veto_n_samples,
            "ci_level": args.ci_level,
        },
        "per_map": per_map,
        "series": series,
    }

    artifact_path = (
        output_dir / args.version / "bootstrap_intervals_report.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # One-line summary: the headline finding is the mean per-category
    # interval width (mean of hi - lo over categories) across maps and
    # across series.
    mean_map_width = float(
        np.mean(
            [
                hi - lo
                for entry in per_map
                for lo, hi in zip(
                    entry["interval_low"], entry["interval_high"]
                )
            ]
        )
    )
    mean_series_width = float(
        np.mean(
            [
                hi - lo
                for entry in series
                for lo, hi in zip(
                    entry["interval_low"], entry["interval_high"]
                )
            ]
        )
    )
    logger.info(
        "bootstrap intervals on %d held-out test-split maps and %d "
        "series (%s/%s, n_bootstrap_map=%d/%d converged, "
        "n_bootstrap_series=%d/%d converged, bootstrap_seed=%d "
        "veto_seed=%d veto_n_samples=%d ci_level=%.2f): mean per-category "
        "interval width %.4f per map, %.4f per series",
        len(per_map),
        len(series),
        output_dir,
        args.version,
        args.n_bootstrap_map,
        n_map_converged,
        args.n_bootstrap_series,
        n_series_converged,
        args.bootstrap_seed,
        args.veto_seed,
        args.veto_n_samples,
        args.ci_level,
        mean_map_width,
        mean_series_width,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
