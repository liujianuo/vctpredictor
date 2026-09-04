"""The M39 ``predict()`` public API and its thin CLI.

Ships the documented library entry point

    predict(team_a, team_b, best_of, map_pool, as_of_date)

(returned by :func:`make_predictor` over the materialised tables and
fitted artifacts for one dataset version) plus a thin command-line
wrapper, and the M39.1 persistent layer: the :class:`Predictor` object
that loads those tables/artifacts once at construction and answers
many ``predict()`` calls in one process, driven from the CLI's
``--stream`` JSONL query-stream mode. This is a **wiring milestone,
not a model**: every underlying
piece already exists and is reviewed/clean —
``models.greedy_veto_simulator`` (M25), ``models.ordinal_logit`` (M20),
``models.temperature_scaling`` (M24), ``models.ancestral_veto_sampler``
(M29), ``evaluation.veto_marginalized_series`` (M31),
``evaluation.bootstrap_intervals`` (M36) and
``evaluation.veto_conditional_variance`` (M37) — and this module only
composes them into the documented result shapes below.

**Placement decision (recorded, do not re-derive).** ``predict()``
lives in ``drivers/``, not ``models/`` or ``evaluation/``. It
orchestrates *sibling* ``evaluation/`` modules
(``veto_marginalized_series``, ``bootstrap_intervals``,
``veto_conditional_variance``) together with ``models.*`` — exactly
the cross-layer composition the module-boundary DAG forbids in both
``models/`` (no sibling ``models/`` imports, no ``evaluation/``) and
``evaluation/`` (no sibling ``evaluation/`` imports, no ``drivers/``).
``drivers/`` is the only layer permitted to import freely across
``drivers.*`` / ``evaluation.*`` / ``models.*`` / ``features.*`` /
``utils.*``. This mirrors the M36/M37 precedent, whose drivers
explicitly note their pure helpers were written "so M39's ``predict()``
public API can reuse them directly". ``drivers/`` is **not** in the
scanned module lists of ``tests/test_module_boundaries.py``, so no
boundary-test change is needed.

**Design decisions D1-D9 (recorded here, do not silently change).**

- **D1.** The public signature is a closure returned by a factory.
  ``predict(team_a, team_b, best_of, map_pool, as_of_date)`` needs the
  three materialised tables and four fitted artifacts, which the
  documented 5-arg signature does not carry; :func:`make_predictor`
  loads tables/artifacts once and returns the 5-arg ``predict``
  closure. ``team_a``/``team_b`` are stable ``team_id`` strings (the
  vocabulary every feature/model consumes, e.g. ``"397"``), **not**
  display names.
- **D2.** ``predicted_veto`` is the M25 greedy simulator
  (:func:`models.greedy_veto_simulator.simulate_veto`), not a trained
  M27/M28 argmax: no deterministic argmax walk over the fitted
  conditional-logit predictors exists anywhere in the repo (M29's
  sampler is stochastic), and building one would be new Stage-1
  modeling, out of scope for a wiring milestone. The played maps for
  ``per_map`` are that greedy sequence's ``pick`` and ``decider``
  actions in step order.
- **D3.** The per-map point model is the M24 temperature-scaled
  ordinal, everywhere: the four per-map probabilities, and the
  ``map_model_fn`` fed to M31 for ``series`` / ``veto_sensitivity``,
  are all built from
  ``drivers.evaluate.MODEL_REGISTRY["ordinal_logit_temperature"]`` —
  which already loads the base ordinal + temperature artifacts,
  enforces the staleness guard, and returns the 6-arg ``ModelFn``.
  (M33b/M36/M37's evaluation drivers wired the *raw* ordinal into M31;
  they predate M24's production role, so ``predict().series`` differs
  slightly from ``series_evaluation_report.json`` — expected and
  documented, not a bug.)
- **D4.** The epistemic interval consumes already-fitted replicate
  models: ``make_predictor(..., bootstrap_models=None)`` accepts an
  optional ``Sequence[OrdinalLogitModel]``; when ``None`` (or empty),
  ``per_map[i].interval_*`` is ``None`` (no interval), and when
  provided, each map's interval is
  :func:`evaluation.bootstrap_intervals.replicate_matrix_intervals`
  over the replicate models' 4-way predictions. The interval is over
  the **raw ordinal** replicates (M36's definition) while the point
  estimate is temperature-scaled (D3) — the same asymmetry M36 itself
  records; kept and stated here. Producing/persisting replicate models
  for real users is the caller's job (or a later milestone); this task
  only consumes them.
- **D5.** One M31 call produces both ``series`` and
  ``veto_sensitivity``: ``series`` = ``prediction.probabilities`` (+
  ``outcome_order`` + ``best_of``) of
  :func:`evaluation.veto_marginalized_series
  .predict_series_outcome_via_veto_marginalization`; ``veto_sensitivity``
  = the M37 summary of ``prediction.samples`` via
  :mod:`evaluation.veto_conditional_variance` (unweighted bands,
  widths, mean width, weighted mean/variance) — one sampling pass, not
  two.
- **D6.** Per-call RNG reconstruction (idempotent public API): each
  ``predict(...)`` call constructs a fresh
  ``numpy.random.default_rng(seed)`` for that call's M31 sampling, so
  identical arguments reproduce identical output. This deliberately
  differs from M37's driver (one rng advanced across a batch); a
  public API's repeated identical calls must not drift.
- **D7.** Defaults: ``n_samples=30`` (M37's measured default — a
  stable *spread* estimate needs more draws than a stable mean;
  ~1.5s per sampled sequence per series on real v1), ``seed=2026``
  (repo convention), and :data:`features.map_win_rate.DEFAULT_K` for
  the greedy veto and ``n_games_backing`` (``k`` does not affect
  ``.games``, and ``DEFAULT_K = 10.0`` is what
  ``models._shared.build_feature_vector`` itself uses).
- **D8.** ``map_pool=None`` resolves the era pool from config for
  ``as_of_date``: both ``simulate_veto`` and ``sample_veto_sequences``
  accept ``map_pool=None`` and resolve via
  ``utils.config.Config.era_as_of``; ``predict`` passes the caller's
  ``map_pool`` straight through unchanged. All supported formats
  require a 7-map pool (the existing ``ACTION_SEQUENCES``), so a
  non-7 ``map_pool`` raises ``ValueError`` from the simulator/sampler —
  propagated, not re-validated.
- **D9.** No labels/splits tables: core prediction needs only
  ``matches`` / ``maps`` / ``player_map_stats`` tables + the
  ``ordinal`` / ``temperature`` / ``conditional_logit_ban`` /
  ``conditional_logit_pick`` artifacts. ``labels``/``splits`` are
  **not** loaded (they are training/eval inputs, and the
  bootstrap-interval path takes pre-fitted models per D4 rather than
  refitting).

**Design decisions E1-E6 (recorded here, do not silently change) — the
M39.1 persistent layer.**

- **E1.** :class:`Predictor` is a thin wrapper over the existing
  :func:`make_predictor` wiring; :func:`make_predictor` is untouched
  internally. ``Predictor.__init__(output_dir, version, *,
  n_samples=DEFAULT_N_SAMPLES, seed=DEFAULT_SEED,
  ci_level=DEFAULT_CI_LEVEL, bootstrap_models=None)`` calls
  :func:`make_predictor` exactly once, forwarding every keyword
  unchanged, and stores the returned 5-arg closure as a private
  attribute; ``Predictor.predict(team_a, team_b, best_of, map_pool,
  as_of_date)`` calls that closure and returns its result unmodified.
  :func:`make_predictor`'s body is not refactored or reordered — zero
  risk to its reviewed-clean tests. A ``Predictor`` instance's
  ``.predict(...)`` call is bitwise identical to calling
  :func:`make_predictor(...)` once and invoking the returned closure
  with the same arguments; D6's per-call fresh-RNG idempotence is
  inherited unchanged since the wrapped closure is the same closure
  :func:`make_predictor` already returns.
- **E2.** ``--stream`` is an explicit CLI flag, not stdin
  auto-detection. A new boolean flag ``--stream``
  (``action="store_true"``, default ``False``) switches ``main()``
  into persistent JSONL query-stream mode. Auto-detecting "stdin has
  data" (e.g. ``sys.stdin.isatty()``) was considered and rejected: it
  is not reliably testable and it silently changes behaviour based on
  how the process happens to be invoked rather than an explicit,
  discoverable flag. ``--stream`` is mutually exclusive with
  ``--team-a`` / ``--team-b`` / ``--best-of`` / ``--as-of-date`` /
  ``--map-pool`` — all five must be at their defaults (``None``) when
  ``--stream`` is given.
- **E3.** Argparse required-arg enforcement moves from
  ``required=True`` to a manual post-parse check. ``--team-a``,
  ``--team-b``, ``--best-of``, ``--as-of-date`` change to
  ``required=False, default=None`` (``--best-of`` keeps its
  ``choices=["Bo1", "Bo3", "Bo5"]`` constraint, which only fires
  when a value is actually given); immediately after
  ``parser.parse_args(argv)`` two manual checks each fire
  ``parser.error(...)`` (raising ``SystemExit(2)``, matching
  argparse's own required-arg behaviour): the four query flags are
  required unless ``--stream`` is given, and ``--stream`` cannot be
  combined with any of the five query flags.
- **E4.** Stream query schema and per-line behaviour. One JSON object
  per stdin line: ``{"team_a": str, "team_b": str, "best_of": str,
  "as_of_date": str, "map_pool": [str, ...] | null}`` (``map_pool``
  optional; absent or ``null`` means ``None``, same era-resolution as
  the one-shot CLI, D8). A present ``map_pool`` JSON array is
  converted to a ``tuple`` before calling ``Predictor.predict``.
  Blank / whitespace-only lines are skipped silently. Extra keys in a
  query object are ignored. There are no per-query knob overrides —
  ``n_samples`` / ``seed`` / ``ci_level`` are fixed for the whole
  stream from the CLI flags at ``Predictor`` construction time
  ("persistent" = one session, one set of knobs, many queries).
- **E5.** Stream-mode errors propagate; nothing is swallowed. A
  malformed JSON line (``json.JSONDecodeError``), a query object
  missing a required key (``KeyError``), or any exception
  ``Predictor.predict(...)`` itself raises propagates uncaught out of
  ``main()`` and terminates the stream — lines already printed stay on
  stdout, nothing after the failing line is processed. No per-line
  ``try``/``except``-and-continue.
- **E6.** Stream-mode output format; the one-shot path stays
  untouched. Each stream result prints as one compact JSON line —
  ``json.dumps(result.to_dict(), sort_keys=True)``, no ``indent=``
  (an indented multi-line object would break the one-line-per-result
  JSONL contract) — via ``print(..., flush=True)`` so a piped
  consumer sees results incrementally. This differs from the existing
  one-shot path's pretty-printed ``indent=2`` output, which is
  unchanged. The one-shot branch of ``main()`` keeps calling
  :func:`make_predictor` directly (not through :class:`Predictor`) —
  this preserves ``test_main_prints_json_result``'s existing
  ``monkeypatch.setattr(pred, "make_predictor", stub)`` with zero
  behaviour change (a one-shot process only ever calls ``predict``
  once, so there is nothing to amortise).

**Probability order.** Every per-map 4-vector and every interval band
in this module is in :data:`models._shared.OUTCOME_LABELS` order —
``("A-regulation", "A-OT", "B-OT", "B-regulation")``; every scoreline
vector is in ``utils.series_paths.series_outcome_order`` order (the
``(a_wins, b_wins)`` terminal scorelines from A's most dominant win to
B's).

**Prerequisite artifacts / tables.** ``matches.parquet``,
``maps.parquet``, ``player_map_stats.parquet``,
``ordinal_logit_model.json``, ``temperature_scaling_model.json``,
``conditional_logit_ban_model.json`` and
``conditional_logit_pick_model.json`` for the requested version (i.e.
``materialize.py`` and the four training drivers have been run). A
missing artifact raises ``FileNotFoundError`` unchanged — the standard
"run the prerequisite first" signal. ``bootstrap_models`` (D4), when
provided, are caller-supplied already-fitted raw ordinal replicates;
they are never fitted or persisted here.

Exit codes (CLI):

- ``0`` — always. The hard failures are raises instead, mirroring the
  rest of ``drivers/``'s raise-for-invariant-break doctrine.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drivers import evaluate
from evaluation import (
    bootstrap_intervals,
    veto_conditional_variance,
    veto_marginalized_series,
)
from features import map_win_rate
from models import (
    conditional_logit_ban,
    conditional_logit_pick,
    greedy_veto_simulator,
    ordinal_logit,
)
from models.greedy_veto_simulator import SimulatedVetoAction
from models.ordinal_logit import OrdinalLogitModel
from utils.table_io import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# The registry key of the production calibrated map model (D3): the
# M24 temperature-scaled ordinal-logit factory in drivers/evaluate.py,
# which loads the base ordinal + temperature artifacts, enforces the
# staleness guard, and returns the 6-arg ModelFn this module closes
# over as the per-map point model and the M31 map_model_fn.
_TEMPERATURE_MAP_MODEL_KEY = "ordinal_logit_temperature"

# The M31 sampling / spread knobs (D7): DEFAULT_N_SAMPLES is M37's
# measured wall-clock choice (~1.5s per sampled veto sequence per
# series on real v1 fitted models — a stable *spread* estimate needs
# more draws than a stable *mean*), DEFAULT_SEED matches the repo's
# "current year" seed convention already used by drivers/evaluate_series.py,
# and DEFAULT_CI_LEVEL mirrors M36/M37's 0.90 convention (5th/95th
# percentile bands).
DEFAULT_N_SAMPLES = 30
DEFAULT_SEED = 2026
DEFAULT_CI_LEVEL = 0.90

# The names this module exposes publicly: the four result dataclasses,
# the factory that returns the documented 5-arg predict closure (the
# closure itself is not a module-level name), and the E1 session-holding
# Predictor wrapper.
__all__ = [
    "PerMapPrediction",
    "PredictionResult",
    "Predictor",
    "SeriesPrediction",
    "VetoSensitivity",
    "make_predictor",
]


@dataclass(frozen=True)
class PerMapPrediction:
    """The per-map prediction record for one played map of a predicted series.

    The ``per_map`` entry type of :class:`PredictionResult`: one played
    map's temperature-scaled four-way point probabilities (D3, in
    :data:`models._shared.OUTCOME_LABELS` order), the optional
    epistemic interval over the raw-ordinal bootstrap replicates (D4:
    ``None``/``None`` when no ``bootstrap_models`` were supplied), and
    the weaker side's as-of per-map game count backing the prediction.

    Attributes:
        map_name: The played map's normalized name (a ``"pick"`` or
            ``"decider"`` map of the predicted greedy veto, in play
            order).
        probabilities: The four temperature-scaled map-outcome
            probabilities in
            :data:`models._shared.OUTCOME_LABELS` order
            (``p_a_regulation, p_a_ot, p_b_ot, p_b_regulation``),
            summing to approximately 1.
        interval_low: The four per-category lower band endpoints over
            the bootstrap replicate models (raw ordinal, M36's
            definition) in the same order; ``None`` when no
            ``bootstrap_models`` were supplied to :func:`make_predictor`.
        interval_high: The four per-category upper band endpoints over
            the same replicate models; ``None`` alongside
            ``interval_low``.
        n_games_backing: ``min(games_a, games_b)`` — the weaker side's
            as-of, map-specific game count backing this prediction
            (:func:`evaluation.bootstrap_intervals.n_games_backing`).
    """

    map_name: str
    probabilities: tuple[float, float, float, float]
    interval_low: tuple[float, float, float, float] | None
    interval_high: tuple[float, float, float, float] | None
    n_games_backing: int

    def to_dict(self) -> dict[str, object]:
        """Serialize this per-map prediction to a JSON-compatible dict.

        Returns:
            A dict with keys ``"map_name"`` (str), ``"probabilities"``
            (list of 4 floats), ``"interval_low"`` /
            ``"interval_high"`` (list of 4 floats, or ``None`` when no
            bootstrap models backed the interval) and
            ``"n_games_backing"`` (int), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "map_name": self.map_name,
            "probabilities": list(self.probabilities),
            "interval_low": (
                None
                if self.interval_low is None
                else list(self.interval_low)
            ),
            "interval_high": (
                None
                if self.interval_high is None
                else list(self.interval_high)
            ),
            "n_games_backing": self.n_games_backing,
        }


@dataclass(frozen=True)
class SeriesPrediction:
    """The veto-marginalised series scoreline prediction (M31 aggregate).

    The ``series`` entry of :class:`PredictionResult`: the
    probability-weighted average scoreline distribution across the
    sampled veto sequences (M31's aggregate — D5), in
    ``utils.series_paths.series_outcome_order`` order, plus the
    outcome-order vocabulary and the parsed ``best_of`` map count so a
    consumer can read the vector without cross-referencing the call.

    Attributes:
        probabilities: The ``best_of + 1`` scoreline probabilities in
            ``outcome_order`` order, summing to 1 within float error.
        outcome_order: The ``best_of + 1`` terminal ``(a_wins, b_wins)``
            scorelines in canonical order
            (``utils.series_paths.series_outcome_order``).
        best_of: The parsed map count (``3`` for ``"Bo3"``, ``5`` for
            ``"Bo5"``, ``1`` for ``"Bo1"``).
    """

    probabilities: tuple[float, ...]
    outcome_order: tuple[tuple[int, int], ...]
    best_of: int

    def to_dict(self) -> dict[str, object]:
        """Serialize this series prediction to a JSON-compatible dict.

        Returns:
            A dict with keys ``"probabilities"`` (list of
            ``best_of + 1`` floats), ``"outcome_order"`` (list of
            ``[a_wins, b_wins]`` int pairs in canonical order) and
            ``"best_of"`` (int), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "probabilities": list(self.probabilities),
            "outcome_order": [
                list(scoreline) for scoreline in self.outcome_order
            ],
            "best_of": self.best_of,
        }


@dataclass(frozen=True)
class VetoSensitivity:
    """The structural (M37) spread summary across the sampled veto sequences.

    The ``veto_sensitivity`` entry of :class:`PredictionResult` — the
    M37 summary of the *same* M31 per-sample detail that produced
    ``series`` (D5, one sampling pass). Unweighted per-category
    percentile bands over the ``n_samples`` ancestral draws are the
    primary metric (each draw is already sampled proportionally to its
    own ``sequence_probability``); the band widths and their mean are
    derived from them; the weighted mean/variance is the
    explicitly-flagged secondary metric using M31's own normalized
    per-sample ``weight`` values. The bands are marginal per category,
    **not** a joint simplex credible region.

    Attributes:
        unweighted_band_low: The per-category lower band endpoints
            over the sampled veto draws (length ``best_of + 1``).
        unweighted_band_high: The per-category upper band endpoints
            (length ``best_of + 1``).
        band_widths: The per-category ``hi - lo`` widths (length
            ``best_of + 1``).
        mean_band_width: The mean of :attr:`band_widths` — the single
            per-series scalar headline "how much does the veto sequence
            move the series outcome" number; exactly ``0.0`` when every
            sampled veto sequence produced the identical scoreline
            distribution.
        weighted_mean: The per-category weighted means over the sample
            rows using M31's normalized per-sample ``weight`` values.
        weighted_variance: The per-category weighted population
            variances about the weighted mean, same weights.
    """

    unweighted_band_low: tuple[float, ...]
    unweighted_band_high: tuple[float, ...]
    band_widths: tuple[float, ...]
    mean_band_width: float
    weighted_mean: tuple[float, ...]
    weighted_variance: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize this veto-sensitivity summary to a JSON-compatible dict.

        Returns:
            A dict with keys ``"unweighted_band_low"``,
            ``"unweighted_band_high"``, ``"band_widths"``,
            ``"weighted_mean"`` and ``"weighted_variance"`` (each a
            list of ``best_of + 1`` floats) and ``"mean_band_width"``
            (a float), all plain JSON types.

        Raises:
            Nothing.
        """
        return {
            "unweighted_band_low": list(self.unweighted_band_low),
            "unweighted_band_high": list(self.unweighted_band_high),
            "band_widths": list(self.band_widths),
            "mean_band_width": self.mean_band_width,
            "weighted_mean": list(self.weighted_mean),
            "weighted_variance": list(self.weighted_variance),
        }


@dataclass(frozen=True)
class PredictionResult:
    """The full M39 ``predict()`` result for one queried match.

    The top-level return of the documented public API: the
    deterministic predicted veto sequence (D2), one
    :class:`PerMapPrediction` per played map in play order, the
    veto-marginalised :class:`SeriesPrediction` (D5), and the
    structural :class:`VetoSensitivity` summary (D5, from the same M31
    sampling pass).

    Attributes:
        predicted_veto: The full deterministic greedy-veto action tuple
            (``"ban"``/``"pick"``/``"decider"`` steps in step order,
            length 7 for every supported ``best_of``).
        per_map: One :class:`PerMapPrediction` per played map, in play
            order (the greedy sequence's ``pick`` steps ascending,
            then the forced ``decider`` map — ``best_of`` entries).
        series: The veto-marginalised series scoreline prediction.
        veto_sensitivity: The structural spread summary across the
            sampled veto sequences.
    """

    predicted_veto: tuple[SimulatedVetoAction, ...]
    per_map: tuple[PerMapPrediction, ...]
    series: SeriesPrediction
    veto_sensitivity: VetoSensitivity

    def to_dict(self) -> dict[str, object]:
        """Serialize this full prediction result to a JSON-compatible dict.

        Nests the four sub-records' own ``to_dict`` outputs: the
        ``predicted_veto`` actions serialize via
        :meth:`SimulatedVetoAction.to_dict`, the ``per_map`` entries
        via :meth:`PerMapPrediction.to_dict`, and the ``series`` /
        ``veto_sensitivity`` records via their own ``to_dict`` methods.

        Returns:
            A dict with keys ``"predicted_veto"`` (list of action
            dicts), ``"per_map"`` (list of per-map dicts), ``"series"``
            (dict) and ``"veto_sensitivity"`` (dict), all plain JSON
            types.

        Raises:
            Nothing.
        """
        return {
            "predicted_veto": [
                action.to_dict() for action in self.predicted_veto
            ],
            "per_map": [entry.to_dict() for entry in self.per_map],
            "series": self.series.to_dict(),
            "veto_sensitivity": self.veto_sensitivity.to_dict(),
        }


def _load_veto_models(
    output_dir: Path, version: str
) -> tuple[object, object]:
    """Load the two fitted Stage-1 veto-step model artifacts.

    Reconstructs the fitted M27 conditional-logit ban model from
    ``conditional_logit_ban_model.json`` and the fitted M28
    conditional-logit pick model from ``conditional_logit_pick_model.json``
    via each module's own ``from_dict`` — the in-driver artifact-loader
    convention the repo's ``evaluate_*.py`` drivers follow (each driver
    independently duplicates its small loader helper rather than
    importing a sibling driver's; a shared loading spot is a flagged
    future refactor, out of scope here). The two ``from_dict`` calls
    are deliberately independent of each other and of the input tables,
    so a missing artifact fails fast with the standard "run the
    training driver first" signal. Note the ordinal / temperature
    artifacts are *not* loaded here — the temperature-scaled Stage-2
    map model comes from
    :data:`drivers.evaluate.MODEL_REGISTRY`'s
    ``"ordinal_logit_temperature"`` factory (which loads and guards
    them itself), per D3.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")``).
        version: The dataset version subdirectory name (e.g. ``"v1"``).

    Returns:
        A ``(ban_model, pick_model)`` tuple of the two deserialized
        fitted models, in the order :func:`make_predictor` wires them
        into the ``predictor_fn_by_action`` dict.

    Raises:
        FileNotFoundError: If either of the two model artifacts does
            not exist for the requested version (i.e. the corresponding
            training driver has not been run for it) — propagated
            unchanged from the file read as a clear "run the training
            driver first" signal, never wrapped or silently skipped.
        KeyError: If an artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        ValueError: If an artifact's shapes are inconsistent (propagated
            from the ``from_dict`` calls).
    """
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
    return ban_model, pick_model


def _n_games_backing_for_map(
    team_a_id: str,
    team_b_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> int:
    """Compute one played map's ``n_games_backing`` via the shared estimator.

    Queries :func:`features.map_win_rate.team_map_win_rate` for both
    teams (with :data:`features.map_win_rate.DEFAULT_K` — the same
    as-of cutoff Stage 2's own ``map_win_rate_diff`` feature uses; the
    shrinkage ``k`` only affects the shrunk ``mean``/``variance``,
    never the ``games`` count, so the specific ``k`` does not change
    the result) and returns
    :func:`evaluation.bootstrap_intervals.n_games_backing` —
    ``min(games_a, games_b)``, the weaker side's as-of, map-specific
    sample size backing this prediction. Mirrors
    ``drivers/evaluate_bootstrap_intervals.py::_games_backing_for_map``
    exactly, copied (not imported) per the repo's per-driver-loader
    convention; M36 records that ``min`` (not ``sum``) is chosen so a
    data-rich side never overstates confidence against a brand-new
    opponent.

    Args:
        team_a_id: The queried team A's stable id.
        team_b_id: The queried team B's stable id.
        map_name: The played map to query backing for.
        date: The as-of cutoff (the queried match's own date; strict
            ``<``).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.

    Returns:
        ``min(team_a_games, team_b_games)`` as an ``int``; ``0`` when
        either side has no as-of games on that map.

    Raises:
        ValueError: If the query date is null/unparseable/timezone-aware
            or an as-of map has a tied/null score (propagated from
            :func:`features.map_win_rate.team_map_win_rate`).
        KeyError: If either table lacks a required column (propagated
            from the same call).
        TypeError: If the query date is list-like (propagated from the
            same call).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from the same call).
    """
    games_a = map_win_rate.team_map_win_rate(
        team_a_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).games
    games_b = map_win_rate.team_map_win_rate(
        team_b_id,
        map_name,
        date,
        matches_df,
        maps_df,
        map_win_rate.DEFAULT_K,
    ).games
    return bootstrap_intervals.n_games_backing(games_a, games_b)


def _veto_sensitivity_from_prediction(
    prediction: veto_marginalized_series.VetoMarginalizedSeriesPrediction,
    ci_level: float,
) -> VetoSensitivity:
    """Summarize one M31 prediction's per-sample spread into a VetoSensitivity.

    Builds the ``(n_samples, best_of + 1)`` sample-row matrix from
    ``prediction.samples[i].scoreline_probabilities`` and the parallel
    weight vector from ``prediction.samples[i].weight`` (the exact
    deterministic per-sample scoreline detail M31 already returns —
    D5: the same M31 call that produced the aggregate also produces
    this structural summary; one sampling pass, not two), then
    computes the M37 summary via the pure helpers in
    :mod:`evaluation.veto_conditional_variance`: the unweighted
    per-category percentile bands
    (:func:`evaluation.veto_conditional_variance.unweighted_scoreline_spread`),
    their widths (:func:`evaluation.veto_conditional_variance.band_widths`),
    the per-series scalar mean width
    (:func:`evaluation.veto_conditional_variance.mean_band_width`),
    and the explicitly-flagged secondary weighted mean/variance
    (:func:`evaluation.veto_conditional_variance.weighted_mean_and_variance`).
    Mirrors
    ``drivers/evaluate_veto_conditional_variance.py::_series_spread_record``
    except it returns the :class:`VetoSensitivity` dataclass rather
    than a report dict. The point estimate (``prediction.probabilities``)
    is *not* included here — the caller carries it as
    ``series.probabilities``.

    Args:
        prediction: The M31 prediction whose per-sample detail is
            summarized; must carry ``best_of + 1``-length
            ``scoreline_probabilities`` per sample and one ``weight``
            per sample (as :func:`evaluation.veto_marginalized_series
            .predict_series_outcome_via_veto_marginalization` returns).
        ci_level: The band level in ``(0, 1)``, passed through to
            :func:`evaluation.veto_conditional_variance
            .unweighted_scoreline_spread`.

    Returns:
        A :class:`VetoSensitivity` with the per-category unweighted
        band endpoints (``unweighted_band_low`` /
        ``unweighted_band_high``), the per-category ``hi - lo``
        ``band_widths``, their ``mean_band_width`` (exactly ``0.0``
        when every sampled veto sequence produced the identical
        scoreline distribution), and the per-category ``weighted_mean``
        / ``weighted_variance`` using the samples' own normalized
        weights.

    Raises:
        ValueError: If a sample's scoreline vector length is
            inconsistent with the others, if ``n_samples`` is zero, or
            if the weight vector mismatches the rows / is malformed
            (all propagated from the pure helpers in
            :mod:`evaluation.veto_conditional_variance`).
    """
    rows = [
        list(sample.scoreline_probabilities)
        for sample in prediction.samples
    ]
    weights = [float(sample.weight) for sample in prediction.samples]
    bands = veto_conditional_variance.unweighted_scoreline_spread(
        rows, ci_level=ci_level
    )
    weighted_means, weighted_variances = (
        veto_conditional_variance.weighted_mean_and_variance(
            rows, weights
        )
    )
    return VetoSensitivity(
        unweighted_band_low=tuple(lo for lo, _hi in bands),
        unweighted_band_high=tuple(hi for _lo, hi in bands),
        band_widths=veto_conditional_variance.band_widths(bands),
        mean_band_width=veto_conditional_variance.mean_band_width(bands),
        weighted_mean=weighted_means,
        weighted_variance=weighted_variances,
    )


def make_predictor(
    output_dir,
    version: str,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
    bootstrap_models: Sequence[OrdinalLogitModel] | None = None,
):
    """Build the documented 5-argument ``predict`` closure for one version.

    The D1 factory: loads the three materialised tables
    (``matches``/``maps``/``player_map_stats`` via the
    ``drivers.evaluate`` loaders) and the four fitted artifacts once,
    and returns a closure with exactly the documented public signature
    ``predict(team_a, team_b, best_of, map_pool, as_of_date)``. The
    Stage-2 map model is the M24 temperature-scaled ordinal from
    :data:`drivers.evaluate.MODEL_REGISTRY`'s
    ``"ordinal_logit_temperature"`` key (D3 — includes the
    temperature/base-model staleness guard; a mismatched pair raises
    ``ValueError`` at factory time), the Stage-1 ban/pick models come
    from :func:`_load_veto_models`, and the
    ``predictor_fn_by_action`` dict is wired from them via each
    module's ``make_veto_step_predictor_fn``. The ``n_samples`` /
    ``seed`` / ``ci_level`` knobs are closed over per-call (D6: every
    ``predict`` call reconstructs ``numpy.random.default_rng(seed)``,
    so identical calls reproduce identical output), and the optional
    ``bootstrap_models`` (D4) are closed over for the per-map
    epistemic intervals.

    Args:
        output_dir: The parent directory the version subdirectory
            lives under (e.g. ``Path("data")`` or the string
            ``"data"``); coerced to a ``Path``.
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        n_samples: How many M29 veto walks each ``predict`` call
            samples for the M31 pipeline (D7: default
            :data:`DEFAULT_N_SAMPLES`, M37's measured wall-clock
            default). Must be a positive integer; enforced by the M31
            sampler at call time (propagated ``ValueError``).
        seed: The per-call ``numpy.random.default_rng`` seed (D7,
            repo convention; default :data:`DEFAULT_SEED`).
        ci_level: The interval/spread level in ``(0, 1)`` for the
            per-map epistemic bands and the veto-sensitivity bands
            (default :data:`DEFAULT_CI_LEVEL`); validated here at
            factory time.
        bootstrap_models: The optional already-fitted raw ordinal
            bootstrap replicate models (D4) the per-map epistemic
            intervals are computed over; ``None`` (the default) or an
            empty sequence means ``per_map[i].interval_*`` is
            ``None``. Replicate models are consumed, never fitted or
            persisted here.

    Returns:
        The 5-argument ``predict(team_a, team_b, best_of, map_pool,
        as_of_date) -> PredictionResult`` closure (D1).

    Raises:
        FileNotFoundError: If any of the required tables/artifacts does
            not exist for the requested version (i.e. the
            ``materialize.py`` / training drivers have not been run) —
            propagated unchanged from the loaders/factories as a clear
            "run the prerequisite first" signal.
        ValueError: If ``ci_level`` is not in ``(0, 1)``; if the
            temperature-scaling artifact was calibrated against a
            different base ordinal artifact (the staleness guard in
            the registry factory); or if any artifact dict is malformed
            (propagated from the ``from_dict`` calls).
        KeyError: If any artifact dict lacks a required key (propagated
            from the ``from_dict`` calls).
        TypeError: If an input type is invalid (propagated from the
            loaders).
    """
    if n_samples < 1:
        raise ValueError(
            f"n_samples must be a positive integer, got {n_samples}"
        )
    if not (0.0 < ci_level < 1.0):
        raise ValueError(
            f"ci_level must be strictly between 0 and 1, got {ci_level}"
        )

    output_dir = Path(output_dir)
    matches_df = evaluate.load_matches_table(output_dir, version)
    maps_df = evaluate.load_maps_table(output_dir, version)
    player_map_stats_df = evaluate.load_player_map_stats_table(
        output_dir, version
    )

    # The production calibrated Stage-2 map model (D3): the registry
    # factory loads the base ordinal + temperature artifacts, enforces
    # the staleness guard, and returns the 6-arg ModelFn this closure
    # uses both for the per-map point probabilities and as the M31
    # map_model_fn. The player_map_stats table it closes over is the
    # same one loaded above (the registry factory loads its own copy
    # for the closure; the one we load is additionally needed for the
    # D4 interval replicates).
    map_model_fn = evaluate.MODEL_REGISTRY[_TEMPERATURE_MAP_MODEL_KEY](
        output_dir, version
    )

    # The fixed Stage-1 predictors (D2): the greedy simulator is
    # deterministic and needs no predictors, but the M31 sampler does.
    ban_model, pick_model = _load_veto_models(output_dir, version)
    predictor_fn_by_action = {
        "ban": conditional_logit_ban.make_veto_step_predictor_fn(ban_model),
        "pick": conditional_logit_pick.make_veto_step_predictor_fn(pick_model),
    }

    def predict(
        team_a: str,
        team_b: str,
        best_of: str,
        map_pool,
        as_of_date: str,
    ) -> PredictionResult:
        """Predict one match's veto, per-map, series and veto sensitivity.

        The documented M39 public API (D1): for one queried match —
        two stable ``team_id`` strings, a ``"Bo<N>"`` series length, an
        optional 7-map pool, and an as-of date — returns the full
        :class:`PredictionResult`. Runs (a) the deterministic M25
        greedy veto via
        :func:`models.greedy_veto_simulator.simulate_veto` (D2, using
        :data:`features.map_win_rate.DEFAULT_K` and the closed-over
        tables); (b) one temperature-scaled four-way prediction per
        played map (the greedy sequence's ``pick``/``decider`` maps in
        step order), with each map's ``n_games_backing`` via
        :func:`_n_games_backing_for_map` and, when
        ``bootstrap_models`` was supplied to :func:`make_predictor`,
        the epistemic interval over the raw-ordinal replicate models'
        four-way predictions (D4); and (c) **one** M31 call
        (:func:`evaluation.veto_marginalized_series
        .predict_series_outcome_via_veto_marginalization` — D5) with
        the temperature-scaled ``map_model_fn``, the closed-over
        ``predictor_fn_by_action``, ``n_samples``, a **fresh**
        ``numpy.random.default_rng(seed)`` per call (D6), and the
        caller's ``map_pool`` passed straight through (D8, ``None``
        resolves the era pool from config), whose aggregate becomes
        ``series`` and whose per-sample detail is summarized into
        ``veto_sensitivity`` via :func:`_veto_sensitivity_from_prediction`.

        Args:
            team_a: The queried team A's stable id (side A of the
                scoreline vocabulary; the even-step veto actor).
            team_b: The queried team B's stable id (side B; the
                odd-step veto actor).
            best_of: The series length as the ``"Bo<N>"`` string
                (``"Bo1"``/``"Bo3"``/``"Bo5"``); anything else raises
                ``ValueError`` from the greedy simulator (which only
                supports the :data:`models.greedy_veto_simulator
                .ACTION_SEQUENCES` keys).
            map_pool: The pool to veto over, as an iterable of map
                names; ``None`` resolves the active era's pool from
                ``config.json`` for ``as_of_date``'s calendar date
                (D8). Every supported format requires a 7-map pool, so
                a pool of any other size raises ``ValueError`` from the
                simulator/sampler — propagated, not re-validated here.
            as_of_date: The as-of cutoff for every feature lookup and
                the era-pool resolution (e.g. the queried match's own
                ISO-8601 timestamp; strict ``<``).

        Returns:
            A :class:`PredictionResult` carrying the full 7-action
            deterministic ``predicted_veto`` (D2), one
            :class:`PerMapPrediction` per played map in play order
            (the greedy sequence's ``pick`` steps ascending, then the
            forced ``decider`` map), the veto-marginalised
            :class:`SeriesPrediction`, and the structural
            :class:`VetoSensitivity` (both from the same single M31
            call, D5). Identical arguments reproduce an identical
            result (D6).

        Raises:
            ValueError: If ``best_of`` is not a supported veto format,
                if ``map_pool`` has the wrong size or contains
                duplicates after normalization, if an as-of map has a
                null/NaN/tied score or ``k`` is invalid (all from
                :func:`models.greedy_veto_simulator.simulate_veto` /
                the M31 sampler — propagated); if ``as_of_date`` is
                null/unparseable/timezone-aware (from ``utils.asof``);
                if a played map's four-way vector is mis-sized (from
                the M31 scorer); or if the M31 sample set is a
                degenerate all-zero-probability set (from the M31
                aggregator). If ``ci_level``/``n_samples`` were invalid
                they were already rejected by :func:`make_predictor`.
            ConfigError: If ``map_pool`` is ``None`` and no configured
                era covers ``as_of_date``'s calendar date, or a map
                name is not a string (from ``utils.config`` —
                propagated).
            KeyError / TypeError: Propagated from the feature
                builders/predictors if a required table column is
                absent or a callable misbehaves.
        """
        # (a) The deterministic greedy veto (D2): keep the full
        # 7-action tuple in step order.
        predicted_veto = tuple(
            greedy_veto_simulator.simulate_veto(
                team_a,
                team_b,
                best_of,
                as_of_date,
                matches_df,
                maps_df,
                k=map_win_rate.DEFAULT_K,
                map_pool=map_pool,
            )
        )

        # The played maps, in play order: the greedy sequence's
        # "pick"/"decider" actions in step order (bans are never
        # played) — D2. The sequence is already step-ordered, so a
        # plain order-preserving filter gives the picks ascending then
        # the forced decider last.
        played_maps = tuple(
            action.map_name
            for action in predicted_veto
            if action.action in ("pick", "decider")
        )

        # (b) Per-map point probabilities + backing + optional
        # epistemic interval.
        per_map_entries: list[PerMapPrediction] = []
        for map_name in played_maps:
            probabilities = tuple(
                float(p)
                for p in map_model_fn(
                    team_a,
                    team_b,
                    map_name,
                    as_of_date,
                    matches_df,
                    maps_df,
                )
            )
            if bootstrap_models:
                # D4: the interval is over the raw ordinal replicates'
                # four-way predictions (M36's definition), while the
                # point estimate above is temperature-scaled (D3) —
                # the same asymmetry M36 records; kept and stated here.
                replicate_rows = [
                    tuple(
                        ordinal_logit.make_model_fn(
                            bootstrap_model, player_map_stats_df
                        )(
                            team_a,
                            team_b,
                            map_name,
                            as_of_date,
                            matches_df,
                            maps_df,
                        )
                    )
                    for bootstrap_model in bootstrap_models
                ]
                bands = bootstrap_intervals.replicate_matrix_intervals(
                    replicate_rows, ci_level=ci_level
                )
                interval_low = tuple(lo for lo, _hi in bands)
                interval_high = tuple(hi for _lo, hi in bands)
            else:
                interval_low = None
                interval_high = None
            per_map_entries.append(
                PerMapPrediction(
                    map_name=map_name,
                    probabilities=probabilities,
                    interval_low=interval_low,
                    interval_high=interval_high,
                    n_games_backing=_n_games_backing_for_map(
                        team_a,
                        team_b,
                        map_name,
                        as_of_date,
                        matches_df,
                        maps_df,
                    ),
                )
            )

        # (c) One M31 call (D5) — a fresh per-call rng (D6), the
        # temperature-scaled map model (D3) and the caller's map_pool
        # passed straight through (D8).
        prediction = (
            veto_marginalized_series.predict_series_outcome_via_veto_marginalization(
                team_a,
                team_b,
                best_of,
                as_of_date,
                matches_df,
                maps_df,
                map_model_fn,
                predictor_fn_by_action,
                n_samples=n_samples,
                rng=np.random.default_rng(seed),
                map_pool=map_pool,
            )
        )
        series = SeriesPrediction(
            probabilities=tuple(prediction.probabilities),
            outcome_order=prediction.outcome_order,
            best_of=prediction.best_of,
        )
        veto_sensitivity = _veto_sensitivity_from_prediction(
            prediction, ci_level=ci_level
        )

        return PredictionResult(
            predicted_veto=predicted_veto,
            per_map=tuple(per_map_entries),
            series=series,
            veto_sensitivity=veto_sensitivity,
        )

    return predict


class Predictor:
    """The E1 session-holding wrapper around the M39 ``make_predictor`` factory.

    A thin persistent object for the M39.1 lifecycle milestone: loading
    the materialised tables and fitted artifacts **once** at
    construction (delegating to :func:`make_predictor` exactly once and
    holding the returned 5-arg closure for the object's lifetime) so a
    process can answer many :meth:`predict` calls without re-loading
    per call. No prediction semantics change — this is lifecycle
    plumbing, not modeling: a ``Predictor`` instance's
    :meth:`predict` call is bitwise identical to calling
    :func:`make_predictor` once and invoking the returned closure with
    the same arguments, and the D6 per-call fresh-RNG idempotence
    (identical arguments reproduce identical output) is inherited
    unchanged since the wrapped closure is exactly what
    :func:`make_predictor` already returns. ``make_predictor`` itself
    is untouched (its body is not refactored or reordered), so its
    reviewed-clean tests keep passing unmodified.

    ``bootstrap_models`` (D4), when given, are forwarded unchanged to
    :func:`make_predictor` for the per-map epistemic intervals. The
    class is the object behind the CLI's persistent ``--stream`` mode:
    one ``Predictor`` built once per process answers the whole JSONL
    query stream (E4).
    """

    def __init__(
        self,
        output_dir,
        version: str,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        seed: int = DEFAULT_SEED,
        ci_level: float = DEFAULT_CI_LEVEL,
        bootstrap_models: Sequence[OrdinalLogitModel] | None = None,
    ) -> None:
        """Construct one Predictor by loading tables/artifacts exactly once.

        E1: calls :func:`make_predictor` exactly once with every
        keyword forwarded unchanged and stores the returned 5-arg
        ``predict`` closure privately; every later :meth:`predict` call
        reuses that loaded state. All validation is delegated to the
        factory — this wrapper adds none of its own (its constructor
        raises exactly what :func:`make_predictor` raises at factory
        time, propagated unchanged).

        Args:
            output_dir: The parent directory the version subdirectory
                lives under (e.g. ``Path("data")`` or the string
                ``"data"``); coerced to a ``Path`` by
                :func:`make_predictor`.
            version: The dataset version subdirectory name (e.g.
                ``"v1"``).
            n_samples: How many M29 veto walks each :meth:`predict`
                call samples for the M31 pipeline (D7: default
                :data:`DEFAULT_N_SAMPLES`). Must be a positive
                integer; enforced by the factory (propagated
                ``ValueError``).
            seed: The per-call ``numpy.random.default_rng`` seed (D7,
                repo convention; default :data:`DEFAULT_SEED`).
            ci_level: The interval/spread level in ``(0, 1)`` (default
                :data:`DEFAULT_CI_LEVEL`); validated at factory time.
            bootstrap_models: The optional already-fitted raw ordinal
                bootstrap replicate models (D4) the per-map epistemic
                intervals are computed over; ``None`` (the default) or
                an empty sequence means ``per_map[i].interval_*`` is
                ``None``. Forwarded unchanged to :func:`make_predictor`.

        Returns:
            Nothing (the loaded state is held on the instance).

        Raises:
            FileNotFoundError: If any required table/artifact does not
                exist for the requested version (propagated unchanged
                from :func:`make_predictor`).
            ValueError: If ``ci_level`` is not in ``(0, 1)`` or the
                temperature/base staleness guard fires (propagated
                unchanged from :func:`make_predictor`).
            KeyError / TypeError: Propagated unchanged from
                :func:`make_predictor` for malformed artifacts or
                invalid input types.
        """
        self._predict_fn = make_predictor(
            output_dir,
            version,
            n_samples=n_samples,
            seed=seed,
            ci_level=ci_level,
            bootstrap_models=bootstrap_models,
        )

    def predict(
        self,
        team_a: str,
        team_b: str,
        best_of: str,
        map_pool,
        as_of_date: str,
    ) -> PredictionResult:
        """Predict one match, delegating to the wrapped 5-arg closure.

        E1: calls the privately-held closure that :func:`make_predictor`
        returned at construction time (loaded once, reused for every
        call) with the exact arguments passed, and returns its result
        unmodified. Bitwise identical to calling
        :func:`make_predictor` once and invoking the returned closure
        with the same arguments; D6's per-call fresh-RNG idempotence
        (identical arguments reproduce identical output) is inherited.

        Args:
            team_a: The queried team A's stable id.
            team_b: The queried team B's stable id.
            best_of: The series length as the ``"Bo<N>"`` string
                (``"Bo1"``/``"Bo3"``/``"Bo5"``).
            map_pool: The pool to veto over, as an iterable of map
                names; ``None`` resolves the active era's pool from
                ``config.json`` for ``as_of_date``'s calendar date
                (D8). Requires a 7-map pool in every supported format.
            as_of_date: The as-of cutoff for every feature lookup and
                the era-pool resolution (strict ``<``).

        Returns:
            The wrapped closure's :class:`PredictionResult` for the
            given arguments, unmodified.

        Raises:
            ValueError: Propagated unchanged from the wrapped closure
                (invalid ``best_of``, wrong-size/duplicate ``map_pool``,
                bad ``as_of_date``, degenerate M31 samples, etc.).
            ConfigError: Propagated unchanged from the wrapped closure
                (no configured era covers ``as_of_date``'s calendar
                date when ``map_pool`` is ``None``).
            KeyError / TypeError: Propagated unchanged from the wrapped
                closure (missing table column, misbehaving callable).
        """
        return self._predict_fn(
            team_a, team_b, best_of, map_pool, as_of_date
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the predict.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with ten attributes: ``team_a``
        (``str`` or ``None``, required unless ``--stream`` is given),
        ``team_b`` (``str`` or ``None``, required unless ``--stream``
        is given), ``best_of`` (``str`` or ``None``, required unless
        ``--stream`` is given; one of ``"Bo1"``/``"Bo3"``/``"Bo5"``
        when a value is present), ``as_of_date`` (``str`` or ``None``,
        required unless ``--stream`` is given), ``map_pool`` (``str``
        or ``None``, the optional comma-separated 7-map pool),
        ``stream`` (``bool``, default ``False`` — switches ``main()``
        into persistent JSONL query-stream mode, E2), ``version``
        (``str``, default ``"v1"``), ``output_dir`` (``str``, default
        ``"data"``), ``n_samples`` (``int``, default
        :data:`DEFAULT_N_SAMPLES`), ``seed`` (``int``, default
        :data:`DEFAULT_SEED`) and ``ci_level`` (``float``, default
        :data:`DEFAULT_CI_LEVEL`).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior)
            — an unknown flag, an unknown ``--best-of`` value (rejected
            by the ``choices=`` constraint, which only fires when a
            value is actually given), or a post-parse validation
            failure (E3): any of ``--team-a``/``--team-b``/
            ``--best-of``/``--as-of-date`` missing while ``--stream``
            is absent, or ``--stream`` combined with any of those four
            flags or ``--map-pool``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the M39 predict() public API for one queried match, or "
            "(--stream) answer a JSONL query stream from stdin with "
            "one persistent Predictor: load the materialised tables "
            "and fitted artifacts for a dataset version once and print "
            "each query's full prediction result "
            "(deterministic greedy veto, temperature-scaled per-map "
            "four-way probabilities with n_games_backing, the "
            "veto-marginalised series scoreline distribution, and the "
            "structural veto-sensitivity spread) as JSON."
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
        "--stream",
        action="store_true",
        default=False,
        help=(
            "persistent JSONL query-stream mode (E4): build one "
            "Predictor and answer one JSON query object per stdin line "
            "({\"team_a\", \"team_b\", \"best_of\", \"as_of_date\", "
            "\"map_pool\"?}), printing one compact JSON result per "
            "line; mutually exclusive with --team-a/--team-b/--best-of/"
            "--as-of-date/--map-pool, which must be omitted"
        ),
    )
    parser.add_argument(
        "--team-a",
        default=None,
        help=(
            "team A's stable team_id (e.g. 397), not its display name; "
            "required unless --stream is given"
        ),
    )
    parser.add_argument(
        "--team-b",
        default=None,
        help=(
            "team B's stable team_id (e.g. 6392), not its display name; "
            "required unless --stream is given"
        ),
    )
    parser.add_argument(
        "--best-of",
        default=None,
        choices=["Bo1", "Bo3", "Bo5"],
        help=(
            "series length (choices: Bo1/Bo3/Bo5); required unless "
            "--stream is given"
        ),
    )
    parser.add_argument(
        "--map-pool",
        default=None,
        help=(
            "optional comma-separated 7-map pool to veto over; when "
            "omitted the active era's pool for --as-of-date is resolved "
            "from config.json"
        ),
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help=(
            "as-of cutoff for every feature lookup (ISO-8601, e.g. "
            "2026-08-23T12:00:00; strictly-earlier data only); required "
            "unless --stream is given"
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            "M29 veto sequences sampled per predict call by the M31 "
            "pipeline (default: "
            f"{DEFAULT_N_SAMPLES} — M37's measured wall-clock default "
            "on real v1 fitted models: ~1.5s per sampled sequence per "
            "series, so a call at the default lands around 45s; a "
            "stable veto-sensitivity spread estimate needs more draws "
            "than a stable mean)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "seed for numpy.random.default_rng, reconstructed fresh per "
            "predict call so identical arguments reproduce identical "
            f"output (default: {DEFAULT_SEED})"
        ),
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help=(
            "interval/spread level in (0, 1): each per-category band "
            "spans the middle ci_level fraction of the replicate/sample "
            f"distribution (default: {DEFAULT_CI_LEVEL} — 5th/95th "
            "percentiles)"
        ),
    )
    args = parser.parse_args(argv)
    # E3: manual post-parse required-arg enforcement. --stream needs the
    # four query flags at their None defaults, and the one-shot path
    # needs all four present; each violation fires parser.error (a
    # SystemExit(2), matching argparse's own required-arg behaviour).
    if not args.stream and (
        args.team_a is None
        or args.team_b is None
        or args.best_of is None
        or args.as_of_date is None
    ):
        parser.error(
            "--team-a, --team-b, --best-of and --as-of-date are required "
            "unless --stream is given"
        )
    if args.stream and (
        args.team_a is not None
        or args.team_b is not None
        or args.best_of is not None
        or args.as_of_date is not None
        or args.map_pool is not None
    ):
        parser.error(
            "--stream cannot be combined with --team-a/--team-b/--best-of/"
            "--as-of-date/--map-pool; supply queries via stdin instead"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run one predict() call end to end and print the JSON result.

    Logging is configured first so the summary line is visible from the
    CLI. The ``--map-pool`` string (comma-separated, each part
    whitespace-stripped) is parsed into a tuple — ``None`` when the
    flag is absent (D8: the era pool for ``--as-of-date`` is resolved
    from config), and a malformed pool (an empty part, e.g. a leading/
    trailing/double comma) raises ``ValueError`` rather than silently
    changing the pool size. The predictor is built via
    :func:`make_predictor` — no bootstrap models (``bootstrap_models``
    is not a CLI flag; intervals are ``None`` from the CLI per D4) —
    and the 5-argument closure is called once with the query
    arguments. The full result is printed to stdout as
    ``json.dumps(result.to_dict(), indent=2, sort_keys=True)`` (the
    repo-wide artifact formatting), and a one-line summary is logged:
    the two teams, the series length, the as-of date, the number of
    played maps, the per-map backing counts, the A-side series win
    probability (the summed probability of the scorelines with
    ``a_wins > b_wins``) and the mean veto-sensitivity band width.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` always. There is no nonzero exit-code path: the hard
            failures are raises that propagate to the caller, matching
            the rest of ``drivers/``'s doctrine.

    Raises:
        FileNotFoundError: If any of the required tables/artifacts does
            not exist for the requested version (from
            :func:`make_predictor`) — propagated unchanged as the
            "run the prerequisite first" signal.
        ValueError: If ``--map-pool`` is malformed (an empty part), if
            ``--ci-level`` is out of ``(0, 1)`` or ``--n-samples`` is
            non-positive (from :func:`make_predictor`), or if the
            simulator/sampler rejects any query input (a non-7 or
            duplicate ``--map-pool``, an unsupported ``--best-of`` —
            though argparse ``choices=`` already constrains it — a
            bad ``--as-of-date``, degenerate M31 samples) — all
            propagated from the pipeline.
        KeyError / TypeError / ConfigError: Propagated from the
            pipeline (see :func:`make_predictor` / the ``predict``
            closure docstrings).
        OSError / TypeError: If the JSON cannot be printed (propagated
            from ``json.dumps``).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.map_pool is None:
        map_pool = None
    else:
        parts = [part.strip() for part in args.map_pool.split(",")]
        if any(part == "" for part in parts):
            raise ValueError(
                f"--map-pool must be a comma-separated list of map "
                f"names with no empty entries, got {args.map_pool!r}"
            )
        map_pool = tuple(parts)

    predictor = make_predictor(
        Path(args.output_dir),
        args.version,
        n_samples=args.n_samples,
        seed=args.seed,
        ci_level=args.ci_level,
    )
    result = predictor(
        args.team_a, args.team_b, args.best_of, map_pool, args.as_of_date
    )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    a_side_series_win = sum(
        probability
        for probability, (a_wins, b_wins) in zip(
            result.series.probabilities, result.series.outcome_order
        )
        if a_wins > b_wins
    )
    logger.info(
        "predict %s vs %s (%s) as of %s on %d played map(s) "
        "(%s/%s, n_samples=%d seed=%d ci_level=%.2f): "
        "per-map n_games_backing=%s, P(A wins series)=%.4f, "
        "mean veto-sensitivity band width %.6f",
        args.team_a,
        args.team_b,
        args.best_of,
        args.as_of_date,
        len(result.per_map),
        Path(args.output_dir),
        args.version,
        args.n_samples,
        args.seed,
        args.ci_level,
        [entry.n_games_backing for entry in result.per_map],
        a_side_series_win,
        result.veto_sensitivity.mean_band_width,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
