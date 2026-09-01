"""Veto-marginalised series prediction (roadmap M31).

Predicts a full series scoreline distribution by marginalising the
Stage-1 veto process: sample ``n_samples`` full veto sequences via
M29's :func:`models.ancestral_veto_sampler.sample_veto_sequences`,
score each sampled sequence's *played* maps with a pluggable Stage-2
four-way per-map model (the natural wiring is M20's
``models.ordinal_logit.make_model_fn`` closure), collapse each map's
four-way distribution to a binary A-win probability (regulation + OT
per side), feed the per-map binary vector through M30's
``utils.series_paths.series_probabilities_in_order`` to get that
sample's exact scoreline distribution, and average across the M29
samples *weighted by sequence probability* per the roadmap's literal
wording. The four-way detail is retained for reporting per sample, not
propagated into the scoreline (the roadmap's explicit requirement).

Design decisions (recorded here, do not re-derive in later
milestones):

1. **Module placement: ``evaluation/``, not ``models/``.** This
   deviates from M32's precedent (``models/flat_series_baseline.py``
   stayed in ``models/``) and needs the same explicit treatment M29 /
   M32 / M33a each gave their own placement reasoning. The reason is
   the module-boundary DAG, not style: ``models/`` modules may depend
   downward on ``features.*`` / ``utils.*`` only, with the one lateral
   exception ``models/_shared.py`` (``tests/test_module_boundaries.py``
   forbids any other ``from models.`` statement). M31 must call M29's
   ``models.ancestral_veto_sampler.sample_veto_sequences`` (to get the
   sampled sequences) *and* score a Stage-2 four-way per-map model —
   if that Stage-2 model itself lived in ``models/`` (M20's
   ``models.ordinal_logit``), invoking it directly from another
   ``models/`` module would be exactly the forbidden
   ``models/ -> models/`` lateral edge (only ``models._shared`` is
   exempt). ``evaluation/`` sits one rung above ``models/`` in the DAG
   and may depend downward on any number of ``models.*`` /
   ``features.*`` / ``utils.*`` modules — the boundary test only
   forbids ``drivers.*`` and *sibling* ``evaluation/`` imports, and
   does not restrict which ``models/`` submodules an ``evaluation/``
   module imports (unlike the single-``_shared``-exception rule one
   rung down). M33a (``evaluation/series_evaluation.py``) already
   established this precedent for a series-level module. Consequently
   this module ``import`` s ``models.ancestral_veto_sampler`` (for
   ``sample_veto_sequences`` / ``SampledVetoSequence`` /
   ``VetoStepPredictorFn``) and imports the ``SeriesModelFn`` type from
   ``models.flat_series_baseline`` freely — both are downward
   ``evaluation/ -> models/`` edges, not lateral ``models/ -> models/``
   ones.
2. **The Stage-2 four-way model is *not* hard-imported.** Mirroring
   M29's own decision 1 (pluggable ``predictor_fn_by_action``, not
   hard-wired to M27/M28), the map-level four-way scorer is accepted as
   a caller-supplied callable (``map_model_fn``) matching M19's
   harness ``ModelFn`` shape — ``(team1_id, team2_id, map_name, date,
   matches_df, maps_df) -> Sequence[float]``, a 4-vector in
   ``models._shared.OUTCOME_LABELS`` order — structurally identical
   to, but **not imported from**, ``evaluation.harness`` (an
   ``evaluation/`` module may not import a sibling ``evaluation/``
   module). :data:`MapOutcomeModelFn` is a locally duplicated type
   alias for it, following the exact duplication precedent M29's
   ``VetoStepPredictorFn`` set for its own type. The natural real
   wiring is M20's ``models.ordinal_logit.make_model_fn`` output, but
   this module depends on none of M20/M27/M28 concretely — a caller
   (tests, or a future driver) wires in whichever concrete model(s) it
   wants.
3. **Played-map order: pick/decider actions in step order.** For a
   given ``SampledVetoSequence`` the maps actually *played*, in play
   order, are the maps of its ``"pick"`` and ``"decider"`` actions
   taken in ascending ``step_index`` order (bans are never played).
   Nothing elsewhere in the repo states this explicitly; it is the
   most natural reading of a veto sequence and is recorded here as an
   assumption (see :func:`_played_maps_in_order`). It is only ever
   applied to *simulated* M29 sequences, never to a real veto log.
4. **Weighted average per the roadmap's literal wording, with the
   tension recorded explicitly.** ``weight_i = sequence_probability_i
   / sum_j(sequence_probability_j)`` over the ``n_samples`` sampled
   sequences, and ``probabilities[k] = sum_i weight_i *
   scoreline_probs_i[k]`` (a ``ValueError`` if the weight sum is
   exactly ``0.0`` — a degenerate all-zero-probability sample set).
   Re-weighting already-probability-proportional ancestral samples by
   their own drawn probability is *not* the same thing as a plain
   equal-weight Monte Carlo mean over the samples (the two coincide
   only asymptotically as ``n_samples -> infinity`` under ancestral
   sampling). The roadmap line is explicit ("average across M29's
   samples weighted by sequence probability"), so this module
   implements the weighted form as specified rather than substituting
   the plain MC mean, and flags the two as a known, discussable
   alternative for REVIEW rather than silently picking one.
5. **``_parse_best_of`` is a *third* independent duplicate.** The
   ``"Bo3"`` -> ``3`` parser is private to
   ``models/flat_series_baseline.py`` (leading underscore, one layer
   below) and duplicated in ``evaluation/series_evaluation.py``
   (sibling layer — importing either copy would be a lateral reach).
   This module carries a third behaviour-identical copy; flagged here
   (not silently fixed) that a future milestone should consider
   promoting the parser to a shared utility.
6. **``models._shared`` may be imported directly from here.** The
   single-exception rule ("only ``models._shared``") is a
   ``models/``-internal restriction; ``evaluation/`` is otherwise free
   to import any ``models/`` submodule.
   :data:`~models._shared.OUTCOME_LABELS` and
   :data:`~models._shared._N_CATEGORIES` make the "index 0/1 = A,
   index 2/3 = B" collapse legible instead of a magic ``[0, 1]`` /
   ``[2, 3]`` slice.
7. **No CLI driver in this milestone.** Mirrors the M25/M29/M30
   library-only precedent; M31's only consumer today is M33b (not yet
   built), which is where the two-arm headline report and any
   artifact-writing CLI belong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models._shared import _N_CATEGORIES, OUTCOME_LABELS
from models.ancestral_veto_sampler import (
    SampledVetoSequence,
    VetoStepPredictorFn,
    sample_veto_sequences,
)
from models.flat_series_baseline import SeriesModelFn
from utils import series_paths

# The generic Stage-2 four-way per-map model interface, locally
# duplicated per decision 2: a callable taking the two team ids, the
# map name, the as-of date, and the full matches/maps tables, and
# returning the 4-vector of map-outcome probabilities in
# ``models._shared.OUTCOME_LABELS`` order
# (p_a_regulation, p_a_ot, p_b_ot, p_b_regulation). Structurally
# identical to the type ``evaluation/harness.py`` names ``ModelFn``
# but deliberately not imported from there (decision 2: an
# ``evaluation/`` module must not import a sibling ``evaluation/``
# module).
MapOutcomeModelFn = Callable[
    [str, str, str, str, pd.DataFrame, pd.DataFrame],
    Sequence[float],
]


def _parse_best_of(best_of: str) -> int:
    """Parse a ``"Bo<N>"`` series-length string into a plain odd int.

    Converts the ``"Bo1"``/``"Bo3"``/``"Bo5"`` strings carried by
    ``matches.parquet``'s ``best_of`` column (and any other
    ``"Bo<N>"`` string whose ``N`` is a positive odd integer) into the
    plain integer map count that ``utils.series_paths`` expects. The
    suffix must be exactly one or more decimal digits: anything else —
    a non-``"Bo"`` prefix, a non-numeric suffix, an even or non-positive
    map count, a non-string input — is rejected with ``ValueError``
    (or ``TypeError`` for a non-string input, which violates the
    annotated contract rather than being a malformed string) rather
    than silently coerced.

    **Third deliberate duplication, not an import.** This helper is a
    local, behaviour-identical copy of
    ``models.flat_series_baseline._parse_best_of`` (private, one layer
    below) and ``evaluation.series_evaluation._parse_best_of`` (sibling
    layer): importing either copy across its boundary would be exactly
    the lateral reach the module-boundary rule forbids (decision 5).
    The three copies stay in sync by convention until a future
    milestone promotes the parser to a shared utility.

    Args:
        best_of: The series-length string to parse; must be exactly a
            ``"Bo"`` prefix followed by decimal digits spelling a
            positive odd integer.

    Returns:
        The parsed map count as a plain ``int`` (``1`` for ``"Bo1"``,
        ``3`` for ``"Bo3"``, ``5`` for ``"Bo5"``).

    Raises:
        TypeError: If ``best_of`` is not a string at all (violates the
            annotated ``str`` contract).
        ValueError: If ``best_of`` is a string that does not start with
            ``"Bo"``, has a non-digit suffix (e.g. ``"BestOf3"``, an
            empty suffix, a trailing-space suffix), or spells an even
            or non-positive map count (e.g. ``"Bo2"``, ``"Bo0"``).
    """
    if not isinstance(best_of, str):
        raise TypeError(
            f"best_of must be a 'Bo<N>' string, got {best_of!r}"
        )
    if not best_of.startswith("Bo") or len(best_of) <= 2:
        raise ValueError(
            f"best_of must be a 'Bo<N>' string like 'Bo3', got {best_of!r}"
        )
    suffix = best_of[2:]
    if not suffix.isdigit():
        raise ValueError(
            f"best_of must be a 'Bo<N>' string with a numeric suffix, "
            f"got {best_of!r}"
        )
    n = int(suffix)
    if n < 1:
        raise ValueError(
            f"best_of must be a positive odd map count, got {best_of!r}"
        )
    if n % 2 == 0:
        raise ValueError(
            f"best_of must be odd (an even map count cannot produce a "
            f"guaranteed series winner), got {best_of!r}"
        )
    return n


def _played_maps_in_order(sample: SampledVetoSequence) -> tuple[str, ...]:
    """Return the maps actually played by a sampled veto, in play order.

    Implements decision 3: the maps of the sample's ``"pick"`` and
    ``"decider"`` actions, taken in ascending ``step_index`` order
    (bans are never played). The resulting count is asserted to equal
    the parsed ``best_of`` map count — an internal desync otherwise
    (a malformed action sequence, or a ``best_of`` string disagreeing
    with the action count), which would silently corrupt the per-map
    probability vector's length.

    Args:
        sample: The sampled veto sequence whose played maps are
            derived.

    Returns:
        A tuple of exactly ``best_of`` map names in play order: the
        picks in pick order (ascending step index), then the forced
        decider map last.

    Raises:
        ValueError: If ``sample.best_of`` is not a valid ``"Bo<N>"``
            string (from :func:`_parse_best_of`), or if the number of
            pick+decider actions does not equal the parsed map count.
    """
    best_of_int = _parse_best_of(sample.best_of)
    played = tuple(
        action.map_name
        for action in sorted(sample.actions, key=lambda a: a.step_index)
        if action.action in ("pick", "decider")
    )
    if len(played) != best_of_int:
        raise ValueError(
            f"sampled veto sequence for best_of {sample.best_of!r} has "
            f"{len(played)} played map(s) (pick/decider actions) but "
            f"{best_of_int} map(s) are expected; the action sequence "
            "and the best_of string are out of sync"
        )
    return played


def _collapse_to_binary_a_win(four_way: Sequence[float]) -> float:
    """Collapse a four-way map distribution to a binary A-win probability.

    Implements the roadmap's core "collapse ... to a binary" step: the
    map-level A-side win probability is the sum of the A-regulation and
    A-OT categories — indices 0 and 1 of
    :data:`models._shared.OUTCOME_LABELS` order
    (``p_a_regulation, p_a_ot, p_b_ot, p_b_regulation``) — because a
    played map is won by side A if and only if it ends in A-regulation
    or A-OT. Kept as its own named, tested pure function since it is
    the one line implementing the roadmap's core instruction; the
    per-call-site validation naming the offending map/sample lives in
    :func:`_score_sample_series` (this function re-validates the
    length so it stands alone safely).

    Args:
        four_way: The 4-vector of map-outcome probabilities in
            :data:`models._shared.OUTCOME_LABELS` order. Must have
            exactly ``_N_CATEGORIES`` (4) entries.

    Returns:
        ``four_way[0] + four_way[1]`` as a ``float`` — P(A wins the
        map) under the four-way model.

    Raises:
        ValueError: If ``four_way`` has other than 4 entries (naming
            the actual length).
    """
    raw = list(four_way)
    if len(raw) != _N_CATEGORIES:
        raise ValueError(
            f"map_model_fn returned {len(raw)} probabilit(ies) but "
            f"{_N_CATEGORIES} are expected (OUTCOME_LABELS order "
            f"{OUTCOME_LABELS}); cannot collapse a non-four-way "
            "vector to a binary A-win probability"
        )
    return float(raw[0] + raw[1])


def _score_sample_series(
    sample: SampledVetoSequence,
    team1_id: str,
    team2_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_model_fn: MapOutcomeModelFn,
    best_of_int: int,
    sample_index: int,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Compute one sampled sequence's played-map scoring detail.

    Implements plan step 5 for a single M29 sample: derives the
    played-map order via :func:`_played_maps_in_order`, calls
    ``map_model_fn(team1_id, team2_id, map_name, date, matches_df,
    maps_df)`` once per played map (validating each returned vector
    has exactly 4 entries before collapsing, with a ``ValueError``
    naming the offending map and sample index), collapses each
    four-way vector to the per-map A-win probability via
    :func:`_collapse_to_binary_a_win`, and feeds the resulting
    ``best_of``-length vector to
    :func:`utils.series_paths.series_probabilities_in_order` to get
    that sample's exact scoreline distribution. Returns everything the
    per-sample reporting dataclass needs — the four-way detail is
    retained pre-collapse (the roadmap's "four-way detail retained for
    reporting").

    Args:
        sample: The sampled veto sequence to score.
        team1_id: The queried team1's stable id ("A"; side A of the
            scoreline vocabulary and the even-step veto actor).
        team2_id: The queried team2's stable id ("B").
        date: The as-of cutoff passed through to ``map_model_fn``
            (every played map's own match date).
        matches_df: The full materialised ``matches`` table, passed
            through to ``map_model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``map_model_fn`` unchanged.
        map_model_fn: The caller-supplied Stage-2 four-way per-map
            model (see :data:`MapOutcomeModelFn`).
        best_of_int: The parsed ``best_of`` map count; the length the
            played-map tuple and the scoreline vector must match.
        sample_index: The sample's position in the caller's sample
            list, for the wrong-length error message.

    Returns:
        A ``(played_maps, per_map_four_way, per_map_win_prob,
        scoreline_probabilities)`` tuple: ``played_maps`` is the tuple
        of played map names in play order; ``per_map_four_way`` is the
        tuple of raw four-way vectors (pre-collapse, for reporting);
        ``per_map_win_prob`` is the tuple of collapsed per-map A-win
        probabilities (the vector fed to the enumeration);
        ``scoreline_probabilities`` is the sample's ``best_of_int + 1``
        -length scoreline distribution in
        ``series_outcome_order`` order.

    Raises:
        ValueError: If the played-map count mismatches ``best_of_int``
            (from :func:`_played_maps_in_order`); if ``map_model_fn``
            returns a vector of other than 4 entries for a played map
            (naming the map and sample index); or if the scoreline
            enumeration rejects the collapsed vector (propagated from
            :func:`utils.series_paths.series_probabilities_in_order`,
            e.g. a non-finite or out-of-``[0, 1]`` collapsed value).
        TypeError / KeyError: Propagated verbatim from
            ``map_model_fn`` (this module does not catch misbehaving
            callables, mirroring M29's predictor contract).
    """
    played_maps = _played_maps_in_order(sample)
    per_map_four_way: list[tuple[float, ...]] = []
    per_map_win_prob: list[float] = []
    for map_name in played_maps:
        four_way = tuple(
            map_model_fn(
                team1_id, team2_id, map_name, date, matches_df, maps_df
            )
        )
        if len(four_way) != _N_CATEGORIES:
            raise ValueError(
                f"map_model_fn returned {len(four_way)} probabilit(ies) "
                f"for map {map_name!r} of sampled sequence "
                f"#{sample_index} (teams {sample.team_a_id!r} vs "
                f"{sample.team_b_id!r}, best_of {sample.best_of!r}); "
                f"expected {_N_CATEGORIES} in OUTCOME_LABELS order "
                f"{OUTCOME_LABELS}"
            )
        per_map_four_way.append(four_way)
        per_map_win_prob.append(_collapse_to_binary_a_win(four_way))
    scoreline_probabilities = tuple(
        series_paths.series_probabilities_in_order(
            per_map_win_prob, best_of_int
        )
    )
    return (
        played_maps,
        tuple(per_map_four_way),
        tuple(per_map_win_prob),
        scoreline_probabilities,
    )


@dataclass(frozen=True)
class SeriesVetoSample:
    """One sampled veto sequence's series-prediction detail (M31).

    The per-sample reporting record of the marginalisation: carries
    the raw M29 :class:`SampledVetoSequence`, its normalized weight in
    the aggregate (decision 4), the played-map order (decision 3), the
    raw per-map four-way vectors (pre-collapse — the roadmap's
    "four-way detail retained for reporting"), the collapsed per-map
    A-win probabilities, and the sample's own exact scoreline
    distribution.

    Attributes:
        sequence: The raw M29 sampled veto sequence (self-contained,
            carrying its own ``sequence_probability``).
        weight: The normalized weight ``sequence_probability /
            sum_j(sequence_probability_j)``; a ``float`` in
            ``[0, 1]``, and the weights sum to 1 across the
            prediction's ``samples`` tuple.
        played_maps: The tuple of map names actually played, in play
            order (``best_of`` entries: the picks in step order, then
            the forced decider map).
        per_map_four_way: The tuple of raw four-way probability
            vectors (``best_of`` entries in ``played_maps`` order,
            each 4 long in OUTCOME_LABELS order) — retained
            pre-collapse for reporting.
        per_map_win_prob: The tuple of collapsed per-map A-win
            probabilities (``best_of`` entries, ``p_a_regulation +
            p_a_ot``) — the vector fed to the scoreline enumeration.
        scoreline_probabilities: This sample's exact ``best_of + 1``
            scoreline distribution in
            :func:`utils.series_paths.series_outcome_order` order.
    """

    sequence: SampledVetoSequence
    weight: float
    played_maps: tuple[str, ...]
    per_map_four_way: tuple[tuple[float, ...], ...]
    per_map_win_prob: tuple[float, ...]
    scoreline_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class VetoMarginalizedSeriesPrediction:
    """A veto-marginalised series-scoreline prediction for one match.

    The top-level M31 output: the aggregated ``best_of + 1`` scoreline
    distribution (the weighted average across the M29 samples, in
    :func:`utils.series_paths.series_outcome_order` order) plus the
    parsed ``best_of``, the outcome-order vocabulary, and the full
    per-sample detail tuple for reporting. :meth:`as_tuple` exposes
    the probabilities as a plain sequence so the prediction satisfies
    the :data:`SeriesModelFn` convention directly (mirroring
    ``FlatSeriesPrediction.as_tuple`` / ``FourWayPrediction.as_tuple``).

    Attributes:
        probabilities: The aggregated ``best_of + 1`` scoreline
            probabilities in ``outcome_order`` order, summing to 1
            within float error.
        best_of: The parsed map count (``3`` for ``"Bo3"``, ``5`` for
            ``"Bo5"``, ``1`` for ``"Bo1"``).
        outcome_order: The ``best_of + 1`` terminal scorelines in
            canonical order
            (``utils.series_paths.series_outcome_order``).
        samples: The tuple of :class:`SeriesVetoSample` reporting
            records, one per sampled sequence, in sample order.
    """

    probabilities: tuple[float, ...]
    best_of: int
    outcome_order: tuple[tuple[int, int], ...]
    samples: tuple[SeriesVetoSample, ...]

    def as_tuple(self) -> tuple[float, ...]:
        """Return the aggregated scoreline probabilities in ordinal order.

        The tuple returned is exactly ``self.probabilities`` — the
        ``best_of + 1`` floats in ``self.outcome_order`` order — so a
        caller consuming the :data:`SeriesModelFn` convention (e.g.
        ``evaluation.series_evaluation.score_held_out_series``) gets
        the ready-to-score probability vector without a remapping
        step. Kept as a method for parity with
        :meth:`models.flat_series_baseline.FlatSeriesPrediction.as_tuple`
        and :meth:`models.four_way_baseline.FourWayPrediction.as_tuple`.

        Returns:
            A tuple of ``best_of + 1`` non-negative floats summing to
            ``1.0``, in the ordinal order of ``self.outcome_order``.

        Raises:
            Nothing.
        """
        return self.probabilities


def predict_series_outcome_via_veto_marginalization(
    team1_id: str,
    team2_id: str,
    best_of: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    map_model_fn: MapOutcomeModelFn,
    predictor_fn_by_action: dict[str, VetoStepPredictorFn],
    n_samples,
    rng: np.random.Generator,
    map_pool=None,
) -> VetoMarginalizedSeriesPrediction:
    """Predict a series scoreline distribution by veto marginalisation (M31).

    Composes the full pipeline (module docstring decisions 1-4):
    parses ``best_of`` via :func:`_parse_best_of`, calls
    :func:`models.ancestral_veto_sampler.sample_veto_sequences` to get
    ``n_samples`` raw sampled sequences, scores and collapses each
    sample's played maps via :func:`_score_sample_series`, normalizes
    the samples' ``sequence_probability`` values into weights (raising
    on a degenerate zero-total sample set, decision 4), and aggregates
    the per-sample scoreline vectors as the probability-weighted
    average. The samples' raw M29 sequences, per-map four-way detail,
    and per-sample scorelines are all retained in the returned
    prediction for reporting.

    Args:
        team1_id: The queried team1's stable id ("A" in the series
            scoreline vocabulary and the even-step veto actor; passed
            to the sampler as ``team_a_id``).
        team2_id: The queried team2's stable id ("B"; passed to the
            sampler as ``team_b_id``).
        best_of: The series length as the ``"Bo<N>"`` string (e.g.
            ``"Bo3"``, ``"Bo5"``, ``"Bo1"``); parsed to a plain odd
            int by :func:`_parse_best_of`.
        date: The as-of cutoff passed through to the sampler, its
            predictors, and ``map_model_fn`` (every played map's own
            match date).
        matches_df: The full materialised ``matches`` table, passed
            through unchanged to the sampler and ``map_model_fn``.
        maps_df: The full materialised ``maps`` table, passed through
            unchanged to the sampler and ``map_model_fn``.
        map_model_fn: The caller-supplied Stage-2 four-way per-map
            model (see :data:`MapOutcomeModelFn`); the natural wiring
            is ``models.ordinal_logit.make_model_fn``'s closure, but
            nothing here depends on M20 concretely.
        predictor_fn_by_action: The caller-supplied Stage-1 per-step
            predictor dict passed straight to
            :func:`models.ancestral_veto_sampler.sample_veto_sequences`
            (keyed ``"ban"`` / ``"pick"``; a Bo1 sequence needs only
            ``"ban"``).
        n_samples: How many M29 walks to sample; must be a positive
            integer (validated inside the sampler).
        rng: The caller-constructed ``numpy.random.Generator`` (e.g.
            ``numpy.random.default_rng(seed)``) the sampler's draws
            consume sequentially. No default — an omitted ``rng`` is a
            ``TypeError`` from Python's own argument binding, not a
            hidden global RNG (mirrors M29's convention).
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves the active era's pool from
            ``config.json`` on ``date``'s calendar date (passed
            through to the sampler unchanged).

    Returns:
        A :class:`VetoMarginalizedSeriesPrediction` whose
        ``probabilities`` (the weighted-average aggregate, summing to
        1 within float error) are in
        ``utils.series_paths.series_outcome_order(best_of)`` order,
        carrying the parsed ``best_of``, the ``outcome_order`` tuple,
        and one :class:`SeriesVetoSample` per sampled sequence (the
        sample weights summing to 1 within float error).

    Raises:
        ValueError: If ``best_of`` is malformed (from
            :func:`_parse_best_of`); if the sampler rejects any input
            (invalid ``n_samples``, missing predictor key, wrong-size
            or duplicate pool, malformed predictor vector — all
            propagated from
            :func:`models.ancestral_veto_sampler.sample_veto_sequences`);
            if the sum of the samples' ``sequence_probability`` values
            is exactly ``0.0`` (a degenerate all-zero-probability
            sample set); if a sampled sequence's played-map count
            mismatches ``best_of`` or a played map's four-way vector
            has the wrong length (from
            :func:`_score_sample_series`).
        TypeError: If ``rng`` is omitted or not a
            ``numpy.random.Generator`` (propagated from the sampler);
            if a predictor or ``map_model_fn`` callable itself raises
            ``TypeError`` (propagated verbatim).
        KeyError: Propagated from the sampler / predictors /
            ``map_model_fn`` if a required table column is absent.
    """
    best_of_int = _parse_best_of(best_of)
    samples = sample_veto_sequences(
        team1_id,
        team2_id,
        best_of,
        date,
        matches_df,
        maps_df,
        predictor_fn_by_action,
        n_samples,
        rng,
        map_pool,
    )

    total_probability = sum(
        sample.sequence_probability for sample in samples
    )
    if total_probability == 0.0:
        raise ValueError(
            f"the {len(samples)} sampled veto sequences have a total "
            "sequence_probability of exactly 0.0; the weighted average "
            "is undefined (a degenerate all-zero-probability sample set)"
        )

    series_samples: list[SeriesVetoSample] = []
    aggregated = [0.0] * (best_of_int + 1)
    for sample_index, sample in enumerate(samples):
        (
            played_maps,
            per_map_four_way,
            per_map_win_prob,
            scoreline_probabilities,
        ) = _score_sample_series(
            sample,
            team1_id,
            team2_id,
            date,
            matches_df,
            maps_df,
            map_model_fn,
            best_of_int,
            sample_index,
        )
        weight = sample.sequence_probability / total_probability
        for category_index, probability in enumerate(
            scoreline_probabilities
        ):
            aggregated[category_index] += weight * probability
        series_samples.append(
            SeriesVetoSample(
                sequence=sample,
                weight=weight,
                played_maps=played_maps,
                per_map_four_way=per_map_four_way,
                per_map_win_prob=per_map_win_prob,
                scoreline_probabilities=scoreline_probabilities,
            )
        )

    return VetoMarginalizedSeriesPrediction(
        probabilities=tuple(aggregated),
        best_of=best_of_int,
        outcome_order=series_paths.series_outcome_order(best_of_int),
        samples=tuple(series_samples),
    )


def make_series_model_fn(
    map_model_fn: MapOutcomeModelFn,
    predictor_fn_by_action: dict[str, VetoStepPredictorFn],
    n_samples,
    rng: np.random.Generator,
    map_pool=None,
) -> SeriesModelFn:
    """Build the :data:`SeriesModelFn` adapter over the M31 pipeline.

    The factory that actually plugs M31 into the live M33a harness: a
    closure factory returning a 6-argument callable — exactly the
    :data:`SeriesModelFn` shape
    ``evaluation.series_evaluation.score_held_out_series`` consumes —
    that calls
    :func:`predict_series_outcome_via_veto_marginalization` and
    returns ``.as_tuple()``. This mirrors
    ``evaluation.series_evaluation.flat_series_baseline_model``'s
    adapter pattern at the factory level: M31's predictions are richer
    than the bare tuple, so the adapter discards the reporting detail
    for scoring (callers that want the detail call the core function
    directly).

    The single ``rng`` object is captured once at factory-construction
    time and consumed sequentially across every call the returned
    closure makes over the life of that closure (e.g. once per
    held-out series when driven by
    ``evaluation.series_evaluation.score_held_out_series``) — so a
    fixed seed reproduces byte-identical output across an entire
    evaluation run, not just one call (mirroring M29's decision 4
    wording).

    Args:
        map_model_fn: The Stage-2 four-way per-map model to score each
            sampled sequence's played maps with (see
            :data:`MapOutcomeModelFn`).
        predictor_fn_by_action: The Stage-1 per-step predictor dict
            passed through to the sampler (see
            :func:`predict_series_outcome_via_veto_marginalization`).
        n_samples: How many M29 walks each prediction samples (a
            positive int; validated inside the sampler).
        rng: The ``numpy.random.Generator`` the returned closure's
            predictions consume sequentially over its whole life. No
            default — an omitted ``rng`` is a ``TypeError`` from
            Python's own argument binding at factory-construction time.
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves the active era's pool from
            ``config.json`` per call.

    Returns:
        A ``SeriesModelFn``-shaped closure ``(team1_id, team2_id,
        best_of, date, matches_df, maps_df) -> tuple[float, ...]`` of
        ``best_of + 1`` probabilities in
        ``utils.series_paths.series_outcome_order`` order.

    Raises:
        TypeError: If ``rng`` is omitted at factory-construction time
            (Python's own argument binding).
        ValueError / KeyError / TypeError: Propagated from
            :func:`predict_series_outcome_via_veto_marginalization` on
            any call of the returned closure (see that function's
            docstring).
    """

    def series_model_fn(
        team1_id: str,
        team2_id: str,
        best_of: str,
        date: str,
        matches_df: pd.DataFrame,
        maps_df: pd.DataFrame,
    ) -> tuple[float, ...]:
        """Predict one series' scoreline distribution via M31.

        Calls :func:`predict_series_outcome_via_veto_marginalization`
        with the closed-over ``map_model_fn``,
        ``predictor_fn_by_action``, ``n_samples``, ``rng`` and
        ``map_pool`` and returns the aggregated probabilities as a
        plain tuple (the :data:`SeriesModelFn` shape). The closed-over
        ``rng`` advances on every call, so a fixed seed reproduces the
        whole evaluation run byte-identically.

        Args:
            team1_id: The queried team1's stable id ("A").
            team2_id: The queried team2's stable id ("B").
            best_of: The ``"Bo<N>"`` series-length string.
            date: The as-of cutoff.
            matches_df: The full materialised ``matches`` table.
            maps_df: The full materialised ``maps`` table.

        Returns:
            A tuple of ``best_of + 1`` floats in
            ``utils.series_paths.series_outcome_order`` order, summing
            to 1 within float error.

        Raises:
            ValueError / KeyError / TypeError: Propagated from
                :func:`predict_series_outcome_via_veto_marginalization`
                (see its docstring).
        """
        prediction = predict_series_outcome_via_veto_marginalization(
            team1_id,
            team2_id,
            best_of,
            date,
            matches_df,
            maps_df,
            map_model_fn,
            predictor_fn_by_action,
            n_samples,
            rng,
            map_pool,
        )
        return prediction.as_tuple()

    return series_model_fn
