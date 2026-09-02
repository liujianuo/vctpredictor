"""Stage-isolation evaluation (roadmap M34).

Measures the cost of Stage-1 error compounding into Stage 2: for every
held-out (test-split) match, Stage 2 (a fitted four-way per-map model,
the natural wiring being M20's ``models.ordinal_logit.make_model_fn``)
is scored twice on the *identical* observed map outcomes — once against
the maps that were actually played (Arm A), and once against the maps
M29's ancestral veto sampler (driven by the fitted M27/M28 ban/pick
predictors) would have predicted at the same positions (Arm B) — and
the gap between the two arms' aggregate metrics is reported as "the
cost of Stage 1 error compounding". Stage 2's model and its inputs are
identical in both arms; only the map identity queried differs, so any
degradation is attributable to Stage 1's own map-identity uncertainty.

Design decisions (recorded here, do not re-derive in later
milestones):

1. **Module placement: ``evaluation/``, not ``models/``.** The reason
   is the module-boundary DAG, not style: this module must call both a
   Stage-2 four-way ``ModelFn``-shaped callable *and*
   ``models.ancestral_veto_sampler.sample_veto_sequences``. A
   ``models/`` module may only depend downward on ``features.*`` /
   ``utils.*`` (with the one lateral ``models/_shared.py`` exception),
   so a module that invokes a sibling ``models/`` Stage-2 model (M20's
   ``models.ordinal_logit``) cannot itself live in ``models/``.
   ``evaluation/`` sits one rung above ``models/`` in the DAG
   (``utils/ -> features/ -> models/ -> evaluation/ -> drivers/``) and
   may depend downward on any number of ``models.*`` / ``features.*`` /
   ``utils.*`` modules. M31 (``evaluation/veto_marginalized_series.py``)
   and M33a (``evaluation/series_evaluation.py``) already established
   this precedent for Stage-2-consuming modules; this task follows it
   unchanged.
2. **No sibling ``evaluation/`` import — everything shared is
   re-derived locally.** The module-boundary test
   (``tests/test_module_boundaries.py::test_evaluation_module_imports_only_features_models_and_utils``)
   forbids any ``evaluation.``-prefixed sibling import statement in an
   ``evaluation/`` module, so this module may **not** import
   ``evaluation.harness`` (the M19 ``build_held_out_maps`` /
   ``score_held_out_maps`` Arm-A machinery), ``evaluation.veto_marginalized_series``
   (the M31 ``_played_maps_in_order`` helper and the weighting
   convention), or ``evaluation.series_evaluation`` /
   ``evaluation.veto_evaluation`` (the multi-arm report scaffolding).
   Concretely, the following are independently reimplemented here,
   behaviour-identical to their siblings but not imported from them:
   - a ``_parse_best_of("Bo<N>") -> int`` helper — this is at least
     the **4th independent copy** in the repo
     (``models/flat_series_baseline.py``,
     ``evaluation/series_evaluation.py`` and
     ``evaluation/veto_marginalized_series.py`` each already carry
     one). Flagged here, per the existing convention: a future
     milestone should promote this to a shared utility rather than
     fixing it silently in this task.
   - a ``_played_maps_in_order``-equivalent helper reading a sampled
     sequence's ``"pick"`` / ``"decider"`` actions in ascending
     ``step_index`` order (the maps actually played by that veto walk,
     per the convention M31's decision 3 records; independently
     reimplemented from
     ``evaluation.veto_marginalized_series._played_maps_in_order``,
     not imported).
   - a local :data:`MapOutcomeModelFn` type alias for the Stage-2
     four-way callable (``(team1_id, team2_id, map_name, date,
     matches_df, maps_df) -> Sequence[float]``, a 4-vector in
     ``models._shared.OUTCOME_LABELS`` order) — structurally identical
     to, but **not imported from**, ``evaluation.harness.ModelFn``.
   - a local Arm-A table builder
     (:func:`build_actual_played_maps`) restricted to what this module
     actually needs — one row per *actually-played* held-out map with
     its true ``outcome_ordinal``, grouped so the caller can find, per
     ``match_id``, the ordered list of ``(map_index, map_name,
     outcome_ordinal)`` and each match's ``best_of`` / ``team1_id`` /
     ``team2_id`` / ``date`` — and two Arm-A/Arm-B scorers
     (:func:`score_actual_played_maps` /
     :func:`score_predicted_played_maps`), re-derived from
     ``evaluation.harness.score_held_out_maps``'s shape.
   To state it explicitly: **no ``evaluation.harness``,
   ``evaluation.series_evaluation``,
   ``evaluation.veto_marginalized_series`` or
   ``evaluation.veto_evaluation`` symbol is imported anywhere in this
   file.** The only intra-repo imports are ``models._shared``,
   ``models.ancestral_veto_sampler``, ``utils.scoring`` and
   ``utils.splits`` (plus whichever ``models/`` module supplies the
   Stage-2 model in tests — kept pluggable via the local
   :data:`MapOutcomeModelFn` type, never hard-imported, mirroring
   ``evaluation.harness``'s own genericity).
3. **The Stage-2 four-way model is *not* hard-imported.** Mirroring
   M29's decision 1 (pluggable ``predictor_fn_by_action``) and M31's
   decision 2 (pluggable ``map_model_fn``), the map-level four-way
   scorer is accepted as a caller-supplied callable matching the local
   :data:`MapOutcomeModelFn` type. The natural real wiring is M20's
   ``models.ordinal_logit.make_model_fn`` output (and the driver wires
   exactly that), but this module depends on none of M20/M27/M28
   concretely.
4. **Arm B is truncated to the ``n_played`` positions that actually
   happened.** A swept Bo3 has ``n_played = 2``: its third slot was
   vetoed and never played, so no ground-truth outcome exists to score
   either arm against at that position. Both arms therefore cover
   exactly the ``(match_id, map_index)`` positions ``0 .. n_played-1``
   materialised in Arm A's table; the un-played decider slot of a
   swept series is dropped from the comparison. ``n_played`` is taken
   as the count of that match's rows in the Arm-A table after the
   inner join (the same convention
   ``evaluation.harness.build_held_out_maps`` /
   ``evaluation.series_evaluation.build_held_out_series`` already
   use); a full veto sequence always has ``best_of`` played maps, so
   truncating to ``n_played <= best_of`` is safe by construction, and
   a sample with fewer than ``n_played`` maps is an internal desync
   that raises loudly.
5. **Arm B's samples are weighted by normalized
   ``sequence_probability``.** Each sampled veto sequence carries its
   own ``sequence_probability`` (the product of every non-decider
   step's drawn probability); the per-position weight is
   ``sequence_probability_i / sum_j(sequence_probability_j)`` over the
   ``n_samples`` draws (a ``ValueError`` on an all-zero-probability
   sample set, mirroring M31's own guard). This inherits M31's
   roadmap-literal choice and its documented tension with a plain
   equal-weight Monte Carlo mean over ancestral samples — inherited,
   not re-litigated here.
6. **One effective row per held-out map position — ``utils.scoring``
   has no per-row weights.** ``utils.scoring``'s batch functions
   (``mean_rps`` / ``mean_log_loss`` /
   ``mean_marginal_binary_accuracy``) accept no ``sample_weight``
   argument (checked at BUILD time; plan item 4's either/or), so Arm
   B collapses each position's ``(sample, weight)`` pairs into one
   effective four-way vector *by hand* — the weighted average of the
   per-sample four-way vectors, weights being the normalized
   ``sequence_probability`` values (identical maps across samples are
   folded in by the weighted sum, so the effective vector is exactly
   the probability-weighted expectation of Stage 2's prediction over
   Stage 1's map-identity uncertainty). Arm A is the weight-1.0
   special case of the same shape. Both arms therefore emit one row
   per held-out map position with the identical column layout — which
   is exactly what :func:`build_stage_isolation_report`'s row-
   alignment guard needs (the ``(match_id, map_index)`` key, one row
   per position, mirroring the M33a/M28 multi-arm guards).
7. **No per-``best_of`` grouping.** Unlike M33a/M33b (series-level,
   varying ``K``), this task scores individual maps — always ``K=4`` —
   so there is no scale-mismatch reason to split the report by
   ``best_of``; a single overall report suffices.

The gap is ``m29_predicted_maps - actual_played_maps`` per metric
(``mean_rps``, ``mean_log_loss``, ``marginal_binary_accuracy``): a
positive RPS/log-loss gap and a negative accuracy gap read as "Stage
1's own map-identity uncertainty makes Stage 2 score worse than it
would if it always knew the real map".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from models._shared import _N_CATEGORIES, OUTCOME_LABELS, _validate_positive_int
from models.ancestral_veto_sampler import (
    SampledVetoSequence,
    VetoStepPredictorFn,
    sample_veto_sequences,
)
from utils import scoring, splits

# Fixed column order for the Arm-A table produced by
# build_actual_played_maps: the per-map identifying columns (plus the
# match's ``best_of`` string, needed to drive the sampler for Arm B)
# and the true ordinal.
ACTUAL_PLAYED_COLUMNS = (
    "match_id",
    "map_index",
    "date",
    "team1_id",
    "team2_id",
    "map_name",
    "best_of",
    "outcome_ordinal",
)

# The four predicted-probability columns of the scored tables, in
# OUTCOME_LABELS order: column ``i`` holds the predicted probability of
# category ``i`` (mirroring evaluation/harness.py's PREDICTION_COLUMNS,
# deliberately not imported — decision 2).
PREDICTION_COLUMNS = ("p_a_regulation", "p_a_ot", "p_b_ot", "p_b_regulation")

# Fixed column order for the scored tables produced by
# score_actual_played_maps / score_predicted_played_maps: the
# identifying columns, the true ordinal, the four predicted
# probabilities, and the three per-position scores. Both arms share
# this exact layout (decision 6), which is what the report's
# row-alignment guard validates against.
SCORED_COLUMNS = (
    "match_id",
    "map_index",
    "date",
    "team1_id",
    "team2_id",
    "map_name",
    "outcome_ordinal",
    *PREDICTION_COLUMNS,
    "rps",
    "log_loss",
    "marginal_correct",
)

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
    plain integer map count. The suffix must be exactly one or more
    decimal digits: anything else — a non-``"Bo"`` prefix, a non-numeric
    suffix, an even or non-positive map count, a non-string input — is
    rejected with ``ValueError`` (or ``TypeError`` for a non-string
    input, which violates the annotated contract rather than being a
    malformed string) rather than silently coerced.

    **Fourth deliberate duplication, not an import.** This helper is a
    local, behaviour-identical copy of
    ``models.flat_series_baseline._parse_best_of`` (private, one layer
    below), ``evaluation.series_evaluation._parse_best_of`` (sibling
    layer) and ``evaluation.veto_marginalized_series._parse_best_of``
    (sibling layer): importing any copy across its boundary would be
    exactly the lateral reach the module-boundary rule forbids
    (decision 2). The four copies stay in sync by convention until a
    future milestone promotes the parser to a shared utility.

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

    Implements decision 2's locally-reimplemented played-map-order
    helper: the maps of the sample's ``"pick"`` and ``"decider"``
    actions, taken in ascending ``step_index`` order (bans are never
    played — the same convention M31's decision 3 records, here
    independently reimplemented from
    ``evaluation.veto_marginalized_series._played_maps_in_order``, not
    imported). The resulting count is asserted to equal the parsed
    ``best_of`` map count — an internal desync otherwise (a malformed
    action sequence, or a ``best_of`` string disagreeing with the
    action count), which would silently corrupt the per-position
    truncation.

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


def build_actual_played_maps(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Assemble the Arm-A table: actually-played held-out maps.

    The Arm-A row table (decision 2's local, independent
    reimplementation of ``evaluation.harness.build_held_out_maps``,
    restricted to what this module needs): joins ``maps_df`` to
    ``matches_df`` (inner, to pick up ``team1_id`` / ``team2_id`` /
    ``date`` — the columns the generic model interface needs, since
    ``maps.parquet`` itself has no date/team columns — plus ``best_of``,
    needed later to drive the M29 sampler for Arm B), joins the result
    to ``labels_df`` on ``(match_id, map_index)`` (inner, to pick up
    the true ``outcome_ordinal``), and restricts to the requested
    ``split`` value by left-attaching the two-valued split column via
    :func:`utils.splits.join_split_to_maps` (the genuine downward
    ``utils/`` dependency) and filtering. The split restriction routes
    through that shared helper so the stale/mismatched-dataset guard it
    performs (every map's ``match_id`` must exist in ``splits_df``)
    applies unchanged, mirroring ``evaluation/harness.py``'s own choice.

    The two inner joins document the same behaviors
    ``evaluation.harness.build_held_out_maps`` records: a map whose
    match is not materialised cannot be scored and is silently dropped,
    and a map whose label row is absent (the skipped-null-score case
    ``drivers.labels`` permits) is silently excluded rather than
    treated as an error. One row per *actually-played* held-out map, in
    the order ``maps_df`` produced them.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_id``, ``team2_id``, ``date``,
            ``best_of``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``map_index``, ``map_name``).
        labels_df: The ``labels`` table (needs ``match_id``,
            ``map_index``, ``outcome_ordinal``).
        splits_df: The ``splits`` table produced by
            :func:`utils.splits.split_matches` (needs ``match_id`` and
            ``split``).
        split: The split value to hold out, ``"test"`` by default (the
            only split ``utils.splits`` defines for final evaluation).

    Returns:
        A ``pandas.DataFrame`` with exactly
        :data:`ACTUAL_PLAYED_COLUMNS` (``match_id, map_index, date,
        team1_id, team2_id, map_name, best_of, outcome_ordinal``), one
        row per actually-played held-out map in the order ``maps_df``
        produced them. Never empty: an empty restricted result raises
        instead.

    Raises:
        ValueError: If the split-restricted, label-joined result is
            empty (no maps in the requested split — e.g. a splits
            table with no rows of that value); or if any map's
            ``match_id`` is absent from ``splits_df`` (propagated from
            :func:`utils.splits.join_split_to_maps`).
        KeyError: If any input table lacks a required column
            (``team1_id``/``team2_id``/``date``/``best_of`` on
            ``matches_df``, ``outcome_ordinal`` on ``labels_df``,
            ``split`` on ``splits_df``; ``match_id``/``map_index`` on
            the maps and labels tables), propagated from pandas/the
            shared helper.
    """
    # Join maps -> matches for the team/date/best_of columns the model
    # and the sampler need. Inner join: a map whose match is not
    # materialised cannot be evaluated, so it is silently dropped
    # rather than erroring.
    joined = maps_df.merge(
        matches_df[
            ["match_id", "team1_id", "team2_id", "date", "best_of"]
        ],
        on="match_id",
        how="inner",
    )
    # Attach the true ordinal (inner join: already-labelled maps only;
    # see the docstring's behavior note).
    joined = joined.merge(
        labels_df[["match_id", "map_index", "outcome_ordinal"]],
        on=["match_id", "map_index"],
        how="inner",
    )
    # Restrict to the requested split via the shared helper, which
    # left-attaches the split column and guards against stale datasets.
    split_maps = splits.join_split_to_maps(joined, splits_df)
    held_out = split_maps[split_maps["split"] == split]
    # ``list(...)``: a bare tuple column selector would be read by
    # pandas as a hierarchical/MultiIndex key rather than a plain
    # column list (mirrors evaluation/harness.py's own comment).
    held_out = held_out[list(ACTUAL_PLAYED_COLUMNS)]
    if len(held_out) == 0:
        raise ValueError(
            f"no actually-played held-out maps for split {split!r}: "
            "joining maps to matches/labels and restricting to that "
            "split yields an empty table"
        )
    return held_out


def sample_predicted_map_identities(
    match_id,
    team1_id: str,
    team2_id: str,
    best_of: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    predictor_fn_by_action: dict[str, VetoStepPredictorFn],
    n_samples,
    rng,
    n_played,
    map_pool=None,
) -> dict[int, list[tuple[str, float]]]:
    """Sample Stage-1's predicted map identities for one match's positions.

    Arm B's map-identity sampler (decision 4): draws ``n_samples`` full
    veto sequences via
    :func:`models.ancestral_veto_sampler.sample_veto_sequences` (one
    call, with ``team1_id``/``team2_id`` as ``team_a_id``/``team_b_id``
    — the same wiring M31's pipeline uses), derives each sample's
    played-map order via :func:`_played_maps_in_order`, truncates it to
    the first ``n_played`` positions (the positions that actually
    happened; the un-played decider slot of a swept series is dropped
    from the comparison, decision 4), and returns, per position
    ``0 .. n_played-1``, the list of ``(sampled_map_name, weight)``
    pairs across the ``n_samples`` draws where each weight is that
    sample's ``sequence_probability`` normalized over the sample set
    (decision 5).

    Args:
        match_id: The match being sampled, for error messages (the
            desync and degenerate-probability guards name it).
        team1_id: The queried team1's stable id ("A" in the scoreline
            vocabulary and the even-step veto actor; passed to the
            sampler as ``team_a_id``).
        team2_id: The queried team2's stable id ("B"; passed to the
            sampler as ``team_b_id``).
        best_of: The series length as the ``"Bo<N>"`` string (e.g.
            ``"Bo3"``, ``"Bo5"``); parsed to a plain odd int by
            :func:`_parse_best_of` inside the sampler's sequence
            handling.
        date: The as-of cutoff passed through to the sampler and its
            predictors.
        matches_df: The full materialised ``matches`` table, passed
            through unchanged to the sampler.
        maps_df: The full materialised ``maps`` table, passed through
            unchanged to the sampler.
        predictor_fn_by_action: The caller-supplied Stage-1 per-step
            predictor dict passed straight to
            :func:`models.ancestral_veto_sampler.sample_veto_sequences`
            (keyed ``"ban"`` / ``"pick"``; a Bo1 sequence needs only
            ``"ban"``).
        n_samples: How many M29 walks to sample; must be a positive
            integer (validated inside the sampler).
        rng: The caller-constructed ``numpy.random.Generator`` the
            sampler's draws consume sequentially. No default — an
            omitted ``rng`` is a ``TypeError`` from Python's own
            argument binding (mirrors M29's convention).
        n_played: How many positions to keep per sample (the count of
            that match's actually-played maps in the Arm-A table);
            must be a positive integer (validated via
            ``models._shared._validate_positive_int``) and must not
            exceed the sample's full played-map count (``best_of``).
        map_pool: The pool to veto over, as an iterable of map names;
            ``None`` (the default) resolves the active era's pool from
            ``config.json`` on ``date``'s calendar date (passed
            through to the sampler unchanged).

    Returns:
        A dict keyed by position index ``0 .. n_played-1``, each value
        a list of ``(map_name, weight)`` pairs with one entry per
        sampled sequence (map names drawn from that sample's played
        order at that position; weights are that sample's normalized
        ``sequence_probability``, so the weights sum to ``1.0`` within
        float error for every position).

    Raises:
        ValueError: If ``n_played`` is not a positive integer; if
            ``best_of`` is malformed (from :func:`_parse_best_of` via
            :func:`_played_maps_in_order`); if a sampled sequence's
            played-map count is less than ``n_played`` (naming the
            match — an internal desync, since a full veto sequence has
            ``best_of >= n_played`` played maps by construction); if
            the sum of the samples' ``sequence_probability`` values is
            exactly ``0.0`` (a degenerate all-zero-probability sample
            set); or if the sampler rejects any input (invalid
            ``n_samples``, missing predictor key, wrong-size or
            duplicate pool, malformed predictor vector — all propagated
            from
            :func:`models.ancestral_veto_sampler.sample_veto_sequences`).
        TypeError: If ``rng`` is omitted or not a
            ``numpy.random.Generator`` (propagated from the sampler);
            if a predictor callable itself raises ``TypeError``
            (propagated verbatim).
        KeyError: Propagated from the sampler / predictors if a
            required table column is absent.
    """
    n_played_int = _validate_positive_int(n_played, "n_played")
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
            f"match {match_id!r}: the {len(samples)} sampled veto "
            "sequences have a total sequence_probability of exactly "
            "0.0; the weighted per-position aggregation is undefined "
            "(a degenerate all-zero-probability sample set)"
        )

    by_position: dict[int, list[tuple[str, float]]] = {
        position: [] for position in range(n_played_int)
    }
    for sample in samples:
        played = _played_maps_in_order(sample)
        if len(played) < n_played_int:
            raise ValueError(
                f"match {match_id!r}: a sampled veto sequence has only "
                f"{len(played)} played map(s) but {n_played_int} "
                f"position(s) are to be scored; a full {best_of!r} veto "
                "sequence must have at least n_played maps (internal "
                "desync between the Arm-A table and the sampler)"
            )
        weight = sample.sequence_probability / total_probability
        for position in range(n_played_int):
            by_position[position].append((played[position], weight))
    return by_position


def _score_position(
    model_fn: MapOutcomeModelFn,
    team1_id: str,
    team2_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    context: str,
) -> list[float]:
    """Call the Stage-2 model once and validate its 4-vector.

    Invokes ``model_fn(team1_id, team2_id, map_name, date,
    matches_df, maps_df)``, materializes the returned sequence, and
    checks it has exactly :data:`_N_CATEGORIES` (4) entries — the
    shared per-call validation both scoring functions use, with a
    ``context`` string (naming the match/position/map) embedded in the
    error message. The simplex validity (finite, non-negative, sums to
    1) is not pre-checked here: the ``utils.scoring`` metric functions
    validate it themselves and their ``ValueError`` propagates
    unchanged.

    Args:
        model_fn: The Stage-2 four-way map model (see
            :data:`MapOutcomeModelFn`).
        team1_id: The queried team1's stable id ("A").
        team2_id: The queried team2's stable id ("B").
        map_name: The map to predict for.
        date: The as-of cutoff passed through to ``model_fn``.
        matches_df: The full materialised ``matches`` table, passed
            through to ``model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``model_fn`` unchanged.
        context: A human-readable description of the position being
            scored (e.g. ``"match '644709' map_index 0 (map 'Haven')"``),
            embedded verbatim in the wrong-length error message.

    Returns:
        The validated 4-vector as a ``list`` of ``float`` in
        :data:`OUTCOME_LABELS` order.

    Raises:
        ValueError: If ``model_fn`` returns a sequence whose length is
            not exactly ``_N_CATEGORIES`` (naming ``context``).
        TypeError / KeyError: Propagated verbatim from ``model_fn``
            (this module does not catch misbehaving callables).
    """
    probs = list(
        model_fn(
            team1_id, team2_id, map_name, date, matches_df, maps_df
        )
    )
    if len(probs) != _N_CATEGORIES:
        raise ValueError(
            f"model_fn returned {len(probs)} probabilit(ies) for {context}; "
            f"expected exactly {_N_CATEGORIES} in OUTCOME_LABELS order "
            f"{OUTCOME_LABELS}"
        )
    return probs


def _per_row_metrics(
    probs: Sequence[float], ordinal: int
) -> tuple[float, float, bool]:
    """Compute the three per-position metrics for one effective prediction.

    Calls ``utils.scoring``'s own per-observation functions —
    :func:`utils.scoring.rps`, :func:`utils.scoring.log_loss` and
    :func:`utils.scoring.marginal_binary_accuracy` — on the given
    four-way vector against the true ordinal, so the metric math lives
    in exactly one place (``utils.scoring``). No ``group_a_indices``
    override is passed: the default first-half convention (side A =
    indices 0, 1 = A-regulation + A-OT) matches the four-way outcome
    vocabulary exactly.

    Args:
        probs: The effective 4-vector of category probabilities in
            :data:`OUTCOME_LABELS` order.
        ordinal: The true ``outcome_ordinal`` of the position.

    Returns:
        A ``(rps, log_loss, marginal_correct)`` tuple: the per-
        observation Ranked Probability Score, the multi-class log
        loss, and the marginal binary correctness ``bool``.

    Raises:
        ValueError: If ``probs`` fails the simplex validation or
            ``ordinal`` is out of range (propagated from the
            ``utils.scoring`` functions — including ``log_loss``'s
            hard error when the true category was assigned exactly
            zero probability).
    """
    return (
        scoring.rps(probs, ordinal),
        scoring.log_loss(probs, ordinal),
        scoring.marginal_binary_accuracy(probs, ordinal),
    )


def score_actual_played_maps(
    model_fn: MapOutcomeModelFn,
    actual_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every actually-played held-out position (Arm A).

    Iterates ``actual_df`` (as produced by :func:`build_actual_played_maps`)
    one row per position, calls ``model_fn(team1_id, team2_id, map_name,
    date, matches_df, maps_df)`` with the row's *actual* ``map_name``
    (weight implicitly 1.0 — the Arm-A special case of decision 6's
    one-effective-row-per-position shape), records the validated
    4-vector under :data:`PREDICTION_COLUMNS`, and computes the three
    per-position metrics via :func:`_per_row_metrics`. The model is
    invoked with the full ``matches_df``/``maps_df`` (never a filtered
    subset): the as-of leakage boundary is the model's own
    responsibility, inherited from ``utils.asof``'s strict ``<``
    cutoff, and is not re-implemented here.

    Args:
        model_fn: The Stage-2 four-way map model to score with (see
            :data:`MapOutcomeModelFn`); the same callable both arms
            use, so only the map identity queried differs.
        actual_df: The Arm-A table from
            :func:`build_actual_played_maps` (needs
            :data:`ACTUAL_PLAYED_COLUMNS`).
        matches_df: The full materialised ``matches`` table, passed
            through to ``model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``model_fn`` unchanged.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SCORED_COLUMNS`
        (the identifying columns, the true ``outcome_ordinal``, the
        four predicted probabilities, and ``rps`` / ``log_loss`` /
        ``marginal_correct``), one row per held-out position in the
        same order as ``actual_df``. ``marginal_correct`` is a ``bool``.

    Raises:
        ValueError: If ``model_fn`` returns a sequence whose length is
            not exactly ``_N_CATEGORIES`` (with the offending
            match/position named); or if any returned vector fails the
            simplex validation or the true ordinal is out of range
            (propagated from :func:`_per_row_metrics`, including
            ``log_loss``'s hard error on a zero-probability true
            category).
        KeyError: If ``actual_df`` lacks an
            :data:`ACTUAL_PLAYED_COLUMNS` column (propagated from
            pandas/``itertuples``).
    """
    rows: list[dict] = []
    for row in actual_df.itertuples(index=False):
        context = f"match {row.match_id!r} map_index {row.map_index!r}"
        probs = _score_position(
            model_fn,
            row.team1_id,
            row.team2_id,
            row.map_name,
            row.date,
            matches_df,
            maps_df,
            context,
        )
        rps, log_loss, marginal_correct = _per_row_metrics(
            probs, row.outcome_ordinal
        )
        scored: dict = {
            "match_id": row.match_id,
            "map_index": row.map_index,
            "date": row.date,
            "team1_id": row.team1_id,
            "team2_id": row.team2_id,
            "map_name": row.map_name,
            "outcome_ordinal": row.outcome_ordinal,
        }
        for column, prob in zip(PREDICTION_COLUMNS, probs):
            scored[column] = prob
        scored["rps"] = rps
        scored["log_loss"] = log_loss
        scored["marginal_correct"] = marginal_correct
        rows.append(scored)
    return pd.DataFrame(rows, columns=SCORED_COLUMNS)


def score_predicted_played_maps(
    model_fn: MapOutcomeModelFn,
    actual_df: pd.DataFrame,
    predicted_by_position: dict[tuple, list[tuple[str, float]]],
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every held-out position against Stage-1's predicted maps (Arm B).

    Iterates ``actual_df`` one row per position (the identical
    positions Arm A scores) and, instead of the row's actual
    ``map_name``, queries Stage 2 with the maps Stage 1 (M29's sampler)
    predicted at that position: for each ``(map_name, weight)`` pair
    from ``predicted_by_position[(match_id, map_index)]`` the model is
    called once and the returned 4-vector is folded into the position's
    *effective* vector as the weight-proportional sum (decision 6 — the
    probability-weighted expectation of Stage 2's prediction over
    Stage 1's map-identity uncertainty; identical maps across samples
    are folded in by the weighted sum). The three per-position metrics
    are then computed on that effective vector via
    :func:`_per_row_metrics`, against the *same* true ``outcome_ordinal``
    Arm A used for the position — the thing that actually happened at
    that point in the series. The only difference between the two arms
    is which ``map_name`` is passed into the identical ``model_fn``.

    Args:
        model_fn: The Stage-2 four-way map model to score with (see
            :data:`MapOutcomeModelFn`); the identical callable Arm A
            uses.
        actual_df: The Arm-A table from
            :func:`build_actual_played_maps` (needs
            :data:`ACTUAL_PLAYED_COLUMNS`); supplies the positions, the
            true ordinals, and the team/date context for every model
            call.
        predicted_by_position: A dict keyed by ``(match_id,
            map_index)`` — one entry per held-out position — mapping to
            the list of ``(map_name, weight)`` pairs from
            :func:`sample_predicted_map_identities` (weights summing
            to 1 within float error per position).
        matches_df: The full materialised ``matches`` table, passed
            through to ``model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``model_fn`` unchanged.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SCORED_COLUMNS` —
        the identical shape and row order as
        :func:`score_actual_played_maps`'s output (one row per held-out
        position, ``map_name`` holding the row's actual map for
        identification while the prediction columns hold the blended
        effective vector) — which is exactly what
        :func:`build_stage_isolation_report`'s row-alignment guard
        requires.

    Raises:
        ValueError: If a row of ``actual_df`` has no entry in
            ``predicted_by_position`` (naming the position — a
            sampling/scoring desync); if ``model_fn`` returns a
            sequence whose length is not exactly ``_N_CATEGORIES``
            (naming the offending sample's position/map); or if any
            effective vector fails the simplex validation or the true
            ordinal is out of range (propagated from
            :func:`_per_row_metrics`).
        KeyError: If ``actual_df`` lacks an
            :data:`ACTUAL_PLAYED_COLUMNS` column (propagated from
            pandas/``itertuples``).
    """
    rows: list[dict] = []
    for row in actual_df.itertuples(index=False):
        key = (row.match_id, row.map_index)
        if key not in predicted_by_position:
            raise ValueError(
                f"no predicted map identities found for match "
                f"{row.match_id!r} map_index {row.map_index!r}; every "
                "held-out position must be sampled by "
                "sample_predicted_map_identities before Arm B scoring"
            )
        context = f"match {row.match_id!r} map_index {row.map_index!r}"
        effective = [0.0] * _N_CATEGORIES
        for map_name, weight in predicted_by_position[key]:
            probs = _score_position(
                model_fn,
                row.team1_id,
                row.team2_id,
                map_name,
                row.date,
                matches_df,
                maps_df,
                f"{context} (sampled map {map_name!r})",
            )
            for i in range(_N_CATEGORIES):
                effective[i] += weight * probs[i]
        rps, log_loss, marginal_correct = _per_row_metrics(
            effective, row.outcome_ordinal
        )
        scored: dict = {
            "match_id": row.match_id,
            "map_index": row.map_index,
            "date": row.date,
            "team1_id": row.team1_id,
            "team2_id": row.team2_id,
            "map_name": row.map_name,
            "outcome_ordinal": row.outcome_ordinal,
        }
        for column, prob in zip(PREDICTION_COLUMNS, effective):
            scored[column] = prob
        scored["rps"] = rps
        scored["log_loss"] = log_loss
        scored["marginal_correct"] = marginal_correct
        rows.append(scored)
    return pd.DataFrame(rows, columns=SCORED_COLUMNS)


def build_stage_isolation_report(
    scored_actual_df: pd.DataFrame,
    scored_predicted_df: pd.DataFrame,
) -> dict:
    """Build the JSON-serializable Arm-A vs Arm-B stage-isolation report.

    An independent two-arm comparison (decision 2: no sibling
    ``evaluation.series_evaluation.build_series_multi_arm_report`` or
    ``evaluation.veto_evaluation.build_veto_multi_arm_report`` import —
    the module-boundary rule; this shape is small enough that a fresh,
    self-contained implementation is appropriate rather than a 4th
    copy of the whole multi-arm scaffolding). Validates the two scored
    tables are row-aligned on ``(match_id, map_index)`` — the
    identifying key here, one row per held-out map position, mirroring
    the row-alignment guards in M33a/M28's multi-arm reports; a
    misaligned comparison would silently pair two different positions'
    scores and corrupt every delta — then recomputes each arm's
    headline metrics from the prediction columns through
    ``utils.scoring``'s shared batch functions
    (:func:`utils.scoring.mean_rps` / ``mean_log_loss`` /
    ``mean_marginal_binary_accuracy``) rather than re-averaging the
    per-row score columns, so the report's headline numbers are
    traceable to the shared metric implementations.

    Args:
        scored_actual_df: The Arm-A scored table from
            :func:`score_actual_played_maps`.
        scored_predicted_df: The Arm-B scored table from
            :func:`score_predicted_played_maps` (needs the identical
            ``(match_id, map_index)`` rows in the identical order).

    Returns:
        A dict with keys ``"actual_played_maps"`` (``n_eval`` int,
        ``mean_rps`` / ``mean_log_loss`` / ``marginal_binary_accuracy``
        floats), ``"m29_predicted_maps"`` (the same four keys), and
        ``"gap"`` (``mean_rps_gap`` / ``mean_log_loss_gap`` /
        ``marginal_binary_accuracy_gap`` floats), where every ``*_gap``
        is ``m29_predicted_maps - actual_played_maps``. Every value is
        a plain str/int/float, so the whole dict is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If the two scored tables have different row counts
            or differ in any ``(match_id, map_index)`` value at the
            same position (the row-alignment contract); or if either
            table is empty (propagated from
            :func:`utils.scoring.mean_rps` — the "mean over zero
            predictions" case); or if any prediction row fails the
            metric validation (propagated from the batch functions,
            e.g. ``log_loss`` on a zero-probability true category).
        KeyError: If either table lacks a prediction column or
            ``outcome_ordinal`` / ``match_id`` / ``map_index``
            (propagated from pandas).
    """
    if len(scored_actual_df) != len(scored_predicted_df):
        raise ValueError(
            f"scored tables have different row counts: actual "
            f"{len(scored_actual_df)} vs predicted "
            f"{len(scored_predicted_df)}; they must describe the same "
            "held-out map positions"
        )
    actual_keys = scored_actual_df[["match_id", "map_index"]].to_numpy()
    predicted_keys = scored_predicted_df[["match_id", "map_index"]].to_numpy()
    mismatch = actual_keys != predicted_keys
    if mismatch.any():
        flat = mismatch.any(axis=1)
        idx = int(flat.argmax())
        raise ValueError(
            "scored tables are not row-aligned: the held-out positions "
            f"differ at row {idx} (actual {actual_keys[idx]!r} vs "
            f"predicted {predicted_keys[idx]!r}); score both arms on "
            "the identical build_actual_played_maps table"
        )

    def _arm_block(scored_df: pd.DataFrame) -> dict:
        """Compute one arm's headline block over its scored table.

        Args:
            scored_df: One arm's scored table (needs the
                :data:`PREDICTION_COLUMNS` columns and
                ``outcome_ordinal``).

        Returns:
            A dict with ``n_eval`` (int), ``mean_rps`` (float),
            ``mean_log_loss`` (float) and
            ``marginal_binary_accuracy`` (float).

        Raises:
            ValueError / KeyError: Propagated from the
                ``utils.scoring`` batch functions / pandas column
                indexing (see the enclosing function's docstring).
        """
        prob_rows = scored_df[list(PREDICTION_COLUMNS)].to_numpy()
        true_indices = scored_df["outcome_ordinal"].to_numpy()
        return {
            "n_eval": len(scored_df),
            "mean_rps": scoring.mean_rps(prob_rows, true_indices),
            "mean_log_loss": scoring.mean_log_loss(
                prob_rows, true_indices
            ),
            "marginal_binary_accuracy": (
                scoring.mean_marginal_binary_accuracy(
                    prob_rows, true_indices
                )
            ),
        }

    actual_block = _arm_block(scored_actual_df)
    predicted_block = _arm_block(scored_predicted_df)
    return {
        "actual_played_maps": actual_block,
        "m29_predicted_maps": predicted_block,
        "gap": {
            "mean_rps_gap": (
                predicted_block["mean_rps"] - actual_block["mean_rps"]
            ),
            "mean_log_loss_gap": (
                predicted_block["mean_log_loss"]
                - actual_block["mean_log_loss"]
            ),
            "marginal_binary_accuracy_gap": (
                predicted_block["marginal_binary_accuracy"]
                - actual_block["marginal_binary_accuracy"]
            ),
        },
    }
