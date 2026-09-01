"""Series-scoreline evaluation harness core (roadmap M33a).

The measurement machinery the two-stage M31 pipeline will be scored
against: it consumes a series-level model callable (the
:data:`SeriesModelFn` convention fixed by M32) plus the *observed*
series scorelines, and aggregates RPS, log loss, and marginal match-win
accuracy via the existing M11 scoring primitives, grouped per
``best_of`` format. This is the harness *core* only: the two-arm
M31-vs-M32 headline report and any CLI/``drivers/`` entry point are
explicitly M33b's job, deferred until M31 exists (assumption 7). M33a
is built and unit-tested against M32 (the one real arm that exists)
plus synthetic stub distributions, so M31 is developed against a
working metric rather than the other way round.

Scope / conventions (recorded here, do not re-derive later):

- **``SeriesModelFn`` reuse, verbatim.** A series model is any callable
  ``(team1_id, team2_id, best_of, date, matches_df, maps_df) ->
  Sequence[float]`` returning the ``best_of + 1`` scoreline
  probabilities in ``utils.series_paths.series_outcome_order`` order
  (assumption 2). This module defines no new model interface; it
  imports the :data:`SeriesModelFn` type from
  ``models.flat_series_baseline`` so the convention lives in one
  place.
- **The observed scoreline is *derived* from ``maps.parquet``, not
  looked up.** No per-series label table exists anywhere in the repo
  (``drivers/labels.py`` (M9) labels only per-*map* four-way outcomes),
  so this module derives each held-out series' observed scoreline
  directly from the map scores, using the exact same "``team1_score``
  is always A's score, ``team2_score`` always B's" convention
  ``drivers/labels.py`` already documents: ``a_wins`` = the number of
  that match's maps with ``team1_score > team2_score``, ``b_wins`` =
  the number with ``team2_score > team1_score`` (assumption 1). The
  ``winner`` column is not read (it holds a team *name* string, not a
  stable id). A tied map score is invalid data and raises loudly
  rather than being silently skipped; a map that resolves to neither
  side's win (e.g. a null score) is likewise an error, never a silent
  undercount.
- **``_parse_best_of`` is deliberately duplicated, not imported.** The
  ``"Bo3"`` -> ``3`` parser is private to
  ``models/flat_series_baseline.py`` (leading underscore, and that
  module is ``models/``, one layer below ``evaluation/``), so importing
  it across that boundary would be exactly the lateral reach the
  module-boundary rule forbids. A local, behaviour-identical copy
  lives here instead (assumption 3); the two copies stay in sync by
  convention until some later milestone promotes the parser to a
  shared utility.
- **No sibling ``evaluation/`` import — the multi-arm scaffolding is an
  independent reimplementation.** ``evaluation/veto_evaluation.py``'s
  ``build_veto_multi_arm_report`` is the closest existing precedent for
  "arm-comparison scaffolding" and is read for its *shape* (per-arm
  report blocks plus a ``deltas_vs_<baseline>`` block, guarded by a
  row-alignment check over the identifying key), but
  :func:`build_series_multi_arm_report` independently reimplements
  that shape adapted to series data — the module-boundary test forbids
  any ``evaluation/`` module from importing a sibling ``evaluation/``
  module (assumption 4), and no exception is added.
- **Metrics come from ``utils.scoring``, grouped per ``best_of``.**
  :func:`score_held_out_series` computes per-series ``rps`` /
  ``log_loss`` / ``marginal_binary_accuracy`` via the per-observation
  scoring functions, and :func:`build_series_evaluation_report`
  aggregates via the batch functions — no metric math is reimplemented
  here. Mixed-``K`` batches (Bo3 ``K=4`` rows beside Bo5 ``K=6`` rows)
  are fine for the batch functions (they validate and score each row
  independently), but the roadmap asks for results "per Bo3 and Bo5"
  separately and blending a 4-category RPS with a 6-category RPS into
  one number is not a meaningful statistic (the RPS scale is
  ``K``-dependent: max value ``K - 1``), so the report produces one
  block per distinct ``best_of`` value present plus an overall
  ``n_eval_total`` count only — no combined-mean-RPS-across-``K``
  number (assumption 5). A ``best_of`` group with zero rows is omitted
  from the report, not an error: v1's ``matches.parquet`` has 96 Bo3
  rows and only 2 Bo5 rows total (across train+test), so the Bo5
  *test-split* group may be empty (it currently is: all 15 held-out v1
  matches are Bo3).
- **``marginal_binary_accuracy``'s default grouping needs no
  override.** ``utils.series_paths.series_outcome_order`` lists all
  ``threshold`` A-win scorelines first and ``K = 2 * threshold``, so
  the default "first ``K // 2`` categories are side A" convention of
  :func:`utils.scoring.marginal_binary_accuracy` matches the series
  vocabulary exactly and no ``group_a_indices`` is passed (assumption
  6). The marginal accuracy therefore answers "did the model pick the
  correct series *winner*".
- **Variable-length probability column, not fixed per-category
  columns.** M19's ``evaluation/harness.py`` stores the 4 predicted
  probabilities as fixed-width :data:`~evaluation.harness.PREDICTION_COLUMNS`
  because every row has the same ``K = 4``. A series scored table
  mixes Bo3 (``K = 4``) and Bo5 (``K = 6``) rows, so a fixed-width
  scheme cannot hold both; the scored table instead stores the
  ``best_of + 1``-length vector as a single list/array-valued
  ``probabilities`` column, per row. This is a deliberate divergence
  from the M19 precedent for a stated, necessary reason; the report's
  per-``best_of`` grouping then hands each homogeneous-``K`` subset to
  the batch functions.
- **No CLI driver in this milestone.** M33a is "the measurement
  machinery," a library-only module (mirroring the M25/M29/M30
  precedent); the CLI/report that runs the M31-vs-M32 comparison and
  writes a JSON artifact is M33b's job (assumption 7).
- **Place in the dependency DAG.** ``utils/ -> features/ -> models/ ->
  evaluation/ -> drivers/``. This module sits in ``evaluation/`` and
  may depend downward on ``models.*`` / ``features.*`` / ``utils.*``
  only — concretely ``models.flat_series_baseline`` (for the thin
  baseline adapter :func:`flat_series_baseline_model`),
  ``utils.series_paths`` (the outcome-order vocabulary),
  ``utils.scoring`` (the metric functions) and ``utils.splits`` (the
  split restriction) — never on ``drivers/`` and never on a sibling
  ``evaluation/`` module (assumption 8, encoded in
  ``tests/test_module_boundaries.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import flat_series_baseline
from models.flat_series_baseline import SeriesModelFn
from utils import scoring, series_paths, splits

# Fixed column order for the held-out series table produced by
# build_held_out_series: one row per held-out match, carrying the
# identifying columns plus the derived observed scoreline
# (``a_wins``/``b_wins``) and its ordinal position within
# utils.series_paths.series_outcome_order (``outcome_index``).
HELD_OUT_SERIES_COLUMNS = (
    "match_id",
    "date",
    "team1_id",
    "team2_id",
    "best_of",
    "best_of_int",
    "a_wins",
    "b_wins",
    "outcome_index",
)

# Fixed column order for the scored table produced by
# score_held_out_series: the held-out identifying columns plus the
# variable-length ``probabilities`` vector column (``best_of_int + 1``
# floats in series_outcome_order order — a list per row, since ``K``
# varies across Bo3/Bo5 rows) and the three per-series scores. The
# variable-length column diverges from the M19 harness's fixed
# PREDICTION_COLUMNS for the stated reason in the module docstring.
SCORED_SERIES_COLUMNS = (
    *HELD_OUT_SERIES_COLUMNS,
    "probabilities",
    "rps",
    "log_loss",
    "marginal_correct",
)


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

    **Deliberate duplication, not an import.** This helper is a local,
    behaviour-identical copy of
    ``models.flat_series_baseline._parse_best_of``: that function is
    private (leading underscore) to a ``models/`` module one layer
    below ``evaluation/``, and importing a private name across that
    boundary would be exactly the lateral reach the module-boundary
    rule forbids (assumption 3). The two copies stay in sync by
    convention until a later milestone promotes the parser to a shared
    utility.

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


def build_held_out_series(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Assemble the held-out series table to evaluate against.

    Restricts ``matches_df`` to the requested ``split`` value (via
    :func:`utils.splits.join_split_to_maps`, reused as-is — it is
    already generic over any ``match_id``-keyed table despite its
    "map-level" docstring framing, and reusing it means the
    stale/mismatched-dataset guard it performs — every match's
    ``match_id`` must exist in ``splits_df`` — applies unchanged,
    mirroring ``evaluation/harness.py``'s own choice to route the
    split restriction through the shared helper rather than a
    hand-rolled filter), then for every remaining match derives the
    observed scoreline directly from ``maps_df`` per assumption 1:
    ``a_wins`` counts that match's maps with
    ``team1_score > team2_score``, ``b_wins`` counts those with
    ``team2_score > team1_score`` (the ``winner`` column is never
    read). The ``"Bo<N>"`` ``best_of`` string is parsed via
    :func:`_parse_best_of`, and the observed scoreline's ordinal index
    within ``utils.series_paths.series_outcome_order`` is computed
    (the terminal scoreline vocabulary M30 fixed, so the ordinal is
    directly scorable by ``utils.scoring``'s index-based metrics).

    Fail-loud doctrine (no silent skipping): an empty split-restricted
    result, a match with zero maps in ``maps_df`` (no played maps, no
    observable scoreline), a map with a tied scoreline, and a map that
    resolves to neither side's win (e.g. a null score) all raise
    ``ValueError`` naming the offending match. A non-terminal derived
    scoreline (data that cannot happen for a real completed series)
    raises via ``series_outcome_order(...).index(...)``.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``,
            ``best_of``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``map_index``, ``team1_score``, ``team2_score``).
        splits_df: The ``splits`` table produced by
            :func:`utils.splits.split_matches` (needs ``match_id`` and
            ``split``).
        split: The split value to hold out, ``"test"`` by default (the
            only split ``utils.splits`` defines for final evaluation).

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`HELD_OUT_SERIES_COLUMNS`
        (``match_id, date, team1_id, team2_id, best_of, best_of_int,
        a_wins, b_wins, outcome_index``), one row per held-out match in
        the order ``matches_df`` produced them. Never empty: an empty
        restricted result raises instead.

    Raises:
        ValueError: If the split-restricted result is empty (no matches
            in the requested split); if a held-out match's ``match_id``
            is absent from ``splits_df`` (propagated from
            :func:`utils.splits.join_split_to_maps`); if a held-out
            match has zero maps in ``maps_df``; if a map has a tied or
            null/unresolvable scoreline (see the docstring's fail-loud
            doctrine); if a ``best_of`` value is not a valid
            ``"Bo<N>"`` string (from :func:`_parse_best_of`); or if the
            derived ``(a_wins, b_wins)`` is not a terminal scoreline
            (from ``series_outcome_order(...).index(...)``).
        KeyError: If any input table lacks a required column
            (``team1_id``/``team2_id``/``date``/``best_of`` on
            ``matches_df``, ``team1_score``/``team2_score``/``map_index``
            on ``maps_df``, ``split`` on ``splits_df``), propagated
            from pandas / the shared helper.
    """
    # Restrict matches to the requested split via the shared helper,
    # which left-attaches the split column and guards against stale
    # datasets (every match_id must be present in splits_df).
    with_split = splits.join_split_to_maps(matches_df, splits_df)
    split_matches = with_split[with_split["split"] == split]

    maps_by_match = {
        match_id: group
        for match_id, group in maps_df.groupby("match_id", sort=True)
    }

    rows: list[dict] = []
    for match_row in split_matches.itertuples(index=False):
        match_id = match_row.match_id
        match_maps = maps_by_match.get(match_id)
        if match_maps is None or len(match_maps) == 0:
            raise ValueError(
                f"match {match_id!r} has zero maps in maps_df; a series "
                "with no played maps cannot have an observed scoreline"
            )
        best_of_int = _parse_best_of(match_row.best_of)
        a_wins = 0
        b_wins = 0
        for map_row in match_maps.itertuples(index=False):
            team1_score = map_row.team1_score
            team2_score = map_row.team2_score
            if pd.isna(team1_score) or pd.isna(team2_score):
                raise ValueError(
                    f"match {match_id!r} map {map_row.map_index!r} has a "
                    "null team score; the observed series scoreline "
                    "cannot be derived from an unfinished map"
                )
            if team1_score == team2_score:
                raise ValueError(
                    f"match {match_id!r} map {map_row.map_index!r} has a "
                    f"tied scoreline ({team1_score}-{team2_score}); a "
                    "tied finished map has no series winner, so the "
                    "observed scoreline cannot be derived"
                )
            if team1_score > team2_score:
                a_wins += 1
            elif team2_score > team1_score:
                b_wins += 1
        if a_wins + b_wins != len(match_maps):
            raise ValueError(
                f"match {match_id!r}: only {a_wins + b_wins} of its "
                f"{len(match_maps)} maps resolved to a team win; every "
                "map must be won by exactly one side (a null or "
                "unparseable score would leave a map uncounted)"
            )
        outcome_index = series_paths.series_outcome_order(best_of_int).index(
            (a_wins, b_wins)
        )
        rows.append(
            {
                "match_id": match_id,
                "date": match_row.date,
                "team1_id": match_row.team1_id,
                "team2_id": match_row.team2_id,
                "best_of": match_row.best_of,
                "best_of_int": best_of_int,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "outcome_index": outcome_index,
            }
        )
    if not rows:
        raise ValueError(
            f"no held-out series for split {split!r}: restricting "
            "matches to that split yields an empty table"
        )
    return pd.DataFrame(rows, columns=HELD_OUT_SERIES_COLUMNS)


def score_held_out_series(
    model_fn: SeriesModelFn,
    held_out_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every held-out series with the model and compute per-series metrics.

    Iterates ``held_out_df`` (as produced by :func:`build_held_out_series`)
    one row per series, calls
    ``model_fn(team1_id, team2_id, best_of, date, matches_df, maps_df)``
    for each (``best_of`` is the original ``"Bo<N>"`` string, matching
    the :data:`SeriesModelFn` convention), validates the returned
    vector's length equals ``best_of_int + 1`` (per-row ``best_of``,
    since Bo3 and Bo5 rows require different lengths — the analogue of
    ``evaluation/harness.py::score_held_out_maps``'s per-map length
    check, which this module mirrors in message style), records the
    vector under a single variable-length ``probabilities`` column, and
    computes the three per-series scores against the row's true
    ``outcome_index`` by calling ``utils.scoring``'s own
    per-observation functions — :func:`utils.scoring.rps`,
    :func:`utils.scoring.log_loss` and
    :func:`utils.scoring.marginal_binary_accuracy` — so the metric math
    lives in exactly one place.

    No ``group_a_indices`` override is passed to
    :func:`utils.scoring.marginal_binary_accuracy`: the default
    first-half convention already matches the series vocabulary exactly
    (``series_outcome_order`` lists the ``threshold`` A-win scorelines
    first and ``K = 2 * threshold``, so side A = "A won the series" —
    assumption 6, confirmed here rather than asserted with an explicit
    override). The model is invoked with the full ``matches_df`` /
    ``maps_df`` (never a filtered subset): the as-of leakage boundary
    is the model's own responsibility, inherited from ``utils.asof``'s
    strict ``<`` cutoff, and is not re-implemented here.

    The returned vector is length-checked here (with a per-series error
    message); its simplex validity (finite, non-negative, sums to 1) is
    not pre-checked — the metric functions validate it themselves and
    their ``ValueError`` propagates unchanged.

    Args:
        model_fn: The model to score; any callable satisfying
            :data:`SeriesModelFn` (returns the ``best_of + 1``
            scoreline probabilities in
            ``utils.series_paths.series_outcome_order`` order).
        held_out_df: The held-out series table from
            :func:`build_held_out_series` (needs
            :data:`HELD_OUT_SERIES_COLUMNS` exactly).
        matches_df: The full materialised ``matches`` table, passed
            through to ``model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``model_fn`` unchanged.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SCORED_SERIES_COLUMNS`
        (the held-out identifying columns, the variable-length
        ``probabilities`` vector column, and ``rps`` / ``log_loss`` /
        ``marginal_correct``), one row per held-out series in the same
        order as ``held_out_df``. ``marginal_correct`` is a ``bool``
        (the per-observation form of
        :func:`utils.scoring.marginal_binary_accuracy`).

    Raises:
        ValueError: If ``model_fn`` returns a sequence whose length is
            not exactly ``best_of_int + 1`` (with the offending
            series' ``match_id``/``best_of`` named); or if any returned
            vector fails the simplex validation or the true ordinal is
            out of range (propagated from :func:`utils.scoring.rps` /
            :func:`utils.scoring.log_loss` /
            :func:`utils.scoring.marginal_binary_accuracy` — including
            ``log_loss``'s hard error when the true scoreline was
            assigned exactly zero probability).
        KeyError: If ``held_out_df`` lacks a
            :data:`HELD_OUT_SERIES_COLUMNS` column (propagated from
            pandas/``itertuples``).
    """
    rows: list[dict] = []
    for row in held_out_df.itertuples(index=False):
        probs = list(
            model_fn(
                row.team1_id,
                row.team2_id,
                row.best_of,
                row.date,
                matches_df,
                maps_df,
            )
        )
        expected_len = row.best_of_int + 1
        if len(probs) != expected_len:
            raise ValueError(
                f"model_fn returned {len(probs)} probabilities for series "
                f"(match {row.match_id!r}, best_of {row.best_of!r}); "
                f"expected exactly {expected_len} in "
                f"series_outcome_order({row.best_of_int}) order"
            )
        scored: dict = {
            column: getattr(row, column) for column in HELD_OUT_SERIES_COLUMNS
        }
        scored["probabilities"] = probs
        scored["rps"] = scoring.rps(probs, row.outcome_index)
        scored["log_loss"] = scoring.log_loss(probs, row.outcome_index)
        # No group_a_indices override (assumption 6): series_outcome_order
        # lists the threshold A-win scorelines first (K = 2 * threshold),
        # so the default first-half grouping already makes side A = "A
        # won the series".
        scored["marginal_correct"] = scoring.marginal_binary_accuracy(
            probs, row.outcome_index
        )
        rows.append(scored)
    return pd.DataFrame(rows, columns=SCORED_SERIES_COLUMNS)


def build_series_evaluation_report(scored_df: pd.DataFrame) -> dict:
    """Build the JSON-serializable per-``best_of`` series evaluation report.

    A pure dict builder (no I/O): turns the scored table from
    :func:`score_held_out_series` into the report every evaluation run
    writes. The headline metrics are recomputed from the
    ``probabilities`` column through ``utils.scoring``'s shared batch
    functions (:func:`utils.scoring.mean_rps` / ``mean_log_loss`` /
    ``mean_marginal_binary_accuracy``) rather than re-averaged from the
    per-row score columns, so the report's headline numbers are
    traceable to the shared metric implementations and there is no
    second, ad hoc mean implementation to drift.

    The report is grouped per distinct ``best_of`` value present
    (assumption 5): each ``"Bo<N>"`` key holds that group's ``n_eval``,
    ``mean_rps``, ``mean_log_loss`` and ``marginal_binary_accuracy``.
    A ``best_of`` group with zero rows is omitted entirely rather than
    erroring (the Bo5 test-split group may legitimately be empty), and
    only an overall ``n_eval_total`` count spans groups — no
    combined-mean-RPS-across-``K`` number, since the RPS scale is
    ``K``-dependent and a blended mean would be meaningless.

    Args:
        scored_df: The scored table from :func:`score_held_out_series`
            (needs ``best_of``, ``probabilities`` and
            ``outcome_index``; the per-row ``rps``/``log_loss``/
            ``marginal_correct`` columns are not read).

    Returns:
        A dict with one key per distinct ``best_of`` value present
        (each holding ``n_eval`` (int), ``mean_rps`` (float),
        ``mean_log_loss`` (float), ``marginal_binary_accuracy``
        (float)) plus ``n_eval_total`` (int, the full row count). Every
        value is a plain str/int/float/dict, so the whole dict is
        directly ``json.dumps``-serializable.

    Raises:
        ValueError: If ``scored_df`` is empty (a mean over zero series
            is undefined); or if any prediction row fails the metric
            validation (propagated from the batch functions, e.g.
            ``log_loss`` on a zero-probability true scoreline).
        KeyError: If ``scored_df`` lacks ``best_of``,
            ``probabilities`` or ``outcome_index`` (propagated from
            pandas).
    """
    if len(scored_df) == 0:
        raise ValueError(
            "cannot build a series evaluation report over zero scored "
            "series"
        )
    report: dict = {}
    for best_of in sorted(scored_df["best_of"].unique()):
        subset = scored_df[scored_df["best_of"] == best_of]
        prob_rows = subset["probabilities"].to_numpy()
        true_indices = subset["outcome_index"].to_numpy()
        report[str(best_of)] = {
            "n_eval": len(subset),
            "mean_rps": scoring.mean_rps(prob_rows, true_indices),
            "mean_log_loss": scoring.mean_log_loss(prob_rows, true_indices),
            "marginal_binary_accuracy": (
                scoring.mean_marginal_binary_accuracy(
                    prob_rows, true_indices
                )
            ),
        }
    report["n_eval_total"] = len(scored_df)
    return report


def build_series_multi_arm_report(
    scored_by_arm: dict[str, pd.DataFrame],
    baseline_arm: str,
) -> dict:
    """Build the N-arm series comparison report over identically-scored tables.

    An independent reimplementation of
    ``evaluation/veto_evaluation.py::build_veto_multi_arm_report``'s
    *shape*, adapted to series data (assumption 4 — the sibling
    ``evaluation/`` module is deliberately not imported): takes one
    scored table per arm from :func:`score_held_out_series` (all
    produced on the *identical* held-out rows, in the identical order),
    validates they are all row-aligned (identical ``match_id`` values at
    identical positions — the identifying key is just ``match_id``
    here, one row per series, where the veto precedent uses
    ``(match_id, step_index)``; a misaligned comparison would silently
    pair two different series' scores and corrupt every delta), and
    returns ``{arm_name: <report>, "deltas_vs_<baseline_arm>":
    {arm_name: {...}}}`` where each arm's block is the
    :func:`build_series_evaluation_report` dict for that arm and the
    delta block holds one per-``best_of`` dict per *non-baseline* arm,
    each entry arm-minus-baseline. No expected sign is stated for any
    delta — the actually-measured values are an empirical finding, not
    an assumed one.

    Deltas are computed per ``best_of`` group (a Bo3 delta and a Bo5
    delta are different questions and must not be blended into one
    number): each non-baseline arm's delta block holds one
    ``{mean_rps_delta, mean_log_loss_delta,
    marginal_binary_accuracy_delta}`` entry per ``best_of`` group the
    arm and the baseline both have, and a group present in only one of
    the two is omitted from that arm's delta block (assumption 5) —
    the group still appears in each arm's own report block.

    Args:
        scored_by_arm: A dict mapping each arm's name to its
            :func:`score_held_out_series` table (needs ``match_id``,
            ``best_of``, ``probabilities`` and ``outcome_index``). At
            least two arms are required (a one-arm "comparison" is
            meaningless).
        baseline_arm: The arm every delta is measured against; must be
            a key of ``scored_by_arm``. The baseline arm's own report
            block appears like any other arm's, but it has no delta
            entry in the delta block.

    Returns:
        A dict with one key per arm name (each the
        :func:`build_series_evaluation_report` dict for that arm) plus
        the key ``"deltas_vs_<baseline_arm>"`` mapping each
        non-baseline arm name to its per-``best_of`` dict of
        ``{"mean_rps_delta": float, "mean_log_loss_delta": float,
        "marginal_binary_accuracy_delta": float}`` arm-minus-baseline
        deltas. Every value is a plain str/int/float/dict, so the whole
        dict is directly ``json.dumps``-serializable.

    Raises:
        ValueError: If ``scored_by_arm`` has fewer than two arms; if
            ``baseline_arm`` is not a key of ``scored_by_arm``; if any
            two arms' scored tables have different row counts or differ
            in any ``match_id`` value at the same position (the
            row-alignment contract, mirroring
            ``evaluation/veto_evaluation.py``'s guard); or if any arm's
            table is empty (propagated from
            :func:`build_series_evaluation_report`).
        KeyError: If any arm's table lacks a required column
            (propagated from pandas column indexing).
    """
    if len(scored_by_arm) < 2:
        raise ValueError(
            f"build_series_multi_arm_report needs at least two arms to "
            f"compare, got {len(scored_by_arm)}"
        )
    if baseline_arm not in scored_by_arm:
        raise ValueError(
            f"baseline_arm {baseline_arm!r} is not a scored arm; got "
            f"arms {sorted(scored_by_arm)}"
        )

    arm_names = list(scored_by_arm)
    reference_ids = scored_by_arm[arm_names[0]]["match_id"].to_numpy()
    for name in arm_names[1:]:
        scored = scored_by_arm[name]
        if len(scored) != len(reference_ids):
            raise ValueError(
                f"scored tables have different row counts: "
                f"{arm_names[0]} {len(reference_ids)} vs {name} "
                f"{len(scored)}; they must describe the same held-out "
                "series"
            )
        arm_ids = scored["match_id"].to_numpy()
        mismatch = arm_ids != reference_ids
        if mismatch.any():
            idx = int(np.argmax(mismatch))
            raise ValueError(
                "scored tables are not row-aligned: the held-out series "
                f"differ at position {idx} ({arm_names[0]} "
                f"{reference_ids[idx]!r} vs {name} {arm_ids[idx]!r}); "
                "score all arms on the identical build_held_out_series "
                "table"
            )

    arm_reports = {
        name: build_series_evaluation_report(scored)
        for name, scored in scored_by_arm.items()
    }
    report = dict(arm_reports)
    baseline_report = arm_reports[baseline_arm]
    deltas_key = f"deltas_vs_{baseline_arm}"
    report[deltas_key] = {}
    for name, arm_report in arm_reports.items():
        if name == baseline_arm:
            continue
        deltas: dict = {}
        for best_of, block in arm_report.items():
            if best_of == "n_eval_total":
                continue
            if best_of not in baseline_report:
                # A best_of group present in this arm but absent from
                # the baseline is omitted from the delta block (assumption
                # 5): a Bo3 delta and a Bo5 delta are different questions
                # and must not be blended or fabricated.
                continue
            deltas[best_of] = {
                "mean_rps_delta": (
                    block["mean_rps"] - baseline_report[best_of]["mean_rps"]
                ),
                "mean_log_loss_delta": (
                    block["mean_log_loss"]
                    - baseline_report[best_of]["mean_log_loss"]
                ),
                "marginal_binary_accuracy_delta": (
                    block["marginal_binary_accuracy"]
                    - baseline_report[best_of]["marginal_binary_accuracy"]
                ),
            }
        report[deltas_key][name] = deltas
    return report


def flat_series_baseline_model(
    team1_id: str,
    team2_id: str,
    best_of: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> tuple[float, ...]:
    """The M32 flat series baseline, adapted to the SeriesModelFn shape.

    Wraps
    :func:`models.flat_series_baseline.predict_series_outcome` — which
    returns a
    :class:`models.flat_series_baseline.FlatSeriesPrediction` dataclass,
    not a bare sequence — into the :data:`SeriesModelFn` callable shape
    by calling ``.as_tuple()``, so the baseline is usable directly by
    :func:`score_held_out_series` and registrable by a later CLI/M33b
    without the harness special-casing the dataclass. This mirrors
    ``evaluation/harness.py::four_way_baseline_model``'s precedent at
    the series level.

    Args:
        team1_id: The queried team1's stable id ("A" in the series
            scoreline vocabulary; the first element of every
            ``(a_wins, b_wins)`` scoreline).
        team2_id: The queried team2's stable id ("B" in the series
            scoreline vocabulary).
        best_of: The series length as the ``"Bo<N>"`` string carried by
            ``matches.parquet`` (e.g. ``"Bo3"``, ``"Bo5"``, ``"Bo1"``).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.

    Returns:
        A tuple of ``best_of + 1`` ``float`` probabilities in
        ``utils.series_paths.series_outcome_order`` order, summing to
        ``1.0`` — exactly what ``FlatSeriesPrediction.as_tuple()``
        returns.

    Raises:
        ValueError: If ``best_of`` is not a valid ``"Bo<N>"`` string;
            if an as-of map has a null/NaN score or tied scores; or if
            the query date or a row date is null/unparseable/
            timezone-aware (all propagated from
            :func:`models.flat_series_baseline.predict_series_outcome`).
        KeyError: If either table lacks a required column (propagated
            from the same call).
        TypeError: If the query date is list-like (propagated from the
            same call).
    """
    return flat_series_baseline.predict_series_outcome(
        team1_id, team2_id, best_of, date, matches_df, maps_df
    ).as_tuple()
