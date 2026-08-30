"""Map-outcome evaluation harness (roadmap M19).

Scores a generic four-way outcome model against observed outcomes on
the held-out test split: RPS (headline), multi-class log loss, marginal
binary accuracy, and per-category calibration (including the predicted
vs observed OT rate and a most-likely-miscalibrated-category flag),
reporting the M18 four-way baseline (``models.four_way_baseline``,
wrapped by :func:`four_way_baseline_model`) as the floor every later
model (M20+) must beat.

Scope / conventions (recorded here, do not re-derive later):

- **Pure and dependency-light.** This module does no file I/O, has no
  CLI / ``argparse`` entry point, and never touches ``drivers/``. It
  takes already-loaded DataFrames and a model callable in, and returns
  DataFrames/dicts out — matching ``utils.splits`` / ``utils.scoring``
  / ``models.four_way_baseline``'s existing convention. All Parquet/
  JSON I/O and CLI wiring lives in ``drivers/evaluate.py``.
- **Place in the dependency DAG.** This module sits one rung above
  ``models/``: ``utils/ -> features/ -> models/ -> evaluation/ ->
  drivers/``. It may depend downward on ``models.*``, ``features.*``
  and ``utils.*`` only; nothing in those layers may depend on
  ``evaluation/``, and this module must never import from ``drivers/``
  (encoded as a regression test in ``tests/test_module_boundaries.py``).
- **Generic over any 4-way model callable.** A model is any callable
  with signature ``(team1_id, team2_id, map_name, date, matches_df,
  maps_df) -> Sequence[float]`` (:data:`ModelFn`) returning the four
  category probabilities in :data:`OUTCOME_LABELS` order. M20+ models
  plug into this harness unchanged; the M18 baseline is exposed as a
  thin adapter (:func:`four_way_baseline_model`) rather than
  special-cased.
- **Held-out set = the M10 ``test`` split.** ``utils.splits``'s own
  module docstring reserves walk-forward / OOF assembly for calibration
  (M24) and hyperparameter selection; ``"test"`` is explicitly "held
  out for final evaluation only", so the harness reads
  ``splits.parquet``, restricts to ``split == "test"`` via
  :func:`utils.splits.join_split_to_maps`, and joins that down to maps.
- **Point-in-time predictions use the map's own match date as the
  as-of cutoff**, exactly as ``models.four_way_baseline.predict_map_outcome``
  already does. Leakage safety is inherited unchanged from
  ``utils.asof``'s strict ``<`` boundary inside the feature calls; the
  harness does not add or re-implement any date filtering itself.
- **All four metrics come from ``utils.scoring``.** ``mean_rps``,
  ``mean_log_loss``, ``mean_marginal_binary_accuracy`` and the
  per-observation ``rps`` / ``log_loss`` / ``marginal_binary_accuracy``
  are called directly — none of the metric math is reimplemented here.
- **Calibration definition.** Per category: ``predicted_mean_prob`` is
  the arithmetic mean of the model's predicted probability for that
  category over every held-out map, and ``observed_frequency`` is the
  fraction of held-out maps whose true ``outcome_ordinal`` equals that
  category. The most-likely-miscalibrated category is whichever has the
  largest ``abs(predicted_mean_prob - observed_frequency)`` gap (ties
  resolve to the earliest category in :data:`OUTCOME_LABELS` order). In
  addition, the report surfaces one explicit
  ``predicted_vs_observed_ot_rate`` field: predicted = mean of
  ``p_a_ot + p_b_ot`` over the held-out set, observed = the fraction of
  held-out maps in ``{A-OT, B-OT}``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from models import four_way_baseline
from utils import scoring, splits

# The four outcome categories in ordinal order, copied from (kept in
# sync with, deliberately *not* imported from)
# ``models.four_way_baseline.OUTCOME_LABELS`` — mirroring that module's
# own precedent of not importing the vocabulary across a layer boundary.
# Index 0 is "A-regulation", 1 "A-OT", 2 "B-OT", 3 "B-regulation";
# "A" = team1, "B" = team2, matching drivers.labels' "A and B are
# column positions, not team identities" convention.
OUTCOME_LABELS = ("A-regulation", "A-OT", "B-OT", "B-regulation")

# The four predicted-probability columns of the scored table, in
# OUTCOME_LABELS order: column ``i`` holds the predicted probability of
# category ``i`` (so ``p_a_ot`` is category index 1, ``p_b_ot`` index
# 2). Named after the ``FourWayPrediction`` dataclass fields so the
# columns are grep-able against ``models.four_way_baseline``.
PREDICTION_COLUMNS = ("p_a_regulation", "p_a_ot", "p_b_ot", "p_b_regulation")

# Fixed column order for the held-out table produced by
# build_held_out_maps: the per-map identifying columns plus the true
# ordinal.
HELD_OUT_COLUMNS = (
    "match_id",
    "map_index",
    "date",
    "team1_id",
    "team2_id",
    "map_name",
    "outcome_ordinal",
)

# Fixed column order for the scored table produced by
# score_held_out_maps: the identifying columns, the true ordinal, the
# four predicted probabilities (in OUTCOME_LABELS order), and the three
# per-map scores.
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

# The generic model interface every evaluated model must satisfy: a
# callable taking the two team ids, the map name, the as-of date, and
# the full matches/maps tables, and returning the four category
# probabilities in OUTCOME_LABELS order (a plain sequence of floats;
# the harness validates its length but delegates simplex validation to
# utils.scoring's metric functions).
ModelFn = Callable[
    [str, str, str, str, pd.DataFrame, pd.DataFrame],
    Sequence[float],
]


def build_held_out_maps(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    split: str = "test",
) -> pd.DataFrame:
    """Assemble the held-out map table to evaluate against.

    Joins ``maps_df`` to ``matches_df`` (to pick up ``team1_id``,
    ``team2_id`` and ``date`` — the columns the generic model interface
    needs — since ``maps.parquet`` itself has no date/team columns),
    joins the result to ``labels_df`` on ``(match_id, map_index)`` (to
    pick up the true ``outcome_ordinal``), and restricts to the
    requested ``split`` value by left-attaching the two-valued split
    column via :func:`utils.splits.join_split_to_maps` and filtering.
    The split restriction happens through that shared helper (not a
    reimplemented merge) so the stale/mismatched-dataset guard it
    performs (every map's ``match_id`` must exist in ``splits_df``)
    applies unchanged.

    The maps-to-matches join is an inner join: a map whose match is not
    materialised cannot be scored and is silently dropped. The
    maps-to-labels join is also an inner join, documenting the chosen
    "already-labelled maps only" behavior: a map whose label row is
    absent (the skipped-null-score case ``drivers.labels`` permits,
    where labels can legitimately have fewer rows than maps) is
    silently excluded from evaluation rather than treated as an error.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_id``, ``team2_id``, ``date``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``map_index``, ``map_name``).
        labels_df: The ``labels`` table (needs ``match_id``,
            ``map_index``, ``outcome_ordinal``).
        splits_df: The ``splits`` table produced by
            :func:`utils.splits.split_matches` (needs ``match_id`` and
            ``split``).
        split: The split value to hold out, ``"test"`` by default
            (the only split ``utils.splits`` defines for final
            evaluation).

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`HELD_OUT_COLUMNS`
        (``match_id, map_index, date, team1_id, team2_id, map_name,
        outcome_ordinal``), one row per held-out map in the order
        ``maps_df`` produced them. Never empty: an empty restricted
        result raises instead.

    Raises:
        ValueError: If the split-restricted, label-joined result is
            empty (no maps in the requested split — e.g. a splits
            table with no rows of that value); or if any map's
            ``match_id`` is absent from ``splits_df`` (propagated from
            :func:`utils.splits.join_split_to_maps`).
        KeyError: If any input table lacks a required column
            (``team1_id``/``team2_id``/``date`` on ``matches_df``,
            ``outcome_ordinal`` on ``labels_df``, ``split`` on
            ``splits_df``; ``match_id``/``map_index`` on the maps and
            labels tables), propagated from pandas/the shared helper.
    """
    # Join maps -> matches for the team/date columns the model needs.
    # Inner join: a map whose match is not materialised cannot be
    # evaluated (its team ids and date are unknowable), so it is
    # silently dropped rather than erroring.
    joined = maps_df.merge(
        matches_df[["match_id", "team1_id", "team2_id", "date"]],
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
    # column list.
    held_out = held_out[list(HELD_OUT_COLUMNS)]
    if len(held_out) == 0:
        raise ValueError(
            f"no held-out maps for split {split!r}: joining maps to "
            "matches/labels and restricting to that split yields an "
            "empty table"
        )
    return held_out


def score_held_out_maps(
    model_fn: ModelFn,
    held_out_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score every held-out map with the model and compute per-map metrics.

    Iterates ``held_out_df`` (as produced by :func:`build_held_out_maps`)
    one row per map, calls ``model_fn(team1_id, team2_id, map_name,
    date, matches_df, maps_df)`` for each, records the returned
    4-vector under :data:`PREDICTION_COLUMNS` (in :data:`OUTCOME_LABELS`
    order), and computes the three per-map scores against the row's
    true ``outcome_ordinal`` by calling ``utils.scoring``'s own
    per-observation functions — :func:`utils.scoring.rps`,
    :func:`utils.scoring.log_loss` and
    :func:`utils.scoring.marginal_binary_accuracy` — so the metric math
    lives in exactly one place. The model is invoked with the full
    ``matches_df``/``maps_df`` (never a filtered subset): the as-of
    leakage boundary is the model's own responsibility, inherited from
    ``utils.asof``'s strict ``<`` cutoff, and is not re-implemented
    here.

    The returned 4-vector is length-checked against
    :data:`OUTCOME_LABELS` here (with a per-map error message); its
    simplex validity (finite, non-negative, sums to 1) is not
    pre-checked — the metric functions validate it themselves and their
    ``ValueError`` propagates unchanged.

    Args:
        model_fn: The model to score; any callable satisfying
            :data:`ModelFn` (returns the four category probabilities in
            :data:`OUTCOME_LABELS` order).
        held_out_df: The held-out map table from
            :func:`build_held_out_maps` (needs :data:`HELD_OUT_COLUMNS`
            exactly).
        matches_df: The full materialised ``matches`` table, passed
            through to ``model_fn`` unchanged.
        maps_df: The full materialised ``maps`` table, passed through
            to ``model_fn`` unchanged.

    Returns:
        A ``pandas.DataFrame`` with exactly :data:`SCORED_COLUMNS`
        (the identifying columns, the true ``outcome_ordinal``, the
        four predicted probabilities, and ``rps`` / ``log_loss`` /
        ``marginal_correct``), one row per held-out map in the same
        order as ``held_out_df``. ``marginal_correct`` is a ``bool``
        (the per-observation form of
        :func:`utils.scoring.marginal_binary_accuracy`).

    Raises:
        ValueError: If ``model_fn`` returns a sequence whose length is
            not exactly ``len(OUTCOME_LABELS)`` (with the offending
            map's ``match_id``/``map_index`` named); or if any
            returned vector fails the simplex validation or the true
            ordinal is out of range (propagated from
            :func:`utils.scoring.rps` / :func:`utils.scoring.log_loss`
            / :func:`utils.scoring.marginal_binary_accuracy` —
            including ``log_loss``'s hard error when the true category
            was assigned exactly zero probability).
        KeyError: If ``held_out_df`` lacks a :data:`HELD_OUT_COLUMNS`
            column (propagated from pandas/``itertuples``).
    """
    rows: list[dict] = []
    for row in held_out_df.itertuples(index=False):
        probs = list(
            model_fn(
                row.team1_id,
                row.team2_id,
                row.map_name,
                row.date,
                matches_df,
                maps_df,
            )
        )
        if len(probs) != len(OUTCOME_LABELS):
            raise ValueError(
                f"model_fn returned {len(probs)} probabilities for map "
                f"(match {row.match_id!r}, map_index {row.map_index!r}); "
                f"expected exactly {len(OUTCOME_LABELS)} in "
                f"{OUTCOME_LABELS} order"
            )
        ordinal = row.outcome_ordinal
        scored: dict = {
            "match_id": row.match_id,
            "map_index": row.map_index,
            "date": row.date,
            "team1_id": row.team1_id,
            "team2_id": row.team2_id,
            "map_name": row.map_name,
            "outcome_ordinal": ordinal,
        }
        for column, prob in zip(PREDICTION_COLUMNS, probs):
            scored[column] = prob
        scored["rps"] = scoring.rps(probs, ordinal)
        scored["log_loss"] = scoring.log_loss(probs, ordinal)
        scored["marginal_correct"] = scoring.marginal_binary_accuracy(
            probs, ordinal
        )
        rows.append(scored)
    return pd.DataFrame(rows, columns=SCORED_COLUMNS)


def build_evaluation_report(
    scored_df: pd.DataFrame,
    category_labels: Sequence[str] = OUTCOME_LABELS,
) -> dict:
    """Build the JSON-serializable evaluation report for a scored table.

    A pure dict builder (no I/O): turns the scored table from
    :func:`score_held_out_maps` into the report every evaluation run
    writes. The headline metrics are recomputed from the prediction
    columns through ``utils.scoring``'s shared batch functions
    (:func:`utils.scoring.mean_rps` / ``mean_log_loss`` /
    ``mean_marginal_binary_accuracy``) rather than re-averaged from the
    per-row score columns, so the report's headline numbers are
    traceable to the shared metric implementations and there is no
    second, ad hoc mean implementation to drift.

    The per-category calibration table holds one entry per
    :data:`category_labels` (default :data:`OUTCOME_LABELS`):
    ``predicted_mean_prob`` (the arithmetic mean of the model's
    predicted probability for that category over every held-out map),
    ``observed_frequency`` (the fraction of held-out maps whose true
    ``outcome_ordinal`` equals the category index), and ``gap`` (their
    absolute difference). ``most_miscalibrated_category`` is the
    category with the largest ``gap`` (ties resolve to the earliest
    category in ``category_labels`` order).
    ``predicted_vs_observed_ot_rate`` is ``{"predicted", "observed",
    "gap"}`` where predicted is the mean of ``p_a_ot + p_b_ot`` over
    the held-out set (equal to the sum of the two OT categories' mean
    probabilities) and observed is the fraction of held-out maps whose
    true ordinal is ``1`` (A-OT) or ``2`` (B-OT).

    Args:
        scored_df: The scored table from :func:`score_held_out_maps`
            (needs :data:`PREDICTION_COLUMNS` and ``outcome_ordinal``;
            the per-row ``rps``/``log_loss``/``marginal_correct``
            columns are not read).
        category_labels: The category vocabulary, one label per
            prediction column in order; defaults to
            :data:`OUTCOME_LABELS`.

    Returns:
        A dict with keys ``n_eval`` (int), ``mean_rps`` (float),
        ``mean_log_loss`` (float), ``marginal_binary_accuracy``
        (float), ``calibration`` (a list of per-category dicts in
        ``category_labels`` order, each with ``category`` /
        ``predicted_mean_prob`` / ``observed_frequency`` / ``gap``),
        ``most_miscalibrated_category`` (str), and
        ``predicted_vs_observed_ot_rate`` (a dict with ``predicted`` /
        ``observed`` / ``gap``). Every value is a plain
        str/int/float/list/dict, so the whole dict is directly
        ``json.dumps``-serializable.

    Raises:
        ValueError: If ``category_labels`` has a different length than
            :data:`PREDICTION_COLUMNS`; if ``scored_df`` is empty
            (propagated from :func:`utils.scoring.mean_rps` — the
            "mean over zero predictions" case the harness deliberately
            does not guard redundantly); or if any prediction row fails
            the metric validation (propagated from the batch
            functions, e.g. ``log_loss`` on a zero-probability true
            category).
        KeyError: If ``scored_df`` lacks a prediction column or
            ``outcome_ordinal`` (propagated from pandas).
    """
    if len(category_labels) != len(PREDICTION_COLUMNS):
        raise ValueError(
            f"category_labels has {len(category_labels)} entries but "
            f"there are {len(PREDICTION_COLUMNS)} prediction columns; "
            "they must match one-to-one"
        )
    prob_rows = scored_df[list(PREDICTION_COLUMNS)].to_numpy()
    true_indices = scored_df["outcome_ordinal"].to_numpy()

    # Headline metrics first: the shared batch functions raise on an
    # empty scored_df before any per-column mean can produce NaN.
    mean_rps = scoring.mean_rps(prob_rows, true_indices)
    mean_log_loss = scoring.mean_log_loss(prob_rows, true_indices)
    marginal_accuracy = scoring.mean_marginal_binary_accuracy(
        prob_rows, true_indices
    )

    calibration: list[dict] = []
    for i, label in enumerate(category_labels):
        predicted_mean_prob = float(scored_df[PREDICTION_COLUMNS[i]].mean())
        observed_frequency = float(
            (scored_df["outcome_ordinal"] == i).mean()
        )
        calibration.append(
            {
                "category": label,
                "predicted_mean_prob": predicted_mean_prob,
                "observed_frequency": observed_frequency,
                "gap": abs(predicted_mean_prob - observed_frequency),
            }
        )
    most_miscalibrated = max(
        calibration, key=lambda entry: entry["gap"]
    )["category"]

    ot_predicted = float(
        scored_df[PREDICTION_COLUMNS[1]].mean()
        + scored_df[PREDICTION_COLUMNS[2]].mean()
    )
    ot_observed = float(scored_df["outcome_ordinal"].isin((1, 2)).mean())

    return {
        "n_eval": len(scored_df),
        "mean_rps": mean_rps,
        "mean_log_loss": mean_log_loss,
        "marginal_binary_accuracy": marginal_accuracy,
        "calibration": calibration,
        "most_miscalibrated_category": most_miscalibrated,
        "predicted_vs_observed_ot_rate": {
            "predicted": ot_predicted,
            "observed": ot_observed,
            "gap": abs(ot_predicted - ot_observed),
        },
    }


def four_way_baseline_model(
    team1_id: str,
    team2_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """The M18 four-way baseline, adapted to the generic ModelFn shape.

    Wraps :func:`models.four_way_baseline.predict_map_outcome` — which
    returns a :class:`models.four_way_baseline.FourWayPrediction`
    dataclass, not a bare sequence — into the :data:`ModelFn` callable
    shape by calling ``.as_tuple()``, so the baseline is usable
    directly by :func:`score_held_out_maps` and registrable by
    ``drivers/evaluate.py``'s ``MODEL_REGISTRY`` without the harness
    special-casing the dataclass.

    The shrinkage strength ``k`` is ``features.map_win_rate.DEFAULT_K``
    (the value ``predict_map_outcome`` defaults to, matching M18's own
    default). It is deliberately not a parameter of this adapter: a
    caller wanting a different ``k`` builds their own ``ModelFn`` from
    ``predict_map_outcome`` directly, e.g.
    ``lambda t1, t2, m, d, mm, mp: predict_map_outcome(
    t1, t2, m, d, mm, mp, k=5.0).as_tuple()``.

    Args:
        team1_id: The queried team1's stable id ("A" in the outcome
            vocabulary).
        team2_id: The queried team2's stable id ("B" in the outcome
            vocabulary).
        map_name: The map to predict for (normalized inside the
            feature estimator, so "breeze" and " Breeze " both match
            "Breeze").
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The full materialised ``matches`` table.
        maps_df: The full materialised ``maps`` table.

    Returns:
        A 4-tuple of ``float`` probabilities in :data:`OUTCOME_LABELS`
        order (``p_a_regulation, p_a_ot, p_b_ot, p_b_regulation``),
        summing to ``1.0`` — exactly what
        ``FourWayPrediction.as_tuple()`` returns.

    Raises:
        ValueError: If ``k`` (the default) is not a positive finite
            real number; if an as-of map has a null/NaN score or tied
            scores; or if the query date or a row date is
            null/unparseable/timezone-aware (all propagated from
            :func:`models.four_way_baseline.predict_map_outcome`).
        KeyError: If either table lacks a required column (propagated
            from the same call).
        TypeError: If the query date is list-like (propagated from the
            same call).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    return four_way_baseline.predict_map_outcome(
        team1_id, team2_id, map_name, date, matches_df, maps_df
    ).as_tuple()
