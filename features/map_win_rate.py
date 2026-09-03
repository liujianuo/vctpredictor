"""Bayesian-shrunk per-map win rates (roadmap M13).

A partial-pooling estimator of a team's win rate on a specific map: the
posterior mean of a Binomial likelihood (``wins`` successes in ``games``
trials) under a ``Beta(alpha0, beta0)`` prior with ``alpha0 = k*prior``
and ``beta0 = k*(1 - prior)``, i.e. ``mean = (wins + k*prior) / (games
+ k)``. The prior ``prior`` is the team's own overall (all-map) win
rate computed from the *same* as-of history — not a fixed constant —
so a team with a great general record gets its small map-specific sample
pulled up toward that record, and vice versa. ``k`` is the effective
prior sample size (``alpha0 + beta0 = k``): it is a *required* parameter
of the estimator and is chosen by cross-validation in :func:`select_k`,
not hard-coded here.

Leakage contract (the whole point of M13 depending on M12):

- Every function obtains match/map history exclusively through
  ``utils.asof`` (``maps_as_of`` / ``features_as_of``); nothing in this
  module reads ``matches.parquet`` / ``maps.parquet`` directly, and no
  computation touches a row whose date is ``>=`` the query date (the
  as-of layer enforces a strict ``<`` boundary). ``select_k`` scores
  every held-out map against an as-of snapshot taken at that map's own
  match timestamp, so no map ever uses its own outcome or a later-in-
  time map to inform its own estimate.
- Win/loss is derived from *scores*, never from the ``winner`` column.
  ``data/v1/maps.parquet``'s ``winner`` holds a display team-name string
  (e.g. ``"FNATIC"``), not a stable ``"team1"``/``"team2"`` marker —
  confirmed by inspection (unique values are all display names). The
  stable side identity is the ``team_is_team1`` orientation column that
  ``utils.asof.maps_as_of`` already attaches: when it is ``True`` the
  queried team's score is ``team1_score``, otherwise ``team2_score``.
  A win is ``our_score > their_score``; a tie raises ``ValueError``
  (a finished map must have a winner, and silently counting it as a
  loss would corrupt the rate).

Data-shape findings recorded per plan item 1 (re-derived against real
``data/v1``, not assumed):

- ``maps.parquet["winner"]`` unique values are display names
  (``FNATIC``, ``Eternal Fire``, ...), confirming the score-comparison
  rule above is mandatory, not optional.
- ``maps.parquet["map_name"]`` values are already title-cased
  (``Breeze``, ``Bind``, ``Lotus``, ``Haven``, ``Pearl``, ``Split``,
  ``Fracture``, ``Ascent``, ``Sunset``, ``Summit``, ``Abyss``) and pass
  ``utils.config.normalize_map_name`` unchanged; the estimator still
  normalizes every name (queried argument *and* per-row value) so a
  caller passing ``"breeze"`` or ``" Breeze "`` matches ``"Breeze"``.
- No map row has ``team1_score == team2_score`` (0 ties in v1); the
  tie guard is defensive fail-loudly coverage, not a live-data fix.
- No map row has a non-null ``winner`` with a null ``team1_score`` or
  ``team2_score`` (0 such rows in v1); the null-score guard added in
  task 017 is defensive fail-loudly coverage, not a live-data fix.

Boundary note: :func:`select_k` imports ``utils.splits``
(``split_matches`` + ``walk_forward_folds``) to reuse the chronological
fold machinery rather than reimplementing it. That dependency is
``features`` -> ``utils`` (feature modules may depend downward on
genuine ``utils/`` utilities; both modules are pure in-memory libraries
with no CLI entry point and no file I/O at import time), so it no
longer inverts the established ``drivers`` -> ``utils`` layering rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features._shared import (
    TEAM1_SCORE_COL,
    TEAM2_SCORE_COL,
    _validate_k,
    _wins_from_oriented_maps,
)
from utils import asof, config, scoring
from utils.splits import (
    DEFAULT_N_FOLDS,
    DEFAULT_TEST_FRAC,
    MIN_FOLD_BLOCK_MATCHES,
    split_matches,
    walk_forward_folds,
)

# The map-name column this module reads; ``team_is_team1`` and ``date``
# come from utils.asof's maps output. The score columns and the
# shrinkage/score-orientation helpers live in features._shared (shared
# with features.h2h_context), so they are imported above, not defined
# here.
MAP_NAME_COL = "map_name"

# Documented fallback shrinkage strength for ad-hoc callers. It is NOT
# what cross-validation reports: the chosen value is :func:`select_k`'s
# ``best_k``, and this constant only gives hand-written calls a sane
# default when no CV has been run.
DEFAULT_K = 10.0

# The default candidate grid :func:`select_k` searches over when the
# caller does not pass one. A pragmatic geometric grid (plan item 9):
# no principled default is specified by roadmap M13, so this is a
# tunable constant, not a magic number buried in the CV loop.
DEFAULT_K_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

# Clip epsilon for the probability handed to binary log loss inside
# :func:`select_k`. The posterior mean is strictly inside (0, 1) whenever
# ``alpha > 0`` and ``beta > 0``, but a team that is 100% wins with
# prior 1.0 can produce ``mean == 1.0`` exactly (and symmetrically
# ``0.0``), where log loss is ``-inf``/raises. Clipping into
# ``[eps, 1 - eps]`` keeps the score finite and consistent with
# ``utils.scoring.log_loss``'s zero-probability-raises convention.
_PROB_CLIP_EPS = 1e-12


@dataclass(frozen=True)
class OverallWinRate:
    """A team's overall (all-map) win rate over its as-of history.

    ``rate`` is ``wins / games`` when ``games > 0`` and exactly ``0.5``
    (the maximally uninformative value) when ``games == 0`` — the empty
    case is a genuine ambiguity call (plan assumption 6), documented
    here rather than silently chosen: an unseen team, or a cutoff before
    the team's first match, has no observable rate, and ``0.5`` is the
    least-committal stand-in.
    """

    wins: int
    games: int
    rate: float


@dataclass(frozen=True)
class ShrunkWinRate:
    """The Beta posterior for a team's win rate on one map.

    ``alpha`` / ``beta`` are the full posterior parameters
    ``Beta(wins + k*prior, losses + k*(1 - prior))`` — exposed, not just
    the point estimate, so callers can read off the uncertainty.
    ``mean`` is the shrinkage point estimate ``alpha / (alpha + beta)``
    (the roadmap's ``(wins + k*prior) / (games + k)``); ``variance`` is
    the Beta variance ``alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))``.
    ``wins`` / ``games`` are the team's record *on this map* (not
    overall); ``prior`` is the team's overall as-of rate fed in as the
    prior mean; ``raw_rate`` is the unshrunk map rate
    ``wins / games``, or exactly ``prior`` when ``games == 0`` (full
    shrinkage — there is no raw sample to compare against).
    """

    wins: int
    games: int
    prior: float
    raw_rate: float
    alpha: float
    beta: float
    mean: float
    variance: float


def team_overall_win_rate(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> OverallWinRate:
    """Compute a team's overall (all-map) win rate over its as-of history.

    The prior source for the shrinkage estimator. It obtains the team's
    completed, strictly-earlier maps through
    :func:`utils.asof.maps_as_of` (never by reading the Parquet tables
    directly) and derives wins via the score-comparison rule: for each
    as-of map, the team won iff its own score (picked via the
    ``team_is_team1`` orientation column) strictly exceeds the
    opponent's. A tied map raises ``ValueError``.

    Args:
        team_id: The queried team's stable id (see
            :func:`utils.asof.matches_as_of`).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.

    Returns:
        An :class:`OverallWinRate` with ``wins``/``games`` counted over
        every as-of map regardless of map name, and ``rate =
        wins / games`` — or exactly ``0.5`` when ``games == 0`` (unseen
        team, or a cutoff before the team's first completed match).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of`).
        ValueError: If an as-of map has a null/NaN score or tied scores
            (see :func:`_wins_from_oriented_maps`); or if the query date
            or a row date is null/unparseable/timezone-aware (propagated
            from :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
    """
    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)
    wins = _wins_from_oriented_maps(maps)
    games = len(maps)
    rate = wins / games if games else 0.5
    return OverallWinRate(wins=wins, games=games, rate=rate)


def team_map_win_rate(
    team_id: str,
    map_name: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k,
) -> ShrunkWinRate:
    """Compute the Beta-posterior win rate for a team on one map.

    The shrinkage estimator itself. It fetches the team's as-of maps
    once via :func:`utils.asof.maps_as_of`, computes the overall win
    rate from *all* of them as the prior, then filters to the queried
    ``map_name`` (both sides normalized through
    :func:`utils.config.normalize_map_name`, so case/whitespace never
    break a match) to get the map-specific ``wins``/``games``. The
    posterior is ``Beta(wins + k*prior, (games - wins) + k*(1 - prior))``;
    ``mean``, ``variance`` and the unshrunk ``raw_rate`` are derived from
    it. With ``games == 0`` on the map the formula degrades to
    ``mean = prior`` exactly (full shrinkage) — the correct behaviour and
    not a special case. No map-pool/era filtering happens here: a map
    name outside the caller's active pool is still a legitimate
    historical map to count (pool filtering is a caller concern, e.g.
    M18).

    Args:
        team_id: The queried team's stable id.
        map_name: The map to estimate for; normalized via
            :func:`utils.config.normalize_map_name` before matching, so
            ``"breeze"``/``" Breeze "`` both match ``"Breeze"``.
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table (needs ``map_name``,
            ``team1_score``, ``team2_score`` in addition to the columns
            ``maps_as_of`` already requires).
        k: The shrinkage strength (effective prior sample size); must be
            a positive finite real number (see :func:`_validate_k`).

    Returns:
        A :class:`ShrunkWinRate` with the map-specific ``wins``/``games``,
        the overall ``prior``, the unshrunk ``raw_rate`` (equal to
        ``prior`` when ``games == 0``), and the posterior ``alpha``,
        ``beta``, ``mean`` and ``variance``.

    Raises:
        ValueError: If ``k`` is not a positive finite real number (see
            :func:`_validate_k`); if an as-of map has a null/NaN score
            or tied scores (see :func:`_wins_from_oriented_maps`); or
            if the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            :func:`utils.asof.maps_as_of`).
        KeyError: If either table lacks a required column (propagated
            from :func:`utils.asof.maps_as_of`; includes ``map_name``,
            ``team1_score``, ``team2_score``).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
        ConfigError: If ``map_name`` or any as-of map's ``map_name``
            value is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    k_value = _validate_k(k)
    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)

    overall_wins = _wins_from_oriented_maps(maps)
    overall_games = len(maps)
    prior = overall_wins / overall_games if overall_games else 0.5

    normalized_map = config.normalize_map_name(map_name)
    map_rows = maps[maps[MAP_NAME_COL].map(config.normalize_map_name) == normalized_map]
    map_wins = _wins_from_oriented_maps(map_rows)
    map_games = len(map_rows)
    raw_rate = map_wins / map_games if map_games else prior

    alpha = map_wins + k_value * prior
    beta = (map_games - map_wins) + k_value * (1.0 - prior)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))

    return ShrunkWinRate(
        wins=map_wins,
        games=map_games,
        prior=prior,
        raw_rate=raw_rate,
        alpha=alpha,
        beta=beta,
        mean=mean,
        variance=variance,
    )


def _collect_validation_instances(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    folds: list[tuple[int, list, list]],
) -> list[tuple[str, str, str, int]]:
    """Build the scored ``(team_id, map_name, date, won)`` validation instances.

    Turns the walk-forward fold assignment into the flat list of
    hold-out map outcomes that :func:`select_k` scores. Every validation
    match's finished maps yield *two* instances — one per side — because
    each side is an independent as-of query and a genuine test of the
    shrinkage estimate (plan assumption 8). ``won`` is derived from the
    map's scores with the same rule as the estimator (1 if that side's
    score strictly exceeds the opponent's, 0 otherwise); a null/NaN
    score or a tie raises ``ValueError``.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``); its
            ``match_id`` values must be unique.
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``map_name``, ``team1_score``, ``team2_score``, ``winner``).
            Only finished maps (``winner`` non-null) contribute an
            outcome.
        folds: The ``(fold_id, train_ids, val_ids)`` tuples from
            :func:`utils.splits.walk_forward_folds`.

    Returns:
        A list of ``(team_id, map_name, date, won)`` tuples in fold
        order, two per finished validation map; ``won`` is ``1`` or
        ``0``.

    Raises:
        ValueError: If a validation ``match_id`` is absent from
            ``matches_df``, if a map has a null/NaN score, or if a map
            has tied scores.
        KeyError: If ``maps_df`` lacks a required column (propagated
            from pandas).
    """
    if not matches_df[asof.MATCH_ID_COL].is_unique:
        raise ValueError(
            "matches_df contains duplicate match_id values; the "
            "validation-instance lookup would silently collapse them"
        )
    match_by_id = {
        getattr(row, asof.MATCH_ID_COL): row
        for row in matches_df.itertuples(index=False)
    }
    finished = maps_df[maps_df[asof.WINNER_COL].notna()]

    instances: list[tuple[str, str, str, int]] = []
    for _fold_id, _train_ids, val_ids in folds:
        for mid in val_ids:
            match = match_by_id.get(mid)
            if match is None:
                raise ValueError(
                    f"validation match_id {mid!r} is absent from matches_df"
                )
            match_maps = finished[finished[asof.MATCH_ID_COL] == mid]
            for map_row in match_maps.itertuples(index=False):
                t1_score = getattr(map_row, TEAM1_SCORE_COL)
                t2_score = getattr(map_row, TEAM2_SCORE_COL)
                if pd.isna(t1_score) or pd.isna(t2_score):
                    raise ValueError(
                        f"map for match {mid!r} has a null/NaN score "
                        f"({t1_score!r} vs {t2_score!r}); a finished map "
                        "must have both scores present"
                    )
                if t1_score == t2_score:
                    raise ValueError(
                        f"map for match {mid!r} has tied scores "
                        f"({t1_score} == {t2_score}); a finished map must "
                        "have a winner"
                    )
                team1_won = 1 if t1_score > t2_score else 0
                instances.append(
                    (
                        getattr(match, asof.TEAM1_ID_COL),
                        getattr(map_row, MAP_NAME_COL),
                        getattr(match, asof.DATE_COL),
                        team1_won,
                    )
                )
                instances.append(
                    (
                        getattr(match, asof.TEAM2_ID_COL),
                        getattr(map_row, MAP_NAME_COL),
                        getattr(match, asof.DATE_COL),
                        1 - team1_won,
                    )
                )
    return instances


def select_k(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k_grid=DEFAULT_K_GRID,
    n_folds: int = DEFAULT_N_FOLDS,
    min_fold_block: int = MIN_FOLD_BLOCK_MATCHES,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple:
    """Choose the shrinkage strength ``k`` by walk-forward cross-validation.

    The CV harness for the shrinkage strength. For each candidate ``k``
    in ``k_grid`` it scores, with binary log loss, the held-out map
    outcomes of a walk-forward fold scheme over the training region
    (``split_matches`` carves out the final test slice, which is never
    scored; ``walk_forward_folds`` then yields the expanding-window
    folds over the train region). Each held-out instance is estimated
    *exactly as it would be live*: the as-of cutoff is that map's own
    match timestamp (not the fold boundary), so the estimate is built
    from a strictly independent snapshot — this is what proves CV itself
    is leakage-safe, and it means the same map/match can be validated
    multiple times across folds, each against a strictly-prior history.
    Both sides of every finished validation map count as separate
    instances (plan assumption 8). The returned ``scores_by_k`` holds
    the *mean* log loss per candidate, and ``best_k`` is the argmin
    (lower is better; ties break toward the earliest ``k`` in the grid).

    Probabilities are clipped into ``[eps, 1 - eps]`` (see
    :data:`_PROB_CLIP_EPS`) before scoring so a degenerate 0/1 posterior
    mean cannot produce an infinite log loss.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``date``, ``team1_id``, ``team2_id``,
            ``status``). Only completed matches participate in the
            split/folds (a live match has no scoreable map outcome).
        maps_df: The materialised ``maps`` table.
        k_grid: The candidate strengths to search; any iterable of
            positive finite reals (default :data:`DEFAULT_K_GRID`).
            Duplicate values collapse to one dict entry.
        n_folds: Passed to :func:`utils.splits.walk_forward_folds`.
        min_fold_block: Passed to :func:`utils.splits.walk_forward_folds`.
        test_frac: Passed to :func:`utils.splits.split_matches`.

    Returns:
        A ``(best_k, scores_by_k)`` tuple. ``best_k`` is the grid value
        with the lowest mean log loss (an element of, and key in,
        ``scores_by_k``). ``scores_by_k`` maps each grid value to its
        mean binary log loss over all validation instances.

    Raises:
        ValueError: If ``k_grid`` is empty; if a candidate ``k`` is not
            a positive finite real number (see :func:`_validate_k`); if
            the completed matches table is too small for the split/fold
            machinery (propagated from
            :func:`utils.splits.split_matches` /
            :func:`utils.splits.walk_forward_folds`); if the folds
            produce zero scoreable validation instances; if a validation
            map has a null/NaN score or tied scores or its ``match_id``
            is missing (see :func:`_collect_validation_instances`); or if
            an as-of query inside scoring fails (propagated from
            :func:`team_map_win_rate`).
        KeyError: If a table lacks a required column (propagated from
            pandas / :func:`team_map_win_rate`).
        ConfigError: If a map name is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    grid = list(k_grid)
    if not grid:
        raise ValueError("k_grid must contain at least one candidate k")
    for k in grid:
        _validate_k(k)

    completed = matches_df[
        matches_df[asof.STATUS_COL] == asof.COMPLETED_STATUS
    ].copy()

    splits_df = split_matches(completed, test_frac=test_frac)
    train_ids = set(
        splits_df.loc[splits_df["split"] == "train", asof.MATCH_ID_COL]
    )
    train_matches = completed[completed[asof.MATCH_ID_COL].isin(train_ids)]
    folds = list(
        walk_forward_folds(
            train_matches,
            n_folds=n_folds,
            min_fold_block=min_fold_block,
        )
    )

    instances = _collect_validation_instances(matches_df, maps_df, folds)
    if not instances:
        raise ValueError(
            "select_k produced zero scoreable validation instances; "
            "cannot choose k from an empty held-out set"
        )

    scores_by_k: dict = {}
    for k in grid:
        k_value = _validate_k(k)
        total = 0.0
        for team_id, map_name, date, won in instances:
            p = team_map_win_rate(
                team_id, map_name, date, matches_df, maps_df, k_value
            ).mean
            p = min(max(p, _PROB_CLIP_EPS), 1.0 - _PROB_CLIP_EPS)
            total += scoring.log_loss([1.0 - p, p], 1 if won else 0)
        scores_by_k[k] = total / len(instances)

    best_k = min(grid, key=lambda candidate: scores_by_k[candidate])
    return best_k, scores_by_k


def _two_sided_map_events(
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the league-wide per-team map event frame used by the batched path.

    Every finished map (``winner`` non-null) of every completed match
    contributes **two** event rows — one per participating team side —
    carrying that side's id, the match date (as-of key), the normalized
    ``map_name`` and a ``won`` flag derived from the raw scores
    (``team1_score > team2_score`` for the team1 side, the reverse for
    the team2 side — never the ``winner`` display string). Score
    validation (null/NaN and tie guards, the same fail-loud class as
    :func:`features._shared._wins_from_oriented_maps`) runs once over
    the whole league here rather than once per query's as-of subset —
    the documented whole-league difference of the batched path; real
    ``data/v1`` has zero such rows, so it never fires on valid data.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_id``, ``team2_id``, ``date``,
            ``status``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``winner``, ``map_name``, ``team1_score``, ``team2_score``).

    Returns:
        A ``pandas.DataFrame`` with columns ``team_id``, ``date``,
        ``match_id``, ``map_index``, ``map_name`` (normalized through
        :func:`utils.config.normalize_map_name`) and ``won`` (``0``/``1``
        ints), two rows per eligible map. Row order is ``maps_df``'s
        original order (each map's team1 row directly above its team2
        row); chronological sorting is the caller's job.

    Raises:
        KeyError: If a table lacks a required column (propagated from
            ``utils.asof.require_columns``).
        ValueError: If ``matches_df`` contains duplicate ``match_id``
            values among the completed rows (the join would fan out and
            duplicate map rows), or if any eligible map has a null/NaN
            score or tied scores (impossible for a finished map).
    """
    asof.require_columns(
        matches_df,
        (
            asof.MATCH_ID_COL,
            asof.TEAM1_ID_COL,
            asof.TEAM2_ID_COL,
            asof.DATE_COL,
            asof.STATUS_COL,
        ),
        "matches_df",
    )
    asof.require_columns(
        maps_df,
        (
            asof.MATCH_ID_COL,
            asof.WINNER_COL,
            MAP_NAME_COL,
            TEAM1_SCORE_COL,
            TEAM2_SCORE_COL,
        ),
        "maps_df",
    )

    completed = matches_df[matches_df[asof.STATUS_COL] == asof.COMPLETED_STATUS]
    if not completed[asof.MATCH_ID_COL].is_unique:
        raise ValueError(
            "matches_df contains duplicate match_id values among the "
            "completed rows; the map-event join would fan out and "
            "duplicate map rows"
        )
    finished = maps_df[maps_df[asof.WINNER_COL].notna()]
    merged = finished.merge(
        completed[
            [
                asof.MATCH_ID_COL,
                asof.TEAM1_ID_COL,
                asof.TEAM2_ID_COL,
                asof.DATE_COL,
            ]
        ],
        on=asof.MATCH_ID_COL,
        how="inner",
    )

    team1_scores = merged[TEAM1_SCORE_COL].to_numpy()
    team2_scores = merged[TEAM2_SCORE_COL].to_numpy()
    null_mask = pd.isna(team1_scores) | pd.isna(team2_scores)
    if null_mask.any():
        offending = merged.loc[null_mask, asof.MATCH_ID_COL].tolist()
        raise ValueError(
            f"{len(offending)} league map(s) have a null/NaN score "
            "(team1_score or team2_score is missing), which is impossible "
            f"for a finished map; offending match_id(s): {offending[:5]}"
        )
    tie_mask = team1_scores == team2_scores
    if tie_mask.any():
        offending = merged.loc[tie_mask, asof.MATCH_ID_COL].tolist()
        raise ValueError(
            f"{len(offending)} league map(s) have tied scores "
            "(team1_score == team2_score), which is impossible for a "
            f"finished map; offending match_id(s): {offending[:5]}"
        )

    normalized_names = merged[MAP_NAME_COL].map(config.normalize_map_name)
    side1 = pd.DataFrame(
        {
            "team_id": merged[asof.TEAM1_ID_COL].to_numpy(),
            "date": merged[asof.DATE_COL].to_numpy(),
            "match_id": merged[asof.MATCH_ID_COL].to_numpy(),
            "map_index": merged["map_index"].to_numpy(),
            "map_name": normalized_names.to_numpy(),
            "won": (team1_scores > team2_scores).astype(int),
        }
    )
    side2 = pd.DataFrame(
        {
            "team_id": merged[asof.TEAM2_ID_COL].to_numpy(),
            "date": merged[asof.DATE_COL].to_numpy(),
            "match_id": merged[asof.MATCH_ID_COL].to_numpy(),
            "map_index": merged["map_index"].to_numpy(),
            "map_name": normalized_names.to_numpy(),
            "won": (team2_scores > team1_scores).astype(int),
        }
    )
    return pd.concat([side1, side2], ignore_index=True)


def batched_map_win_rate_diff(
    rows_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k=DEFAULT_K,
) -> np.ndarray:
    """Return per-row shrunk per-map win-rate differentials (A minus B).

    The batched sibling of the two :func:`team_map_win_rate` calls
    :func:`models._shared.build_feature_vector` makes per row (task
    052): builds the league-wide per-team map event frame once via
    :func:`_two_sided_map_events`, records per-key running totals over
    the chronologically-sorted events (per-team overall wins/games for
    the prior, and per-``(team, map_name)`` wins/games for the
    numerator — both via ``groupby(...).cumcount()``/``cumsum()``), and
    resolves each row's two sides against those static ledgers with
    ``utils.asof.merge_asof_lookup`` — one team1 lookup and one team2
    lookup per ledger, so every row's count is the count of that side's
    events dated strictly before the row's own date (the strict-``<``
    boundary enforced per query by the as-of primitive, never by
    trusting a pre-filtered view). The Beta formula is then applied
    vectorized with the exact same arithmetic and operand order as the
    single-row estimator (``prior = wins/games`` when ``games > 0``
    else ``0.5``; ``alpha = map_wins + k*prior``; ``beta =
    (map_games - map_wins) + k*(1 - prior)``; ``mean =
    alpha/(alpha+beta)``), so a zero-history side degrades to the full
    shrinkage value exactly as the single-row path does, and the output
    is bit-for-bit identical to looping the single-row calls.

    Args:
        rows_df: The row table; needs ``team1_id``, ``team2_id``,
            ``map_name`` and ``date`` columns. Row order is preserved
            in the output.
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        k: The shrinkage strength (effective prior sample size); must be
            a positive finite real number (see :func:`_validate_k`).

    Returns:
        A ``(n,)`` numpy array of ``float``:
        ``team_map_win_rate(team1, map, date, ...).mean`` minus
        ``team_map_win_rate(team2, map, date, ...).mean`` per row, where
        a side with zero prior maps on the queried map degrades to its
        own overall-rate prior exactly as the single-row estimator does.

    Raises:
        KeyError: If a table lacks a required column (propagated from
            the event builder / as-of helpers).
        ValueError: If ``k`` is invalid (see :func:`_validate_k`), if
            any league map has a null/NaN score or tied scores (see
            :func:`_two_sided_map_events`), or if a row date is
            null/unparseable (propagated from
            ``utils.asof.parse_date_column``).
        ConfigError: If a map name is not a string (propagated from
            :func:`utils.config.normalize_map_name`).
    """
    k_value = _validate_k(k)
    events = _two_sided_map_events(matches_df, maps_df)

    parsed = asof.parse_date_column(events[asof.DATE_COL])
    events = (
        events.assign(_parsed_date=parsed)
        .sort_values(
            ["_parsed_date", asof.MATCH_ID_COL, "map_index", "team_id"],
            kind="stable",
        )
        .drop(columns=["_parsed_date"])
    )
    events = events.reset_index(drop=True)

    events["games_after"] = events.groupby("team_id").cumcount() + 1
    events["wins_after"] = events.groupby("team_id")["won"].cumsum()
    events["map_games_after"] = (
        events.groupby(["team_id", "map_name"]).cumcount() + 1
    )
    events["map_wins_after"] = (
        events.groupby(["team_id", "map_name"])["won"].cumsum()
    )

    def _side_result(team_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve one side's overall + per-map running totals for every row.

        Args:
            team_col: The row-table team id column to resolve
                (``"team1_id"`` or ``"team2_id"``).

        Returns:
            A ``(games_overall, wins_overall, map_games, map_wins)``
            tuple of float numpy arrays aligned to ``rows_df`` (``NaN``
            where the side has no strictly-prior event for that ledger).

        Raises:
            Nothing (validation propagated from the lookups).
        """
        queries_overall = rows_df[[team_col, asof.DATE_COL]].reset_index(drop=True)
        overall = asof.merge_asof_lookup(
            events,
            ("team_id",),
            asof.DATE_COL,
            ("games_after", "wins_after"),
            queries_overall,
            (team_col,),
            asof.DATE_COL,
        )
        query_map = rows_df[[team_col, MAP_NAME_COL, asof.DATE_COL]].copy()
        query_map[MAP_NAME_COL] = query_map[MAP_NAME_COL].map(
            config.normalize_map_name
        )
        query_map = query_map.reset_index(drop=True)
        per_map = asof.merge_asof_lookup(
            events,
            ("team_id", "map_name"),
            asof.DATE_COL,
            ("map_games_after", "map_wins_after"),
            query_map,
            (team_col, MAP_NAME_COL),
            asof.DATE_COL,
        )
        return (
            overall["games_after"].to_numpy(dtype=float),
            overall["wins_after"].to_numpy(dtype=float),
            per_map["map_games_after"].to_numpy(dtype=float),
            per_map["map_wins_after"].to_numpy(dtype=float),
        )

    games_a, wins_a, map_games_a, map_wins_a = _side_result(asof.TEAM1_ID_COL)
    games_b, wins_b, map_games_b, map_wins_b = _side_result(asof.TEAM2_ID_COL)

    def _shrunk_mean(games_o, wins_o, map_games, map_wins):
        """Apply the Beta formula vectorized for one side.

        Args:
            games_o: Overall as-of games (float array, NaN = none).
            wins_o: Overall as-of wins (float array, NaN = none).
            map_games: Per-map as-of games (float array, NaN = none).
            map_wins: Per-map as-of wins (float array, NaN = none).

        Returns:
            The per-row posterior mean array.

        Raises:
            Nothing.
        """
        games_o = np.where(np.isnan(games_o), 0.0, games_o)
        wins_o = np.where(np.isnan(wins_o), 0.0, wins_o)
        map_games = np.where(np.isnan(map_games), 0.0, map_games)
        map_wins = np.where(np.isnan(map_wins), 0.0, map_wins)
        prior = np.full(len(games_o), 0.5)
        has_games = games_o > 0.0
        prior[has_games] = wins_o[has_games] / games_o[has_games]
        alpha = map_wins + k_value * prior
        beta = (map_games - map_wins) + k_value * (1.0 - prior)
        return alpha / (alpha + beta)

    return (_shrunk_mean(games_a, wins_a, map_games_a, map_wins_a) -
            _shrunk_mean(games_b, wins_b, map_games_b, map_wins_b))
