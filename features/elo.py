"""Sequential Elo ratings and team-parity differentials (roadmap M14).

A pure in-memory library that replays every finished map in
chronological order and produces, for any as-of cutoff date, each
team's Elo rating as well as the signed and absolute rating
differential between two teams (the parity feature the overtime model,
M15/M20, needs). Like the rest of ``features/``, this module has no CLI,
no ``argparse`` entry point, and no file I/O of its own — it operates
on the already-materialised ``matches_df``/``maps_df`` DataFrames a
caller passes in.

**Leakage contract (the hard requirement).** Elo is sequential:
``rating(team, t)`` depends on every prior map's outcome, not a fixed
window. An as-of query therefore replays the *entire league's* finished
maps in true global chronological order, filtered to rows dated
**strictly before** the query date — reusing ``utils.asof``'s own date
parsing and strict-``<`` boundary conventions so this module's cutoff
semantics are byte-for-byte identical to every other as-of consumer,
not a second reimplementation that could drift. The boundary is
enforced by construction (filter-then-replay), not by trusting the
replay loop to stop itself. A map dated equal to or after the query
date is never replayed, for any team, so no rating used in a query can
have seen its own or any later outcome.

**Approach: full deterministic replay, not an incremental cache.** Each
query is O(all history) rather than O(1): the filter below is computed
from scratch, then the whole surviving sequence is folded through the
Elo update rule. This is a deliberate, documented performance tradeoff
(acceptable at v1's 244-map scale, matching ``utils.asof``'s own
"correctness first" stance). A running-state snapshot table
(one row per team per map, rating *after* that map, queried by "last
snapshot strictly before the cutoff") is a valid equivalent design and
a good future performance follow-up, but it is *not* built here to
avoid a subtle staleness/off-by-one bug in a cache-invalidation path
the size budget does not justify.

**Resolved design decisions (do not re-derive in later milestones):**

- **Initial rating 1500** and **fixed K-factor 32** are conventional
  Elo defaults chosen in the absence of roadmap guidance; they are
  isolated as module constants (:data:`INITIAL_RATING`,
  :data:`DEFAULT_K`), not magic numbers, so a later change is a
  one-line diff.
- **No K-factor cross-validation in M14.** Unlike M13's ``select_k``,
  roadmap M14 asks for a rating and a parity feature, not a tuned
  update rule. A future milestone (or a revision of M20) may add a
  ``select_k`` analogue that scores candidate K values by the log loss
  of :func:`_expected_score` against held-out map outcomes via
  walk-forward CV — recorded here so it is a known gap, not a silent
  omission.
- **Global Elo, not per-map-name.** Roadmap says "a sequential rating",
  singular, and does not mention per-map splitting (contrast M13, which
  is explicitly per-map-name). One rating per team, updated by every
  map regardless of which map it was played on. A "per-map-name Elo"
  variant is a candidate future extension, explicitly out of scope here.
- **Per-map updates, not per-match.** A Bo3/Bo5 match contributes one
  update per finished map, each treating that individual map as a fresh
  binary outcome with the *pre-update* ratings of both sides feeding
  the expected score.
- **No home-field/first-pick advantage term.** VCT has no venue home
  team; the expected score uses the raw rating difference only.
- **Win/loss from scores, never the ``winner`` column.** As M13
  established, ``maps.parquet["winner"]`` is a display team-name string
  (e.g. ``"FNATIC"``), not a stable ``"team1"``/``"team2"`` marker.
  Team 1 won a map iff ``team1_score > team2_score``; team 2 won
  otherwise. A tied map raises ``ValueError`` (a finished map must have
  a winner; a draw score is never a live code path and is not silently
  defined). A finished map with a null/NaN ``team1_score`` or
  ``team2_score`` also raises ``ValueError`` *before* the tie check:
  NaN compares neither equal nor greater to anything (including
  itself), so an unguarded comparison would silently record team 2 as
  the winner — the guard makes that fail loudly instead.
- **Same-date tie-break is ``(date, match_id, map_index)``.** ``date``
  and ``match_id`` match ``utils.splits._chronological_order``'s
  existing convention; ``map_index`` (0-indexed per match in M8's
  ``maps.parquet``) is the necessary third key because multiple maps
  share one match timestamp and must replay in the order they were
  actually played. This total order is asserted deterministic via a
  repeated-run test.

**Data-shape findings (re-derived against real ``data/v1``, item 1):**

- ``maps.parquet`` carries ``map_index``, 0-indexed per ``match_id``
  (confirmed: ``[0,1,2]``, ``[0,1]`` sequences per match).
- ``utils.asof.maps_as_of`` retains ``map_index`` in its output (it is
  an inner merge of ``maps_df`` against a small join frame, so every
  original ``maps_df`` column survives). Verified empirically for
  ``data/v1``.
- No map row has ``team1_score == team2_score`` (0 ties in v1); the
  tie guard is defensive fail-loudly coverage, not a live-data fix.
- No map row has a non-null ``winner`` with a null ``team1_score`` or
  ``team2_score`` (0 such rows in v1); the null-score guard added in
  task 017 is defensive fail-loudly coverage, not a live-data fix.

**Module docstring note on the league-wide filter (item 2).**
``utils.asof``'s filters are single-team by design, so this module
writes its own small league-wide filter (:func:`_league_maps_as_of`)
that mirrors ``matches_as_of``'s completed + strictly-before boolean
masks *without* the team-id mask, then inner-joins to ``maps_df`` with
the same finished-map (``winner.notna()``) filter ``maps_as_of``
applies. The duplication is ~3 lines of boolean-mask logic and is a
conscious choice: bending the existing single-team API into a
multi-team shape it was not designed for would be more fragile than a
two-mask duplicate. A candidate ``matches_as_of_all``/
``maps_as_of_all`` (all-teams) extension of ``utils.asof`` is flagged
for a later milestone if a third consumer appears.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from utils import asof

# Elo constants, isolated so a later change is a one-line diff.
INITIAL_RATING = 1500.0
DEFAULT_K = 32.0

# Column names this module reads from the M8 maps table. ``team1_id``,
# ``team2_id``, ``date`` and ``status`` come from the matches table via
# ``utils.asof``'s constants; the score columns and ``map_index`` are
# M8's maps-table columns, named here once so functions and tests share
# one spelling.
MAP_INDEX_COL = "map_index"
TEAM1_SCORE_COL = "team1_score"
TEAM2_SCORE_COL = "team2_score"

# The columns :func:`_league_maps_as_of` needs on each table.
_MATCHES_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.TEAM1_ID_COL,
    asof.TEAM2_ID_COL,
    asof.DATE_COL,
    asof.STATUS_COL,
)
_MAPS_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.WINNER_COL,
    MAP_INDEX_COL,
    TEAM1_SCORE_COL,
    TEAM2_SCORE_COL,
)


@dataclass(frozen=True)
class EloDifferential:
    """The signed and absolute Elo differential between two teams.

    The single object a caller (M15's team-parity input) consumes so the
    "expose both the signed differential and its absolute value"
    requirement is a literal field pair computed from *one* shared
    replay — never two independent calls a caller could accidentally
    run at different as-of dates.

    Attributes:
        team_a_id: The first queried team's stable id, echoed unchanged.
        team_b_id: The second queried team's stable id, echoed unchanged.
        date: The as-of cutoff, exactly as passed in (original string).
        rating_a: ``team_a_id``'s rating after the replay.
        rating_b: ``team_b_id``'s rating after the replay.
        differential: The signed difference ``rating_a - rating_b``.
        abs_differential: ``abs(differential)``.
    """

    team_a_id: str
    team_b_id: str
    date: str
    rating_a: float
    rating_b: float
    differential: float
    abs_differential: float


def _validate_k(k) -> float:
    """Validate the K-factor and return it as a float.

    ``k`` is the per-map rating-change scale factor. It must be a
    positive finite real number: ``k <= 0`` would make a winner's rating
    fall and a loser's rise (inverting the update), and non-finite ``k``
    would poison every rating it touched.

    Args:
        k: The K-factor. Any real number (``int``/``float``/numpy
            scalar) is coerced to ``float``.

    Returns:
        ``k`` as a ``float``.

    Raises:
        ValueError: If ``k`` cannot be coerced to a ``float``, or if the
            result is NaN, infinite, or ``<= 0``.
    """
    try:
        value = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"k must be a positive real number, got {k!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"k must be a positive finite real number, got {k!r}")
    return value


def _validate_initial_rating(initial_rating) -> float:
    """Validate the initial rating and return it as a float.

    The starting rating every team carries before its first as-of map.
    It must be a finite real number (any finite value is a legal Elo
    origin; NaN/inf would poison the replay).

    Args:
        initial_rating: The starting rating. Any real number
            (``int``/``float``/numpy scalar) is coerced to ``float``.

    Returns:
        ``initial_rating`` as a ``float``.

    Raises:
        ValueError: If it cannot be coerced to a ``float``, or if the
            result is NaN or infinite.
    """
    try:
        value = float(initial_rating)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"initial_rating must be a finite real number, got {initial_rating!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"initial_rating must be a finite real number, got {initial_rating!r}"
        )
    return value


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Return team A's expected score against team B (standard Elo logistic).

    The classic Elo expectation ``1 / (1 + 10^((R_b - R_a) / 400))``.
    Team B's expectation is always ``1 - E_a`` (the two sum to 1.0),
    which the update rule relies on for its zero-sum property.

    Args:
        rating_a: Team A's pre-update rating.
        rating_b: Team B's pre-update rating.

    Returns:
        Team A's expected score as a ``float`` in ``(0, 1)``.

    Raises:
        Nothing (the formula is total for any finite input, including
            extreme rating gaps, which saturate toward 1/0 but never
            reach them).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_pair(
    rating_a: float,
    rating_b: float,
    team_a_won: bool,
    k: float,
) -> tuple[float, float]:
    """Apply one map's Elo update to a pair of ratings.

    The single-map update rule: ``R_a' = R_a + K*(S_a - E_a)`` and
    ``R_b' = R_b + K*(S_b - E_b)`` with ``S_a = 1`` if team A won else
    ``0``, ``S_b = 1 - S_a``, and ``E_a``/``E_b`` from
    :func:`_expected_score` (``E_b = 1 - E_a``). Because ``S_a + S_b ==
    1`` and ``E_a + E_b == 1``, the pair is exactly zero-sum:
    ``(R_a' - R_a) == -(R_b' - R_b)``.

    A tied map is **not expressible** here: ``team_a_won`` is a strict
    ``bool``, so "neither team won" has no representation. Ties are
    rejected upstream in :func:`_replay_ratings_as_of`, which reads the
    two scores and raises ``ValueError`` before calling this function.

    Args:
        rating_a: Team A's pre-update rating.
        rating_b: Team B's pre-update rating.
        team_a_won: ``True`` if team A won this map, ``False`` if team
            B won.
        k: The K-factor (already validated positive finite real; see
            :func:`_validate_k`).

    Returns:
        A ``(new_rating_a, new_rating_b)`` tuple of the two post-update
        ratings, in that order.

    Raises:
        Nothing (a strict bool has only two states and both are valid;
            the K-factor is expected pre-validated).
    """
    expected_a = _expected_score(rating_a, rating_b)
    score_a = 1.0 if team_a_won else 0.0
    score_b = 1.0 - score_a
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


def _league_maps_as_of(
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return every finished map of every team strictly before a cutoff.

    The league-wide as-of filter item 2 calls for. It mirrors
    ``utils.asof.matches_as_of``'s three boolean masks (completed,
    strictly-before) *without* the per-team mask — because Elo needs the
    **global** map sequence (every team's maps, not just one queried
    team's) so opponent-side ratings are tracked correctly — then
    inner-joins to ``maps_df`` with the same finished-map
    (``winner.notna()``) filter ``utils.asof.maps_as_of`` applies.

    Date parsing reuses ``utils.asof``'s public parse helpers
    (``parse_query_date`` / ``parse_date_column``) rather than
    duplicating them: both modules are pure in-memory libraries and the
    dependency is ``features`` -> ``utils`` (feature modules may depend
    downward on genuine ``utils/`` utilities, not a driver import), and
    reusing the same parse functions guarantees the strict-``<``
    boundary, null-date rejection and timezone-naive-only rules are
    byte-for-byte identical to every
    other as-of consumer. The private access is a deliberate, documented
    choice over a reimplementation.

    Args:
        date: The as-of cutoff; rows dated equal to or after this are
            excluded (strict ``<``).
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_id``, ``team2_id``, ``date``,
            ``status``).
        maps_df: The materialised ``maps`` table (needs ``match_id``,
            ``winner``, ``map_index``, ``team1_score``, ``team2_score``).

    Returns:
        A ``pandas.DataFrame`` with columns ``match_id``, ``map_index``,
        ``team1_score``, ``team2_score``, ``team1_id``, ``team2_id`` and
        ``date`` — exactly the rows whose parent match is completed and
        strictly before the cutoff *and* whose own ``winner`` is
        non-null (a finished map). The output is unsorted (ordering is
        :func:`_replay_ratings_as_of`'s job) and preserves no input
        index (the merge resets it).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from ``utils.asof.require_columns``).
        ValueError: If the filtered ``matches`` frame contains duplicate
            ``match_id`` values (the join would fan out and duplicate
            map rows); or if the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            ``utils.asof.parse_query_date`` /
            ``utils.asof.cached_parsed_date_column``).
        TypeError: If the query date is list-like (propagated from
            ``utils.asof.parse_query_date``).
    """
    asof.require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")
    asof.require_columns(maps_df, _MAPS_REQUIRED, "maps_df")

    parsed_dates = asof.cached_parsed_date_column(matches_df)
    query = asof.parse_query_date(date)

    is_completed = matches_df[asof.STATUS_COL] == asof.COMPLETED_STATUS
    is_before = parsed_dates < query
    matches = matches_df[is_completed & is_before]

    if not matches[asof.MATCH_ID_COL].is_unique:
        duplicates = (
            matches.loc[
                matches[asof.MATCH_ID_COL].duplicated(keep=False),
                asof.MATCH_ID_COL,
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "matches_df contains duplicate match_id value(s) "
            f"{duplicates} after as-of filtering; the maps join would "
            "fan out and duplicate map rows"
        )

    finished_maps = maps_df[maps_df[asof.WINNER_COL].notna()][
        [asof.MATCH_ID_COL, MAP_INDEX_COL, TEAM1_SCORE_COL, TEAM2_SCORE_COL]
    ]
    join_frame = matches[
        [asof.MATCH_ID_COL, asof.TEAM1_ID_COL, asof.TEAM2_ID_COL, asof.DATE_COL]
    ]
    return finished_maps.merge(join_frame, on=asof.MATCH_ID_COL, how="inner")


def _replay_ratings_as_of(
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k: float = DEFAULT_K,
    initial_rating: float = INITIAL_RATING,
) -> dict[str, float]:
    """Replay the league's as-of maps and return the post-replay ratings.

    The single shared replay implementation behind both public queries.
    It obtains the global as-of map sequence via
    :func:`_league_maps_as_of` (so the strict-``<`` boundary is applied
    exactly once, by that filter, before any update runs), sorts it by
    ``(date, match_id, map_index)``, and folds :func:`_update_pair` over
    every row in that order, starting each team at ``initial_rating`` on
    its first appearance. A team that never appears is simply absent
    from the returned dict (callers default it to ``initial_rating``;
    this function does not invent a zero-history entry).

    Args:
        date: The as-of cutoff; maps dated equal to or after this are
            never replayed (strict ``<``, enforced by
            :func:`_league_maps_as_of`).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        k: The K-factor; must be a positive finite real (see
            :func:`_validate_k`).
        initial_rating: The starting rating; must be a finite real (see
            :func:`_validate_initial_rating`).

    Returns:
        A ``dict`` mapping each ``team_id`` that appears in at least one
        as-of map to its rating after the full replay (a ``float``).
        A team with no as-of maps is absent from the dict.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`_league_maps_as_of`).
        ValueError: If an as-of map has a null/NaN score
            (``team1_score`` or ``team2_score`` is missing, checked
            before the tie comparison) or tied scores
            (``team1_score == team2_score``); if ``k`` or
            ``initial_rating`` is invalid (see :func:`_validate_k` /
            :func:`_validate_initial_rating`); or if the query date or a
            row date is null/unparseable/timezone-aware (propagated from
            :func:`_league_maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`_league_maps_as_of`).
    """
    k_value = _validate_k(k)
    initial = _validate_initial_rating(initial_rating)

    maps = _league_maps_as_of(date, matches_df, maps_df)

    # Dates were already validated non-null/parseable by
    # _league_maps_as_of, so this re-parse cannot raise a null-date
    # error; it exists purely to establish the sort key.
    sort_keys = pd.DataFrame(
        {
            "date": pd.to_datetime(maps[asof.DATE_COL]).to_numpy(),
            "match_id": maps[asof.MATCH_ID_COL].to_numpy(),
            "map_index": maps[MAP_INDEX_COL].to_numpy(),
        }
    )
    order = sort_keys.sort_values(
        ["date", "match_id", "map_index"], kind="stable"
    ).index.to_numpy()

    ratings: dict[str, float] = {}
    for position in order:
        team1_id = maps[asof.TEAM1_ID_COL].iloc[position]
        team2_id = maps[asof.TEAM2_ID_COL].iloc[position]
        rating_a = ratings.get(team1_id, initial)
        rating_b = ratings.get(team2_id, initial)

        team1_score = maps[TEAM1_SCORE_COL].iloc[position]
        team2_score = maps[TEAM2_SCORE_COL].iloc[position]
        if pd.isna(team1_score) or pd.isna(team2_score):
            raise ValueError(
                f"map for match {maps[asof.MATCH_ID_COL].iloc[position]!r} has "
                f"a null/NaN score ({team1_score!r} vs {team2_score!r}); a "
                "finished map must have both scores present"
            )
        if team1_score == team2_score:
            raise ValueError(
                f"map for match {maps[asof.MATCH_ID_COL].iloc[position]!r} has "
                f"tied scores ({team1_score} == {team2_score}); a finished "
                "map must have a winner"
            )

        team1_won = team1_score > team2_score
        new_a, new_b = _update_pair(rating_a, rating_b, team1_won, k_value)
        ratings[team1_id] = new_a
        ratings[team2_id] = new_b

    return ratings


def elo_rating(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k: float = DEFAULT_K,
    initial_rating: float = INITIAL_RATING,
) -> float:
    """Return one team's Elo rating as of a cutoff date.

    Replays the full league history strictly before ``date`` (via
    :func:`_replay_ratings_as_of`) and returns the queried team's rating
    after the replay. A team with zero as-of maps (unseen team, or a
    cutoff before its first completed match) is not an error: it returns
    exactly ``initial_rating``, matching ``utils.asof``'s "unknown team
    is empty, not an error" convention.

    Args:
        team_id: The queried team's stable id (a string matching the
            dtype of ``team1_id``/``team2_id``).
        date: The as-of cutoff; maps dated equal to or after this are
            excluded (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        k: The K-factor (see :func:`_validate_k`).
        initial_rating: The starting rating (see
            :func:`_validate_initial_rating`); also the exact value
            returned for a team with no as-of maps.

    Returns:
        The team's rating as a ``float`` after replaying every as-of
        map; ``initial_rating`` (coerced to ``float``) if the team has
        no as-of maps.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`_replay_ratings_as_of`).
        ValueError: If ``k`` or ``initial_rating`` is invalid; if an
            as-of map has a null/NaN score or tied scores; or if the
            query date or a row date is null/unparseable/timezone-aware
            (all propagated from :func:`_replay_ratings_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`_replay_ratings_as_of`).
    """
    initial = _validate_initial_rating(initial_rating)
    ratings = _replay_ratings_as_of(date, matches_df, maps_df, k, initial)
    return ratings.get(team_id, initial)


def elo_differential(
    team_a_id: str,
    team_b_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    k: float = DEFAULT_K,
    initial_rating: float = INITIAL_RATING,
) -> EloDifferential:
    """Return the signed and absolute Elo differential between two teams.

    Computes both teams' ratings from **one shared replay** (a single
    :func:`_replay_ratings_as_of` call — not two independent
    ``elo_rating`` calls, which would replay the whole league twice) and
    packages them, plus the signed ``rating_a - rating_b`` and its
    absolute value, into an :class:`EloDifferential`. This is the
    primary consumer-facing function for M15's team-parity input.

    Args:
        team_a_id: The first team's stable id.
        team_b_id: The second team's stable id.
        date: The as-of cutoff; maps dated equal to or after this are
            excluded (strict ``<``).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.
        k: The K-factor (see :func:`_validate_k`).
        initial_rating: The starting rating (see
            :func:`_validate_initial_rating`); also the default for a
            side with no as-of maps.

    Returns:
        An :class:`EloDifferential` holding both ratings, the signed
        ``differential = rating_a - rating_b`` and
        ``abs_differential = abs(differential)``. Two unseen teams yield
        ``differential == 0.0`` and ``abs_differential == 0.0``.

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`_replay_ratings_as_of`).
        ValueError: If ``k`` or ``initial_rating`` is invalid; if an
            as-of map has a null/NaN score or tied scores; or if the
            query date or a row date is null/unparseable/timezone-aware
            (all propagated from :func:`_replay_ratings_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`_replay_ratings_as_of`).
    """
    initial = _validate_initial_rating(initial_rating)
    ratings = _replay_ratings_as_of(date, matches_df, maps_df, k, initial)

    rating_a = ratings.get(team_a_id, initial)
    rating_b = ratings.get(team_b_id, initial)
    differential = rating_a - rating_b

    return EloDifferential(
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        date=date,
        rating_a=rating_a,
        rating_b=rating_b,
        differential=differential,
        abs_differential=abs(differential),
    )
