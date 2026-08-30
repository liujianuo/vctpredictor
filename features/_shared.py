"""Shared helper logic for the ``features/`` package (not a genuine utility).

This module is the single home for helper logic genuinely shared by more
than one ``features/`` module, so no feature module ever imports a
private helper from a sibling feature module. It deliberately lives
under ``features/`` (not ``utils/``): it is feature-support code, not a
leaf-level utility, and placing it in ``utils/`` would reopen the exact
lateral-dependency problem this package layout closes.

It holds the Beta-shrinkage and score-orientation helpers lifted from
``features.map_win_rate`` (``_validate_k``, ``_wins_from_oriented_maps``
and the score-column constants) and the per-match name-resolution/roster
helpers lifted from ``features.player_form`` (``_build_match_name_lookup``,
``_chronological_maps``, ``_validated_roster`` and the name-column
constants). Each is a private name here (leading underscore) so it is
not promoted to public API, but the module itself is importable from any
``features/`` module. Like the rest of ``features/`` it has no CLI and
no file I/O of its own; history still flows exclusively through
``utils.asof``.
"""

from __future__ import annotations

import math

import pandas as pd

from utils import asof

# Column names shared by the score-orientation helpers. ``team_is_team1``
# comes from utils.asof's maps output; the score columns are M8's
# maps-table columns, named here once so features and tests share one
# spelling.
TEAM1_SCORE_COL = "team1_score"
TEAM2_SCORE_COL = "team2_score"

# Column names shared by the per-match name-resolution helpers. The
# ``team_name`` column is a player_map_stats-table column; the two side
# names are matches-table columns, named here once so features and tests
# share one spelling.
TEAM1_NAME_COL = "team1_name"
TEAM2_NAME_COL = "team2_name"
TEAM_NAME_COL = "team_name"


def _validate_k(k) -> float:
    """Validate the shrinkage strength ``k`` and return it as a float.

    ``k`` is the effective prior sample size: ``alpha0 = k*prior`` and
    ``beta0 = k*(1 - prior)``, so ``alpha0 + beta0 = k``. It must be a
    positive finite real number. ``k <= 0`` is rejected rather than
    tolerated: ``k == 0`` would drop the prior term entirely (the
    estimator degenerates to the raw rate with no graceful fallback),
    and negative ``k`` would produce negative pseudo-counts.

    Args:
        k: The shrinkage strength. Any real number (``int``/``float``/
            numpy scalar) is coerced to ``float``.

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


def _wins_from_oriented_maps(maps: pd.DataFrame) -> int:
    """Count wins for the queried team across an as-of-oriented maps frame.

    The input is the output of ``utils.asof.maps_as_of`` (or a subset of
    it), which already carries the ``team_is_team1`` orientation column.
    For each row the queried team's score is ``team1_score`` when
    ``team_is_team1`` is truthy, else ``team2_score``; the opponent's is
    the other column. A win is the queried team's score strictly
    exceeding the opponent's. A null/NaN score raises ``ValueError``
    before the tie check (NaN compares neither equal nor greater to
    anything, so an unguarded comparison would silently count the row
    as a loss); a tie also raises ``ValueError`` (fail loudly, matching
    the ``drivers/labels.py`` convention) rather than being silently
    counted as a loss.

    Args:
        maps: The as-of-filtered maps DataFrame; needs at least
            ``team1_score``, ``team2_score`` and
            :data:`asof.TEAM_ORIENTATION_COL`. An empty frame is valid
            and yields 0 wins.

    Returns:
        The number of wins as an ``int``.

    Raises:
        ValueError: If any row has a null/NaN ``team1_score`` or
            ``team2_score`` (an impossible finished map), or if any row
            has ``team1_score == team2_score`` (an impossible finished
            map); the message lists the offending ``match_id`` values.
        KeyError: If a required column is missing (propagated from
            pandas).
    """
    is_team1 = maps[asof.TEAM_ORIENTATION_COL].astype(bool)
    our = maps[TEAM1_SCORE_COL].where(is_team1, maps[TEAM2_SCORE_COL])
    their = maps[TEAM2_SCORE_COL].where(is_team1, maps[TEAM1_SCORE_COL])
    null_mask = our.isna() | their.isna()
    if null_mask.any():
        offending = maps.loc[null_mask, asof.MATCH_ID_COL].tolist()
        raise ValueError(
            f"{len(offending)} as-of map(s) have a null/NaN score "
            "(team1_score or team2_score is missing), which is "
            "impossible for a finished map; offending match_id(s): "
            f"{offending[:5]}"
        )
    tie_mask = our == their
    if tie_mask.any():
        offending = maps.loc[tie_mask, asof.MATCH_ID_COL].tolist()
        raise ValueError(
            f"{len(offending)} as-of map(s) have tied scores "
            "(team1_score == team2_score), which is impossible for a "
            f"finished map; offending match_id(s): {offending[:5]}"
        )
    return int((our > their).sum())


def _build_match_name_lookup(matches_df: pd.DataFrame) -> dict:
    """Build a ``match_id -> (team1_name, team2_name)`` lookup.

    The per-match name-resolution source. ``maps_as_of``'s output carries
    ``match_id`` and ``team_is_team1`` but not the display names, so this
    lookup lets each as-of map row resolve the queried team's display
    name from the same match row that already carries its orientation.
    It also validates the two extra ``matches_df`` columns this module
    needs (``team1_name``/``team2_name``) beyond what ``utils.asof``
    itself requires, and rejects the same-team-name collision case
    (``team1_name == team2_name``) before any roster filtering can run.

    Args:
        matches_df: The materialised ``matches`` table (needs
            ``match_id``, ``team1_name``, ``team2_name``). Duplicate
            ``match_id`` values are harmless here (later rows overwrite
            earlier ones in the dict) because ``maps_as_of`` already
            rejects duplicates within the as-of-filtered subset, and a
            single match's two side names are constant across its rows.

    Returns:
        A ``dict`` mapping each ``match_id`` to a
        ``(team1_name, team2_name)`` tuple.

    Raises:
        KeyError: If ``matches_df`` lacks ``team1_name`` or
            ``team2_name`` (propagated from
            :func:`utils.asof.require_columns`).
        ValueError: If any match row has
            ``team1_name == team2_name`` (a side-name collision). The
            message names the ``match_id`` and the colliding name. This
            is the fail-loud guard that keeps opponent rows from being
            silently merged into the queried team's roster: with the two
            names identical, the ``team_name == resolved_name`` filter
            in :func:`_validated_roster` cannot distinguish the two
            sides.
    """
    asof.require_columns(matches_df, (TEAM1_NAME_COL, TEAM2_NAME_COL), "matches_df")
    lookup: dict = {}
    for row in matches_df.itertuples(index=False):
        match_id = getattr(row, asof.MATCH_ID_COL)
        team1_name = getattr(row, TEAM1_NAME_COL)
        team2_name = getattr(row, TEAM2_NAME_COL)
        if team1_name == team2_name:
            raise ValueError(
                f"matches_df match {match_id!r} has team1_name == team2_name "
                f"== {team1_name!r}; the two side names must be distinct, "
                "otherwise opponent rows cannot be distinguished from the "
                "queried team's roster and would be silently merged into it"
            )
        lookup[match_id] = (team1_name, team2_name)
    return lookup


def _chronological_maps(maps: pd.DataFrame) -> pd.DataFrame:
    """Return the as-of maps sorted by ``(date, match_id, map_index)``.

    The recency window needs the team's maps in true play order. The
    ``date`` column carried by :func:`utils.asof.maps_as_of` is a
    validated, non-null ISO-8601 string, but it is re-parsed with
    ``pandas.to_datetime`` here (matching ``features.elo``'s established
    sort-key convention) so ordering never depends on string collation.
    ``match_id`` and ``map_index`` are the tie-breaks: ``map_index`` is
    the per-match 0-indexed play order, so a match's maps sort in play
    order rather than in whatever order they appear in the input.

    Args:
        maps: The output of :func:`utils.asof.maps_as_of`; needs
            ``date``, ``match_id`` and ``map_index``.

    Returns:
        A copy of ``maps`` sorted ascending by
        ``(date, match_id, map_index)`` (stable), preserving all columns
        and the original (post-merge) index values.

    Raises:
        KeyError: If a required column is missing (propagated from
            pandas).
        ValueError: If a date value cannot be parsed (propagated from
            ``pandas.to_datetime``; a null date cannot reach this point
            because ``maps_as_of`` already rejected null dates).
    """
    parsed = pd.to_datetime(maps[asof.DATE_COL])
    return (
        maps.assign(_parsed_date=parsed)
        .sort_values(
            ["_parsed_date", asof.MATCH_ID_COL, asof.MAP_INDEX_COL],
            kind="stable",
        )
        .drop(columns=["_parsed_date"])
    )


def _validated_roster(
    group: pd.DataFrame,
    resolved_name: str,
    valid_names: set,
    match_id: str,
    map_index: int,
) -> pd.DataFrame:
    """Validate a map's player rows and return the queried team's roster.

    Applies the name-mismatch fail-loud guard to one ``(match_id,
    map_index)`` group of ``player_map_stats`` rows: every row's
    ``team_name`` must equal one of the match's two side names
    (``team1_name``/``team2_name``). The first name that matches neither
    raises ``ValueError`` — corruption is reported, never silently
    dropped or guessed at. The queried team's roster (``team_name ==
    resolved_name``) is then returned; it may be empty, which the caller
    treats as "skip this map" (a map whose group holds only the
    opponent's rows).

    Args:
        group: The ``player_map_stats`` rows for one ``(match_id,
            map_index)`` key. The caller short-circuits an *absent*
            group before calling this, so ``group`` is expected non-empty
            here (an empty-but-present group passes vacuously and yields
            an empty roster).
        resolved_name: The queried team's display name for this match
            (``team1_name`` if ``team_is_team1`` else ``team2_name``).
        valid_names: The match's two side names
            (``{team1_name, team2_name}``); any row ``team_name`` outside
            this set is a mismatch.
        match_id: The match id, used only in the error message.
        map_index: The map index, used only in the error message.

    Returns:
        The subset of ``group`` whose ``team_name`` equals
        ``resolved_name`` (possibly empty).

    Raises:
        ValueError: If any row's ``team_name`` matches neither side of
            the match; the message lists the offending names and the two
            valid names.
    """
    invalid = group[~group[TEAM_NAME_COL].isin(valid_names)]
    if not invalid.empty:
        bad = invalid[TEAM_NAME_COL].drop_duplicates().tolist()
        raise ValueError(
            f"player_map_stats for match {match_id!r} map_index {map_index} "
            f"contains team_name(s) {bad} matching neither team1_name nor "
            f"team2_name of that match (valid names: {sorted(valid_names)})"
        )
    return group[group[TEAM_NAME_COL] == resolved_name]
