"""Recency-weighted team player-form features (roadmap M16).

A pure in-memory library producing, for one team as of a cutoff date, a
recency-weighted rolling mean of two per-player statistics — ``acs``
(Average Combat Score) and ``rating`` — aggregated up to team level from
``player_map_stats``. The roadmap calls this "recency-weighted rolling
mean ACS/rating over the last N maps, aggregated to team level"; this
module is the M16 implementation of exactly that.

Like the rest of ``utils/``, this module has no CLI, no ``argparse``
entry point, and (except for :func:`load_player_form_tables`) no file
I/O of its own — it operates on the already-materialised
``matches_df`` / ``maps_df`` / ``player_map_stats_df`` DataFrames a
caller passes in.

**Leakage contract (the hard requirement).** Feature history originates
from :func:`utils.asof.maps_as_of` — never ``pd.read_parquet`` on
``matches.parquet`` / ``maps.parquet`` / ``player_map_stats.parquet``
directly. The strict ``<`` boundary is enforced by ``utils.asof``'s date
parsing (reused here, not reimplemented), so a map dated equal to or
after the query date never enters any estimate. ``player_map_stats``
carries no date of its own; its rows enter a query *only* through the
``(match_id, map_index)`` keys of the as-of-filtered maps, so it
inherits the same strict-``<`` boundary automatically.

**Linkage design (per-match name resolution, no global crosswalk).**
``player_map_stats`` has a ``team_name`` (display string) but no
``team_id``, while every as-of feature is keyed by ``team_id``. Rather
than a global/precomputed ``name <-> id`` table (which would collide if
one display string is ever reused by two ids in different eras), the
name is resolved **per as-of map row**, using the same match row that
already carries the queried team's orientation:

1. ``utils.asof.maps_as_of(team_id, date, matches_df, maps_df)`` returns
   one row per finished, strictly-prior map the team played, each
   carrying ``match_id``, ``map_index``, ``team_is_team1`` and ``date``.
2. The queried team's display name *for that specific match* is
   ``team1_name`` when ``team_is_team1`` is truthy, else ``team2_name``
   — read off ``matches_df`` (which therefore must carry ``team1_name``
   and ``team2_name``, extra required columns beyond what ``utils.asof``
   itself needs).
3. ``player_map_stats_df`` is then filtered to
   ``(match_id, map_index, team_name == resolved_name)`` to obtain that
   team's roster rows for that map.

**Three-way failure split (deliberate, load-bearing — do not collapse).**
There are exactly three distinct "something is missing" cases, handled
three different ways on purpose:

- **Name/id mismatch -> fail loud (``ValueError``).** If any
  ``player_map_stats`` row for a processed ``(match_id, map_index)``
  carries a ``team_name`` matching *neither* ``team1_name`` nor
  ``team2_name`` of that match (data corruption, or a mid-event
  rebrand not reflected in ``matches.parquet``), the query raises
  ``ValueError`` rather than silently dropping the row or guessing.
  This mirrors the repo's established fail-loud convention in
  ``utils/map_win_rate.py`` / ``utils/elo.py`` /
  ``utils/closeness.py``.
- **A map with zero matching player rows -> skip-and-count.** A finished
  as-of map whose ``(match_id, map_index)`` has no ``player_map_stats``
  rows at all (the real 242/244 gap), or whose group has rows only for
  the opponent, is skipped and counted in
  ``PlayerFormResult.skipped_maps`` — a missing player-stat row for an
  otherwise-valid map is "no player-form signal for that map", the same
  class as a brand-new team having no signal at all, *not* corruption
  of the map's own meaning.
- **Per-row null ``acs``/``rating`` -> skip-and-count.** A roster row
  whose ``acs`` (or ``rating``) is null is excluded from that map's
  *unweighted* mean for that stat (counted in the stat's
  ``null_rows_skipped``), and the other stat still computes normally
  from the same rows. If **every** roster row for a team on a map is
  null for one stat, that map contributes no value for *that stat only*
  (the two stats are computed independently; no division by zero, no
  knock-on effect on the other stat). A null *score* is a different
  failure class — it corrupts a map's win/loss meaning — and is already
  guarded loudly in ``utils/elo.py`` / ``utils/map_win_rate.py``; this
  module does not re-litigate that.

**Aggregation design (two stages, in this exact order).**

1. **Per-map, per-team unweighted roster mean.** For each qualifying
   as-of map, the team's ``acs`` (and, independently, ``rating``) mean
   is the plain mean over however many roster rows actually joined
   (typically 5; never a hardcoded "5" divisor), after null rows are
   dropped. Recency weighting is *never* applied within a map — every
   roster row on the same map counts equally.
2. **Recency weighting across maps, windowed to the last N.** The
   team's most recent ``N`` qualifying maps (ordered by
   ``(date, match_id, map_index)`` — the same tie-break tuple
   ``utils/elo.py`` established for its league-wide replay) receive
   weights ``w_i = decay_rate ** rank`` where rank 0 is the most recent
   map (weight 1.0), rank 1 the next, etc.; the reported value is
   ``sum(w_i * map_mean_i) / sum(w_i)``.

**Empty / partial history.** With zero qualifying maps the result is
``mean = None`` and ``maps_used = 0`` — an honest sentinel, *not* a
fabricated numeric default (a form feature has no principled
"uninformative" number the way a win-rate prior has the coin-flip 0.5;
see ``utils.map_win_rate.OverallWinRate`` for the contrast). With
``1..N-1`` qualifying maps, all of them are used with no padding and no
reweighting — the same "partial window uses what exists" principle as
``utils.elo``'s full replay.

**Chosen constants (documented defaults, not CV-tuned; roadmap M16 asks
for no CV routine, matching M14's documented-default precedent):**

- ``DEFAULT_FORM_WINDOW = 10`` — roughly three Bo3 matches of maps; long
  enough to smooth single-map noise, short enough to track recent
  roster/form swings.
- ``DEFAULT_DECAY_RATE = 0.9`` — with N=10 the oldest map in the window
  carries weight ``0.9**9 ≈ 0.387`` relative to the most recent, so
  recent maps dominate while the tail is not discarded; near 1.0 the
  window would be almost uniform and near 0.5 it would collapse to ~2
  effective maps. Both may be overridden per call.

**Data-shape findings (re-derived against real ``data/v1``, not
assumed):**

- ``matches.parquet`` (98 x 12) carries both the display name and the
  stable id per side (``team1_name``/``team1_id``,
  ``team2_name``/``team2_id``), which is what makes the per-match name
  resolution above possible without a crosswalk.
- ``player_map_stats.parquet`` (2420 x 15) has columns ``match_id,
  map_index, player_name, team_name, rating, acs, ...``; every
  ``(match_id, map_index)`` group has exactly 2 distinct ``team_name``
  values and 10 rows (5 per side), across 242 groups.
- ``player_map_stats``'s key set is a strict subset of ``maps.parquet``'s:
  242 of 244 maps have player rows; the 2 missing maps are
  ``(match_id=712803, map_index=0)`` and ``(match_id=712803,
  map_index=1)`` — the real, live gap the skip-and-count rule handles.
- 0 ``team_name`` mismatches (every row's ``team_name`` equals its
  match's ``team1_name`` or ``team2_name``); 16 distinct teams, all
  name<->id mappings currently 1:1 (a favourable current-data fact, not
  a guaranteed invariant — which is why no global crosswalk is built).
- 0 null ``acs`` and 0 null ``rating`` values across all 2420 rows; the
  per-row null skip is defensive-only (matching task 017's precedent).

**Scope note.** ``utils.asof`` explicitly leaves ``player_map_stats`` out
of its own wiring "pending an id-resolution step"; this module *is* that
step, so it performs the name resolution itself on top of
``maps_as_of``'s output rather than asking ``utils.asof`` to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils import asof
from utils.table_io import DEFAULT_OUTPUT_DIR

# Default recency window (number of most-recent qualifying maps used)
# and exponential decay rate. Rationale in the module docstring.
DEFAULT_FORM_WINDOW = 10
DEFAULT_DECAY_RATE = 0.9

# Column names this module reads. ``match_id``, ``date`` and
# ``team_is_team1`` come from utils.asof's constants / maps_as_of
# output; ``map_index`` is an original maps-table column carried through
# maps_as_of (shared constant in utils.asof); the rest are the
# matches/player_map_stats columns, named here once so functions and
# tests share one spelling.
TEAM1_NAME_COL = "team1_name"
TEAM2_NAME_COL = "team2_name"
TEAM_NAME_COL = "team_name"
ACS_COL = "acs"
RATING_COL = "rating"
PLAYER_NAME_COL = "player_name"

# Extra columns this module needs on each table, beyond what
# ``utils.asof.maps_as_of`` already requires. ``player_map_stats``'s
# required set is exactly what the aggregation actually reads
# (``player_name`` is deliberately not required: the per-map mean counts
# roster rows, not identities).
_MATCHES_REQUIRED = (TEAM1_NAME_COL, TEAM2_NAME_COL)
_MAPS_REQUIRED = (asof.MAP_INDEX_COL,)
_PMS_REQUIRED = (
    asof.MATCH_ID_COL,
    asof.MAP_INDEX_COL,
    TEAM_NAME_COL,
    ACS_COL,
    RATING_COL,
)


@dataclass(frozen=True)
class FormStat:
    """One stat's recency-weighted team form and the provenance to recompute it.

    ``per_map_means`` and ``weights`` are the exact windowed inputs the
    weighted mean was computed from, listed **most-recent-first** (so
    ``weights[i] == decay_rate ** i``), enabling a caller or test to
    independently recompute ``mean == sum(w_i * m_i) / sum(w_i)``
    without duplicating this module's internals.

    Attributes:
        mean: The recency-weighted mean over the window; ``None`` when
            ``maps_used == 0`` (no qualifying maps — an honest sentinel,
            not a fabricated default).
        maps_used: Number of qualifying maps actually included in the
            window (``<= n``).
        per_map_means: The windowed per-map roster means, most-recent
            first; length ``maps_used``.
        weights: The decay weights matching ``per_map_means``
            (``weights[i] == decay_rate ** i``); length ``maps_used``.
        null_rows_skipped: Roster rows excluded from the per-map means
            because this stat's value was null (a skip-and-count total
            over the whole history, not just the window).
    """

    mean: float | None
    maps_used: int
    per_map_means: tuple[float, ...]
    weights: tuple[float, ...]
    null_rows_skipped: int


@dataclass(frozen=True)
class PlayerFormResult:
    """The bundled ``acs`` and ``rating`` team-form result for one as-of query.

    Both stats are computed from the same as-of history but *independently*
    (one stat being fully null on a map never knocks out the other), so
    they are bundled here side by side rather than requiring two calls.
    ``as_of_maps`` and ``skipped_maps`` describe the shared history
    feeding both.

    Attributes:
        team_id: The queried team id (echoed unchanged).
        date: The as-of cutoff, exactly as passed in (original string).
        acs: The :class:`FormStat` for ``acs``.
        rating: The :class:`FormStat` for ``rating``.
        as_of_maps: Total finished, strictly-prior maps returned by
            :func:`utils.asof.maps_as_of` (the history considered).
        skipped_maps: Of those, the maps skipped-and-counted because they
            had zero matching player rows (no ``player_map_stats`` group
            at all, or a group with no rows for the queried team).
    """

    team_id: str
    date: str
    acs: FormStat
    rating: FormStat
    as_of_maps: int
    skipped_maps: int


def _validate_n(n) -> int:
    """Validate the window size ``n`` and return it as an ``int``.

    ``n`` is the number of most-recent qualifying maps kept in the
    recency window. It must be a positive integer: a ``bool`` is
    rejected explicitly (even though it is ``int``-coercible), and a
    non-integral value (a float with a fraction, a numeric string, etc.)
    is rejected rather than truncated. ``n <= 0`` would produce an empty
    window for every query, which is never a caller's intent.

    Args:
        n: The window size. An ``int`` (or numpy integer) is accepted;
            bools, non-integral floats, strings and other types are
            rejected.

    Returns:
        ``n`` as a plain ``int``.

    Raises:
        ValueError: If ``n`` is a ``bool``, not integer-valued, or
            ``<= 0``.
    """
    if type(n) is bool:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    try:
        value = int(n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"n must be a positive integer, got {n!r}") from exc
    if value != n or value <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    return value


def _validate_decay_rate(decay_rate) -> float:
    """Validate the recency decay rate and return it as a float.

    The decay rate is the per-rank multiplier in the exponential weight
    ``w_i = decay_rate ** rank``. It must be a finite real in ``[0, 1]``:
    a rate ``> 1`` would weight *older* maps more heavily, a negative
    rate would alternate sign, and NaN/inf would poison every weight.
    ``0.0`` (only the most-recent map counts) and ``1.0`` (a uniform
    window, no decay) are both legal limit cases.

    Args:
        decay_rate: The decay rate. Any real number (``int``/``float``/
            numpy scalar) is coerced to ``float``.

    Returns:
        ``decay_rate`` as a ``float``.

    Raises:
        ValueError: If it cannot be coerced to a ``float``, or if the
            result is NaN, infinite, ``< 0`` or ``> 1``.
    """
    try:
        value = float(decay_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"decay_rate must be a real number in [0, 1], got {decay_rate!r}"
        ) from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"decay_rate must be a finite real number in [0, 1], got {decay_rate!r}"
        )
    return value


def _build_match_name_lookup(matches_df: pd.DataFrame) -> dict:
    """Build a ``match_id -> (team1_name, team2_name)`` lookup.

    The per-match name-resolution source. ``maps_as_of``'s output carries
    ``match_id`` and ``team_is_team1`` but not the display names, so this
    lookup lets each as-of map row resolve the queried team's display
    name from the same match row that already carries its orientation.
    It also validates the two extra ``matches_df`` columns this module
    needs (``team1_name``/``team2_name``) beyond what ``utils.asof``
    itself requires.

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
            :func:`utils.asof._require_columns`).
    """
    asof._require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")
    return {
        getattr(row, asof.MATCH_ID_COL): (
            getattr(row, TEAM1_NAME_COL),
            getattr(row, TEAM2_NAME_COL),
        )
        for row in matches_df.itertuples(index=False)
    }


def _chronological_maps(maps: pd.DataFrame) -> pd.DataFrame:
    """Return the as-of maps sorted by ``(date, match_id, map_index)``.

    The recency window needs the team's maps in true play order. The
    ``date`` column carried by :func:`utils.asof.maps_as_of` is a
    validated, non-null ISO-8601 string, but it is re-parsed with
    ``pandas.to_datetime`` here (matching ``utils.elo``'s established
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


def _recency_weighted(
    means: list[float],
    n: int,
    decay_rate: float,
) -> tuple[float | None, tuple[float, ...], tuple[float, ...]]:
    """Window ``means`` to the last ``n`` and apply exponential recency weights.

    The second aggregation stage: keep the most recent ``n`` per-map means
    (the input is already in chronological ascending order), then weight
    them most-recent-first as ``w_i = decay_rate ** i`` (rank 0 = most
    recent gets weight 1.0). The returned weighted mean is
    ``sum(w_i * m_i) / sum(w_i)``. An empty window returns
    ``(None, (), ())`` — the empty-history sentinel (no numeric default).

    Args:
        means: The team's qualifying per-map means in chronological
            ascending order (oldest first). May be empty.
        n: The window size (already validated positive integer).
        decay_rate: The decay rate (already validated ``[0, 1]``).

    Returns:
        A ``(weighted_mean, per_map_means, weights)`` tuple:
        ``weighted_mean`` is ``None`` when the window is empty;
        ``per_map_means`` and ``weights`` are parallel tuples listed
        most-recent-first (so ``weights[i] == decay_rate ** i``), of
        length ``min(n, len(means))``.

    Raises:
        Nothing (``n``/``decay_rate`` are expected pre-validated, and the
            formula is total for any finite means and ``decay_rate`` in
            ``[0, 1]`` — ``decay_rate == 0.0`` leaves only the most
            recent weight 1.0 with total 1.0).
    """
    window = means[-n:]
    if not window:
        return None, (), ()
    most_recent_first = tuple(reversed(window))
    weights = tuple(decay_rate ** i for i in range(len(most_recent_first)))
    total = sum(weights)
    weighted_mean = sum(w * m for w, m in zip(weights, most_recent_first)) / total
    return weighted_mean, most_recent_first, weights


def team_player_form(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
    player_map_stats_df: pd.DataFrame,
    n: int = DEFAULT_FORM_WINDOW,
    decay_rate: float = DEFAULT_DECAY_RATE,
) -> PlayerFormResult:
    """Compute a team's recency-weighted ACS/rating form as of a cutoff.

    The single public entry point. It (1) fetches the team's finished,
    strictly-prior maps via :func:`utils.asof.maps_as_of` (the leakage
    boundary), (2) resolves, per map, the queried team's display name
    from that map's own match row and joins its ``player_map_stats``
    roster on ``(match_id, map_index, team_name)``, (3) computes the
    per-map unweighted roster mean of ``acs`` and ``rating``
    independently (null rows skipped per stat), and (4) windows to the
    last ``n`` qualifying maps and applies exponential recency decay to
    produce the final weighted means. See the module docstring for the
    exact three-way failure split (name mismatch raises; missing player
    rows and per-row null stats are skipped and counted).

    Args:
        team_id: The queried team's stable id (a string matching the
            dtype of ``team1_id``/``team2_id``).
        date: The as-of cutoff; maps dated ``>=`` this are excluded
            (strict ``<``).
        matches_df: The materialised ``matches`` table. Needs
            ``team1_name``/``team2_name`` in addition to the columns
            :func:`utils.asof.maps_as_of` already requires.
        maps_df: The materialised ``maps`` table. Needs ``map_index`` in
            addition to the columns :func:`utils.asof.maps_as_of`
            already requires.
        player_map_stats_df: The materialised ``player_map_stats`` table
            (needs ``match_id``, ``map_index``, ``team_name``, ``acs``,
            ``rating``). Its rows enter the query only via the as-of
            maps' keys, so it inherits the strict-``<`` boundary.
        n: The recency window size (see :func:`_validate_n`).
        decay_rate: The recency decay rate (see
            :func:`_validate_decay_rate`).

    Returns:
        A :class:`PlayerFormResult` bundling the ``acs`` and ``rating``
        :class:`FormStat` results (each with its weighted ``mean``,
        ``maps_used``, ``per_map_means``/``weights`` and
        ``null_rows_skipped``) plus the shared ``as_of_maps`` /
        ``skipped_maps`` counts. Zero qualifying maps yields
        ``mean is None`` and ``maps_used == 0`` for the affected stat.

    Raises:
        KeyError: If any table lacks a required column (propagated from
            :func:`utils.asof.maps_as_of` /
            :func:`utils.asof._require_columns`; includes
            ``team1_name``/``team2_name``, ``map_index``, and the
            ``player_map_stats`` columns).
        ValueError: If ``n`` or ``decay_rate`` is invalid (see the
            validate helpers); if a ``player_map_stats`` ``team_name``
            matches neither side of its match (name mismatch); or if the
            query date or a row date is null/unparseable/timezone-aware
            (propagated from :func:`utils.asof.maps_as_of`).
        TypeError: If the query date is list-like (propagated from
            :func:`utils.asof.maps_as_of`).
    """
    n_value = _validate_n(n)
    decay = _validate_decay_rate(decay_rate)

    asof._require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")
    asof._require_columns(maps_df, _MAPS_REQUIRED, "maps_df")
    asof._require_columns(player_map_stats_df, _PMS_REQUIRED, "player_map_stats_df")

    maps = asof.maps_as_of(team_id, date, matches_df, maps_df)
    match_names = _build_match_name_lookup(matches_df)
    maps_sorted = _chronological_maps(maps)

    pms_groups = {
        (key[0], int(key[1])): group
        for key, group in player_map_stats_df.groupby(
            [asof.MATCH_ID_COL, asof.MAP_INDEX_COL], sort=False
        )
    }

    acs_means: list[float] = []
    rating_means: list[float] = []
    null_acs_skipped = 0
    null_rating_skipped = 0
    skipped_maps = 0

    for row in maps_sorted.itertuples(index=False):
        match_id = getattr(row, asof.MATCH_ID_COL)
        map_index = int(getattr(row, asof.MAP_INDEX_COL))
        team_is_team1 = bool(getattr(row, asof.TEAM_ORIENTATION_COL))

        team1_name, team2_name = match_names[match_id]
        resolved_name = team1_name if team_is_team1 else team2_name

        group = pms_groups.get((match_id, map_index))
        if group is None:
            skipped_maps += 1
            continue

        roster = _validated_roster(
            group, resolved_name, {team1_name, team2_name}, match_id, map_index
        )
        if roster.empty:
            skipped_maps += 1
            continue

        acs_nonnull = roster[ACS_COL].dropna()
        rating_nonnull = roster[RATING_COL].dropna()
        null_acs_skipped += len(roster) - len(acs_nonnull)
        null_rating_skipped += len(roster) - len(rating_nonnull)
        if len(acs_nonnull):
            acs_means.append(float(acs_nonnull.mean()))
        if len(rating_nonnull):
            rating_means.append(float(rating_nonnull.mean()))

    acs_mean, acs_per_map, acs_weights = _recency_weighted(
        acs_means, n_value, decay
    )
    rating_mean, rating_per_map, rating_weights = _recency_weighted(
        rating_means, n_value, decay
    )

    return PlayerFormResult(
        team_id=team_id,
        date=date,
        acs=FormStat(
            mean=acs_mean,
            maps_used=len(acs_per_map),
            per_map_means=acs_per_map,
            weights=acs_weights,
            null_rows_skipped=null_acs_skipped,
        ),
        rating=FormStat(
            mean=rating_mean,
            maps_used=len(rating_per_map),
            per_map_means=rating_per_map,
            weights=rating_weights,
            null_rows_skipped=null_rating_skipped,
        ),
        as_of_maps=len(maps),
        skipped_maps=skipped_maps,
    )


def load_player_form_tables(
    version: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the materialised matches, maps and player_map_stats tables.

    A thin disk-I/O convenience wrapper mirroring
    :func:`utils.asof.load_asof_tables` but also loading
    ``player_map_stats.parquet``. It reads
    ``<output_dir>/<version>/{matches,maps,player_map_stats}.parquet``
    via ``pandas.read_parquet`` and hands all three to
    :func:`team_player_form` — it re-implements no filtering or feature
    logic.

    Args:
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        output_dir: The parent directory the version subdirectory lives
            under (default :data:`utils.table_io.DEFAULT_OUTPUT_DIR`,
            i.e. ``Path("data")``).

    Returns:
        A ``(matches_df, maps_df, player_map_stats_df)`` tuple of the
        three loaded DataFrames, in that order.

    Raises:
        FileNotFoundError: If any Parquet file does not exist for this
            version (i.e. ``materialize.py`` has not been run for it) —
            propagated as-is from ``pandas.read_parquet`` as a clear
            "run materialize.py first" signal rather than wrapped.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    version_dir = Path(output_dir) / version
    matches_df = pd.read_parquet(version_dir / "matches.parquet")
    maps_df = pd.read_parquet(version_dir / "maps.parquet")
    player_map_stats_df = pd.read_parquet(version_dir / "player_map_stats.parquet")
    return matches_df, maps_df, player_map_stats_df
