"""Point-in-time (as-of) feature access layer (roadmap M12).

The leakage-safety scaffolding that every downstream feature milestone
(M13-M17: win rate, Elo, closeness/OT, player form, head-to-head) must
go through instead of reading ``data/<version>/matches.parquet`` or
``data/<version>/maps.parquet`` directly. It does **not** compute any
feature value itself — it only answers one question, and answers it
safely: *which already-completed matches and maps does a given team
have as of a given cutoff date?*

The whole module is built around a single, non-negotiable contract:

- **Strict ``<`` boundary.** A row whose date is *equal to* the query
  date is excluded. Only rows strictly before the cutoff are returned.
  This is the direct reading of roadmap M12's requirement ("no row
  dated ≥ the match date can enter a feature") and is locked in by
  ``tests/test_asof.py``'s core leakage-proof test.
- **Same-day ties need no tie-break.** Because the boundary is strict
  ``<``, two matches carrying the *identical* timestamp string never
  see each other, regardless of insertion order. This is the chosen
  (safe) resolution of the "how are ties on the same date handled"
  ambiguity, documented here so a later contributor does not add a
  tie-breaker that reintroduces leakage.
- **Dates are parsed, not string-sorted.** The ``date`` columns are
  ISO-8601 strings; they sort correctly as plain strings only while
  every row shares one exact format/precision (true for v1). Rather
  than rely on that, dates are parsed with ``pandas.to_datetime``
  (matching ``utils/splits.py``'s ``_chronological_order``
  convention) and a null or unparseable date — in the query or in any
  row — raises ``ValueError`` instead of silently mis-ordering.
- **The team key is ``team_id``, not ``team_name``.** ``team_id`` is
  the stable identifier (a string such as ``"2593"``, matching
  ``matches.team1_id``/``team2_id``); display names are not guaranteed
  unique/stable. No name-to-id resolution is attempted here.
- **Single-team API.** ``features_as_of(team_id, date, ...)`` filters
  history for *one* team. A pairwise feature (e.g. M17 head-to-head)
  is expected to call it twice, once per side, rather than the
  framework growing a second two-team entry point.
- **Completed rows only.** Only ``status == "completed"`` matches (and
  their maps) are eligible: a live/upcoming match carries no usable
  outcome signal for win-rate/Elo/closeness-style features, so the
  as-of contract filters it out rather than leaving that to each
  caller. The match-level half of this is the ``status`` check above;
  the map-level half is enforced separately in :func:`maps_as_of`,
  which drops any map whose ``winner`` is null (an unfinished map)
  even when its parent match is ``completed`` — see that function's
  docstring for the exact rule.
- **Unknown team is empty, not an error.** A new/unseen ``team_id``
  (or a ``team_id`` whose first match is still in the future of the
  query date) legitimately yields zero history rows. Feature functions
  must handle the empty case; this layer returns empty DataFrames
  rather than raising.

Scope of what is wired up today:

- ``matches.parquet`` and ``maps.parquet`` join cleanly on
  ``match_id`` and are therefore wired through this framework now.
- ``veto_actions.parquet`` and ``player_map_stats.parquet`` are
  **deliberately out of scope** pending an id-resolution step: both
  carry only a team *name* (or, for veto actions, a short abbreviation
  like ``"FNC"``) rather than a ``team_id``, so there is no direct
  join key today. Wiring them (M16/M17) requires resolving those names
  to ids first; this module's API shape does not preclude that, but it
  also does not silently paper over the gap with a fragile
  name-matching join.
- No per-team performance work (history caching/indexing) is done
  here. What is cached is narrower and orthogonal: the parsed
  ``matches`` table date column is parsed once per table per process
  (keyed on the table's identity via
  :func:`cached_parsed_date_column`) instead of on every as-of call,
  since date parsing is the one pure cost identical across every
  query against the same table. The per-team filtering logic itself
  still runs on every call, and correctness and the leakage proof
  still come first; per-team caching remains a candidate follow-up,
  not this module's job.

This module lives in ``utils/`` next to ``utils/scoring.py`` and
``utils/table_io.py`` per the boundary rule: it is a pure in-memory
filter over already-materialised DataFrames, with no CLI, no
``argparse`` entry point, and (except for :func:`load_asof_tables`)
no file I/O of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.table_io import DEFAULT_OUTPUT_DIR

# Column-name constants for the M8 matches/maps tables this module
# reads. Kept as module-level names so the functions, the docstrings,
# and the tests all reference one spelling.
DATE_COL = "date"
MATCH_ID_COL = "match_id"
TEAM1_ID_COL = "team1_id"
TEAM2_ID_COL = "team2_id"
STATUS_COL = "status"
COMPLETED_STATUS = "completed"

# M8's "this map is finished" signal on the maps table: ``winner`` is
# ``"team1"``/``"team2"`` for a finished map and ``None`` for an
# unfinished one (``materialize.build_maps_table`` skips the latter).
WINNER_COL = "winner"

# The added orientation column on :func:`maps_as_of`'s output: True
# when the queried team is the match's ``team1`` (so its score is
# ``team1_score``), False when it is ``team2`` (its score is
# ``team2_score``).
TEAM_ORIENTATION_COL = "team_is_team1"

# The per-match map-ordering key carried through on :func:`maps_as_of`'s
# output (an original maps-table column, 0-indexed per match in M8's
# ``maps.parquet``). Feature modules use it to order a match's maps in
# play order; it is a shared constant here so they do not each redefine
# the spelling.
MAP_INDEX_COL = "map_index"

# The columns :func:`matches_as_of` needs on the matches table.
_MATCHES_REQUIRED = (TEAM1_ID_COL, TEAM2_ID_COL, DATE_COL, STATUS_COL)

# The columns :func:`maps_as_of` needs on the maps table: the join key
# plus the map-completion signal used by the map-level filter.
_MAPS_REQUIRED = (MATCH_ID_COL, WINNER_COL)

# Process-lifetime cache of parsed matches-table date columns, keyed on
# ``id(matches_df)`` (the DataFrame's identity — stable for the lifetime
# of a feature-build/evaluation loop, unlike a Series-keyed cache, whose
# key would silently invalidate on any unrelated column reassignment).
# Each value pins a strong reference to the DataFrame it was parsed from
# so the id cannot be freed and reused by an unrelated DataFrame while
# the entry is alive (id-reuse would otherwise risk a stale hit);
# :func:`cached_parsed_date_column` additionally checks the pinned frame
# is the same object and its length is unchanged before returning a hit.
_DATE_COLUMN_CACHE: dict[int, tuple[pd.DataFrame, pd.Series]] = {}


@dataclass(eq=False)
class AsOfBundle:
    """Structured result returned by :func:`features_as_of`.

    A small, explicit container (rather than a bare ``dict`` or a raw
    tuple) so downstream feature functions have named attribute access
    to the two filtered tables. ``eq=False`` is deliberate: the two
    fields are DataFrames, whose ``==`` produces an element-wise
    DataFrame rather than a single ``bool``, so a generated ``__eq__``
    would raise on any bundle comparison.

    Attributes:
        team_id: The queried team id (echoed unchanged from the call,
            for convenience/traceability).
        date: The queried as-of cutoff, exactly as passed in (the
            original string, not the parsed timestamp).
        matches: The completed, strictly-earlier matches the team
            played in (see :func:`matches_as_of` for the exact filter
            and shape).
        maps: The completed, strictly-earlier maps belonging to those
            matches, with an added ``team_is_team1`` orientation flag
            and the match's ``date`` carried over (see
            :func:`maps_as_of` for the exact shape).
    """

    team_id: str
    date: str
    matches: pd.DataFrame
    maps: pd.DataFrame


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    """Raise ``KeyError`` if a table is missing required columns.

    The single shared missing-column check used by the as-of functions,
    so every entry point reports a missing M8 column the same way
    (matching the existing ``drivers/*`` convention of surfacing the
    missing name rather than a generic pandas error).

    Args:
        df: The table to check.
        columns: The column names that must all be present.
        table: A human-readable table name for the error message
            (e.g. ``"matches_df"``).

    Returns:
        None.

    Raises:
        KeyError: If any name in ``columns`` is absent from ``df``;
            the message lists every missing name.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{table} is missing required column(s): {missing}")


def parse_query_date(date: str) -> pd.Timestamp:
    """Parse and validate a single as-of cutoff date.

    Turns the caller's cutoff into a single ``pandas.Timestamp`` that
    can be compared against the parsed table date column. The parse is
    done with ``pandas.to_datetime`` in its ``format="ISO8601"``
    precision-adaptive mode (the convention shared with
    ``utils.splits``, with the ISO-8601 format pinned so any precision
    within the ISO-8601 family — with or without sub-second digits —
    parses identically) rather than left as a string, and the result
    is validated to be exactly one real (non-null, timezone-naive)
    timestamp, so a null or list-like cutoff fails loudly instead of
    silently producing an empty or mis-ordered result.

    Args:
        date: The as-of cutoff. Must be a single ISO-8601 date string
            (or anything ``pandas.to_datetime`` accepts for a scalar),
            e.g. ``"2026-04-01T11:00:00"``. List-like inputs are
            rejected.

    Returns:
        The cutoff as a single ``pandas.Timestamp``.

    Raises:
        ValueError: If ``date`` cannot be parsed by
            ``pandas.to_datetime``; if it is null (``None``/``NaN``,
            which parse to ``None``/``NaT`` and have no chronological
            position); or if it parses to a timezone-aware timestamp
            (the v1 date column is timezone-naive, so a tz-aware
            cutoff cannot be compared against it).
        TypeError: If ``date`` is list-like rather than a single
            scalar timestamp.
    """
    try:
        parsed = pd.to_datetime(date, format="ISO8601")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"query date {date!r} is not a parseable timestamp: {exc}"
        ) from exc
    if parsed is None or pd.isna(parsed) is True:
        raise ValueError(
            f"query date {date!r} is null (None/NaN -> NaT) and has no "
            "chronological position, so it cannot be an as-of cutoff"
        )
    if not isinstance(parsed, pd.Timestamp):
        raise TypeError(
            f"query date {date!r} must be a single timestamp, got a "
            f"{type(parsed).__name__}"
        )
    if parsed.tzinfo is not None:
        raise ValueError(
            f"query date {date!r} is timezone-aware; the v1 date column "
            "is timezone-naive, so pass a naive timestamp"
        )
    return parsed


def parse_date_column(dates: pd.Series) -> pd.Series:
    """Parse and null-check a table's date column for as-of filtering.

    Parses the raw (ISO-8601 string) date column with
    ``pandas.to_datetime`` in its ``format="ISO8601"`` mode (pandas'
    precision-adaptive ISO-8601 parser, so rows with differing
    second/microsecond precision still parse) and rejects any null
    value, mirroring the null-date guard already established in
    ``utils.splits``: a null date parses to ``NaT`` rather than
    raising, and ``NaT`` has no chronological position, so it must not
    silently pass (or fail) an as-of comparison.

    Args:
        dates: The raw date column (typically the ``"date"`` column of
            M8's matches table).

    Returns:
        The parsed column as a ``pandas.Series`` of ``datetime64[ns]``
        values, aligned to the input index.

    Raises:
        ValueError: If any value cannot be parsed (propagated from
            ``pandas.to_datetime`` with its default
            ``errors="raise"``), or if any parsed value is null
            (``None``/``NaN`` -> ``NaT``).
    """
    try:
        parsed = pd.to_datetime(dates, format="ISO8601")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"date column contains an unparseable value: {exc}") from exc
    null_mask = parsed.isna()
    if null_mask.any():
        positions = [int(i) for i in np.flatnonzero(null_mask.to_numpy())]
        raise ValueError(
            f"date column contains {len(positions)} null value(s) at row(s) "
            f"{positions}: a null date (None/NaN) parses to NaT and has no "
            "chronological position, so it cannot be as-of filtered"
        )
    return parsed


def cached_parsed_date_column(matches_df: pd.DataFrame) -> pd.Series:
    """Return ``matches_df``'s parsed date column, parsed once per table.

    The performance wrapper over :func:`parse_date_column` that every
    as-of date-parse call site routes through (``matches_as_of`` and
    the league-wide helpers in ``features/``/``evaluation/`` that call
    ``parse_date_column(matches_df[DATE_COL])`` directly): parsing the
    date column is a pure cost identical for every as-of query against
    the same table, so it is done once per table per process and
    reused, rather than re-parsed on every call.

    The cache is keyed on ``id(matches_df)`` — the DataFrame's object
    identity, which is stable for the whole lifetime of a feature-build
    or evaluation loop because the same ``matches_df`` object is
    threaded through unchanged. It is deliberately *not* keyed on
    ``matches_df[DATE_COL]`` (the Series): pandas clears its internal
    per-column cache whenever any column is reassigned, so a Series
    key would silently degrade to perpetual misses under an unrelated
    mutation elsewhere, and a bare Series carries no back-reference to
    its parent DataFrame anyway. Each cache value is the
    ``(matches_df, parsed_series)`` pair, not just the parsed series:
    pinning a strong reference to the DataFrame prevents its ``id``
    from being freed and reused by a new, unrelated DataFrame while
    the entry is alive, which would otherwise risk a stale hit. A hit
    additionally requires the pinned frame to *be* the queried object
    (``entry[0] is matches_df``) and to have unchanged length
    (``len(entry[1]) == len(matches_df)`` — a defense-in-depth
    backstop against in-place row mutation, the documented-immutable
    claim being the primary safety argument). Any of those checks
    failing is treated as a miss: the column is re-parsed, the entry
    is overwritten, and the fresh result is returned — a stale-cache
    detection never raises, because a fresh parse is always a safe
    fallback.

    This is purely a performance cache, not a new validation layer:
    behavior on a miss is identical to calling :func:`parse_date_column`
    directly (null/parse-error handling is not duplicated here), and a
    hit performs no parsing at all and raises nothing.

    Args:
        matches_df: The materialised ``matches`` table whose ``date``
            column (see :data:`DATE_COL`) is to be parsed. Only its
            identity (``id``) and length are read on a cache hit; the
            ``date`` column itself is read only on a miss.

    Returns:
        The parsed ``date`` column as a ``pandas.Series`` of
        ``datetime64[ns]`` values aligned to ``matches_df``'s index.
        Repeat calls with the same (unmutated) ``matches_df`` object
        return the exact same cached Series object; a different
        ``matches_df`` (even one with identical contents) or a length-
        mutated one re-parses and returns a fresh Series.

    Raises:
        KeyError: If ``matches_df`` has no ``date`` column and no
            valid cache entry exists (propagated from the
            ``matches_df[DATE_COL]`` indexing on a miss).
        ValueError: If the ``date`` column contains a null or
            unparseable value and no valid cache entry exists
            (propagated from :func:`parse_date_column` on a miss).
            A cache hit raises nothing.
    """
    key = id(matches_df)
    entry = _DATE_COLUMN_CACHE.get(key)
    if entry is not None and entry[0] is matches_df and len(entry[1]) == len(
        matches_df
    ):
        return entry[1]
    parsed = parse_date_column(matches_df[DATE_COL])
    _DATE_COLUMN_CACHE[key] = (matches_df, parsed)
    return parsed


def matches_as_of(team_id: str, date: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Return the team's completed matches strictly before a cutoff date.

    The match-level half of the as-of access layer. It applies three
    filters, in this order of intent: the team must be one of the two
    sides (``team1_id`` or ``team2_id``), the match must be completed
    (``status == "completed"``), and the match date must be strictly
    less than the query date. All three are boolean masks, so the
    original row order and index are preserved and the output is a
    faithful *subset* of ``matches_df`` — no columns are added, removed,
    or reordered.

    Args:
        team_id: The queried team's stable id, as a string matching the
            dtype of ``team1_id``/``team2_id`` (M8 stores both as
            object/string). No type coercion is performed: a ``team_id``
            whose type does not match the stored ids simply matches
            nothing and yields an empty result.
        date: The as-of cutoff (see :func:`parse_query_date`). Rows
            dated equal to or after this are excluded (strict ``<``).
        matches_df: The materialised ``matches`` table (M8's
            ``matches.parquet``). Only ``team1_id``, ``team2_id``,
            ``date`` and ``status`` are read; every other column is
            carried through untouched.

    Returns:
        A ``pandas.DataFrame`` with the same columns and (original)
        index as ``matches_df``, containing exactly the rows where the
        team appears on either side, the match is completed, and the
        parsed match date is strictly before the parsed query date. An
        unseen team, a team with no prior completed matches, or an
        empty-but-well-formed input table yields a zero-row frame with
        the full original column set — an empty result is a normal,
        non-error outcome.

    Raises:
        KeyError: If ``matches_df`` lacks any of ``team1_id``,
            ``team2_id``, ``date`` or ``status``.
        ValueError: If the query date is null/unparseable/timezone-aware
            (see :func:`parse_query_date`), or if any row date is
            null/unparseable (see :func:`parse_date_column`).
        TypeError: If the query date is list-like rather than a single
            scalar (see :func:`parse_query_date`).
    """
    require_columns(matches_df, _MATCHES_REQUIRED, "matches_df")

    parsed_dates = cached_parsed_date_column(matches_df)
    query = parse_query_date(date)

    team1 = matches_df[TEAM1_ID_COL]
    team2 = matches_df[TEAM2_ID_COL]
    is_team = (team1 == team_id) | (team2 == team_id)
    is_completed = matches_df[STATUS_COL] == COMPLETED_STATUS
    is_before = parsed_dates < query

    return matches_df[is_team & is_completed & is_before]


def maps_as_of(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the team's completed maps (with dates + orientation) before a cutoff.

    The map-level half of the as-of access layer. A map has no date or
    team identity of its own in M8's schema — both only exist via a
    join back to ``matches`` on ``match_id`` — so this function first
    runs the match-level filter (:func:`matches_as_of`) and then inner-
    joins ``maps_df`` against those matches. The inner join is what
    applies the match-level filters to the maps in one step: a map whose
    match is the wrong team, not completed, or not strictly before the
    cutoff is absent from the join keys and therefore dropped.

    A map-level completeness filter is applied in addition to (and
    before) that join: only maps whose ``winner`` is non-null are
    eligible. ``winner`` is M8's "this map is finished" signal —
    ``materialize.build_maps_table`` already skips winner-null maps, but
    a ``completed`` match can in principle still carry an unfinished
    child map, so the as-of contract enforces the map-level half of
    "completed rows only" here rather than assuming it away. The join is
    also guarded against fan-out: the as-of-filtered ``matches`` frame
    must have unique ``match_id`` values, otherwise an inner join would
    silently duplicate every map row of the offending match.

    Two columns are added to each surviving map row:

    - ``date`` — the match's original (unparsed) date string, carried
      over from ``matches_df`` so a feature function can see *when* the
      map happened without re-joining;
    - ``team_is_team1`` (see :data:`TEAM_ORIENTATION_COL`) — a boolean
      orientation flag: ``True`` means the queried team is that match's
      ``team1`` and its score is ``team1_score``; ``False`` means it is
      ``team2`` and its score is ``team2_score``. This lets a caller
      tell which score belongs to the queried team without re-deriving
      it from the team-id columns.

    Args:
        team_id: The queried team's stable id (see
            :func:`matches_as_of`).
        date: The as-of cutoff (see :func:`parse_query_date`).
        matches_df: The materialised ``matches`` table; read by
            :func:`matches_as_of` and used as the join source for
            ``date``/orientation.
        maps_df: The materialised ``maps`` table (M8's
            ``maps.parquet``). Every column is carried through; only
            ``match_id`` (join key) and ``winner`` (map-completion
            signal) are read.

    Returns:
        A ``pandas.DataFrame`` with all of ``maps_df``'s columns (in
        their original order) followed by ``date`` and then
        ``team_is_team1``, containing exactly the rows whose match
        passes the match-level filter *and* whose own ``winner`` is
        non-null (a finished map). A zero-row result (unseen team, no
        prior completed maps, or an empty join) carries the full
        ``maps`` + ``date`` + ``team_is_team1`` column set with
        ``team_is_team1`` still boolean.

    Raises:
        KeyError: If ``maps_df`` lacks ``match_id`` or ``winner``, or if
            ``matches_df`` lacks a required column (propagated from
            :func:`matches_as_of`).
        ValueError: If the as-of-filtered ``matches`` frame contains
            duplicate ``match_id`` values (the join would fan out and
            duplicate map rows); or if the query date or a row date is
            null/unparseable/timezone-aware (propagated from
            :func:`matches_as_of` / the parse helpers).
        TypeError: If the query date is list-like (propagated from
            :func:`parse_query_date` via :func:`matches_as_of`).
    """
    matches = matches_as_of(team_id, date, matches_df)
    require_columns(maps_df, _MAPS_REQUIRED, "maps_df")

    if not matches[MATCH_ID_COL].is_unique:
        duplicates = matches.loc[
            matches[MATCH_ID_COL].duplicated(keep=False), MATCH_ID_COL
        ].unique().tolist()
        raise ValueError(
            "matches_df contains duplicate match_id value(s) "
            f"{duplicates} after as-of filtering; the maps join would "
            "fan out and duplicate map rows"
        )

    finished_maps = maps_df[maps_df[WINNER_COL].notna()]

    is_team1 = (matches[TEAM1_ID_COL] == team_id).to_numpy()
    join_frame = pd.DataFrame(
        {
            MATCH_ID_COL: matches[MATCH_ID_COL].to_numpy(),
            DATE_COL: matches[DATE_COL].to_numpy(),
            TEAM_ORIENTATION_COL: is_team1,
        }
    )
    return finished_maps.merge(join_frame, on=MATCH_ID_COL, how="inner")


def features_as_of(
    team_id: str,
    date: str,
    matches_df: pd.DataFrame,
    maps_df: pd.DataFrame,
) -> AsOfBundle:
    """Return the as-of bundle a feature function should consume.

    The single public entry point roadmap M12 asks for: it filters both
    tables as of ``(team_id, date)`` and returns them together in an
    :class:`AsOfBundle`. Downstream feature functions (M13+) are
    expected to call this (or the two individual filters) and read the
    resulting rows instead of touching ``matches.parquet``/
    ``maps.parquet`` directly — that indirection is the whole point of
    the leakage-safety layer, because it guarantees the strict-``<``
    boundary is applied in exactly one place.

    Args:
        team_id: The queried team's stable id (see
            :func:`matches_as_of`).
        date: The as-of cutoff (see :func:`parse_query_date`).
        matches_df: The materialised ``matches`` table.
        maps_df: The materialised ``maps`` table.

    Returns:
        An :class:`AsOfBundle` whose ``matches`` field is the
        :func:`matches_as_of` result and whose ``maps`` field is the
        :func:`maps_as_of` result (the two filters are computed
        independently, so the match-level filter runs twice — once for
        the bundle and once inside ``maps_as_of``. That redundant
        *filtering* work is deliberate and unchanged; only its
        date-parsing cost is gone, since the second
        :func:`matches_as_of` call reuses the parsed date column cached
        by :func:`cached_parsed_date_column`. Removing the redundant
        filtering itself is a separate structural refactor, not this
        module's job; per-team caching/indexing remains an explicit
        follow-up).

    Raises:
        KeyError: If either table lacks a required column (propagated
            from :func:`matches_as_of` / :func:`maps_as_of`).
        ValueError: If the as-of-filtered ``matches`` frame contains
            duplicate ``match_id`` values (propagated from
            :func:`maps_as_of`); or if the query date or a row date is
            null/unparseable/timezone-aware (propagated from the parse
            helpers).
        TypeError: If the query date is list-like (propagated from
            :func:`parse_query_date`).
    """
    matches = matches_as_of(team_id, date, matches_df)
    maps = maps_as_of(team_id, date, matches_df, maps_df)
    return AsOfBundle(team_id=team_id, date=date, matches=matches, maps=maps)


def load_asof_tables(
    version: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the materialised matches and maps tables for a version.

    A thin disk-I/O convenience wrapper so callers who do not already
    hold the DataFrames in memory can still get started with one call.
    It reads ``<output_dir>/<version>/matches.parquet`` and
    ``<output_dir>/<version>/maps.parquet`` via ``pandas.read_parquet``
    (the same convention as ``drivers.splits.load_matches_table`` /
    ``drivers.labels.load_maps_table``) and hands both to the pure
    functions above — it does not re-implement any filtering logic.

    Args:
        version: The dataset version subdirectory name (e.g. ``"v1"``).
        output_dir: The parent directory the version subdirectory
            lives under (default :data:`utils.table_io.DEFAULT_OUTPUT_DIR`,
            i.e. ``Path("data")``).

    Returns:
        A ``(matches_df, maps_df)`` tuple of the two loaded DataFrames,
        in that order.

    Raises:
        FileNotFoundError: If either Parquet file does not exist for
            this version (i.e. ``materialize.py`` has not been run for
            it) — propagated as-is from ``pandas.read_parquet`` as a
            clear "run materialize.py first" signal rather than
            wrapped.
        OSError: On any other file-access failure (permissions, etc.),
            also propagated as-is.
    """
    version_dir = Path(output_dir) / version
    matches_df = pd.read_parquet(version_dir / "matches.parquet")
    maps_df = pd.read_parquet(version_dir / "maps.parquet")
    return matches_df, maps_df
