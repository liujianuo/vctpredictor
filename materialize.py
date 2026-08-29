"""Command-line dataset materialisation (roadmap M8).

Reads every completed match out of the SQLite cache
(``scraper.cache``), flattens the nested ``Match``/``MapResult``/
``VetoAction``/``PlayerStats`` dataclasses into four flat, versioned
Parquet tables — ``matches``, ``maps``, ``veto_actions``,
``player_map_stats`` — under ``data/<version>/``, and writes a
``report.json`` sanity report (row counts, map count, an ad hoc OT
rate, and the ``best_of`` format mix) alongside them, so a silently
broken scrape (e.g. one event's matches never made it into the cache)
is visible the moment the dataset is materialised rather than
discovered three milestones later.

This module sits at the repo root (next to ``scrape.py``/``config.py``)
per the same boundary rule task 008 established: it is the driver that
ties persistence (``scraper.cache``) and data shape
(``scraper.models``) together, and it does not need ``config`` — it
reads only what is already in the cache, never re-deriving the map
pool or era.

Design rules:

- **Completed matches only.** ``Match.status`` can be
  ``"completed"``/``"live"``/``"upcoming"``; live and upcoming rows
  carry partial or absent scores (per task 007's finished-map gate)
  and are not training data, so only ``status == "completed"``
  matches are materialised. A later milestone that needs live rows
  gets its own flag; it does not change this default.
- **One bad row must not abort the run.** Each cached match is
  deserialized via ``cache.get_cached_match`` under that module's
  corrupt-vs-illegal contract: a corrupt row returns ``None``
  (counted and skipped here), a row that deserializes to an illegal
  scoreline raises :class:`IllegalScoreError` (caught, counted and
  skipped here) — the same per-unit isolation ``scrape.py`` applies to
  events, so a single poisoned cache row cannot discard the whole
  dataset.
- **Flat tables.** The only nested field in any table is
  ``PlayerStats.agents`` (a ``list[str]``), stored as a JSON string
  column (``agents_json``) so every column is a flat scalar —
  consistent with how ``scraper.cache`` itself serializes nested
  structures as JSON.
- **Versioned output.** Each run rewrites ``data/<version>/`` in
  place (``--version``, default ``v1``); a schema change or a
  re-scrape gets a new version directory instead of silently
  clobbering the old one, so a downstream consumer can pin an exact
  path.

Exit codes:

- ``0`` — materialisation succeeded and at least one completed match
  was written.
- ``1`` — the run completed mechanically (the four tables and
  ``report.json`` are still written, possibly empty) but zero
  completed matches were found: an empty v1 dataset is almost
  certainly a bug upstream (an empty/cleared cache, a scrape that
  never ran), so it must not look identical to a healthy run in an
  automation exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from scraper import cache
from scraper.models import IllegalScoreError, MapResult, Match
from table_io import DEFAULT_OUTPUT_DIR, write_parquet

logger = logging.getLogger(__name__)

# Fixed column order for each table, used both to build non-empty
# DataFrames in a deterministic order and to give empty runs a
# schema-correct zero-row table (a bare pd.DataFrame([]) has no
# columns at all, which would write a column-less Parquet file).
MATCHES_COLUMNS = (
    "match_id",
    "url",
    "event_name",
    "date",
    "team1_name",
    "team1_id",
    "team2_name",
    "team2_id",
    "team1_score",
    "team2_score",
    "best_of",
    "status",
)
MAPS_COLUMNS = (
    "match_id",
    "map_index",
    "map_name",
    "team1_score",
    "team2_score",
    "winner",
    "duration",
    "team1_first_half_rounds",
    "team1_second_half_rounds",
    "team1_atk_rounds",
    "team1_def_rounds",
    "team2_first_half_rounds",
    "team2_second_half_rounds",
    "team2_atk_rounds",
    "team2_def_rounds",
)
VETO_ACTIONS_COLUMNS = ("match_id", "step_index", "team", "action", "map_name")
PLAYER_MAP_STATS_COLUMNS = (
    "match_id",
    "map_index",
    "player_name",
    "team_name",
    "rating",
    "acs",
    "kills",
    "deaths",
    "assists",
    "adr",
    "kast",
    "hs_pct",
    "first_kills",
    "first_deaths",
    "agents_json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the materialize.py command line.

    Args:
        argv: The argument list to parse; ``None`` (the default) uses
            ``sys.argv[1:]`` (argparse's standard behavior). Passed
            through explicitly so tests can exercise the flags without
            touching the process-wide ``sys.argv``.

    Returns:
        An ``argparse.Namespace`` with three attributes: ``version``
        (``str``, the output subdirectory name, default ``"v1"``),
        ``output_dir`` (``str``, the parent directory the version
        subdirectory is created under, default ``"data"``) and
        ``db_path`` (``Optional[str]``, the cache database path,
        default ``None`` meaning ``scraper.cache``'s own default).

    Raises:
        SystemExit: On invalid arguments (argparse's standard behavior,
            e.g. an unknown flag).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Materialise every completed cached match into four versioned "
            "Parquet tables (matches, maps, veto_actions, player_map_stats) "
            "plus a report.json sanity report."
        )
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="output subdirectory name under --output-dir (default: v1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "parent directory the version subdirectory is created under "
            "(default: data)"
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            "path to the cache SQLite database (default: scraper.cache's "
            "own default)"
        ),
    )
    return parser.parse_args(argv)


def load_completed_matches(
    db_path: Path | None = None,
) -> tuple[list[Match], dict[str, int]]:
    """Load every completed match currently stored in the cache.

    Enumerates the cache via :func:`scraper.cache.list_cached_match_ids`
    (one new primitive; there is no existing bulk reader), then
    deserializes each id through the existing single-row
    :func:`scraper.cache.get_cached_match`, so the two error kinds that
    function already distinguishes are reused rather than reimplemented
    here: a corrupt row (bad JSON, missing keys, unparseable date)
    returns ``None`` and is counted and skipped; a row that
    deserializes but fails score validity raises
    :class:`scraper.models.IllegalScoreError`, which is caught here,
    counted and skipped — one poisoned cache row can never abort the
    whole materialisation. Surviving matches are then filtered to
    ``status == "completed"`` (live/upcoming matches carry partial or
    absent scores and are not training data; see the module docstring).

    Args:
        db_path: Path to the SQLite cache database, forwarded to every
            ``scraper.cache`` call. ``None`` uses that module's default.

    Returns:
        A tuple of ``(completed_matches, counts)`` where
        ``completed_matches`` is the list of deserialized matches whose
        ``status == "completed"`` (in cache id order), and ``counts``
        is a dict with three int keys: ``total_cached`` (ids listed by
        the cache, before any skip), ``matches_skipped_invalid``
        (corrupt rows and rows whose cached data violates score
        validity) and ``matches_skipped_not_completed`` (live or
        upcoming matches).

    Raises:
        sqlite3.OperationalError: If the cache database cannot be
            opened or queried (propagated from
            :func:`scraper.cache.list_cached_match_ids` /
            :func:`scraper.cache.get_cached_match`). This is a real
            storage failure, not a per-row problem, so it is not
            swallowed.
    """
    match_ids = cache.list_cached_match_ids(db_path=db_path)
    completed: list[Match] = []
    counts = {
        "total_cached": len(match_ids),
        "matches_skipped_invalid": 0,
        "matches_skipped_not_completed": 0,
    }
    for match_id in match_ids:
        try:
            match = cache.get_cached_match(match_id, db_path=db_path)
        except IllegalScoreError as exc:
            counts["matches_skipped_invalid"] += 1
            logger.warning(
                "match %s has an illegal cached scoreline (%s); skipping",
                match_id,
                exc,
            )
            continue
        if match is None:
            counts["matches_skipped_invalid"] += 1
            logger.warning(
                "match %s has a corrupt or unreadable cache row; skipping",
                match_id,
            )
            continue
        if match.status != "completed":
            counts["matches_skipped_not_completed"] += 1
            logger.info("match %s is %s; skipping", match_id, match.status)
            continue
        completed.append(match)
    return completed, counts


def _is_finished_map(map_result: MapResult) -> bool:
    """Return whether a map result is finished (materialisable).

    The single definition of "finished" shared by
    :func:`_iter_finished_maps` and :func:`build_maps_table` so the two
    cannot drift apart on what counts as a materialisable map: a map is
    finished exactly when its ``winner`` is set — the same finished
    signal :meth:`MapResult.__post_init__` gates its validation on, and
    the one invariant that always holds for a completed match's maps in
    practice. Any future refinement to the finished-map definition is
    made here once and flows to both tables.

    Args:
        map_result: The :class:`MapResult` to test.

    Returns:
        ``True`` if ``map_result.winner`` is not ``None``, ``False``
        otherwise.

    Raises:
        Nothing.
    """
    return map_result.winner is not None


def _iter_finished_maps(match: Match) -> Iterator[tuple[int, MapResult]]:
    """Yield ``(map_index, MapResult)`` pairs for a match's finished maps.

    Iterates a match's maps using :func:`_is_finished_map` — the single
    shared finished-map definition also used by
    :func:`build_maps_table` — so the two tables cannot drift apart on
    what counts as a materialisable map. A map skipped here is skipped
    by the player-stats table too, so a player row is never orphaned
    against a map that does not exist in the ``maps`` table; the skip is
    counted and logged by :func:`build_maps_table` (the layer that
    defines what "finished" means), which is the one warning the two
    tables share.

    Args:
        match: The :class:`Match` whose maps to iterate.

    Returns:
        An iterator of ``(map_index, map_result)`` pairs, ``map_index``
        being the 0-based position of the map in ``match.maps``, for
        every map whose ``winner`` is not ``None``.

    Raises:
        Nothing.
    """
    for map_index, map_result in enumerate(match.maps):
        if not _is_finished_map(map_result):
            continue
        yield map_index, map_result


def build_matches_table(matches: list[Match]) -> pd.DataFrame:
    """Build the flat ``matches`` table from a list of matches.

    One row per match. The nested ``Team`` objects are flattened to
    ``team1_name``/``team1_id``/``team2_name``/``team2_id`` scalar
    columns (there is no separate teams table in the M8 spec), and the
    naive-UTC ``date`` is rendered as an ISO-8601 string (or ``None``)
    so pandas does not re-interpret the timezone. Columns follow
    ``MATCHES_COLUMNS`` in exactly that order.

    Args:
        matches: The list of :class:`Match` objects to flatten (already
            filtered to completed matches by
            :func:`load_completed_matches`; this function does not
            re-filter).

    Returns:
        A ``pandas.DataFrame`` with one row per match and columns
        ``match_id, url, event_name, date, team1_name, team1_id,
        team2_name, team2_id, team1_score, team2_score, best_of,
        status``. Missing optional values (e.g. ``date``, ``best_of``,
        a ``team_id``) materialize as ``NaN``.

    Raises:
        Nothing (pandas type-inference errors would surface as
            ``ValueError``/``TypeError`` and are not caught).
    """
    rows = []
    for match in matches:
        rows.append(
            {
                "match_id": match.match_id,
                "url": match.url,
                "event_name": match.event_name,
                "date": match.date.isoformat() if match.date is not None else None,
                "team1_name": match.team1.name,
                "team1_id": match.team1.team_id,
                "team2_name": match.team2.name,
                "team2_id": match.team2.team_id,
                "team1_score": match.team1_score,
                "team2_score": match.team2_score,
                "best_of": match.best_of,
                "status": match.status,
            }
        )
    return pd.DataFrame(rows, columns=MATCHES_COLUMNS)


def build_maps_table(matches: list[Match]) -> tuple[pd.DataFrame, int]:
    """Build the flat ``maps`` table from a list of matches.

    One row per *finished* map (see :func:`_iter_finished_maps`): a
    completed match's map with ``winner is None`` — the "completed but
    a child map isn't finished" case that ``Match.status`` does not
    rule out — is skipped, counted in the returned integer, and logged,
    rather than assumed away. The 0-based ``map_index`` column is the
    map's position in ``match.maps``; :func:`build_player_map_stats_table`
    assigns the same value to its rows so the two tables join on
    ``(match_id, map_index)``. The eight half-split columns are
    carried through unchanged: a finished map whose header parsed no
    half data legitimately has ``None``/``NaN`` there, which is a real
    expected value, not an error.

    Args:
        matches: The list of completed :class:`Match` objects to
            flatten.

    Returns:
        A tuple of ``(dataframe, maps_skipped_incomplete)`` where the
        dataframe has one row per finished map with columns
        ``match_id, map_index, map_name, team1_score, team2_score,
        winner, duration`` plus all eight
        ``team{1,2}_{first_half,second_half,atk,def}_rounds`` columns
        (``MATCHES_COLUMNS`` order), and ``maps_skipped_incomplete``
        is the number of maps skipped because ``winner`` was ``None``.

    Raises:
        Nothing (pandas type-inference errors would surface as
            ``ValueError``/``TypeError`` and are not caught).
    """
    rows = []
    maps_skipped_incomplete = 0
    for match in matches:
        for map_index, map_result in enumerate(match.maps):
            if not _is_finished_map(map_result):
                maps_skipped_incomplete += 1
                logger.warning(
                    "match %s map %d (%s) has no winner; skipping the "
                    "incomplete map (and any player stats attached to it)",
                    match.match_id,
                    map_index,
                    map_result.map_name,
                )
                continue
            rows.append(
                {
                    "match_id": match.match_id,
                    "map_index": map_index,
                    "map_name": map_result.map_name,
                    "team1_score": map_result.team1_score,
                    "team2_score": map_result.team2_score,
                    "winner": map_result.winner,
                    "duration": map_result.duration,
                    "team1_first_half_rounds": map_result.team1_first_half_rounds,
                    "team1_second_half_rounds": map_result.team1_second_half_rounds,
                    "team1_atk_rounds": map_result.team1_atk_rounds,
                    "team1_def_rounds": map_result.team1_def_rounds,
                    "team2_first_half_rounds": map_result.team2_first_half_rounds,
                    "team2_second_half_rounds": map_result.team2_second_half_rounds,
                    "team2_atk_rounds": map_result.team2_atk_rounds,
                    "team2_def_rounds": map_result.team2_def_rounds,
                }
            )
    return pd.DataFrame(rows, columns=MAPS_COLUMNS), maps_skipped_incomplete


def build_veto_actions_table(matches: list[Match]) -> pd.DataFrame:
    """Build the flat ``veto_actions`` table from a list of matches.

    One row per :class:`VetoAction`, in the order the actions appear
    within each match's ``veto_actions`` list (a match with no veto
    note contributes zero rows). Matches without any parsed veto
    actions are simply absent from this table — an empty veto table is
    a valid outcome for a cache whose matches predate veto parsing.

    Args:
        matches: The list of completed :class:`Match` objects to
            flatten.

    Returns:
        A ``pandas.DataFrame`` with one row per veto action and columns
        ``match_id, step_index, team, action, map_name``. ``team`` is
        ``None``/``NaN`` for a decider action (which is forced rather
        than chosen — see :class:`VetoAction`).

    Raises:
        Nothing (pandas type-inference errors would surface as
            ``ValueError``/``TypeError`` and are not caught).
    """
    rows = []
    for match in matches:
        for action in match.veto_actions:
            rows.append(
                {
                    "match_id": match.match_id,
                    "step_index": action.step_index,
                    "team": action.team,
                    "action": action.action,
                    "map_name": action.map_name,
                }
            )
    return pd.DataFrame(rows, columns=VETO_ACTIONS_COLUMNS)


def build_player_map_stats_table(matches: list[Match]) -> pd.DataFrame:
    """Build the flat ``player_map_stats`` table from a list of matches.

    One row per :class:`PlayerStats` entry, for finished maps only:
    the table iterates the exact same ``(map_index, map_result)``
    pairs :func:`build_maps_table` writes rows for (via
    :func:`_iter_finished_maps`), so a player row is never orphaned
    against a map that does not exist in the ``maps`` table — a stats
    row attached to a skipped-incomplete map is dropped alongside that
    map, covered by the single warning :func:`build_maps_table` logs.
    The ``agents`` list field is encoded as a JSON string
    (``agents_json``; empty list becomes ``"[]"``) per the module's
    flat-tables design rule — decodable with a plain ``json.loads`` by
    any downstream reader.

    Args:
        matches: The list of completed :class:`Match` objects to
            flatten.

    Returns:
        A ``pandas.DataFrame`` with one row per player-map stat line
        and columns ``match_id, map_index, player_name, team_name,
        rating, acs, kills, deaths, assists, adr, kast, hs_pct,
        first_kills, first_deaths, agents_json``. Optional numeric
        stats materialize as ``NaN``.

    Raises:
        Nothing (pandas type-inference errors would surface as
            ``ValueError``/``TypeError`` and are not caught).
    """
    rows = []
    for match in matches:
        for map_index, map_result in _iter_finished_maps(match):
            for player in map_result.player_stats:
                rows.append(
                    {
                        "match_id": match.match_id,
                        "map_index": map_index,
                        "player_name": player.player_name,
                        "team_name": player.team_name,
                        "rating": player.rating,
                        "acs": player.acs,
                        "kills": player.kills,
                        "deaths": player.deaths,
                        "assists": player.assists,
                        "adr": player.adr,
                        "kast": player.kast,
                        "hs_pct": player.hs_pct,
                        "first_kills": player.first_kills,
                        "first_deaths": player.first_deaths,
                        "agents_json": json.dumps(player.agents),
                    }
                )
    return pd.DataFrame(rows, columns=PLAYER_MAP_STATS_COLUMNS)


def build_sanity_report(
    tables: dict[str, pd.DataFrame],
    load_counts: dict[str, int],
    maps_skipped_incomplete: int,
) -> dict:
    """Build the JSON-serializable sanity report for a materialisation.

    Pure function (no disk I/O): computes the report's numbers from the
    four built tables and the loader's skip counters so it can be unit
    tested without touching disk. It may emit a ``logger.warning`` when
    a null-score map is excluded from the OT rate; that does not affect
    the returned dict. The OT rate is deliberately an ad
    hoc *report-only* heuristic, not a persisted column and not M9's
    formal outcome label: a finished map counts as OT when its winning
    score exceeds 13 rounds (``max(team1_score, team2_score) > 13``),
    which the task 002/007 score-validity invariant makes exact — a
    regulation win is always exactly 13, so ``> 13`` is precisely "went
    to overtime" for every row that reached the ``maps`` table. This
    heuristic is deliberately independent of ``labels.py``'s canonical
    OT criterion (``min(scores) >= 12``); the two agree on every legal
    scoreline, and that agreement is enforced by
    ``tests/test_labels.py::test_ot_heuristic_agrees_with_canonical_ot_criterion``
    rather than left as an unstated external invariant. A finished map
    with a winner but a null score (which can only bypass
    ``MapResult.__post_init__``'s validation, e.g. a hand-edited cache
    row) is warned about and excluded from the OT denominator, so it
    cannot silently deflate the rate. The heuristic is written to
    ``report.json`` only; nothing in the ``maps`` table encodes it.

    Note on the signature: the plan sketched this function as taking
    ``(matches_df, maps_df, ...)``, but its contract is per-table row
    counts, which need all four tables — it therefore takes the single
    ``tables`` dict :func:`main` already holds, so the four row counts
    cannot drift from the four files actually written.

    Args:
        tables: A dict mapping each table name (``"matches"``,
            ``"maps"``, ``"veto_actions"``, ``"player_map_stats"``) to
            the DataFrame that will be written under that name.
        load_counts: The counts dict returned by
            :func:`load_completed_matches` (keys ``total_cached``,
            ``matches_skipped_invalid``,
            ``matches_skipped_not_completed``).
        maps_skipped_incomplete: The number of maps
            :func:`build_maps_table` skipped because ``winner`` was
            ``None``.

    Returns:
        A JSON-serializable dict with keys ``row_counts`` (per-table
        row counts), ``total_cached``, ``matches_skipped_invalid``,
        ``matches_skipped_not_completed``, ``maps_skipped_incomplete``,
        ``map_count`` (row count of the ``maps`` table),
        ``maps_skipped_null_score`` (the number of finished maps
        excluded from ``ot_rate`` because ``team1_score`` or
        ``team2_score`` was null), ``ot_rate`` (a fraction in
        ``[0, 1]`` — ``0.083`` means 8.3% — or ``None`` when there is
        no classifiable map) and ``format_mix`` (dict
        mapping each ``best_of`` value to its match count, with the
        key ``"unknown"`` for matches whose ``best_of`` is ``None``).

    Raises:
        KeyError: If ``tables`` is missing any of the four expected
            table names, or ``load_counts`` is missing a documented
            key.
    """
    row_counts = {name: len(df) for name, df in tables.items()}
    map_count = row_counts["maps"]
    maps_df = tables["maps"]
    if map_count == 0:
        ot_rate = None
        maps_skipped_null_score = 0
    else:
        team1_score = maps_df["team1_score"]
        team2_score = maps_df["team2_score"]
        null_score = team1_score.isna() | team2_score.isna()
        maps_skipped_null_score = int(null_score.sum())
        if maps_skipped_null_score:
            logger.warning(
                "%d finished map(s) have a null team1_score/team2_score "
                "and are excluded from the OT rate; this bypasses "
                "MapResult.__post_init__'s validation and should not "
                "happen via the scraper path",
                maps_skipped_null_score,
            )
        ot_denominator = map_count - maps_skipped_null_score
        if ot_denominator == 0:
            ot_rate = None
        else:
            winner_scores = maps_df[["team1_score", "team2_score"]].max(axis=1)
            ot_count = int((winner_scores[~null_score] > 13).sum())
            ot_rate = ot_count / ot_denominator
    format_mix: dict[str, int] = {}
    for value in tables["matches"]["best_of"]:
        key = "unknown" if pd.isna(value) else str(value)
        format_mix[key] = format_mix.get(key, 0) + 1
    return {
        "row_counts": row_counts,
        "total_cached": load_counts["total_cached"],
        "matches_skipped_invalid": load_counts["matches_skipped_invalid"],
        "matches_skipped_not_completed": load_counts[
            "matches_skipped_not_completed"
        ],
        "maps_skipped_incomplete": maps_skipped_incomplete,
        "map_count": map_count,
        "maps_skipped_null_score": maps_skipped_null_score,
        "ot_rate": ot_rate,
        "format_mix": format_mix,
    }


def write_dataset(
    tables: dict[str, pd.DataFrame],
    report: dict,
    output_dir: Path,
) -> None:
    """Write the four Parquet tables and the sanity report to disk.

    Creates ``output_dir`` (including parents) if it does not exist,
    writes each of the four tables as ``<name>.parquet`` via
    :func:`table_io.write_parquet` (``index=False`` — the tables carry
    no meaningful row index, only their columns) in the fixed order
    ``matches, maps, veto_actions, player_map_stats``, and writes
    ``report.json`` (``json.dumps`` with indent and sorted keys) into
    the same directory. Overwrites any previous contents in place —
    re-materialising the same version replaces the files rather than
    erroring, matching task 008's idempotent re-run story.

    Args:
        tables: A dict mapping each table name to the DataFrame to
            write as ``<name>.parquet``.
        report: The JSON-serializable report dict (see
            :func:`build_sanity_report`) to write as ``report.json``.
        output_dir: The directory to write into (e.g.
            ``data/v1``); created with parents if missing.

    Returns:
        None.

    Raises:
        OSError: If ``output_dir`` cannot be created or a file cannot
            be written (e.g. permissions or disk errors).
        ValueError: If any table contains a value that cannot be
            serialized to Parquet (propagated from
            ``DataFrame.to_parquet``, e.g. an unserializable object
            type).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("matches", "maps", "veto_actions", "player_map_stats"):
        write_parquet(tables[name], output_dir / f"{name}.parquet")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset materialisation end to end.

    Logging is configured first so the loader's per-match skip warnings
    (a corrupt row, an illegal cached scoreline, a live/upcoming
    match) are visible from the CLI. All completed matches are then
    loaded from the cache (see :func:`load_completed_matches`), the
    four flat tables are built (see the ``build_*_table`` functions),
    the sanity report is computed (see :func:`build_sanity_report`),
    and everything is written under ``<output-dir>/<version>`` (see
    :func:`write_dataset`) — the write happens even when zero completed
    matches were found, so an empty run still produces schema-correct
    empty tables and a report that says so instead of nothing. A
    one-line summary of the materialised counts is logged at the end.

    Args:
        argv: The argument list to parse (see :func:`parse_args`);
            ``None`` means ``sys.argv[1:]``.

    Returns:
        ``0`` if the run wrote at least one completed match; ``1`` if
        the run completed mechanically but zero completed matches were
        found (tables and report are still written, possibly empty — an
        empty v1 dataset almost certainly means a bug upstream, so it
        must not look identical to a healthy run in an automation exit
        code).

    Raises:
        sqlite3.OperationalError: If the cache database cannot be
            opened or queried (propagated from
            :func:`load_completed_matches`).
        OSError / ValueError: If the output cannot be written
            (propagated from :func:`write_dataset`).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    matches, load_counts = load_completed_matches(db_path=args.db_path)
    maps_df, maps_skipped_incomplete = build_maps_table(matches)
    tables = {
        "matches": build_matches_table(matches),
        "maps": maps_df,
        "veto_actions": build_veto_actions_table(matches),
        "player_map_stats": build_player_map_stats_table(matches),
    }
    report = build_sanity_report(tables, load_counts, maps_skipped_incomplete)

    output_dir = Path(args.output_dir) / args.version
    write_dataset(tables, report, output_dir)

    ot_text = "n/a" if report["ot_rate"] is None else f"{report['ot_rate']:.1%}"
    logger.info(
        "wrote %s: %d matches, %d maps, %d veto actions, %d player-map "
        "stat rows, OT rate %s, format mix %s",
        output_dir,
        report["row_counts"]["matches"],
        report["map_count"],
        report["row_counts"]["veto_actions"],
        report["row_counts"]["player_map_stats"],
        ot_text,
        report["format_mix"],
    )
    if not matches:
        logger.warning(
            "zero completed matches materialised; an empty dataset almost "
            "certainly means a bug upstream (empty/cleared cache or a "
            "scrape that never ran), not a healthy zero-match run"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
