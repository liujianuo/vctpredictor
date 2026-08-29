"""Tests for materialize (roadmap M8): v1 dataset materialisation.

Follows the tests/test_scrape.py pattern of testing a root-level
module: small in-memory Match/MapResult/VetoAction/PlayerStats fixtures
built directly (not from real HTML), a temp SQLite cache via
--db-path, and a temp output dir via --output-dir/--version. No live
network, no real cache/data directories.
"""

import json
import logging
from datetime import datetime

import pandas as pd
import pytest

import materialize
from scraper import cache
from scraper.models import MapResult, Match, PlayerStats, Team, VetoAction

# Sentinel distinguishing "caller did not pass date" (use the fixed
# fixture date) from "caller explicitly wants date=None" (a dateless
# match) — the dateless-match test needs to force None while every
# other caller wants the fixed date without spelling it out.
_UNSET = object()
# Match.date is documented as naive UTC (scraper.models), so the
# fixture mirrors that rather than attaching a tzinfo; DTZ001 is
# deliberately suppressed for the same reason test_cache.py's make_match
# fixture builds a naive datetime.
_FIXED_DATE = datetime(2026, 7, 15, 11, 0, 0)  # noqa: DTZ001


def _make_match(
    match_id="1",
    status="completed",
    date=_UNSET,
    scores=None,
    unfinished=False,
    veto=False,
    player_stats=False,
):
    """Build a small Match fixture for materialisation tests.

    Args:
        match_id: The match id to use (also becomes the URL suffix).
        status: The match's ``status`` field ("completed", "live" or
            "upcoming").
        date: The match's naive-UTC ``date``. The default (the
            module-level ``_UNSET`` sentinel) means "use the fixed
            2026-07-15 11:00 date"; pass ``None`` explicitly to build
            a dateless match.
        scores: List of ``(team1_score, team2_score)`` tuples; one
            finished :class:`MapResult` is built per tuple (winner
            derived from the higher score).
        unfinished: When ``True``, append a fully-absent map (all
            scores and winner ``None``) after the finished maps — the
            only "unfinished" MapResult that can be constructed without
            raising (a partially-filled one fails
            ``MapResult.__post_init__`` itself).
        veto: When ``True``, attach a two-action veto sequence (a ban
            and a decider) to the match.
        player_stats: When ``True``, attach two :class:`PlayerStats`
            rows to every map in ``maps`` (including the unfinished
            one, so tests can assert unfinished-map stats are dropped).

    Returns:
        The constructed :class:`Match`.

    Raises:
        Nothing.
    """
    maps = []
    for s1, s2 in scores or []:
        winner = "Alpha" if s1 > s2 else "Beta"
        maps.append(
            MapResult(
                map_name=f"Map{len(maps)}",
                team1_score=s1,
                team2_score=s2,
                winner=winner,
            )
        )
    if unfinished:
        maps.append(
            MapResult(
                map_name="Unfinished",
                team1_score=None,
                team2_score=None,
                winner=None,
            )
        )
    if player_stats:
        for m in maps:
            m.player_stats = [
                PlayerStats(
                    player_name="player-a",
                    team_name="Alpha",
                    rating=1.2,
                    kills=20,
                    deaths=10,
                    assists=5,
                    agents=["Jett"],
                ),
                PlayerStats(
                    player_name="player-b",
                    team_name="Beta",
                    rating=0.9,
                    kills=12,
                    deaths=18,
                    assists=7,
                    agents=["Omen", "Kayo"],
                ),
            ]
    veto_actions = []
    if veto:
        veto_actions = [
            VetoAction(step_index=0, team="Alpha", action="ban", map_name="Split"),
            VetoAction(step_index=4, team=None, action="decider", map_name="Haven"),
        ]
    return Match(
        match_id=match_id,
        url=f"https://www.vlr.gg/{match_id}/alpha-vs-beta",
        event_name="Test Event",
        date=_FIXED_DATE if date is _UNSET else date,
        team1=Team(name="Alpha", team_id="1"),
        team2=Team(name="Beta", team_id="2"),
        team1_score=2 if scores else None,
        team2_score=0 if scores else None,
        best_of="Bo3",
        maps=maps,
        veto_actions=veto_actions,
        status=status,
    )


def _seed_illegal_row(db, match_id="2"):
    """Insert a raw cache row whose data fails score validity.

    Mirrors the seeding technique in tests/test_cache.py: a finished
    map at 13-12 with a declared winner is an illegal OT scoreline
    (winning margin below 2), so ``MapResult.__post_init__`` raises
    :class:`IllegalScoreError` when the row is deserialized. This is
    the "genuine data problem" kind of bad row, distinct from a
    corrupt row (bad JSON), and :func:`materialize.load_completed_matches`
    must skip and count it rather than aborting the whole run.

    Args:
        db: Path to the temp cache database.
        match_id: The id to store the illegal row under.

    Returns:
        Nothing; the row is committed to ``db``.

    Raises:
        Nothing (SQLite errors would propagate).
    """
    conn = cache.get_connection(db)
    try:
        conn.execute(
            "INSERT INTO matches (match_id, url, data, cached_at) VALUES (?, ?, ?, ?)",
            (
                match_id,
                "https://x",
                json.dumps(
                    {
                        "match_id": match_id,
                        "url": "https://x",
                        "event_name": "Test Event",
                        "date": None,
                        "team1": {"name": "Alpha", "team_id": "1"},
                        "team2": {"name": "Beta", "team_id": "2"},
                        "team1_score": 1,
                        "team2_score": 0,
                        "best_of": "Bo3",
                        "maps": [
                            {
                                "map_name": "Ascent",
                                "team1_score": 13,
                                "team2_score": 12,
                                "winner": "Alpha",
                                "duration": "41:10",
                            }
                        ],
                        "status": "completed",
                    }
                ),
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    # Defaults: version v1, output dir "data", db path None (meaning
    # scraper.cache's own default).
    args = materialize.parse_args([])
    assert args.version == "v1"
    assert args.output_dir == "data"
    assert args.db_path is None


def test_parse_args_overrides():
    # Each flag overrides its default and is passed through verbatim.
    args = materialize.parse_args(
        ["--version", "v2", "--output-dir", "/tmp/out", "--db-path", "/tmp/c.sqlite3"]
    )
    assert args.version == "v2"
    assert args.output_dir == "/tmp/out"
    assert args.db_path == "/tmp/c.sqlite3"


# --------------------------------------------------------------------------
# load_completed_matches
# --------------------------------------------------------------------------


def test_load_completed_matches_filters_statuses(tmp_path):
    # Only completed matches survive; live/upcoming matches are counted
    # and skipped, not materialised (they carry partial or absent
    # scores and are not training rows).
    db = tmp_path / "c.sqlite3"
    cache.set_cached_match(_make_match("1", status="completed"), db_path=db)
    cache.set_cached_match(_make_match("2", status="live"), db_path=db)
    cache.set_cached_match(_make_match("3", status="upcoming"), db_path=db)
    matches, counts = materialize.load_completed_matches(db_path=db)
    assert [m.match_id for m in matches] == ["1"]
    assert counts == {
        "total_cached": 3,
        "matches_skipped_invalid": 0,
        "matches_skipped_not_completed": 2,
    }


def test_load_completed_matches_skips_illegal_row(tmp_path):
    # A cached row that deserializes to an illegal scoreline raises
    # IllegalScoreError out of get_cached_match; the loader must catch
    # it, count it as skipped-invalid, and keep loading the rest rather
    # than aborting the whole materialisation.
    db = tmp_path / "c.sqlite3"
    cache.set_cached_match(_make_match("1", status="completed"), db_path=db)
    _seed_illegal_row(db, match_id="2")
    matches, counts = materialize.load_completed_matches(db_path=db)
    assert [m.match_id for m in matches] == ["1"]
    assert counts == {
        "total_cached": 2,
        "matches_skipped_invalid": 1,
        "matches_skipped_not_completed": 0,
    }


def test_load_completed_matches_skips_corrupt_row(tmp_path):
    # A corrupt row (unparseable JSON) returns None from
    # get_cached_match; the loader counts it as skipped-invalid rather
    # than treating it as a completed match.
    db = tmp_path / "c.sqlite3"
    cache.set_cached_match(_make_match("1", status="completed"), db_path=db)
    conn = cache.get_connection(db)
    try:
        conn.execute(
            "INSERT INTO matches (match_id, url, data, cached_at) VALUES (?, ?, ?, ?)",
            ("2", "https://x", "{not json", "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    matches, counts = materialize.load_completed_matches(db_path=db)
    assert [m.match_id for m in matches] == ["1"]
    assert counts == {
        "total_cached": 2,
        "matches_skipped_invalid": 1,
        "matches_skipped_not_completed": 0,
    }


def test_load_completed_matches_empty_cache(tmp_path):
    # An empty cache yields no matches and zero skips, not an error.
    db = tmp_path / "c.sqlite3"
    matches, counts = materialize.load_completed_matches(db_path=db)
    assert matches == []
    assert counts == {
        "total_cached": 0,
        "matches_skipped_invalid": 0,
        "matches_skipped_not_completed": 0,
    }


# --------------------------------------------------------------------------
# build_matches_table
# --------------------------------------------------------------------------


def test_build_matches_table_columns_and_values():
    # One row per match, fixed column order, team objects flattened to
    # scalar columns, date rendered as an ISO string.
    df = materialize.build_matches_table([_make_match("1", scores=[(13, 11)])])
    assert list(df.columns) == list(materialize.MATCHES_COLUMNS)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["match_id"] == "1"
    assert row["url"] == "https://www.vlr.gg/1/alpha-vs-beta"
    assert row["event_name"] == "Test Event"
    assert row["date"] == "2026-07-15T11:00:00"
    assert row["team1_name"] == "Alpha"
    assert row["team1_id"] == "1"
    assert row["team2_name"] == "Beta"
    assert row["team1_score"] == 2
    assert row["best_of"] == "Bo3"
    assert row["status"] == "completed"


def test_build_matches_table_dateless_match():
    # A match with date=None materializes its date column as NaN, not
    # a fabricated timestamp.
    df = materialize.build_matches_table([_make_match("1", date=None)])
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["date"])


# --------------------------------------------------------------------------
# build_maps_table
# --------------------------------------------------------------------------


def test_build_maps_table_map_index_and_skip():
    # One row per finished map; map_index is the 0-based position in
    # match.maps; a fully-absent map (winner None) is excluded and
    # counted in the returned skip integer.
    match = _make_match("1", scores=[(13, 11), (16, 14)], unfinished=True)
    df, skipped = materialize.build_maps_table([match])
    assert skipped == 1
    assert len(df) == 2
    assert list(df.columns) == list(materialize.MAPS_COLUMNS)
    assert list(df["match_id"]) == ["1", "1"]
    assert list(df["map_index"]) == [0, 1]
    assert list(df["map_name"]) == ["Map0", "Map1"]
    # (13, 11) -> Alpha wins; (16, 14) -> Alpha wins (16 > 14).
    assert list(df["winner"]) == ["Alpha", "Alpha"]


def test_build_maps_table_half_split_columns_carried_through():
    # The eight half-split columns are present; a finished map with no
    # parsed half data legitimately has NaN there.
    df, skipped = materialize.build_maps_table([_make_match("1", scores=[(13, 6)])])
    assert skipped == 0
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["team1_first_half_rounds"])
    assert pd.isna(df.iloc[0]["team2_def_rounds"])


def test_is_finished_map_predicate():
    # The shared finished-map predicate (review finding: build_maps_table
    # used to inline ``winner is None`` instead of sharing the
    # definition with build_player_map_stats_table). winner set ->
    # finished; winner None -> not finished, so both tables funnel
    # through one definition that cannot drift.
    finished = MapResult(
        map_name="Ascent", team1_score=13, team2_score=11, winner="Alpha"
    )
    unfinished = MapResult(
        map_name="Ascent", team1_score=None, team2_score=None, winner=None
    )
    assert materialize._is_finished_map(finished) is True
    assert materialize._is_finished_map(unfinished) is False


# --------------------------------------------------------------------------
# build_veto_actions_table
# --------------------------------------------------------------------------


def test_build_veto_actions_table_rows_and_values():
    # One row per veto action, only for matches that have one; a match
    # with no veto note contributes zero rows.
    m1 = _make_match("1", veto=True)
    m2 = _make_match("2")
    df = materialize.build_veto_actions_table([m1, m2])
    assert list(df.columns) == list(materialize.VETO_ACTIONS_COLUMNS)
    assert len(df) == 2
    assert list(df["match_id"]) == ["1", "1"]
    assert list(df["step_index"]) == [0, 4]
    assert list(df["action"]) == ["ban", "decider"]
    # Decider actions have no acting team.
    assert pd.isna(df.iloc[1]["team"])


# --------------------------------------------------------------------------
# build_player_map_stats_table
# --------------------------------------------------------------------------


def test_build_player_map_stats_join_and_agents_json():
    # One row per player per finished map; (match_id, map_index) aligns
    # with the maps table; agents round-trips through agents_json.
    match = _make_match("1", scores=[(13, 11), (16, 14)], player_stats=True)
    df = materialize.build_player_map_stats_table([match])
    assert list(df.columns) == list(materialize.PLAYER_MAP_STATS_COLUMNS)
    assert len(df) == 4  # 2 players x 2 finished maps
    assert list(df["match_id"]) == ["1", "1", "1", "1"]
    assert list(df["map_index"]) == [0, 0, 1, 1]
    assert json.loads(df.iloc[0]["agents_json"]) == ["Jett"]
    assert json.loads(df.iloc[1]["agents_json"]) == ["Omen", "Kayo"]
    assert df.iloc[0]["kills"] == 20


def test_build_player_map_stats_drops_unfinished_map_stats():
    # Stats attached to an unfinished map are dropped alongside that
    # map (the maps table has no row for it, so the FK cannot dangle):
    # player rows exist only for finished maps.
    match = _make_match(
        "1", scores=[(13, 11)], unfinished=True, player_stats=True
    )
    df = materialize.build_player_map_stats_table([match])
    assert len(df) == 2  # only the finished map's two players
    assert set(df["map_index"]) == {0}


# --------------------------------------------------------------------------
# build_sanity_report
# --------------------------------------------------------------------------


def test_build_sanity_report_ot_rate_and_format_mix():
    # ot_rate is the fraction of finished maps whose winning score
    # exceeds 13 (16-14 is OT; 13-11 and 13-6 are regulation);
    # format_mix counts best_of values with an "unknown" bucket for
    # None.
    tables = {
        "matches": pd.DataFrame(
            [{"best_of": "Bo3"}, {"best_of": "Bo5"}, {"best_of": None}],
            columns=["best_of"],
        ),
        "maps": pd.DataFrame(
            [
                {"team1_score": 13, "team2_score": 11},
                {"team1_score": 16, "team2_score": 14},
                {"team1_score": 13, "team2_score": 6},
            ],
            columns=["team1_score", "team2_score"],
        ),
        "veto_actions": pd.DataFrame(columns=materialize.VETO_ACTIONS_COLUMNS),
        "player_map_stats": pd.DataFrame(
            columns=materialize.PLAYER_MAP_STATS_COLUMNS
        ),
    }
    report = materialize.build_sanity_report(
        tables,
        {
            "total_cached": 4,
            "matches_skipped_invalid": 1,
            "matches_skipped_not_completed": 0,
        },
        maps_skipped_incomplete=2,
    )
    assert report["row_counts"] == {
        "matches": 3,
        "maps": 3,
        "veto_actions": 0,
        "player_map_stats": 0,
    }
    assert report["map_count"] == 3
    assert report["ot_rate"] == pytest.approx(1 / 3)
    assert report["format_mix"] == {"Bo3": 1, "Bo5": 1, "unknown": 1}
    assert report["total_cached"] == 4
    assert report["matches_skipped_invalid"] == 1
    assert report["matches_skipped_not_completed"] == 0
    assert report["maps_skipped_incomplete"] == 2


def test_build_sanity_report_zero_maps_ot_rate_none():
    # With no finished maps the OT rate is None (not a
    # ZeroDivisionError), the format mix is empty, and row counts are
    # all zero.
    tables = {
        "matches": pd.DataFrame(columns=["best_of"]),
        "maps": pd.DataFrame(columns=["team1_score", "team2_score"]),
        "veto_actions": pd.DataFrame(columns=materialize.VETO_ACTIONS_COLUMNS),
        "player_map_stats": pd.DataFrame(
            columns=materialize.PLAYER_MAP_STATS_COLUMNS
        ),
    }
    report = materialize.build_sanity_report(
        tables,
        {
            "total_cached": 0,
            "matches_skipped_invalid": 0,
            "matches_skipped_not_completed": 0,
        },
        maps_skipped_incomplete=0,
    )
    assert report["map_count"] == 0
    assert report["ot_rate"] is None
    assert report["format_mix"] == {}
    assert report["row_counts"]["matches"] == 0


def test_build_sanity_report_null_score_map_excluded_from_ot_rate(caplog):
    # A finished map with a winner but a null score (which bypasses
    # MapResult.__post_init__'s validation) must be warned about and
    # excluded from the OT denominator, not silently deflate ot_rate
    # while still counting toward map_count.
    tables = {
        "matches": pd.DataFrame([{"best_of": "Bo3"}], columns=["best_of"]),
        "maps": pd.DataFrame(
            [
                {"team1_score": 16, "team2_score": 14},
                {"team1_score": None, "team2_score": 11},
            ],
            columns=["team1_score", "team2_score"],
        ),
        "veto_actions": pd.DataFrame(columns=materialize.VETO_ACTIONS_COLUMNS),
        "player_map_stats": pd.DataFrame(
            columns=materialize.PLAYER_MAP_STATS_COLUMNS
        ),
    }
    with caplog.at_level(logging.WARNING):
        report = materialize.build_sanity_report(
            tables,
            {
                "total_cached": 1,
                "matches_skipped_invalid": 0,
                "matches_skipped_not_completed": 0,
            },
            maps_skipped_incomplete=0,
        )
    assert report["map_count"] == 2
    assert report["maps_skipped_null_score"] == 1
    # Only the (16, 14) map is classifiable; it is OT, so 1/1.
    assert report["ot_rate"] == pytest.approx(1.0)
    assert "null" in caplog.text


# --------------------------------------------------------------------------
# main — end to end
# --------------------------------------------------------------------------


def test_main_end_to_end(tmp_path, caplog):
    # Seeding two completed matches (with maps, veto actions and player
    # stats) plus one live match, materialising to a temp output dir:
    # the four parquet files and report.json appear under
    # output_dir/version/, row counts round-trip, the report's counts
    # are internally consistent, the live match is absent from every
    # table, and main returns 0.
    caplog.set_level(logging.INFO)
    db = tmp_path / "cache.sqlite3"
    outdir = tmp_path / "out"
    cache.set_cached_match(
        _make_match("1", scores=[(13, 11)], veto=True, player_stats=True),
        db_path=db,
    )
    cache.set_cached_match(
        _make_match("2", scores=[(13, 6), (16, 14)], player_stats=True),
        db_path=db,
    )
    cache.set_cached_match(_make_match("3", status="live"), db_path=db)

    rc = materialize.main(
        ["--db-path", str(db), "--output-dir", str(outdir), "--version", "v1"]
    )
    assert rc == 0

    version_dir = outdir / "v1"
    for name in ("matches", "maps", "veto_actions", "player_map_stats"):
        assert (version_dir / f"{name}.parquet").exists()
    assert (version_dir / "report.json").exists()

    matches_df = pd.read_parquet(version_dir / "matches.parquet")
    maps_df = pd.read_parquet(version_dir / "maps.parquet")
    veto_df = pd.read_parquet(version_dir / "veto_actions.parquet")
    player_df = pd.read_parquet(version_dir / "player_map_stats.parquet")

    assert len(matches_df) == 2
    assert set(matches_df["match_id"]) == {"1", "2"}
    assert len(maps_df) == 3  # 1 finished map + 2 finished maps
    assert set(maps_df["match_id"]) == {"1", "2"}
    assert len(veto_df) == 2  # match 1's two veto actions
    assert len(player_df) == 6  # 2 players x (1 + 2) finished maps

    report = json.loads((version_dir / "report.json").read_text())
    assert report["row_counts"] == {
        "matches": 2,
        "maps": 3,
        "veto_actions": 2,
        "player_map_stats": 6,
    }
    # One of the three maps is OT (16-14).
    assert report["ot_rate"] == pytest.approx(1 / 3)
    assert report["format_mix"] == {"Bo3": 2}
    assert report["total_cached"] == 3
    assert report["matches_skipped_not_completed"] == 1
    assert "wrote" in caplog.text


def test_main_zero_completed_matches_returns_1(tmp_path):
    # With only a live match cached, the run still completes
    # mechanically — schema-correct empty parquet files and a report
    # are written — but returns 1 so an empty v1 dataset never looks
    # like a healthy run in an automation exit code.
    db = tmp_path / "cache.sqlite3"
    outdir = tmp_path / "out"
    cache.set_cached_match(_make_match("3", status="live"), db_path=db)

    rc = materialize.main(
        ["--db-path", str(db), "--output-dir", str(outdir), "--version", "v1"]
    )
    assert rc == 1

    version_dir = outdir / "v1"
    expected_columns = {
        "matches": materialize.MATCHES_COLUMNS,
        "maps": materialize.MAPS_COLUMNS,
        "veto_actions": materialize.VETO_ACTIONS_COLUMNS,
        "player_map_stats": materialize.PLAYER_MAP_STATS_COLUMNS,
    }
    for name, columns in expected_columns.items():
        df = pd.read_parquet(version_dir / f"{name}.parquet")
        assert len(df) == 0
        assert list(df.columns) == list(columns)
    report = json.loads((version_dir / "report.json").read_text())
    assert report["row_counts"]["matches"] == 0
    assert report["ot_rate"] is None
