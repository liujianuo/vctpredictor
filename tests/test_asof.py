"""Tests for the point-in-time (as-of) feature access layer (M12).

The centerpiece is the leakage-safety proof: for a synthetic fixture
with rows dated exactly at, after, and before the query date (plus an
unseen team, a non-completed match, and null/unparseable dates), every
as-of entry point is asserted — by inspecting the returned rows' parsed
dates directly — to never contain a row dated ``>=`` the query date.
A second, skip-guarded test repeats the same no-leakage assertion at
real ``data/v1`` scale so a Parquet round-trip or dtype surprise the
tiny synthetic fixture cannot express still gets caught.
"""

from pathlib import Path

import pandas as pd
import pytest

from utils import asof

QUERY_DATE = "2026-01-03T10:00:00"


def _synthetic_tables():
    """Build a small synthetic matches+maps pair exercising every filter branch.

    The matches table contains, for the queried team ``"T1"``: two
    strictly-before completed matches (one as ``team1``, one as
    ``team2``), one match dated *exactly equal* to :data:`QUERY_DATE`,
    one match dated *after* it, one match that is not ``T1``'s at all,
    and one ``T1`` match dated before the cutoff but not completed. The
    maps table carries one finished map for each of the four
    ``T1``-involving matches (the not-completed and non-T1 matches have
    no map), plus one *unfinished* map (``winner=None``) attached to the
    strictly-before ``m001`` so the map-level completeness filter has a
    row to drop.

    Returns:
        A ``(matches_df, maps_df)`` tuple. ``matches_df`` has columns
        ``match_id, date, team1_id, team2_id, status``;
        ``maps_df`` has columns ``match_id, map_index, team1_score,
        team2_score, winner`` (the subset ``maps_as_of`` and the tests
        need — ``maps_as_of`` reads ``match_id`` and ``winner`` and
        carries the rest through untouched).

    Raises:
        Nothing.
    """
    matches_rows = [
        {
            "match_id": "m001",
            "date": "2026-01-01T10:00:00",
            "team1_id": "T1",
            "team2_id": "T2",
            "status": "completed",
        },
        {
            "match_id": "m002",
            "date": "2026-01-02T10:00:00",
            "team1_id": "T2",
            "team2_id": "T1",
            "status": "completed",
        },
        {
            "match_id": "m003",
            "date": QUERY_DATE,
            "team1_id": "T1",
            "team2_id": "T3",
            "status": "completed",
        },
        {
            "match_id": "m004",
            "date": "2026-01-04T10:00:00",
            "team1_id": "T1",
            "team2_id": "T2",
            "status": "completed",
        },
        {
            "match_id": "m005",
            "date": "2026-01-01T11:00:00",
            "team1_id": "T3",
            "team2_id": "T2",
            "status": "completed",
        },
        {
            "match_id": "m006",
            "date": "2025-12-31T10:00:00",
            "team1_id": "T1",
            "team2_id": "T2",
            "status": "upcoming",
        },
    ]
    matches_df = pd.DataFrame(
        matches_rows,
        columns=["match_id", "date", "team1_id", "team2_id", "status"],
    )
    maps_rows = [
        {"match_id": "m001", "map_index": 0, "team1_score": 13, "team2_score": 8, "winner": "team1"},
        {"match_id": "m001", "map_index": 1, "team1_score": 3, "team2_score": 5, "winner": None},
        {"match_id": "m002", "map_index": 0, "team1_score": 9, "team2_score": 13, "winner": "team2"},
        {"match_id": "m003", "map_index": 0, "team1_score": 10, "team2_score": 13, "winner": "team2"},
        {"match_id": "m004", "map_index": 0, "team1_score": 13, "team2_score": 5, "winner": "team1"},
    ]
    maps_df = pd.DataFrame(
        maps_rows,
        columns=["match_id", "map_index", "team1_score", "team2_score", "winner"],
    )
    return matches_df, maps_df


def test_matches_as_of_strict_lt_boundary():
    # The core match-level leakage proof: only the two strictly-before
    # completed T1 matches (m001/m002) survive; the exactly-equal
    # (m003) and after (m004) rows are excluded, and every returned
    # date is strictly < the query date.
    matches_df, _ = _synthetic_tables()
    result = asof.matches_as_of("T1", QUERY_DATE, matches_df)
    assert set(result["match_id"]) == {"m001", "m002"}
    query = pd.to_datetime(QUERY_DATE)
    assert (pd.to_datetime(result["date"]) < query).all()
    assert "m003" not in set(result["match_id"])
    assert "m004" not in set(result["match_id"])


def test_maps_as_of_strict_lt_boundary_and_orientation():
    # The map-level leakage proof plus the orientation flag: m001 (T1
    # is team1, score 13) and m002 (T1 is team2, score 13) survive,
    # the equal/after maps do not, and team_is_team1 points at the
    # correct score column on both sides.
    matches_df, maps_df = _synthetic_tables()
    result = asof.maps_as_of("T1", QUERY_DATE, matches_df, maps_df)
    assert set(result["match_id"]) == {"m001", "m002"}
    query = pd.to_datetime(QUERY_DATE)
    assert (pd.to_datetime(result["date"]) < query).all()
    row1 = result[result["match_id"] == "m001"].iloc[0]
    row2 = result[result["match_id"] == "m002"].iloc[0]
    assert bool(row1["team_is_team1"]) is True
    assert row1["team1_score"] == 13
    assert bool(row2["team_is_team1"]) is False
    assert row2["team2_score"] == 13
    assert result["team_is_team1"].dtype == bool


def test_features_as_of_bundle_excludes_equal_and_after_rows():
    # The roadmap-centered proof over the single public entry point:
    # the bundle's matches AND maps both exclude the exactly-equal
    # (m003) and after (m004) rows, with the team_id/date echoed back.
    matches_df, maps_df = _synthetic_tables()
    bundle = asof.features_as_of("T1", QUERY_DATE, matches_df, maps_df)
    assert bundle.team_id == "T1"
    assert bundle.date == QUERY_DATE
    query = pd.to_datetime(QUERY_DATE)
    for frame in (bundle.matches, bundle.maps):
        assert (pd.to_datetime(frame["date"]) < query).all()
        assert "m003" not in set(frame["match_id"])
        assert "m004" not in set(frame["match_id"])


def test_unseen_team_returns_empty_without_raising():
    # An unknown team is a normal, non-error case: empty matches and
    # maps, but the full column sets are preserved (including the
    # added date/orientation columns on the maps side).
    matches_df, maps_df = _synthetic_tables()
    bundle = asof.features_as_of("UNSEEN", QUERY_DATE, matches_df, maps_df)
    assert len(bundle.matches) == 0
    assert len(bundle.maps) == 0
    assert list(bundle.matches.columns) == list(matches_df.columns)
    assert list(bundle.maps.columns) == list(maps_df.columns) + [
        asof.DATE_COL,
        asof.TEAM_ORIENTATION_COL,
    ]
    assert bundle.maps["team_is_team1"].dtype == bool


def test_non_completed_matches_excluded():
    # A live/upcoming match dated before the cutoff must be excluded:
    # it carries no usable outcome signal for a feature.
    matches_df, _ = _synthetic_tables()
    result = asof.matches_as_of("T1", QUERY_DATE, matches_df)
    assert "m006" not in set(result["match_id"])
    assert (result["status"] == "completed").all()


def test_query_at_team_first_match_date_returns_empty():
    # Querying exactly at the team's very first match timestamp yields
    # an empty history (strict <), not an error and not that match.
    matches_df, maps_df = _synthetic_tables()
    bundle = asof.features_as_of("T1", "2026-01-01T10:00:00", matches_df, maps_df)
    assert len(bundle.matches) == 0
    assert len(bundle.maps) == 0


@pytest.mark.parametrize("null_value", [None, float("nan")])
def test_null_table_date_raises(null_value):
    # A null date in the table parses to NaT (not a pd.to_datetime
    # error) and must be rejected explicitly rather than silently
    # mis-ordered against the cutoff.
    matches_df, _ = _synthetic_tables()
    matches_df.loc[matches_df["match_id"] == "m001", "date"] = null_value
    with pytest.raises(ValueError, match="null"):
        asof.matches_as_of("T1", QUERY_DATE, matches_df)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_unparseable_table_date_raises():
    # An unparseable date propagates as a ValueError (the same
    # fail-loudly contract as drivers/splits) rather than being sorted
    # as a string.
    matches_df, _ = _synthetic_tables()
    matches_df.loc[matches_df["match_id"] == "m001", "date"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable"):
        asof.matches_as_of("T1", QUERY_DATE, matches_df)


@pytest.mark.parametrize("null_query", [None, float("nan")])
def test_null_query_date_raises(null_query):
    # A null cutoff has no chronological position and must raise, not
    # silently return everything (or nothing).
    matches_df, maps_df = _synthetic_tables()
    with pytest.raises(ValueError, match="null"):
        asof.features_as_of("T1", null_query, matches_df, maps_df)


def test_unparseable_query_date_raises():
    # An unparseable cutoff is rejected before any filtering happens.
    matches_df, maps_df = _synthetic_tables()
    with pytest.raises(ValueError, match="parseable"):
        asof.features_as_of("T1", "not-a-date", matches_df, maps_df)


def test_list_like_query_date_raises():
    # A list-like cutoff is an invalid type for the single-team as-of
    # primitive, so it raises TypeError rather than being half-matched.
    matches_df, maps_df = _synthetic_tables()
    with pytest.raises(TypeError, match="single timestamp"):
        asof.features_as_of(
            "T1", ["2026-01-01T10:00:00", "2026-01-02T10:00:00"], matches_df, maps_df
        )


def test_timezone_aware_query_date_raises():
    # The v1 date column is timezone-naive; a tz-aware cutoff cannot be
    # compared against it and is rejected with a clear message.
    matches_df, maps_df = _synthetic_tables()
    with pytest.raises(ValueError, match="timezone-aware"):
        asof.features_as_of("T1", "2026-01-03T10:00:00+00:00", matches_df, maps_df)


def test_matches_as_of_missing_column_raises():
    # A missing required column surfaces as KeyError naming it (same
    # contract as labels.py/splits.py), not a confusing pandas error.
    matches_df, _ = _synthetic_tables()
    for column in ["team1_id", "team2_id", "date", "status"]:
        df = matches_df.drop(columns=[column])
        with pytest.raises(KeyError, match=column):
            asof.matches_as_of("T1", QUERY_DATE, df)


def test_maps_as_of_missing_match_id_raises():
    # maps_df must carry the join key; its absence is a KeyError.
    matches_df, maps_df = _synthetic_tables()
    with pytest.raises(KeyError, match="match_id"):
        asof.maps_as_of("T1", QUERY_DATE, matches_df, maps_df.drop(columns=["match_id"]))


def test_maps_as_of_missing_winner_raises():
    # maps_df must carry the map-completion signal too; its absence is
    # a KeyError rather than a silent skip of the completeness filter.
    matches_df, maps_df = _synthetic_tables()
    with pytest.raises(KeyError, match="winner"):
        asof.maps_as_of("T1", QUERY_DATE, matches_df, maps_df.drop(columns=["winner"]))


def test_maps_as_of_drops_unfinished_map():
    # The map-level completeness filter: a winner-null map attached to
    # a completed, strictly-before match is dropped, while its finished
    # sibling map survives — so "completed rows only" holds at the map
    # level, not just the match level.
    matches_df, maps_df = _synthetic_tables()
    result = asof.maps_as_of("T1", QUERY_DATE, matches_df, maps_df)
    m001_maps = result[result["match_id"] == "m001"]
    assert set(m001_maps["map_index"]) == {0}
    assert (result["winner"].notna()).all()


def test_maps_as_of_duplicate_match_id_raises():
    # A duplicate match_id in the as-of-filtered matches would fan out
    # the maps join and silently duplicate map rows; the framework must
    # raise instead.
    matches_df, maps_df = _synthetic_tables()
    dup_row = matches_df[matches_df["match_id"] == "m001"].copy()
    matches_df = pd.concat([matches_df, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate match_id"):
        asof.maps_as_of("T1", QUERY_DATE, matches_df, maps_df)


def test_load_asof_tables_roundtrip(tmp_path):
    # Both Parquet files under tmp_path/<version> read back intact and
    # in the right order (matches first, then maps).
    matches_df, maps_df = _synthetic_tables()
    version_dir = tmp_path / "v1"
    version_dir.mkdir()
    matches_df.to_parquet(version_dir / "matches.parquet", index=False)
    maps_df.to_parquet(version_dir / "maps.parquet", index=False)
    loaded_matches, loaded_maps = asof.load_asof_tables("v1", output_dir=tmp_path)
    assert list(loaded_matches.columns) == list(matches_df.columns)
    assert list(loaded_maps.columns) == list(maps_df.columns)
    assert len(loaded_matches) == len(matches_df)
    assert len(loaded_maps) == len(maps_df)


def test_load_asof_tables_missing_raises(tmp_path):
    # A missing maps.parquet (materialize.py never ran for this version)
    # surfaces as FileNotFoundError — a clear "run materialize.py first"
    # signal.
    with pytest.raises(FileNotFoundError):
        asof.load_asof_tables("v1", output_dir=tmp_path)


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_no_leakage_at_v1_scale():
    # The same no-leakage assertion at real v1 scale: pick the most
    # frequently appearing team, query just after the dataset's latest
    # date, and verify no returned match/map date is >= the cutoff, the
    # team filter and completed filter held, and the orientation flag
    # agrees with the matches table's own team1_id column.
    matches_df, maps_df = asof.load_asof_tables("v1")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    latest = pd.to_datetime(matches_df["date"]).max()
    query = (latest + pd.Timedelta(hours=1)).isoformat()

    bundle = asof.features_as_of(team_id, query, matches_df, maps_df)

    assert len(bundle.matches) > 0
    query_ts = pd.to_datetime(query)
    for frame in (bundle.matches, bundle.maps):
        parsed = pd.to_datetime(frame["date"])
        assert not parsed.isna().any()
        assert (parsed < query_ts).all()
    assert (
        (bundle.matches["team1_id"] == team_id)
        | (bundle.matches["team2_id"] == team_id)
    ).all()
    assert (bundle.matches["status"] == "completed").all()
    check = bundle.maps.merge(
        bundle.matches[["match_id", "team1_id"]], on="match_id", how="inner"
    )
    assert (check["team_is_team1"] == (check["team1_id"] == team_id)).all()


@pytest.mark.skipif(
    not (
        Path("data/v1/matches.parquet").exists()
        and Path("data/v1/maps.parquet").exists()
    ),
    reason="materialised v1 dataset not present (run materialize.py first)",
)
def test_real_data_strict_boundary_excludes_equal_date():
    # The strict-< boundary exercised at real v1 scale, not just on the
    # synthetic fixture: query at a real team's latest match timestamp
    # exactly and prove that match (date == query) is excluded while its
    # strictly-earlier history is still returned. This would fail if the
    # implementation regressed to <= at real-data scale.
    matches_df, maps_df = asof.load_asof_tables("v1")
    appearances = pd.concat(
        [matches_df["team1_id"], matches_df["team2_id"]]
    ).dropna()
    team_id = appearances.value_counts().idxmax()
    team_matches = matches_df[
        (matches_df["team1_id"] == team_id) | (matches_df["team2_id"] == team_id)
    ].copy()
    team_matches["_parsed"] = pd.to_datetime(team_matches["date"])
    team_matches = team_matches.sort_values("_parsed")
    assert team_matches["_parsed"].nunique() >= 2

    cutoff_row = team_matches.iloc[-1]
    query = str(cutoff_row["date"])
    excluded_id = cutoff_row["match_id"]

    bundle = asof.features_as_of(team_id, query, matches_df, maps_df)

    assert len(bundle.matches) > 0
    assert excluded_id not in set(bundle.matches["match_id"])
    query_ts = pd.to_datetime(query)
    for frame in (bundle.matches, bundle.maps):
        assert (pd.to_datetime(frame["date"]) < query_ts).all()
