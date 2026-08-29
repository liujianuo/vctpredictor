"""Tests for config (map pool + era, roadmap M0).

Follows the style of tests/test_cache.py: pytest, tmp_path, no live
network.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from utils import config
from utils.config import Config, ConfigError, Era, load_config, normalize_map_name

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "config.json"

EXPECTED_ACTIVE_POOL = ("Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def base_config_data():
    """A minimal valid config dict (two non-overlapping eras)."""
    return {
        "region": "emea",
        "active_era": "2026-abyss",
        "eras": [
            {
                "name": "2026-s1",
                "start": "2026-04-01",
                "end": "2026-07-15",
                "map_pool": ["Ascent", "Breeze", "Fracture", "Haven", "Lotus", "Pearl", "Split"],
            },
            {
                "name": "2026-abyss",
                "start": "2026-07-15",
                "end": None,
                "map_pool": ["Abyss", "Ascent", "Haven", "Lotus", "Split", "Summit", "Sunset"],
            },
        ],
        "event_urls": [
            "https://www.vlr.gg/event/matches/2976/vct-2026-emea-stage-2/?group=completed",
        ],
    }


def write_json(tmp_path, data, filename="config.json"):
    path = tmp_path / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The real config.json
# --------------------------------------------------------------------------


def test_real_config_loads():
    cfg = load_config()
    assert cfg.region == "emea"
    assert cfg.active_era.name == "2026-abyss"
    assert cfg.active_era.map_pool == EXPECTED_ACTIVE_POOL
    assert len(cfg.event_urls) == 2
    assert all(u.startswith("https://www.vlr.gg/event/") for u in cfg.event_urls)


def test_module_level_active_matches_real_config():
    assert config.ACTIVE.region == "emea"
    assert config.ACTIVE.active_era.map_pool == EXPECTED_ACTIVE_POOL
    assert config.ACTIVE.active_era.name == "2026-abyss"


def test_active_is_lazy_and_cached(monkeypatch):
    # ACTIVE must not be computed at import time: a corrupt config.json
    # would otherwise abort `import config` and break collection of this
    # very test file. It should load once, on first access, then cache.
    calls = []
    real_load = config.load_config

    def counting(*args, **kwargs):
        calls.append(1)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(config, "load_config", counting)
    monkeypatch.setattr(config, "_ACTIVE", None)

    assert config.ACTIVE.region == "emea"
    assert len(calls) == 1
    assert config.ACTIVE.region == "emea"
    assert len(calls) == 1


def test_real_config_has_four_eras_and_rotation():
    # The v1 window (Stage 1 + Stage 2) straddles real pool rotations;
    # each era must exist so every match date resolves to its pool.
    cfg = load_config()
    assert len(cfg.eras) == 4
    pools = {e.name: e.map_pool for e in cfg.eras}
    assert pools["2026-s1-bind"] == ("Bind", "Breeze", "Fracture", "Haven", "Lotus", "Pearl", "Split")
    assert pools["2026-s1-ascent"] == ("Ascent", "Breeze", "Fracture", "Haven", "Lotus", "Pearl", "Split")
    assert pools["2026-s2-breeze"] == ("Ascent", "Breeze", "Haven", "Lotus", "Split", "Summit", "Sunset")
    assert pools["2026-abyss"] == EXPECTED_ACTIVE_POOL


# --------------------------------------------------------------------------
# normalize_map_name
# --------------------------------------------------------------------------


def test_normalize_map_name_case_and_whitespace():
    assert normalize_map_name("sunset") == "Sunset"
    assert normalize_map_name("SUNSET") == "Sunset"
    assert normalize_map_name("  abyss ") == "Abyss"
    assert normalize_map_name("  sPlIt  ") == "Split"
    assert normalize_map_name("\tpearl\n") == "Pearl"


# --------------------------------------------------------------------------
# is_active_map / contains_map
# --------------------------------------------------------------------------


def test_is_active_map():
    cfg = load_config()
    assert cfg.is_active_map("Breeze") is False
    assert cfg.is_active_map("abyss") is True
    assert cfg.is_active_map("ABYSS") is True
    assert cfg.is_active_map(" sunset ") is True


def test_contains_map_normalises():
    era = Era(name="e", start=date(2026, 1, 1), end=None, map_pool=("Sunset", "Abyss"))
    assert era.contains_map("sunset") is True
    assert era.contains_map("  ABYSS ") is True
    assert era.contains_map("Breeze") is False


# --------------------------------------------------------------------------
# era_as_of
# --------------------------------------------------------------------------


def test_era_as_of_inside_window():
    cfg = load_config()
    assert cfg.era_as_of(date(2026, 4, 1)).name == "2026-s1-bind"
    assert cfg.era_as_of(date(2026, 5, 1)).name == "2026-s1-bind"
    assert cfg.era_as_of(date(2026, 5, 7)).name == "2026-s1-ascent"
    assert cfg.era_as_of(date(2026, 7, 15)).name == "2026-s2-breeze"
    assert cfg.era_as_of(date(2026, 8, 16)).name == "2026-s2-breeze"
    assert cfg.era_as_of(date(2026, 8, 17)).name == "2026-abyss"
    assert cfg.era_as_of(date(2026, 8, 23)).name == "2026-abyss"
    assert cfg.era_as_of(date(2026, 12, 31)).name == "2026-abyss"


def test_era_as_of_raises_outside_window():
    cfg = load_config()
    with pytest.raises(ConfigError):
        cfg.era_as_of(date(2026, 1, 1))
    with pytest.raises(ConfigError):
        cfg.era_as_of(date(2025, 12, 31))


def test_era_as_of_resolves_fixture_date_to_breeze_pool():
    # The Stage 2 W1 fixture match is dated 2026-07-15 and its veto log
    # names a Breeze pool (no Abyss). era_as_of must agree with the fixture.
    cfg = load_config()
    era = cfg.era_as_of(date(2026, 7, 15))
    assert era.contains_map("Breeze") is True
    assert era.contains_map("Abyss") is False


def test_era_as_of_accepts_datetime():
    from datetime import datetime

    cfg = load_config()
    assert cfg.era_as_of(datetime(2026, 8, 23, 11, 0, 0)).name == "2026-abyss"


# --------------------------------------------------------------------------
# Validation rules — one ConfigError case per rule (temp JSON files)
# --------------------------------------------------------------------------


def test_error_missing_required_key(tmp_path):
    data = base_config_data()
    del data["region"]
    with pytest.raises(ConfigError, match="region"):
        load_config(write_json(tmp_path, data))


def test_error_eras_empty(tmp_path):
    data = base_config_data()
    data["eras"] = []
    with pytest.raises(ConfigError, match="eras"):
        load_config(write_json(tmp_path, data))


def test_error_duplicate_map_after_normalisation(tmp_path):
    data = base_config_data()
    data["eras"][1]["map_pool"] = ["Abyss", "abyss", "Ascent", "Haven", "Lotus", "Split", "Summit"]
    with pytest.raises(ConfigError, match="duplicate map"):
        load_config(write_json(tmp_path, data))


def test_error_non_string_map_pool_entry(tmp_path):
    # A stray null from a trailing comma must not be coerced into a pool
    # containing a phantom map named "None".
    data = base_config_data()
    data["eras"][1]["map_pool"] = [
        "Abyss", None, "Ascent", "Haven", "Lotus", "Split", "Summit"
    ]
    with pytest.raises(ConfigError, match="must be a string"):
        load_config(write_json(tmp_path, data))


def test_error_nested_list_map_pool_entry(tmp_path):
    data = base_config_data()
    data["eras"][1]["map_pool"] = [
        "Abyss", ["Ascent"], "Haven", "Lotus", "Split", "Summit", "Sunset"
    ]
    with pytest.raises(ConfigError, match="must be a string"):
        load_config(write_json(tmp_path, data))


def test_error_empty_map_pool(tmp_path):
    data = base_config_data()
    data["eras"][0]["map_pool"] = []
    with pytest.raises(ConfigError, match="map_pool"):
        load_config(write_json(tmp_path, data))


def test_error_bad_start_date(tmp_path):
    data = base_config_data()
    data["eras"][0]["start"] = "not-a-date"
    with pytest.raises(ConfigError, match="start"):
        load_config(write_json(tmp_path, data))


def test_error_bad_end_date(tmp_path):
    data = base_config_data()
    data["eras"][0]["end"] = "2026/07/15"
    with pytest.raises(ConfigError, match="end"):
        load_config(write_json(tmp_path, data))


def test_error_end_not_after_start(tmp_path):
    data = base_config_data()
    data["eras"][1]["end"] = "2026-07-15"  # equal to start
    with pytest.raises(ConfigError, match="strictly after"):
        load_config(write_json(tmp_path, data))


def test_error_overlapping_eras(tmp_path):
    data = base_config_data()
    data["eras"][0]["end"] = "2026-08-01"  # overlaps 2026-abyss (starts 2026-07-15)
    with pytest.raises(ConfigError, match="overlap"):
        load_config(write_json(tmp_path, data))


def test_error_gap_between_eras(tmp_path):
    # A hole between era windows must fail at load time, not later when a
    # match date lands in it and era_as_of has no answer.
    data = base_config_data()
    data["eras"][1]["start"] = "2026-07-20"  # previous era ends 2026-07-15
    with pytest.raises(ConfigError, match="gap"):
        load_config(write_json(tmp_path, data))


def test_error_two_open_ended_eras(tmp_path):
    # Rule 4: at most one era open-ended. Caught by the ordering rule — the
    # first open-ended era is not last, so its window would swallow the
    # following one.
    data = base_config_data()
    data["eras"][0]["end"] = None
    with pytest.raises(ConfigError, match="open-ended"):
        load_config(write_json(tmp_path, data))


def test_error_active_era_unknown(tmp_path):
    data = base_config_data()
    data["active_era"] = "2026-nonexistent"
    with pytest.raises(ConfigError, match="active_era"):
        load_config(write_json(tmp_path, data))


def test_error_active_era_not_current(tmp_path):
    # Rule 5 (as_of): active_era must cover the as_of date, not merely name
    # an existing era: a stale pointer would silently answer from a retired
    # pool after a rotation.
    data = base_config_data()
    data["active_era"] = "2026-s1"  # past era; 2026-abyss covers 2026-08-23
    with pytest.raises(ConfigError, match="does not cover"):
        load_config(write_json(tmp_path, data), as_of=date(2026, 8, 23))


def test_error_no_era_covers_as_of(tmp_path):
    data = base_config_data()
    data["eras"][1]["end"] = "2026-08-01"  # bounded in the past; as_of uncovered
    with pytest.raises(ConfigError, match="no era covers"):
        load_config(write_json(tmp_path, data), as_of=date(2026, 8, 23))


def test_error_event_urls_empty(tmp_path):
    data = base_config_data()
    data["event_urls"] = []
    with pytest.raises(ConfigError, match="event_urls"):
        load_config(write_json(tmp_path, data))


def test_error_event_url_not_vlr(tmp_path):
    data = base_config_data()
    data["event_urls"] = ["https://example.com/event/matches/1/foo"]
    with pytest.raises(ConfigError, match="vlr.gg"):
        load_config(write_json(tmp_path, data))


def test_error_duplicate_event_url(tmp_path):
    data = base_config_data()
    url = "https://www.vlr.gg/event/matches/2976/vct-2026-emea-stage-2/?group=completed"
    data["event_urls"] = [url, url]
    with pytest.raises(ConfigError, match="duplicate event_urls"):
        load_config(write_json(tmp_path, data))


def test_error_event_url_substring_not_enough(tmp_path):
    # "/event/" anywhere in the string is not enough — the URL must start
    # with the https://www.vlr.gg/event/ prefix.
    data = base_config_data()
    data["event_urls"] = ["https://www.vlr.gg/eventmatches/../event/x"]
    with pytest.raises(ConfigError, match="vlr.gg"):
        load_config(write_json(tmp_path, data))


def test_error_corrupt_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="read config"):
        load_config(path)


def test_valid_temp_config_loads(tmp_path):
    cfg = load_config(write_json(tmp_path, base_config_data()))
    assert cfg.region == "emea"
    assert cfg.active_era.name == "2026-abyss"
    assert cfg.active_era.map_pool == EXPECTED_ACTIVE_POOL
    assert cfg.eras[0].name == "2026-s1"


def test_load_config_skips_wall_clock_check_without_as_of(tmp_path):
    # The "covers today" check is opt-in via as_of: a config whose eras are
    # entirely in the past must still load, so an archived config can be
    # backtested and pre-registering the next rotation does not start
    # failing at a future midnight.
    data = base_config_data()
    data["eras"][1]["end"] = "2026-08-01"
    data["active_era"] = "2026-s1"
    cfg = load_config(write_json(tmp_path, data))
    assert cfg.era_as_of(date(2026, 5, 1)).name == "2026-s1"
    # Same file, validated against an uncovered date, fails cleanly.
    with pytest.raises(ConfigError, match="no era covers"):
        load_config(write_json(tmp_path, data), as_of=date(2026, 8, 23))


def test_load_config_as_of_valid_active_era(tmp_path):
    data = base_config_data()
    cfg = load_config(write_json(tmp_path, data), as_of=date(2026, 8, 23))
    assert cfg.active_era.name == "2026-abyss"


def test_load_config_as_of_rejects_non_date(tmp_path):
    data = base_config_data()
    with pytest.raises(ConfigError, match="as_of"):
        load_config(write_json(tmp_path, data), as_of="2026-08-23")


def test_error_unicode_decode(tmp_path):
    # A stray non-UTF-8 byte (editor saving UTF-16/Latin-1) must surface as
    # ConfigError, not a raw UnicodeDecodeError.
    path = tmp_path / "config.json"
    path.write_bytes(b'{"region": "emea",\xff}')
    with pytest.raises(ConfigError, match="read config"):
        load_config(path)


def test_era_as_of_none_raises_configerror():
    # Upcoming/TBD matches carry date=None; must be ConfigError, not TypeError.
    cfg = load_config()
    with pytest.raises(ConfigError, match="expects a date or datetime"):
        cfg.era_as_of(None)


def test_era_as_of_string_raises_configerror():
    cfg = load_config()
    with pytest.raises(ConfigError, match="expects a date or datetime"):
        cfg.era_as_of("2026-08-17")


def test_era_as_of_aware_datetime_uses_utc_date():
    # Era boundaries and Match.date are UTC calendar dates: an aware
    # datetime is converted to UTC before deciding its era.
    from datetime import datetime, timedelta, timezone

    cfg = load_config()
    # 23:30 CEST Aug 16 == 21:30 UTC Aug 16 -> still Breeze pool.
    late_evening_local = datetime(2026, 8, 16, 23, 30, tzinfo=timezone(timedelta(hours=2)))
    assert cfg.era_as_of(late_evening_local).name == "2026-s2-breeze"
    # 02:30 CEST Aug 17 == 00:30 UTC Aug 17 -> Abyss pool.
    early_local = datetime(2026, 8, 17, 2, 30, tzinfo=timezone(timedelta(hours=2)))
    assert cfg.era_as_of(early_local).name == "2026-abyss"


def test_era_as_of_empty_eras_raises_configerror():
    # A directly-built Config with no eras must raise ConfigError from
    # era_as_of, not IndexError on self.eras[0].
    empty = Config(
        region="emea",
        eras=(),
        active_era=Era(name="2026-abyss", start=date(2026, 8, 17), end=None, map_pool=("Abyss",)),
        event_urls=(),
    )
    with pytest.raises(ConfigError, match="no eras configured"):
        empty.era_as_of(date(2026, 8, 23))


def test_normalize_map_name_rejects_non_string():
    # The read path must reject phantom names too: a missing scraped map
    # name must not become a map literally called "None".
    with pytest.raises(ConfigError, match="expects a string"):
        normalize_map_name(None)
    with pytest.raises(ConfigError, match="expects a string"):
        normalize_map_name(["Ascent"])


def test_is_active_map_tracks_rotation_across_midnight(monkeypatch, tmp_path):
    # A long-running process crossing a rotation must not keep answering
    # from the pool it started with: is_active_map derives the era covering
    # date.today() on every call instead of trusting the frozen active_era.
    data = base_config_data()
    cfg = load_config(write_json(tmp_path, data))

    class FakeDate(date):
        _today = None

        @classmethod
        def today(cls):
            return cls._today

    FakeDate._today = FakeDate(2026, 5, 1)  # inside 2026-s1 (no Abyss)
    monkeypatch.setattr(config, "date", FakeDate)
    assert cfg.is_active_map("Abyss") is False

    FakeDate._today = FakeDate(2026, 8, 23)  # inside 2026-abyss
    assert cfg.is_active_map("Abyss") is True
