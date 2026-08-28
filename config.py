"""Map-pool + era configuration (roadmap M0).

The active map pool rotates, and this file is the *only* place that
rotation is declared. ``config.json`` holds the data; this module
loads and validates it; every downstream consumer (M7 scrape driver,
M13 map win rates, M25 veto simulator, M27/M28 conditional logit)
reads the pool from here.

Evidence that the pool rotates, already in the repo: the Stage 2 W1
fixture (``tests/fixtures/match_page.html``) vetoes a pool containing
Breeze and no Abyss — not the current pool. Live event pages confirm
the v1 window (VCT 2026 EMEA Stage 1 + Stage 2) straddles *four* pools:

- 2026-s1-bind    (Apr 1-May 1):    Bind-era pool, Stage 1 group stage
- 2026-s1-ascent  (May 2-Jul 14):   Ascent replaces Bind, Stage 1 playoffs
- 2026-s2-breeze  (Jul 15-Aug 16):  Summit/Sunset-era pool, Stage 2 group
- 2026-abyss      (Aug 17-...):     Abyss replaces Breeze (current pool)

Design rules:

- **Fail loudly at load time.** A wrong pool silently corrupts every
  downstream feature, so any invalid ``config.json`` raises
  :class:`ConfigError` here rather than surfacing later as wrong data.
- **Score each match against the pool live at its date.** Use
  :meth:`Config.era_as_of`; era windows are half-open ``[start, end)``
  and ``end=None`` means open-ended. A date with no matching era
  raises :class:`ConfigError` — a match from outside the configured
  window must not silently inherit today's pool.
- This module imports nothing from ``scraper/`` and ``scraper/`` must
  not import it: parsers stay pure and fixture-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

# Default config location: <project root>/config.json
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

_PATH_T = Union[str, Path, None]


class ConfigError(Exception):
    """A config.json file failed validation, or a query has no answer.

    Mirrors the ``VlrError`` convention in ``scraper.vlr``: fail
    loudly rather than returning a silently wrong pool.
    """


def normalize_map_name(name: str) -> str:
    """Normalise a map name: strip and collapse whitespace, title-case.

    vlr.gg markup mixes cases (the fixtures contain both ``Sunset``
    and ``sunset``); comparisons always go through this.
    """
    return " ".join(str(name).split()).title()


def _parse_iso_date(value, era_name: str, field: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"era {era_name!r}: {field} must be an ISO date string "
            f"(YYYY-MM-DD), got {value!r}"
        )
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ConfigError(
            f"era {era_name!r}: {field} is not a valid ISO date: {value!r}"
        ) from exc


def _normalize_pool(pool, era_name: str) -> tuple[str, ...]:
    if not isinstance(pool, list) or len(pool) == 0:
        raise ConfigError(
            f"era {era_name!r}: map_pool must be a non-empty list of map names"
        )
    seen: set[str] = set()
    normalized = []
    for entry in pool:
        name = normalize_map_name(entry)
        if not name:
            raise ConfigError(
                f"era {era_name!r}: map_pool entry is empty after normalisation: {entry!r}"
            )
        if name in seen:
            raise ConfigError(
                f"era {era_name!r}: duplicate map in pool after normalisation: {name!r}"
            )
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


@dataclass(frozen=True)
class Era:
    """A map-pool era: the pool that was live between ``start`` and ``end``.

    Windows are half-open: the era covers ``start <= d < end``, and
    ``end=None`` means open-ended. ``map_pool`` is stored already
    normalised (see :func:`normalize_map_name`).
    """

    name: str
    start: date
    end: Optional[date]
    map_pool: tuple[str, ...]

    def contains_map(self, name: str) -> bool:
        """True if ``name`` (any case/whitespace) is in this era's pool."""
        return normalize_map_name(name) in self.map_pool


@dataclass(frozen=True)
class Config:
    """Validated project configuration loaded from ``config.json``."""

    region: str
    eras: tuple[Era, ...]
    active_era: Era
    event_urls: tuple[str, ...]

    def era_as_of(self, d: date) -> Era:
        """The era whose window contains ``d``; raises ConfigError if none.

        This is what lets a 2026-07-15 match be scored against the
        pool that was live then rather than today's.
        """
        if isinstance(d, datetime):
            d = d.date()
        for era in self.eras:
            if era.start <= d and (era.end is None or d < era.end):
                return era
        raise ConfigError(
            f"no era covers {d.isoformat()}; eras span "
            f"{self.eras[0].start.isoformat()}.."
            f"{self.eras[-1].end.isoformat() if self.eras[-1].end else 'open'}"
        )

    def is_active_map(self, name: str) -> bool:
        """True if ``name`` is in the *active* (current) era's pool."""
        return self.active_era.contains_map(name)


def load_config(path: _PATH_T = None) -> Config:
    """Load and validate ``config.json``.

    ``path=None`` falls back to the module-level ``DEFAULT_CONFIG_PATH``
    (``<project root>/config.json``). Accepts ``str | Path | None`` so
    tests can point it at a temp file.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a JSON object, got {type(raw).__name__}"
        )

    # Rule 1: required keys present.
    required = {"region", "active_era", "eras", "event_urls"}
    missing = required - raw.keys()
    if missing:
        raise ConfigError(f"config missing required key(s): {sorted(missing)}")

    region = raw["region"]
    if not isinstance(region, str) or not region.strip():
        raise ConfigError("region must be a non-empty string")
    region = region.strip().lower()

    # Rule 1: eras non-empty.
    eras_raw = raw["eras"]
    if not isinstance(eras_raw, list) or len(eras_raw) == 0:
        raise ConfigError("eras must be a non-empty list")

    eras: list[Era] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(eras_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"eras[{i}] must be an object, got {type(entry).__name__}")
        era_missing = {"name", "start", "end", "map_pool"} - entry.keys()
        if era_missing:
            raise ConfigError(f"eras[{i}] missing required key(s): {sorted(era_missing)}")

        name = entry["name"]
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"eras[{i}] name must be a non-empty string")
        name = name.strip()
        if name in seen_names:
            raise ConfigError(f"duplicate era name: {name!r}")
        seen_names.add(name)

        # Rule 3: start parses as ISO date; end null or strictly after start.
        start = _parse_iso_date(entry["start"], name, "start")
        end = None if entry["end"] is None else _parse_iso_date(entry["end"], name, "end")
        if end is not None and end <= start:
            raise ConfigError(
                f"era {name!r}: end ({end.isoformat()}) must be strictly after "
                f"start ({start.isoformat()})"
            )

        # Rule 2: non-empty pool, no duplicates after normalisation.
        pool = _normalize_pool(entry["map_pool"], name)

        eras.append(Era(name=name, start=start, end=end, map_pool=pool))

    # Rule 4: windows do not overlap; at most one era is open-ended.
    eras.sort(key=lambda e: e.start)
    for prev, cur in zip(eras, eras[1:]):
        if prev.end is None:
            raise ConfigError(
                f"era {prev.name!r} is open-ended (end: null) but is not the last "
                f"era; its window swallows {cur.name!r}"
            )
        if cur.start < prev.end:
            raise ConfigError(
                f"era windows overlap: {prev.name!r} [{prev.start.isoformat()}, "
                f"{prev.end.isoformat()}) vs {cur.name!r} [{cur.start.isoformat()}, ...)"
            )
    if len([e for e in eras if e.end is None]) > 1:
        raise ConfigError("at most one era may be open-ended (end: null)")

    # Rule 5: active_era names an era that exists.
    active_name = raw["active_era"]
    if not isinstance(active_name, str) or not active_name.strip():
        raise ConfigError("active_era must be a non-empty string naming an era")
    active = next((e for e in eras if e.name == active_name.strip()), None)
    if active is None:
        raise ConfigError(
            f"active_era {active_name!r} does not name an era in eras: "
            f"{[e.name for e in eras]}"
        )

    # Rule 6: non-empty event_urls, all absolute vlr.gg URLs with /event/.
    urls = raw["event_urls"]
    if not isinstance(urls, list) or len(urls) == 0:
        raise ConfigError("event_urls must be a non-empty list")
    event_urls: list[str] = []
    for u in urls:
        if (
            not isinstance(u, str)
            or not u.startswith("https://www.vlr.gg/")
            or "/event/" not in u
        ):
            raise ConfigError(
                "event_urls entry must be an absolute vlr.gg URL containing "
                f"/event/: {u!r}"
            )
        event_urls.append(u)

    return Config(
        region=region,
        eras=tuple(eras),
        active_era=active,
        event_urls=tuple(event_urls),
    )


# Module-level convenience: from config import ACTIVE
ACTIVE = load_config()
