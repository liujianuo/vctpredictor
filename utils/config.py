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
  window must not silently inherit today's pool. Dates and era
  boundaries are UTC calendar dates: ``Match.date`` is naive UTC and
  an aware ``datetime`` is converted to UTC before deciding its era.
- **Validation is wall-clock independent by default.** ``load_config``
  checks ``active_era`` against an explicit ``as_of`` date only; with
  ``as_of=None`` no "covers today" check runs, so archived configs
  load for backtesting and pre-registering the next rotation does not
  break a nightly run at midnight.
- This module imports nothing from ``scraper/`` and ``scraper/`` must
  not import it: parsers stay pure and fixture-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Union

# Default config location: <project root>/config.json
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_PATH_T = Union[str, Path, None]


class ConfigError(Exception):
    """A config.json file failed validation, or a query has no answer.

    Mirrors the ``VlrError`` convention in ``scraper.vlr``: fail
    loudly rather than returning a silently wrong pool.
    """


def normalize_map_name(name: str) -> str:
    """Normalise a map name: strip and collapse whitespace, title-case.

    vlr.gg markup mixes cases (the fixtures contain both ``Sunset``
    and ``sunset``); comparisons always go through this. Non-strings
    are rejected rather than coerced, so a missing scraped name cannot
    silently become a map literally called ``'None'``.

    Args:
        name: The raw map name to normalise. Must be a ``str``.

    Returns:
        The normalised name: leading/trailing whitespace stripped,
        internal runs of whitespace collapsed to single spaces, and
        the result title-cased (e.g. ``"  sunset "`` -> ``"Sunset"``).

    Raises:
        ConfigError: If ``name`` is not a ``str`` (a non-string is
            never coerced, only rejected).
    """
    if not isinstance(name, str):
        raise ConfigError(
            f"normalize_map_name expects a string, got "
            f"{type(name).__name__}: {name!r}"
        )
    return " ".join(name.split()).title()


def _parse_iso_date(value, era_name: str, field: str) -> date:
    """Parse and validate one ISO-8601 date field of an era entry.

    Args:
        value: The raw value read from ``config.json`` for this field.
            Must be a non-empty ``str`` in ``YYYY-MM-DD`` format.
        era_name: Name of the era this field belongs to, used only to
            make error messages identify which era is malformed.
        field: Name of the field being parsed (``"start"`` or
            ``"end"``), used only to make error messages specific.

    Returns:
        The parsed ``date``.

    Raises:
        ConfigError: If ``value`` is not a non-empty string, or is a
            string that does not parse as an ISO-8601 date.
    """
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
    """Validate and normalise one era's ``map_pool`` list.

    Every entry is passed through :func:`normalize_map_name`, and the
    result is checked for post-normalisation duplicates (e.g.
    ``"Sunset"`` and ``"sunset"`` colliding into the same map).

    Args:
        pool: The raw ``map_pool`` value read from ``config.json`` for
            this era. Must be a non-empty ``list`` of ``str``.
        era_name: Name of the era this pool belongs to, used only to
            make error messages identify which era is malformed.

    Returns:
        A ``tuple`` of normalised map names, in the same order as
        ``pool``, with no duplicates.

    Raises:
        ConfigError: If ``pool`` is not a non-empty list, if any entry
            is not a string, if an entry normalises to an empty
            string, or if two entries normalise to the same name.
    """
    if not isinstance(pool, list) or len(pool) == 0:
        raise ConfigError(
            f"era {era_name!r}: map_pool must be a non-empty list of map names"
        )
    seen: set[str] = set()
    normalized = []
    for entry in pool:
        if not isinstance(entry, str):
            raise ConfigError(
                f"era {era_name!r}: map_pool entry must be a string, "
                f"got {type(entry).__name__}: {entry!r}"
            )
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
        """Check whether a map is in this era's pool.

        Args:
            name: The map name to check. Normalised via
                :func:`normalize_map_name` before comparison, so case
                and whitespace differences do not affect the result.

        Returns:
            ``True`` if the normalised ``name`` is in ``self.map_pool``,
            ``False`` otherwise.

        Raises:
            ConfigError: If ``name`` is not a ``str`` (propagated from
                :func:`normalize_map_name`).
        """
        return normalize_map_name(name) in self.map_pool


@dataclass(frozen=True)
class Config:
    """Validated project configuration loaded from ``config.json``."""

    region: str
    eras: tuple[Era, ...]
    active_era: Era
    event_urls: tuple[str, ...]

    def era_as_of(self, d: date) -> Era:
        """Find the era whose window contains a given date.

        This is what lets a 2026-07-15 match be scored against the
        pool that was live then rather than today's. Era boundaries
        are half-open (``start <= d < end``), so a date exactly on an
        era's ``start`` belongs to that era, not the previous one.

        Args:
            d: The date to resolve to an era. May be a ``date`` or a
                ``datetime``. Era boundaries and ``Match.date``
                (``scraper.models``) are UTC calendar dates: a naive
                ``datetime`` is treated as already UTC and narrowed to
                its UTC date; a timezone-aware ``datetime`` is
                converted to UTC first. Either way, the UTC calendar
                date is what decides the era.

        Returns:
            The :class:`Era` whose ``[start, end)`` window contains
            ``d``.

        Raises:
            ConfigError: If ``d`` is not a ``date``/``datetime`` (e.g.
                ``None`` for an upcoming match, or a string) — raised
                here rather than letting an opaque ``TypeError``
                propagate. Also raised if ``self.eras`` is empty, or
                if no configured era's window covers ``d``.
        """
        if not isinstance(d, date):
            raise ConfigError(
                f"era_as_of expects a date or datetime, got "
                f"{type(d).__name__}: {d!r}"
            )
        if isinstance(d, datetime):
            if d.tzinfo is not None:
                d = d.astimezone(timezone.utc)
            d = d.date()
        if not self.eras:
            raise ConfigError(
                "no eras configured; cannot resolve a date to an era"
            )
        for era in self.eras:
            if era.start <= d and (era.end is None or d < era.end):
                return era
        raise ConfigError(
            f"no era covers {d.isoformat()}; eras span "
            f"{self.eras[0].start.isoformat()}.."
            f"{self.eras[-1].end.isoformat() if self.eras[-1].end else 'open'}"
        )

    def is_active_map(self, name: str) -> bool:
        """Check whether a map is in the pool live today.

        Derived from :meth:`era_as_of` at call time rather than from the
        frozen ``active_era`` field, so a long-running process (a looped
        scrape driver, a notebook kernel) that crosses a rotation
        midnight keeps answering from the pool that actually covers
        today instead of the one it started with.

        Args:
            name: The map name to check (any case/whitespace).

        Returns:
            ``True`` if ``name`` is in the pool of the era covering
            today's date, ``False`` otherwise.

        Raises:
            ConfigError: If no configured era covers today's date, or
                if ``name`` is not a ``str`` (propagated from
                :meth:`era_as_of` / :func:`normalize_map_name`).
        """
        return self.era_as_of(date.today()).contains_map(name)


def load_config(path: _PATH_T = None, as_of: Optional[date] = None) -> Config:
    """Load, parse and validate ``config.json`` into a :class:`Config`.

    Reads the JSON file at ``path``, then runs it through six
    validation rules in order (required keys present; each era has a
    non-empty, duplicate-free map pool; each era's dates parse and
    ``end`` is strictly after ``start``; era windows are contiguous
    and non-overlapping with at most one open-ended era; ``active_era``
    names a real era, optionally checked against ``as_of``; and
    ``event_urls`` is a non-empty list of unique absolute vlr.gg event
    URLs). Any rule failing raises :class:`ConfigError` immediately —
    this function never returns a partially-valid ``Config``.

    Args:
        path: Path to the config JSON file. ``None`` (the default)
            falls back to the module-level ``DEFAULT_CONFIG_PATH``
            (``<project root>/config.json``). Accepts ``str | Path |
            None`` so tests can point it at a temp file.
        as_of: The date (or datetime) against which the "active_era
            covers this date" check runs. When ``None`` (the default)
            that check is skipped entirely: loading is wall-clock
            independent, so an archived config loads for backtesting
            and a config that pre-declares the next rotation does not
            start failing at midnight. Pass ``as_of`` (e.g.
            ``date.today()``) when you specifically want to assert the
            declared ``active_era`` is current for that date.

    Returns:
        A fully validated :class:`Config`, with ``eras`` sorted by
        start date and all map names normalised.

    Raises:
        ConfigError: If the file cannot be read or decoded, if the
            JSON root is not an object, if any required key is
            missing, if any era's dates/map pool are malformed, if era
            windows overlap or have gaps, if more than one era is
            open-ended, if ``active_era`` does not name a configured
            era, if ``as_of`` is given and is not a
            ``date``/``datetime`` or does not fall within
            ``active_era``'s window, or if ``event_urls`` is empty,
            malformed, or contains duplicates.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # OSError: unreadable/missing file. ValueError: JSONDecodeError
        # and UnicodeDecodeError (a stray non-UTF-8 byte from an editor
        # saving UTF-16 or Latin-1) both subclass it.
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

    # Rule 4: windows do not overlap and are contiguous (no gaps); at most
    # one era is open-ended — which follows from the loop below, since an
    # open-ended era that is not last would swallow the following era, and
    # the eras are sorted, so at most one can be last.
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
        if cur.start > prev.end:
            raise ConfigError(
                f"era windows have a gap: {prev.name!r} ends {prev.end.isoformat()} "
                f"but {cur.name!r} starts {cur.start.isoformat()}"
            )

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

    # Rule 5 (cont.): when as_of is given, active_era must be the era
    # covering it, so a stale pointer fails here instead of silently
    # serving a retired pool. Opt-in by design: a config valid for
    # backtesting (or pre-declaring the next rotation) must not start
    # failing at a future midnight, so with as_of=None no wall-clock
    # check runs.
    if as_of is not None:
        if not isinstance(as_of, date):
            raise ConfigError(
                f"as_of must be a date or datetime, got "
                f"{type(as_of).__name__}: {as_of!r}"
            )
        if isinstance(as_of, datetime):
            if as_of.tzinfo is not None:
                as_of = as_of.astimezone(timezone.utc)
            as_of = as_of.date()
        covering = next(
            (e for e in eras if e.start <= as_of and (e.end is None or as_of < e.end)),
            None,
        )
        if covering is None:
            raise ConfigError(
                f"no era covers {as_of.isoformat()}; eras are stale: "
                f"{[e.name for e in eras]}"
            )
        if active is not covering:
            raise ConfigError(
                f"active_era {active_name!r} does not cover "
                f"{as_of.isoformat()}; the era covering it is {covering.name!r}"
            )

    # Rule 6: non-empty event_urls, all absolute vlr.gg URLs with /event/.
    urls = raw["event_urls"]
    if not isinstance(urls, list) or len(urls) == 0:
        raise ConfigError("event_urls must be a non-empty list")
    event_urls: list[str] = []
    seen_urls: set[str] = set()
    for u in urls:
        if not isinstance(u, str) or not u.startswith("https://www.vlr.gg/event/"):
            raise ConfigError(
                "event_urls entry must be an absolute vlr.gg event URL "
                f"starting with https://www.vlr.gg/event/: {u!r}"
            )
        if u in seen_urls:
            raise ConfigError(f"duplicate event_urls entry: {u!r}")
        seen_urls.add(u)
        event_urls.append(u)

    return Config(
        region=region,
        eras=tuple(eras),
        active_era=active,
        event_urls=tuple(event_urls),
    )


# Module-level convenience: from config import ACTIVE
#
# Lazy on purpose (PEP 562 module __getattr__): an eager load would make a
# bad config.json abort ``import config`` — and with it pytest collection
# of tests/test_config.py, the very tests written to assert that invalid
# configs raise cleanly. The ConfigError instead surfaces on first ACTIVE
# *use*, still "at load time" from any consumer's point of view.
_ACTIVE: Optional[Config] = None


def __getattr__(name: str) -> Config:
    """Module-level attribute hook implementing lazy ``config.ACTIVE``.

    PEP 562 hook: Python calls this only when a normal attribute
    lookup on the module fails, i.e. only for ``config.ACTIVE`` here.
    The config is loaded (via :func:`load_config`) on first access and
    cached in the module-level ``_ACTIVE`` global for subsequent
    accesses, so a bad ``config.json`` fails at first *use* of
    ``ACTIVE`` rather than aborting ``import config`` (which would also
    break pytest collection of tests asserting that invalid configs
    raise cleanly).

    Args:
        name: The attribute name Python failed to find on this module.

    Returns:
        The cached (or newly loaded) :class:`Config` when
        ``name == "ACTIVE"``.

    Raises:
        AttributeError: If ``name`` is anything other than
            ``"ACTIVE"``.
        ConfigError: If ``name == "ACTIVE"`` and ``config.json`` fails
            validation (propagated from :func:`load_config`).
    """
    if name == "ACTIVE":
        global _ACTIVE
        if _ACTIVE is None:
            _ACTIVE = load_config()
        return _ACTIVE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
