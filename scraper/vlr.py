"""Scraping logic for vlr.gg.

vls.gg has no public API, so this module scrapes its HTML with
requests + BeautifulSoup. Page structure may change over time and
break the selectors below — that is a known fragility. All network
errors surface as :class:`VlrFetchError` and all missing-selector
problems as :class:`VlrParseError`, never as raw bs4 exceptions.

Parse functions (``parse_match``, ``parse_event_match_links``) are
pure: they take an HTML string and do no I/O, so they can be tested
against saved fixtures.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .cache import (
    get_cached_match,
    get_cached_page,
    set_cached_match,
    set_cached_page,
)
from .models import MapResult, Match, Team

BASE_URL = "https://www.vlr.gg"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
# Delay between consecutive uncached match fetches, to be polite to vlr.gg.
POLITE_DELAY_SECONDS = 1.0

_MATCH_ID_RE = re.compile(r"/(\d+)/")
_BO_RE = re.compile(r"Bo\d+")
_TEAM_ID_RE = re.compile(r"/team/(\d+)")


class VlrError(Exception):
    """Base class for all errors raised by this module."""


class VlrFetchError(VlrError):
    """A network/HTTP failure while fetching a page."""


class VlrParseError(VlrError):
    """Expected structure was missing from a fetched page."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def extract_match_id(url: str) -> str:
    """Return the numeric match id from a vlr.gg match URL.

    Example: ``https://www.vlr.gg/712803/...`` -> ``"712803"``.
    A bare numeric id (``"712803"``) is accepted as-is. Event-page
    URLs (which also contain a numeric id, the event id) are rejected
    so a wrong id never becomes a cache key.

    Args:
        url: A vlr.gg match URL (absolute or relative), a bare numeric
            match id string, or an event-page URL. Any query string
            (``?...``) or fragment (``#...``) is stripped before
            parsing.

    Returns:
        The numeric match id as a string, e.g. ``"712803"``.

    Raises:
        VlrParseError: If ``url`` is an event page (contains
            ``"/event/"``), or if no numeric id can be found in it.
    """
    url = url.split("?", 1)[0].split("#", 1)[0]
    if "/event/" in url:
        raise VlrParseError(f"URL is an event page, not a match page: {url!r}")
    if url.isdigit():
        return url
    m = _MATCH_ID_RE.search(url)
    if m is None:
        raise VlrParseError(f"could not extract a numeric match id from URL: {url!r}")
    return m.group(1)


def _parse_int(text: str) -> Optional[int]:
    """Best-effort parse of a stripped string as an integer.

    Args:
        text: The text to parse, e.g. text scraped from a score span.
            Leading/trailing whitespace is stripped before parsing.

    Returns:
        The parsed ``int``, or ``None`` if ``text`` (after stripping)
        is not a valid integer literal (e.g. ``""``, ``"-"``, ``"TBD"``).
    """
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return None


def _parse_team(link_el) -> Team:
    """Parse a team from a ``.match-header-link`` element.

    Args:
        link_el: A BeautifulSoup ``Tag`` for one ``.match-header-link``
            element from a match header, containing the team name
            (``.wf-title-med``) and, in its ``href``, the team's vlr.gg
            page URL.

    Returns:
        A :class:`scraper.models.Team` with the parsed name and, when
        the ``href`` matches ``/team/<id>``, its numeric ``team_id``
        (``None`` if the id could not be extracted).

    Raises:
        VlrParseError: If no team name (``.wf-title-med``) is found,
            or it is empty.
    """
    name_el = link_el.select_one(".wf-title-med")
    name = name_el.get_text(strip=True) if name_el is not None else ""
    if not name:
        raise VlrParseError("team link found without a team name (.wf-title-med)")
    href = link_el.get("href") or ""
    m = _TEAM_ID_RE.search(href)
    return Team(name=name, team_id=m.group(1) if m else None)


def _parse_map(game_el, team1: Team, team2: Team) -> MapResult:
    """Parse one played map's result from a ``.vm-stats-game`` block.

    Args:
        game_el: A BeautifulSoup ``Tag`` for one ``.vm-stats-game``
            element, expected to contain a ``.vm-stats-game-header``
            with the map name, per-team score, win indicator and
            duration.
        team1: The match's team1, used to resolve the map winner's
            name when the ``.score.mod-win`` element is on the
            left-hand (``mod-1``) side.
        team2: The match's team2, used to resolve the map winner's
            name when the ``.score.mod-win`` element is on the
            right-hand (``mod-right``) side.

    Returns:
        A :class:`scraper.models.MapResult` with the parsed map name,
        both teams' scores (``None`` for any that could not be parsed
        as an integer), the winner's team name (``None`` if no
        ``.score.mod-win`` element was found), and the map duration.
        ``agent_picks`` is always ``None`` (not yet parsed from stats
        tables).

    Raises:
        VlrParseError: If ``game_el`` has no ``.vm-stats-game-header``,
            or the header has no map name (``.map div span``).
    """
    header_el = game_el.select_one(".vm-stats-game-header")
    if header_el is None:
        raise VlrParseError("vm-stats-game block without .vm-stats-game-header")

    # Map name is the direct text of the innermost <span> in the map
    # header (the "PICK"/"BAN" labels are nested spans, so recursive
    # text would include them).
    map_span = header_el.select_one(".map div span")
    map_name = ""
    if map_span is not None:
        direct = map_span.find(string=True, recursive=False)
        map_name = direct.strip() if direct else ""
    if not map_name:
        raise VlrParseError("map block without a map name (.map div span)")

    score_els = header_el.select(".team .score")
    team1_score = (
        _parse_int(score_els[0].get_text(strip=True)) if len(score_els) >= 1 else None
    )
    team2_score = (
        _parse_int(score_els[1].get_text(strip=True)) if len(score_els) >= 2 else None
    )

    win_el = header_el.select_one(".score.mod-win")
    winner = None
    if win_el is not None:
        parent = win_el.find_parent("div", class_="team")
        if parent is not None and "mod-right" in (parent.get("class") or []):
            winner = team2.name
        else:
            winner = team1.name

    duration_el = header_el.select_one(".map-duration")
    duration = duration_el.get_text(strip=True) if duration_el is not None else None

    return MapResult(
        map_name=map_name,
        team1_score=team1_score,
        team2_score=team2_score,
        winner=winner,
        duration=duration,
        agent_picks=None,  # reserved; agent picks not parsed from stats tables yet
    )


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def fetch_page(url: str, use_cache: bool = True, force_refresh: bool = False) -> str:
    """Return the HTML of ``url``, using the local page cache.

    Checks ``cache.get_cached_page`` first unless ``force_refresh`` is
    True, otherwise fetches over HTTP with a real User-Agent header,
    raises :class:`VlrFetchError` on non-200/network failure, and
    stores the result in the cache (when ``use_cache``).

    Args:
        url: The absolute URL to fetch.
        use_cache: When ``True`` (the default), read from and write to
            the local page cache. When ``False``, always fetch over
            HTTP and never touch the cache.
        force_refresh: When ``True``, skip the cache read and always
            fetch over HTTP, even if ``use_cache`` is ``True`` (the
            result is still written back to the cache when
            ``use_cache`` is ``True``).

    Returns:
        The page's HTML as a string.

    Raises:
        VlrFetchError: If the HTTP request fails (network error,
            timeout, or non-2xx status via ``raise_for_status``).
    """
    if use_cache and not force_refresh:
        cached = get_cached_page(url)
        if cached is not None:
            return cached
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise VlrFetchError(f"failed to fetch {url}: {exc}") from exc
    html = resp.text
    if use_cache:
        set_cached_page(url, html)
    return html


# --------------------------------------------------------------------------
# Parsing (pure, no I/O)
# --------------------------------------------------------------------------


def parse_match(html: str, url: str) -> Match:
    """Parse a vlr.gg match page HTML string into a :class:`Match`.

    Pure function: does no network I/O, so it can be run against saved
    HTML fixtures in tests. Extracts the event name, both teams, the
    scheduled/played date, match status (``"completed"``/``"live"``/
    ``"upcoming"``), best-of format, overall scores, and the list of
    per-map results (skipping the "All Maps" overview block and any
    placeholder ``"TBD"`` maps rendered for upcoming matches).

    Args:
        html: The full HTML of a vlr.gg match page.
        url: The URL the HTML was fetched from. Used to derive
            ``match_id`` (via :func:`extract_match_id`) and stored
            verbatim on the returned ``Match``.

    Returns:
        The parsed :class:`scraper.models.Match`.

    Raises:
        VlrParseError: If ``url`` is not a parseable match URL, or if
            any expected element is missing from ``html`` — no
            ``div.match-header``, no ``.match-header-event`` (and no
            fallback event slug), fewer than two
            ``.match-header-link`` team elements, or a per-map block
            with no map name.
    """
    soup = BeautifulSoup(html, "lxml")
    header = soup.select_one("div.match-header")
    if header is None:
        raise VlrParseError('no <div class="match-header"> found; is this a match page?')

    match_id = extract_match_id(url)

    # Event name.
    event_el = header.select_one(".match-header-event")
    if event_el is None:
        raise VlrParseError("no .match-header-event element found")
    name_el = event_el.select_one("div > div")
    event_name = name_el.get_text(strip=True) if name_el is not None else ""
    if not event_name:
        # Fall back to the event slug from the href, e.g. /event/2976/<slug>/...
        event_name = (event_el.get("href") or "").rsplit("/", 1)[-1].replace("-", " ")
    if not event_name:
        raise VlrParseError("could not determine event name from match header")

    # Teams (document order: team1 is mod-1/left, team2 is mod-2/right).
    team_els = header.select(".match-header-link")
    if len(team_els) < 2:
        raise VlrParseError(f"expected 2 team links in match header, found {len(team_els)}")
    team1 = _parse_team(team_els[0])
    team2 = _parse_team(team_els[1])

    # Date: naive UTC string from data-utc-ts, e.g. "2026-07-15 11:00:00".
    date_el = header.select_one(".match-header-date")
    utc_el = date_el.find(attrs={"data-utc-ts": True}) if date_el is not None else None
    date = None
    if utc_el is not None:
        try:
            date = datetime.fromisoformat(utc_el["data-utc-ts"].strip())
        except ValueError:
            date = None

    # Notes: [status ("final" | countdown), ..., best_of ("Bo3")].
    notes = [
        n.get_text(strip=True)
        for n in header.select(".match-header-vs-score .match-header-vs-note")
    ]
    best_of = next((n for n in reversed(notes) if _BO_RE.fullmatch(n)), None)
    status_note = notes[0].lower() if notes else ""

    # Status + scores. The displayed score is always team1:team2 from
    # left to right, so the numeric spans in DOM order map directly to
    # (team1_score, team2_score) regardless of winner/loser labels.
    if header.select_one(".match-header-vs-score-winner") is not None:
        status = "completed"
    elif "live" in status_note:
        status = "live"
    else:
        status = "upcoming"

    score_spans = header.select(".match-header-vs-score .sp-hide span")
    numeric_scores = [
        s.get_text(strip=True) for s in score_spans if s.get_text(strip=True).isdigit()
    ]
    team1_score = _parse_int(numeric_scores[0]) if len(numeric_scores) >= 1 else None
    team2_score = _parse_int(numeric_scores[1]) if len(numeric_scores) >= 2 else None

    # Per-map results. The "All Maps" overview block has no
    # .vm-stats-game-header, so it is naturally skipped. Upcoming
    # matches render placeholder map blocks (name "TBD", score 0:0,
    # duration "-"); those are not real results, so they are skipped.
    maps: List[MapResult] = []
    for g in soup.select(".vm-stats-game"):
        if g.select_one(".vm-stats-game-header") is None:
            continue
        map_result = _parse_map(g, team1, team2)
        if map_result.map_name.strip().upper() == "TBD":
            continue
        maps.append(map_result)

    return Match(
        match_id=match_id,
        url=url,
        event_name=event_name,
        date=date,
        team1=team1,
        team2=team2,
        team1_score=team1_score,
        team2_score=team2_score,
        best_of=best_of,
        maps=maps,
        status=status,
    )


def parse_event_match_links(html: str) -> List[str]:
    """Extract the list of match URLs from an event's matches page.

    Pure function: does no network I/O, so it can be run against saved
    HTML fixtures in tests.

    Args:
        html: The full HTML of a vlr.gg event matches page.

    Returns:
        Relative match URLs (e.g.
        ``/712833/fnatic-vs-team-heretics-...``) in page order, with
        duplicates removed (first occurrence kept). An empty list if
        no match links are found — this is not treated as an error,
        since an event with no matches yet is a valid state.
    """
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    seen = set()
    for a in soup.select("a.wf-module-item.match-item"):
        href = a.get("href")
        if href and href.startswith("/") and href not in seen:
            seen.add(href)
            links.append(href)
    return links


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def get_match(url: str, use_cache: bool = True) -> Match:
    """Fetch (or load from cache) and parse a single match.

    Checks the parsed-match cache first (keyed by match id); on a
    miss, fetches the page HTML (:func:`fetch_page`), parses it
    (:func:`parse_match`), and stores the result in the match cache
    before returning it.

    Args:
        url: The vlr.gg match page URL to fetch and parse.
        use_cache: When ``True`` (the default), read from and write to
            both the page cache and the parsed-match cache. When
            ``False``, always fetch and parse fresh and never touch
            either cache.

    Returns:
        The parsed :class:`scraper.models.Match`, either from cache or
        freshly fetched.

    Raises:
        VlrParseError: If ``url`` is not a parseable match URL, or if
            the fetched page is missing expected structure
            (propagated from :func:`extract_match_id` /
            :func:`parse_match`).
        VlrFetchError: If fetching the page over HTTP fails
            (propagated from :func:`fetch_page`).
    """
    match_id = extract_match_id(url)
    if use_cache:
        cached = get_cached_match(match_id)
        if cached is not None:
            return cached
    html = fetch_page(url, use_cache=use_cache)
    match = parse_match(html, url)
    if use_cache:
        set_cached_match(match)
    return match


def get_matches_from_event(event_url: str, use_cache: bool = True) -> List[Match]:
    """Fetch (or load from cache) every match listed on an event page.

    Fetches the event page, extracts its match links
    (:func:`parse_event_match_links`), then fetches/parses each match
    in page order (:func:`get_match`). A small delay
    (``POLITE_DELAY_SECONDS``) is inserted between consecutive
    *uncached* match fetches to be polite to vlr.gg; cached matches
    incur no delay.

    Args:
        event_url: The vlr.gg event matches page URL.
        use_cache: When ``True`` (the default), read from and write to
            the page and match caches for both the event page and
            every individual match. When ``False``, always fetch
            fresh and never touch either cache (and every match fetch
            is treated as uncached, so the polite delay applies
            between all of them).

    Returns:
        A list of parsed :class:`scraper.models.Match` objects, one
        per match link found on the event page, in page order.

    Raises:
        VlrFetchError: If fetching the event page or any match page
            over HTTP fails.
        VlrParseError: If the event page or any match page is missing
            expected structure.
    """
    html = fetch_page(event_url, use_cache=use_cache)
    links = parse_event_match_links(html)
    matches: List[Match] = []
    fetched = False
    for link in links:
        url = link if link.startswith("http") else BASE_URL + link
        if use_cache and get_cached_match(extract_match_id(url)) is not None:
            matches.append(get_match(url, use_cache=True))
            continue
        if fetched:
            time.sleep(POLITE_DELAY_SECONDS)
        matches.append(get_match(url, use_cache=use_cache))
        fetched = True
    return matches
