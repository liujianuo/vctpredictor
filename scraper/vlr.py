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

import logging
import re
import time
from datetime import datetime
from typing import List, Optional
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from .cache import (
    get_cached_match,
    get_cached_page,
    set_cached_match,
    set_cached_page,
)
from .models import IllegalScoreError, MapResult, Match, PlayerStats, Team, VetoAction

logger = logging.getLogger(__name__)

BASE_URL = "https://www.vlr.gg"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# The token robots.txt rules are matched against. RobotFileParser matches
# only the token before the first "/" in the agent string it is given, so
# passing USER_AGENT ("Mozilla/5.0 ...") would silently resolve to
# "mozilla" and never hit a targeted "User-agent: <botname>" block. The
# actual HTTP requests keep the browser UA (USER_AGENT) so vlr.gg's edge
# layer treats them as ordinary browsing; robots permission is judged
# under this explicit bot token instead.
ROBOTS_USER_AGENT = "vct-predictor-scraper"
REQUEST_TIMEOUT = 15
# Delay between consecutive uncached match fetches, to be polite to vlr.gg.
POLITE_DELAY_SECONDS = 1.0
# Where vlr.gg publishes its robots.txt; checked once per CLI run, before any
# event URL is fetched (see scrape.py).
ROBOTS_URL = BASE_URL + "/robots.txt"

_MATCH_ID_RE = re.compile(r"/(\d+)/")
_BO_RE = re.compile(r"Bo\d+")
_TEAM_ID_RE = re.compile(r"/team/(\d+)")
# A veto-note segment naming an acting team and an action, e.g.
# "NAVI ban Haven" / "FUT pick Split". The team token is any
# non-empty run of characters before the action keyword (vlr.gg uses
# short abbreviations like "NAVI"/"FUT"/"KRX"); the map is everything
# after it. "ban|pick" is matched before the decider pattern below.
_VETO_BAN_PICK_RE = re.compile(r"^(?P<team>.+?) (?P<action>ban|pick) (?P<map>.+)$")
# A veto-note segment naming the decider map, e.g. "Sunset remains".
_VETO_DECIDER_RE = re.compile(r"^(?P<map>.+) remains$")


class VlrError(Exception):
    """Base class for all errors raised by this module."""


class VlrFetchError(VlrError):
    """A network/HTTP failure while fetching a page."""


class VlrParseError(VlrError):
    """Expected structure was missing from a fetched page."""


class VlrRobotsError(VlrError):
    """A URL is disallowed by the site's robots.txt."""


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


def _parse_float(text: str) -> Optional[float]:
    """Best-effort parse of a stripped string as a float.

    Mirrors :func:`_parse_int`'s best-effort-``None`` convention, with
    one extra accommodation: vlr.gg renders percentage columns (KAST,
    HS%) as ``"74%"`` / ``"27%"``, so an optional trailing ``%``
    sign is stripped before parsing. Whole numbers parse to floats
    (``"171"`` -> ``171.0``) so callers can rely on a uniform float
    type for all continuous stats.

    Args:
        text: The text to parse, e.g. text scraped from a stats cell.
            Leading/trailing whitespace and one optional trailing
            ``%`` are stripped before parsing.

    Returns:
        The parsed ``float``, or ``None`` if ``text`` (after
        stripping) is not a valid float literal (e.g. ``""``,
        ``"-"``, ``"TBD"``).
    """
    text = text.strip().rstrip("%")
    try:
        return float(text)
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


def _parse_player_stats_table(table_el, team_name: str) -> List[PlayerStats]:
    """Parse one team's player-stats table from a map's stats block.

    vlr.gg renders each completed map's per-player stats as two
    ``.ovw-table`` divs (not ``<table>`` elements) inside its
    ``.vm-stats-game`` block — one per team, in document order
    (team1 first, team2 second) — mirroring the DOM-order convention
    ``_parse_map`` already uses for scores. Each table has one header
    row (``.ovw-row.mod-head``, skipped) and one row per player
    (``.ovw-row``). A player row carries a ``.ovw-player-name``, a
    ``.ovw-agents`` cell with one ``<img>`` per agent the player used
    on the map, and one ``[data-col]`` cell per stat column; the
    column name lives in the ``data-col`` attribute (on the cell
    itself, or on inner ``.ovw-kda-stat`` spans for the K/D/A trio)
    and the map-total value in that cell's nested ``.side.mod-both``
    span. Only the ``mod-both`` value is read — the ``mod-t``/
    ``mod-ct`` half splits belong to a later milestone (roadmap M6).

    Args:
        table_el: A BeautifulSoup ``Tag`` for one ``.ovw-table``
            element within a per-map ``.vm-stats-game`` block.
        team_name: The resolved ``Team.name`` this table's players
            belong to (the plan's positional convention — the table's
            DOM position pins it to team1 or team2, so the
            ``.ovw-player-tag`` abbreviation is never needed). Stored
            on every returned ``PlayerStats``.

    Returns:
        A list of :class:`scraper.models.PlayerStats`, one per player
        row in table order. Empty/unparseable numeric cells parse to
        ``None`` via the same best-effort convention as
        ``_parse_int``/``_parse_float`` (e.g. a future ``"-"`` value),
        never raising; ``agents`` holds every agent image's ``alt``
        text in render order (an agent swap mid-map yields more than
        one entry).

    Raises:
        VlrParseError: If a player row has no ``.ovw-player-name``
            or it is empty — the table exists but its shape is
            broken, so this fails loudly (per the module's
            fail-loud-on-unrecognized-structure convention) rather
            than silently dropping the row.
    """
    players: List[PlayerStats] = []
    for row_el in table_el.select(".ovw-row"):
        if "mod-head" in (row_el.get("class") or []):
            continue
        name_el = row_el.select_one(".ovw-player-name")
        player_name = name_el.get_text(strip=True) if name_el is not None else ""
        if not player_name:
            raise VlrParseError(
                "player stats row without a player name (.ovw-player-name)"
            )
        cells = {}
        for el in row_el.select("[data-col]"):
            side_el = el.select_one(".side.mod-both")
            cells[el.get("data-col")] = (
                side_el.get_text(strip=True) if side_el is not None else ""
            )
        agents = [
            img.get("alt") for img in row_el.select(".ovw-agents img") if img.get("alt")
        ]
        players.append(
            PlayerStats(
                player_name=player_name,
                team_name=team_name,
                rating=_parse_float(cells.get("rating2", "")),
                acs=_parse_float(cells.get("acs", "")),
                kills=_parse_int(cells.get("kills", "")),
                deaths=_parse_int(cells.get("deaths", "")),
                assists=_parse_int(cells.get("assists", "")),
                adr=_parse_float(cells.get("adr", "")),
                kast=_parse_float(cells.get("kast", "")),
                hs_pct=_parse_float(cells.get("hsp", "")),
                first_kills=_parse_int(cells.get("fb", "")),
                first_deaths=_parse_int(cells.get("fd", "")),
                agents=agents,
            )
        )
    return players


def _parse_half_split(team_div_el):
    """Parse one team's attack/defense half-split round counts from a map header.

    vlr.gg renders each played map's per-team half breakdown as two or
    three sibling ``<span>`` elements inside the team's ``.team`` div
    in its ``.vm-stats-game-header`` (e.g. ``<span class="mod-ct">4</span>
    / <span class="mod-t">2</span>``). Span **DOM order is the half
    order**: the first span is always the team's first-half round
    count, the second its second-half count — which side a team
    started on is not fixed, so the half slot comes from position,
    never from the class — while each span's class names the side:
    ``mod-t`` = attacking that half, ``mod-ct`` = defending. A third
    ``mod-ot`` span (maps that went to overtime) carries the team's
    total OT rounds; it is read for completeness but not returned,
    since the header markup exposes OT only as a combined per-team
    total, not per side — so the returned atk/def totals are
    regulation-only by design (plan assumption).

    Args:
        team_div_el: A BeautifulSoup ``Tag`` for one ``.team`` div
            inside a ``.vm-stats-game-header``, containing the team's
            ``.score`` div plus the half ``<span>`` siblings.

    Returns:
        A 4-tuple ``(first_half_rounds, second_half_rounds,
        atk_rounds, def_rounds)``. The half values are the parsed
        round counts of the spans at DOM positions 0 and 1 (the
        regulation halves); ``atk_rounds``/``def_rounds`` are the sums
        of the ``mod-t``/``mod-ct`` spans (regulation only — a
        ``mod-ot`` value is excluded), or ``None`` for a side whose
        span never parsed — e.g. a live match's in-progress half
        rendering only the currently active side's count — never a
        fabricated ``0``, which would be indistinguishable from a team
        genuinely winning zero rounds on that side. ``(None, None,
        None, None)`` when no span carries a recognized
        ``mod-t``/``mod-ct`` class — e.g. an upcoming match's TBD
        placeholder block, whose spans have a bare ``mod-`` class —
        the same soft-missing treatment ``duration``/``winner`` get
        elsewhere in ``_parse_map``, not a raise.

    Raises:
        Nothing; unparseable span text (e.g. ``"-"``) is treated as
        missing via :func:`_parse_int`'s best-effort-``None``
        convention, and an unrecognized span class is ignored.
    """
    first_half_rounds: Optional[int] = None
    second_half_rounds: Optional[int] = None
    atk_rounds: Optional[int] = None
    def_rounds: Optional[int] = None
    recognized = 0
    for idx, span_el in enumerate(team_div_el.select("span")):
        classes = span_el.get("class") or []
        if "mod-t" in classes:
            side = "atk"
        elif "mod-ct" in classes:
            side = "def"
        else:
            # mod-ot (third span on OT maps) and the upcoming
            # placeholder's bare "mod-" class contribute to neither
            # the atk/def totals nor the regulation half slots.
            continue
        value = _parse_int(span_el.get_text(strip=True))
        if value is None:
            continue
        recognized += 1
        if side == "atk":
            atk_rounds = value if atk_rounds is None else atk_rounds + value
        else:
            def_rounds = value if def_rounds is None else def_rounds + value
        if idx == 0:
            first_half_rounds = value
        elif idx == 1:
            second_half_rounds = value
    if recognized == 0:
        return None, None, None, None
    return first_half_rounds, second_half_rounds, atk_rounds, def_rounds


def _parse_map(game_el, team1: Team, team2: Team) -> MapResult:
    """Parse one played map's result from a ``.vm-stats-game`` block.

    Args:
        game_el: A BeautifulSoup ``Tag`` for one ``.vm-stats-game``
            element, expected to contain a ``.vm-stats-game-header``
            with the map name, per-team score, win indicator and
            duration, plus (for real maps) two ``.ovw-table`` stat
            blocks, one per team.
        team1: The match's team1, used to resolve the map winner's
            name when the ``.score.mod-win`` element is on the
            left-hand (``mod-1``) side, and to label the first
            ``.ovw-table``'s players.
        team2: The match's team2, used to resolve the map winner's
            name when the ``.score.mod-win`` element is on the
            right-hand (``mod-right``) side, and to label the second
            ``.ovw-table``'s players.

    Returns:
        A :class:`scraper.models.MapResult` with the parsed map name,
        both teams' scores (``None`` for any that could not be parsed
        as an integer), the winner's team name (``None`` if no
        ``.score.mod-win`` element was found, or if either score is
        missing so the declared winner cannot be verified against the
        final scores), the map duration, and — when the map rendered
        two ``.ovw-table`` stat blocks — ``player_stats`` (every
        player-map stat line, team1 rows then team2 rows) and
        ``agent_picks`` (a dict mapping each team's resolved name to
        the list of agents its players used, one entry per player in
        table row order, first-listed agent only), and the eight
        half-split fields (``team1_first_half_rounds``,
        ``team1_second_half_rounds``, ``team1_atk_rounds``,
        ``team1_def_rounds`` and the team2 counterparts) parsed from
        the header's per-team ``mod-t``/``mod-ct``/``mod-ot`` spans
        (regulation-only atk/def totals; see
        :func:`_parse_half_split`). All eight are ``None`` when the
        header rendered no recognized half spans (e.g. an
        upcoming/TBD placeholder block), and the unparsed side's
        atk/def field is ``None`` (not a fabricated ``0``) when only
        one of the two sides' spans parsed (e.g. a live in-progress
        map mid-half). A map block with
        no ``.ovw-table`` at all (e.g. a future awarded/abandoned map
        that never rendered stats) yields ``player_stats == []`` and
        ``agent_picks is None`` — the same soft-missing treatment
        ``duration``/``winner`` get — while a block with exactly one
        table is a structural break and raises (see Raises).

    Raises:
        VlrParseError: If ``game_el`` has no ``.vm-stats-game-header``,
            or the header has no map name (``.map div span``), or the
            parsed half-split data violates a round-count invariant
            (combined first half != 12, or combined second half > 12,
            propagated from :meth:`MapResult.__post_init__` — checked
            only on finished maps, since a live in-progress map's
            partial counts legitimately violate them), or the
            parsed final score is illegal (a winner with fewer than
            13 rounds, or an overtime scoreline with margin < 2), or
            the winner label contradicts the final scores (the
            winning side must be the one with more rounds), or the
            block has a non-0/non-2 count of ``.ovw-table`` elements
            (one team's stats missing is a broken render, not a valid
            soft-missing state), or a player row inside a present
            table has no ``.ovw-player-name`` (propagated from
            :func:`_parse_player_stats_table`). The illegal-score,
            half-split-invariant and winner-mismatch cases include the
            map name and both scores in the message.
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

    # Attack/defense half splits from the header's per-team spans.
    # One .team div per side in DOM order (team1 first), the same
    # positional convention used for .team .score above. The
    # upcoming/TBD placeholder blocks render spans with a bare
    # "mod-" class; _parse_half_split returns all-None for those, so
    # placeholder data never reaches MapResult's half invariants.
    team_divs = header_el.select(".team")
    team1_half = (
        _parse_half_split(team_divs[0]) if len(team_divs) >= 1 else (None,) * 4
    )
    team2_half = (
        _parse_half_split(team_divs[1]) if len(team_divs) >= 2 else (None,) * 4
    )

    win_el = header_el.select_one(".score.mod-win")
    winner = None
    if win_el is not None:
        parent = win_el.find_parent("div", class_="team")
        if parent is not None and "mod-right" in (parent.get("class") or []):
            winner = team2.name
        else:
            winner = team1.name

    # A declared winner is only trustworthy when both final scores
    # parsed. If either score is missing (e.g. a forfeited/awarded map
    # renders "-" for one side), the win-element fallback above can
    # label the wrong team (it defaults to team1 when the ancestor
    # lookup fails), and MapResult's own validation skips maps with a
    # None score — so an unverified, possibly-wrong winner would reach
    # the cache unchecked. Drop the winner (None) instead: the map is
    # kept with its known score(s) but no asserted winner.
    if winner is not None and (team1_score is None or team2_score is None):
        winner = None

    # Cross-check the winner label against the final scores: a
    # finished map's winner must be the side with more rounds. This
    # catches the win-element fallback above (when the
    # ``.score.mod-win`` ancestor lookup fails and the code defaults
    # to ``team1`` even if team2 actually won) before an inconsistent
    # MapResult reaches the cache. ``MapResult`` itself cannot do this
    # check - it stores only the winner's *name*, not which side it
    # was - so it lives here where both team names and scores are
    # known. Equal scores with a declared winner are already rejected
    # below by ``MapResult`` score validation (a winner must reach 13
    # and OT margins must be >= 2), so they are skipped here. The
    # missing-score case (one side None) never reaches this check: the
    # winner was already dropped above as unverifiable.
    if (
        winner is not None
        and team1_score is not None
        and team2_score is not None
        and team1_score != team2_score
    ):
        expected_winner = (
            team1.name if team1_score > team2_score else team2.name
        )
        if winner != expected_winner:
            raise VlrParseError(
                f"map {map_name!r} winner {winner!r} does not match final "
                f"score {team1_score}-{team2_score} (expected "
                f"{expected_winner!r})"
            )

    duration_el = header_el.select_one(".map-duration")
    duration = duration_el.get_text(strip=True) if duration_el is not None else None

    # Per-player stats and agent picks. A real map renders exactly two
    # .ovw-table blocks (one per team, team1 in document order first),
    # each parsed positionally like the scores above. Zero tables is
    # the soft-missing case (e.g. a future awarded/abandoned map that
    # never rendered stats): player_stats stays [] and agent_picks
    # None, the same treatment _parse_map already gives duration/
    # winner. Any other count is a structural break — a half-rendered
    # table set would silently drop one team's stats, so it fails
    # loudly (per the fail-loud-on-unrecognized-structure convention)
    # instead. The upcoming-match placeholder blocks also carry two
    # tables; their rows have player names but empty stat cells and no
    # agents, which parse to None/[] here — parse_match discards the
    # whole block by its "TBD" map name afterwards, so no placeholder
    # data ever reaches the cache.
    tables = game_el.select(".ovw-table")
    player_stats: List[PlayerStats] = []
    agent_picks = None
    if len(tables) not in (0, 2):
        raise VlrParseError(
            f"map {map_name!r} has {len(tables)} .ovw-table blocks, "
            f"expected 0 (no stats rendered) or 2 (one per team)"
        )
    if len(tables) == 2:
        team1_stats = _parse_player_stats_table(tables[0], team1.name)
        team2_stats = _parse_player_stats_table(tables[1], team2.name)
        player_stats = team1_stats + team2_stats
        # agent_picks is a composition-summary convenience: exactly one
        # entry per player row, in table order, using only the
        # first-listed agent for players who swapped mid-map (vlr.gg's
        # markup does not label which agent was primary). The full swap
        # history is never lost — it stays on PlayerStats.agents. A
        # player with no agents at all contributes an empty string so
        # the one-entry-per-player / row-order alignment holds.
        agent_picks = {
            team1.name: [ps.agents[0] if ps.agents else "" for ps in team1_stats],
            team2.name: [ps.agents[0] if ps.agents else "" for ps in team2_stats],
        }

    try:
        return MapResult(
            map_name=map_name,
            team1_score=team1_score,
            team2_score=team2_score,
            winner=winner,
            duration=duration,
            agent_picks=agent_picks,
            player_stats=player_stats,
            team1_first_half_rounds=team1_half[0],
            team1_second_half_rounds=team1_half[1],
            team1_atk_rounds=team1_half[2],
            team1_def_rounds=team1_half[3],
            team2_first_half_rounds=team2_half[0],
            team2_second_half_rounds=team2_half[1],
            team2_atk_rounds=team2_half[2],
            team2_def_rounds=team2_half[3],
        )
    except ValueError as exc:
        # An illegal final score or a broken half-split invariant is a
        # data problem, not a programming error, so it surfaces
        # through the module's error taxonomy (VlrParseError) rather
        # than as a raw ValueError. It still aborts the whole match
        # parse — fail loudly, never silently skip-and-continue with
        # a wrong label.
        raise VlrParseError(
            f"invalid map data for map {map_name!r} "
            f"({team1_score}-{team2_score}): {exc}"
        ) from exc


def _parse_veto_note(note_text: str) -> List[VetoAction]:
    """Parse a match page's veto log free text into structured actions.

    vlr.gg renders a Bo3 match's bans/picks/decider as a single
    semicolon-separated string in the ``.match-header-note`` element,
    e.g. ``"NAVI ban Haven; FUT ban Breeze; NAVI pick Split; ...;
    Sunset remains"``. Each non-empty segment is matched against the
    ban/pick pattern (``"<team> ban <map>"`` / ``"<team> pick <map>"``)
    first, then the decider pattern (``"<map> remains"``); anything
    matching neither fails loudly via :class:`VlrParseError` rather
    than being silently skipped, so an unrecognized phrasing (e.g. a
    new vlr.gg wording) never turns into a silently wrong or missing
    veto action.

    Args:
        note_text: The stripped text of the ``.match-header-note``
            element, semicolon-separated. May be empty or contain
            empty/whitespace-only segments (e.g. a stray trailing
            ``";"``); those are dropped, not errors.

    Returns:
        A list of :class:`scraper.models.VetoAction` in segment order,
        with ``step_index`` numbered 0, 1, 2, ... over the emitted
        actions (empty segments do not consume an index). An empty or
        all-whitespace ``note_text`` yields an empty list.

    Raises:
        VlrParseError: If any non-empty segment matches neither the
            ban/pick pattern nor the decider pattern. The message
            includes the raw segment text and its 0-based index within
            the semicolon-separated note.
    """
    actions: List[VetoAction] = []
    for idx, segment in enumerate(note_text.split(";")):
        segment = segment.strip()
        if not segment:
            continue
        m = _VETO_BAN_PICK_RE.match(segment)
        if m is not None:
            actions.append(
                VetoAction(
                    step_index=len(actions),
                    team=m.group("team"),
                    action=m.group("action"),
                    map_name=m.group("map"),
                )
            )
            continue
        m = _VETO_DECIDER_RE.match(segment)
        if m is not None:
            actions.append(
                VetoAction(
                    step_index=len(actions),
                    team=None,  # forced, not chosen by either team
                    action="decider",
                    map_name=m.group("map"),
                )
            )
            continue
        raise VlrParseError(
            f"unrecognized veto note segment #{idx}: {segment!r}"
        )
    return actions


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _http_get(url: str) -> requests.Response:
    """GET ``url`` over HTTP with the standard UA header and timeout.

    Shared by :func:`fetch_page` and :func:`fetch_robots_parser` so the
    two fetches cannot drift apart in transport-level behavior (header,
    timeout, network-error conversion). Deliberately does *not* call
    ``raise_for_status``: status handling differs per caller
    (``fetch_page`` treats any non-2xx as fatal;
    ``fetch_robots_parser`` treats 404 as allow-all and any other
    non-2xx as fatal), so that part stays with the callers via
    :func:`_raise_for_status`.

    Args:
        url: The absolute URL to GET.

    Returns:
        The raw :class:`requests.Response` (status not yet checked).

    Raises:
        VlrFetchError: If the HTTP request fails at the transport level
            (network error or timeout — anything
            ``requests.RequestException`` covers before status
            checking). Non-2xx status codes are not converted here; see
            :func:`_raise_for_status`.
    """
    try:
        return requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        raise VlrFetchError(f"failed to fetch {url}: {exc}") from exc


def _raise_for_status(resp: requests.Response, url: str) -> None:
    """Raise :class:`VlrFetchError` if ``resp``'s status is not 2xx.

    Thin shared wrapper over ``requests.Response.raise_for_status`` used
    by both :func:`fetch_page` and :func:`fetch_robots_parser`, so the
    status-code-to-``VlrFetchError`` conversion (and its error-message
    shape) lives in exactly one place.

    Args:
        resp: The response whose status is being checked.
        url: The URL that was fetched, embedded in the error message to
            keep it identifiable.

    Returns:
        Nothing; returns normally for 2xx responses.

    Raises:
        VlrFetchError: If ``resp.raise_for_status()`` raises (a non-2xx
            status, e.g. 404 or 500).
    """
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise VlrFetchError(f"failed to fetch {url}: {exc}") from exc


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
    resp = _http_get(url)
    _raise_for_status(resp, url)
    html = resp.text
    if use_cache:
        set_cached_page(url, html)
    return html


# --------------------------------------------------------------------------
# Robots.txt (driver boundary)
# --------------------------------------------------------------------------
#
# The robots check lives one level up from fetch_page — it is a property of
# the CLI entry point (scrape.py), not of the page-fetch layer. Wiring it
# into fetch_page itself would fetch robots.txt on every cache miss and break
# the exact ``requests.get`` call-count assertions in tests/test_vlr.py, plus
# raise a recursion question (checking robots permission to fetch robots.txt).
# These are the reusable primitives: the CLI driver checks each event URL up
# front, and get_matches_from_event gates every individual match page against
# the same parser when one is passed in.


def fetch_robots_parser() -> RobotFileParser:
    """Fetch vlr.gg's robots.txt and return a parsed ``RobotFileParser``.

    Fetches ``ROBOTS_URL`` over HTTP (via :func:`_http_get`) rather than
    through :func:`fetch_page`, so the robots fetch never touches the
    page cache and never affects any existing fetch-page call-count
    assertion. The response body is fed to a fresh
    :class:`RobotFileParser` via :meth:`RobotFileParser.parse`, which
    populates its allow/disallow rules for subsequent
    :meth:`RobotFileParser.can_fetch` checks. Rules are matched under
    ``ROBOTS_USER_AGENT``, the scraper's explicit bot token.

    A missing robots.txt (HTTP 404) is *not* a fatal condition: by
    standard robots-exclusion convention (and
    ``urllib.robotparser.RobotFileParser.read``'s own behavior) a site
    with no robots.txt allows everything, so an empty parser whose
    ``can_fetch`` always returns ``True`` is returned instead of raising
    — a site that simply does not publish a robots.txt must not make the
    whole scrape run abort.

    Returns:
        A parsed :class:`RobotFileParser` for vlr.gg's robots.txt; an
        empty (allow-all) parser when vlr.gg returns 404.

    Raises:
        VlrFetchError: If the HTTP request fails (network error,
            timeout, or a non-2xx status other than 404) — the same
            conversion :func:`fetch_page` applies.
    """
    resp = _http_get(ROBOTS_URL)
    if resp.status_code == 404:
        # No robots.txt published: allow-all, not an error. The empty
        # parser must still be *parsed* (parse([]) sets last_checked):
        # an unparsed RobotFileParser conservatively denies everything
        # (mirroring urllib.robotparser's own read(), which treats a 404
        # as allow-all).
        parser = RobotFileParser()
        parser.parse([])
        return parser
    _raise_for_status(resp, ROBOTS_URL)
    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    return parser


def assert_allowed(url: str, rp: Optional[RobotFileParser] = None) -> None:
    """Assert that a URL is permitted by vlr.gg's robots.txt.

    When ``rp`` is ``None`` (the default), the robots file is fetched and
    parsed first via :func:`fetch_robots_parser`; callers that already
    fetched it once (e.g. ``scrape.main``, which checks every configured
    event URL against a single parser) pass it in to avoid N fetches of
    the same file.

    Args:
        url: The absolute URL to check, e.g. an event matches page from
            ``config.ACTIVE.event_urls``. Permission is checked at this
            URL's granularity (typically a path prefix, e.g. an
            ``/event/`` page); the CLI checks each configured event URL,
            not every individual match URL discovered from it.
        rp: An already-parsed :class:`RobotFileParser` to check against.
            ``None`` fetches a fresh one via
            :func:`fetch_robots_parser` (see Raises).

    Returns:
        Nothing; returns normally when
        ``rp.can_fetch(ROBOTS_USER_AGENT, url)`` is ``True`` (including
        the default-allow case where the robots file has no rule matching
        the URL). Robots rules are matched under ``ROBOTS_USER_AGENT``,
        the scraper's explicit bot token, rather than the browser-
        spoofing ``USER_AGENT`` string: ``RobotFileParser`` only matches
        the token before the first ``/``, so a targeted
        ``User-agent: <botname>`` block could never match ``Mozilla/...``.

    Raises:
        VlrRobotsError: If ``rp.can_fetch(ROBOTS_USER_AGENT, url)`` is
            ``False`` — the URL is disallowed and must not be fetched.
        VlrFetchError: If ``rp`` was ``None`` and the robots.txt fetch
            itself failed (propagated from :func:`fetch_robots_parser`).
    """
    if rp is None:
        rp = fetch_robots_parser()
    if not rp.can_fetch(ROBOTS_USER_AGENT, url):
        raise VlrRobotsError(f"robots.txt disallows fetching {url}")


# --------------------------------------------------------------------------
# Parsing (pure, no I/O)
# --------------------------------------------------------------------------


def parse_match(html: str, url: str) -> Match:
    """Parse a vlr.gg match page HTML string into a :class:`Match`.

    Pure function: does no network I/O, so it can be run against saved
    HTML fixtures in tests. Extracts the event name, both teams, the
    scheduled/played date, match status (``"completed"``/``"live"``/
    ``"upcoming"``), best-of format, overall scores, the list of
    per-map results (skipping the "All Maps" overview block and any
    placeholder ``"TBD"`` maps rendered for upcoming matches) — each
    map now also carrying its per-player stats
    (``MapResult.player_stats``), agent-pick summary
    (``MapResult.agent_picks``) parsed from the map's ``.ovw-table``
    blocks, and its attack/defense half-split round counts
    (``MapResult.team{1,2}_{first,second}_half_rounds`` and
    ``MapResult.team{1,2}_{atk,def}_rounds``) parsed from the
    header's per-team ``mod-t``/``mod-ct``/``mod-ot`` spans — and the
    ordered ban/pick/decider veto sequence parsed
    from the page's ``.match-header-note`` element (see
    :func:`_parse_veto_note`).

    Args:
        html: The full HTML of a vlr.gg match page.
        url: The URL the HTML was fetched from. Used to derive
            ``match_id`` (via :func:`extract_match_id`) and stored
            verbatim on the returned ``Match``.

    Returns:
        The parsed :class:`scraper.models.Match`, including a
        ``veto_actions`` list (empty when the page has no
        ``.match-header-note`` element, e.g. upcoming matches, or
        when the note's phrasing is unrecognized — see Raises).

    Raises:
        VlrParseError: If ``url`` is not a parseable match URL, or if
            any expected element is missing from ``html`` — no
            ``div.match-header``, no ``.match-header-event`` (and no
            fallback event slug), fewer than two
            ``.match-header-link`` team elements, or a per-map block
            with no map name. Unrecognized veto-note phrasing does
            *not* raise here: :func:`_parse_veto_note` still fails
            loudly on it at the unit level, but ``parse_match``
            catches that error, logs a warning, and leaves
            ``veto_actions`` empty — the same state as a match with
            no note — so one match's note can never abort the whole
            event fetch via :func:`get_matches_from_event`.
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

    # Veto log: free text in a single .match-header-note element, a
    # direct child of div.match-header. Most completed matches render
    # it; upcoming matches (and a few completed ones, e.g.
    # match_page_single_ot.html) have no such element at all. An
    # absent or empty note yields an empty list, matching the existing
    # convention for maps/scores that aren't available yet.
    # _parse_veto_note raises VlrParseError on unrecognized phrasing
    # rather than silently dropping an action; that error is caught
    # here (not propagated) so one match's unrecognized note can never
    # discard every other match already parsed from the same event
    # fetch via get_matches_from_event. The veto data is left empty —
    # the same state as a match with no note element — and the
    # failure is logged loudly rather than being silent.
    note_el = header.select_one(".match-header-note")
    veto_actions: List[VetoAction] = []
    if note_el is not None:
        try:
            veto_actions = _parse_veto_note(note_el.get_text(strip=True))
        except VlrParseError as exc:
            logger.warning(
                "match %s (%s): unrecognized veto note phrasing; "
                "veto_actions left empty: %s",
                match_id,
                url,
                exc,
            )

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
        veto_actions=veto_actions,
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
            :func:`parse_match`). Unrecognized veto-note phrasing is
            not one of these cases: ``parse_match`` catches it
            internally, logs a warning, and leaves
            ``Match.veto_actions`` empty.
        VlrFetchError: If fetching the page over HTTP fails
            (propagated from :func:`fetch_page`).
        IllegalScoreError: If a cached row for the match deserializes
            to an illegal final map score (propagated from
            :func:`cache.get_cached_match`; a ``ValueError`` subclass,
            so callers catching ``ValueError`` still see it). The same
            failure on a freshly fetched page is raised as
            :class:`VlrParseError` instead (via :func:`_parse_map`).
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


def get_matches_from_event(
    event_url: str,
    use_cache: bool = True,
    robots_parser: Optional[RobotFileParser] = None,
) -> List[Match]:
    """Fetch (or load from cache) every match listed on an event page.

    Fetches the event page, extracts its match links
    (:func:`parse_event_match_links`), then fetches/parses each match
    in page order (:func:`get_match`). A small delay
    (``POLITE_DELAY_SECONDS``) is inserted between consecutive
    *uncached* match fetch attempts to be polite to vlr.gg; cached
    matches incur no delay. Two per-match gates sit between the event
    page and each match fetch:

    - When ``robots_parser`` is given, every match URL is checked with
      :func:`assert_allowed` first and a disallowed match is logged and
      skipped. (The CLI driver checks the event URL up front, but that
      only covers the listing page; without this per-match gate the
      majority of the requests this function makes would go out with
      zero permission check.)
    - A match that fails to fetch or parse (``VlrFetchError`` /
      ``VlrParseError`` / ``IllegalScoreError``) is logged and skipped
      rather than aborting the whole event, so matches already parsed
      and durably cached before the failure still count in the caller's
      run summary. The cached fast path is covered by the same
      isolation: a cached row that deserializes to an illegal final map
      score (``IllegalScoreError`` raised by
      :func:`cache.get_cached_match`) is logged and skipped too, not
      propagated to the caller.

    Args:
        event_url: The vlr.gg event matches page URL.
        use_cache: When ``True`` (the default), read from and write to
            the page and match caches for both the event page and
            every individual match. When ``False``, always fetch
            fresh and never touch either cache (and every match fetch
            is treated as uncached, so the polite delay applies
            between all of them).
        robots_parser: An already-parsed :class:`RobotFileParser`
            (normally the one ``scrape.main`` fetched once up front)
            to gate every match URL with before fetching it. ``None``
            (the default) performs no per-match robots check, keeping
            the function's existing behavior for callers that do not
            run under the robots gate.

    Returns:
        A list of parsed :class:`scraper.models.Match` objects, one
        per match link found on the event page that was not skipped,
        in page order. Matches skipped by the robots gate or by a
        per-match fetch/parse failure are not included (each skip is
        logged as a warning).

    Raises:
        VlrFetchError: If fetching the *event page* over HTTP fails.
            Per-match fetch failures no longer raise here: they are
            logged and the match is skipped.
        VlrParseError: If the event page is missing expected structure.
            Unrecognized veto-note phrasing does not raise —
            ``parse_match`` catches it, logs a warning, and leaves that
            match's ``veto_actions`` empty — so a single match's veto
            note can never discard the other matches already parsed
            from the event.
        IllegalScoreError: No longer raised from a per-match failure on
            either the uncached path or the cached fast path — a
            cached row deserializing to an illegal final map score
            (raised by :func:`cache.get_cached_match`) is caught,
            logged, and skipped like the other two per-match failures.
            It can still escape :func:`get_match` when that function is
            called directly.
    """
    html = fetch_page(event_url, use_cache=use_cache)
    links = parse_event_match_links(html)
    matches: List[Match] = []
    fetched = False
    for link in links:
        url = link if link.startswith("http") else BASE_URL + link
        if robots_parser is not None:
            try:
                assert_allowed(url, robots_parser)
            except VlrRobotsError as exc:
                logger.warning(
                    "robots.txt disallows match %s: %s; skipping", url, exc
                )
                continue
        try:
            if use_cache and get_cached_match(extract_match_id(url)) is not None:
                matches.append(get_match(url, use_cache=True))
                continue
            if fetched:
                time.sleep(POLITE_DELAY_SECONDS)
            fetched = True
            matches.append(get_match(url, use_cache=use_cache))
        except (VlrFetchError, VlrParseError, IllegalScoreError) as exc:
            # One bad match must not discard the matches already
            # parsed/cached for this event: log and move on, so a
            # partial run's summary still counts what succeeded. The
            # same catch covers the cached fast path, where a corrupt
            # cached row deserializing to an illegal score raises
            # IllegalScoreError from get_cached_match (both the
            # condition check above and the call inside get_match).
            logger.warning("match %s failed: %s; skipping", url, exc)
    return matches
