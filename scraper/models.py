"""Custom match data format for vctpredictor.

Pure dataclasses with no scraping/caching dependencies, so that
``scraper.cache`` can serialize/deserialize them without importing any
parsing logic. ``to_dict``/``from_dict`` are the canonical
serialization contract used by the SQLite cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


class IllegalScoreError(ValueError):
    """A finished map's scoreline or half-split data violates VCT rules.

    Raised by :meth:`MapResult.__post_init__` when a finished map's
    scoreline is impossible (a winner with fewer than 13 rounds, an
    overtime scoreline — both teams >= 12 — with a winning margin
    below 2 rounds, or a regulation scoreline whose winner exceeds 13
    rounds), or when a parsed half-split invariant is broken (the
    combined first-half round count is not exactly 12, or the combined
    second-half count exceeds 12). Subclasses ``ValueError`` so
    existing ``except ValueError`` handlers keep working, but gives
    callers a way to distinguish a score/half-split validity failure
    from unrelated ``ValueError`` sources (e.g. a corrupt cache row
    whose ``date`` field fails ``datetime.fromisoformat``).
    """


@dataclass
class Team:
    """A competing team.

    Attributes:
        name: The team's display name (e.g. ``"Fnatic"``).
        team_id: vlr.gg's numeric team id (as a string), or ``None``
            if it could not be determined (e.g. no team-page link was
            found on the source page).
    """

    name: str
    team_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize this team to a JSON-compatible dict.

        Returns:
            A dict with keys ``"name"`` and ``"team_id"``, suitable
            for ``json.dumps`` and for round-tripping via
            :meth:`from_dict`.
        """
        return {"name": self.name, "team_id": self.team_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        """Reconstruct a ``Team`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing a required ``"name"`` key and an optional
                ``"team_id"`` key.

        Returns:
            The reconstructed ``Team``.

        Raises:
            KeyError: If ``data`` has no ``"name"`` key.
        """
        return cls(name=data["name"], team_id=data.get("team_id"))


@dataclass
class PlayerStats:
    """Per-player, per-map statistics from a vlr.gg stats table.

    One instance per player row in a map's ``.ovw-table`` block (see
    :func:`scraper.vlr._parse_player_stats_table`). All numeric
    fields are parsed best-effort from the table's ``.side.mod-both``
    cells — the map-total values, not the per-half ``mod-t``/``mod-ct``
    splits, which belong to a later milestone — and are ``None`` when
    the cell is empty or unparseable (e.g. a future ``"-"`` value).
    ``kast`` and ``hs_pct`` are percentages stored without the ``%``
    sign (``74.0`` for ``"74%"``).

    Attributes:
        player_name: The player's in-game name.
        team_name: Name of the team the player was on for this stat
            line (matches a ``Team.name``, not a foreign key).
        rating: The player's vlr.gg "Rating 2.0" stat, or ``None`` if
            unavailable/unparseable.
        acs: The player's Average Combat Score, or ``None`` if
            unavailable/unparseable.
        kills: Kills on this map, or ``None`` if unavailable.
        deaths: Deaths on this map, or ``None`` if unavailable.
        assists: Assists on this map, or ``None`` if unavailable.
        adr: Average Damage per Round, or ``None`` if unavailable.
        kast: KAST percentage as a plain float (``74.0`` for
            ``"74%"``), or ``None`` if unavailable.
        hs_pct: Headshot percentage as a plain float (``27.0`` for
            ``"27%"``), or ``None`` if unavailable.
        first_kills: First-kill count on this map, or ``None`` if
            unavailable. The raw count vlr.gg renders under its "FK"
            column — not a per-round rate (plan assumption 2).
        first_deaths: First-death count on this map, or ``None`` if
            unavailable. The raw count from vlr.gg's "FD" column —
            not a per-round rate (plan assumption 2).
        agents: The agents this player used on the map, in the order
            vlr.gg lists them (an agent swap mid-map yields more than
            one entry, in swap order — plan assumption 1). Defaults
            to an empty list.
    """

    player_name: str
    team_name: str
    rating: Optional[float] = None
    acs: Optional[float] = None
    kills: Optional[int] = None
    deaths: Optional[int] = None
    assists: Optional[int] = None
    adr: Optional[float] = None
    kast: Optional[float] = None
    hs_pct: Optional[float] = None
    first_kills: Optional[int] = None
    first_deaths: Optional[int] = None
    agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize these stats to a JSON-compatible dict.

        Returns:
            A dict with keys ``"player_name"``, ``"team_name"``,
            ``"rating"``, ``"acs"``, ``"kills"``, ``"deaths"``,
            ``"assists"``, ``"adr"``, ``"kast"``, ``"hs_pct"``,
            ``"first_kills"``, ``"first_deaths"`` and ``"agents"``
            (a copy of the agents list), suitable for ``json.dumps``
            and for round-tripping via :meth:`from_dict`.
        """
        return {
            "player_name": self.player_name,
            "team_name": self.team_name,
            "rating": self.rating,
            "acs": self.acs,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "adr": self.adr,
            "kast": self.kast,
            "hs_pct": self.hs_pct,
            "first_kills": self.first_kills,
            "first_deaths": self.first_deaths,
            "agents": list(self.agents),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerStats":
        """Reconstruct a ``PlayerStats`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing required ``"player_name"`` and
                ``"team_name"`` keys and optional keys for every
                other field (``"agents"`` defaults to an empty list
                when absent, so rows cached before this task parsed
                stats still deserialize).

        Returns:
            The reconstructed ``PlayerStats``.

        Raises:
            KeyError: If ``data`` has no ``"player_name"`` or
                ``"team_name"`` key.
        """
        return cls(
            player_name=data["player_name"],
            team_name=data["team_name"],
            rating=data.get("rating"),
            acs=data.get("acs"),
            kills=data.get("kills"),
            deaths=data.get("deaths"),
            assists=data.get("assists"),
            adr=data.get("adr"),
            kast=data.get("kast"),
            hs_pct=data.get("hs_pct"),
            first_kills=data.get("first_kills"),
            first_deaths=data.get("first_deaths"),
            agents=list(data.get("agents", [])),
        )


@dataclass
class MapResult:
    """Result of a single played map.

    Attributes:
        map_name: Name of the map played (e.g. ``"Ascent"``).
        team1_score: Rounds won by team1 on this map, or ``None`` if
            unknown/not yet played.
        team2_score: Rounds won by team2 on this map, or ``None`` if
            unknown/not yet played.
        winner: Name of the winning team, or ``None`` if unknown (this
            is a team *name* string, not a ``Team`` reference).
        duration: Map duration as displayed on vlr.gg, e.g.
            ``"59:20"``, or ``None`` if not shown.
        agent_picks: Composition-summary of each team's agent picks
            on this map: a dict mapping each team's resolved name to
            the list of agents its players used, one entry per player
            in table row order (only the first-listed agent for a
            player who swapped mid-map — the full swap history stays
            on that player's ``PlayerStats.agents``). ``None`` when
            the map rendered no stats tables at all.
        player_stats: Every player-map stat line for this map, both
            teams combined, in the order the tables render
            ``(team1 rows..., team2 rows...)``. Empty when the map
            rendered no stats tables.
        team1_first_half_rounds: Rounds team1 won in the map's first
            half (regulation only), or ``None`` when the header
            rendered no recognized half spans (e.g. an upcoming/TBD
            placeholder block). A regulation first half always runs
            its full 12 rounds, so team1's and team2's first-half
            counts always sum to exactly 12.
        team1_second_half_rounds: Rounds team1 won in the map's
            second half (regulation only), or ``None`` when
            unavailable. Unlike the first half, the second half may be
            truncated — a team reaching 13 rounds ends the game
            mid-half — so the combined second-half count can fall
            below 12 (it can never exceed it).
        team1_atk_rounds: Total regulation rounds team1 won while
            attacking (the sum of its ``mod-t`` half spans), or
            ``None`` when no half data parsed. OT rounds are excluded
            (vlr.gg's header markup reports OT only as a combined
            per-team total, not per side), so ``atk + def`` always
            equals ``first + second`` half rounds.
        team1_def_rounds: Total regulation rounds team1 won while
            defending (the sum of its ``mod-ct`` half spans), or
            ``None`` when no half data parsed.
        team2_first_half_rounds: Rounds team2 won in the map's first
            half (regulation only), or ``None`` when unavailable.
        team2_second_half_rounds: Rounds team2 won in the map's
            second half (regulation only), or ``None`` when
            unavailable.
        team2_atk_rounds: Total regulation rounds team2 won while
            attacking (the sum of its ``mod-t`` half spans), or
            ``None`` when no half data parsed.
        team2_def_rounds: Total regulation rounds team2 won while
            defending (the sum of its ``mod-ct`` half spans), or
            ``None`` when no half data parsed.

    Raises:
        IllegalScoreError (a ``ValueError`` subclass): In
            :meth:`__post_init__`, if the map is finished (all three
            of ``team1_score``, ``team2_score`` and ``winner`` set)
            with an illegal scoreline: a winner with fewer than 13
            rounds, an overtime scoreline (both teams >= 12) with a
            winning margin below 2, or a regulation scoreline (loser
            below 12) whose winner exceeds 13 rounds; or, whenever
            both teams' half-split data parsed (even for unfinished
            maps), if the combined first-half round count is not
            exactly 12 or the combined second-half count exceeds 12.
    """

    map_name: str
    team1_score: Optional[int]
    team2_score: Optional[int]
    winner: Optional[str]  # name of the winning team (None if unknown)
    duration: Optional[str] = None  # e.g. "59:20" as displayed on vlr.gg
    agent_picks: Optional[dict[str, list[str]]] = None  # team name -> per-player agent list
    player_stats: list[PlayerStats] = field(default_factory=list)
    team1_first_half_rounds: Optional[int] = None  # regulation first half
    team1_second_half_rounds: Optional[int] = None  # regulation second half
    team1_atk_rounds: Optional[int] = None  # regulation rounds attacking
    team1_def_rounds: Optional[int] = None  # regulation rounds defending
    team2_first_half_rounds: Optional[int] = None  # regulation first half
    team2_second_half_rounds: Optional[int] = None  # regulation second half
    team2_atk_rounds: Optional[int] = None  # regulation rounds attacking
    team2_def_rounds: Optional[int] = None  # regulation rounds defending

    def __post_init__(self) -> None:
        """Validate a finished map's score and half-split data after construction.

        Two independent validation layers run here, both raising
        :class:`IllegalScoreError` (a ``ValueError`` subclass).

        The half-split layer runs whenever *both* teams' half data
        parsed — it is independent of the finished-map gate, so it
        also fires on live/unfinished maps whose scores/winner are
        still ``None``. It enforces the two round-count invariants
        that hold on every real vlr.gg header: the combined first-half
        round count of both teams is always exactly 12 (a regulation
        first half always runs its full 12 rounds), and the combined
        second-half count never exceeds 12 (the second half *may* be
        truncated — a team reaching 13 rounds ends the game mid-half,
        so e.g. a 13-6 map's second half sums to fewer than 12 — but
        can never exceed it).

        The finished-map layer runs only when ``team1_score``,
        ``team2_score`` and ``winner`` are all not ``None``
        (unfinished/live/upcoming maps have no final labels yet and
        are skipped). For finished maps it enforces the standard VCT
        rules: the winner must have at least 13 rounds
        (``winner_score >= 13``); when both teams reached 12 rounds
        (overtime), the winning margin must be at least 2 rounds
        (``winner_score - loser_score >= 2``); and a regulation win
        (loser below 12) must end at exactly 13 rounds
        (``winner_score == 13``) since a regulation game stops the
        moment a team reaches 13 — so a score like 30-3 or 14-11 is
        impossible. This is validation only — no ``is_overtime`` field
        is persisted here; deriving an OT flag belongs to a later
        milestone against the materialized dataset.

        Raises:
            IllegalScoreError (a ``ValueError`` subclass): If both
                teams' half-split data parsed and the combined
                first-half round count is not exactly 12, or the
                combined second-half count exceeds 12 (the message
                includes ``map_name`` and both teams' half values); or
                if the map is finished and ``winner_score < 13`` (a
                winner cannot have fewer than 13 rounds), the map went
                to overtime (both scores >= 12) and the winning
                margin is less than 2 rounds (e.g. 13-12), or the map
                is a regulation scoreline (loser below 12) whose
                winner exceeds 13 rounds (e.g. 30-3). The score
                messages include ``map_name`` and both scores.
        """
        # Half-split invariant, independent of the finished-map gate
        # below: runs whenever both teams' half data parsed, even when
        # scores/winner are still None. A regulation first half always
        # runs its full 12 rounds, so the combined first-half count
        # must be exactly 12. The combined second-half count is capped
        # at 12 but may be less: a team reaching 13 rounds mid-half
        # ends the game, so truncated second halves (e.g. a 13-6 map)
        # sum to under 12 — never over. (The upcoming-placeholder
        # "TBD" blocks render no recognized half spans, so they parse
        # to None and skip this check entirely.)
        if (
            self.team1_first_half_rounds is not None
            and self.team2_first_half_rounds is not None
        ):
            first_half_sum = (
                self.team1_first_half_rounds + self.team2_first_half_rounds
            )
            if first_half_sum != 12:
                raise IllegalScoreError(
                    f"map {self.map_name!r} has an illegal combined "
                    f"first-half round count {first_half_sum} (team1 "
                    f"{self.team1_first_half_rounds}, team2 "
                    f"{self.team2_first_half_rounds}): a regulation "
                    f"first half is always exactly 12 rounds"
                )
        if (
            self.team1_second_half_rounds is not None
            and self.team2_second_half_rounds is not None
        ):
            second_half_sum = (
                self.team1_second_half_rounds + self.team2_second_half_rounds
            )
            if second_half_sum > 12:
                raise IllegalScoreError(
                    f"map {self.map_name!r} has an illegal combined "
                    f"second-half round count {second_half_sum} (team1 "
                    f"{self.team1_second_half_rounds}, team2 "
                    f"{self.team2_second_half_rounds}): a second half "
                    f"can be truncated (a team reaching 13 ends the "
                    f"game) but never exceeds 12 rounds"
                )

        if (
            self.team1_score is None
            or self.team2_score is None
            or self.winner is None
        ):
            return
        winner_score = max(self.team1_score, self.team2_score)
        loser_score = min(self.team1_score, self.team2_score)
        if winner_score < 13:
            raise IllegalScoreError(
                f"map {self.map_name!r} has winner score {winner_score} < 13 "
                f"(scores {self.team1_score}-{self.team2_score})"
            )
        is_overtime = loser_score >= 12
        if is_overtime:
            if winner_score - loser_score < 2:
                raise IllegalScoreError(
                    f"map {self.map_name!r} went to overtime with an illegal "
                    f"margin: winner {winner_score} vs loser {loser_score} "
                    f"(scores {self.team1_score}-{self.team2_score}); "
                    f"overtime wins must have margin >= 2"
                )
        elif winner_score > 13:
            # A regulation game (loser below 12) ends the moment a
            # team reaches 13 rounds, so a regulation win is always
            # exactly 13-<loser>. A winner with more than 13 rounds
            # and a sub-12 loser (e.g. 30-3 or 14-11) is impossible:
            # the only way past 13 is overtime, which requires both
            # teams to have reached 12 (handled above).
            raise IllegalScoreError(
                f"map {self.map_name!r} has an impossible regulation score "
                f"{self.team1_score}-{self.team2_score}: a regulation win "
                f"ends at 13 rounds, but the winner has {winner_score} "
                f"rounds and the loser only {loser_score} (< 12, so not "
                f"overtime)"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this map result to a JSON-compatible dict.

        Reads fields off the already-constructed instance and never
        re-runs :meth:`__post_init__` validation, so it does not
        raise ``ValueError``; an illegal scoreline is rejected at
        construction time (:meth:`__init__` / :meth:`from_dict`)
        before this is ever called.

        Returns:
            A dict with keys ``"map_name"``, ``"team1_score"``,
            ``"team2_score"``, ``"winner"``, ``"duration"``,
            ``"agent_picks"``, ``"player_stats"`` (each entry
            serialized via :meth:`PlayerStats.to_dict`) and the eight
            half-split fields (``team1_first_half_rounds``,
            ``team1_second_half_rounds``, ``team1_atk_rounds``,
            ``team1_def_rounds``, ``team2_first_half_rounds``,
            ``team2_second_half_rounds``, ``team2_atk_rounds``,
            ``team2_def_rounds``), suitable for ``json.dumps`` and for
            round-tripping via :meth:`from_dict`.
        """
        return {
            "map_name": self.map_name,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "winner": self.winner,
            "duration": self.duration,
            "agent_picks": self.agent_picks,
            "player_stats": [ps.to_dict() for ps in self.player_stats],
            "team1_first_half_rounds": self.team1_first_half_rounds,
            "team1_second_half_rounds": self.team1_second_half_rounds,
            "team1_atk_rounds": self.team1_atk_rounds,
            "team1_def_rounds": self.team1_def_rounds,
            "team2_first_half_rounds": self.team2_first_half_rounds,
            "team2_second_half_rounds": self.team2_second_half_rounds,
            "team2_atk_rounds": self.team2_atk_rounds,
            "team2_def_rounds": self.team2_def_rounds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapResult":
        """Reconstruct a ``MapResult`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing a required ``"map_name"`` key and optional
                ``"team1_score"``, ``"team2_score"``, ``"winner"``,
                ``"duration"``, ``"agent_picks"``, ``"player_stats"``
                (a list of :meth:`PlayerStats.to_dict` dicts) and the
                eight half-split keys. ``"player_stats"`` defaults to
                an empty list when absent, and each half-split key
                defaults to ``None``, so cache rows written before
                stats or half-split parsing existed still deserialize.

        Returns:
            The reconstructed ``MapResult``.

        Raises:
            KeyError: If ``data`` has no ``"map_name"`` key.
            IllegalScoreError (a ``ValueError`` subclass): If the
                reconstructed map is finished with an illegal
                scoreline, or its half-split data violates the
                round-count invariants (propagated from
                :meth:`__post_init__`).
        """
        return cls(
            map_name=data["map_name"],
            team1_score=data.get("team1_score"),
            team2_score=data.get("team2_score"),
            winner=data.get("winner"),
            duration=data.get("duration"),
            agent_picks=data.get("agent_picks"),
            player_stats=[
                PlayerStats.from_dict(ps) for ps in data.get("player_stats", [])
            ],
            team1_first_half_rounds=data.get("team1_first_half_rounds"),
            team1_second_half_rounds=data.get("team1_second_half_rounds"),
            team1_atk_rounds=data.get("team1_atk_rounds"),
            team1_def_rounds=data.get("team1_def_rounds"),
            team2_first_half_rounds=data.get("team2_first_half_rounds"),
            team2_second_half_rounds=data.get("team2_second_half_rounds"),
            team2_atk_rounds=data.get("team2_atk_rounds"),
            team2_def_rounds=data.get("team2_def_rounds"),
        )


@dataclass
class VetoAction:
    """One step of a match's veto/bans-and-picks sequence.

    Parsed from the free-text ``.match-header-note`` element on a
    vlr.gg match page (see :func:`scraper.vlr._parse_veto_note`). The
    action sequence is ordered: two bans, two picks, two bans and a
    decider in a standard Bo3 (``ban, ban, pick, pick, ban, ban,
    decider``), with different shapes for Bo1/Bo5 formats.

    Attributes:
        step_index: 0-based position of this action in the veto
            sequence (0 for the first ban, 1 for the second, etc.).
        team: The acting team's token exactly as it appears in the
            note (e.g. ``"NAVI"``, ``"FUT"``) — a vlr.gg
            abbreviation, *not* resolved to a ``Team.name`` (the
            short forms are not mechanically derivable from the full
            names, so resolving them risks silently mislabeling the
            acting team). ``None`` for a decider action, which is
            forced rather than chosen.
        action: One of ``"ban"``, ``"pick"`` or ``"decider"``.
        map_name: The map named in the segment (e.g. ``"Haven"``).
    """

    step_index: int
    team: Optional[str]
    action: str
    map_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize this veto action to a JSON-compatible dict.

        Returns:
            A dict with keys ``"step_index"``, ``"team"``,
            ``"action"`` and ``"map_name"``, suitable for
            ``json.dumps`` and for round-tripping via
            :meth:`from_dict`.
        """
        return {
            "step_index": self.step_index,
            "team": self.team,
            "action": self.action,
            "map_name": self.map_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VetoAction":
        """Reconstruct a ``VetoAction`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing required ``"step_index"``, ``"action"``
                and ``"map_name"`` keys and an optional ``"team"``
                key (``None`` for a decider action).

        Returns:
            The reconstructed ``VetoAction``.

        Raises:
            KeyError: If ``data`` has no ``"step_index"``,
                ``"action"`` or ``"map_name"`` key.
        """
        return cls(
            step_index=data["step_index"],
            team=data.get("team"),
            action=data["action"],
            map_name=data["map_name"],
        )


@dataclass
class Match:
    """A single vlr.gg match (completed, live or upcoming).

    Attributes:
        match_id: vlr.gg's numeric match id (as a string); see
            :func:`scraper.vlr.extract_match_id`.
        url: The source match page URL.
        event_name: Name of the event/tournament the match belongs to.
        date: A naive UTC ``datetime`` parsed from vlr.gg's
            ``data-utc-ts`` attribute, or ``None`` if unavailable
            (e.g. an upcoming match with no schedule yet).
        team1: The first-listed (left-side) competing team.
        team2: The second-listed (right-side) competing team.
        team1_score: Overall maps won by team1 (not rounds), or
            ``None`` if unknown/not yet played.
        team2_score: Overall maps won by team2 (not rounds), or
            ``None`` if unknown/not yet played.
        best_of: The match format, e.g. ``"Bo3"``, ``"Bo5"``, or
            ``None`` if not shown.
        maps: Per-map results in the order they were played. Defaults
            to an empty list.
        veto_actions: The ordered ban/pick/decider sequence parsed
            from the page's ``.match-header-note`` element (see
            :class:`VetoAction`). Defaults to an empty list — the
            value for matches whose page renders no note (e.g.
            upcoming matches).
        status: One of ``"completed"``, ``"live"``, ``"upcoming"``.
            Defaults to ``"upcoming"``.
    """

    match_id: str
    url: str
    event_name: str
    date: Optional[datetime]
    team1: Team
    team2: Team
    team1_score: Optional[int]
    team2_score: Optional[int]
    best_of: Optional[str]  # e.g. "Bo3", "Bo5"
    maps: list[MapResult] = field(default_factory=list)
    veto_actions: list[VetoAction] = field(default_factory=list)
    status: str = "upcoming"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this match to a JSON-compatible dict.

        ``team1``/``team2``, each entry of ``maps`` and each entry of
        ``veto_actions`` are recursively serialized via their own
        ``to_dict``, and ``date`` is rendered as an ISO-8601 string
        (or ``None``), so the result is safe to pass to ``json.dumps``
        — this is exactly what ``scraper.cache.set_cached_match`` does
        before storing a match.

        Returns:
            A dict with keys ``"match_id"``, ``"url"``,
            ``"event_name"``, ``"date"``, ``"team1"``, ``"team2"``,
            ``"team1_score"``, ``"team2_score"``, ``"best_of"``,
            ``"maps"``, ``"veto_actions"`` and ``"status"``,
            suitable for round-tripping via :meth:`from_dict`.
        """
        return {
            "match_id": self.match_id,
            "url": self.url,
            "event_name": self.event_name,
            "date": self.date.isoformat() if self.date is not None else None,
            "team1": self.team1.to_dict(),
            "team2": self.team2.to_dict(),
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "best_of": self.best_of,
            "maps": [m.to_dict() for m in self.maps],
            "veto_actions": [v.to_dict() for v in self.veto_actions],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Match":
        """Reconstruct a ``Match`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing required ``"match_id"``, ``"url"``,
                ``"event_name"``, ``"team1"`` and ``"team2"`` keys,
                and optional ``"date"`` (an ISO-8601 string or
                ``None``), ``"team1_score"``, ``"team2_score"``,
                ``"best_of"``, ``"maps"`` (a list of
                :meth:`MapResult.to_dict` dicts), ``"veto_actions"`` (a
                list of :meth:`VetoAction.to_dict` dicts) and
                ``"status"`` keys.

        Returns:
            The reconstructed ``Match``, with nested ``team1``/
            ``team2``/``maps``/``veto_actions`` reconstructed via
            their own ``from_dict``.

        Raises:
            KeyError: If ``data`` is missing ``"match_id"``, ``"url"``,
                ``"event_name"``, ``"team1"`` or ``"team2"``.
            ValueError: If ``"date"`` is present and not a valid
                ISO-8601 string (propagated from
                ``datetime.fromisoformat``).
            IllegalScoreError (a ``ValueError`` subclass): If any map
                in ``"maps"`` deserializes to an illegal final score
                (propagated from :meth:`MapResult.from_dict`, which
                validates each map via
                :meth:`MapResult.__post_init__`).
        """
        raw_date = data.get("date")
        return cls(
            match_id=data["match_id"],
            url=data["url"],
            event_name=data["event_name"],
            date=datetime.fromisoformat(raw_date) if raw_date else None,
            team1=Team.from_dict(data["team1"]),
            team2=Team.from_dict(data["team2"]),
            team1_score=data.get("team1_score"),
            team2_score=data.get("team2_score"),
            best_of=data.get("best_of"),
            maps=[MapResult.from_dict(m) for m in data.get("maps", [])],
            veto_actions=[
                VetoAction.from_dict(v) for v in data.get("veto_actions", [])
            ],
            status=data.get("status", "upcoming"),
        )
