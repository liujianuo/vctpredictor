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
    """Minimal per-player stats.

    Reserved for future stats scraping (not populated by the current
    parsers). Kept in the model so the schema exists before the
    stats tables on vlr.gg get parsed.

    Attributes:
        player_name: The player's in-game name.
        team_name: Name of the team the player was on for this stat
            line (matches a ``Team.name``, not a foreign key).
        rating: The player's vlr.gg "Rating" stat, or ``None`` if not
            yet populated.
        acs: The player's Average Combat Score, or ``None`` if not yet
            populated.
    """

    player_name: str
    team_name: str
    rating: Optional[float] = None
    acs: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize these stats to a JSON-compatible dict.

        Returns:
            A dict with keys ``"player_name"``, ``"team_name"``,
            ``"rating"`` and ``"acs"``, suitable for ``json.dumps`` and
            for round-tripping via :meth:`from_dict`.
        """
        return {
            "player_name": self.player_name,
            "team_name": self.team_name,
            "rating": self.rating,
            "acs": self.acs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerStats":
        """Reconstruct a ``PlayerStats`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing required ``"player_name"`` and
                ``"team_name"`` keys and optional ``"rating"`` and
                ``"acs"`` keys.

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
        agent_picks: Reserved for future agent-pick data; not
            populated by the current parsers.
    """

    map_name: str
    team1_score: Optional[int]
    team2_score: Optional[int]
    winner: Optional[str]  # name of the winning team (None if unknown)
    duration: Optional[str] = None  # e.g. "59:20" as displayed on vlr.gg
    agent_picks: Optional[dict[str, Any]] = None  # reserved; not populated yet

    def to_dict(self) -> dict[str, Any]:
        """Serialize this map result to a JSON-compatible dict.

        Returns:
            A dict with keys ``"map_name"``, ``"team1_score"``,
            ``"team2_score"``, ``"winner"``, ``"duration"`` and
            ``"agent_picks"``, suitable for ``json.dumps`` and for
            round-tripping via :meth:`from_dict`.
        """
        return {
            "map_name": self.map_name,
            "team1_score": self.team1_score,
            "team2_score": self.team2_score,
            "winner": self.winner,
            "duration": self.duration,
            "agent_picks": self.agent_picks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapResult":
        """Reconstruct a ``MapResult`` from :meth:`to_dict` output.

        Args:
            data: A dict as produced by :meth:`to_dict`, i.e.
                containing a required ``"map_name"`` key and optional
                ``"team1_score"``, ``"team2_score"``, ``"winner"``,
                ``"duration"`` and ``"agent_picks"`` keys.

        Returns:
            The reconstructed ``MapResult``.

        Raises:
            KeyError: If ``data`` has no ``"map_name"`` key.
        """
        return cls(
            map_name=data["map_name"],
            team1_score=data.get("team1_score"),
            team2_score=data.get("team2_score"),
            winner=data.get("winner"),
            duration=data.get("duration"),
            agent_picks=data.get("agent_picks"),
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
    status: str = "upcoming"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this match to a JSON-compatible dict.

        ``team1``/``team2`` and each entry of ``maps`` are recursively
        serialized via their own ``to_dict``, and ``date`` is rendered
        as an ISO-8601 string (or ``None``), so the result is safe to
        pass to ``json.dumps`` — this is exactly what
        ``scraper.cache.set_cached_match`` does before storing a match.

        Returns:
            A dict with keys ``"match_id"``, ``"url"``,
            ``"event_name"``, ``"date"``, ``"team1"``, ``"team2"``,
            ``"team1_score"``, ``"team2_score"``, ``"best_of"``,
            ``"maps"`` and ``"status"``, suitable for round-tripping
            via :meth:`from_dict`.
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
                :meth:`MapResult.to_dict` dicts) and ``"status"`` keys.

        Returns:
            The reconstructed ``Match``, with nested ``team1``/
            ``team2``/``maps`` reconstructed via their own
            ``from_dict``.

        Raises:
            KeyError: If ``data`` is missing ``"match_id"``, ``"url"``,
                ``"event_name"``, ``"team1"`` or ``"team2"``.
            ValueError: If ``"date"`` is present and not a valid
                ISO-8601 string (propagated from
                ``datetime.fromisoformat``).
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
            status=data.get("status", "upcoming"),
        )
