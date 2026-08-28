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
    """A competing team. ``team_id`` is vlr.gg's numeric team id."""

    name: str
    team_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "team_id": self.team_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        return cls(name=data["name"], team_id=data.get("team_id"))


@dataclass
class PlayerStats:
    """Minimal per-player stats.

    Reserved for future stats scraping (not populated by the current
    parsers). Kept in the model so the schema exists before the
    stats tables on vlr.gg get parsed.
    """

    player_name: str
    team_name: str
    rating: Optional[float] = None
    acs: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "team_name": self.team_name,
            "rating": self.rating,
            "acs": self.acs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerStats":
        return cls(
            player_name=data["player_name"],
            team_name=data["team_name"],
            rating=data.get("rating"),
            acs=data.get("acs"),
        )


@dataclass
class MapResult:
    """Result of a single played map."""

    map_name: str
    team1_score: Optional[int]
    team2_score: Optional[int]
    winner: Optional[str]  # name of the winning team (None if unknown)
    duration: Optional[str] = None  # e.g. "59:20" as displayed on vlr.gg
    agent_picks: Optional[dict[str, Any]] = None  # reserved; not populated yet

    def to_dict(self) -> dict[str, Any]:
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

    ``date`` is a naive UTC datetime parsed from vlr.gg's
    ``data-utc-ts`` attribute. ``status`` is one of
    "completed", "live", "upcoming".
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
