from __future__ import annotations
"""Persist fetched ESPN + odds data to/from a local JSON cache.

The cache stores the full state of team_registry, group_standings, completed matches,
and remaining fixtures (including any derived λ values from the odds API).
This allows the app to run completely offline after one successful fetch.
"""

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from simulator.models import MatchFixture, MatchResult, GroupStanding, Team


def save(
    team_registry: dict,
    group_standings: dict,
    completed: list,
    fixtures: list,
    path: Path,
    elo_ratings: dict | None = None,
) -> None:
    """Serialise all data to a JSON cache file."""
    data = {
        "cached_at": datetime.now().isoformat(timespec="seconds"),
        "teams": {tid: dataclasses.asdict(t) for tid, t in team_registry.items()},
        "group_standings": {
            gl: [dataclasses.asdict(s) for s in standings]
            for gl, standings in group_standings.items()
        },
        "completed": [dataclasses.asdict(m) for m in completed],
        "fixtures": [dataclasses.asdict(f) for f in fixtures],
    }
    if elo_ratings:
        data["elo_ratings"] = elo_ratings
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(data, fp, indent=2)


def load(path: Path) -> tuple[dict, dict, list, list, str, dict]:
    """Load cached data and return (team_registry, group_standings, completed, fixtures, cached_at, elo_ratings)."""
    with open(path) as fp:
        data = json.load(fp)

    team_registry = {tid: Team(**t) for tid, t in data["teams"].items()}

    group_standings = {
        gl: [GroupStanding(**s) for s in standings]
        for gl, standings in data["group_standings"].items()
    }

    completed = [MatchResult(**m) for m in data["completed"]]
    fixtures = [MatchFixture(**f) for f in data["fixtures"]]
    elo_ratings = data.get("elo_ratings", {})

    return team_registry, group_standings, completed, fixtures, data.get("cached_at", ""), elo_ratings


def timestamp(path: Path) -> str:
    """Return the cached_at timestamp string, or '' if no cache exists."""
    if not path.exists():
        return ""
    try:
        with open(path) as fp:
            return json.load(fp).get("cached_at", "")
    except Exception:
        return ""
