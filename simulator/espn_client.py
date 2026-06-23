from __future__ import annotations
"""Fetch current WC 2026 standings and fixtures from the ESPN public API."""
import concurrent.futures
import datetime
import json
import os

import requests

from simulator.models import GroupStanding, MatchFixture, MatchResult, Team
from simulator.paths import CONFIG_DIR

_STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings"
_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
_TIMEOUT = 10
_CONFIG_DIR = CONFIG_DIR


def fetch_all(
    start_date: str = "20260611",
    end_date: str = "20260627",
    force_fallback: bool = False,
) -> tuple[dict, dict, list[MatchResult], list[MatchFixture]]:
    """Return (team_registry, group_standings, completed_matches, remaining_fixtures).

    team_registry: team_id -> Team
    group_standings: group_letter -> list[GroupStanding]  (pre-populated from ESPN)
    """
    if not force_fallback:
        try:
            team_registry, group_standings = _fetch_standings()
            completed, fixtures = _fetch_scoreboard(start_date, end_date, team_registry)
            return team_registry, group_standings, completed, fixtures
        except Exception as exc:
            print(f"[yellow]ESPN API failed ({exc}); using config/standings.json fallback.[/yellow]")

    return _load_fallback()


def _fetch_standings() -> tuple[dict, dict]:
    resp = requests.get(_STANDINGS_URL, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return _parse_standings(data)


def _parse_standings(data: dict) -> tuple[dict, dict]:
    team_registry: dict[str, Team] = {}
    group_standings: dict[str, list[GroupStanding]] = {}

    for child in data.get("children", []):
        group_name = child.get("name", "")
        if "Group " not in group_name:
            continue
        group_letter = group_name.split("Group ")[-1].strip()[0].upper()

        entries = child.get("standings", {}).get("entries", [])
        standings: list[GroupStanding] = []

        for entry in entries:
            team_data = entry.get("team", {})
            team_id = str(team_data.get("id", ""))
            if not team_id:
                continue

            team_name = team_data.get("displayName", team_data.get("name", "Unknown"))
            team_abbr = team_data.get("abbreviation", "???")

            if team_id not in team_registry:
                team_registry[team_id] = Team(
                    id=team_id,
                    name=team_name,
                    abbreviation=team_abbr,
                    group=group_letter,
                )

            stats = {s["name"]: float(s.get("value", 0)) for s in entry.get("stats", [])}
            standings.append(
                GroupStanding(
                    team_id=team_id,
                    group=group_letter,
                    played=int(stats.get("gamesPlayed", 0)),
                    wins=int(stats.get("wins", 0)),
                    draws=int(stats.get("ties", 0)),
                    losses=int(stats.get("losses", 0)),
                    gf=int(stats.get("pointsFor", 0)),
                    ga=int(stats.get("pointsAgainst", 0)),
                    points=int(stats.get("points", 0)),
                )
            )

        group_standings[group_letter] = standings

    return team_registry, group_standings


def _scoreboard_for_date(date_str: str) -> dict:
    resp = requests.get(_SCOREBOARD_URL, params={"dates": date_str}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fetch_scoreboard(
    start_date: str,
    end_date: str,
    team_registry: dict,
) -> tuple[list[MatchResult], list[MatchFixture]]:
    start = datetime.datetime.strptime(start_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y%m%d"))
        cur += datetime.timedelta(days=1)

    completed: list[MatchResult] = []
    fixtures: list[MatchFixture] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_scoreboard_for_date, d): d for d in dates}
        for fut in concurrent.futures.as_completed(futures):
            try:
                data = fut.result()
            except Exception:
                continue
            for event in data.get("events", []):
                parsed = _parse_event(event, team_registry)
                if isinstance(parsed, MatchResult):
                    completed.append(parsed)
                elif isinstance(parsed, MatchFixture):
                    fixtures.append(parsed)

    fixtures.sort(key=lambda f: f.utc_date)
    return completed, fixtures


def _parse_event(
    event: dict, team_registry: dict
) -> MatchResult | MatchFixture | None:
    comps = event.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]

    group_letter = _extract_group(event, comp)
    if not group_letter:
        return None

    competitors = comp.get("competitors", [])
    home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home_c or not away_c:
        return None

    home_id = str(home_c.get("team", {}).get("id", ""))
    away_id = str(away_c.get("team", {}).get("id", ""))

    for competitor, _ in [(home_c, "home"), (away_c, "away")]:
        tid = str(competitor.get("team", {}).get("id", ""))
        if tid and tid not in team_registry:
            td = competitor.get("team", {})
            team_registry[tid] = Team(
                id=tid,
                name=td.get("displayName", td.get("name", "Unknown")),
                abbreviation=td.get("abbreviation", "???"),
                group=group_letter,
            )

    state = comp.get("status", {}).get("type", {}).get("state", "pre")
    match_id = str(comp.get("id", event.get("id", "")))
    utc_date = event.get("date", "")

    if state == "post":
        try:
            home_goals = int(home_c.get("score", 0) or 0)
            away_goals = int(away_c.get("score", 0) or 0)
        except (ValueError, TypeError):
            home_goals = away_goals = 0
        return MatchResult(
            match_id=match_id,
            home_team_id=home_id,
            away_team_id=away_id,
            home_goals=home_goals,
            away_goals=away_goals,
            group=group_letter,
            is_completed=True,
        )
    else:
        return MatchFixture(
            match_id=match_id,
            home_team_id=home_id,
            away_team_id=away_id,
            group=group_letter,
            utc_date=utc_date,
        )


def _extract_group(event: dict, comp: dict) -> str | None:
    sources = [
        comp.get("notes", []),
        event.get("notes", []),
        [{"headline": comp.get("altGameNote", "")}],
        [{"headline": event.get("name", "")}],
    ]
    for notes in sources:
        for note in notes:
            headline = note.get("headline", "") or note.get("type", {}).get("text", "")
            if "Group " in headline:
                after = headline.split("Group ")[-1].strip()
                if after and after[0].isalpha():
                    return after[0].upper()
    return None


def _load_fallback() -> tuple[dict, dict, list[MatchResult], list[MatchFixture]]:
    fallback_path = _CONFIG_DIR / "standings.json"
    if not fallback_path.exists():
        raise FileNotFoundError(
            f"ESPN API unavailable and {fallback_path} not found. "
            "Create config/standings.json — see config/standings_template.json for format."
        )
    with open(fallback_path) as f:
        data = json.load(f)

    team_registry: dict[str, Team] = {}
    group_standings: dict[str, list[GroupStanding]] = {}
    completed: list[MatchResult] = []
    fixtures: list[MatchFixture] = []

    for group_letter, entries in data.get("standings", {}).items():
        standings = []
        for e in entries:
            tid = str(e["team_id"])
            team_registry[tid] = Team(
                id=tid,
                name=e["team_name"],
                abbreviation=e.get("abbreviation", e["team_name"][:3].upper()),
                group=group_letter,
            )
            standings.append(
                GroupStanding(
                    team_id=tid,
                    group=group_letter,
                    played=e.get("played", 0),
                    wins=e.get("wins", 0),
                    draws=e.get("draws", 0),
                    losses=e.get("losses", 0),
                    gf=e.get("gf", 0),
                    ga=e.get("ga", 0),
                    points=e.get("points", 0),
                )
            )
        group_standings[group_letter] = standings

    for m in data.get("fixtures", {}).get("completed", []):
        completed.append(
            MatchResult(
                match_id=str(m["match_id"]),
                home_team_id=str(m["home_team_id"]),
                away_team_id=str(m["away_team_id"]),
                home_goals=m["home_goals"],
                away_goals=m["away_goals"],
                group=m["group"],
            )
        )
    for m in data.get("fixtures", {}).get("remaining", []):
        fixtures.append(
            MatchFixture(
                match_id=str(m["match_id"]),
                home_team_id=str(m["home_team_id"]),
                away_team_id=str(m["away_team_id"]),
                group=m["group"],
                utc_date=m.get("utc_date", ""),
            )
        )

    return team_registry, group_standings, completed, fixtures
