from __future__ import annotations
"""Full-tournament knockout bracket simulation (R32 → R16 → QF → SF → Final).

Reads the bracket topology from config/world_cup_2026_bracket.json and
config/bracket.json, then simulates each knockout round using the Elo model.

Draws in knockout matches are resolved by re-drawing until a winner emerges
(the GS "burning draws" approach). This implicitly models extra time and
penalty shootouts while slightly favouring the stronger team, consistent with
empirical evidence that the better team wins more often from the spot.

Elo ratings are updated after each simulated match so that later-round
λ values reflect mid-tournament form.
"""

import json
import random
from pathlib import Path

import numpy as np

from simulator.elo_model import compute_match_lambdas, update_elo
from simulator.paths import CONFIG_DIR

# ---------------------------------------------------------------------------
# Bracket topology
# ---------------------------------------------------------------------------

def _load_bracket_topology() -> dict:
    path = CONFIG_DIR / "world_cup_2026_bracket.json"
    with open(path) as f:
        return json.load(f)


def _load_fixed_bracket() -> dict:
    path = CONFIG_DIR / "bracket.json"
    with open(path) as f:
        return json.load(f)


# Build match-tree: match_id -> (feeder_home_match, feeder_away_match) or (slot1, slot2)
def _build_match_tree() -> tuple[dict, dict, list]:
    """Return (r32_slots, feed_map, ko_round_order).

    r32_slots:    {match_id: (slot1, slot2)} where slotX is '1A'/'2B'/None (Annexe C)
    feed_map:     {match_id: (home_feeder_match, away_feeder_match)} for R16/QF/SF/Final
    ko_round_order: list of round names in order ['round_of_32', 'round_of_16', ...]
    """
    topo = _load_bracket_topology()
    rounds = topo["rounds"]

    def _pm(s: str) -> str:
        return f"M{s.split()[-1]}"

    def _parse_slot(s: str) -> str | None:
        p = s.split()
        if p[0] == "Winner" and p[1] == "Group":
            return f"1{p[2]}"
        if p[0] == "Runner-up" and p[1] == "Group":
            return f"2{p[2]}"
        return None  # Annexe C (Best 3rd place …)

    r32_slots: dict[str, tuple[str | None, str | None]] = {}
    for m in rounds["round_of_32"]:
        mid = f"M{m['match']}"
        r32_slots[mid] = (_parse_slot(m["home"]), _parse_slot(m["away"]))

    feed_map: dict[str, tuple[str, str]] = {}
    for rnd in ("round_of_16", "quarter_finals", "semi_finals", "final"):
        for m in rounds[rnd]:
            mid = f"M{m['match']}"
            if "Winner Match" in m.get("home", "") and "Winner Match" in m.get("away", ""):
                feed_map[mid] = (_pm(m["home"]), _pm(m["away"]))

    ko_round_order = [
        "round_of_32",
        "round_of_16",
        "quarter_finals",
        "semi_finals",
        "final",
    ]
    return r32_slots, feed_map, ko_round_order


_R32_SLOTS, _FEED_MAP, _KO_ROUNDS = _build_match_tree()

# ---------------------------------------------------------------------------
# Knockout match venue altitudes (from venues.json knockout_match_venues)
# ---------------------------------------------------------------------------

def _load_ko_venue_altitudes() -> dict[str, int]:
    """Return {match_id: altitude_m} for knockout matches with known venues."""
    try:
        with open(CONFIG_DIR / "venues.json") as f:
            vd = json.load(f)
        venue_defs = vd.get("venues", {})
        ko_venues = vd.get("knockout_match_venues", {})
        out: dict[str, int] = {}
        for mid, vkey in ko_venues.items():
            if mid.startswith("_"):
                continue
            alt = venue_defs.get(vkey, {}).get("altitude_m", 0)
            out[mid] = alt
        return out
    except Exception:
        return {}


_KO_VENUE_ALTITUDES: dict[str, int] = _load_ko_venue_altitudes()

# Round match IDs grouped
def _round_match_ids() -> dict[str, list[str]]:
    topo = _load_bracket_topology()
    out: dict[str, list[str]] = {}
    for rnd in ("round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"):
        out[rnd] = [f"M{m['match']}" for m in topo["rounds"][rnd]]
    return out


_ROUND_MATCH_IDS = _round_match_ids()

# Map round name → our SimResult field key
_ROUND_TO_FIELD = {
    "round_of_32": "r32",
    "round_of_16": "r16",
    "quarter_finals": "qf",
    "semi_finals": "sf",
    "final": "final",
}


# ---------------------------------------------------------------------------
# R32 bracket resolution
# ---------------------------------------------------------------------------

def resolve_r32_teams(
    group_ranks: dict[str, list[str]],
    best_8_groups: frozenset,
    annexe_c: dict,
) -> dict[str, tuple[str, str]]:
    """Return {match_id: (home_team_id, away_team_id)} for all 16 R32 matches.

    Uses the group stage outcome (group_ranks, best_8_groups, annexe_c) to
    concretely assign teams to each bracket slot.

    Parameters
    ----------
    group_ranks:   {group_letter: [team_id at pos1, pos2, pos3, pos4]}
    best_8_groups: frozenset of group letters whose 3rd-place team qualified
    annexe_c:      {slot: opponent_group} from r32_third_place.allocation()
    """
    from simulator.r32_third_place import SLOT_MATCH

    # Resolve slot → team_id
    def _slot_team(slot: str | None, group_letter_override: str | None = None) -> str | None:
        if slot is None:
            return None
        pos = int(slot[0])  # 1 or 2
        grp = slot[1].upper()
        ranked = group_ranks.get(grp, [])
        if len(ranked) < pos:
            return None
        return ranked[pos - 1]

    # Annexe C: slot (e.g. "1A") → the actual 3rd-place team from opponent_group
    # annexe_c maps "1A" (= SLOT_MATCH slot) → opponent_group letter
    annexe_c_team: dict[str, str | None] = {}
    for slot, opp_group in annexe_c.items():
        ranked = group_ranks.get(opp_group, [])
        annexe_c_team[slot] = ranked[2] if len(ranked) >= 3 else None

    matchups: dict[str, tuple[str, str]] = {}
    fixed_bracket = _load_fixed_bracket()

    # --- Annexe C matches (groups whose winners face 3rd-place teams) ---
    for slot, (match_num, winner_group) in SLOT_MATCH.items():
        home_team = _slot_team(f"1{winner_group}")
        away_team = annexe_c_team.get(slot)
        if home_team and away_team:
            matchups[match_num] = (home_team, away_team)

    # --- Fixed bracket matches ---
    for m in fixed_bracket.get("r32_fixed", []):
        mid = m["match_id"]
        t1 = _slot_team(m["slot_1"])
        t2 = _slot_team(m["slot_2"])
        if t1 and t2:
            matchups[mid] = (t1, t2)

    return matchups


# ---------------------------------------------------------------------------
# Single-match simulation
# ---------------------------------------------------------------------------

def simulate_ko_match(
    home_id: str,
    away_id: str,
    elo_ratings: dict[str, int],
    team_registry: dict,
    rng: random.Random,
    venue_altitude_m: int = 0,
    max_redraw: int = 50,
) -> tuple[str, str, int, int]:
    """Simulate one knockout match, returning (winner_id, loser_id, home_g, away_g).

    Draws are resolved by re-drawing (GS "burning draws" approach).
    This makes the expected goal difference the deciding factor when teams are
    level, matching empirical penalty-shootout outcomes.
    """
    lh, la = compute_match_lambdas(
        home_team_id=home_id,
        away_team_id=away_id,
        elo_ratings=elo_ratings,
        team_registry=team_registry,
        venue_altitude_m=venue_altitude_m,
    )

    for _ in range(max_redraw):
        hg = int(np.random.poisson(lh))
        ag = int(np.random.poisson(la))
        if hg != ag:
            break
    # In the very unlikely event we never broke the draw, pick by higher λ
    if hg == ag:
        if lh >= la:
            hg += 1
        else:
            ag += 1

    winner = home_id if hg > ag else away_id
    loser = away_id if hg > ag else home_id
    return winner, loser, hg, ag


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def simulate_tournament(
    group_ranks: dict[str, list[str]],
    best_8_groups: frozenset,
    annexe_c: dict,
    elo_ratings: dict[str, int],
    team_registry: dict,
    rng: random.Random,
) -> dict[str, dict[str, dict]]:
    """Simulate R32 through Final for one Monte Carlo run.

    Returns {round_name: {match_id: {"winner": str, "home": str, "away": str,
                                     "home_g": int, "away_g": int}}}.
    Elo ratings are updated after each match within this run (not mutating
    the caller's dict — a local copy is made at the start).
    """
    local_elo = dict(elo_ratings)

    # Resolve R32 matchups from group stage results
    r32_matchups = resolve_r32_teams(group_ranks, best_8_groups, annexe_c)

    # match_id → winner_team_id (populated round by round)
    winners: dict[str, str] = {}
    results_by_round: dict[str, dict[str, dict]] = {}

    for rnd in _KO_ROUNDS:
        round_match_ids = _ROUND_MATCH_IDS[rnd]
        results_by_round[rnd] = {}

        for mid in round_match_ids:
            if rnd == "round_of_32":
                matchup = r32_matchups.get(mid)
                if matchup is None:
                    continue
                home_id, away_id = matchup
            else:
                feeder = _FEED_MAP.get(mid)
                if feeder is None:
                    continue
                home_feeder, away_feeder = feeder
                home_id = winners.get(home_feeder)
                away_id = winners.get(away_feeder)
                if home_id is None or away_id is None:
                    continue

            winner, loser, hg, ag = simulate_ko_match(
                home_id=home_id,
                away_id=away_id,
                elo_ratings=local_elo,
                team_registry=team_registry,
                rng=rng,
                venue_altitude_m=_KO_VENUE_ALTITUDES.get(mid, 0),
            )
            winners[mid] = winner
            results_by_round[rnd][mid] = {
                "winner": winner,
                "home": home_id,
                "away": away_id,
                "home_g": hg,
                "away_g": ag,
            }

            # Update Elo with this match result
            local_elo = update_elo(home_id, away_id, hg, ag, local_elo)

    return results_by_round


def simulate_bracket_once(
    group_ranks: dict[str, list[str]],
    best_8_groups: frozenset,
    annexe_c: dict,
    elo_ratings: dict[str, int],
    team_registry: dict,
    rng: random.Random,
) -> dict[str, dict]:
    """Simulate the full knockout bracket and return detailed per-match results.

    Unlike simulate_tournament (which returns only winners per round), this
    function returns goal scores alongside winner/loser info for every match.

    Returns
    -------
    {match_id: {"home": team_id, "away": team_id, "home_g": int,
                "away_g": int, "winner": team_id, "round": str}}
    """
    local_elo = dict(elo_ratings)
    r32_matchups = resolve_r32_teams(group_ranks, best_8_groups, annexe_c)

    winners: dict[str, str] = {}
    match_details: dict[str, dict] = {}

    for rnd in _KO_ROUNDS:
        round_match_ids = _ROUND_MATCH_IDS[rnd]
        for mid in round_match_ids:
            if rnd == "round_of_32":
                matchup = r32_matchups.get(mid)
                if matchup is None:
                    continue
                home_id, away_id = matchup
            else:
                feeder = _FEED_MAP.get(mid)
                if feeder is None:
                    continue
                home_feeder, away_feeder = feeder
                home_id = winners.get(home_feeder)
                away_id = winners.get(away_feeder)
                if home_id is None or away_id is None:
                    continue

            winner, loser, hg, ag = simulate_ko_match(
                home_id=home_id,
                away_id=away_id,
                elo_ratings=local_elo,
                team_registry=team_registry,
                rng=rng,
                venue_altitude_m=_KO_VENUE_ALTITUDES.get(mid, 0),
            )
            winners[mid] = winner
            match_details[mid] = {
                "home": home_id,
                "away": away_id,
                "home_g": hg,
                "away_g": ag,
                "winner": winner,
                "round": rnd,
            }
            local_elo = update_elo(home_id, away_id, hg, ag, local_elo)

    return match_details
