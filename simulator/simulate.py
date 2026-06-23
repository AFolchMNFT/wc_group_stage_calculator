from __future__ import annotations
"""Monte Carlo simulation of the full WC 2026 tournament (group stage → Final)."""
import copy
import random
from collections import defaultdict

import numpy as np

from simulator.blend import blend_lambdas as _blend_lambdas
from simulator.elo_model import compute_match_lambdas as _elo_lambdas
from simulator.knockout import simulate_tournament as _simulate_tournament
from simulator.match_model import DEFAULT_LAMBDA_TOTAL, solve_lambdas as _solve_lambdas
from simulator.models import (
    GroupStanding,
    MatchFixture,
    MatchResult,
    SimResult,
    Team,
)
from simulator.r32_third_place import SLOT_MATCH
from simulator.r32_third_place import allocation as _annexe_c_allocation

# Groups whose winners face 3rd-place teams (Annexe C slots)
_ANNEXE_C_WINNER_GROUPS = set(SLOT_MATCH.keys())

# Valid model mode values
_MODEL_MODES = frozenset({"odds", "elo", "blend"})


def run(
    team_registry: dict,
    group_standings: dict,
    completed: list[MatchResult],
    fixtures: list[MatchFixture],
    n_simulations: int = 10_000,
    poisson_params: dict | None = None,
    seed: int | None = None,
    manual_results: dict | None = None,
    model_mode: str = "blend",
    elo_weight: float = 0.4,
    elo_ratings: dict | None = None,
    simulate_knockout: bool = True,
) -> SimResult:
    """Run n_simulations of the full tournament and return aggregated SimResult.

    Goal scoring is modelled as independent Poisson processes with per-fixture
    (λ_home, λ_away) derived from the betting markets, the Elo model, or a blend
    of both depending on model_mode.

    model_mode:        'odds'  — use betting odds only (original behaviour)
                       'elo'   — use Elo/factor model only
                       'blend' — weighted geometric mean of both (default)
    elo_weight:        Elo model weight in blend mode (0=pure odds, 1=pure Elo).
    elo_ratings:       {team_id: elo} pre-fetched ratings; None skips Elo model.
    simulate_knockout: If True (default), simulate R32→Final after group stage.

    manual_results: optional dict of match_id -> (home_goals, away_goals) for fixtures
    whose score is already known.  Applied deterministically; remaining fixtures are
    Monte Carlo simulated.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Optional override for the fallback total-goals rate (from settings.json)
    default_lt = (poisson_params or {}).get("default_lambda_total", DEFAULT_LAMBDA_TOTAL)

    # Build groups dict: group_letter -> [team_id, ...]
    groups: dict[str, list[str]] = {}
    for group_letter, standings in group_standings.items():
        groups[group_letter] = [s.team_id for s in standings]

    # Build initial standing state from ESPN standings (already reflects completed matches)
    initial_state = _build_initial_state(group_standings, completed, groups)

    # Apply any manually-fixed future results to the initial state
    manual_results = manual_results or {}
    fixture_map = {f.match_id: f for f in fixtures}
    for match_id, (hg, ag) in manual_results.items():
        fix = fixture_map.get(match_id)
        if fix:
            _update_state(initial_state, fix.home_team_id, fix.away_team_id, hg, ag)

    # Determine which fixtures are truly remaining (exclude completed + manual)
    excluded_ids = {m.match_id for m in completed} | set(manual_results.keys())
    remaining = [f for f in fixtures if f.match_id not in excluded_ids]

    n_fix = len(remaining)

    # avg_goals accumulates (avg_home, avg_away) per match_id for the results tab.
    # Manual/fixed scores go in as exact values; simulated ones as means over all runs.
    avg_goals: dict[str, tuple[float, float]] = {
        mid: (float(hg), float(ag)) for mid, (hg, ag) in manual_results.items()
    }

    if n_fix == 0:
        rng_state = random.Random(seed)
        one_run = _rank_all_groups(groups, copy.deepcopy(initial_state), rng_state)
        ko_elo = elo_ratings or {}
        knockout_runs: list[dict] = []
        if simulate_knockout and ko_elo:
            for _ in range(n_simulations):
                knockout_runs.append(
                    _simulate_tournament(
                        group_ranks=one_run["group_ranks"],
                        best_8_groups=one_run["best_8_groups"],
                        annexe_c=one_run["annexe_c"],
                        elo_ratings=ko_elo,
                        team_registry=team_registry,
                        rng=rng_state,
                    )
                )
        result = _aggregate([one_run] * n_simulations, team_registry, groups, knockout_runs)
        result.fixture_avg_goals = avg_goals
        return result

    # ---- Resolve per-fixture Poisson goal rates --------------------------------
    # Blend odds-based and Elo-based λ values according to model_mode.
    _use_elo = model_mode in ("elo", "blend") and elo_ratings is not None
    _use_odds = model_mode in ("odds", "blend")

    lambda_homes = np.empty(n_fix)
    lambda_aways = np.empty(n_fix)
    for j, fix in enumerate(remaining):
        # Odds-based λ
        if _use_odds and fix.lambda_home is not None and fix.lambda_away is not None:
            lh_odds, la_odds = fix.lambda_home, fix.lambda_away
        elif _use_odds:
            lh_odds, la_odds = _solve_lambdas(fix.prob_home, default_lt)
        else:
            lh_odds, la_odds = None, None

        # Elo-based λ
        if _use_elo:
            lh_elo, la_elo = _elo_lambdas(
                home_team_id=fix.home_team_id,
                away_team_id=fix.away_team_id,
                elo_ratings=elo_ratings,
                team_registry=team_registry,
            )
        else:
            # Fall back to solve_lambdas when Elo not available
            lh_elo, la_elo = _solve_lambdas(fix.prob_home, default_lt)

        lh, la = _blend_lambdas(lh_odds, la_odds, lh_elo, la_elo, elo_weight)
        lambda_homes[j] = lh
        lambda_aways[j] = la

    # ---- Pre-generate all goals: shape (n_simulations, n_fix) -----------------
    # Each fixture column j is drawn from Pois(λ_home[j]) / Pois(λ_away[j]).
    # Using a loop over fixtures keeps the per-fixture λ vectorised over sims.
    home_goals_all = np.zeros((n_simulations, n_fix), dtype=np.int32)
    away_goals_all = np.zeros((n_simulations, n_fix), dtype=np.int32)
    for j in range(n_fix):
        home_goals_all[:, j] = np.random.poisson(lambda_homes[j], n_simulations)
        away_goals_all[:, j] = np.random.poisson(lambda_aways[j], n_simulations)

    # Record mean simulated score per fixture
    for j, fix in enumerate(remaining):
        avg_goals[fix.match_id] = (
            float(home_goals_all[:, j].mean()),
            float(away_goals_all[:, j].mean()),
        )

    # ---- Main simulation loop -------------------------------------------------
    all_runs = []
    knockout_runs: list[dict] = []  # per-simulation knockout results
    rng = random.Random(seed)

    # Use fallback elo_ratings for knockout even if Elo was not used in group stage
    ko_elo = elo_ratings or {}

    for i in range(n_simulations):
        state = copy.deepcopy(initial_state)
        for j, fix in enumerate(remaining):
            _update_state(
                state,
                fix.home_team_id,
                fix.away_team_id,
                int(home_goals_all[i, j]),
                int(away_goals_all[i, j]),
            )
        run_result = _rank_all_groups(groups, state, rng)
        all_runs.append(run_result)

        if simulate_knockout and ko_elo:
            ko_result = _simulate_tournament(
                group_ranks=run_result["group_ranks"],
                best_8_groups=run_result["best_8_groups"],
                annexe_c=run_result["annexe_c"],
                elo_ratings=ko_elo,
                team_registry=team_registry,
                rng=rng,
            )
            knockout_runs.append(ko_result)

    result = _aggregate(all_runs, team_registry, groups, knockout_runs)
    result.fixture_avg_goals = avg_goals
    return result


def _build_initial_state(
    group_standings: dict,
    completed: list[MatchResult],
    groups: dict,
) -> dict:
    """Build per-team state dict from ESPN standings + completed match list for H2H."""
    state: dict[str, dict] = {}

    for group_letter, standings in group_standings.items():
        for s in standings:
            state[s.team_id] = {
                "pts": s.points,
                "gf": s.gf,
                "ga": s.ga,
                "h2h": {},
            }

    # Populate H2H from completed matches
    for m in completed:
        if m.home_team_id not in state or m.away_team_id not in state:
            continue
        if m.home_goals > m.away_goals:
            hpts, apts = 3, 0
        elif m.home_goals == m.away_goals:
            hpts = apts = 1
        else:
            hpts, apts = 0, 3

        _add_h2h(state, m.home_team_id, m.away_team_id, hpts, m.home_goals, m.away_goals)
        _add_h2h(state, m.away_team_id, m.home_team_id, apts, m.away_goals, m.home_goals)

    return state


def _add_h2h(state, tid, opp_id, pts, gf, ga):
    if opp_id not in state[tid]["h2h"]:
        state[tid]["h2h"][opp_id] = {"pts": 0, "gf": 0, "ga": 0}
    state[tid]["h2h"][opp_id]["pts"] += pts
    state[tid]["h2h"][opp_id]["gf"] += gf
    state[tid]["h2h"][opp_id]["ga"] += ga


def _update_state(state, home_id, away_id, hg, ag):
    if hg > ag:
        hpts, apts = 3, 0
    elif hg == ag:
        hpts = apts = 1
    else:
        hpts, apts = 0, 3

    state[home_id]["pts"] += hpts
    state[home_id]["gf"] += hg
    state[home_id]["ga"] += ag

    state[away_id]["pts"] += apts
    state[away_id]["gf"] += ag
    state[away_id]["ga"] += hg

    _add_h2h(state, home_id, away_id, hpts, hg, ag)
    _add_h2h(state, away_id, home_id, apts, ag, hg)


def _rank_all_groups(groups, state, rng) -> dict:
    """Return dict with group rankings and 3rd-place info for one simulation run."""
    group_ranks: dict[str, list[str]] = {}
    thirds: list[dict] = []

    for group_letter, team_ids in groups.items():
        ranked = _rank_group(team_ids, state, rng)
        group_ranks[group_letter] = ranked
        third_id = ranked[2]
        thirds.append(
            {
                "group": group_letter,
                "team_id": third_id,
                "pts": state[third_id]["pts"],
                "gd": state[third_id]["gf"] - state[third_id]["ga"],
                "gf": state[third_id]["gf"],
            }
        )

    best_8_groups = _select_best_8_thirds(thirds, rng)

    try:
        annexe_c = _annexe_c_allocation(best_8_groups)
    except Exception:
        annexe_c = {}

    return {
        "group_ranks": group_ranks,
        "best_8_groups": best_8_groups,
        "annexe_c": annexe_c,
    }


def _rank_group(team_ids: list, state: dict, rng: random.Random) -> list:
    """Sort team_ids by FIFA 2026 group-stage tiebreaker rules.

    Order: points → H2H (pts/GD/GF among tied) → overall GD → overall GF → lots.
    H2H is the first tiebreaker after points (before overall GD), per the updated rules.
    """
    by_pts = sorted(team_ids, key=lambda tid: -state[tid]["pts"])

    result: list[str] = []
    i = 0
    while i < len(by_pts):
        j = i + 1
        while j < len(by_pts) and state[by_pts[j]]["pts"] == state[by_pts[i]]["pts"]:
            j += 1
        tied = by_pts[i:j]
        if len(tied) == 1:
            result.extend(tied)
        else:
            result.extend(_resolve_h2h(tied, state, rng))
        i = j

    return result


def _resolve_h2h(tied: list, state: dict, rng: random.Random) -> list:
    """Break ties: H2H record among the tied teams → overall GD/GF → lots."""

    def h2h_key(tid):
        pts = gf = ga = 0
        for opp in tied:
            if opp == tid:
                continue
            h = state[tid]["h2h"].get(opp, {})
            pts += h.get("pts", 0)
            gf += h.get("gf", 0)
            ga += h.get("ga", 0)
        return (-pts, -(gf - ga), -gf)

    def overall_gd_key(tid):
        s = state[tid]
        return (-(s["gf"] - s["ga"]), -s["gf"])

    sorted_h2h = sorted(tied, key=h2h_key)

    result: list[str] = []
    i = 0
    while i < len(sorted_h2h):
        j = i + 1
        while j < len(sorted_h2h) and h2h_key(sorted_h2h[j]) == h2h_key(sorted_h2h[i]):
            j += 1
        sub = sorted_h2h[i:j]
        if len(sub) == 1:
            result.extend(sub)
        else:
            # Still tied after H2H — apply overall GD/GF before lots
            sorted_gd = sorted(sub, key=overall_gd_key)
            k = 0
            while k < len(sorted_gd):
                m = k + 1
                while m < len(sorted_gd) and overall_gd_key(sorted_gd[m]) == overall_gd_key(sorted_gd[k]):
                    m += 1
                final = sorted_gd[k:m]
                if len(final) == 1:
                    result.extend(final)
                else:
                    rng.shuffle(final)  # drawing of lots
                    result.extend(final)
                k = m
        i = j

    return result


def _select_best_8_thirds(thirds: list[dict], rng: random.Random) -> frozenset:
    """Return frozenset of the 8 group letters whose 3rd-place team qualifies."""
    sorted_thirds = sorted(
        thirds,
        key=lambda t: (-t["pts"], -t["gd"], -t["gf"], rng.random()),
    )
    return frozenset(t["group"] for t in sorted_thirds[:8])


def _aggregate(
    all_runs: list,
    team_registry: dict,
    groups: dict,
    knockout_runs: list | None = None,
) -> SimResult:
    n = len(all_runs)

    group_finish_counts: dict[str, dict[int, int]] = {
        tid: {1: 0, 2: 0, 3: 0, 4: 0} for tid in team_registry
    }
    r32_counts: dict[str, int] = {tid: 0 for tid in team_registry}
    third_qualified_counts: dict[str, int] = {g: 0 for g in groups}
    # slot_key e.g. "1A" -> {opponent_group: count}
    annexe_c_opponent_counts: dict[str, dict[str, int]] = {
        slot: defaultdict(int) for slot in SLOT_MATCH
    }

    for run in all_runs:
        ranks = run["group_ranks"]
        best_8 = run["best_8_groups"]
        annexe_c = run["annexe_c"]

        for group_letter, ranked in ranks.items():
            for pos, tid in enumerate(ranked, 1):
                if tid in group_finish_counts:
                    group_finish_counts[tid][pos] += 1
                # R32 qualification
                if pos <= 2:
                    r32_counts[tid] = r32_counts.get(tid, 0) + 1
                elif pos == 3 and group_letter in best_8:
                    r32_counts[tid] = r32_counts.get(tid, 0) + 1

        for g in best_8:
            third_qualified_counts[g] += 1

        for slot, opp_group in annexe_c.items():
            annexe_c_opponent_counts[slot][opp_group] += 1

    # ---- Aggregate knockout round advancement counts -------------------------
    _ko_round_fields = {
        "round_of_32":   "r32_ko",   # distinct from group-stage r32_counts
        "round_of_16":   "r16",
        "quarter_finals": "qf",
        "semi_finals":   "sf",
        "final":         "final",
    }
    r16_counts: dict[str, int] = {tid: 0 for tid in team_registry}
    qf_counts:  dict[str, int] = {tid: 0 for tid in team_registry}
    sf_counts:  dict[str, int] = {tid: 0 for tid in team_registry}
    final_counts: dict[str, int] = {tid: 0 for tid in team_registry}
    champion_counts: dict[str, int] = {tid: 0 for tid in team_registry}
    ko_match_stats: dict = {}

    if knockout_runs:
        for ko_run in knockout_runs:
            # Participants in each round = set of winners from the previous round
            # plus both participants of each match. We track who PLAYED in the round.
            for rnd, match_data in ko_run.items():
                for mid, mdata in match_data.items():
                    winner_tid = mdata["winner"]
                    home_id = mdata["home"]
                    away_id = mdata["away"]
                    home_g = mdata["home_g"]
                    away_g = mdata["away_g"]

                    if winner_tid not in team_registry:
                        continue
                    if rnd == "round_of_16":
                        r16_counts[winner_tid] = r16_counts.get(winner_tid, 0) + 1
                    elif rnd == "quarter_finals":
                        qf_counts[winner_tid] = qf_counts.get(winner_tid, 0) + 1
                    elif rnd == "semi_finals":
                        sf_counts[winner_tid] = sf_counts.get(winner_tid, 0) + 1
                    elif rnd == "final":
                        final_counts[winner_tid] = final_counts.get(winner_tid, 0) + 1
                        champion_counts[winner_tid] = champion_counts.get(winner_tid, 0) + 1

                    # Accumulate per-match stats for MC results display
                    ms = ko_match_stats.setdefault(mid, {
                        "played": 0,
                        "home_slot_wins": 0,
                        "home_goals_sum": 0.0,
                        "away_goals_sum": 0.0,
                        "home_teams": defaultdict(int),
                        "away_teams": defaultdict(int),
                        "round": rnd,
                    })
                    ms["played"] += 1
                    ms["home_goals_sum"] += home_g
                    ms["away_goals_sum"] += away_g
                    if winner_tid == home_id:
                        ms["home_slot_wins"] += 1
                    ms["home_teams"][home_id] += 1
                    ms["away_teams"][away_id] += 1

    return SimResult(
        n_simulations=n,
        teams=team_registry,
        groups=groups,
        group_finish_counts=group_finish_counts,
        r32_counts=r32_counts,
        third_qualified_counts=third_qualified_counts,
        annexe_c_opponent_counts=dict(annexe_c_opponent_counts),
        r16_counts=r16_counts,
        qf_counts=qf_counts,
        sf_counts=sf_counts,
        final_counts=final_counts,
        champion_counts=champion_counts,
        ko_match_stats=ko_match_stats,
    )
