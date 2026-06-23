from __future__ import annotations
"""GS-inspired Elo-based Poisson goal model.

Predicts per-match (λ_home, λ_away) using:
  - Elo rating difference (main driver, from eloratings.net)
  - Scoring talent (global top-scorer count, capped at 4)
  - Team momentum (recent goals scored / conceded in competitive matches)
  - Mentality factors: defending champion slump, first WC boost, big-nation boost
  - Opponent difficulty: scoring vs European teams is harder
  - Geography: home advantage for host nations, altitude penalty in Mexico

Coefficients are calibrated to reproduce the marginal effects shown in
GS Exhibit 4 and to yield a mean λ_total of ~2.6 for an average WC group-stage
match (the historical average since 1978).

Elo update formula (used within simulation runs):
  K = 60 for WC matches (eloratings.net standard for major tournaments)
  Expected = 1 / (1 + 10^((Elo_opp - Elo_team) / 400))
  New Elo = Old Elo + K * (Score - Expected)
  Score: 1 = win, 0.5 = draw, 0 = loss
"""

import json
import math
from pathlib import Path

from simulator.paths import CONFIG_DIR

# ---------------------------------------------------------------------------
# Model coefficients (calibrated to GS Exhibit 4 marginal effects)
# ---------------------------------------------------------------------------
# All coefficients represent the additive effect on log(λ_i), evaluated
# at the stated marginal values.
#
# GS Exhibit 4 baseline (marginal effects at average values):
#   Home advantage:           +0.30 expected goals
#   Elo diff +200pt:          +0.28 expected goals
#   Scoring talent +2 top scorers: +0.15 expected goals
#   Opponent goals conceded +1/game: +0.07 expected goals
#   Recent goals scored +1/game:  +0.06 expected goals
#   Winner's slump (defending champion): −0.12 expected goals
#   High-altitude malus:      −0.18 expected goals
#
# We convert these to log-scale coefficients assuming a base λ of 1.3
# per team (since 2 × 1.3 ≈ 2.6 total goals per game).
# β = Δλ / λ_base ≈ Δλ / 1.3
#
# Intercept anchored so that at Elo parity, average talent, average momentum,
# and no special factors, λ = 1.3 (→ λ_total ≈ 2.6).

_BASE_LAMBDA: float = 1.3   # per-team base (sum = 2.6 WC average)
_LOG_BASE: float = math.log(_BASE_LAMBDA)

# Coefficient for Elo difference (per 400-point difference)
# Effect: +200pt → +0.28 goals → β_elo × (200/400) = 0.28/1.3 → β_elo ≈ 0.43 × 2 = 0.86
_BETA_ELO: float = 0.86         # multiplied by Δelo / 400

# Scoring talent: each additional top-scorer adds ~0.075 goals
# (talent+2 → +0.15 goals → Δlog(λ) = 0.15/1.3 ≈ 0.115 per 2 scorers)
_BETA_TALENT: float = 0.058     # multiplied by talent (0–4)

# Momentum — recent goals scored (per game in last 10 games)
# Effect: +1 → +0.06 expected → Δlog(λ) ≈ 0.046
_BETA_MOMENTUM_SCORED: float = 0.046

# Opponent momentum — opponent recent goals conceded (per game in last 5 games)
# Effect: +1 → +0.07 expected → Δlog(λ) ≈ 0.054
_BETA_OPP_CONCEDED: float = 0.054

# Defending champion slump (Argentina 2026)
# Effect: −0.12 expected goals → Δlog(λ) ≈ −0.092
_BETA_CHAMPION_SLUMP: float = -0.092

# First World Cup appearance boost
# GS notes debut teams tend to outperform — estimated +0.05 expected goals
_BETA_FIRST_WC: float = 0.038

# Big-nation competition boost (ARG, GER, BRA, FRA, NLD, POR, SPA, ENG)
# Not explicitly quantified by GS; we assign a modest boost ~+0.08 goals
_BETA_BIG_NATION: float = 0.061

# Scoring vs European teams is harder (negative effect on team i when j is European)
# Estimated from GS Exhibit 4 "European opponent" implied effect ~−0.06 goals
_BETA_VS_EUROPEAN: float = -0.046

# Home advantage for host nations (US, Canada, Mexico)
# Effect: +0.30 expected goals → Δlog(λ) ≈ 0.23
_BETA_HOME: float = 0.23

# High-altitude penalty for sea-level teams
# Effect: −0.18 expected goals → Δlog(λ) ≈ −0.138
_BETA_ALTITUDE: float = -0.138
_ALTITUDE_THRESHOLD_M: int = 1500   # venue altitude above which penalty applies
_SEA_LEVEL_THRESHOLD_M: int = 500   # team's home altitude below which they suffer

# WC historical underperformance (England)
# GS Exhibit 8: England underperforms Elo due to historical tournament disappointment.
# Effect: ~−0.12 expected goals per game → Δlog(λ) ≈ −0.095
# Note: England also loses the big-nation boost (set is_big_nation=false in team_factors).
_BETA_WC_UNDERPERFORMER: float = -0.095

# Reference momentum values (average WC qualifier team)
_AVG_GOALS_SCORED_10: float = 1.6
_AVG_GOALS_CONCEDED_5: float = 1.1

# Elo K-factor for WC matches (eloratings.net major tournament standard)
_K_FACTOR: int = 60

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_team_factors() -> dict[str, dict]:
    path = CONFIG_DIR / "team_factors.json"
    with open(path) as f:
        data = json.load(f)
    return data["teams"]   # {abbreviation: {scoring_talent, ...}}


def _load_venues() -> dict:
    path = CONFIG_DIR / "venues.json"
    with open(path) as f:
        return json.load(f)


# Module-level cache
_TEAM_FACTORS: dict[str, dict] | None = None
_VENUES: dict | None = None


def _factors() -> dict[str, dict]:
    global _TEAM_FACTORS
    if _TEAM_FACTORS is None:
        _TEAM_FACTORS = _load_team_factors()
    return _TEAM_FACTORS


def _venue_data() -> dict:
    global _VENUES
    if _VENUES is None:
        _VENUES = _load_venues()
    return _VENUES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_abb_index(team_registry: dict) -> dict[str, str]:
    """Return {abbreviation.upper(): team_id} from the ESPN team_registry."""
    return {t.abbreviation.upper(): tid for tid, t in team_registry.items()}


def get_factors(team_id: str, team_registry: dict, abb_index: dict[str, str] | None = None) -> dict:
    """Return team factors for a given ESPN team_id, or a default if unknown."""
    tf = _factors()
    # Look up by abbreviation
    team = team_registry.get(team_id)
    if team:
        f = tf.get(team.abbreviation.upper())
        if f:
            return f
    # Fallback defaults — treat as average team
    return {
        "scoring_talent": 0,
        "recent_goals_scored_10": _AVG_GOALS_SCORED_10,
        "recent_goals_conceded_5": _AVG_GOALS_CONCEDED_5,
        "is_defending_champion": False,
        "is_first_world_cup": False,
        "is_european": False,
        "is_big_nation": False,
    }


def compute_lambdas(
    team_i_id: str,
    team_j_id: str,
    elo_ratings: dict[str, int],
    team_registry: dict,
    is_home_i: bool = False,
    venue_altitude_m: int = 0,
) -> tuple[float, float]:
    """Compute (λ_i, λ_j) — expected goals for team i vs team j.

    Parameters
    ----------
    team_i_id:       ESPN team_id of the attacking team
    team_j_id:       ESPN team_id of the defending team
    elo_ratings:     {team_id: elo_rating} — current ratings (updated mid-tournament)
    team_registry:   ESPN team registry {team_id: Team}
    is_home_i:       True if team i is playing in their home country
    venue_altitude_m: Altitude of the venue in metres

    Returns both directions so callers can get (λ_home, λ_away) in one call:
        λ_home, λ_away = compute_lambdas(home_id, away_id, ..., is_home_i=True)
    """
    fi = get_factors(team_i_id, team_registry)
    fj = get_factors(team_j_id, team_registry)

    elo_i = elo_ratings.get(team_i_id, 1700)
    elo_j = elo_ratings.get(team_j_id, 1700)

    # ---- Build log(λ_i) ------------------------------------------------
    log_lambda = _LOG_BASE

    # Elo advantage
    log_lambda += _BETA_ELO * (elo_i - elo_j) / 400.0

    # Scoring talent (global)
    log_lambda += _BETA_TALENT * fi["scoring_talent"]

    # Momentum: how much team i has been scoring lately
    log_lambda += _BETA_MOMENTUM_SCORED * (
        fi["recent_goals_scored_10"] - _AVG_GOALS_SCORED_10
    )

    # Opponent's defensive weakness
    log_lambda += _BETA_OPP_CONCEDED * (
        fj["recent_goals_conceded_5"] - _AVG_GOALS_CONCEDED_5
    )

    # Mentality: defending champion slump
    if fi["is_defending_champion"]:
        log_lambda += _BETA_CHAMPION_SLUMP

    # Mentality: first World Cup appearance
    if fi["is_first_world_cup"]:
        log_lambda += _BETA_FIRST_WC

    # Mentality: big-nation competition boost
    if fi["is_big_nation"]:
        log_lambda += _BETA_BIG_NATION

    # Opponent defence: European teams are harder to score against
    if fj["is_european"]:
        log_lambda += _BETA_VS_EUROPEAN

    # Geography: home advantage for host nations
    if is_home_i:
        log_lambda += _BETA_HOME

    # Geography: high-altitude penalty for sea-level teams
    team_i_obj = team_registry.get(team_i_id)
    if (
        venue_altitude_m >= _ALTITUDE_THRESHOLD_M
        and team_i_obj is not None
        and not _team_is_host(team_i_obj.abbreviation.upper())
    ):
        log_lambda += _BETA_ALTITUDE

    # WC historical underperformance (e.g. England)
    if fi.get("wc_underperformer", False):
        log_lambda += _BETA_WC_UNDERPERFORMER

    return math.exp(log_lambda), _compute_lambda_j(team_j_id, team_i_id, elo_ratings, team_registry, venue_altitude_m)


def compute_match_lambdas(
    home_team_id: str,
    away_team_id: str,
    elo_ratings: dict[str, int],
    team_registry: dict,
    venue_altitude_m: int = 0,
    high_altitude_match_ids: frozenset[str] | None = None,
    match_id: str | None = None,
) -> tuple[float, float]:
    """Convenience wrapper returning (λ_home, λ_away) for a single match.

    Handles home/away symmetry and altitude lookup.
    """
    home_team = team_registry.get(home_team_id)
    away_team = team_registry.get(away_team_id)

    # Determine if either team gets the home-nation advantage
    home_abb = home_team.abbreviation.upper() if home_team else ""
    away_abb = away_team.abbreviation.upper() if away_team else ""

    vd = _venue_data()
    host_nations: set[str] = set(vd.get("host_nations", []))

    is_home_for_home_team = home_abb in host_nations
    is_home_for_away_team = away_abb in host_nations

    lh = _compute_lambda_j(
        team_i_id=home_team_id,
        team_j_id=away_team_id,
        elo_ratings=elo_ratings,
        team_registry=team_registry,
        venue_altitude_m=venue_altitude_m,
        is_home_i=is_home_for_home_team,
    )
    la = _compute_lambda_j(
        team_i_id=away_team_id,
        team_j_id=home_team_id,
        elo_ratings=elo_ratings,
        team_registry=team_registry,
        venue_altitude_m=venue_altitude_m,
        is_home_i=is_home_for_away_team,
    )
    return lh, la


def update_elo(
    team_i_id: str,
    team_j_id: str,
    goals_i: int,
    goals_j: int,
    elo_ratings: dict[str, int],
    k: int = _K_FACTOR,
) -> dict[str, int]:
    """Return updated elo_ratings after team_i vs team_j with given score.

    Returns a shallow copy of elo_ratings with updated values for the two teams.
    """
    elo_i = elo_ratings.get(team_i_id, 1700)
    elo_j = elo_ratings.get(team_j_id, 1700)

    expected_i = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))
    expected_j = 1.0 - expected_i

    if goals_i > goals_j:
        score_i, score_j = 1.0, 0.0
    elif goals_i == goals_j:
        score_i = score_j = 0.5
    else:
        score_i, score_j = 0.0, 1.0

    updated = dict(elo_ratings)
    updated[team_i_id] = round(elo_i + k * (score_i - expected_i))
    updated[team_j_id] = round(elo_j + k * (score_j - expected_j))
    return updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_lambda_j(
    team_i_id: str,
    team_j_id: str,
    elo_ratings: dict[str, int],
    team_registry: dict,
    venue_altitude_m: int = 0,
    is_home_i: bool = False,
) -> float:
    """Compute λ_i from team_i's perspective attacking against team_j."""
    fi = get_factors(team_i_id, team_registry)
    fj = get_factors(team_j_id, team_registry)

    elo_i = elo_ratings.get(team_i_id, 1700)
    elo_j = elo_ratings.get(team_j_id, 1700)

    log_lambda = _LOG_BASE

    log_lambda += _BETA_ELO * (elo_i - elo_j) / 400.0
    log_lambda += _BETA_TALENT * fi["scoring_talent"]
    log_lambda += _BETA_MOMENTUM_SCORED * (
        fi["recent_goals_scored_10"] - _AVG_GOALS_SCORED_10
    )
    log_lambda += _BETA_OPP_CONCEDED * (
        fj["recent_goals_conceded_5"] - _AVG_GOALS_CONCEDED_5
    )

    if fi["is_defending_champion"]:
        log_lambda += _BETA_CHAMPION_SLUMP
    if fi["is_first_world_cup"]:
        log_lambda += _BETA_FIRST_WC
    if fi["is_big_nation"]:
        log_lambda += _BETA_BIG_NATION
    if fj["is_european"]:
        log_lambda += _BETA_VS_EUROPEAN
    if is_home_i:
        log_lambda += _BETA_HOME

    team_i_obj = team_registry.get(team_i_id)
    if (
        venue_altitude_m >= _ALTITUDE_THRESHOLD_M
        and team_i_obj is not None
        and not _team_is_host(team_i_obj.abbreviation.upper())
    ):
        log_lambda += _BETA_ALTITUDE

    # WC historical underperformance (e.g. England)
    if fi.get("wc_underperformer", False):
        log_lambda += _BETA_WC_UNDERPERFORMER

    return math.exp(log_lambda)


def _team_is_host(abbreviation: str) -> bool:
    """Return True if this team is one of the three host nations."""
    vd = _venue_data()
    return abbreviation in set(vd.get("host_nations", []))
