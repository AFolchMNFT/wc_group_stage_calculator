"""Fetch H2H and totals odds from The Odds API, derive per-match Poisson λ values."""
from __future__ import annotations

import requests

from simulator.match_model import (
    DEFAULT_LAMBDA_TOTAL,
    lambda_total_from_over_under,
    solve_lambdas,
)

_BASE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
_TIMEOUT = 15

# Map Odds API team names → ESPN displayName where they differ
NAME_ALIASES: dict[str, str] = {
    "USA": "United States",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Republic of Ireland": "Republic of Ireland",
    "Ivory Coast": "Ivory Coast",
    "Trinidad & Tobago": "Trinidad and Tobago",
}


def fetch_odds(
    api_key: str,
    sport_key: str = "soccer_fifa_world_cup",
    regions: str = "uk",
) -> dict[str, dict]:
    """Fetch H2H + totals markets and return per-match odds with derived λ values.

    Returns a dict keyed by match_key (sorted team names, lower-cased, pipe-joined).
    Each value has: prob_home, prob_draw, prob_away, lambda_home, lambda_away,
    lambda_total (for diagnostics), has_totals (bool).
    """
    url = _BASE.format(sport=sport_key)
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    events = resp.json()

    result: dict[str, dict] = {}

    for event in events:
        home_name = _normalise(event.get("home_team", ""))
        away_name = _normalise(event.get("away_team", ""))
        if not home_name or not away_name:
            continue

        # Collect raw prices across bookmakers
        h2h_home: list[float] = []
        h2h_draw: list[float] = []
        h2h_away: list[float] = []
        tot_over_prices: list[float] = []
        tot_under_prices: list[float] = []
        tot_lines: list[float] = []

        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                key = market.get("key")

                if key == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    h = outcomes.get(event["home_team"]) or outcomes.get(home_name)
                    d = outcomes.get("Draw")
                    a = outcomes.get(event["away_team"]) or outcomes.get(away_name)
                    if h and d and a and all(x > 1.0 for x in (h, d, a)):
                        h2h_home.append(h)
                        h2h_draw.append(d)
                        h2h_away.append(a)

                elif key == "totals":
                    by_name = {o["name"]: o for o in market.get("outcomes", [])}
                    if "Over" in by_name and "Under" in by_name:
                        o = by_name["Over"]
                        u = by_name["Under"]
                        o_price = o.get("price", 0)
                        u_price = u.get("price", 0)
                        line = o.get("point") or u.get("point")
                        if o_price > 1.0 and u_price > 1.0 and line is not None:
                            tot_over_prices.append(o_price)
                            tot_under_prices.append(u_price)
                            tot_lines.append(float(line))

        if not h2h_home:
            continue  # no H2H data — skip this event

        # ---- H2H → normalised win probabilities ----
        avg_h = sum(h2h_home) / len(h2h_home)
        avg_d = sum(h2h_draw) / len(h2h_draw)
        avg_a = sum(h2h_away) / len(h2h_away)

        imp_h, imp_d, imp_a = 1 / avg_h, 1 / avg_d, 1 / avg_a
        total_imp = imp_h + imp_d + imp_a
        prob_home = imp_h / total_imp
        prob_draw = imp_d / total_imp
        prob_away = imp_a / total_imp

        # ---- Totals → λ_total ----
        has_totals = bool(tot_lines)
        if has_totals:
            avg_line = sum(tot_lines) / len(tot_lines)
            avg_over_price = sum(tot_over_prices) / len(tot_over_prices)
            avg_under_price = sum(tot_under_prices) / len(tot_under_prices)

            # Normalise to remove bookmaker margin on the totals market
            raw_over = 1 / avg_over_price
            raw_under = 1 / avg_under_price
            prob_over = raw_over / (raw_over + raw_under)

            lambda_total = lambda_total_from_over_under(avg_line, prob_over)
        else:
            lambda_total = DEFAULT_LAMBDA_TOTAL

        # ---- Solve for per-team λ given λ_total and P(home wins) ----
        lh, la = solve_lambdas(prob_home, lambda_total)

        result[_match_key(home_name, away_name)] = {
            "prob_home": prob_home,
            "prob_draw": prob_draw,
            "prob_away": prob_away,
            "lambda_home": lh,
            "lambda_away": la,
            "lambda_total": lambda_total,
            "has_totals": has_totals,
        }

    return result


def merge_into_fixtures(fixtures, odds: dict, teams: dict) -> list:
    """Attach odds and λ values to each MatchFixture; leave unchanged if not found."""
    updated = []
    unmatched = 0

    for fix in fixtures:
        home_team = teams.get(fix.home_team_id)
        away_team = teams.get(fix.away_team_id)

        if home_team is None or away_team is None:
            updated.append(fix)
            continue

        key = _match_key(home_team.name, away_team.name)
        if key in odds:
            od = odds[key]
            fix.prob_home = od["prob_home"]
            fix.prob_draw = od["prob_draw"]
            fix.prob_away = od["prob_away"]
            fix.lambda_home = od["lambda_home"]
            fix.lambda_away = od["lambda_away"]
            fix.has_market_odds = True
        else:
            unmatched += 1

        updated.append(fix)

    if unmatched > 0 and fixtures:
        pct = 100 * unmatched / len(fixtures)
        print(
            f"[yellow]Odds unmatched for {unmatched}/{len(fixtures)} fixtures "
            f"({pct:.0f}%) — uniform 1/3 + default λ used for those.[/yellow]"
        )

    return updated


def _normalise(name: str) -> str:
    return NAME_ALIASES.get(name, name)


def _match_key(name1: str, name2: str) -> str:
    return "|".join(sorted([name1.lower(), name2.lower()]))
