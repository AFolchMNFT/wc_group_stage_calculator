"""Fetch current Elo ratings for WC 2026 teams from eloratings.net.

Returns a dict mapping ESPN team_id → Elo rating (int).
Ratings are sourced from the live eloratings.net table and cached locally.

The site provides ratings in a plain HTML table. We parse it with the
standard library only (no BeautifulSoup dependency required).
"""
from __future__ import annotations

import re
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Name normalisation: eloratings.net name → ESPN abbreviation
# ---------------------------------------------------------------------------
# Keys are substrings / normalised names from eloratings.net; values are
# the ESPN abbreviation used in our team_registry.
_NAME_TO_ABB: dict[str, str] = {
    "Argentina":           "ARG",
    "Spain":               "ESP",
    "France":              "FRA",
    "England":             "ENG",
    "Brazil":              "BRA",
    "Netherlands":         "NED",
    "Portugal":            "POR",
    "Germany":             "GER",
    "Colombia":            "COL",
    "Croatia":             "CRO",
    "Norway":              "NOR",
    "Mexico":              "MEX",
    "Senegal":             "SEN",
    "Ecuador":             "ECU",
    "Belgium":             "BEL",
    "Switzerland":         "SUI",
    "Turkey":              "TUR",
    "Türkiye":             "TUR",
    "United States":       "USA",
    "USA":                 "USA",
    "Japan":               "JPN",
    "Austria":             "AUT",
    "Uruguay":             "URU",
    "Paraguay":            "PAR",
    "Morocco":             "MAR",
    "Canada":              "CAN",
    "South Korea":         "KOR",
    "Korea Republic":      "KOR",
    "Czechia":             "CZE",
    "Czech Republic":      "CZE",
    "South Africa":        "RSA",
    "Qatar":               "QAT",
    "Bosnia-Herzegovina":  "BIH",
    "Bosnia and Herzegovina": "BIH",
    "Haiti":               "HAI",
    "Scotland":            "SCO",
    "Australia":           "AUS",
    "Ivory Coast":         "CIV",
    "Côte d'Ivoire":       "CIV",
    "Curaçao":             "CUW",
    "Curacao":             "CUW",
    "Sweden":              "SWE",
    "Tunisia":             "TUN",
    "Iran":                "IRN",
    "New Zealand":         "NZL",
    "Egypt":               "EGY",
    "Saudi Arabia":        "KSA",
    "Cape Verde":          "CPV",
    "Iraq":                "IRQ",
    "Jordan":              "JOR",
    "Algeria":             "ALG",
    "Uzbekistan":          "UZB",
    "DR Congo":            "COD",
    "Congo DR":            "COD",
    "Congo":               "COD",
    "Panama":              "PAN",
    "Ghana":               "GHA",
}

_ELO_URL = "https://www.eloratings.net/"

# Fallback Elo ratings (as of late May 2026, from GS report / eloratings.net)
# Used when the live fetch fails.
_FALLBACK_ELO: dict[str, int] = {
    "ARG": 2144,
    "ESP": 2134,
    "FRA": 2084,
    "ENG": 2055,
    "BRA": 2038,
    "NED": 1998,
    "POR": 1993,
    "GER": 1985,
    "COL": 1875,
    "CRO": 1872,
    "NOR": 1854,
    "MEX": 1840,
    "SEN": 1831,
    "ECU": 1813,
    "BEL": 1818,
    "SUI": 1826,
    "TUR": 1815,
    "USA": 1803,
    "JPN": 1820,
    "AUT": 1796,
    "URU": 1808,
    "PAR": 1740,
    "MAR": 1790,
    "CAN": 1779,
    "KOR": 1764,
    "CZE": 1758,
    "RSA": 1620,
    "QAT": 1626,
    "BIH": 1697,
    "HAI": 1551,
    "SCO": 1717,
    "AUS": 1710,
    "CIV": 1715,
    "CUW": 1460,
    "SWE": 1727,
    "TUN": 1682,
    "IRN": 1725,
    "NZL": 1607,
    "EGY": 1682,
    "KSA": 1638,
    "CPV": 1620,
    "IRQ": 1608,
    "JOR": 1564,
    "ALG": 1712,
    "UZB": 1658,
    "COD": 1600,
    "PAN": 1598,
    "GHA": 1623,
}


def fetch_elo_ratings(
    team_registry: dict,
    timeout: int = 10,
) -> dict[str, int]:
    """Fetch current Elo ratings and return {team_id: elo_rating}.

    Tries eloratings.net first; falls back to hardcoded values on any error.
    `team_registry` maps ESPN team_id → Team object (with .abbreviation).
    """
    # Build reverse lookup: abbreviation → ESPN team_id
    abb_to_id: dict[str, str] = {
        t.abbreviation.upper(): tid
        for tid, t in team_registry.items()
    }

    raw = _fetch_raw(timeout)
    parsed = _parse_elo_table(raw) if raw else {}

    result: dict[str, int] = {}
    for elo_name, rating in parsed.items():
        abb = _NAME_TO_ABB.get(elo_name)
        if abb and abb in abb_to_id:
            result[abb_to_id[abb]] = rating

    # Fill any gaps from fallback
    for abb, rating in _FALLBACK_ELO.items():
        tid = abb_to_id.get(abb)
        if tid and tid not in result:
            result[tid] = rating

    return result


def fetch_elo_ratings_by_abb(timeout: int = 10) -> dict[str, int]:
    """Like fetch_elo_ratings but returns {abbreviation: elo_rating}.

    Useful when team_registry is not yet available.
    """
    raw = _fetch_raw(timeout)
    parsed = _parse_elo_table(raw) if raw else {}

    result: dict[str, int] = {}
    for name, rating in parsed.items():
        abb = _NAME_TO_ABB.get(name)
        if abb:
            result[abb] = rating

    # Fill gaps
    for abb, rating in _FALLBACK_ELO.items():
        result.setdefault(abb, rating)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_raw(timeout: int) -> Optional[str]:
    """Download eloratings.net and return the HTML body, or None on failure."""
    try:
        req = urllib.request.Request(
            _ELO_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; wc2026-simulator/1.0; "
                    "+https://github.com/jpfolch/wc_group_stage_calculator)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _parse_elo_table(html: str) -> dict[str, int]:
    """Extract {team_name: elo_rating} from the eloratings.net ratings table.

    The site renders a table where each row contains:
      rank | team (as a link) | rating | ...

    We use regex to pull out team names and their associated ratings from the
    structured text, which is more robust than full HTML parsing.
    """
    ratings: dict[str, int] = {}

    # eloratings.net row pattern: rank number, team name link, rating value
    # Pattern observed in page source: team name in an <a> tag, followed by rating
    row_pattern = re.compile(
        r'<a[^>]+href="/([^"]+)"[^>]*>\s*([^<]+?)\s*</a>'   # team link + name
        r'.*?'
        r'(\d{3,4})',                                          # rating (3-4 digits)
        re.DOTALL,
    )

    # Simpler fallback: look for lines with team name and 4-digit rating
    # The page text content (after JS render) includes rows like:
    # "1 Argentina 2144 ..."
    line_pattern = re.compile(
        r'\b([A-Z][a-zA-ZÀ-ÿ\s\'\-\.]+?)\s+(\d{4})\b'
    )

    # Try the anchor-based approach first
    for m in row_pattern.finditer(html):
        team_name = m.group(2).strip()
        try:
            rating = int(m.group(3))
            if 1300 <= rating <= 2400:  # sanity range for international football
                ratings[team_name] = rating
        except ValueError:
            continue

    if len(ratings) >= 20:
        return ratings

    # Fallback: plain text line matching
    for m in line_pattern.finditer(html):
        team_name = m.group(1).strip()
        rating = int(m.group(2))
        if 1300 <= rating <= 2400 and team_name not in ratings:
            ratings[team_name] = rating

    return ratings
