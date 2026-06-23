"""Blend odds-derived and Elo-derived Poisson goal rates.

Strategy:
- Both sources available  → weighted geometric mean of (λ_home, λ_away)
- Only odds available     → pass odds λ through unchanged
- Only Elo available      → use Elo λ (replaces the old uniform-1/3 fallback)
- Neither available       → Elo model always provides a value; this branch is unreachable

Geometric mean is preferred over arithmetic because λ values are rates on a
log scale: blending in log-space preserves their multiplicative structure.
"""
from __future__ import annotations

import math


def blend_lambdas(
    lh_odds: float | None,
    la_odds: float | None,
    lh_elo: float,
    la_elo: float,
    weight_elo: float = 0.4,
) -> tuple[float, float]:
    """Return blended (λ_home, λ_away).

    Parameters
    ----------
    lh_odds / la_odds : λ values from betting markets, or None if unavailable.
    lh_elo / la_elo   : λ values from the Elo regression model (always present).
    weight_elo        : weight to give the Elo model in [0, 1].
                        0.0 = pure odds, 1.0 = pure Elo.
                        Default 0.4 means 60% odds / 40% Elo when both available.

    Returns
    -------
    (λ_home, λ_away) blended rates ready for Poisson simulation.
    """
    weight_elo = max(0.0, min(1.0, weight_elo))
    weight_odds = 1.0 - weight_elo

    if lh_odds is None or la_odds is None or weight_odds <= 0.0:
        # No odds data (or pure-Elo mode): fall back fully to Elo
        return lh_elo, la_elo

    if weight_elo <= 0.0:
        # Pure-odds mode
        return lh_odds, la_odds

    # Geometric mean in log-space
    lh_blend = math.exp(weight_odds * math.log(max(lh_odds, 1e-6)) + weight_elo * math.log(max(lh_elo, 1e-6)))
    la_blend = math.exp(weight_odds * math.log(max(la_odds, 1e-6)) + weight_elo * math.log(max(la_elo, 1e-6)))
    return lh_blend, la_blend
