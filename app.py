"""WC 2026 R32 Matchup Predictor — Streamlit frontend."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="WC 2026 R32 Predictor",
    page_icon="⚽",
    layout="wide",
)

# ---- Config ----
_ROOT = Path(__file__).parent
_SETTINGS = json.loads((_ROOT / "config" / "settings.json").read_text())
_POISSON = _SETTINGS.get("poisson_lambdas")
_START = str(_SETTINGS.get("group_stage_start", "20260611"))
_END = str(_SETTINGS.get("group_stage_end", "20260627"))
_CACHE_PATH = _ROOT / "config" / "data_cache.json"

# LOCAL_DEV=true in .env means we're running locally and can fetch/cache live ESPN data.
# When deployed, the app uses only the cache pushed to the repo.
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")
_IS_LOCAL = os.environ.get("LOCAL_DEV", "").lower() in ("1", "true", "yes")

# ---- Session state ----
for _k, _v in {
    "loaded": False,
    "team_registry": {},
    "group_standings": {},
    "completed": [],
    "fixtures": [],
    "elo_ratings": {},
    "sim_result": None,
    "matchup_probs": None,
    "last_updated": "",
    "odds_note": "",
    "cache_loaded_at": "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---- Helpers ----
def _team_name(team_registry: dict, tid: str) -> str:
    t = team_registry.get(tid)
    return t.name if t else tid


def _best_in_slot(slot: str, result) -> tuple:
    """(team_id, %) for the most likely team to fill a slot like '1A' or '2H'."""
    pos = int(slot[0])
    group = slot[1].upper()
    tids = result.groups.get(group, [])
    if not tids:
        return None, 0.0
    best = max(tids, key=lambda t: result.group_finish_counts.get(t, {}).get(pos, 0))
    pct = result.group_finish_counts.get(best, {}).get(pos, 0) / result.n_simulations * 100
    return best, pct


def _best_third_in_group(group: str, result) -> tuple:
    """(team_id, %) for the most likely 3rd-place finisher in a group."""
    tids = result.groups.get(group, [])
    if not tids:
        return None, 0.0
    best = max(tids, key=lambda t: result.group_finish_counts.get(t, {}).get(3, 0))
    pct = result.group_finish_counts.get(best, {}).get(3, 0) / result.n_simulations * 100
    return best, pct


def _compute_group_lineups(result) -> dict[str, dict[int, str]]:
    """Greedy dedup: assign each team in a group to exactly one position.

    For each group, position 1 gets the team with the highest P(1st); position 2 gets
    the team with the highest P(2nd) from the remaining teams; etc.  This prevents the
    same team appearing in two bracket slots (e.g. Spain at both 1E and 2E).

    Returns {group_letter: {1: team_id, 2: team_id, 3: team_id, 4: team_id}}
    """
    lineups: dict[str, dict[int, str]] = {}
    for group, tids in result.groups.items():
        remaining = set(tids)
        assignment: dict[int, str] = {}
        for pos in range(1, 5):
            if not remaining:
                break
            _pos = pos  # local alias avoids lambda-capture gotcha
            best = max(remaining,
                       key=lambda t: result.group_finish_counts.get(t, {}).get(_pos, 0))
            assignment[pos] = best
            remaining.discard(best)
        lineups[group] = assignment
    return lineups


def _load_bracket_topology():
    """Parse world_cup_2026_bracket.json into ordered pair lists for the bracket table.

    Returns (r32_order, r16_pairs, qf_pairs, sf_pairs, final_id, r32_slots) where:
    - r32_order  : list of R32 match IDs in bracket-display order (left→right, top→bottom)
    - *_pairs    : list of (match_id, feeder_home, feeder_away) in that display order
    - final_id   : match ID string of the Final (e.g. "M104")
    - r32_slots  : dict match_id → (slot1, slot2) e.g. ("2A","2B") or ("1E",None) for Annexe C
    """
    wc = json.loads((_ROOT / "config" / "world_cup_2026_bracket.json").read_text())

    def _pm(s):
        return f"M{s.split()[-1]}"

    def _parse_slot(s):
        p = s.split()
        if p[0] == "Winner" and p[1] == "Group":
            return f"1{p[2]}"
        if p[0] == "Runner-up" and p[1] == "Group":
            return f"2{p[2]}"
        return None  # "Best 3rd place …" → Annexe C

    # Feed map: match_id → (feeder_home, feeder_away)
    _feed: dict = {}
    for rnd in ("round_of_16", "quarter_finals", "semi_finals", "final"):
        for m in wc["rounds"][rnd]:
            mid = f"M{m['match']}"
            if "Winner Match" in m["home"] and "Winner Match" in m["away"]:
                _feed[mid] = (_pm(m["home"]), _pm(m["away"]))

    _r32_ids = {f"M{m['match']}" for m in wc["rounds"]["round_of_32"]}
    _r16_ids = {f"M{m['match']}" for m in wc["rounds"]["round_of_16"]}
    _qf_ids  = {f"M{m['match']}" for m in wc["rounds"]["quarter_finals"]}
    _sf_ids  = {f"M{m['match']}" for m in wc["rounds"]["semi_finals"]}
    final_id = f"M{wc['rounds']['final'][0]['match']}"

    def _ordered(mid, target):
        if mid in target:
            return [mid]
        if mid not in _feed:
            return []
        h, a = _feed[mid]
        return _ordered(h, target) + _ordered(a, target)

    r32_order = _ordered(final_id, _r32_ids)
    r16_pairs = [(m,) + _feed[m] for m in _ordered(final_id, _r16_ids)]
    qf_pairs  = [(m,) + _feed[m] for m in _ordered(final_id, _qf_ids)]
    sf_pairs  = [(m,) + _feed[m] for m in _ordered(final_id, _sf_ids)]

    r32_slots = {
        f"M{m['match']}": (_parse_slot(m["home"]), _parse_slot(m["away"]))
        for m in wc["rounds"]["round_of_32"]
    }
    return r32_order, r16_pairs, qf_pairs, sf_pairs, final_id, r32_slots


_BRACKET_TOPOLOGY = _load_bracket_topology()


def _get_odds_api_key() -> str:
    import os
    from dotenv import load_dotenv
    from simulator.paths import ENV_FILE

    load_dotenv(ENV_FILE)

    try:
        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
    except FileNotFoundError:
        pass

    return os.environ.get("ODDS_API_KEY", "")


def _fetch_data() -> None:
    import datetime
    from simulator import espn_client, odds_client

    tr, gs, comp, fix = espn_client.fetch_all(start_date=_START, end_date=_END)

    # Enrich fixtures with per-match Poisson λ values from the odds API
    odds_note = ""
    api_key = _get_odds_api_key()
    if api_key:
        try:
            raw_odds = odds_client.fetch_odds(api_key)
            fix = odds_client.merge_into_fixtures(fix, raw_odds, tr)
            n_with = sum(1 for f in fix if f.lambda_home is not None)
            n_with_totals = sum(
                1 for f in fix
                if f.lambda_home is not None
                and abs(f.lambda_home + (f.lambda_away or 0) - 2.6) > 0.05
            )
            odds_note = (
                f"odds for {n_with}/{len(fix)} fixtures "
                f"({n_with_totals} with totals market)"
            )
        except Exception as exc:
            odds_note = f"odds fetch failed: {exc}"
    else:
        odds_note = "no ODDS_API_KEY — using default λ"

    # Pre-fill manual odds from existing cache for fixtures still lacking API odds
    _try_prefill_manual_odds_from_cache(fix)

    st.session_state.update(
        loaded=True,
        team_registry=tr,
        group_standings=gs,
        completed=comp,
        fixtures=fix,
        odds_note=odds_note,
        sim_result=None,
        matchup_probs=None,
        last_updated=datetime.datetime.now().strftime("%H:%M:%S"),
        cache_loaded_at="",
    )

    # Auto-save to cache so the data is available offline next time
    try:
        from simulator import cache as _cache
        from simulator.elo_client import fetch_elo_ratings as _fetch_elo
        _elo = {}
        try:
            _elo = _fetch_elo(tr)
            st.session_state["elo_ratings"] = _elo
        except Exception:
            pass
        _cache.save(tr, gs, comp, fix, _CACHE_PATH, elo_ratings=_elo or None)
    except Exception:
        pass  # cache write failure is non-fatal


def _load_from_cache() -> None:
    from simulator import cache as _cache
    tr, gs, comp, fix, cached_at, elo = _cache.load(_CACHE_PATH)
    st.session_state.update(
        loaded=True,
        team_registry=tr,
        group_standings=gs,
        completed=comp,
        fixtures=fix,
        elo_ratings=elo,
        sim_result=None,
        matchup_probs=None,
        last_updated="",
        cache_loaded_at=cached_at,
        odds_note=f"{sum(1 for f in fix if f.has_market_odds)}/{len(fix)} fixtures with market odds (cached)",
    )


def _apply_custom_odds(fixtures: list) -> list:
    """Return a copy of fixtures with any manually entered decimal odds applied."""
    import copy
    from simulator.match_model import solve_lambdas, DEFAULT_LAMBDA_TOTAL

    updated = []
    for fix in fixtures:
        oh = st.session_state.get(f"oh_{fix.match_id}")
        od_val = st.session_state.get(f"od_{fix.match_id}")
        oa = st.session_state.get(f"oa_{fix.match_id}")
        if oh and od_val and oa and oh > 1.0 and od_val > 1.0 and oa > 1.0:
            fix = copy.copy(fix)
            raw_h, raw_d, raw_a = 1 / oh, 1 / od_val, 1 / oa
            total = raw_h + raw_d + raw_a
            fix.prob_home = raw_h / total
            fix.prob_draw = raw_d / total
            fix.prob_away = raw_a / total
            lh, la = solve_lambdas(fix.prob_home, DEFAULT_LAMBDA_TOTAL)
            fix.lambda_home = lh
            fix.lambda_away = la
            fix.has_market_odds = True
        updated.append(fix)
    return updated


def _try_prefill_manual_odds_from_cache(fresh_fixtures: list) -> None:
    """For fixtures that didn't get API odds, pre-fill oh_/od_/oa_ session-state keys
    from any manually-saved odds in the cache so the user can re-deploy unchanged odds
    without re-typing.  Only fills keys that are currently None (won't overwrite user input)."""
    if not _CACHE_PATH.exists():
        return
    try:
        from simulator import cache as _cache
        _, _, _, cached_fix, _ = _cache.load(_CACHE_PATH)
    except Exception:
        return
    cached_by_id = {f.match_id: f for f in cached_fix}
    prefilled = 0
    for fix in fresh_fixtures:
        if fix.has_market_odds:
            continue  # already has live API odds
        cf = cached_by_id.get(fix.match_id)
        if cf is None or not cf.has_market_odds:
            continue  # no cached manual odds for this fixture
        ph, pd_, pa = cf.prob_home, cf.prob_draw, cf.prob_away
        if ph <= 0 or pd_ <= 0 or pa <= 0:
            continue
        k_h = f"oh_{fix.match_id}"
        k_d = f"od_{fix.match_id}"
        k_a = f"oa_{fix.match_id}"
        # Only pre-fill if the user hasn't typed anything yet this session
        if (st.session_state.get(k_h) is None
                and st.session_state.get(k_d) is None
                and st.session_state.get(k_a) is None):
            st.session_state[k_h] = round(1.0 / ph, 2)
            st.session_state[k_d] = round(1.0 / pd_, 2)
            st.session_state[k_a] = round(1.0 / pa, 2)
            prefilled += 1
    if prefilled:
        st.session_state["_prefill_note"] = (
            f"✏️ Pre-filled odds from cache for {prefilled} fixture(s) — "
            "check the Missing Odds section below to review."
        )


def _collect_manual_results(fixtures: list) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for fix in fixtures:
        hg = st.session_state.get(f"h_{fix.match_id}")
        ag = st.session_state.get(f"a_{fix.match_id}")
        if hg is not None and ag is not None:
            out[fix.match_id] = (int(hg), int(ag))
    return out


def _run_sim(manual_results: dict, n_sims: int) -> None:
    from simulator import simulate
    from simulator.bracket import load_bracket, resolve_r32_matchups
    fixtures = _apply_custom_odds(st.session_state.fixtures)
    model_cfg = _SETTINGS.get("model", {})
    model_mode = model_cfg.get("mode", "blend")
    elo_weight = model_cfg.get("elo_weight", 0.4)
    elo_ratings = st.session_state.get("elo_ratings") or None
    # If no Elo ratings in session, attempt a live fetch
    if not elo_ratings:
        try:
            from simulator.elo_client import fetch_elo_ratings as _fe
            elo_ratings = _fe(st.session_state.team_registry)
            st.session_state["elo_ratings"] = elo_ratings
        except Exception:
            pass
    result = simulate.run(
        team_registry=st.session_state.team_registry,
        group_standings=st.session_state.group_standings,
        completed=st.session_state.completed,
        fixtures=fixtures,
        n_simulations=n_sims,
        poisson_params=_POISSON,
        manual_results=manual_results,
        model_mode=model_mode,
        elo_weight=elo_weight,
        elo_ratings=elo_ratings,
        simulate_knockout=True,
    )
    bracket = load_bracket()
    st.session_state.sim_result = result
    st.session_state.matchup_probs = resolve_r32_matchups(bracket, result)


# ---- Sidebar ----
with st.sidebar:
    st.title("⚽ WC 2026")
    if st.button("🔄  Fetch Results & Odds", use_container_width=True, type="primary"):
        with st.spinner("Fetching live data…"):
            try:
                _fetch_data()
                teams = st.session_state.team_registry
                completed = st.session_state.completed
                fixtures = st.session_state.fixtures
                st.success(
                    f"Loaded {len(teams)} teams · "
                    f"{len(completed)} completed · "
                    f"{len(fixtures)} remaining"
                )
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")

    if st.session_state.last_updated:
        st.caption(f"Last updated: {st.session_state.last_updated}")
    if st.session_state.cache_loaded_at:
        st.caption(f"📂 From cache: {st.session_state.cache_loaded_at}")
    if st.session_state.get("odds_note"):
        st.caption(f"📈 {st.session_state.odds_note}")
    if st.session_state.get("_prefill_note"):
        st.info(st.session_state["_prefill_note"])
        st.session_state["_prefill_note"] = ""

    # Offline / cache button
    from simulator import cache as _cache_mod
    _cache_ts = _cache_mod.timestamp(_CACHE_PATH)
    if _cache_ts:
        _cache_label = f"📂  Load from Cache\n{_cache_ts}"
        if st.button(_cache_label, use_container_width=True):
            try:
                _load_from_cache()
                st.success("Loaded from cache.")
            except Exception as exc:
                st.error(f"Cache load failed: {exc}")
    else:
        st.button("📂  Load from Cache", use_container_width=True, disabled=True,
                  help="No cache yet — fetch live data first.")

    st.divider()
    n_sims = st.select_slider(
        "Simulations",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        format_func=lambda v: f"{v:,}",
    )


# ---- Main ----
st.title("WC 2026 — R32 Matchup Predictor")
st.caption(
    "Set known upcoming scores, then run the Monte Carlo simulation to see who Mexico plays."
)

if not st.session_state.loaded:
    if not _IS_LOCAL:
        try:
            _load_from_cache()
        except Exception as exc:
            st.error(f"Could not load cache: {exc}")
            st.stop()
    else:
        st.info("Press **🔄 Fetch Results & Odds** in the sidebar to load live standings and fixtures.")
        st.stop()

teams = st.session_state.team_registry
group_standings = st.session_state.group_standings
fixtures: list = st.session_state.fixtures

# ---- Current Standings ----
with st.expander("📊 Current Group Standings", expanded=False):
    gl_list = sorted(group_standings.keys())
    n_cols = min(len(gl_list), 4)
    cols = st.columns(n_cols)
    for idx, gl in enumerate(gl_list):
        with cols[idx % n_cols]:
            entries = sorted(
                group_standings[gl],
                key=lambda s: (-s.points, -(s.gf - s.ga), -s.gf),
            )
            df_stand = pd.DataFrame([
                {
                    "#": i + 1,
                    "Team": _team_name(teams, s.team_id),
                    "Pts": s.points,
                    "GD": s.gd,
                    "GF": s.gf,
                    "P": s.played,
                }
                for i, s in enumerate(entries)
            ])
            st.markdown(f"**Group {gl}**")
            st.dataframe(df_stand, hide_index=True, use_container_width=True, height=175)

# ---- Remaining Fixtures ----
st.subheader("Remaining Fixtures")
if not fixtures:
    st.success("All group stage matches are complete — run the simulation to see standings.")
else:
    st.caption(
        "Enter both goals to fix a score; leave either blank to let it be simulated.  "
        "Odds badge: ✅ market odds  ✏️ custom odds entered  ⚠️ default λ (no odds)"
    )
    by_group: dict[str, list] = {}
    for fix in fixtures:
        by_group.setdefault(fix.group, []).append(fix)

    for gl in sorted(by_group):
        with st.expander(f"Group {gl}  —  {len(by_group[gl])} remaining", expanded=True):
            for fix in by_group[gl]:
                home = _team_name(teams, fix.home_team_id)
                away = _team_name(teams, fix.away_team_id)

                # Determine odds badge from current widget state
                oh = st.session_state.get(f"oh_{fix.match_id}")
                od_v = st.session_state.get(f"od_{fix.match_id}")
                oa = st.session_state.get(f"oa_{fix.match_id}")
                has_custom = bool(oh and od_v and oa and oh > 1.0 and od_v > 1.0 and oa > 1.0)
                if fix.has_market_odds:
                    badge = "✅"
                elif has_custom:
                    badge = "✏️"
                else:
                    badge = "⚠️"

                c1, c2, c3, c4, c5, c6 = st.columns([3.5, 1, 0.4, 1, 3.5, 0.6])
                with c1:
                    st.markdown(
                        f"<p style='text-align:right;padding-top:6px'><b>{home}</b></p>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.number_input(
                        "Home goals", min_value=0, max_value=20, value=None,
                        key=f"h_{fix.match_id}", label_visibility="collapsed",
                    )
                with c3:
                    st.markdown(
                        "<p style='text-align:center;padding-top:6px;color:#888'>–</p>",
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.number_input(
                        "Away goals", min_value=0, max_value=20, value=None,
                        key=f"a_{fix.match_id}", label_visibility="collapsed",
                    )
                with c5:
                    st.markdown(
                        f"<p style='padding-top:6px'><b>{away}</b></p>",
                        unsafe_allow_html=True,
                    )
                with c6:
                    st.markdown(
                        f"<p style='text-align:center;padding-top:6px;font-size:1.1em'>{badge}</p>",
                        unsafe_allow_html=True,
                    )

# ---- Missing Odds section ----
missing_odds_fixtures = [f for f in fixtures if not f.has_market_odds]
if missing_odds_fixtures:
    with st.expander(
        f"⚠️  Enter custom odds for {len(missing_odds_fixtures)} fixture(s) without market data",
        expanded=False,
    ):
        st.caption(
            "Enter decimal odds (e.g. 1.80 / 3.50 / 4.20). "
            "Leave all blank to use the default λ = 2.6 symmetric split.  "
            "Values marked ✏️ were pre-filled from the cache — review and adjust as needed."
        )
        miss_by_group: dict[str, list] = {}
        for fix in missing_odds_fixtures:
            miss_by_group.setdefault(fix.group, []).append(fix)

        for gl in sorted(miss_by_group):
            st.markdown(f"**Group {gl}**")
            for fix in miss_by_group[gl]:
                home = _team_name(teams, fix.home_team_id)
                away = _team_name(teams, fix.away_team_id)
                st.markdown(
                    f"<span style='color:#555'>{home} vs {away}</span>",
                    unsafe_allow_html=True,
                )
                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    st.number_input(
                        f"Home ({home})", min_value=1.01, max_value=100.0,
                        value=None, step=0.05, format="%.2f",
                        key=f"oh_{fix.match_id}", label_visibility="visible",
                    )
                with oc2:
                    st.number_input(
                        "Draw", min_value=1.01, max_value=100.0,
                        value=None, step=0.05, format="%.2f",
                        key=f"od_{fix.match_id}", label_visibility="visible",
                    )
                with oc3:
                    st.number_input(
                        f"Away ({away})", min_value=1.01, max_value=100.0,
                        value=None, step=0.05, format="%.2f",
                        key=f"oa_{fix.match_id}", label_visibility="visible",
                    )

        st.divider()
        if st.button(
            "💾  Add to cache",
            key="save_odds_cache",
            help="Bake the entered odds into the cache file so they load automatically next time",
        ):
            from simulator import cache as _cache_so
            _upd = _apply_custom_odds(st.session_state.fixtures)
            _n_new = (
                sum(1 for f in _upd if f.has_market_odds)
                - sum(1 for f in st.session_state.fixtures if f.has_market_odds)
            )
            st.session_state.fixtures = _upd
            _cache_so.save(
                st.session_state.team_registry,
                st.session_state.group_standings,
                st.session_state.completed,
                _upd,
                _CACHE_PATH,
                elo_ratings=st.session_state.get("elo_ratings") or None,
            )
            st.success(
                f"Saved odds for {_n_new} fixture(s) to cache. "
                "They will be pre-loaded next time you click **Load from Cache**."
            )

# ---- Run simulation ----
manual = _collect_manual_results(fixtures)
n_fixed = len(manual)
n_free = len(fixtures) - n_fixed
btn_label = f"▶  Run Simulation  ({n_fixed} fixed · {n_free} simulated)"

if st.button(btn_label, type="primary", use_container_width=True):
    with st.spinner(f"Running {n_sims:,} Monte Carlo simulations…"):
        try:
            _run_sim(manual, n_sims)
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            st.stop()

if st.session_state.sim_result is None:
    st.stop()

# ---- Results ----
result = st.session_state.sim_result
matchup_probs = st.session_state.matchup_probs
n = result.n_simulations

# Dedup group lineups: each team assigned to exactly one position per group
_group_lineups = _compute_group_lineups(result)

# Locate Mexico
mexico_id: str | None = next(
    (
        tid for tid, t in result.teams.items()
        if "mexico" in t.name.lower() or t.abbreviation.upper() == "MEX"
    ),
    None,
)


def _compute_mc_bracket(result) -> dict:
    """Build bracket display data from Monte Carlo ko_match_stats.

    For each knockout match uses the modal (most frequent) teams from the full
    simulation ensemble and the MC-average expected goals.
    Returns {match_id: {"home": team_id, "away": team_id,
                        "home_g": float, "away_g": float,
                        "winner": team_id, "home_win_pct": float, "round": str}}
    """
    ko_stats = result.ko_match_stats
    if not ko_stats:
        return {}
    mc: dict[str, dict] = {}
    for mid, ms in ko_stats.items():
        pl = ms.get("played", 0)
        if pl == 0 or not ms.get("home_teams") or not ms.get("away_teams"):
            continue
        h_id = max(ms["home_teams"], key=ms["home_teams"].get)
        a_id = max(ms["away_teams"], key=ms["away_teams"].get)
        hwp = ms["home_slot_wins"] / pl * 100
        mc[mid] = {
            "home": h_id,
            "away": a_id,
            "home_g": ms["home_goals_sum"] / pl,
            "away_g": ms["away_goals_sum"] / pl,
            "winner": h_id if hwp >= 50 else a_id,
            "home_win_pct": hwp,
            "round": ms.get("round", ""),
        }
    return mc


tab_mex, tab_groups, tab_third, tab_annexe, tab_bracket, tab_knockout, tab_scores, tab_how = st.tabs(
    ["🇲🇽  Mexico's R32", "📊  Group Finish", "🏅  3rd Place", "📋  Annexe C", "🏆  Predicted Bracket", "⚽  Knockout Odds", "📈  Avg Scores", "ℹ️  How It Works"]
)

# ---- Mexico tab ----
with tab_mex:
    st.header(f"Mexico's R32 Opponent Probabilities  ({n:,} sims)")
    st.caption(
        "Mexico play match **M79** as Group A winners. Their opponent is the best 3rd-place "
        "team from groups C/E/F/H/I, determined by Annexe C."
    )

    if mexico_id is None:
        st.warning("Could not identify Mexico in simulation results.")
    else:
        opps = matchup_probs.get(mexico_id, {})
        if not opps:
            st.info("No R32 matchup data available for Mexico.")
        else:
            from collections import defaultdict as _dd

            # Group opponents by their group letter
            _by_grp: dict[str, list] = _dd(list)
            for _oid, _p in opps.items():
                if _p > 0.001:
                    _opp = result.teams.get(_oid)
                    if _opp:
                        _by_grp[_opp.group].append((_oid, _p))

            _grp_totals = {g: sum(p for _, p in ts) for g, ts in _by_grp.items()}
            _sorted_grps = sorted(_by_grp.keys(), key=lambda g: -_grp_totals[g])
            _max_p = max((p for ts in _by_grp.values() for _, p in ts), default=1.0)

            # Distinct highlight color per group
            _GRP_PALETTE = {
                "C": ("#1565c0", "#e3f2fd"),
                "E": ("#00695c", "#e0f2f1"),
                "F": ("#bf360c", "#fff3e0"),
                "H": ("#4a148c", "#f3e5f5"),
                "I": ("#b71c1c", "#ffebee"),
            }
            _DEF_COLOR = ("#424242", "#f5f5f5")

            for _grp in _sorted_grps:
                _grp_teams = sorted(_by_grp[_grp], key=lambda x: -x[1])
                _total_pct = _grp_totals[_grp] * 100
                _bar_clr, _bg_clr = _GRP_PALETTE.get(_grp, _DEF_COLOR)

                st.markdown(
                    f"<div style='background:{_bg_clr};border-left:4px solid {_bar_clr};"
                    f"padding:8px 12px;margin:14px 0 6px 0;border-radius:4px;'>"
                    f"<b style='color:{_bar_clr};font-size:1em'>Group {_grp}</b>"
                    f"<span style='color:#666;font-size:0.88em;margin-left:10px'>"
                    f"combined {_total_pct:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                for _oid, _prob in _grp_teams:
                    _opp = result.teams.get(_oid)
                    if _opp is None:
                        continue
                    _pct = _prob * 100
                    _bw = _prob / _max_p * 100
                    _fw = "bold" if _pct >= 5 else "normal"
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:12px;margin:4px 0 4px 12px'>"
                        f"<div style='min-width:200px;font-weight:{_fw}'>{_opp.name}</div>"
                        f"<div style='flex:1;background:#e8e8e8;border-radius:4px;height:20px'>"
                        f"<div style='background:{_bar_clr};width:{_bw:.1f}%;height:100%;border-radius:4px'></div>"
                        f"</div>"
                        f"<div style='min-width:48px;text-align:right;font-weight:{_fw}'>{_pct:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ---- Group finish tab ----
with tab_groups:
    st.header(f"Group Finish Probabilities  ({n:,} simulations)")

    # Per-group tables in a 4-column grid
    _gl_list = sorted(result.groups.keys())
    _ncols = 4
    for _chunk_start in range(0, len(_gl_list), _ncols):
        _chunk = _gl_list[_chunk_start : _chunk_start + _ncols]
        _gcols = st.columns(len(_chunk))
        for _ci, _gl in enumerate(_chunk):
            with _gcols[_ci]:
                st.markdown(f"**Group {_gl}**")
                _grows = []
                for _tid in result.groups[_gl]:
                    _fc = result.group_finish_counts.get(_tid, {})
                    _tm = result.teams.get(_tid)
                    _grows.append({
                        "Team": _tm.name if _tm else _tid,
                        "1st %": round(_fc.get(1, 0) / n * 100, 1),
                        "2nd %": round(_fc.get(2, 0) / n * 100, 1),
                        "3rd %": round(_fc.get(3, 0) / n * 100, 1),
                        "Out %": round(_fc.get(4, 0) / n * 100, 1),
                        "R32 %": round(result.r32_counts.get(_tid, 0) / n * 100, 1),
                    })
                _grows.sort(key=lambda r: -r["R32 %"])
                _df_g = pd.DataFrame(_grows)
                st.dataframe(
                    _df_g.style.background_gradient(subset=["R32 %"], cmap="Greens"),
                    hide_index=True,
                    use_container_width=True,
                    height=35 * len(_df_g) + 38,
                )

    st.divider()
    st.subheader("Best 3rd-Place Teams")
    st.caption("8 of 12 groups' 3rd-place teams qualify for the R32. The table shows each group's qualification probability.")
    _third_rows = []
    for _gl in sorted(result.groups):
        _q_pct = result.third_qualified_counts.get(_gl, 0) / n * 100
        _thirds = sorted(
            [
                (
                    result.teams.get(tid).name if result.teams.get(tid) else tid,
                    result.group_finish_counts.get(tid, {}).get(3, 0) / n * 100,
                )
                for tid in result.groups[_gl]
            ],
            key=lambda x: -x[1],
        )
        _cands_list = [(nm, p) for nm, p in _thirds if p > 0.5][:3]
        _cands = ", ".join(f"{nm} {p:.0f}%" for nm, p in _cands_list)
        _third_rows.append({
            "Group": _gl,
            "3rd Qualifies %": round(_q_pct, 1),
            "Most Likely 3rd-Place Teams": _cands,
        })
    _df_third_g = pd.DataFrame(_third_rows)
    st.dataframe(
        _df_third_g.style.background_gradient(subset=["3rd Qualifies %"], cmap="Greens"),
        hide_index=True,
        use_container_width=True,
    )

# ---- 3rd place tab ----
with tab_third:
    st.header("3rd-Place Qualification Probability")
    rows = []
    for gl in sorted(result.groups):
        q_pct = result.third_qualified_counts.get(gl, 0) / n * 100
        thirds = [
            (
                (result.teams.get(tid).name if result.teams.get(tid) else tid),
                result.group_finish_counts.get(tid, {}).get(3, 0) / n * 100,
            )
            for tid in result.groups[gl]
        ]
        thirds = sorted([(nm, p) for nm, p in thirds if p > 0.5], key=lambda x: -x[1])
        candidates = ", ".join(f"{nm} {p:.0f}%" for nm, p in thirds[:3])
        rows.append({
            "Group": gl,
            "3rd Qualifies %": round(q_pct, 1),
            "Most Likely 3rd-Place Teams": candidates,
        })
    df_third = pd.DataFrame(rows)
    st.dataframe(
        df_third.style.background_gradient(subset=["3rd Qualifies %"], cmap="Greens"),
        hide_index=True,
        use_container_width=True,
    )

# ---- Annexe C tab ----
with tab_annexe:
    from simulator.r32_third_place import SLOT_MATCH
    st.header("Annexe C — Winner vs 3rd-Place Matchup Probabilities")
    st.caption(
        "The 8 group winners who face a best 3rd-place team in the R32. "
        "**Most Likely Winner** shows the team most likely to win that group (✓ = virtually guaranteed). "
        "Each opponent group also shows the most likely 3rd-place finisher from that group."
    )
    rows = []
    _winner_guaranteed_flags: dict[str, bool] = {}
    for slot, (match_num, winner_group) in sorted(SLOT_MATCH.items(), key=lambda x: x[1][0]):
        opp_counts = result.annexe_c_opponent_counts.get(slot, {})
        total = sum(opp_counts.values())
        if total == 0:
            continue
        parts = sorted(opp_counts.items(), key=lambda x: -x[1])

        # Most likely group winner
        winner_tid, winner_pct = _best_in_slot(f"1{winner_group}", result)
        winner_name = result.teams[winner_tid].name if winner_tid else "?"
        guaranteed = winner_pct >= 99.5
        _winner_guaranteed_flags[match_num] = guaranteed
        winner_str = f"{winner_name} ({winner_pct:.0f}%)" + (" ✓" if guaranteed else "")

        # Opponent groups with most likely 3rd-place team per group
        parts_desc = []
        for g, c in parts:
            if c / total > 0.02:
                third_tid, _ = _best_third_in_group(g, result)
                third_name = result.teams[third_tid].name if third_tid else "?"
                parts_desc.append(f"Grp {g}: {c / total * 100:.0f}%  ({third_name})")
        desc = "  ·  ".join(parts_desc)

        rows.append({
            "Match": match_num,
            "Slot": f"1{winner_group}",
            "Most Likely Winner": winner_str,
            "Possible 3rd-Place Opponent (% of sims)": desc,
        })
    df_annexe = pd.DataFrame(rows)

    def _style_annexe_df(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for idx, row in df.iterrows():
            if _winner_guaranteed_flags.get(row["Match"], False):
                styles.at[idx, "Most Likely Winner"] = (
                    "background-color:#e8f5e9; color:#1b5e20; font-weight:bold"
                )
        return styles

    st.dataframe(
        df_annexe.style.apply(_style_annexe_df, axis=None),
        hide_index=True,
        use_container_width=True,
    )

# ---- Predicted Bracket tab ----
with tab_bracket:
    from simulator.r32_third_place import SLOT_MATCH as _SM_B

    st.header(f"Simulated Tournament Bracket  ({n:,} simulations)")
    st.caption(
        "R32 slots: most likely team from group-stage simulation. "
        "R16 → Final: Monte Carlo expected outcomes across all simulations. "
        "Each match shows win probability and expected goals; most likely winner in green. Mexico 🇲🇽 highlighted."
    )

    # ── Bracket topology ─────────────────────────────────────────────────
    _R32_ORDER, _R16_PAIRS, _QF_PAIRS, _SF_PAIRS, _final_id, _bkt_r32_slots = _BRACKET_TOPOLOGY

    # ── Annexe C slot map: match_id -> (slot_key, winner_group) ─────────
    _bkt_annexe = {_mn: (_sk, _wg) for _sk, (_mn, _wg) in _SM_B.items()}
    _mex_name_b = result.teams[mexico_id].name if mexico_id else ""

    def _bkt_esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")
    def _bkt_mex(name): return bool(_mex_name_b and _mex_name_b.lower() in name.lower())

    def _slot_info(mid, slot_num):
        """Return {"name":…, "slot":…, "pct":…} for the given R32 match + slot position.
        Uses _group_lineups for dedup: no team appears in two slots of the same group."""
        _s1, _s2 = _bkt_r32_slots[mid]
        _sl = _s1 if slot_num == 1 else _s2
        if _sl is not None:
            _pos = int(_sl[0]); _grp = _sl[1].upper()
            _t = _group_lineups.get(_grp, {}).get(_pos)
            _p = (result.group_finish_counts.get(_t, {}).get(_pos, 0) / result.n_simulations * 100
                  if _t else 0.0)
            return {"name": result.teams[_t].name if _t and result.teams.get(_t) else "?",
                    "slot": _sl, "pct": _p}
        # Annexe C slot (None = 3rd-place side)
        _sk, _wg = _bkt_annexe[mid]
        if slot_num == 1:
            _t = _group_lineups.get(_wg, {}).get(1)
            _p = (result.group_finish_counts.get(_t, {}).get(1, 0) / result.n_simulations * 100
                  if _t else 0.0)
            return {"name": result.teams[_t].name if _t and result.teams.get(_t) else "?",
                    "slot": f"1{_wg}", "pct": _p}
        _oc = result.annexe_c_opponent_counts.get(_sk, {})
        if _oc:
            _ot = sum(_oc.values())
            _bg = max(_oc, key=_oc.get)
            _t2 = _group_lineups.get(_bg, {}).get(3)
            return {"name": result.teams[_t2].name if _t2 and result.teams.get(_t2) else "?",
                    "slot": f"3rd Grp {_bg}", "pct": _oc[_bg] / _ot * 100}
        return {"name": "?", "slot": "3rd", "pct": 0.0}

    # ── MC bracket: modal teams + expected scores from all simulations ──────
    _mc_bracket = _compute_mc_bracket(result)

    def _r32_row_info(mid, slot_num):
        """Return R32 row data using the MC bracket's modal team assignment.

        Falls back to _slot_info (aggregate group-stage stats) when no MC data.
        """
        _s_data = _mc_bracket.get(mid, {})
        _side = "home" if slot_num == 1 else "away"
        _tid = _s_data.get(_side) if _s_data else None
        if not _tid:
            return _slot_info(mid, slot_num)
        _t = result.teams.get(_tid)
        if not _t:
            return _slot_info(mid, slot_num)
        # Derive the position label from how often this team finished at each rank
        _fc = result.group_finish_counts.get(_tid, {})
        _pos = max(range(1, 5), key=lambda p: _fc.get(p, 0)) if _fc else 1
        _pct = _fc.get(_pos, 0) / result.n_simulations * 100
        _lbl = f"3rd Grp {_t.group}" if _pos == 3 else f"{_pos}{_t.group}"
        return {"name": _t.name, "slot": _lbl, "pct": _pct}

    # ── 32 R32 team rows (from MC bracket for full consistency) ─────────
    _bkt_rows = []
    for _mid in _R32_ORDER:
        _bkt_rows.append({**_r32_row_info(_mid, 1), "match": _mid})
        _bkt_rows.append({**_r32_row_info(_mid, 2), "match": _mid})

    def _ko_cell(mid, rowspan):
        """Build an HTML <td> showing MC win probability and expected score."""
        data = _mc_bracket.get(mid, {})
        if data:
            _h_id   = data["home"]
            _a_id   = data["away"]
            _hg     = data["home_g"]
            _ag     = data["away_g"]
            _win_id = data["winner"]
            _hwp    = data["home_win_pct"]
            _awp    = 100 - _hwp
            _h_t  = result.teams.get(_h_id)
            _a_t  = result.teams.get(_a_id)
            _h_name = _h_t.name if _h_t else _h_id
            _a_name = _a_t.name if _a_t else _a_id
            _is_m = _bkt_mex(_h_name) or _bkt_mex(_a_name)
            _h_s = "font-weight:bold;color:#1b5e20" if _win_id == _h_id else "color:#999"
            _a_s = "font-weight:bold;color:#1b5e20" if _win_id == _a_id else "color:#999"
            _content = (
                f'<div style="color:#aaa;font-size:0.70em;margin-bottom:2px">{mid}</div>'
                f'<div style="{_h_s}">{_bkt_esc(_h_name)}&nbsp;<b>{_hwp:.0f}%</b></div>'
                f'<div style="color:#ccc;font-size:0.72em;margin:1px 0">{_hg:.1f}&nbsp;&#8211;&nbsp;{_ag:.1f}</div>'
                f'<div style="{_a_s}"><b>{_awp:.0f}%</b>&nbsp;{_bkt_esc(_a_name)}</div>'
            )
        else:
            _is_m = False
            _content = f'<div style="color:#aaa;font-size:0.70em">{mid}</div><div style="color:#bbb">–</div>'
        _bg = "background:#edf7ed;border:1.5px solid #1b5e20;" if _is_m else "background:#f8f8f8;border:1px solid #e0e0e0;"
        return (
            f'<td rowspan="{rowspan}" style="vertical-align:middle;text-align:center;'
            f'padding:4px 6px;font-size:0.78em;min-width:108px;{_bg}border-radius:4px;">'
            f'{_content}</td>'
        )

    # ── HTML bracket table (32 rows × 5 columns with rowspan) ────────────
    _brows = []
    for _ri in range(32):
        _rd   = _bkt_rows[_ri]
        _is_m = _bkt_mex(_rd["name"])
        _is_last_in_match = (_ri % 2 == 1)
        _is_r16_boundary  = (_ri % 4 == 3) and _ri < 31
        _bdr_b = ("2px solid #bbb" if _is_r16_boundary
                  else ("1px solid #eee" if _is_last_in_match else "none"))

        _r32_td = (
            f'<td style="padding:3px 6px;font-size:0.78em;vertical-align:middle;'
            f'max-width:155px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'border-bottom:{_bdr_b};'
            f'{"background:#edf7ed;border-left:3px solid #1b5e20;" if _is_m else "border-left:2px solid #e8e8e8;"}'
            f'">'
            f'{"🇲🇽 " if _is_m else ""}<b>{_bkt_esc(_rd["name"])}</b>'
            f'<span style="color:#999;font-size:0.82em"> {_rd["slot"]} {_rd["pct"]:.0f}%</span>'
            f'</td>'
        )
        _row = f'<tr style="height:22px;">{_r32_td}'

        if _ri % 4 == 0:
            _r16m = _R16_PAIRS[_ri // 4][0]
            _row += _ko_cell(_r16m, 4)

        if _ri % 8 == 0:
            _qfm = _QF_PAIRS[_ri // 8][0]
            _row += _ko_cell(_qfm, 8)

        if _ri % 16 == 0:
            _sfm = _SF_PAIRS[_ri // 16][0]
            _row += _ko_cell(_sfm, 16)

        if _ri == 0:
            _fin_data = _mc_bracket.get(_final_id, {})
            if _fin_data:
                _fh_t  = result.teams.get(_fin_data["home"])
                _fa_t  = result.teams.get(_fin_data["away"])
                _fw_t  = result.teams.get(_fin_data["winner"])
                _fh_name = _fh_t.name if _fh_t else "?"
                _fa_name = _fa_t.name if _fa_t else "?"
                _fw_name = _fw_t.name if _fw_t else "?"
                _fhwp = _fin_data["home_win_pct"]
                _fawp = 100 - _fhwp
                _fhg, _fag = _fin_data["home_g"], _fin_data["away_g"]
                _fh_s = "font-weight:bold;color:#1b5e20" if _fin_data["winner"] == _fin_data["home"] else "color:#999"
                _fa_s = "font-weight:bold;color:#1b5e20" if _fin_data["winner"] == _fin_data["away"] else "color:#999"
                _row += (
                    f'<td rowspan="32" style="vertical-align:middle;text-align:center;'
                    f'padding:10px;font-size:0.8em;min-width:115px;'
                    f'background:#fffde7;border:2px solid #ffd700;border-radius:6px;">'
                    f'<div style="color:#aaa;font-size:0.72em;margin-bottom:4px">{_final_id} · Final</div>'
                    f'<div style="{_fh_s}">{_bkt_esc(_fh_name)}&nbsp;<b>{_fhwp:.0f}%</b></div>'
                    f'<div style="color:#bbb;margin:3px 0;font-size:0.76em">{_fhg:.1f} – {_fag:.1f} exp.</div>'
                    f'<div style="{_fa_s}"><b>{_fawp:.0f}%</b>&nbsp;{_bkt_esc(_fa_name)}</div>'
                    f'<div style="margin-top:10px;padding-top:6px;border-top:1px solid #e8d000">'
                    f'<div style="color:#888;font-size:0.72em">Most likely winner</div>'
                    f'<div style="font-weight:bold;font-size:0.95em;color:#1b5e20">🏆 {_bkt_esc(_fw_name)}</div>'
                    f'</div></td>'
                )
            else:
                _row += (
                    f'<td rowspan="32" style="vertical-align:middle;text-align:center;'
                    f'padding:10px;font-size:0.8em;min-width:115px;'
                    f'background:#fffde7;border:2px solid #ffd700;border-radius:6px;">'
                    f'<div style="color:#aaa;font-size:0.72em">Final</div>'
                    f'<div style="color:#bbb;font-size:0.8em">Run simulation to see result</div>'
                    f'</td>'
                )

        _row += "</tr>"
        _brows.append(_row)

    _bkt_html = (
        '<div style="overflow-x:auto;padding:6px 0;">'
        '<table style="border-collapse:separate;border-spacing:3px 1px;'
        'font-family:sans-serif;width:100%;">'
        "<thead><tr style=\"font-size:0.78em;color:#555;\">"
        "<th style=\"text-align:left;padding:4px 8px;border-bottom:2px solid #bbb\">R32</th>"
        "<th style=\"text-align:center;padding:4px 8px;border-bottom:2px solid #bbb\">R16</th>"
        "<th style=\"text-align:center;padding:4px 8px;border-bottom:2px solid #bbb\">QF</th>"
        "<th style=\"text-align:center;padding:4px 8px;border-bottom:2px solid #bbb\">SF</th>"
        "<th style=\"text-align:center;padding:4px 8px;border-bottom:2px solid #bbb\">Final 🏆</th>"
        "</tr></thead><tbody>"
        + "".join(_brows)
        + "</tbody></table></div>"
    )
    st.markdown(_bkt_html, unsafe_allow_html=True)


# ---- Knockout Odds tab ----
with tab_knockout:
    st.header(f"Knockout Stage — Monte Carlo Results  ({n:,} simulations)")

    has_ko = bool(result.ko_match_stats)
    if not has_ko:
        st.info(
            "No knockout data available. Re-run the simulation to generate full-tournament "
            "probabilities (requires Elo ratings to be loaded)."
        )
    else:
        _ko_stats = result.ko_match_stats

        # Helper: pick modal team (most common) from a {team_id: count} dict
        def _modal(team_counts: dict) -> tuple[str, int]:
            if not team_counts:
                return "", 0
            best = max(team_counts, key=team_counts.get)
            return best, team_counts[best]

        _ROUND_LABELS = {
            "round_of_32":   ("R32",    "Round of 32"),
            "round_of_16":   ("R16",    "Round of 16"),
            "quarter_finals": ("QF",    "Quarter-Finals"),
            "semi_finals":   ("SF",     "Semi-Finals"),
            "final":         ("Final",  "Final 🏆"),
        }

        for rnd_key, (rnd_short, rnd_label) in _ROUND_LABELS.items():
            # Gather matches for this round from ko_match_stats
            rnd_matches = sorted(
                [(mid, ms) for mid, ms in _ko_stats.items() if ms.get("round") == rnd_key],
                key=lambda x: x[0],
            )
            if not rnd_matches:
                continue

            st.subheader(rnd_label)

            _rows = []
            for mid, ms in rnd_matches:
                played = ms["played"]
                if played == 0:
                    continue

                h_modal_id, h_modal_cnt = _modal(ms["home_teams"])
                a_modal_id, a_modal_cnt = _modal(ms["away_teams"])
                h_modal_t = result.teams.get(h_modal_id)
                a_modal_t = result.teams.get(a_modal_id)
                h_name = h_modal_t.name if h_modal_t else h_modal_id
                a_name = a_modal_t.name if a_modal_t else a_modal_id

                avg_hg = ms["home_goals_sum"] / played
                avg_ag = ms["away_goals_sum"] / played
                home_win_pct = ms["home_slot_wins"] / played * 100
                away_win_pct = 100 - home_win_pct

                _rows.append({
                    "Match": mid,
                    "Home (most likely)": f"{h_name} ({h_modal_cnt / played * 100:.0f}%)",
                    "Exp. Score": f"{avg_hg:.2f} – {avg_ag:.2f}",
                    "Away (most likely)": f"{a_name} ({a_modal_cnt / played * 100:.0f}%)",
                    "Home Slot Win %": round(home_win_pct, 1),
                    "Away Slot Win %": round(away_win_pct, 1),
                    "Sims Played": played,
                })

            if _rows:
                _df_rnd = pd.DataFrame(_rows)
                st.dataframe(
                    _df_rnd.style.background_gradient(
                        subset=["Home Slot Win %"], cmap="RdYlGn", vmin=0, vmax=100
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=35 * len(_df_rnd) + 38,
                )

        st.divider()
        st.subheader("Championship Probabilities")
        st.caption("Team-level probabilities aggregated across all simulation runs.")
        rows_ko = []
        for tid, team in result.teams.items():
            champ = result.champion_counts.get(tid, 0)
            r32 = result.r32_counts.get(tid, 0)
            if r32 == 0 and champ == 0:
                continue
            rows_ko.append({
                "Team": team.name,
                "Group": team.group,
                "R32 %": round(r32 / n * 100, 1),
                "R16 %": round(result.r16_counts.get(tid, 0) / n * 100, 1),
                "QF %":  round(result.qf_counts.get(tid, 0) / n * 100, 1),
                "SF %":  round(result.sf_counts.get(tid, 0) / n * 100, 1),
                "Final %": round(result.final_counts.get(tid, 0) / n * 100, 1),
                "Win %":   round(champ / n * 100, 1),
            })
        rows_ko.sort(key=lambda r: -r["Win %"])
        df_ko = pd.DataFrame(rows_ko)
        st.dataframe(
            df_ko.style.background_gradient(subset=["Win %"], cmap="Greens"),
            hide_index=True,
            use_container_width=True,
            height=min(800, 35 * len(df_ko) + 38),
        )

        st.divider()
        st.subheader("Championship Paths")
        st.caption(
            "Select a team to see how they can win the tournament — "
            "who they beat at each round and the most common complete paths to the title."
        )

        _all_champs_ko = sorted(
            [
                (result.teams[t].name if result.teams.get(t) else t, t, c)
                for t, c in result.champion_counts.items() if c > 0
            ],
            key=lambda x: -x[2],
        )
        if not _all_champs_ko:
            st.info("No championship data yet — run the simulation first.")
        else:
            _sel_name_ko = st.selectbox(
                "Team",
                [nm for nm, _, _ in _all_champs_ko],
                key="champ_path_team",
                label_visibility="collapsed",
            )
            _sel_tid_ko = next((t for nm, t, _ in _all_champs_ko if nm == _sel_name_ko), None)
            if _sel_tid_ko:
                _champ_cnt_ko = result.champion_counts.get(_sel_tid_ko, 0)
                _r32_played = result.r32_counts.get(_sel_tid_ko, 0)

                _col1_ko, _col2_ko, _col3_ko, _col4_ko, _col5_ko, _col6_ko = st.columns(6)
                _col1_ko.metric("R32", f"{_r32_played / n * 100:.1f}%")
                _col2_ko.metric("R16", f"{sum(result.ko_team_paths.get(_sel_tid_ko, {}).get('round_of_32', {}).values()) / n * 100:.1f}%")
                _col3_ko.metric("QF",  f"{result.r16_counts.get(_sel_tid_ko, 0) / n * 100:.1f}%")
                _col4_ko.metric("SF",  f"{result.qf_counts.get(_sel_tid_ko, 0) / n * 100:.1f}%")
                _col5_ko.metric("Final", f"{result.sf_counts.get(_sel_tid_ko, 0) / n * 100:.1f}%")
                _col6_ko.metric("🏆 Win", f"{_champ_cnt_ko / n * 100:.1f}%")

                st.markdown("**Most common opponents beaten per round**")
                _tp_ko = result.ko_team_paths.get(_sel_tid_ko, {})
                _rnd_denominators = {
                    "round_of_32":   _r32_played,
                    "round_of_16":   sum(_tp_ko.get("round_of_32", {}).values()) or 1,
                    "quarter_finals": result.r16_counts.get(_sel_tid_ko, 0) or 1,
                    "semi_finals":   result.qf_counts.get(_sel_tid_ko, 0) or 1,
                    "final":         result.sf_counts.get(_sel_tid_ko, 0) or 1,
                }
                _rnd_labels_ko = {
                    "round_of_32": "R32", "round_of_16": "R16",
                    "quarter_finals": "QF", "semi_finals": "SF", "final": "Final",
                }
                _path_rows = []
                for _rk_ko, _rl_ko in _rnd_labels_ko.items():
                    _opps_ko = _tp_ko.get(_rk_ko, {})
                    _denom_ko = _rnd_denominators[_rk_ko]
                    if not _opps_ko:
                        continue
                    _top_opps_ko = sorted(_opps_ko.items(), key=lambda x: -x[1])[:5]
                    for _oid_ko, _ocnt_ko in _top_opps_ko:
                        _oname_ko = (result.teams[_oid_ko].name
                                     if result.teams.get(_oid_ko) else _oid_ko)
                        _path_rows.append({
                            "Round": _rl_ko,
                            "Opponent": _oname_ko,
                            "Times beaten": _ocnt_ko,
                            "Win % (when playing)": round(_ocnt_ko / _denom_ko * 100, 1),
                        })
                if _path_rows:
                    st.dataframe(
                        pd.DataFrame(_path_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

                # Top complete championship paths
                _cp_ko = result.ko_champion_paths.get(_sel_tid_ko, {})
                if _cp_ko and _champ_cnt_ko > 0:
                    st.markdown("**Top championship paths to the title**")
                    st.caption(f"Each path is a sequence of opponents beaten from R32 → Final "
                               f"across {_champ_cnt_ko:,} simulations where {_sel_name_ko} won.")
                    _top_paths_ko = sorted(_cp_ko.items(), key=lambda x: -x[1])[:10]
                    _path_display = []
                    for _pk_ko, _pc_ko in _top_paths_ko:
                        _names_ko = [
                            (result.teams[o].name if result.teams.get(o) else o)
                            for o in _pk_ko
                        ]
                        _path_display.append({
                            "Path (R32 → Final)": " → ".join(_names_ko),
                            "Sims": _pc_ko,
                            "% of titles": round(_pc_ko / _champ_cnt_ko * 100, 1),
                        })
                    st.dataframe(
                        pd.DataFrame(_path_display),
                        hide_index=True,
                        use_container_width=True,
                    )

# ---- Avg Scores tab ----
with tab_scores:
    st.header(f"Simulated Match Scores  ({n:,} simulations)")
    st.caption(
        "**Final** = already played · **Fixed** = score you set · **Simulated** = mean over all runs"
    )

    completed_list = st.session_state.completed
    all_fixtures = st.session_state.fixtures
    avg_goals = result.fixture_avg_goals

    # Build a lookup: match_id -> MatchFixture for remaining fixtures
    fixture_by_id = {f.match_id: f for f in all_fixtures}

    # Collect all matches into rows
    rows: list[dict] = []

    for m in completed_list:
        rows.append({
            "group": m.group,
            "home": _team_name(teams, m.home_team_id),
            "score": f"{m.home_goals} – {m.away_goals}",
            "away": _team_name(teams, m.away_team_id),
            "status": "Final",
            "_sort": 0,
        })

    for fix in all_fixtures:
        if fix.match_id in manual and fix.match_id in avg_goals:
            hg, ag = avg_goals[fix.match_id]
            score_str = f"{int(hg)} – {int(ag)}"
            status = "Fixed"
        elif fix.match_id in avg_goals:
            hg, ag = avg_goals[fix.match_id]
            score_str = f"{hg:.2f} – {ag:.2f}"
            status = "Simulated"
        else:
            continue
        rows.append({
            "group": fix.group,
            "home": _team_name(teams, fix.home_team_id),
            "score": score_str,
            "away": _team_name(teams, fix.away_team_id),
            "status": status,
            "_sort": 1 if status == "Fixed" else 2,
        })

    if not rows:
        st.info("No match data to display.")
    else:
        # Display one table per group
        all_groups = sorted({r["group"] for r in rows})
        for gl in all_groups:
            group_rows = sorted(
                [r for r in rows if r["group"] == gl],
                key=lambda r: r["_sort"],
            )
            df = pd.DataFrame(
                [
                    {
                        "Home": r["home"],
                        "Score": r["score"],
                        "Away": r["away"],
                        "Status": r["status"],
                    }
                    for r in group_rows
                ]
            )

            def _row_style(row):
                if row["Status"] == "Final":
                    return ["color: #888"] * len(row)
                if row["Status"] == "Fixed":
                    return ["font-weight: bold; color: #1565c0"] * len(row)
                return ["color: #2e7d32"] * len(row)

            st.markdown(f"**Group {gl}**")
            st.dataframe(
                df.style.apply(_row_style, axis=1),
                hide_index=True,
                use_container_width=True,
                height=35 * len(df) + 38,
            )
            st.write("")


# ---- How It Works tab ----
with tab_how:
    st.header("How It Works")

    st.subheader("1 · H2H market → win probabilities")
    st.markdown("""
Bookmakers publish **decimal odds** for each outcome (home win / draw / away win).
For example, odds of **2.50** on a home win imply a probability of 1 ÷ 2.50 = 40 %.

Because bookmakers build in a profit margin (the *overround*), the three raw implied
probabilities sum to more than 100 %. We remove that margin by **normalising**:

```
raw_home  = 1 / odds_home        # e.g. 1/2.50 = 0.400
raw_draw  = 1 / odds_draw        # e.g. 1/3.40 = 0.294
raw_away  = 1 / odds_away        # e.g. 1/3.00 = 0.333
                                 # sum = 1.027  (2.7 % overround)
prob_home = raw_home / 1.027     # 38.9 %
prob_draw = raw_draw / 1.027     # 28.6 %
prob_away = raw_away / 1.027     # 32.4 %
```

Odds from **all available bookmakers** are averaged before conversion, so no single
bookmaker dominates.
""")

    st.subheader("2 · Totals market → expected goals")
    st.markdown("""
The **over/under (totals) market** prices whether the total goals in a match will
exceed a line — usually **2.5** for football. We use this to anchor the expected
total goals λ_total for each match.

Given a 2.5-goal line with prices P_over and P_under, we:

1. **Normalise** to get the fair P(over): strip the bookmaker margin the same way
   as step 1.
2. **Invert the Poisson CDF**: find λ_total such that
   P(Pois(λ_total) ≥ 3) = P(over). This is solved numerically with bisection.

```
# Example: line = 2.5, fair P(over) = 0.52
# Find λ such that 1 − CDF(2, λ) = 0.52
# → CDF(2, λ) = 0.48
# Bisection → λ_total ≈ 2.75 goals expected
```

If no totals market is available, λ_total falls back to **2.6** (WC group-stage
historical average).
""")

    st.subheader("3 · Solving for per-team goal rates")
    st.markdown("""
Given λ_total (how many total goals the market expects) and P(home win) (from the
H2H market), we solve for the individual rates **λ_home** and **λ_away**:

**Constraint 1** — total goals:  `λ_home + λ_away = λ_total`

**Constraint 2** — win probability:
  `P(Pois(λ_home) > Pois(λ_away)) = P(home win)`

Because constraint 1 lets us write `λ_away = λ_total − λ_home`, this reduces to
a one-dimensional bisection on λ_home. The home-win probability is a strictly
monotone function of λ_home, so the bisection always converges.

```
# Example: P(home win) = 0.45, λ_total = 2.75
# Bisect λ_home in (0, 2.75) until
#   P(Pois(λ_h) > Pois(2.75 − λ_h)) ≈ 0.45
# → λ_home ≈ 1.55,  λ_away ≈ 1.20
```

A match between roughly equal teams (P(home win) ≈ 33 %) gets a symmetric split
`λ_home = λ_away = λ_total / 2`.  A heavy favourite gets a much higher λ, producing
more goals on average and bigger realistic winning margins.
""")

    st.subheader("4 · Monte Carlo simulation")
    st.markdown("""
The simulation is run **N times** (default 10 000) independently. In each run:

1. For every remaining fixture, **two independent Poisson draws** are made:
   ```
   home_goals ~ Pois(λ_home)
   away_goals ~ Pois(λ_away)
   ```
   The result (win/draw/loss) follows naturally from the scores — no ad-hoc
   outcome sampling or goal-forcing needed.  The H2H win probability is reproduced
   *implicitly* because that is exactly what the λ values were calibrated to.

2. **Group standings** (points, GF, GA, H2H records) are updated after each match.

3. Groups are **ranked** using the FIFA 2026 tiebreaker rules (see below).

4. The best 8 third-place teams are selected and **Annexe C** is looked up to
   determine each winner-vs-3rd matchup.

After all N runs, every outcome is expressed as a **percentage of simulations**.
""")

    st.subheader("5 · FIFA 2026 group-stage tiebreaker rules")
    st.markdown("""
When two or more teams finish level on points, FIFA 2026 applies these criteria
**in order**:

| Priority | Criterion |
|---|---|
| 1 | Points in **all** group matches |
| 2 | Points in **head-to-head** (H2H) matches among the tied teams |
| 3 | Goal difference in H2H matches |
| 4 | Goals scored in H2H matches |
| 5 | Goal difference in **all** group matches |
| 6 | Goals scored in all group matches |
| 7 | Drawing of lots |

> **Key change from 2022:** H2H record (criteria 2–4) now comes *before* overall
> goal difference (5–6). This is why Mexico's first-place finish is already
> confirmed — their H2H record guarantees it regardless of their final match result.

When **you fix a score** in this app, that result is applied deterministically
before the Monte Carlo runs: the team standings and H2H records are updated, and
only the remaining unfixed matches are simulated.
""")

    st.subheader("6 · Round-of-32 matchup probabilities")
    st.markdown("""
Mexico (**Group A winner, slot 1A**) plays in **match M79**. Their opponent is
the best 3rd-place team from groups **C, E, F, H or I**, determined by Annexe C
of the FIFA regulations.

- **Fixed bracket matches**: probability = fraction of sims where each team
  finished in the required position.
- **Annexe C matches**: each simulation independently determines which eight
  3rd-place groups qualify and looks up the exact Annexe C allocation, so all
  uncertainty in the 3rd-place standings is propagated correctly.
""")

# ---- Share Summary section ----
st.divider()
with st.expander("📋  Share Summary — copy-paste text for any tab"):
    st.caption(
        "Select all text in a box below (Ctrl+A / Cmd+A inside the box) and copy to share."
    )

    from simulator.r32_third_place import SLOT_MATCH as _SM_SHARE

    _share_tabs = st.tabs(
        ["🇲🇽 Mexico", "📊 Groups", "🏅 3rd Place", "📋 Annexe C", "🏆 Bracket", "⚽ Knockout"]
    )

    # ── Mexico summary ───────────────────────────────────────────────────
    with _share_tabs[0]:
        _lines = [f"Mexico R32 Opponent Probabilities ({n:,} sims)\n"]
        if mexico_id:
            _opps = matchup_probs.get(mexico_id, {})
            for _rank, (_oid, _p) in enumerate(
                sorted(_opps.items(), key=lambda x: -x[1])[:12], 1
            ):
                if _p < 0.001:
                    break
                _ot = result.teams.get(_oid)
                _lines.append(
                    f"  {_rank:2d}. {(_ot.name if _ot else _oid):<22s} {_p*100:5.1f}%"
                    + (f"  (Grp {_ot.group})" if _ot else "")
                )
        st.text_area("Mexico R32", "\n".join(_lines), height=280, label_visibility="collapsed")

    # ── Group Finish summary ─────────────────────────────────────────────
    with _share_tabs[1]:
        _lines = [f"Group Finish Probabilities ({n:,} sims)\n"]
        for _gl in sorted(result.groups):
            _lines.append(f"Group {_gl}:")
            _rows_g = []
            for _tid in result.groups[_gl]:
                _fc = result.group_finish_counts.get(_tid, {})
                _tm = result.teams.get(_tid)
                _rows_g.append((_tm.name if _tm else _tid,
                                _fc.get(1,0)/n*100, _fc.get(2,0)/n*100,
                                _fc.get(3,0)/n*100, result.r32_counts.get(_tid,0)/n*100))
            _rows_g.sort(key=lambda x: -x[4])
            for _nm, _p1, _p2, _p3, _pr in _rows_g:
                _lines.append(f"  {_nm:<22s}  1st:{_p1:5.1f}%  2nd:{_p2:5.1f}%  3rd:{_p3:5.1f}%  R32:{_pr:5.1f}%")
            _lines.append("")
        st.text_area("Groups", "\n".join(_lines), height=320, label_visibility="collapsed")

    # ── 3rd Place summary ────────────────────────────────────────────────
    with _share_tabs[2]:
        _lines = [f"3rd-Place Qualification Probabilities ({n:,} sims)\n"]
        for _gl in sorted(result.groups):
            _qp = result.third_qualified_counts.get(_gl, 0) / n * 100
            _thirds_sh = sorted(
                [(result.teams[_t].name if result.teams.get(_t) else _t,
                  result.group_finish_counts.get(_t,{}).get(3,0)/n*100)
                 for _t in result.groups[_gl]],
                key=lambda x: -x[1]
            )
            _cands = ", ".join(f"{_nm} {_p:.0f}%" for _nm, _p in _thirds_sh[:2] if _p > 1)
            _lines.append(f"Grp {_gl}: qualifies {_qp:5.1f}%   candidates: {_cands}")
        st.text_area("3rd Place", "\n".join(_lines), height=260, label_visibility="collapsed")

    # ── Annexe C summary ─────────────────────────────────────────────────
    with _share_tabs[3]:
        _lines = [f"Annexe C — Winner vs 3rd-Place Matchup ({n:,} sims)\n"]
        for _slot_s, (_mn_s, _wg_s) in sorted(_SM_SHARE.items(), key=lambda x: x[1][0]):
            _oc_s = result.annexe_c_opponent_counts.get(_slot_s, {})
            _tot_s = sum(_oc_s.values())
            if _tot_s == 0:
                continue
            _wt, _wp = _best_in_slot(f"1{_wg_s}", result)
            _wn = result.teams[_wt].name if _wt else "?"
            _guar = " (guaranteed)" if _wp >= 99.5 else f" ({_wp:.0f}%)"
            _lines.append(f"{_mn_s}  Slot 1{_wg_s}: {_wn}{_guar}")
            for _g_s, _c_s in sorted(_oc_s.items(), key=lambda x: -x[1]):
                if _c_s / _tot_s < 0.02:
                    continue
                _t3, _ = _best_third_in_group(_g_s, result)
                _n3 = result.teams[_t3].name if _t3 else "?"
                _lines.append(f"      Grp {_g_s}: {_c_s/_tot_s*100:.0f}%  ({_n3})")
            _lines.append("")
        st.text_area("Annexe C", "\n".join(_lines), height=320, label_visibility="collapsed")

    # ── Bracket summary ──────────────────────────────────────────────────
    with _share_tabs[4]:
        _lines = [f"Predicted Bracket ({n:,} sims — R16+ projections)\n"]
        _R32_ORD_S, _R16_S, _QF_S, _SF_S, _fin_id_s, _r32sl_s = _BRACKET_TOPOLOGY
        _bkt_ax_s = {_mn: (_sk, _wg) for _sk, (_mn, _wg) in _SM_SHARE.items()}

        def _si_s(mid, sn):
            _s1, _s2 = _r32sl_s[mid]
            _sl = _s1 if sn == 1 else _s2
            if _sl is not None:
                _pos = int(_sl[0]); _grp = _sl[1].upper()
                _t = _group_lineups.get(_grp, {}).get(_pos)
                _p = (result.group_finish_counts.get(_t, {}).get(_pos, 0) / result.n_simulations * 100
                      if _t else 0.0)
                return result.teams[_t].name if _t and result.teams.get(_t) else "?", _sl, _p
            _sk2, _wg2 = _bkt_ax_s[mid]
            if sn == 1:
                _t = _group_lineups.get(_wg2, {}).get(1)
                _p = (result.group_finish_counts.get(_t, {}).get(1, 0) / result.n_simulations * 100
                      if _t else 0.0)
                return result.teams[_t].name if _t and result.teams.get(_t) else "?", f"1{_wg2}", _p
            _oc2 = result.annexe_c_opponent_counts.get(_sk2, {})
            if _oc2:
                _ot2 = sum(_oc2.values())
                _bg2 = max(_oc2, key=_oc2.get)
                _t2 = _group_lineups.get(_bg2, {}).get(3)
                return (result.teams[_t2].name if _t2 and result.teams.get(_t2) else "?",
                        f"3rd Grp {_bg2}", _oc2[_bg2]/_ot2*100)
            return "?", "3rd", 0.0

        _r32w_s = {}
        for _mid_s in _R32_ORD_S:
            _n1s, _sl1s, _p1s = _si_s(_mid_s, 1)
            _n2s, _sl2s, _p2s = _si_s(_mid_s, 2)
            _lines.append(f"R32  {_mid_s}: {_n1s} ({_sl1s} {_p1s:.0f}%)  vs  {_n2s} ({_sl2s} {_p2s:.0f}%)")
            _r32w_s[_mid_s] = (_n1s, _p1s) if _p1s >= _p2s else (_n2s, _p2s)
        _lines.append("")

        def _proj_s(pairs, src):
            res = {}
            for _m, _a, _b in pairs:
                _wa, _pa = src[_a]
                _wb, _pb = src[_b]
                _rnd = "R16" if len(pairs)==8 else ("QF" if len(pairs)==4 else "SF")
                _lines.append(f"{_rnd}  {_m}: {_wa} ({_pa:.0f}%)  vs  {_wb} ({_pb:.0f}%)")
                res[_m] = (_wa, _pa) if _pa >= _pb else (_wb, _pb)
            _lines.append("")
            return res

        _r16w_s = _proj_s(_R16_S, _r32w_s)
        _qfw_s  = _proj_s(_QF_S,  _r16w_s)
        _sfw_s  = _proj_s(_SF_S,  _qfw_s)
        _sf1s, _sf2s = _SF_S[0][0], _SF_S[1][0]
        _ft1s, _fp1s = _sfw_s.get(_sf1s, ("?", 0))
        _ft2s, _fp2s = _sfw_s.get(_sf2s, ("?", 0))
        _lines.append(f"Final {_fin_id_s}: {_ft1s} ({_fp1s:.0f}%)  vs  {_ft2s} ({_fp2s:.0f}%)")
        _fw_s = _ft1s if _fp1s >= _fp2s else _ft2s
        _lines.append(f"\nPredicted winner: {_fw_s}")

        st.text_area("Bracket", "\n".join(_lines), height=360, label_visibility="collapsed")

    # ── Knockout simulation summary ──────────────────────────────────────
    with _share_tabs[5]:
        _ko_ms = result.ko_match_stats
        _lines = [f"Knockout Simulation Results ({n:,} sims)\n"]
        _lines.append(
            "Format per match: [Simulated score] | MC avg: H.h – A.a  · home-slot wins X%"
        )
        _lines.append("Most likely team shown; appearance % in parentheses.\n")

        _RND_ORDER_SHARE = [
            ("round_of_32",    "Round of 32",    _R32_ORD_S),
            ("round_of_16",    "Round of 16",    [m for m, _, _ in _R16_S]),
            ("quarter_finals", "Quarter-Finals", [m for m, _, _ in _QF_S]),
            ("semi_finals",    "Semi-Finals",    [m for m, _, _ in _SF_S]),
            ("final",          "Final",          [_fin_id_s]),
        ]

        for _rk_s, _rl_s, _mids_s in _RND_ORDER_SHARE:
            _lines.append(f"--- {_rl_s} ---")
            for _mid_s in _mids_s:
                # Showcase (single-run) score
                _sc = _showcase.get(_mid_s, {})
                if _sc:
                    _sh_t = result.teams.get(_sc["home"])
                    _sa_t = result.teams.get(_sc["away"])
                    _sh_n = _sh_t.name if _sh_t else _sc["home"]
                    _sa_n = _sa_t.name if _sa_t else _sc["away"]
                    _sw_t = result.teams.get(_sc["winner"])
                    _sw_n = _sw_t.name if _sw_t else _sc["winner"]
                    _score_str = f"{_sh_n} {_sc['home_g']} – {_sc['away_g']} {_sa_n}  (winner: {_sw_n})"
                else:
                    _score_str = "(no showcase data)"

                # MC aggregate stats
                _ms = _ko_ms.get(_mid_s, {})
                if _ms and _ms["played"] > 0:
                    _pl = _ms["played"]
                    _avg_h = _ms["home_goals_sum"] / _pl
                    _avg_a = _ms["away_goals_sum"] / _pl
                    _hwp = _ms["home_slot_wins"] / _pl * 100
                    _awp = 100 - _hwp
                    _hm_id = max(_ms["home_teams"], key=_ms["home_teams"].get) if _ms["home_teams"] else ""
                    _am_id = max(_ms["away_teams"], key=_ms["away_teams"].get) if _ms["away_teams"] else ""
                    _hm_t = result.teams.get(_hm_id)
                    _am_t = result.teams.get(_am_id)
                    _hm_n = _hm_t.name if _hm_t else _hm_id
                    _am_n = _am_t.name if _am_t else _am_id
                    _hm_pct = _ms["home_teams"].get(_hm_id, 0) / _pl * 100
                    _am_pct = _ms["away_teams"].get(_am_id, 0) / _pl * 100
                    _mc_str = (
                        f"MC: {_hm_n} ({_hm_pct:.0f}%) {_avg_h:.2f} – {_avg_a:.2f} "
                        f"{_am_n} ({_am_pct:.0f}%)  ·  home wins {_hwp:.0f}% / away {_awp:.0f}%"
                    )
                else:
                    _mc_str = "MC: no data"

                _lines.append(f"  {_mid_s}  {_score_str}")
                _lines.append(f"       {_mc_str}")
            _lines.append("")

        # Championship probabilities (top 15)
        _lines.append("--- Championship Probabilities (top 15) ---")
        _champ_rows = sorted(
            [(result.teams[t].name if result.teams.get(t) else t, c)
             for t, c in result.champion_counts.items() if c > 0],
            key=lambda x: -x[1],
        )[:15]
        for _rank_s, (_cn_s, _cc_s) in enumerate(_champ_rows, 1):
            _lines.append(f"  {_rank_s:2d}. {_cn_s:<22s} {_cc_s/n*100:5.1f}%")

        st.text_area(
            "Knockout", "\n".join(_lines), height=420, label_visibility="collapsed"
        )
