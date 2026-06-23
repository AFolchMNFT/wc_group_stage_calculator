from __future__ import annotations
"""Rich terminal output for the WC 2026 group stage simulator."""
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns

from simulator.models import GroupStanding, SimResult, Team


def print_live_standings(
    group_standings: dict,
    teams: dict,
    console: Console,
) -> None:
    """Print a table of current live group standings."""
    console.print(Panel("[bold cyan]Current Group Standings[/bold cyan]", expand=False))

    tables = []
    for group_letter in sorted(group_standings.keys()):
        entries = sorted(
            group_standings[group_letter],
            key=lambda s: (-s.points, -s.gd, -s.gf),
        )
        t = Table(
            title=f"[bold]Group {group_letter}[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
            min_width=38,
        )
        t.add_column("#", width=2, justify="right")
        t.add_column("Team", width=18)
        t.add_column("P", justify="center", width=2)
        t.add_column("GD", justify="center", width=3)
        t.add_column("GF", justify="center", width=2)
        t.add_column("Pts", justify="center", width=3)

        for pos, s in enumerate(entries, 1):
            team_name = teams.get(s.team_id, Team(s.team_id, s.team_id, "???", "?")).name
            if pos <= 2:
                style = "green"
            elif pos == 3:
                style = "yellow"
            else:
                style = "red"
            gd = f"+{s.gd}" if s.gd > 0 else str(s.gd)
            t.add_row(str(pos), team_name, str(s.played), gd, str(s.gf), str(s.points), style=style)

        tables.append(t)

    # Print 4 per row
    for i in range(0, len(tables), 4):
        console.print(Columns(tables[i : i + 4]))


def print_simulation_summary(result: SimResult, console: Console) -> None:
    """Print group finish probabilities table."""
    console.print(
        Panel(
            f"[bold cyan]Simulation Summary[/bold cyan]  "
            f"[dim]({result.n_simulations:,} simulations)[/dim]",
            expand=False,
        )
    )

    n = result.n_simulations
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Group", width=6)
    t.add_column("Team", width=22)
    t.add_column("1st%", justify="right", width=6)
    t.add_column("2nd%", justify="right", width=6)
    t.add_column("3rd%", justify="right", width=6)
    t.add_column("Out%", justify="right", width=6)
    t.add_column("R32%", justify="right", width=6)

    for group_letter in sorted(result.groups.keys()):
        team_ids = result.groups[group_letter]
        rows = []
        for tid in team_ids:
            fc = result.group_finish_counts.get(tid, {})
            p1 = fc.get(1, 0) / n * 100
            p2 = fc.get(2, 0) / n * 100
            p3 = fc.get(3, 0) / n * 100
            p4 = fc.get(4, 0) / n * 100
            r32 = result.r32_counts.get(tid, 0) / n * 100
            rows.append((p1, tid, p1, p2, p3, p4, r32))

        rows.sort(reverse=True)

        for idx, (_, tid, p1, p2, p3, p4, r32) in enumerate(rows):
            team = result.teams.get(tid, Team(tid, tid, "???", group_letter))
            g_label = group_letter if idx == 0 else ""
            style = "green" if r32 >= 70 else ("yellow" if r32 >= 40 else "")
            t.add_row(
                g_label,
                team.name,
                f"{p1:.1f}",
                f"{p2:.1f}",
                f"{p3:.1f}",
                f"{p4:.1f}",
                f"[bold]{r32:.1f}[/bold]",
                style=style,
            )
        t.add_section()

    console.print(t)


def print_third_place_summary(result: SimResult, console: Console) -> None:
    """Print the probability of each group's 3rd-place team qualifying."""
    console.print(
        Panel("[bold cyan]3rd-Place Qualification Probability[/bold cyan]", expand=False)
    )
    n = result.n_simulations
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Group", width=6)
    t.add_column("3rd qualifies%", justify="right", width=15)
    t.add_column("Most likely teams finishing 3rd", width=45)

    for group_letter in sorted(result.groups.keys()):
        q_pct = result.third_qualified_counts.get(group_letter, 0) / n * 100
        team_ids = result.groups[group_letter]
        thirds = []
        for tid in team_ids:
            p3 = result.group_finish_counts.get(tid, {}).get(3, 0) / n * 100
            if p3 > 0.5:
                name = result.teams.get(tid, Team(tid, tid, "???", "")).name
                thirds.append(f"{name} {p3:.0f}%")
        thirds.sort(key=lambda s: -float(s.split()[-1].rstrip("%")))
        style = "green" if q_pct >= 70 else ("yellow" if q_pct >= 40 else "")
        t.add_row(group_letter, f"{q_pct:.1f}%", ", ".join(thirds[:3]), style=style)

    console.print(t)


def print_knockout_summary(result: SimResult, console: Console) -> None:
    """Print advancement probabilities for all knockout rounds."""
    n = result.n_simulations
    has_ko = any(result.champion_counts.get(tid, 0) > 0 for tid in result.teams)
    if not has_ko:
        return

    console.print(
        Panel(
            f"[bold cyan]Knockout Stage Probabilities[/bold cyan]  "
            f"[dim]({n:,} simulations)[/dim]",
            expand=False,
        )
    )

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Team", width=22)
    t.add_column("R32%", justify="right", width=6)
    t.add_column("R16%", justify="right", width=6)
    t.add_column("QF%",  justify="right", width=6)
    t.add_column("SF%",  justify="right", width=6)
    t.add_column("Final%", justify="right", width=7)
    t.add_column("Win%",  justify="right", width=6)

    # Sort by winner probability descending
    sorted_teams = sorted(
        result.teams.keys(),
        key=lambda tid: -result.champion_counts.get(tid, 0),
    )

    for tid in sorted_teams:
        champ = result.champion_counts.get(tid, 0)
        if champ == 0 and result.r32_counts.get(tid, 0) == 0:
            continue
        team = result.teams[tid]
        r32_pct   = result.r32_counts.get(tid, 0) / n * 100
        r16_pct   = result.r16_counts.get(tid, 0) / n * 100
        qf_pct    = result.qf_counts.get(tid, 0) / n * 100
        sf_pct    = result.sf_counts.get(tid, 0) / n * 100
        final_pct = result.final_counts.get(tid, 0) / n * 100
        win_pct   = champ / n * 100

        style = "green" if win_pct >= 10 else ("yellow" if win_pct >= 3 else "")
        t.add_row(
            team.name,
            f"{r32_pct:.1f}",
            f"{r16_pct:.1f}",
            f"{qf_pct:.1f}",
            f"{sf_pct:.1f}",
            f"{final_pct:.1f}",
            f"[bold]{win_pct:.1f}[/bold]",
            style=style,
        )

    console.print(t)


    """Print the Annexe C opponent distribution for each group winner that faces a 3rd-place team."""
    from simulator.r32_third_place import SLOT_MATCH

    console.print(
        Panel("[bold cyan]Annexe C — Winner vs 3rd-Place Matchup Probabilities[/bold cyan]", expand=False)
    )
    n = result.n_simulations
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold white")
    t.add_column("Match", width=6)
    t.add_column("Winner slot", width=10)
    t.add_column("Possible 3rd-place opponent group (probability)", width=55)

    for slot, (match_num, winner_group) in sorted(SLOT_MATCH.items(), key=lambda x: x[1][0]):
        opp_counts = result.annexe_c_opponent_counts.get(slot, {})
        total = sum(opp_counts.values())
        if total == 0:
            continue
        parts = sorted(opp_counts.items(), key=lambda x: -x[1])
        desc = "  ".join(f"Grp {g}: {c/total*100:.0f}%" for g, c in parts if c / total > 0.02)
        t.add_row(match_num, f"1{winner_group} (Winner of {winner_group})", desc)

    console.print(t)


def print_sweepstake_report(
    result: SimResult,
    assignments: dict,
    console: Console,
) -> None:
    """Print per-participant sweepstake report."""
    console.print(Panel("[bold magenta]Sweepstake — Participant Outlook[/bold magenta]", expand=False))
    n = result.n_simulations

    participant_scores: list[tuple[float, str]] = []

    for participant, team_names in assignments.items():
        t = Table(
            title=f"[bold]{participant}[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold white",
            min_width=55,
        )
        t.add_column("Team", width=22)
        t.add_column("Group", width=6)
        t.add_column("1st%", justify="right", width=5)
        t.add_column("2nd%", justify="right", width=5)
        t.add_column("R32%", justify="right", width=5)

        total_r32 = 0.0
        for team_name in team_names:
            # Look up team by name
            tid = _find_team_by_name(team_name, result.teams)
            if tid is None:
                t.add_row(team_name, "?", "—", "—", "—")
                continue
            team = result.teams[tid]
            fc = result.group_finish_counts.get(tid, {})
            p1 = fc.get(1, 0) / n * 100
            p2 = fc.get(2, 0) / n * 100
            r32 = result.r32_counts.get(tid, 0) / n * 100
            total_r32 += r32
            style = "green" if r32 >= 70 else ("yellow" if r32 >= 40 else "")
            t.add_row(
                team.name,
                team.group,
                f"{p1:.0f}",
                f"{p2:.0f}",
                f"[bold]{r32:.0f}[/bold]",
                style=style,
            )

        participant_scores.append((total_r32, participant))
        console.print(t)
        console.print()

    # Leaderboard
    participant_scores.sort(reverse=True)
    lb = Table(
        title="[bold]Sweepstake Leaderboard — Sum of R32% across 4 teams[/bold]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
    )
    lb.add_column("#", width=3)
    lb.add_column("Participant", width=18)
    lb.add_column("Sum R32%", justify="right", width=10)
    for rank, (score, name) in enumerate(participant_scores, 1):
        lb.add_row(str(rank), name, f"{score:.1f}")
    console.print(lb)


_SWEEPSTAKE_ALIASES: dict[str, str] = {
    "DR Congo": "Congo DR",
    "Türkiye": "Türkiye",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
    "United States": "United States",
    "South Korea": "South Korea",
    "Ivory Coast": "Ivory Coast",
}


def _find_team_by_name(name: str, teams: dict) -> str | None:
    """Find team_id by display name (case-insensitive; handles common aliases)."""
    candidates = {name, _SWEEPSTAKE_ALIASES.get(name, name)}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        # Exact match
        for tid, team in teams.items():
            if team.name.lower() == candidate_lower:
                return tid
        # Words overlap match (handles "DR Congo" vs "Congo DR")
        candidate_words = set(candidate_lower.split())
        for tid, team in teams.items():
            team_words = set(team.name.lower().split())
            if candidate_words and candidate_words == team_words:
                return tid
        # Substring match
        for tid, team in teams.items():
            if candidate_lower in team.name.lower() or team.name.lower() in candidate_lower:
                return tid
    return None
