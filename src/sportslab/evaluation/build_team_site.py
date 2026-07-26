# ruff: noqa: E501
"""Build static team site for Cloudflare Pages.

Generates per-team pages with 2026 schedule, predictions, and rosters.
Output goes to site/ for Cloudflare Pages deployment.

Usage:
    python -m sportslab.evaluation.build_team_site
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[3]
PREDICTIONS_CSV = BASE / "reports" / "predictions" / "2026_season_predictions.csv"
ROSTER_CACHE = BASE / "data" / "features" / "nfl" / "rosters_2026.parquet"
PLAYER_VALUES_PATH = BASE / "data" / "features" / "nfl" / "player_values_2026.parquet"
OUTPUT = BASE / "site"

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

TEAM_DIVISIONS = {
    "ARI": "NFC West", "ATL": "NFC South", "BAL": "AFC North", "BUF": "AFC East",
    "CAR": "NFC South", "CHI": "NFC North", "CIN": "AFC North", "CLE": "AFC North",
    "DAL": "NFC East", "DEN": "AFC West", "DET": "NFC North", "GB": "NFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "KC": "AFC West",
    "LA": "NFC West", "LAC": "AFC West", "LV": "AFC West", "MIA": "AFC East",
    "MIN": "NFC North", "NE": "AFC East", "NO": "NFC South", "NYG": "NFC East",
    "NYJ": "AFC East", "PHI": "NFC East", "PIT": "AFC North", "SEA": "NFC West",
    "SF": "NFC West", "TB": "NFC South", "TEN": "AFC South", "WAS": "NFC East",
}

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LA": "#003594", "LAC": "#0080C6", "LV": "#000000", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#002244", "WAS": "#5A1414",
}

POSITION_GROUPS = {
    "QB": "Quarterbacks", "RB": "Running Backs", "TE": "Tight Ends",
    "WR": "Wide Receivers", "OL": "Offensive Line",
    "DL": "Defensive Line", "LB": "Linebackers", "DB": "Defensive Backs",
    "K": "Kickers", "P": "Punters", "LS": "Long Snappers", "KR": "Kick Returners",
}

POS_ORDER = ["QB", "RB", "TE", "WR", "OL", "DL", "LB", "DB", "K", "P", "LS", "KR"]

# Map roster positions to value-system position groups
POS_TO_VALUE_GROUP = {
    "QB": "qb",
    "RB": "skill", "FB": "skill", "TE": "skill", "WR": "skill",
    "OL": "ol", "C": "ol", "G": "ol", "T": "ol",
    "DL": "front", "DE": "front", "DT": "front", "NT": "front", "EDGE": "front",
    "LB": "lb", "ILB": "lb", "OLB": "lb", "MLB": "lb",
    "DB": "coverage", "CB": "coverage", "S": "coverage", "SS": "coverage", "FS": "coverage",
    "K": "st", "P": "st", "LS": "st", "KR": "st",
}

VALUE_GROUP_LABELS = {
    "qb": "QB", "skill": "Skill Positions", "ol": "Offensive Line",
    "front": "Defensive Front", "lb": "Linebackers",
    "coverage": "Secondary", "st": "Special Teams",
}


def load_predictions():
    df = pd.read_csv(PREDICTIONS_CSV)
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["week", "gameday"])
    return df


def load_rosters():
    if ROSTER_CACHE.exists():
        return pd.read_parquet(ROSTER_CACHE)
    print("Fetching 2026 rosters from nflreadpy...")
    import nfl_data_py as nfl
    rosters = nfl.import_seasonal_rosters([2026])
    rosters.to_parquet(ROSTER_CACHE)
    return rosters


def load_player_values():
    if not PLAYER_VALUES_PATH.exists():
        print("  Player values not found — skipping roster strength display")
        return None, None, None, None
    values = pd.read_parquet(PLAYER_VALUES_PATH)
    from sportslab.features.player_value import aggregate_by_team
    team_vals = aggregate_by_team(values)

    additions, departures = compute_additions_departures(values)
    return values, team_vals, additions, departures


ADDITIONS_DEPARTURES_TOP_N = 8


def compute_additions_departures(player_values):
    pv = player_values.copy()
    pv["team_2025"] = pv["team_2025"].fillna("").astype(str).str.strip().str.upper()
    pv["team_2026"] = pv["team_2026"].fillna("").astype(str).str.strip().str.upper()
    pv["value"] = pv["value"].fillna(0)

    valid_teams = set(TEAM_NAMES.keys())
    teams = sorted(set(pv["team_2026"].unique()) | set(pv["team_2025"].unique()))
    teams = [t for t in teams if t and t in valid_teams]

    additions = {}
    departures = {}

    for team in teams:
        added = pv[
            (pv["team_2026"] == team)
            & (pv["team_2025"] != team)
            & (pv["value"] != 0)
        ].nlargest(ADDITIONS_DEPARTURES_TOP_N, "value")

        departed = pv[
            (pv["team_2025"] == team)
            & (pv["team_2026"] != team)
            & (pv["value"] != 0)
        ].nlargest(ADDITIONS_DEPARTURES_TOP_N, "value")

        additions[team] = added
        departures[team] = departed

    return additions, departures


def compute_team_stats(preds):
    teams = sorted(set(preds["home_team"].unique()) | set(preds["away_team"].unique()))
    rows = []
    for team in teams:
        h = preds[preds["home_team"] == team]
        a = preds[preds["away_team"] == team]
        pw = h["incumbent_home_win_prob"].sum() + (1 - a["incumbent_home_win_prob"]).sum()
        n = len(h) + len(a)
        rows.append({"team": team, "pred_wins": round(pw, 2), "games": n})
    return pd.DataFrame(rows).sort_values("pred_wins", ascending=False)


def team_tier(wins):
    if wins >= 10:
        return "elite"
    if wins >= 9:
        return "strong"
    if wins >= 8:
        return "mid"
    if wins >= 7:
        return "weak"
    return "rebuild"


def team_schedule(preds, team):
    home = preds[preds["home_team"] == team].copy()
    home["opponent"] = home["away_team"]
    home["team_score"] = home["incumbent_home_win_prob"]
    home["is_home"] = True

    away = preds[preds["away_team"] == team].copy()
    away["opponent"] = away["home_team"]
    away["team_score"] = 1 - away["incumbent_home_win_prob"]
    away["is_home"] = False

    sched = pd.concat([home, away], ignore_index=True)
    sched = sched.sort_values(["week", "gameday"])
    return sched


def position_sort_key(pos):
    return POS_ORDER.index(pos) if pos in POS_ORDER else 99


def _h(s): return str(s) if pd.notna(s) else ""


CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0f; color: #e0e0e0; line-height: 1.6;
    min-height: 100vh;
}
a { color: #4fc3f7; text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 1100px; margin: 0 auto; padding: 0 20px; }
header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 20px 0; border-bottom: 2px solid #2a2a4a;
    margin-bottom: 30px;
}
header h1 { font-size: 1.6rem; color: #fff; }
header h1 span { color: #4fc3f7; }
header .subtitle { color: #888; font-size: 0.85rem; margin-top: 4px; }
nav { margin-top: 10px; }
nav a { color: #aaa; margin-right: 20px; font-size: 0.9rem; }
nav a:hover { color: #fff; }
nav a.active { color: #4fc3f7; font-weight: 600; }

h2 { font-size: 1.3rem; margin: 24px 0 12px; color: #fff; }
h3 { font-size: 1.1rem; margin: 20px 0 10px; color: #ccc; }

.team-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px; margin: 20px 0;
}
.team-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #1e1e38 100%);
    border: 1px solid #2a2a4a; border-radius: 10px; padding: 16px;
    transition: transform 0.15s, box-shadow 0.15s;
}
.team-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
.team-card .abbr { font-size: 1.2rem; font-weight: 700; color: #fff; }
.team-card .name { font-size: 0.8rem; color: #888; }
.team-card .wins { font-size: 2rem; font-weight: 700; margin: 8px 0; }
.team-card .tier { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; padding: 2px 8px; border-radius: 4px; display: inline-block; }
.tier-elite { background: #1b5e20; color: #a5d6a7; }
.tier-strong { background: #0d47a1; color: #90caf9; }
.tier-mid { background: #e65100; color: #ffe0b2; }
.tier-weak { background: #b71c1c; color: #ef9a9a; }
.tier-rebuild { background: #4a148c; color: #ce93d8; }

table { width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 0.9rem; }
th {
    background: #1a1a2e; color: #888; font-weight: 600; text-transform: uppercase;
    font-size: 0.75rem; letter-spacing: 0.5px; padding: 10px 12px; text-align: left;
    border-bottom: 2px solid #2a2a4a;
}
td { padding: 8px 12px; border-bottom: 1px solid #1e1e30; }
tr:hover td { background: rgba(79, 195, 247, 0.04); }
td.team-col { font-weight: 600; }
td.win-cell { font-weight: 700; }
td.win-high { color: #66bb6a; }
td.win-mid { color: #ffa726; }
td.win-low { color: #ef5350; }
td.bucket { font-size: 0.75rem; }
.bucket-80p { color: #66bb6a; }
.bucket-70-80 { color: #81c784; }
.bucket-65-70 { color: #aed581; }
.bucket-60-65 { color: #dce775; }
.bucket-55-60 { color: #fff9c4; }
.bucket-50-55 { color: #aaa; }
.home-badge { display: inline-block; background: #1a3a5c; color: #4fc3f7; font-size: 0.65rem; padding: 1px 6px; border-radius: 3px; margin-left: 4px; }
.away-badge { display: inline-block; background: #3a1a1a; color: #ef5350; font-size: 0.65rem; padding: 1px 6px; border-radius: 3px; margin-left: 4px; }
.caution-flag { display: inline-block; background: #e65100; color: #fff; font-size: 0.6rem; padding: 1px 5px; border-radius: 2px; margin-left: 4px; }

.roster-section { margin: 16px 0; }
.position-group {
    background: #111122; border: 1px solid #1e1e30; border-radius: 8px;
    margin-bottom: 12px; overflow: hidden;
}
.position-header {
    padding: 8px 14px; font-weight: 600; font-size: 0.85rem;
    background: #1a1a2e; color: #aaa; text-transform: uppercase;
    letter-spacing: 0.5px; cursor: pointer;
}
.position-header:hover { background: #222240; }
.player-row {
    display: grid; grid-template-columns: 2fr 1fr 1fr 1fr;
    padding: 6px 14px; font-size: 0.85rem; border-top: 1px solid #1a1a2e;
}
.player-row .name { font-weight: 500; }
.player-row .meta { color: #888; }
.player-row:nth-child(odd) { background: rgba(255,255,255,0.02); }
.roster-count { color: #666; font-size: 0.75rem; margin-left: 8px; }

/* ── Player Value Badge ── */
.player-value {
    display: inline-block; font-size: 0.65rem; font-weight: 700; padding: 0 5px; border-radius: 3px;
    margin-left: 6px; vertical-align: middle; line-height: 1.5;
}
.val-high { background: #1b5e20; color: #a5d6a7; }
.val-mid { background: #663d00; color: #ffe082; }
.val-low { background: #5f2120; color: #ef9a9a; }

/* ── Roster Strength Bars ── */
.strength-bars { margin-bottom: 24px; }
.strength-bar-row {
    display: grid; grid-template-columns: 130px 1fr 32px;
    align-items: center; gap: 8px; margin-bottom: 6px;
}
.strength-label { font-size: 0.8rem; color: #aaa; text-align: right; }
.strength-bar-track {
    height: 14px; background: #1a1a2e; border-radius: 7px; overflow: hidden;
}
.strength-bar-fill { height: 100%; border-radius: 7px; transition: width 0.5s; }
.strength-pctl { font-size: 0.75rem; color: #888; }

/* ── Week-by-Week Schedule ── */
.week-block { margin-bottom: 32px; }
.week-header {
    font-size: 1.1rem; color: #fff; margin-bottom: 8px;
    padding-bottom: 6px; border-bottom: 2px solid #2a2a4a;
}

/* ── Offseason Moves ── */
.moves-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
.moves-col { background: #111122; border: 1px solid #1e1e30; border-radius: 8px; padding: 12px; }
.moves-col h3 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px; }
.move-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; font-size: 0.85rem; border-top: 1px solid #1a1a2e;
}
.move-row:first-of-type { border-top: none; }
.move-name { font-weight: 500; }
.move-meta { color: #888; font-size: 0.75rem; }
@media (max-width: 600px) { .moves-grid { grid-template-columns: 1fr; } }

footer {
    text-align: center; color: #555; font-size: 0.8rem;
    padding: 40px 0 20px; border-top: 1px solid #1a1a2e; margin-top: 40px;
}

@media (max-width: 600px) {
    .team-grid { grid-template-columns: 1fr 1fr; }
    .player-row { grid-template-columns: 1fr 1fr; font-size: 0.8rem; }
}
"""


def render_page(title, body_html, active_nav=""):
    nav_items = [
        ("index.html", "Teams"),
        ("schedule.html", "Schedule"),
        ("standings.html", "Standings"),
    ]
    nav_links = "".join(
        '<a href="{}"{}>{}</a>'.format(
            href,
            ' class="active"' if active_nav == label else "",
            label,
        )
        for href, label in nav_items
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Sports ML Lab</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<header>
<div class="container">
<h1>Sports <span>ML</span> Lab</h1>
<div class="subtitle">2026 NFL Season Predictions — v3.0.0 Frozen QB Overlay</div>
<nav>{nav_links}</nav>
</div>
</header>
<main class="container">
{body_html}
</main>
<footer>
<div class="container">
Research output from <a href="https://github.com/SecuritahGuy/sports-ml-lab">Sports ML Lab</a>.
Not betting advice. QB data is preseason-only; accuracy improves when weekly starters are confirmed.
Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
</div>
</footer>
</body>
</html>"""


def render_index(preds, stats):
    teams = stats.to_dict("records")
    cards = []
    for r in teams:
        t = r["team"]
        tier = team_tier(r["pred_wins"])
        name = TEAM_NAMES.get(t, t)
        color = TEAM_COLORS.get(t, "#2a2a4a")
        cards.append(f"""<a href="teams/{t}.html" class="team-card" style="border-left:4px solid {color}">
<div class="abbr" style="color:{color}">{t}</div>
<div class="name">{name}</div>
<div class="wins">{r["pred_wins"]}</div>
<div class="tier tier-{tier}">{tier}</div>
</a>""")

    html = f"""<h2>2026 Season Predictions</h2>
<p style="color:#888;margin-bottom:20px">{len(preds)} games predicted across all 18 weeks.
Predicted win totals based on Elo + qb_changed + rolling_mov_3 + Platt + Frozen QB Overlay.
Click a team for full schedule and roster.</p>
<div class="team-grid">{"".join(cards)}</div>
<p style="color:#666;font-size:0.8rem">Tiers: elite (10+) · strong (9-10) · mid (8-9) · weak (7-8) · rebuild (&lt;7)</p>"""
    return render_page("Teams", html, "Teams")


def render_standings(stats):
    rows_html = "".join(
        f"<tr style=\"border-left:3px solid {TEAM_COLORS.get(r['team'], '#2a2a4a')}\">"
        f"<td>{i}</td><td class=\"team-col\" style=\"color:{TEAM_COLORS.get(r['team'], '#ccc')}\">{r['team']}</td>"
        f"<td>{TEAM_NAMES.get(r['team'], r['team'])}</td>"
        f"<td>{TEAM_DIVISIONS.get(r['team'], '')}</td>"
        f"<td class=\"win-cell win-{'high' if r['pred_wins'] >= 10 else 'mid' if r['pred_wins'] >= 8 else 'low'}\">"
        f"{r['pred_wins']}</td><td>{r['games']}</td></tr>\n"
        for i, r in enumerate(stats.to_dict("records"), 1)
    )
    html = f"""<h2>2026 Predicted Standings</h2>
<p style="color:#888;margin-bottom:12px">Win totals from Elo-based model. QB data is preseason; will update as weekly starters are confirmed.</p>
<table>
<thead><tr><th>#</th><th>Team</th><th>Name</th><th>Division</th><th>Pred Wins</th><th>Games</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>"""
    return render_page("Standings", html, "Standings")


def render_schedule(preds):
    weeks = sorted(preds["week"].unique())
    sections = []
    for wk in weeks:
        games = preds[preds["week"] == wk].sort_values("gameday")
        date_range = _week_date_range(games)
        rows = []
        for _, g in games.iterrows():
            away = _h(g["away_team"])
            home = _h(g["home_team"])
            prob = g["incumbent_home_win_prob"]
            bucket = g.get("confidence_bucket", "")
            away_color = TEAM_COLORS.get(away, "#ccc")
            home_color = TEAM_COLORS.get(home, "#ccc")

            prob_class = "win-high" if prob >= 0.65 else ("win-mid" if prob >= 0.55 else "win-low")
            bucket_class = f"bucket-{bucket.replace('-','_').replace('+','p')}" if bucket else ""

            cautions = ""
            if str(g.get("caution_qb_change", "")).strip() in ("1", "1.0"):
                cautions += '<span class="caution-flag">QB change</span>'
            if str(g.get("caution_early_season", "")).strip() in ("1", "1.0"):
                cautions += '<span class="caution-flag">Early</span>'
            if str(g.get("caution_neutral", "")).strip() in ("1", "1.0"):
                cautions += '<span class="caution-flag">Neutral</span>'

            day_str = g["gameday"].strftime("%a %b %d") if pd.notna(g["gameday"]) else ""

            rowspan_away = f'<a href="teams/{away}.html" style="color:{away_color};font-weight:600">{TEAM_NAMES.get(away, away)}</a>'
            rowspan_home = f'<a href="teams/{home}.html" style="color:{home_color};font-weight:600">{TEAM_NAMES.get(home, home)}</a>'

            rows.append(f"""<tr>
<td style="color:#666;font-size:0.8rem">{day_str}</td>
<td class="team-col">{rowspan_away}</td>
<td class="team-col">{rowspan_home}</td>
<td class="win-cell {prob_class}">{prob:.0%}</td>
<td class="bucket {bucket_class}">{bucket}</td>
<td>{cautions}</td>
</tr>""")

        sections.append(f"""<div class="week-block">
<h3 class="week-header">Week {wk} <span style="color:#666;font-weight:400;font-size:0.85rem">— {date_range}</span></h3>
<table>
<thead><tr><th>Date</th><th>Away</th><th>Home</th><th>Home Win%</th><th>Confidence</th><th>Cautions</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>""")

    html = f"""<h2>2026 Season — Week-by-Week Predictions</h2>
<p style="color:#888;margin-bottom:20px">{len(preds)} games across {len(weeks)} weeks.
QB data is preseason; accuracy improves as weekly starters are confirmed.</p>
<div class="week-container">{"".join(sections)}</div>"""
    return render_page("Schedule", html, "Schedule")


def _week_date_range(games):
    dates = games["gameday"].dropna()
    if len(dates) == 0:
        return ""
    start = dates.min().strftime("%b %d")
    end = dates.max().strftime("%b %d")
    return f"{start} – {end}"


def render_team_page(team, preds, rosters, stats, player_values=None, team_values=None, additions=None, departures=None):
    name = TEAM_NAMES.get(team, team)
    div = TEAM_DIVISIONS.get(team, "")
    team_stat = stats[stats["team"] == team]
    pw = team_stat["pred_wins"].values[0] if len(team_stat) else 0
    tier = team_tier(pw)

    # ── Roster Strength Summary ──
    strength_html = ""
    if team_values is not None and len(team_values):
        tv = team_values[team_values["team"] == team.upper()]
        if len(tv):
            tv = tv.iloc[0]
            bars = []
            vg_order = ["qb", "skill", "ol", "front", "lb", "coverage"]
            for vg in vg_order:
                pctl = tv.get(f"{vg}_avg_pctl", 0)
                label = VALUE_GROUP_LABELS.get(vg, vg)
                bar_color = "var(--val-high)" if pctl >= 60 else ("var(--val-mid)" if pctl >= 40 else "var(--val-low)")
                bars.append(f"""<div class="strength-bar-row">
<span class="strength-label">{label}</span>
<div class="strength-bar-track">
<div class="strength-bar-fill" style="width:{pctl}%;background:{bar_color}"></div>
</div>
<span class="strength-pctl">{pctl:.0f}</span>
</div>""")
            strength_html = f"""<h2>Roster Strength</h2>
<p style="color:#888;font-size:0.8rem;margin-bottom:8px">Position-group percentile ranks vs all NFL teams (based on 2025 PBP EPA)</p>
<div class="strength-bars">{"".join(bars)}</div>"""

    # ── Schedule ──
    sched = team_schedule(preds, team)
    sched_rows = []
    for _, g in sched.iterrows():
        opponent = g["opponent"]
        opp_name = TEAM_NAMES.get(opponent, opponent)
        team_prob = g["team_score"]
        is_home = g["is_home"]
        bucket = g["confidence_bucket"]
        caution = ""
        if g.get("caution_qb_change") == 1 or str(g.get("caution_qb_change", "")).strip() == "1.0":
            caution += '<span class="caution-flag">QB change</span>'
        if g.get("caution_early_season") == 1 or str(g.get("caution_early_season", "")).strip() == "1.0":
            caution += '<span class="caution-flag">Early</span>'

        prob_class = "win-high" if team_prob >= 0.65 else ("win-mid" if team_prob >= 0.55 else "win-low")
        loc_badge = '<span class="home-badge">HOME</span>' if is_home else '<span class="away-badge">AWAY</span>'
        bucket_class = f"bucket-{bucket.replace('-','_').replace('+','p')}" if bucket else ""

        week_day = g["gameday"].strftime("%a") if pd.notna(g["gameday"]) else ""
        date_str = g["gameday"].strftime("%b %d") if pd.notna(g["gameday"]) else ""

        sched_rows.append(f"""<tr>
<td>W{g['week']}</td>
<td>{week_day}<br><span style="color:#666;font-size:0.75rem">{date_str}</span></td>
<td class="team-col">{opp_name} {loc_badge}</td>
<td class="win-cell {prob_class}">{team_prob:.0%}</td>
<td class="bucket {bucket_class}">{bucket}</td>
<td>{caution}</td>
</tr>""")

    sched_html = f"""<h2>2026 Schedule</h2>
<p style="color:#888;margin-bottom:12px">Predicted win total: <strong style="color:#fff">{pw:.1f}</strong>
<span class="tier tier-{tier}" style="margin-left:8px">{tier}</span></p>
<table>
<thead><tr><th>Week</th><th>Date</th><th>Opponent</th><th>Win Prob</th><th>Confidence</th><th>Cautions</th></tr></thead>
<tbody>{"".join(sched_rows)}</tbody>
</table>"""

    # ── Key Additions / Departures ──
    moves_html = ""
    if additions is not None and departures is not None:
        team_add = additions.get(team, pd.DataFrame())
        team_dep = departures.get(team, pd.DataFrame())

        def _render_moves_list(rows, is_incoming):
            if len(rows) == 0:
                return '<p style="color:#555;font-style:italic">None</p>'
            items = ""
            for _, r in rows.iterrows():
                name = _h(r.get("player_name", "?"))
                pos = _h(r.get("position", ""))
                val = r.get("value", 0)
                other_team = _h(r.get("team_2025", "")) if is_incoming else _h(r.get("team_2026", ""))
                prefix = "from " if is_incoming else "→ "
                team_tag = f" {prefix}{other_team}" if other_team else ""
                items += f"""<div class="move-row">
<span class="move-name">{name}</span>
<span class="move-meta">{pos}{team_tag} · {val:.0f}</span>
</div>"""
            return items

        has_add = len(team_add) > 0
        has_dep = len(team_dep) > 0
        if has_add or has_dep:
            cols = []
            if has_add:
                cols.append(f"""<div class="moves-col">
<h3 style="color:#2ea043">Key Additions</h3>
{_render_moves_list(team_add, True)}
</div>""")
            if has_dep:
                cols.append(f"""<div class="moves-col">
<h3 style="color:#ef5350">Key Departures</h3>
{_render_moves_list(team_dep, False)}
</div>""")
            moves_html = f"""<h2>Offseason Moves</h2>
<p style="color:#888;font-size:0.8rem;margin-bottom:8px">Highest-value players added / lost (based on 2025 PBP EPA)</p>
<div class="moves-grid">{"".join(cols)}</div>"""

    # ── Roster ──
    team_rosters = rosters[rosters["team"] == team] if rosters is not None else pd.DataFrame()
    roster_html = "<h2>Roster</h2>"
    if len(team_rosters) == 0:
        roster_html += '<p style="color:#666">Roster data not yet available for 2026.</p>'
    else:
        active = team_rosters[team_rosters["status"] == "ACT"]
        if len(active) == 0:
            active = team_rosters
        for pos in POS_ORDER:
            group = active[active["position"] == pos]
            if len(group) == 0:
                continue
            group_name = POSITION_GROUPS.get(pos, pos)
            players = group.sort_values("jersey_number")
            player_rows = "".join(
                _render_player_row(p, player_values)
                for _, p in players.iterrows()
            )
            roster_html += f"""<div class="position-group">
<div class="position-header">{group_name} <span class="roster-count">({len(group)})</span></div>
{player_rows}
</div>"""

    color = TEAM_COLORS.get(team, "#2a2a4a")
    body = f"""
<div style="margin-bottom:20px"><a href="../index.html" style="color:#666">&larr; All Teams</a></div>
<div style="padding:16px 20px;border-radius:10px;border-left:4px solid {color};background:rgba(255,255,255,0.02);margin-bottom:20px">
<h2 style="margin:0;color:{color}">{name}</h2>
<div style="color:#888;margin-top:2px">{div}</div>
</div>
{strength_html}
{moves_html}
{sched_html}
{roster_html}"""
    return render_page(f"{team} — {name}", body)


def _render_player_row(p, player_values):
    name = _h(p["player_name"])
    jersey = _h(p["jersey_number"])
    exp = _h(p.get("years_exp", ""))
    college = _h(p.get("college", ""))

    value_tag = ""
    if player_values is not None:
        pid = p.get("player_id", "")
        matches = player_values[player_values["player_id"] == pid]
        if len(matches):
            val = matches.iloc[0]["value"]
            detail = matches.iloc[0].get("detail", "")
            pctl = matches.iloc[0].get("pctl", 50)
            if val != 0:
                rank = "val-high" if pctl >= 60 else ("val-mid" if pctl >= 30 else "val-low")
                value_tag = f"""<span class="player-value {rank}" title="{detail}">{pctl:.0f}</span>"""

    return f"""<div class="player-row">
<span class="name">{name} {value_tag}</span>
<span class="meta">#{jersey}</span>
<span class="meta">{exp} yr</span>
<span class="meta">{college}</span>
</div>"""


def build_site():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "teams").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "assets").mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    preds = load_predictions()
    print(f"  {len(preds)} games loaded")

    print("Loading 2026 rosters...")
    rosters = load_rosters()
    print(f"  {len(rosters)} players loaded")

    print("Loading player values...")
    player_values, team_values, additions, departures = load_player_values()
    if player_values is not None:
        print(f"  {len(player_values)} players with value scores")
        print(f"  Additions/departures computed for {len(additions)} teams")
    else:
        print("  (not found — skipping)")
        additions, departures = None, None

    print("Computing team stats...")
    stats = compute_team_stats(preds)
    print(f"  {len(stats)} teams")

    print("Writing CSS...")
    with open(OUTPUT / "assets" / "style.css", "w") as f:
        f.write(CSS)

    print("Writing index...")
    html = render_index(preds, stats)
    with open(OUTPUT / "index.html", "w") as f:
        f.write(html)

    print("Writing standings...")
    html = render_standings(stats)
    with open(OUTPUT / "standings.html", "w") as f:
        f.write(html)

    print("Writing schedule...")
    html = render_schedule(preds)
    with open(OUTPUT / "schedule.html", "w") as f:
        f.write(html)

    teams = stats["team"].tolist()
    for i, team in enumerate(teams):
        if (i + 1) % 8 == 0:
            print(f"  Writing team pages ({i + 1}/{len(teams)})...")
        html = render_team_page(team, preds, rosters, stats, player_values, team_values, additions, departures)
        with open(OUTPUT / "teams" / f"{team}.html", "w") as f:
            f.write(html)

    print(f"\nDone! Site written to {OUTPUT}")
    print(f"  {len(teams)} team pages")
    print(f"  {len(preds)} games indexed")
    print(f"  {len(rosters)} players in roster database")
    file_count = sum(1 for _ in OUTPUT.rglob("*") if _.is_file())
    print(f"  {file_count} total files")
    print("\nTo test locally:  python -m http.server 8080 -d site")
    print("Cloudflare Pages: set build command to 'python src/sportslab/evaluation/build_team_site.py'")
    print("                  and publish directory to 'site'")


if __name__ == "__main__":
    build_site()
