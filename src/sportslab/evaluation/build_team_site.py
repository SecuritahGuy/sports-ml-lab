# ruff: noqa: E501
"""Build static prediction site for Cloudflare Pages.

Generates:
  - Dashboard homepage with summary cards + current-week matchups
  - Per-team pages with schedule, roster, and offseason moves
  - Schedule page with week selector
  - Standings with conference/division views
  - Model methodology page
  - JSON data files for client-side interactivity

Output: site/ directory
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
PREDICTIONS_CSV = BASE / "reports" / "predictions" / "2026_season_predictions.csv"
ROSTER_CACHE = BASE / "data" / "features" / "nfl" / "rosters_2026.parquet"
PLAYER_VALUES_PATH = BASE / "data" / "features" / "nfl" / "player_values_2026.parquet"
OUTPUT = BASE / "site"

# ── Team Metadata ──

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

CONFERENCE = {
    "ARI": "NFC", "ATL": "NFC", "BAL": "AFC", "BUF": "AFC",
    "CAR": "NFC", "CHI": "NFC", "CIN": "AFC", "CLE": "AFC",
    "DAL": "NFC", "DEN": "AFC", "DET": "NFC", "GB": "NFC",
    "HOU": "AFC", "IND": "AFC", "JAX": "AFC", "KC": "AFC",
    "LA": "NFC", "LAC": "AFC", "LV": "AFC", "MIA": "AFC",
    "MIN": "NFC", "NE": "AFC", "NO": "NFC", "NYG": "NFC",
    "NYJ": "AFC", "PHI": "NFC", "PIT": "AFC", "SEA": "NFC",
    "SF": "NFC", "TB": "NFC", "TEN": "AFC", "WAS": "NFC",
}

POSITION_GROUPS = {
    "QB": "Quarterbacks", "RB": "Running Backs", "TE": "Tight Ends",
    "WR": "Wide Receivers", "OL": "Offensive Line",
    "DL": "Defensive Line", "LB": "Linebackers", "DB": "Defensive Backs",
    "K": "Kickers", "P": "Punters", "LS": "Long Snappers", "KR": "Kick Returners",
}

POS_ORDER = ["QB", "RB", "TE", "WR", "OL", "DL", "LB", "DB", "K", "P", "LS", "KR"]

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
    "qb": "QB", "skill": "Skill", "ol": "OL", "front": "Front",
    "lb": "LB", "coverage": "Secondary", "st": "Sp Teams",
}

VALUE_GROUP_ORDER = ["qb", "skill", "ol", "front", "lb", "coverage"]

TIER_THRESHOLDS = [
    (10, "contender", "Contender"),
    (9, "playoff-mix", "Playoff Mix"),
    (8, "competitive", "Competitive"),
    (7, "long-shot", "Long Shot"),
]

TIER_NAMES = {
    "contender": "Contender", "playoff-mix": "Playoff Mix",
    "competitive": "Competitive", "long-shot": "Long Shot", "underdog": "Underdog",
}

ADDITIONS_DEPARTURES_TOP_N = 8

# ── CSS Design Tokens ──

CSS = r""":root {
  --surface-page: #0b1018;
  --surface-card: #121a26;
  --surface-raised: #182231;
  --surface-hover: #1e2a3a;
  --text-primary: #f3f6fa;
  --text-secondary: #9eabbc;
  --text-muted: #5a6a7e;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-card: rgba(255, 255, 255, 0.06);
  --accent-model: #7ce7d3;
  --accent-blue: #5b9aff;
  --positive: #76d39b;
  --negative: #f08b8b;
  --warning: #efc66a;
  --font-sans: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'Geist Mono', 'IBM Plex Mono', 'JetBrains Mono', 'SF Mono', monospace;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --transition: 0.15s ease;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 16px; }
body {
  font-family: var(--font-sans);
  background: var(--surface-page);
  color: var(--text-primary);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--accent-blue); text-decoration: none; transition: color var(--transition); }
a:hover { color: #8bb9ff; }

.container { max-width: 1120px; margin: 0 auto; padding: 0 24px; }

/* ── Sticky Header ── */
.site-header {
  position: sticky; top: 0; z-index: 100;
  background: rgba(11, 16, 24, 0.92);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border-subtle);
}
.header-inner {
  display: flex; align-items: center; justify-content: space-between;
  height: 56px; gap: 24px;
}
.header-left { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.header-logo {
  font-size: 1.05rem; font-weight: 700; color: var(--text-primary);
  letter-spacing: -0.02em;
}
.header-logo span { color: var(--accent-model); }
.header-tagline {
  font-size: 0.72rem; color: var(--text-muted);
  display: none;
}
@media (min-width: 768px) { .header-tagline { display: block; } }

.header-nav { display: flex; align-items: center; gap: 4px; }
.header-nav a {
  font-size: 0.82rem; font-weight: 500; color: var(--text-secondary);
  padding: 6px 12px; border-radius: var(--radius-sm);
  transition: background var(--transition), color var(--transition);
}
.header-nav a:hover, .header-nav a.active {
  background: var(--surface-raised); color: var(--text-primary);
}
.header-nav a.active { color: var(--accent-model); }
.header-search { display: none; }
@media (min-width: 768px) {
  .header-search {
    display: block; position: relative; margin-left: auto;
  }
  .header-search input {
    background: var(--surface-card); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm); padding: 6px 12px;
    font-size: 0.8rem; color: var(--text-primary); width: 180px;
    font-family: var(--font-sans);
  }
  .header-search input:focus {
    outline: none; border-color: var(--accent-model);
  }
  .header-search input::placeholder { color: var(--text-muted); }
}

/* ── Model Status Bar ── */
.model-status {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 8px;
  background: var(--surface-raised); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md); padding: 10px 16px; margin-bottom: 24px;
  font-size: 0.78rem;
}
.model-status-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.model-status-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: rgba(124, 231, 211, 0.1); border: 1px solid rgba(124, 231, 211, 0.2);
  color: var(--accent-model); padding: 2px 8px; border-radius: 4px;
  font-family: var(--font-mono); font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.3px;
}
.model-status-badge::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent-model); display: inline-block;
}
.model-status-text { color: var(--text-secondary); }
.model-status-time { color: var(--text-muted); font-family: var(--font-mono); font-size: 0.7rem; }

/* ── Dashboard Summary Cards ── */
.summary-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px; margin-bottom: 28px;
}
.summary-card {
  background: var(--surface-card); border: 1px solid var(--border-card);
  border-radius: var(--radius-md); padding: 16px 18px;
  transition: background var(--transition);
}
.summary-card:hover { background: var(--surface-hover); }
.summary-card-label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--text-muted); margin-bottom: 4px;
}
.summary-card-value {
  font-size: 1.2rem; font-weight: 700; color: var(--text-primary);
  font-family: var(--font-mono);
}
.summary-card-sub {
  font-size: 0.75rem; color: var(--text-secondary); margin-top: 2px;
}

/* ── Matchup Cards ── */
.matchup-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 14px; margin-bottom: 28px;
}
.matchup-card {
  background: var(--surface-card); border: 1px solid var(--border-card);
  border-radius: var(--radius-md); padding: 14px 16px;
  transition: background var(--transition);
}
.matchup-card:hover { background: var(--surface-hover); }
.matchup-teams {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 8px;
}
.matchup-team { display: flex; flex-direction: column; }
.matchup-team-abbr {
  font-size: 1rem; font-weight: 700;
}
.matchup-team-name { font-size: 0.72rem; color: var(--text-secondary); }
.matchup-vs { font-size: 0.75rem; color: var(--text-muted); padding: 0 8px; }
.matchup-location {
  text-align: center; font-size: 0.68rem; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.matchup-bar-track {
  height: 6px; background: var(--surface-page); border-radius: 3px;
  overflow: hidden; margin-bottom: 6px;
}
.matchup-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.matchup-meta {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.78rem;
}
.matchup-pick { font-weight: 600; }
.matchup-confidence { color: var(--text-muted); font-size: 0.72rem; }
.matchup-cautions { display: flex; gap: 4px; flex-wrap: wrap; }

/* ── Week Selector ── */
.week-selector {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 20px; flex-wrap: wrap;
}
.week-selector-btn {
  background: var(--surface-card); border: 1px solid var(--border-subtle);
  color: var(--text-secondary); padding: 6px 14px; border-radius: var(--radius-sm);
  font-size: 0.82rem; cursor: pointer; transition: all var(--transition);
  font-family: var(--font-sans);
}
.week-selector-btn:hover { background: var(--surface-raised); color: var(--text-primary); }
.week-selector-btn.active { background: rgba(124, 231, 211, 0.1); border-color: var(--accent-model); color: var(--accent-model); }
.week-date-range { font-size: 0.78rem; color: var(--text-muted); }

/* ── Section Headers ── */
.section-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 14px;
}
.section-header h2 {
  font-size: 1.1rem; font-weight: 600;
}
.section-link { font-size: 0.8rem; }

/* ── Team Grid ── */
.team-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px; margin-bottom: 28px;
}
.team-card {
  background: var(--surface-card); border: 1px solid var(--border-card);
  border-radius: var(--radius-md); padding: 14px;
  border-left: 4px solid var(--text-muted);
  transition: background var(--transition), transform var(--transition);
  display: block; color: var(--text-primary);
}
.team-card:hover { background: var(--surface-hover); transform: translateY(-2px); color: var(--text-primary); }
.team-card-rank { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); }
.team-card-abbr { font-size: 1.05rem; font-weight: 700; margin-top: 2px; }
.team-card-name { font-size: 0.72rem; color: var(--text-secondary); }
.team-card-wins { font-size: 1.5rem; font-weight: 700; margin-top: 6px; }
.team-card-wins small { font-size: 0.65rem; font-weight: 400; color: var(--text-muted); }
.team-card-tier {
  font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.8px;
  padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 4px;
}
.tier-contender { background: rgba(118, 211, 155, 0.12); color: var(--positive); }
.tier-playoff-mix { background: rgba(91, 154, 255, 0.12); color: var(--accent-blue); }
.tier-competitive { background: rgba(239, 198, 106, 0.12); color: var(--warning); }
.tier-long-shot { background: rgba(240, 139, 139, 0.12); color: var(--negative); }
.tier-underdog { background: rgba(180, 180, 180, 0.12); color: var(--text-muted); }

/* ── Tables ── */
.table-wrap { overflow-x: auto; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th {
  background: var(--surface-card);
  color: var(--text-muted); font-weight: 600; text-transform: uppercase;
  font-size: 0.68rem; letter-spacing: 0.5px; padding: 10px 12px;
  text-align: left; border-bottom: 1px solid var(--border-subtle);
  white-space: nowrap;
}
td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.03); }
tr:hover td { background: rgba(255,255,255,0.02); }
td.team-col { font-weight: 600; }
td.num { font-family: var(--font-mono); font-weight: 600; }
td.pos { color: var(--positive); }
td.neg { color: var(--negative); }
td.neutral { color: var(--warning); }

/* ── Probability coloring ── */
.prob-bar {
  display: inline-block; min-width: 60px; text-align: right;
  font-family: var(--font-mono); font-size: 0.85rem; font-weight: 600;
}
.prob-fav { color: var(--positive); }
.prob-even { color: var(--warning); }
.prob-dog { color: var(--negative); }

/* ── Badges ── */
.badge {
  display: inline-block; font-size: 0.62rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.3px;
  padding: 1px 6px; border-radius: 3px;
}
.badge-home { background: rgba(118, 211, 155, 0.12); color: var(--positive); }
.badge-away { background: rgba(240, 139, 139, 0.1); color: var(--negative); }
.badge-neutral { background: rgba(255,255,255,0.06); color: var(--text-muted); }
.caution-flag {
  display: inline-flex; align-items: center; gap: 3px;
  font-size: 0.6rem; font-weight: 600; text-transform: uppercase;
  padding: 1px 6px; border-radius: 3px;
  background: rgba(239, 198, 106, 0.12); color: var(--warning);
}

/* ── Team Hero ── */
.team-hero {
  margin-bottom: 24px;
}
.team-hero-header {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 8px;
}
.team-hero-nav { display: flex; gap: 8px; }
.team-hero-nav a {
  font-size: 0.75rem; color: var(--text-muted); padding: 4px 10px;
  border: 1px solid var(--border-subtle); border-radius: var(--radius-sm);
  transition: all var(--transition);
}
.team-hero-nav a:hover { background: var(--surface-raised); color: var(--text-primary); }
.team-hero-name { font-size: 1.5rem; font-weight: 700; }
.team-hero-meta { color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 12px; }
.team-hero-stats {
  display: flex; gap: 24px; flex-wrap: wrap;
}
.hero-stat { }
.hero-stat-value {
  font-size: 1.8rem; font-weight: 700; font-family: var(--font-mono);
  line-height: 1.2;
}
.hero-stat-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.3px; }

/* ── Roster Strength ── */
.strength-section { margin-bottom: 28px; }
.strength-row {
  display: grid; grid-template-columns: 90px 1fr 36px;
  align-items: center; gap: 8px; margin-bottom: 5px;
}
.strength-label { font-size: 0.78rem; color: var(--text-secondary); text-align: right; }
.strength-track {
  height: 10px; background: var(--surface-page); border-radius: 5px; overflow: hidden;
}
.strength-fill { height: 100%; border-radius: 5px; transition: width 0.5s; }
.strength-pctl { font-size: 0.72rem; color: var(--text-muted); font-family: var(--font-mono); }

/* ── Offseason Moves ── */
.moves-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 24px; }
.moves-col {
  background: var(--surface-card); border: 1px solid var(--border-card);
  border-radius: var(--radius-md); padding: 12px;
}
.moves-col h3 { font-size: 0.8rem; margin-bottom: 8px; letter-spacing: 0.3px; }
.move-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 0; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.04);
}
.move-row:first-of-type { border-top: none; }
.move-name { font-weight: 500; }
.move-meta { color: var(--text-muted); font-size: 0.72rem; }
@media (max-width: 600px) { .moves-grid { grid-template-columns: 1fr; } }

/* ── Roster ── */
.roster-section { margin-bottom: 24px; }
.position-group {
  background: var(--surface-card); border: 1px solid var(--border-card);
  border-radius: var(--radius-sm); margin-bottom: 8px; overflow: hidden;
}
.position-header {
  padding: 8px 14px; font-weight: 600; font-size: 0.82rem;
  color: var(--text-secondary); cursor: pointer; user-select: none;
  display: flex; justify-content: space-between;
}
.position-header:hover { background: var(--surface-raised); }
.position-count { color: var(--text-muted); font-weight: 400; font-size: 0.75rem; }
.player-row {
  display: grid; grid-template-columns: 2fr 1fr 1fr;
  padding: 5px 14px; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.03);
}
.player-row .name { font-weight: 500; }
.player-row .meta { color: var(--text-muted); }
.player-row:nth-child(even) { background: rgba(255,255,255,0.015); }

.player-value {
  display: inline-block; font-size: 0.6rem; font-weight: 700;
  padding: 0 5px; border-radius: 3px; margin-left: 5px;
  vertical-align: middle; line-height: 1.5;
}
.val-high { background: rgba(118, 211, 155, 0.15); color: var(--positive); }
.val-mid { background: rgba(239, 198, 106, 0.15); color: var(--warning); }
.val-low { background: rgba(240, 139, 139, 0.12); color: var(--negative); }

/* ── Model Page ── */
.model-section { margin-bottom: 28px; }
.model-section h2 { font-size: 1.1rem; margin-bottom: 12px; }
.model-section p { color: var(--text-secondary); font-size: 0.88rem; line-height: 1.7; margin-bottom: 8px; }
.pipeline {
  display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
  margin: 16px 0; padding: 14px; background: var(--surface-card);
  border-radius: var(--radius-md); border: 1px solid var(--border-card);
}
.pipeline-step {
  background: var(--surface-raised); padding: 6px 14px;
  border-radius: var(--radius-sm); font-size: 0.8rem; font-weight: 500;
}
.pipeline-arrow { color: var(--text-muted); font-size: 0.8rem; }
.metric-cards {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.metric-card {
  background: var(--surface-card); border: 1px solid var(--border-card);
  border-radius: var(--radius-md); padding: 14px;
}
.metric-value {
  font-size: 1.3rem; font-weight: 700; font-family: var(--font-mono);
}
.metric-label { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }
.metric-desc { font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px; }

.version-list { list-style: none; }
.version-list li {
  display: flex; gap: 12px; padding: 10px 0;
  border-bottom: 1px solid var(--border-subtle); font-size: 0.85rem;
}
.version-list li:last-child { border-bottom: none; }
.version-tag {
  font-family: var(--font-mono); font-weight: 600; font-size: 0.78rem;
  color: var(--accent-model); white-space: nowrap;
}

/* ── Footer ── */
.site-footer {
  text-align: center; color: var(--text-muted); font-size: 0.78rem;
  padding: 40px 0 24px; border-top: 1px solid var(--border-subtle); margin-top: 40px;
}
.site-footer a { color: var(--text-secondary); }
.site-footer a:hover { color: var(--text-primary); }

/* ── Responsive ── */
@media (max-width: 760px) {
  .container { padding: 0 16px; }
  .summary-grid { grid-template-columns: 1fr 1fr; }
  .matchup-grid { grid-template-columns: 1fr; }
  .team-grid { grid-template-columns: 1fr 1fr; }
  .team-hero-stats { gap: 16px; }
  .hero-stat-value { font-size: 1.4rem; }
  .header-nav a { font-size: 0.75rem; padding: 6px 8px; }
}
@media (max-width: 480px) {
  .summary-grid { grid-template-columns: 1fr; }
  .team-grid { grid-template-columns: 1fr; }
  .team-hero-name { font-size: 1.2rem; }
}
"""

# ── JavaScript ──

JS = r"""// Sports ML Lab — Client-side interactivity

document.addEventListener('DOMContentLoaded', function () {
  // Week selector
  const weekBtns = document.querySelectorAll('.week-selector-btn');
  const weekBlocks = document.querySelectorAll('.week-block');
  weekBtns.forEach(btn => {
    btn.addEventListener('click', function () {
      weekBtns.forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      const wk = this.dataset.week;
      weekBlocks.forEach(b => {
        b.style.display = b.dataset.week === wk ? 'block' : 'none';
      });
    });
  });
  if (weekBtns.length > 0) weekBtns[0].click();

  // Position group collapsible
  document.querySelectorAll('.position-header').forEach(hdr => {
    hdr.addEventListener('click', function () {
      const body = this.nextElementSibling;
      while (body && body.classList.contains('player-row')) {
        body.style.display = body.style.display === 'none' ? '' : 'none';
        body = body.nextElementSibling;
      }
    });
  });

  // Team search
  const searchInput = document.getElementById('team-search');
  if (searchInput) {
    searchInput.addEventListener('input', function () {
      const q = this.value.toLowerCase().trim();
      document.querySelectorAll('.team-card').forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = (!q || text.includes(q)) ? '' : 'none';
      });
    });
  }

  // Rel time
  const timestamps = document.querySelectorAll('[data-relative]');
  timestamps.forEach(el => {
    const d = new Date(el.dataset.relative.replace(' ', 'T') + 'Z');
    const now = new Date();
    const diff = Math.floor((now - d) / 60000);
    let label = '';
    if (diff < 1) label = 'just now';
    else if (diff < 60) label = diff + 'm ago';
    else if (diff < 1440) label = Math.floor(diff / 60) + 'h ago';
    else label = Math.floor(diff / 1440) + 'd ago';
    el.textContent = label;
  });
});
"""

# ── Helpers ──

def _h(v):
    return str(v) if pd.notna(v) else ""

def fmt(v):
    """Format a percentage value for display."""
    v = float(v)
    if v >= 0.5:
        return f"{v:.0%}"
    return f"{(1-v):.0%}"

def pick_team(prob, home, away):
    """Return (model_pick_team, model_pick_prob, is_home_favored)."""
    if prob >= 0.5:
        return home, prob, True
    return away, 1 - prob, False

def team_tier(wins):
    if wins >= 10: return "contender"
    if wins >= 9: return "playoff-mix"
    if wins >= 8: return "competitive"
    if wins >= 7: return "long-shot"
    return "underdog"

def gen_id(g):
    return f"{g['season']}_{g['week']:02d}_{g['away_team']}_{g['home_team']}"

# ── Data Loading ──

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
        return None, None, None, None
    values = pd.read_parquet(PLAYER_VALUES_PATH)
    from sportslab.features.player_value import aggregate_by_team
    team_vals = aggregate_by_team(values)
    additions, departures = compute_additions_departures(values)
    return values, team_vals, additions, departures

def compute_additions_departures(player_values):
    pv = player_values.copy()
    pv["team_2025"] = pv["team_2025"].fillna("").astype(str).str.strip().str.upper()
    pv["team_2026"] = pv["team_2026"].fillna("").astype(str).str.strip().str.upper()
    pv["value"] = pv["value"].fillna(0)
    valid_teams = set(TEAM_NAMES.keys())
    teams = sorted(set(pv["team_2026"].unique()) | set(pv["team_2025"].unique()))
    teams = [t for t in teams if t and t in valid_teams]
    additions, departures = {}, {}
    for team in teams:
        added = pv[(pv["team_2026"] == team) & (pv["team_2025"] != team) & (pv["value"] != 0)].nlargest(ADDITIONS_DEPARTURES_TOP_N, "value")
        departed = pv[(pv["team_2025"] == team) & (pv["team_2026"] != team) & (pv["value"] != 0)].nlargest(ADDITIONS_DEPARTURES_TOP_N, "value")
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
    df = pd.DataFrame(rows).sort_values("pred_wins", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["tier"] = df["pred_wins"].apply(team_tier)
    return df

def team_schedule(preds, team):
    home = preds[preds["home_team"] == team].copy()
    home["opponent"] = home["away_team"]
    home["team_prob"] = home["incumbent_home_win_prob"]
    home["is_home"] = True
    away = preds[preds["away_team"] == team].copy()
    away["opponent"] = away["home_team"]
    away["team_prob"] = 1 - away["incumbent_home_win_prob"]
    away["is_home"] = False
    sched = pd.concat([home, away], ignore_index=True)
    sched = sched.sort_values(["week", "gameday"])
    return sched

def _week_date_range(games):
    dates = games["gameday"].dropna()
    if len(dates) == 0: return ""
    start = dates.min().strftime("%b %d")
    end = dates.max().strftime("%b %d")
    return f"{start} – {end}"

def position_sort_key(pos):
    return POS_ORDER.index(pos) if pos in POS_ORDER else 99

# ── JSON Data Generation ──

def generate_json(preds, stats, output_dir):
    """Generate JSON files for client-side filtering."""
    assets_data = output_dir / "assets" / "data"
    assets_data.mkdir(parents=True, exist_ok=True)

    # Teams JSON
    teams_data = []
    for _, r in stats.iterrows():
        teams_data.append({
            "rank": int(r["rank"]),
            "abbr": r["team"],
            "name": TEAM_NAMES.get(r["team"], r["team"]),
            "division": TEAM_DIVISIONS.get(r["team"], ""),
            "conference": CONFERENCE.get(r["team"], ""),
            "wins": float(r["pred_wins"]),
            "games": int(r["games"]),
            "tier": r["tier"],
            "color": TEAM_COLORS.get(r["team"], "#999"),
        })
    import json as _json
    (assets_data / "teams.json").write_text(_json.dumps(teams_data, indent=2))

    # Schedule JSON (by week)
    schedule_data = {}
    for wk in sorted(preds["week"].unique()):
        games = preds[preds["week"] == wk].sort_values("gameday")
        games_list = []
        for _, g in games.iterrows():
            pick, pick_prob, _ = pick_team(g["incumbent_home_win_prob"], g["home_team"], g["away_team"])
            games_list.append({
                "id": g["game_id"],
                "week": int(g["week"]),
                "gameday": str(g["gameday"].date()) if pd.notna(g["gameday"]) else "",
                "away_team": _h(g["away_team"]),
                "home_team": _h(g["home_team"]),
                "prob": float(g["incumbent_home_win_prob"]),
                "model_pick": pick,
                "model_pick_prob": round(float(pick_prob), 4),
                "confidence_bucket": _h(g.get("confidence_bucket", "")),
                "caution_qb_change": int(g.get("caution_qb_change", 0)),
                "caution_neutral": int(g.get("caution_neutral", 0)),
                "caution_early_season": int(g.get("caution_early_season", 0)),
            })
        schedule_data[f"week_{wk}"] = games_list
    (assets_data / "schedule.json").write_text(_json.dumps(schedule_data, indent=2))

    # Standings JSON
    standings_data = {"afc": {}, "nfc": {}}
    for conf in ["AFC", "NFC"]:
        conf_teams = stats[stats["team"].isin([t for t in stats["team"] if CONFERENCE.get(t) == conf])]
        for div in sorted(set(TEAM_DIVISIONS.values())):
            if div.startswith(conf[:3]):
                div_teams = conf_teams[conf_teams["team"].isin([t for t in conf_teams["team"] if TEAM_DIVISIONS.get(t) == div])]
                standings_data[conf.lower()][div] = []
                for _, r in div_teams.sort_values("pred_wins", ascending=False).iterrows():
                    standings_data[conf.lower()][div].append({
                        "abbr": r["team"],
                        "name": TEAM_NAMES.get(r["team"], r["team"]),
                        "wins": float(r["pred_wins"]),
                        "rank": int(r["rank"]),
                        "tier": r["tier"],
                    })
    (assets_data / "standings.json").write_text(_json.dumps(standings_data, indent=2))

    print(f"  JSON data written to {assets_data}")

# ── HTML Components ──

def _render_header(active_nav=""):
    nav_links = [
        ("index.html", "Predictions"),
        ("schedule.html", "Schedule"),
        ("standings.html", "Standings"),
        ("model.html", "Model"),
    ]
    links_html = "".join(
        f'<a href="{href}"{" class=\"active\"" if label == active_nav else ""}>{label}</a>'
        for href, label in nav_links
    )
    return f"""<header class="site-header">
<div class="container header-inner">
<div class="header-left">
<a href="index.html" class="header-logo">Sports <span>ML</span> Lab</a>
<div class="header-tagline">2026 NFL Predictions</div>
</div>
<nav class="header-nav">{links_html}</nav>
</div>
</header>"""

def _render_footer():
    ts = datetime.now(timezone.utc)
    return f"""<footer class="site-footer">
<div class="container">
<p>Research output from <a href="https://github.com/SecuritahGuy/sports-ml-lab">Sports ML Lab</a>.
Not betting advice. Data is preseason; accuracy improves as weekly starters are confirmed.</p>
<p style="margin-top:6px">Generated {ts.strftime("%Y-%m-%d %H:%M UTC")}</p>
</div>
</footer>"""

def _render_model_banner():
    return """<div class="model-status">
<div class="model-status-left">
<span class="model-status-badge">MODEL v3.0.0</span>
<span class="model-status-badge">FROZEN QB OVERLAY</span>
<span class="model-status-badge">PRESEASON</span>
<span class="model-status-text">Independent, market-free projections powered by Elo and quarterback-adjusted modeling.</span>
</div>
</div>"""

def render_page(title, body_html, active_nav="", extra_head=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Sports ML Lab</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">{extra_head}
</head>
<body>
{_render_header(active_nav)}
<main class="container">
{body_html}
</main>
{_render_footer()}
<script src="assets/app.js"></script>
</body>
</html>"""

# ── Dashboard Homepage ──

def render_index(preds, stats):
    teams = stats.to_dict("records")
    top_team = teams[0]
    week1 = preds[preds["week"] == 1].sort_values("gameday")

    # Summary cards
    most_confident = None
    closest = None
    best_conf = 0
    closest_diff = 1.0
    for _, g in week1.iterrows():
        p = g["incumbent_home_win_prob"]
        cp = max(p, 1 - p)
        if cp > best_conf:
            best_conf = cp
            most_confident = g
        diff = abs(p - 0.5)
        if diff < closest_diff:
            closest_diff = diff
            closest = g

    def summary_card(label, value, sub=""):
        return f"""<div class="summary-card">
<div class="summary-card-label">{label}</div>
<div class="summary-card-value">{value}</div>
{f'<div class="summary-card-sub">{sub}</div>' if sub else ''}
</div>"""

    mct = most_confident
    mc_pick, mc_prob, _ = pick_team(mct["incumbent_home_win_prob"], mct["home_team"], mct["away_team"]) if mct is not None else ("", 0)
    cl = closest
    cl_pick, cl_prob, _ = pick_team(cl["incumbent_home_win_prob"], cl["home_team"], cl["away_team"]) if cl is not None else ("", 0)

    summary_cards = ""
    summary_cards += summary_card("Top Projected Team",
        f"{top_team['team']} · {top_team['pred_wins']} wins",
        TEAM_NAMES.get(top_team["team"], ""))
    summary_cards += summary_card("Most Confident Week 1",
        f"{mc_pick} · {mc_prob:.0%}",
        f"{mct['away_team']} @ {mct['home_team']}" if mct is not None else "")
    summary_cards += summary_card("Closest Week 1 Matchup",
        f"{cl_pick} · {cl_prob:.0%}",
        f"{cl['away_team']} @ {cl['home_team']}" if cl is not None else "")
    summary_cards += summary_card("Model Status",
        "Preseason Projections",
        "QB starters not yet confirmed")

    # Week 1 matchup cards
    matchup_cards = ""
    for _, g in week1.iterrows():
        prob = g["incumbent_home_win_prob"]
        pick, pick_prob, is_home_fav = pick_team(prob, g["home_team"], g["away_team"])
        conf_bucket = _h(g.get("confidence_bucket", ""))
        away_c = TEAM_COLORS.get(g["away_team"], "#999")
        home_c = TEAM_COLORS.get(g["home_team"], "#999")
        day_str = g["gameday"].strftime("%a %b %d") if pd.notna(g["gameday"]) else ""

        cautions = ""
        if str(g.get("caution_neutral", "")).strip() in ("1", "1.0"):
            cautions += '<span class="caution-flag">Neutral site</span>'
        if str(g.get("caution_qb_change", "")).strip() in ("1", "1.0"):
            cautions += '<span class="caution-flag">QB change</span>'

        bar_color = "var(--positive)" if is_home_fav else "var(--negative)"
        bar_pct = max(pick_prob * 100, 50)

        loc_label = "at" if not is_home_fav else "vs"

        matchup_cards += f"""<a href="teams/{g['home_team']}.html" style="display:block;color:inherit;text-decoration:none">
<div class="matchup-card">
<div class="matchup-teams">
<div class="matchup-team" style="text-align:left">
<span class="matchup-team-abbr" style="color:{away_c}">{g['away_team']}</span>
<span class="matchup-team-name">{TEAM_NAMES.get(g['away_team'], '')}</span>
</div>
<div class="matchup-vs">{loc_label}</div>
<div class="matchup-team" style="text-align:right">
<span class="matchup-team-abbr" style="color:{home_c}">{g['home_team']}</span>
<span class="matchup-team-name">{TEAM_NAMES.get(g['home_team'], '')}</span>
</div>
</div>
<div class="matchup-location">{day_str}</div>
<div class="matchup-bar-track"><div class="matchup-bar-fill" style="width:{bar_pct:.0f}%;background:{bar_color}"></div></div>
<div class="matchup-meta">
<span class="matchup-pick">Model pick: <span style="color:{bar_color}">{pick}</span> — {pick_prob:.0%}</span>
<span class="matchup-confidence">{conf_bucket}</span>
</div>
{f'<div class="matchup-cautions">{cautions}</div>' if cautions else ''}
</div>
</a>"""

    # Top teams grid
    top8 = teams[:8]
    team_cards = ""
    for r in top8:
        t = r["team"]
        tier = r["tier"]
        name = TEAM_NAMES.get(t, t)
        color = TEAM_COLORS.get(t, "#999")
        tier_name = TIER_NAMES.get(tier, tier)
        team_cards += f"""<a href="teams/{t}.html" class="team-card" style="border-left-color:{color}">
<div class="team-card-rank">#{r['rank']}</div>
<div class="team-card-abbr" style="color:{color}">{t}</div>
<div class="team-card-name">{name}</div>
<div class="team-card-wins">{r['pred_wins']}<small> wins</small></div>
<div class="team-card-tier tier-{tier}">{tier_name}</div>
</a>"""

    # Previous rank simulation (would need rank tracking; for now show tier)
    html = f"""{_render_model_banner()}

<div class="summary-grid">{summary_cards}</div>

<div class="section-header">
<h2>Week 1 — Matchups</h2>
<a href="schedule.html" class="section-link">Full schedule →</a>
</div>
<div class="matchup-grid">{matchup_cards}</div>

<div class="section-header">
<h2>Top Projected Teams</h2>
<a href="standings.html" class="section-link">Full standings →</a>
</div>
<div class="team-grid">{team_cards}</div>
<p style="color:var(--text-muted);font-size:0.78rem;margin-bottom:28px">
Tiers: Contender (10+) · Playoff Mix (9–10) · Competitive (8–9) · Long Shot (7–8) · Underdog ({"<"}7)
</p>"""

    return render_page("2026 Predictions", html, "Predictions")

# ── Schedule Page ──

def render_schedule(preds):
    weeks = sorted(preds["week"].unique())

    week_btns = "".join(
        f'<button class="week-selector-btn" data-week="{wk}">Week {wk}</button>'
        for wk in weeks
    )

    sections = []
    for wk in weeks:
        games = preds[preds["week"] == wk].sort_values("gameday")
        date_range = _week_date_range(games)
        rows = []
        for _, g in games.iterrows():
            away = _h(g["away_team"])
            home = _h(g["home_team"])
            prob = g["incumbent_home_win_prob"]
            bucket = _h(g.get("confidence_bucket", ""))
            away_c = TEAM_COLORS.get(away, "#999")
            home_c = TEAM_COLORS.get(home, "#999")

            pick, pick_prob, is_home_fav = pick_team(prob, home, away)
            prob_class = "prob-fav" if prob >= 0.55 else ("prob-even" if prob >= 0.45 else "prob-dog")

            loc_badge = ""
            if str(g.get("caution_neutral", "")).strip() in ("1", "1.0"):
                loc_badge = '<span class="badge badge-neutral">Neutral</span>'
            elif g.get("is_home") if "is_home" else True:
                loc_badge = '<span class="badge badge-home">Home</span>'

            cautions = ""
            if str(g.get("caution_qb_change", "")).strip() in ("1", "1.0"):
                cautions += '<span class="caution-flag">QB change</span>'
            if str(g.get("caution_early_season", "")).strip() in ("1", "1.0"):
                cautions += '<span class="caution-flag">Early</span>'

            day_str = g["gameday"].strftime("%a %b %d") if pd.notna(g["gameday"]) else ""

            rows.append(f"""<tr>
<td style="color:var(--text-muted);font-size:0.78rem">{day_str}</td>
<td class="team-col"><a href="teams/{away}.html" style="color:{away_c};font-weight:600">{away}</a></td>
<td class="team-col"><a href="teams/{home}.html" style="color:{home_c};font-weight:600">{home}</a></td>
<td>{loc_badge}</td>
<td class="prob-bar {prob_class}">{prob:.0%}</td>
<td style="font-weight:600">Model pick: {pick} ({pick_prob:.0%})</td>
<td style="font-size:0.75rem;color:var(--text-muted)">{bucket}</td>
<td>{cautions}</td>
</tr>""")

        sections.append(f"""<div class="week-block" data-week="{wk}">
<h3 style="font-size:1rem;margin-bottom:8px">Week {wk} <span style="color:var(--text-muted);font-weight:400;font-size:0.82rem">— {date_range}</span></h3>
<div class="table-wrap">
<table>
<thead><tr><th>Date</th><th>Away</th><th>Home</th><th>Loc</th><th>Home Win%</th><th>Model Pick</th><th>Confidence</th><th></th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
</div>""")

    html = f"""<div class="section-header">
<h2>2026 Season Schedule</h2>
</div>
{_render_model_banner()}
<div class="week-selector">{week_btns}</div>
<div class="week-container">{"".join(sections)}</div>"""
    return render_page("Schedule", html, "Schedule")

# ── Standings Page ──

def render_standings(stats):
    def _div_table(div_teams, div_name):
        rows = "".join(
            f"""<tr style="border-left:3px solid {TEAM_COLORS.get(r['team'], '#999')}">
<td class="num">#{r['rank']}</td>
<td class="team-col" style="color:{TEAM_COLORS.get(r['team'], '#ccc')}">{r['team']}</td>
<td>{TEAM_NAMES.get(r['team'], r['team'])}</td>
<td class="num {'pos' if float(r['pred_wins']) >= 10 else 'neutral' if float(r['pred_wins']) >= 8 else 'neg'}">{r['pred_wins']}</td>
<td><span class="team-card-tier tier-{r['tier']}" style="font-size:0.6rem">{TIER_NAMES.get(r['tier'], r['tier'])}</span></td>
</tr>"""
            for _, r in div_teams.iterrows()
        )
        return f"""<div style="margin-bottom:20px">
<h3 style="font-size:0.9rem;margin-bottom:8px;color:var(--text-secondary)">{div_name}</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Rank</th><th>Team</th><th>Name</th><th>Wins</th><th>Tier</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
</div>"""

    afc_html = "".join(
        _div_table(stats[stats["team"].isin([t for t in stats["team"] if TEAM_DIVISIONS.get(t) == div])], div)
        for div in ["AFC East", "AFC North", "AFC South", "AFC West"]
    )
    nfc_html = "".join(
        _div_table(stats[stats["team"].isin([t for t in stats["team"] if TEAM_DIVISIONS.get(t) == div])], div)
        for div in ["NFC East", "NFC North", "NFC South", "NFC West"]
    )

    # League table (top 10)
    league_rows = "".join(
        f"""<tr style="border-left:3px solid {TEAM_COLORS.get(r['team'], '#999')}">
<td class="num">#{r['rank']}</td>
<td class="team-col" style="color:{TEAM_COLORS.get(r['team'], '#ccc')}">{r['team']}</td>
<td>{TEAM_NAMES.get(r['team'], r['team'])}</td>
<td>{TEAM_DIVISIONS.get(r['team'], '')}</td>
<td class="num {'pos' if float(r['pred_wins']) >= 10 else 'neutral' if float(r['pred_wins']) >= 8 else 'neg'}">{r['pred_wins']}</td>
<td><span class="team-card-tier tier-{r['tier']}" style="font-size:0.6rem">{TIER_NAMES.get(r['tier'], r['tier'])}</span></td>
</tr>"""
        for _, r in stats.head(16).iterrows()
    )

    html = f"""<div class="section-header">
<h2>2026 Predicted Standings</h2>
</div>
{_render_model_banner()}

<h3 style="font-size:0.95rem;margin-bottom:12px">League — Top 16</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Rank</th><th>Team</th><th>Name</th><th>Division</th><th>Wins</th><th>Tier</th></tr></thead>
<tbody>{league_rows}</tbody>
</table>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px">
<div><h2 style="font-size:1rem;margin-bottom:12px">AFC</h2>{afc_html}</div>
<div><h2 style="font-size:1rem;margin-bottom:12px">NFC</h2>{nfc_html}</div>
</div>"""
    return render_page("Standings", html, "Standings")

# ── Team Page ──

def render_team_page(team, preds, rosters, stats, player_values=None, team_values=None, additions=None, departures=None):
    name = TEAM_NAMES.get(team, team)
    div = TEAM_DIVISIONS.get(team, "")
    conf = CONFERENCE.get(team, "")
    color = TEAM_COLORS.get(team, "#999")
    team_stat = stats[stats["team"] == team]
    pw = team_stat["pred_wins"].values[0] if len(team_stat) else 0
    rank = int(team_stat["rank"].values[0]) if len(team_stat) else 0
    tier = team_tier(pw)
    tier_name = TIER_NAMES.get(tier, tier)

    # Prev/next team nav
    all_teams = stats["team"].tolist()
    idx = all_teams.index(team) if team in all_teams else -1
    prev_t = all_teams[idx - 1] if idx > 0 else None
    next_t = all_teams[idx + 1] if idx >= 0 and idx < len(all_teams) - 1 else None
    nav_html = ""
    if prev_t:
        nav_html += f'<a href="{prev_t}.html">← {prev_t}</a>'
    if next_t:
        nav_html += f'<a href="{next_t}.html">{next_t} →</a>'

    # Hero
    hero_html = f"""<div class="team-hero">
<div class="team-hero-header">
<h1 class="team-hero-name" style="color:{color}">{name}</h1>
<div class="team-hero-nav">{nav_html}</div>
</div>
<div class="team-hero-meta">{conf} · {div}</div>
<div class="team-hero-stats">
<div class="hero-stat">
<div class="hero-stat-value" style="color:{color}">{pw:.1f}</div>
<div class="hero-stat-label">Projected Wins</div>
</div>
<div class="hero-stat">
<div class="hero-stat-value">#{rank}</div>
<div class="hero-stat-label">Overall Rank</div>
</div>
<div class="hero-stat">
<div class="hero-stat-value"><span class="team-card-tier tier-{tier}">{tier_name}</span></div>
<div class="hero-stat-label">Model Tier</div>
</div>
</div>
</div>"""

    # Roster Strength
    strength_html = ""
    if team_values is not None and len(team_values):
        tv = team_values[team_values["team"] == team.upper()]
        if len(tv):
            tv = tv.iloc[0]
            bars = []
            for vg in VALUE_GROUP_ORDER:
                pctl = float(tv.get(f"{vg}_avg_pctl", 0))
                label = VALUE_GROUP_LABELS.get(vg, vg)
                bar_color = "var(--positive)" if pctl >= 60 else ("var(--warning)" if pctl >= 40 else "var(--negative)")
                bars.append(f"""<div class="strength-row">
<span class="strength-label">{label}</span>
<div class="strength-track">
<div class="strength-fill" style="width:{pctl:.0f}%;background:{bar_color}"></div>
</div>
<span class="strength-pctl">{pctl:.0f}</span>
</div>""")
            strength_html = f"""<div class="strength-section">
<h3 style="font-size:0.9rem;margin-bottom:8px">Roster Strength</h3>
<p style="color:var(--text-muted);font-size:0.75rem;margin-bottom:8px">Position-group percentile ranks vs all NFL teams</p>
{"".join(bars)}
</div>"""

    # Schedule
    sched = team_schedule(preds, team)
    sched_rows = []
    for _, g in sched.iterrows():
        opponent = g["opponent"]
        team_prob = g["team_prob"]
        is_home = g["is_home"]
        bucket = _h(g["confidence_bucket"])
        opp_c = TEAM_COLORS.get(opponent, "#999")

        prob_class = "prob-fav" if team_prob >= 0.55 else ("prob-even" if team_prob >= 0.45 else "prob-dog")
        loc_badge = '<span class="badge badge-home">HOME</span>' if is_home else '<span class="badge badge-away">AWAY</span>'

        cautions = ""
        if str(g.get("caution_qb_change", "")).strip() in ("1", "1.0"):
            cautions += '<span class="caution-flag">QB change</span>'
        if str(g.get("caution_early_season", "")).strip() in ("1", "1.0"):
            cautions += '<span class="caution-flag">Early</span>'

        day_str = g["gameday"].strftime("%a %b %d") if pd.notna(g["gameday"]) else ""

        # Prob bar
        bar_color = "var(--positive)" if team_prob >= 0.5 else "var(--negative)"
        bar_w = max(team_prob, 1 - team_prob) * 100

        sched_rows.append(f"""<tr>
<td class="num">W{g['week']}</td>
<td style="color:var(--text-muted);font-size:0.78rem">{day_str}</td>
<td class="team-col"><a href="{opponent}.html" style="color:{opp_c};font-weight:600">{opponent}</a> {loc_badge}</td>
<td class="prob-bar {prob_class}">{team_prob:.0%}</td>
<td><div class="matchup-bar-track" style="width:80px;display:inline-block;vertical-align:middle"><div class="matchup-bar-fill" style="width:{bar_w:.0f}%;background:{bar_color}"></div></div></td>
<td style="font-size:0.75rem;color:var(--text-muted)">{bucket}</td>
<td>{cautions}</td>
</tr>""")

    sched_html = f"""<div style="margin-bottom:24px">
<h3 style="font-size:0.9rem;margin-bottom:8px">2026 Schedule</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Week</th><th>Date</th><th>Opponent</th><th>Win Prob</th><th></th><th>Confidence</th><th></th></tr></thead>
<tbody>{"".join(sched_rows)}</tbody>
</table>
</div>
</div>"""

    # Offseason Moves
    moves_html = ""
    if additions is not None and departures is not None:
        team_add = additions.get(team, pd.DataFrame())
        team_dep = departures.get(team, pd.DataFrame())

        def _render_moves(rows, is_incoming):
            if len(rows) == 0: return '<p style="color:var(--text-muted);font-style:italic;font-size:0.82rem">None</p>'
            items = ""
            for _, r in rows.iterrows():
                pname = _h(r.get("player_name", "?"))
                pos = _h(r.get("position", ""))
                val = float(r.get("value", 0))
                other = _h(r.get("team_2025", "")) if is_incoming else _h(r.get("team_2026", ""))
                prefix = "from" if is_incoming else "→"
                team_tag = f" {prefix} {other}" if other else ""
                val_class = "pos" if val > 0 else "neg"
                items += f"""<div class="move-row">
<span class="move-name">{pname}</span>
<span class="move-meta">{pos}{team_tag} · <span class="{val_class}">{val:.0f}</span></span>
</div>"""
            return items

        has_add = len(team_add) > 0
        has_dep = len(team_dep) > 0
        if has_add or has_dep:
            cols = []
            if has_add:
                cols.append(f"""<div class="moves-col">
<h3 style="color:var(--positive)">Key Additions</h3>
{_render_moves(team_add, True)}
</div>""")
            if has_dep:
                cols.append(f"""<div class="moves-col">
<h3 style="color:var(--negative)">Key Departures</h3>
{_render_moves(team_dep, False)}
</div>""")
            moves_html = f"""<div style="margin-bottom:24px">
<h3 style="font-size:0.9rem;margin-bottom:8px">Offseason Movement</h3>
<p style="color:var(--text-muted);font-size:0.75rem;margin-bottom:8px">Player value scores based on 2025 EPA contribution. Top {ADDITIONS_DEPARTURES_TOP_N} added/lost.</p>
<div class="moves-grid">{"".join(cols)}</div>
</div>"""

    # Roster
    team_rost = rosters[rosters["team"] == team.upper()].copy() if "team" in rosters.columns else rosters[rosters["team_abbr"] == team.upper()].copy() if "team_abbr" in rosters.columns else pd.DataFrame()
    roster_html = ""
    if len(team_rost):
        team_rost["pos_group"] = team_rost.get("position", "").map(POS_TO_VALUE_GROUP).fillna("OTHER")
        team_rost["pos_sort"] = team_rost.get("position", "").apply(position_sort_key)
        team_rost = team_rost.sort_values(["pos_sort", "jersey_number" if "jersey_number" in team_rost.columns else "position"])

        pv_index = {}
        if player_values is not None:
            for _, r in player_values.iterrows():
                pv_index[r.get("player_name", "").upper()] = float(r.get("value", 0))

        sections = []
        for pos, label in POSITION_GROUPS.items():
            group = team_rost[team_rost.get("position", "") == pos]
            if len(group) == 0: continue
            players = []
            for _, p in group.iterrows():
                pname = _h(p.get("full_name", p.get("player_name", "?")))
                pname_display = _h(p.get("display_name", p.get("player_name", pname)))
                num = _h(p.get("jersey_number", ""))
                exp = _h(p.get("years_exp", p.get("season", "")))
                value = pv_index.get(pname_display.upper(), pv_index.get(pname.upper(), 0))
                val_badge = ""
                if value > 0:
                    vc = "val-high" if value >= 50 else ("val-mid" if value >= 20 else "val-low")
                    val_badge = f'<span class="player-value {vc}">{value:.0f}</span>'
                players.append(f'<div class="player-row"><span class="name">{pname_display}{val_badge}</span><span class="meta">#{num}</span><span class="meta">{exp} yr</span></div>')

            sections.append(f"""<div class="position-group">
<div class="position-header"><span>{label}</span><span class="position-count">{len(group)}</span></div>
{"".join(players)}
</div>""")

        if sections:
            roster_html = f"""<div class="roster-section">
<h3 style="font-size:0.9rem;margin-bottom:8px">Roster</h3>
{"".join(sections)}
</div>"""

    body_html = hero_html + strength_html + sched_html + moves_html + roster_html
    return render_page(f"{team} — {name}", body_html, "Teams")

# ── Model Page ──

def render_model():
    html = f"""<div class="section-header"><h2>Model Methodology</h2></div>
{_render_model_banner()}

<div class="model-section">
<h2>How It Works</h2>
<p>The Sports ML Lab prediction system is a two-layer architecture that combines a Bayesian rating system with game-context adjustments. Every prediction is pregame-safe, market-free, and leakage-verified.</p>
<div class="pipeline">
<span class="pipeline-step">Historical games</span>
<span class="pipeline-arrow">→</span>
<span class="pipeline-step">Base Elo rating</span>
<span class="pipeline-arrow">→</span>
<span class="pipeline-step">Recent form + QB context</span>
<span class="pipeline-arrow">→</span>
<span class="pipeline-step">Probability calibration</span>
<span class="pipeline-arrow">→</span>
<span class="pipeline-step">Frozen QB overlay</span>
<span class="pipeline-arrow">→</span>
<span class="pipeline-step">Game prediction</span>
</div>
</div>

<div class="model-section">
<h2>Performance</h2>
<div class="metric-cards">
<div class="metric-card">
<div class="metric-value">0.6200</div>
<div class="metric-label">Log Loss</div>
<div class="metric-desc">Measures probability quality — lower is better</div>
</div>
<div class="metric-card">
<div class="metric-value">0.2157</div>
<div class="metric-label">Brier Score</div>
<div class="metric-desc">Measures calibration and accuracy — lower is better</div>
</div>
<div class="metric-card">
<div class="metric-value">0.7098</div>
<div class="metric-label">ROC AUC</div>
<div class="metric-desc">Measures ranking discrimination — higher is better</div>
</div>
<div class="metric-card">
<div class="metric-value">0.6090</div>
<div class="metric-label">Market Baseline</div>
<div class="metric-desc">No-vig moneyline baseline — diagnostic only, not a target</div>
</div>
</div>
<p>Our model trails the market closing line (0.6090 vs 0.6200 log loss), which is expected — the market has access to injury reports, weather forecasts, and late-breaking information that our pregame-independent model does not. This gap represents the information ceiling for market-free prediction.</p>
</div>

<div class="model-section">
<h2>Version History</h2>
<ul class="version-list">
<li><span class="version-tag">v3.0.0</span> <span>Frozen QB Overlay — Current champion. Adds two-layer adjustment: QB stability gate (changed OR starts &lt; 17) followed by roster position-group overlay. Holdout LL 0.6200.</span></li>
<li><span class="version-tag">v2.0.0</span> <span>Feature-augmented Elo — First model to beat incumbent on both validation and holdout. Added qb_changed + rolling_mov_3 as Platt features. Holdout LL 0.6262.</span></li>
<li><span class="version-tag">v1.x</span> <span>Base research models — Elo-only, MOV Elo, season regression, and O/D Elo. Each built on the previous but all superseded by v2.0.0.</span></li>
</ul>
</div>

<div class="model-section">
<h2>What the Model Does Not Know</h2>
<p style="color:var(--text-secondary);font-size:0.85rem;line-height:1.8">
• <strong style="color:var(--text-primary)">Injuries</strong> — Injury report data was tested and rejected as too noisy at this sample size.<br>
• <strong style="color:var(--text-primary)">Unconfirmed starters</strong> — QB data is preseason-only; accuracy improves when weekly starters are confirmed.<br>
• <strong style="color:var(--text-primary)">Weather</strong> — Weather features tested and rejected; model performs comparably in all conditions.<br>
• <strong style="color:var(--text-primary)">Market information</strong> — Market odds are diagnostic-only. The model is market-free by design.<br>
• <strong style="color:var(--text-primary)">Coaching or scheme changes</strong> — Coach tenure features tested and rejected.<br>
• <strong style="color:var(--text-primary)">Playoff stakes</strong> — No situational weighting for playoff-elimination games.
</p>
</div>

<div class="model-section">
<h2>Research Integrity</h2>
<p>The model has been hardened against data leakage through chronological feature computation, holdout exclusion, and schema verification. Every experiment is audited for leakage risk before promotion. See the <a href="https://github.com/SecuritahGuy/sports-ml-lab">GitHub repository</a> for the full experiment ledger and benchmark history.</p>
</div>
"""
    return render_page("Model Methodology", html, "Model")

# ── Main ──

def write_assets():
    assets_dir = OUTPUT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "style.css").write_text(CSS)
    (assets_dir / "app.js").write_text(JS)
    print(f"  Assets written to {assets_dir}")

def build_site():
    OUTPUT.mkdir(parents=True, exist_ok=True)

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
        print(f"  Additions/departures computed for 32 teams")
    else:
        print("  No player values found — skipping roster strength display")

    print("Computing team stats...")
    stats = compute_team_stats(preds)
    print(f"  {len(stats)} teams")

    # Write assets
    write_assets()

    # Generate JSON
    generate_json(preds, stats, OUTPUT)

    # Generate pages
    pages = [
        ("index.html", render_index(preds, stats)),
        ("schedule.html", render_schedule(preds)),
        ("standings.html", render_standings(stats)),
        ("model.html", render_model()),
    ]

    teams_dir = OUTPUT / "teams"
    teams_dir.mkdir(exist_ok=True)

    for t in stats["team"].tolist():
        html = render_team_page(t, preds, rosters, stats, player_values, team_values, additions, departures)
        (teams_dir / f"{t}.html").write_text(html)

    for name, html in pages:
        (OUTPUT / name).write_text(html)

    n_teams = len(stats)
    n_players = len(rosters) if len(rosters) > 0 else 0
    n_files = sum(1 for _ in OUTPUT.rglob("*.html")) + sum(1 for _ in OUTPUT.rglob("*.css")) + sum(1 for _ in OUTPUT.rglob("*.js"))
    print(f"\nDone! Site written to {OUTPUT}")
    print(f"  {n_teams} team pages")
    print(f"  {len(preds)} games indexed")
    print(f"  {n_players} players in roster database")
    print(f"  {n_files} total files")
    print(f"\nTo test locally:  python -m http.server 8080 -d site")
    print(f"Cloudflare Pages: python src/sportslab/evaluation/build_team_site.py")

main = build_site

if __name__ == "__main__":
    build_site()
