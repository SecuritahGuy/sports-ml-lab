"""Prediction vintage comparison — track how predictions drift across a week.

A "vintage" is a prediction snapshot taken at a specific point in the
pre-game week. The three standard vintages are:

  - early:        Early-week prediction (e.g., Tuesday).
  - final-injury: After Friday's final injury reports.
  - locked:       Final locked prediction (Sunday, pre-kickoff).

Each vintage preserves: expected starting QB, QB source, timestamp,
home-win probability, confidence bucket, whether the QB overlay gate
fired, and the overlay magnitude.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from sportslab.evaluation.weekly_pipeline import (
    _iso_now,
    _read_manifest,
)

COMPARISON_COLUMNS = [
    "game_id", "season", "week", "away_team", "home_team",
    "incumbent_home_win_prob", "qb_source", "confidence_bucket",
]


def _get_vintage_entry(
    manifest_entry: dict, vintage: str,
) -> Optional[dict]:
    """Verify a manifest entry matches the requested vintage."""
    if manifest_entry is None:
        return None
    if manifest_entry.get("vintage", "locked") != vintage:
        return None
    return manifest_entry


def list_vintages(season: int, week: int, mode: str = "live",
                  verbose: bool = False) -> List[dict]:
    """List all vintage entries in the manifest for a given season/week/mode.

    Returns a list of manifest entry dicts, one per vintage.
    """
    manifest = _read_manifest()
    matches = [
        s for s in manifest["snapshots"]
        if s["season"] == season and s["week"] == week
        and s.get("mode", "live") == mode
        and s.get("status", "initial") != "superseded"
    ]
    if verbose:
        print(f"\n=== Vintages for {season} Week {week} ({mode}) ===")
        if not matches:
            print("  No vintages found.")
            return matches
        for s in matches:
            vintage = s.get("vintage", "locked")
            qb_src = s.get("qb_source", "?")
            n_games = s.get("n_games", "?")
            created = s.get("created_at", "?")[:19]
            graded = "✓" if s.get("graded") else " "
            print(f"  [{graded}] {vintage:15s} | {n_games:2d} games | "
                  f"QB: {qb_src:12s} | {created}")
    return matches


def load_vintages(season: int, week: int, mode: str = "live"
                  ) -> Dict[str, pd.DataFrame]:
    """Load all vintage DataFrames for a given season/week/mode.

    Returns a dict mapping vintage label → DataFrame of predictions.
    """
    manifest = _read_manifest()
    matches = [
        s for s in manifest["snapshots"]
        if s["season"] == season and s["week"] == week
        and s.get("mode", "live") == mode
        and s.get("status", "initial") != "superseded"
    ]
    result: Dict[str, pd.DataFrame] = {}
    for entry in matches:
        vintage = entry.get("vintage", "locked")
        path = Path(entry["path"])
        if path.exists():
            result[vintage] = pd.read_csv(path)
        else:
            print(f"  WARNING: Snapshot file missing for vintage '{vintage}': {path}")
    return result


def compare_vintages(season: int, week: int, mode: str = "live"
                     ) -> Optional[pd.DataFrame]:
    """Compare predictions across all vintages for a given season/week/mode.

    Returns a DataFrame with one row per game and one column per vintage
    for home-win probability, plus columns for QB source and confidence
    bucket per vintage.
    """
    vintages = load_vintages(season, week, mode)
    if len(vintages) < 2:
        print(f"  Need at least 2 vintages to compare ({len(vintages)} found).")
        return None

    # Build a comparison table aligned by game_id
    prob_cols: Dict[str, pd.Series] = {}
    qb_cols: Dict[str, pd.Series] = {}
    conf_cols: Dict[str, pd.Series] = {}
    gate_cols: Dict[str, pd.Series] = {}
    first_df = None

    for vintage in sorted(vintages.keys()):
        df = vintages[vintage].set_index("game_id")
        if first_df is None:
            first_df = df[["season", "week", "away_team", "home_team"]].copy()
        prob_cols[vintage] = df["incumbent_home_win_prob"]
        default_str = pd.Series("?", index=df.index)
        qb_cols[vintage] = df["qb_source"] if "qb_source" in df.columns else default_str
        conf_cols[vintage] = (
            df["confidence_bucket"]
            if "confidence_bucket" in df.columns
            else pd.Series("?", index=df.index)
        )
        if "qb_gate_fired" in df.columns:
            gate_cols[vintage] = df["qb_gate_fired"]

    if first_df is None:
        return None

    result = first_df.copy()
    for vintage in sorted(vintages.keys()):
        result[f"prob_{vintage}"] = prob_cols.get(vintage, pd.Series(dtype=float))
        result[f"qb_{vintage}"] = qb_cols.get(vintage, pd.Series(dtype=str))
        result[f"conf_{vintage}"] = conf_cols.get(vintage, pd.Series(dtype=str))
        if vintage in gate_cols:
            result[f"gate_{vintage}"] = gate_cols[vintage]

    # Add probability drift columns
    vintage_list = sorted(vintages.keys())
    for i in range(len(vintage_list) - 1):
        v_from = vintage_list[i]
        v_to = vintage_list[i + 1]
        col_from = f"prob_{v_from}"
        col_to = f"prob_{v_to}"
        if col_from in result.columns and col_to in result.columns:
            result[f"drift_{v_from}_to_{v_to}"] = (
                result[col_to] - result[col_from]
            )

    result = result.reset_index()
    return result


def _format_drift(val: float) -> str:
    """Format a drift value as a signed percentage string."""
    if pd.isna(val):
        return "   N/A  "
    pct = val * 100
    if abs(pct) < 0.01:
        return "    0.0%"
    return f"{pct:+.1f}%"


def _print_game_diff(game_row: pd.Series, vintage_list: List[str]) -> str:
    """Format a single game's comparison as a string."""
    gid = game_row.get("game_id", "?")
    away = game_row.get("away_team", "?")
    home = game_row.get("home_team", "?")
    lines = [f"  {away} @ {home} ({gid})"]
    for v in vintage_list:
        prob = game_row.get(f"prob_{v}", float("nan"))
        qb = game_row.get(f"qb_{v}", "?")
        conf = game_row.get(f"conf_{v}", "?")
        prob_str = f"{prob*100:.1f}%" if pd.notna(prob) else "  N/A  "
        lines.append(f"    {v:15s} prob={prob_str}  qb={qb:12s}  conf={conf}")
    drift_cols = [c for c in game_row.index if c.startswith("drift_")]
    for dc in drift_cols:
        val = game_row[dc]
        lines.append(f"    {dc:25s} {_format_drift(val)}")
    return "\n".join(lines)


def vintage_diff_report(season: int, week: int, mode: str = "live",
                        output_path: Optional[str] = None) -> str:
    """Generate a markdown comparison report across vintages.

    Args:
        season: Season year.
        week: Week number.
        mode: Snapshot mode.
        output_path: Optional file path to write the report.

    Returns:
        Report as a string.
    """
    vintages = load_vintages(season, week, mode)
    if len(vintages) < 2:
        msg = f"Need at least 2 vintages ({len(vintages)} found)."
        return msg

    vintage_list = sorted(vintages.keys())
    comp = compare_vintages(season, week, mode)
    if comp is None:
        return "No comparison available."

    report_lines = [
        f"# Vintage Comparison: {season} Week {week} ({mode})",
        "",
        f"Generated: {_iso_now()[:19]}",
        f"Vintages: {', '.join(vintage_list)}",
        "",
        "---",
        "",
        "## Per-Game Comparison",
        "",
    ]

    for _, row in comp.iterrows():
        report_lines.append(_print_game_diff(row, vintage_list))
        report_lines.append("")

    # Summary statistics
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Vintage | Games |")
    report_lines.append("|---------|-------|")
    for v in vintage_list:
        df = vintages[v]
        report_lines.append(f"| {v} | {len(df)} |")

    # Biggest drifts
    drift_cols = [c for c in comp.columns if c.startswith("drift_")]
    if drift_cols:
        report_lines.append("")
        report_lines.append("### Biggest Probability Drifts")
        report_lines.append("")
        for dc in drift_cols:
            report_lines.append(f"**{dc}**")
            report_lines.append("")
            report_lines.append("| Game | Drift | From | To |")
            report_lines.append("|------|-------|------|-----|")
            sorted_rows = comp.sort_values(dc, key=abs, ascending=False).head(10)
            # drift column is like "drift_early_to_final-injury"
            prefix = dc.replace("drift_", "")
            parts = prefix.split("_to_", 1)
            from_vintage = parts[0]
            to_vintage = parts[1] if len(parts) > 1 else "unknown"
            from_col = f"prob_{from_vintage}"
            to_col = f"prob_{to_vintage}"
            for _, row in sorted_rows.iterrows():
                val = row[dc]
                away = row.get("away_team", "?")
                home = row.get("home_team", "?")
                label = f"{away}@{home}"
                from_val = row.get(from_col, float("nan"))
                to_val = row.get(to_col, float("nan"))
                from_str = f"{from_val*100:.1f}%" if pd.notna(from_val) else "N/A"
                to_str = f"{to_val*100:.1f}%" if pd.notna(to_val) else "N/A"
                report_lines.append(
                    f"| {label} | {_format_drift(val).strip()} | {from_str} | {to_str} |"
                )
            report_lines.append("")

    report = "\n".join(report_lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)
        print(f"  Report: {output_path}")

    return report
